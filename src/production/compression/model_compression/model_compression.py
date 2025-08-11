from numba import jit
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    import cupy as cp  # type: ignore

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    cp = None  # type: ignore
    CUPY_AVAILABLE = False
import logging
from typing import Any

from langroid import ChatAgent, ChatAgentConfig, Task

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Bitlinearization components
class TernaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        scale = torch.mean(torch.abs(input)).clamp(min=1e-8)
        return torch.round(torch.clamp(input / scale, -1, 1)).to(torch.int8), scale

    @staticmethod
    def backward(ctx, grad_output, grad_scale):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer("weight_scale", torch.ones(1))
        self.register_buffer("quantized_weight", torch.zeros_like(self.weight, dtype=torch.int8))

    def quantize_weight(self) -> None:
        self.quantized_weight, self.weight_scale = TernaryQuantizer.apply(self.weight)

    def forward(self, x):
        x_norm = F.layer_norm(x, x.shape[-1:])
        x_quant = x_norm + (quantize_activations(x_norm) - x_norm).detach()
        w_quant = self.quantized_weight.float() * self.weight_scale
        return F.linear(x_quant, w_quant, self.bias)


def quantize_activations(x):
    scale = 127.0 / torch.max(torch.abs(x), dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return torch.round(torch.clamp(x * scale, -127, 127)) / scale


def convert_to_bitnet(model: nn.Module) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            bit_linear = BitLinear(module.in_features, module.out_features, module.bias is not None)
            bit_linear.weight.data = module.weight.data
            if module.bias is not None:
                bit_linear.bias.data = module.bias.data
            bit_linear.quantize_weight()
            setattr(model, name, bit_linear)
        else:
            convert_to_bitnet(module)
    return model


class BitNetModel(nn.Module):
    def __init__(self, original_model: nn.Module) -> None:
        super().__init__()
        self.model = convert_to_bitnet(original_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Hypercompression components
if CUPY_AVAILABLE:
    cuda_reconstruct_group = cp.RawKernel(
        r"""
extern "C" __global__
void reconstruct_group(float theta, int K, int8_t* output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < K) {
        float value = theta / (M_PI + (idx + 1));
        value = value - floor(value);
        output[idx] = round(value * 2 - 1);
    }
}
""",
        "reconstruct_group",
    )
    cuda_compress = cp.RawKernel(
        r"""
extern "C" __global__
void compress(const int8_t* weights, int num_groups, int K, float U, float* thetas) {
    int group_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (group_idx < num_groups) {
        float best_theta = 0;
        float min_loss = INFINITY;
        for (float theta = 0; theta <= U; theta += U/1000) {
            float loss = 0;
            for (int k = 0; k < K; k++) {
                float value = theta / (M_PI + (k + 1));
                value = value - floor(value);
                int8_t reconstructed = round(value * 2 - 1);
                loss += abs(reconstructed - weights[group_idx * K + k]);
            }
            if (loss < min_loss) {
                min_loss = loss;
                best_theta = theta;
            }
        }
        thetas[group_idx] = best_theta;
    }
}
""",
        "compress",
    )
else:
    cuda_reconstruct_group = None  # type: ignore
    cuda_compress = None  # type: ignore


@jit(nopython=True)
def adaptive_parameter_selection(weights: np.ndarray, target_compression: float) -> tuple[int, float]:
    total_params = weights.size
    K_candidates = [128, 256, 512, 1024]
    U_candidates = [1e5, 5e5, 1e6, 5e6]

    best_K, best_U = K_candidates[0], U_candidates[0]
    best_ratio = 0

    for K in K_candidates:
        for U in U_candidates:
            num_groups = total_params // K
            compressed_size = num_groups * 4  # 4 bytes for float32 theta
            original_size = total_params
            ratio = original_size / compressed_size

            if ratio > best_ratio and ratio <= target_compression:
                best_ratio = ratio
                best_K, best_U = K, U

    return best_K, best_U


class HyperCompressor:
    def __init__(self, K: int = 256, U: float = 1000000) -> None:
        self.K = K
        self.U = U

    def compress(self, ternary_weights: torch.Tensor, scale_factors: torch.Tensor) -> dict[str, Any]:
        if CUPY_AVAILABLE:
            W = cp.asarray(ternary_weights.numpy())
            num_groups = len(W) // self.K
            thetas = cp.zeros(num_groups, dtype=cp.float32)

            threads_per_block = 256
            blocks = (num_groups + threads_per_block - 1) // threads_per_block

            cuda_compress((blocks,), (threads_per_block,), (W, num_groups, self.K, self.U, thetas))
            theta_out = cp.asnumpy(thetas)
        else:
            W = ternary_weights.numpy()
            num_groups = len(W) // self.K
            theta_out = np.zeros(num_groups, dtype=np.float32)
            for i in range(num_groups):
                group = W[i * self.K : (i + 1) * self.K]
                best_theta = 0.0
                min_loss = float("inf")
                for theta in np.linspace(0, self.U, 1001):
                    vals = theta / (np.pi + np.arange(1, self.K + 1))
                    vals = vals - np.floor(vals)
                    recon = np.round(vals * 2 - 1).astype(np.int8)
                    loss = np.abs(recon - group).sum()
                    if loss < min_loss:
                        min_loss = loss
                        best_theta = theta
                theta_out[i] = best_theta

        return {
            "thetas": theta_out,
            "original_shape": ternary_weights.shape,
            "scale_factors": scale_factors.numpy(),
        }

    def decompress(self, compressed_data: dict[str, Any]) -> torch.Tensor:
        original_shape = compressed_data["original_shape"]
        if CUPY_AVAILABLE:
            thetas = cp.asarray(compressed_data["thetas"])
            scale_factors = cp.asarray(compressed_data["scale_factors"])

            num_groups = len(thetas)
            reconstructed = cp.zeros(num_groups * self.K, dtype=cp.int8)

            threads_per_block = 256
            blocks = (self.K + threads_per_block - 1) // threads_per_block

            for i in range(num_groups):
                cuda_reconstruct_group(
                    (blocks,),
                    (threads_per_block,),
                    (thetas[i], self.K, reconstructed[i * self.K :]),
                )

            reconstructed = cp.asnumpy(reconstructed).reshape(original_shape) * cp.asnumpy(scale_factors)
        else:
            thetas = np.asarray(compressed_data["thetas"])
            scale_factors = np.asarray(compressed_data["scale_factors"])

            num_groups = len(thetas)
            reconstructed = np.zeros(num_groups * self.K, dtype=np.int8)
            for i in range(num_groups):
                theta = thetas[i]
                vals = theta / (np.pi + np.arange(1, self.K + 1))
                vals = vals - np.floor(vals)
                reconstructed[i * self.K : (i + 1) * self.K] = np.round(vals * 2 - 1).astype(np.int8)
            reconstructed = reconstructed.reshape(original_shape) * scale_factors

        return torch.tensor(reconstructed)


# Combined ModelCompressionTask
class ModelCompressionTask(Task):
    def __init__(self, agent: ChatAgent, model: nn.Module) -> None:
        super().__init__(agent)
        self.model = model
        self.compressor = HyperCompressor()

    async def fine_tune(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 1e-4,
    ) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0
        history = {"train_loss": [], "val_acc": []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Re-quantize weights after each update
                for module in self.model.modules():
                    if isinstance(module, BitLinear):
                        module.quantize_weight()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            accuracy = correct / total
            history["val_acc"].append(accuracy)

            if accuracy > best_acc:
                best_acc = accuracy
                best_model = self.model.state_dict()

            await self.agent.llm_response(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {accuracy:.4f}"
            )
            scheduler.step()

        return {"best_model": best_model, "best_acc": best_acc, "history": history}

    async def compress_model(self, chunk_size: int = 1000000) -> dict[str, Any]:
        compressed_state_dict = {}

        for name, param in self.model.state_dict().items():
            if "weight" in name and param.dim() > 1:
                ternary_weights = torch.sign(param).to(torch.int8)
                scale_factors = param.abs().mean().to(torch.float32)

                chunks = ternary_weights.split(chunk_size)
                compressed_chunks = []

                for chunk in chunks:
                    compressed_chunk = self.compressor.compress(chunk, scale_factors)
                    compressed_chunks.append(compressed_chunk)

                compressed_state_dict[name] = {
                    "chunks": compressed_chunks,
                    "original_shape": param.shape,
                }
            else:
                compressed_state_dict[name] = param

        return compressed_state_dict

    async def decompress_model(self, compressed_state_dict: dict[str, Any]) -> nn.Module:
        decompressed_state_dict = {}

        for name, compressed_param in compressed_state_dict.items():
            if isinstance(compressed_param, dict) and "chunks" in compressed_param:
                decompressed_chunks = [self.compressor.decompress(chunk) for chunk in compressed_param["chunks"]]
                decompressed_param = torch.cat(decompressed_chunks).reshape(compressed_param["original_shape"])
                decompressed_state_dict[name] = decompressed_param
            else:
                decompressed_state_dict[name] = compressed_param

        self.model.load_state_dict(decompressed_state_dict)
        return self.model

    async def benchmark_compression(self, compressed_model: dict[str, Any]) -> dict[str, float]:
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        compressed_size = sum(
            (
                sum(
                    chunk["thetas"].size * chunk["thetas"].itemsize
                    + chunk["scale_factors"].size * chunk["scale_factors"].itemsize
                    for chunk in param["chunks"]
                )
                if isinstance(param, dict) and "chunks" in param
                else param.numel() * param.element_size()
            )
            for param in compressed_model.values()
        )
        compression_ratio = original_size / compressed_size

        return {
            "original_size_mb": original_size / (1024 * 1024),
            "compressed_size_mb": compressed_size / (1024 * 1024),
            "compression_ratio": compression_ratio,
        }

    async def run(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 1e-4,
    ) -> dict[str, Any]:
        await self.agent.llm_response("Starting model compression process...")

        # Step 1: Convert to BitNet model
        self.model = BitNetModel(self.model)
        await self.agent.llm_response("Converted model to BitNet")

        # Step 2: Fine-tune the BitNet model
        fine_tune_results = await self.fine_tune(train_loader, val_loader, epochs, lr)
        await self.agent.llm_response(
            f"Fine-tuned BitNet model. Best validation accuracy: {fine_tune_results['best_acc']:.4f}"
        )

        # Step 3: Apply hypercompression
        compressed_model = await self.compress_model()
        await self.agent.llm_response("Applied hypercompression")

        # Capture output before decompression for verification later
        with torch.no_grad():
            verification_input = torch.randn(1, 10000)
            original_output = self.model(verification_input)

        # Step 4: Benchmark compression
        metrics = await self.benchmark_compression(compressed_model)
        await self.agent.llm_response(f"Compression metrics: {metrics}")

        # Step 5: Decompress the model
        decompressed_model = await self.decompress_model(compressed_model)
        await self.agent.llm_response("Decompressed model")

        # Step 6: Verify decompression
        with torch.no_grad():
            decompressed_output = decompressed_model(verification_input)
            error = torch.mean((original_output - decompressed_output).abs())
            await self.agent.llm_response(f"Mean absolute error after decompression: {error.item():.6f}")

        return {
            "fine_tune_results": fine_tune_results,
            "compressed_model": compressed_model,
            "decompressed_model": decompressed_model,
            "metrics": metrics,
            "error": error.item(),
        }


# Example usage
if __name__ == "__main__":
    import asyncio

    from langroid.language_models.openai_gpt import OpenAIGPTConfig

    async def main() -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        if CUPY_AVAILABLE:
            cp.random.seed(0)

        config = ChatAgentConfig(
            name="ModelCompressionAgent",
            llm=OpenAIGPTConfig(chat_model="gpt-3.5-turbo"),
        )
        agent = ChatAgent(config)

        # Create a large model
        model = nn.Sequential(
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 2500),
            nn.ReLU(),
            nn.Linear(2500, 1000),
        )

        # Create dummy data loaders (replace with your actual data)
        train_loader = torch.utils.data.DataLoader(torch.randn(1000, 10000), batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(torch.randn(200, 10000), batch_size=32, shuffle=False)

        task = ModelCompressionTask(agent, model)
        results = await task.run(train_loader, val_loader)

        print(f"Compression ratio: {results['metrics']['compression_ratio']:.2f}")
        print(f"Mean absolute error: {results['error']:.6f}")

    asyncio.run(main())
