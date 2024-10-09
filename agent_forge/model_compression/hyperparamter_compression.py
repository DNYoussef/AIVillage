import torch
import numpy as np
from scipy.optimize import minimize_scalar
import concurrent.futures
from numba import jit, prange
import cupy as cp
import logging
from typing import Dict, Any, Tuple
from langroid import Task, ChatAgent, ChatAgentConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary functions and classes from hyperparameter_compression.py
from .hyperparameter_compression import HyperCompressor, adaptive_parameter_selection, cuda_reconstruct_group, cuda_compress

class HyperparamterCompressionTask(Task):
    def __init__(self, agent: ChatAgent):
        super().__init__(agent)
        self.compressor = HyperCompressor()

    async def run(self, model: torch.nn.Module) -> Dict[str, Any]:
        await self.agent.llm_response("Warning: This file (hyperparamter_compression.py) appears to be a duplicate of hyperparameter_compression.py with a typo in the name. Consider removing or renaming this file.")
        
        # The rest of the implementation is similar to HyperCompressionTask in hyperparameter_compression.py
        await self.agent.llm_response("Starting model compression process...")

        # Simulate 1.58-bit quantization
        with torch.no_grad():
            for param in model.parameters():
                if param.dim() > 1:
                    param.data = torch.sign(param.data) * param.data.abs().mean()

        await self.agent.llm_response("Quantized model to 1.58-bit")

        # Adaptive parameter selection
        sample_weights = model.state_dict()[next(iter(model.state_dict()))].flatten().numpy()
        target_compression = 10  # Aim for 10x compression
        K, U = adaptive_parameter_selection(sample_weights, target_compression)
        await self.agent.llm_response(f"Selected parameters: K={K}, U={U}")

        # Compress the model
        compressed_model = await self.stream_compress_model(model)
        await self.agent.llm_response("Compressed model")

        # Benchmark compression
        metrics = await self.benchmark_compression(model, compressed_model)
        await self.agent.llm_response(f"Compression metrics: {metrics}")

        # Decompress the model
        decompressed_model = await self.stream_decompress_model(compressed_model, model)
        await self.agent.llm_response("Decompressed model")

        # Verify decompression
        with torch.no_grad():
            input = torch.randn(1, 10000)
            original_output = model(input)
            decompressed_output = decompressed_model(input)
            error = torch.mean((original_output - decompressed_output).abs())
            await self.agent.llm_response(f"Mean absolute error after decompression: {error.item():.6f}")

        return {
            'compressed_model': compressed_model,
            'decompressed_model': decompressed_model,
            'metrics': metrics,
            'error': error.item()
        }

    # Include the stream_compress_model, stream_decompress_model, and benchmark_compression methods
    # from HyperCompressionTask in hyperparameter_compression.py
    async def stream_compress_model(self, model: torch.nn.Module, chunk_size: int = 1000000) -> Dict[str, Any]:
        compressed_state_dict = {}

        for name, param in model.state_dict().items():
            if 'weight' in name and param.dim() > 1:
                ternary_weights = torch.sign(param).to(torch.int8)
                scale_factors = param.abs().mean().to(torch.float32)

                chunks = ternary_weights.split(chunk_size)
                compressed_chunks = []

                for chunk in chunks:
                    compressed_chunk = self.compressor.compress(chunk, scale_factors)
                    compressed_chunks.append(compressed_chunk)

                compressed_state_dict[name] = {
                    'chunks': compressed_chunks,
                    'original_shape': param.shape
                }
            else:
                compressed_state_dict[name] = param

        return compressed_state_dict

    async def stream_decompress_model(self, compressed_state_dict: Dict[str, Any], original_model: torch.nn.Module) -> torch.nn.Module:
        decompressed_state_dict = {}

        for name, compressed_param in compressed_state_dict.items():
            if isinstance(compressed_param, dict) and 'chunks' in compressed_param:
                decompressed_chunks = [self.compressor.decompress(chunk) for chunk in compressed_param['chunks']]
                decompressed_param = torch.cat(decompressed_chunks).reshape(compressed_param['original_shape'])
                decompressed_state_dict[name] = decompressed_param
            else:
                decompressed_state_dict[name] = compressed_param

        original_model.load_state_dict(decompressed_state_dict)
        return original_model

    async def benchmark_compression(self, model: torch.nn.Module, compressed_model: Dict[str, Any]) -> Dict[str, float]:
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        compressed_size = sum(
            sum(chunk['thetas'].size * chunk['thetas'].itemsize + chunk['scale_factors'].size * chunk['scale_factors'].itemsize 
                for chunk in param['chunks'])
            if isinstance(param, dict) and 'chunks' in param
            else param.numel() * param.element_size()
            for param in compressed_model.values()
        )
        compression_ratio = original_size / compressed_size

        return {
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': compression_ratio
        }

# Example usage
if __name__ == "__main__":
    import asyncio
    from langroid.language_models.openai_gpt import OpenAIGPTConfig

    async def main():
        torch.manual_seed(0)
        np.random.seed(0)
        cp.random.seed(0)

        config = ChatAgentConfig(
            name="HyperparamterCompressionAgent",
            llm=OpenAIGPTConfig(chat_model="gpt-3.5-turbo"),
        )
        agent = ChatAgent(config)

        # Create a large model
        model = torch.nn.Sequential(
            torch.nn.Linear(10000, 5000),
            torch.nn.ReLU(),
            torch.nn.Linear(5000, 2500),
            torch.nn.ReLU(),
            torch.nn.Linear(2500, 1000)
        )

        task = HyperparamterCompressionTask(agent)
        results = await task.run(model)

        print(f"Compression ratio: {results['metrics']['compression_ratio']:.2f}")
        print(f"Mean absolute error: {results['error']:.6f}")

    asyncio.run(main())
