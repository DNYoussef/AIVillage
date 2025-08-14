import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer, TrainerCallback, TrainingArguments


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for BitNet stabilization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return x / (norm + self.eps) * self.weight


class BitNetLinear(nn.Module):
    """BitNet Linear layer with ternary quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full precision weights for training
        self.weight_fp = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Quantization parameters
        self.lambda_val = 0.0  # Interpolation parameter
        self.alpha = nn.Parameter(torch.ones(1))  # Scaling factor

    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to ternary {-1, 0, 1}."""
        # Calculate threshold for sparsity
        threshold = weights.abs().mean()

        # Ternary quantization
        mask = weights.abs() > threshold
        quantized = torch.sign(weights) * mask.float()

        return quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # During training, interpolate between full precision and quantized
            quantized_weights = self.quantize_weights(self.weight_fp)
            effective_weights = (
                1 - self.lambda_val
            ) * self.weight_fp + self.lambda_val * quantized_weights
        else:
            # During inference, use quantized weights
            effective_weights = self.quantize_weights(self.weight_fp)

        # Scale weights
        effective_weights = effective_weights * self.alpha

        return F.linear(x, effective_weights, self.bias)

    def to_float(self) -> torch.Tensor:
        """Return quantized weights as float tensor."""
        return self.quantize_weights(self.weight_fp) * self.alpha


def convert_to_bitnet(model, threshold: float = 0.02):
    """In-place replace every nn.Linear with BitNet implementation."""
    # Handle case where model itself is a Linear layer
    if isinstance(model, nn.Linear):
        bitnet_layer = BitNetLinear(
            model.in_features, model.out_features, bias=model.bias is not None
        )

        # Initialize with original weights
        with torch.no_grad():
            bitnet_layer.weight_fp.copy_(model.weight)
            if model.bias is not None:
                bitnet_layer.bias.copy_(model.bias)

        return bitnet_layer

    def replace_linear_recursive(module, name="") -> None:
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with BitNet linear
                bitnet_layer = BitNetLinear(
                    child.in_features, child.out_features, bias=child.bias is not None
                )

                # Initialize with original weights
                with torch.no_grad():
                    bitnet_layer.weight_fp.copy_(child.weight)
                    if child.bias is not None:
                        bitnet_layer.bias.copy_(child.bias)

                setattr(module, child_name, bitnet_layer)
            else:
                replace_linear_recursive(
                    child, f"{name}.{child_name}" if name else child_name
                )

    # Use custom implementation directly since bitsandbytes doesn't have LinearBitNet
    replace_linear_recursive(model)

    # Add RMSNorm after attention layers for stability
    add_rmsnorm_to_attention(model)

    return model


def add_rmsnorm_to_attention(model) -> None:
    """Add RMSNorm layers after attention blocks for stability."""
    for name, module in model.named_modules():
        if "attention" in name.lower() or "attn" in name.lower():
            # Add RMSNorm after attention if it doesn't exist
            if hasattr(module, "out_proj") and not hasattr(module, "norm"):
                hidden_size = module.out_proj.out_features
                module.norm = RMSNorm(hidden_size)

                # Wrap the forward method to apply norm
                original_forward = module.forward

                def forward_with_norm(*args, **kwargs):
                    output = original_forward(*args, **kwargs)
                    if isinstance(output, tuple):
                        return (module.norm(output[0]), *output[1:])
                    return module.norm(output)

                module.forward = forward_with_norm


class GradualBitnetCallback(TrainerCallback):
    """Lambda schedule ramping 0->1 over first 40% of steps."""

    def __init__(self, total_steps: int, warmup_ratio: float = 0.4) -> None:
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)

    def on_step_begin(self, args, state, control, **kwargs):
        lam = min(state.global_step / self.warmup_steps, 1.0)
        for m in kwargs["model"].modules():
            if isinstance(m, bnb.nn.LinearBitNet | BitNetLinear):
                m.lambda_val = lam
        return control


def apply_hf_bitnet_finetune(model, train_dataset, config):
    """Finetune model with HuggingFace BitNet recipe."""
    model = convert_to_bitnet(model, threshold=config.bitnet_zero_threshold)

    args = TrainingArguments(
        output_dir="checkpoints/bitnet",
        per_device_train_batch_size=config.bitnet_batch_size,
        num_train_epochs=config.bitnet_finetuning_epochs,
        learning_rate=config.bitnet_learning_rate,
        fp16=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=50,
        save_steps=0,
        optim="adamw_torch",
    )

    total_steps_est = (
        len(train_dataset)
        // (args.per_device_train_batch_size * torch.cuda.device_count())
    ) * args.num_train_epochs

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=args,
        callbacks=[GradualBitnetCallback(total_steps_est)],
    )

    trainer.train()
    return model
