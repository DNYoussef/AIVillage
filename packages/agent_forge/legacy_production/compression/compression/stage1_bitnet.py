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
    """BitNet Linear layer with ternary quantization and gradual λ schedule."""

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
        self.lambda_val = 0.0  # Interpolation parameter (0=fp, 1=ternary)
        self.alpha = nn.Parameter(torch.ones(1))  # Scaling factor

    def quantize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to ternary {-1, 0, 1} with threshold-based sparsity."""
        # Calculate threshold for sparsity (consistent with bitnet.py compressor)
        threshold = weights.abs().mean()

        # Ternary quantization: zero if below threshold, sign if above
        mask = weights.abs() > threshold
        quantized = torch.sign(weights) * mask.float()

        return quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Gradual λ interpolation: (1-λ)*fp + λ*quantized
            quantized_weights = self.quantize_weights(self.weight_fp)
            effective_weights = (1 - self.lambda_val) * self.weight_fp + self.lambda_val * quantized_weights
        else:
            # During inference, use pure ternary weights (λ=1.0 path)
            effective_weights = self.quantize_weights(self.weight_fp)

        # Apply learned scaling factor
        effective_weights = effective_weights * self.alpha

        return F.linear(x, effective_weights, self.bias)

    def to_float(self) -> torch.Tensor:
        """Return quantized weights as float tensor."""
        return self.quantize_weights(self.weight_fp) * self.alpha


def convert_to_bitnet(model, threshold: float = 0.02, rmsnorm_post_attn: bool = True):
    """In-place replace every nn.Linear with BitNet implementation.

    Args:
        model: Model to convert
        threshold: Quantization threshold for sparsity (deprecated, now calculated dynamically)
        rmsnorm_post_attn: Whether to add RMSNorm after attention layers for stability
    """
    # Handle case where model itself is a Linear layer
    if isinstance(model, nn.Linear):
        bitnet_layer = BitNetLinear(model.in_features, model.out_features, bias=model.bias is not None)

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
                bitnet_layer = BitNetLinear(child.in_features, child.out_features, bias=child.bias is not None)

                # Initialize with original weights
                with torch.no_grad():
                    bitnet_layer.weight_fp.copy_(child.weight)
                    if child.bias is not None:
                        bitnet_layer.bias.copy_(child.bias)

                setattr(module, child_name, bitnet_layer)
            else:
                replace_linear_recursive(child, f"{name}.{child_name}" if name else child_name)

    # Replace all Linear layers with BitNet equivalents
    replace_linear_recursive(model)

    # Add RMSNorm after attention layers for stability if enabled
    if rmsnorm_post_attn:
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
    """Lambda schedule ramping λ: 0→1 over first 40% of training steps."""

    def __init__(self, total_steps: int, warmup_ratio: float = 0.4) -> None:
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.warmup_ratio = warmup_ratio
        self.current_lambda = 0.0

        print(f"[BitNet λ Schedule] Warmup steps: {self.warmup_steps}/{total_steps} ({warmup_ratio:.1%})")

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step <= self.warmup_steps:
            # Gradual λ ramp from 0 to 1 over warmup period
            self.current_lambda = state.global_step / self.warmup_steps if self.warmup_steps > 0 else 1.0
        else:
            # After warmup, keep λ=1.0 for pure ternary training
            self.current_lambda = 1.0

        # Update lambda_val for all BitNetLinear layers
        for module in kwargs["model"].modules():
            if isinstance(module, BitNetLinear):
                module.lambda_val = self.current_lambda

        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Log current lambda value for W&B monitoring
        if logs is not None:
            logs["lambda_val"] = self.current_lambda
            logs["bitnet_phase"] = "warmup" if self.current_lambda < 1.0 else "ternary"
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
        len(train_dataset) // (args.per_device_train_batch_size * torch.cuda.device_count())
    ) * args.num_train_epochs

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=args,
        callbacks=[GradualBitnetCallback(total_steps_est)],
    )

    trainer.train()
    return model
