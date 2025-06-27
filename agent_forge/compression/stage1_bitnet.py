import torch
import bitsandbytes as bnb
from transformers import TrainerCallback, TrainingArguments, Trainer


def convert_to_bitnet(model, threshold: float = 0.02):
    """In-place replace every nn.Linear with bnb.nn.LinearBitNet."""
    if not hasattr(bnb.nn, "LinearBitNet"):
        raise ImportError("bitsandbytes LinearBitNet unavailable")
    bnb.nn.LinearBitNet.convert(model, threshold=threshold)
    return model


class GradualBitnetCallback(TrainerCallback):
    """Lambda schedule ramping 0->1 over first 40% of steps."""

    def __init__(self, total_steps: int, warmup_ratio: float = 0.4):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)

    def on_step_begin(self, args, state, control, **kwargs):
        lam = min(state.global_step / self.warmup_steps, 1.0)
        for m in kwargs["model"].modules():
            if isinstance(m, bnb.nn.LinearBitNet):
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
