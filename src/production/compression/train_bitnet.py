#!/usr/bin/env python3
"""
BitNet training with gradual λ schedule.
Trains a model with self-generated data using the λ warmup approach.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import wandb

from .compression.stage1_bitnet import GradualBitnetCallback, convert_to_bitnet


class BitNetDataset:
    """Dataset handler for BitNet training."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        """Initialize dataset from JSONL file."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_jsonl(jsonl_path)

    def load_jsonl(self, path: str) -> list[dict]:
        """Load data from JSONL file."""
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def format_sample(self, sample: dict) -> str:
        """Format sample as instruction-following text."""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")

        if input_text:
            formatted = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            formatted = (
                f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
            )

        return formatted

    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        # Format all examples
        texts = [self.format_sample(sample) for sample in examples]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )

        # Labels are the same as input_ids for language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def create_dataset(self) -> Dataset:
        """Create HuggingFace Dataset object."""
        dataset = Dataset.from_list(self.data)
        tokenized_dataset = dataset.map(
            self.tokenize_function, batched=True, remove_columns=dataset.column_names
        )
        return tokenized_dataset


class BitNetTrainer:
    """BitNet trainer with gradual λ schedule."""

    def __init__(
        self,
        base_model_path: str,
        output_dir: str,
        lambda_warmup_frac: float = 0.4,
        rmsnorm_post_attn: bool = True,
    ):
        """Initialize BitNet trainer.

        Args:
            base_model_path: Path to base model directory
            output_dir: Output directory for trained model
            lambda_warmup_frac: Fraction of steps for λ warmup
            rmsnorm_post_attn: Whether to add RMSNorm after attention
        """
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.lambda_warmup_frac = lambda_warmup_frac
        self.rmsnorm_post_attn = rmsnorm_post_attn

        # Set up environment
        self.setup_environment()

        # Load model and tokenizer
        print(f"Loading base model from: {base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Convert to BitNet
        print("Converting model to BitNet...")
        self.model = convert_to_bitnet(
            self.model, rmsnorm_post_attn=self.rmsnorm_post_attn
        )
        print("BitNet conversion complete")

    def setup_environment(self):
        """Set up environment variables."""
        defaults = {
            "AIV_ROOT": "D:\\AIVillage",
            "AIV_MODELS_DIR": "D:\\AIVillage\\models",
            "AIV_ARTIFACTS_DIR": "D:\\AIVillage\\artifacts",
            "WANDB_DIR": "D:\\AIVillage\\wandb",
            "WANDB_MODE": "offline",
        }

        for key, value in defaults.items():
            if key not in os.environ:
                os.environ[key] = value

        # Create directories
        Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        dataset_path: str,
        steps: int = 1000,
        batch_size: int = 2,
        grad_accum: int = 8,
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.1,
    ):
        """Train the BitNet model.

        Args:
            dataset_path: Path to training dataset JSONL
            steps: Number of training steps
            batch_size: Per-device batch size
            grad_accum: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_ratio: Learning rate warmup ratio
        """
        # Load dataset
        print(f"Loading dataset from: {dataset_path}")
        dataset_handler = BitNetDataset(dataset_path, self.tokenizer)
        train_dataset = dataset_handler.create_dataset()
        print(f"Dataset loaded: {len(train_dataset)} samples")

        # Set up W&B
        model_name = Path(self.base_model_path).name
        wandb.init(
            project="AIVillage-BitNet158",
            name=f"bitnet-{model_name}",
            tags=["stage:bitnet158", "source:selfgen", f"model:{model_name}"],
            config={
                "base_model": self.base_model_path,
                "dataset_size": len(train_dataset),
                "lambda_warmup_frac": self.lambda_warmup_frac,
                "rmsnorm_post_attn": self.rmsnorm_post_attn,
                "steps": steps,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "learning_rate": learning_rate,
            },
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=1,  # Use max_steps instead
            max_steps=steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            fp16=True,
            gradient_checkpointing=True,
            dataloader_drop_last=True,
            logging_steps=10,
            save_steps=steps // 4,  # Save 4 checkpoints during training
            save_total_limit=2,
            report_to="wandb",
            optim="adamw_torch",
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8
        )

        # Gradual BitNet callback
        callback = GradualBitnetCallback(
            total_steps=steps, warmup_ratio=self.lambda_warmup_frac
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[callback],
        )

        # Train
        print("Starting BitNet training...")
        trainer.train()

        # Save final model
        print("Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        # Save training manifest
        manifest = {
            "base_model": self.base_model_path,
            "dataset_path": dataset_path,
            "training_steps": steps,
            "lambda_warmup_frac": self.lambda_warmup_frac,
            "rmsnorm_post_attn": self.rmsnorm_post_attn,
            "final_lambda": 1.0,  # Always 1.0 after training
            "compression_ready": True,  # Ready for bitnet.py::compress()
            "model_type": "bitnet158",
        }

        with open(self.output_dir / "training_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Training complete! Model saved to: {self.output_dir}")
        wandb.finish()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train BitNet model with gradual λ schedule"
    )
    parser.add_argument(
        "--base_model", required=True, help="Path to base model directory"
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to training dataset JSONL"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for trained model"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of training steps"
    )
    parser.add_argument("--bsz", type=int, default=2, help="Per-device batch size")
    parser.add_argument(
        "--grad_accum", type=int, default=8, help="Gradient accumulation steps"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--lambda_warmup_frac",
        type=float,
        default=0.4,
        help="Fraction of steps for λ warmup",
    )
    parser.add_argument(
        "--rmsnorm_post_attn",
        type=int,
        default=1,
        help="Add RMSNorm after attention (1=True, 0=False)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.base_model).exists():
        print(f"Error: Base model path does not exist: {args.base_model}")
        return 1

    if not Path(args.dataset).exists():
        print(f"Error: Dataset path does not exist: {args.dataset}")
        return 1

    try:
        # Initialize trainer
        trainer = BitNetTrainer(
            base_model_path=args.base_model,
            output_dir=args.out_dir,
            lambda_warmup_frac=args.lambda_warmup_frac,
            rmsnorm_post_attn=bool(args.rmsnorm_post_attn),
        )

        # Train model
        trainer.train(
            dataset_path=args.dataset,
            steps=args.steps,
            batch_size=args.bsz,
            grad_accum=args.grad_accum,
            learning_rate=args.lr,
        )

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
