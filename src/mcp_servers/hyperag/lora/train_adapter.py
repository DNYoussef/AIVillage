#!/usr/bin/env python3
"""HypeRAG LoRA Adapter Trainer.

Trains domain-specific LoRA adapters using PEFT (Parameter-Efficient Fine-Tuning).
Integrates with Guardian Gate for signing and validation.
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class RepairDataset(Dataset):
    """Dataset for knowledge graph repair tasks."""

    def __init__(self, data_path: Path, tokenizer, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load JSONL data
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Combine prompt and completion
        text = f"{example['prompt']}\n\nResponse:\n{example['completion']}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels (same as input_ids for causal LM)
        encoding["labels"] = encoding["input_ids"].clone()

        return {k: v.squeeze() for k, v in encoding.items()}


class LoRATrainer:
    def __init__(self, base_model: str = "microsoft/phi-2", device: str = "cuda") -> None:
        self.base_model_name = base_model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.peft_model = None

    def load_base_model(self) -> None:
        """Load the base model and tokenizer."""
        logger.info(f"Loading base model: {self.base_model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Move to device if needed
        if self.device == "cpu":
            self.model = self.model.to(self.device)

    def create_lora_model(self, lora_config: dict[str, Any] | None = None) -> None:
        """Create LoRA model with specified configuration."""
        if lora_config is None:
            lora_config = {
                "r": 16,  # LoRA rank
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM,
            }

        logger.info(f"Creating LoRA model with config: {lora_config}")

        # Create LoRA configuration
        peft_config = LoraConfig(**lora_config)

        # Get PEFT model
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

    def train(
        self,
        train_data_path: Path,
        eval_data_path: Path | None = None,
        output_dir: Path = Path("./lora_output"),
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 50,
        save_steps: int = 500,
    ):
        """Train the LoRA adapter."""
        # Create datasets
        train_dataset = RepairDataset(train_data_path, self.tokenizer)
        eval_dataset = RepairDataset(eval_data_path, self.tokenizer) if eval_data_path else None

        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            save_total_limit=2,
            load_best_model_at_end=bool(eval_dataset),
            metric_for_best_model="eval_loss" if eval_dataset else None,
            fp16=self.device == "cuda",
            push_to_hub=False,
            report_to=["tensorboard"],
            logging_dir=str(output_dir / "logs"),
        )

        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save the final model
        self.peft_model.save_pretrained(output_dir / "final_adapter")
        self.tokenizer.save_pretrained(output_dir / "final_adapter")

        return trainer.state.log_history

    def evaluate(self, eval_data_path: Path) -> dict[str, float]:
        """Evaluate the trained adapter."""
        logger.info("Evaluating adapter...")

        eval_dataset = RepairDataset(eval_data_path, self.tokenizer)
        eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

        self.peft_model.eval()

        all_predictions = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in eval_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.peft_model(**inputs)

                # Calculate perplexity
                loss = outputs.loss
                total_loss += loss.item()

                # Get predictions
                predictions = outputs.logits.argmax(dim=-1)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(inputs["labels"].cpu().numpy().flatten())

        # Calculate metrics
        perplexity = np.exp(total_loss / len(eval_loader))

        # Filter out padding tokens for accuracy
        mask = np.array(all_labels) != self.tokenizer.pad_token_id
        filtered_predictions = np.array(all_predictions)[mask]
        filtered_labels = np.array(all_labels)[mask]

        accuracy = accuracy_score(filtered_labels, filtered_predictions)

        metrics = {
            "perplexity": float(perplexity),
            "accuracy": float(accuracy),
            "eval_loss": float(total_loss / len(eval_loader)),
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def compute_adapter_hash(self, adapter_path: Path) -> str:
        """Compute SHA256 hash of the adapter weights."""
        sha256_hash = hashlib.sha256()

        # Hash all adapter weight files
        for weight_file in sorted(adapter_path.glob("*.bin")):
            with open(weight_file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def generate_registry_entry(self, adapter_path: Path, domain: str, metrics: dict[str, float]) -> dict[str, Any]:
        """Generate a registry entry for the trained adapter."""
        adapter_hash = self.compute_adapter_hash(adapter_path)

        entry = {
            "adapter_id": f"{domain}_lora_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "sha256": adapter_hash,
            "domain": domain,
            "base_model": self.base_model_name,
            "metrics": metrics,
            "training_config": {
                "peft_type": "lora",
                "r": self.peft_model.peft_config["default"].r,
                "lora_alpha": self.peft_model.peft_config["default"].lora_alpha,
                "target_modules": self.peft_model.peft_config["default"].target_modules,
                "lora_dropout": self.peft_model.peft_config["default"].lora_dropout,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "guardian_signature": None,  # To be filled by Guardian
        }

        return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA adapter for HypeRAG")
    parser.add_argument("--train-data", required=True, type=Path, help="Path to training JSONL file")
    parser.add_argument("--eval-data", type=Path, help="Path to evaluation JSONL file")
    parser.add_argument("--domain", required=True, help="Domain name for the adapter")
    parser.add_argument(
        "--base-model",
        default="microsoft/phi-2",
        help="Base model to use (default: microsoft/phi-2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./lora_output"),
        help="Output directory for trained adapter",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create trainer
    trainer = LoRATrainer(base_model=args.base_model, device=args.device)

    # Load base model
    trainer.load_base_model()

    # Create LoRA model
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }
    trainer.create_lora_model(lora_config)

    # Train
    trainer.train(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Evaluate if eval data provided
    metrics = {}
    if args.eval_data:
        metrics = trainer.evaluate(args.eval_data)

    # Generate registry entry
    adapter_path = args.output_dir / "final_adapter"
    registry_entry = trainer.generate_registry_entry(adapter_path, args.domain, metrics)

    # Save registry entry
    registry_path = adapter_path / "registry_entry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry_entry, f, indent=2)

    logger.info(f"Training complete. Adapter saved to: {adapter_path}")
    logger.info(f"Registry entry saved to: {registry_path}")

    # Print summary
    print("\nTraining Summary:")
    print(f"  Domain: {args.domain}")
    print(f"  Base Model: {args.base_model}")
    print(f"  Adapter Hash: {registry_entry['sha256']}")
    if metrics:
        print("  Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"    - {k}: {v:.4f}")


if __name__ == "__main__":
    main()
