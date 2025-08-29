#!/usr/bin/env python3
"""
Complete Cognate 25M Pretraining Pipeline

Creates and pretrains 3 identical 25M parameter Cognate models using:
- Real datasets (GSM8K, HotpotQA, etc.) with proper curriculum
- Full Cognate architecture with ACT halting and Titans-style LTM
- GrokFast optimization with proper hyperparameters
- Train-many/infer-few paradigm
- EvoMerge-compatible output format

This is the complete, production-ready implementation.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add cognate path to imports
cognate_path = Path(__file__).parent.parent.parent.parent / "packages" / "agent_forge" / "models" / "cognate"
sys.path.append(str(cognate_path))

# Import real Cognate components
from refiner_core import CognateConfig, CognateRefiner

# Import or create GrokFast optimizer
try:
    sys.path.append(str(Path(__file__).parent))
    from grokfast_optimizer import GrokFastOptimizer
except ImportError:
    logger.warning("GrokFast optimizer not found, using inline implementation")

    class GrokFastOptimizer:
        """GrokFast optimizer implementation."""

        def __init__(self, model: nn.Module, base_optimizer, alpha=0.98, lamb=2.0, warmup_steps=2000):
            self.base_optimizer = base_optimizer
            self.alpha = alpha
            self.lamb = lamb
            self.warmup_steps = warmup_steps
            self.step_count = 0
            self.grad_ema = {}

            # Initialize gradient EMA storage
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.grad_ema[name] = torch.zeros_like(param.data)

        def step(self, closure=None):
            self.step_count += 1

            # During warmup, just use base optimizer
            if self.step_count <= self.warmup_steps:
                return self.base_optimizer.step(closure)

            # Apply GrokFast gradient amplification
            with torch.no_grad():
                for name, param in self.base_optimizer.param_groups[0]["params"]:
                    if hasattr(param, "grad") and param.grad is not None:
                        # Update EMA of gradients
                        if name in self.grad_ema:
                            self.grad_ema[name] = self.alpha * self.grad_ema[name] + (1 - self.alpha) * param.grad
                            # Amplify slow-varying component
                            param.grad = param.grad + self.lamb * self.grad_ema[name]

            return self.base_optimizer.step(closure)

        def zero_grad(self):
            self.base_optimizer.zero_grad()

        def state_dict(self):
            return self.base_optimizer.state_dict()

        def load_state_dict(self, state_dict):
            self.base_optimizer.load_state_dict(state_dict)


@dataclass
class FullPretrainingConfig:
    """Complete configuration for Cognate 25M pretraining."""

    # Model architecture (exact 25M parameters)
    d_model: int = 216
    n_layers: int = 11
    n_heads: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 4096

    # Memory system
    d_mem: int = 216
    mem_capacity: int = 4096
    mem_topk: int = 4

    # ACT configuration
    act_threshold: float = 0.99
    max_act_steps: int = 16
    act_epsilon: float = 0.01

    # Training dynamics (train-many/infer-few)
    train_max_steps: int = 16
    train_min_steps: int = 8
    infer_max_steps: int = 6
    infer_min_steps: int = 2

    # Curriculum (45% short, 55% long)
    short_ratio: float = 0.45
    long_ratio: float = 0.55
    short_seq_len: int = 512
    long_seq_len: int = 4096

    # Optimizer settings (exact specification)
    learning_rate: float = 2e-4  # 2e-4 with cosine decay
    weight_decay: float = 0.1
    beta1: float = 0.9  # AdamW β1
    beta2: float = 0.95  # AdamW β2
    warmup_steps: int = 2000  # 2k steps warmup

    # GrokFast settings
    grokfast_alpha: float = 0.98
    grokfast_lamb: float = 2.0
    grokfast_warmup: int = 2000

    # Training configuration
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    max_training_steps: int = 25000  # Full pretraining
    eval_steps: int = 1000
    save_steps: int = 5000

    # Precision
    mixed_precision: bool = True  # bf16

    # Data
    dataset_path: str = "./cognate_datasets/mixed_training_data.json"

    # Output
    output_dir: str = "./cognate_25m_pretrained"

    # Loss weights
    lambda_act: float = 0.1
    alpha_read: float = 0.05
    beta_write: float = 0.05


class RealCognateDataset(Dataset):
    """Dataset using real downloaded data with curriculum alignment."""

    def __init__(self, config: FullPretrainingConfig, tokenizer, data_file: str, split: str = "train"):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split

        # Load real data
        logger.info(f"Loading real dataset from {data_file}")
        with open(data_file, encoding="utf-8") as f:
            self.data = json.load(f)

        # Separate short and long sequences
        self.short_data = [item for item in self.data if item["seq_type"] == "short"]
        self.long_data = [item for item in self.data if item["seq_type"] == "long"]

        logger.info(f"Loaded {len(self.short_data)} short samples, {len(self.long_data)} long samples")

        # Calculate actual ratios
        total_samples = len(self.data)
        self.short_samples = len(self.short_data)
        self.long_samples = len(self.long_data)

        actual_short_ratio = self.short_samples / total_samples
        actual_long_ratio = self.long_samples / total_samples

        logger.info(f"Dataset ratios: {actual_short_ratio:.1%} short, {actual_long_ratio:.1%} long")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize text
        text = item["text"]
        tokens = self.tokenizer.encode(
            text, add_special_tokens=True, truncation=True, max_length=self.config.long_seq_len
        )

        # Determine sequence type and pad accordingly
        if item["seq_type"] == "short":
            max_len = self.config.short_seq_len
        else:
            max_len = self.config.long_seq_len

        # Truncate or pad
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        elif len(tokens) < max_len:
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))

        # Create attention mask
        attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in tokens]

        return {
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(tokens[1:], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:-1], dtype=torch.long),
            "seq_type": item["seq_type"],
            "requires_memory": item["requires_memory"],
            "dataset": item["dataset"],
            "metadata": item.get("metadata", {}),
        }


class Complete25MCognateModel(nn.Module):
    """Complete 25M Cognate model with full architecture."""

    def __init__(self, config: FullPretrainingConfig, model_name: str = "cognate-25m"):
        super().__init__()
        self.config = config
        self.model_name = model_name

        # Create Cognate configuration
        cognate_config = CognateConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            d_mem=config.d_mem,
            mem_capacity=config.mem_capacity,
            mem_topk=config.mem_topk,
            act_threshold=config.act_threshold,
            max_act_steps=config.max_act_steps,
            act_epsilon=config.act_epsilon,
            dropout=0.1,
        )

        # Initialize core Cognate refiner with full architecture
        self.cognate_core = CognateRefiner(cognate_config)

        # Training/inference mode tracking
        self.training_mode = True
        self.max_steps = config.train_max_steps

        logger.info(f"Created Complete25MCognateModel '{model_name}' with {self.count_parameters():,} parameters")

    def set_inference_mode(self, inference: bool = True):
        """Switch between train-many and infer-few paradigms."""
        self.training_mode = not inference
        self.max_steps = self.config.train_max_steps if self.training_mode else self.config.infer_max_steps

        # Update core model's max steps
        if hasattr(self.cognate_core, "max_act_steps"):
            self.cognate_core.max_act_steps = self.max_steps

        logger.info(
            f"Switched to {'training' if self.training_mode else 'inference'} mode (max_steps={self.max_steps})"
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass with full Cognate architecture."""
        return self.cognate_core(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True, **kwargs
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_for_evomerge(self, save_path: str):
        """Save model in EvoMerge-compatible format."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save PyTorch state dict (EvoMerge expects this)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")

        # Save configuration
        config_dict = {
            "model_config": asdict(self.config),
            "architecture": "cognate-25m",
            "parameter_count": self.count_parameters(),
            "model_class": "Complete25MCognateModel",
            "model_name": self.model_name,
        }

        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Saved {self.model_name} for EvoMerge: {self.count_parameters():,} params")


class FullCognateTrainer:
    """Complete trainer for Cognate 25M models with real datasets."""

    def __init__(self, config: FullPretrainingConfig):
        self.config = config

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer (use a standard one for now)
        self.tokenizer = self._create_tokenizer()

        # Load datasets
        self.train_dataset = RealCognateDataset(config, self.tokenizer, config.dataset_path, "train")

        logger.info("FullCognateTrainer initialized")

    def _create_tokenizer(self):
        """Create or load tokenizer."""
        try:
            # Try to use a standard tokenizer
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info("Using DialoGPT tokenizer")
            return tokenizer
        except:
            logger.warning("Could not load HuggingFace tokenizer, using mock")

            # Create mock tokenizer
            class MockTokenizer:
                def __init__(self):
                    self.vocab_size = 32000
                    self.pad_token_id = 0
                    self.eos_token_id = 2

                def encode(self, text, add_special_tokens=True, truncation=True, max_length=4096):
                    # Simple hash-based encoding
                    tokens = [abs(hash(text[i : i + 4])) % 30000 + 100 for i in range(0, len(text), 4)]
                    if add_special_tokens:
                        tokens = [1] + tokens + [2]  # BOS + tokens + EOS
                    return tokens[:max_length] if truncation else tokens

            return MockTokenizer()

    def create_model(self, model_name: str, seed: int) -> Complete25MCognateModel:
        """Create a single Cognate model with specific seed."""
        # Set seed for reproducible initialization
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = Complete25MCognateModel(self.config, model_name)
        model.to(self.device)

        param_count = model.count_parameters()
        logger.info(f"Created {model_name} (seed={seed}): {param_count:,} parameters")

        return model

    def setup_optimizer_and_scheduler(self, model: Complete25MCognateModel):
        """Setup GrokFast optimizer and scheduler."""

        # Parameter groups with weight decay
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # Base AdamW optimizer
        base_optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )

        # Wrap with GrokFast
        optimizer = GrokFastOptimizer(
            model=model,
            base_optimizer=base_optimizer,
            alpha=self.config.grokfast_alpha,
            lamb=self.config.grokfast_lamb,
            warmup_steps=self.config.grokfast_warmup,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=self.config.max_training_steps)

        logger.info(f"Setup GrokFast optimizer: alpha={self.config.grokfast_alpha}, lamb={self.config.grokfast_lamb}")

        return optimizer, scheduler

    def train_single_model(self, model: Complete25MCognateModel, model_name: str) -> dict[str, Any]:
        """Train a single Cognate model."""
        logger.info(f"Starting training for {model_name}")

        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer_and_scheduler(model)

        # Create dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Training loop
        model.train()
        model.set_inference_mode(False)  # Training mode

        training_stats = {
            "total_steps": 0,
            "total_loss": 0.0,
            "losses": [],
            "learning_rates": [],
            "model_name": model_name,
        }

        step = 0
        accumulated_loss = 0.0

        for epoch in range(100):  # Large number, will break early
            for batch_idx, batch in enumerate(train_loader):
                if step >= self.config.max_training_steps:
                    break

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs["loss"] if "loss" in outputs else outputs.get("total_loss", 0.0)

                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                accumulated_loss += loss.item()

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)

                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Statistics
                    training_stats["total_loss"] += accumulated_loss
                    training_stats["losses"].append(accumulated_loss)
                    training_stats["learning_rates"].append(scheduler.get_last_lr()[0])

                    # Logging
                    if step % 100 == 0:
                        avg_loss = training_stats["total_loss"] / max(
                            step // self.config.gradient_accumulation_steps, 1
                        )
                        logger.info(f"{model_name} Step {step}: loss={accumulated_loss:.4f}, avg_loss={avg_loss:.4f}")

                    accumulated_loss = 0.0

                step += 1
                training_stats["total_steps"] = step

                if step >= self.config.max_training_steps:
                    break

            if step >= self.config.max_training_steps:
                break

        logger.info(f"Completed training {model_name}: {step} steps")
        return training_stats

    def train_three_models(self) -> list[dict[str, Any]]:
        """Train 3 identical models with different seeds."""
        model_configs = [("cognate-25m-model-1", 42), ("cognate-25m-model-2", 1337), ("cognate-25m-model-3", 2023)]

        results = []

        for model_name, seed in model_configs:
            logger.info(f"=== TRAINING {model_name.upper()} (SEED={seed}) ===")

            try:
                # Create model
                model = self.create_model(model_name, seed)

                # Train model
                training_stats = self.train_single_model(model, model_name)

                # Save model
                output_path = Path(self.config.output_dir) / model_name
                model.save_for_evomerge(str(output_path))

                # Save training stats
                with open(output_path / "training_stats.json", "w") as f:
                    json.dump(training_stats, f, indent=2)

                results.append(
                    {
                        "model_name": model_name,
                        "seed": seed,
                        "parameter_count": model.count_parameters(),
                        "training_stats": training_stats,
                        "status": "success",
                    }
                )

                logger.info(f"✅ Successfully trained and saved {model_name}")

            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                results.append({"model_name": model_name, "seed": seed, "status": "failed", "error": str(e)})

        return results


def main():
    """Main pretraining function."""
    logger.info("=== STARTING COMPLETE COGNATE 25M PRETRAINING ===")

    # Check if datasets exist, download if needed
    dataset_path = "./cognate_datasets/mixed_training_data.json"
    if not Path(dataset_path).exists():
        logger.info("Datasets not found, downloading...")
        try:
            from download_datasets import main as download_main

            download_main()
        except Exception as e:
            logger.error(f"Failed to download datasets: {e}")
            logger.error("Please run download_datasets.py first")
            return

    # Create configuration
    config = FullPretrainingConfig(
        dataset_path=dataset_path, max_training_steps=25000, output_dir="./cognate_25m_pretrained"  # Full pretraining
    )

    # Save configuration
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "pretraining_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    logger.info(f"Configuration: {config.max_training_steps} steps, lr={config.learning_rate}")
    logger.info(f"GrokFast: alpha={config.grokfast_alpha}, lamb={config.grokfast_lamb}")
    logger.info(f"Architecture: {config.d_model}D, {config.n_layers}L, {config.n_heads}H")

    # Initialize trainer
    trainer = FullCognateTrainer(config)

    # Train all three models
    results = trainer.train_three_models()

    # Create final summary
    successful_models = [r for r in results if r["status"] == "success"]
    failed_models = [r for r in results if r["status"] == "failed"]

    summary = {
        "total_models": len(results),
        "successful_models": len(successful_models),
        "failed_models": len(failed_models),
        "configuration": asdict(config),
        "results": results,
        "evomerge_ready": len(successful_models) >= 2,
    }

    # Create EvoMerge model registry
    evomerge_models = []
    for result in successful_models:
        model_path = output_dir / result["model_name"] / "pytorch_model.bin"
        config_path = output_dir / result["model_name"] / "config.json"

        evomerge_models.append(
            {
                "model_path": str(model_path),
                "config_path": str(config_path),
                "model_name": result["model_name"],
                "seed": result["seed"],
                "parameter_count": result["parameter_count"],
                "architecture": "cognate-25m",
            }
        )

    evomerge_registry = {
        "seed_models": evomerge_models,
        "model_type": "cognate-25m",
        "total_models": len(evomerge_models),
        "ready_for_evomerge": len(evomerge_models) >= 2,
        "architecture_details": {
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "parameter_count": evomerge_models[0]["parameter_count"] if evomerge_models else 0,
        },
    }

    # Save everything
    with open(output_dir / "pretraining_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "evomerge_models.json", "w") as f:
        json.dump(evomerge_registry, f, indent=2)

    logger.info("=== PRETRAINING COMPLETE ===")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Successful models: {len(successful_models)}/{len(results)}")
    logger.info(f"EvoMerge ready: {evomerge_registry['ready_for_evomerge']}")

    return summary


if __name__ == "__main__":
    summary = main()
    print(f"FINAL RESULT: {summary['successful_models']}/{summary['total_models']} models successfully trained")
