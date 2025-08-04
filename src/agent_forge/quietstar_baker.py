#!/usr/bin/env python3
"""Quiet-STaR Baker - Reasoning Token Injection and Baking.

Implements Quiet-STaR (Self-Taught Reasoner) methodology:
- Injects thought tokens (<|startofthought|> / <|endofthought|>) during forward pass
- A/B tests reasoning improvements on evaluation set
- Bakes successful reasoning patterns into model weights
- Full W&B tracking with reasoning metrics

Based on "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking"
"""

import asyncio
import json
import logging
from pathlib import Path
import time
from typing import Any

import click
from datasets import load_dataset
import numpy as np
from pydantic import BaseModel, Field, validator
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================


class QuietSTaRConfig(BaseModel):
    """Configuration for Quiet-STaR baking."""

    # Model configuration
    model_path: str = Field(..., description="Path to champion model from EvoMerge")
    output_path: str = Field(..., description="Path for baked model output")
    tokenizer_path: str | None = Field(
        None, description="Custom tokenizer path (defaults to model_path)"
    )

    # Thought tokens
    start_thought_token: str = Field(default="<|startofthought|>")
    end_thought_token: str = Field(default="<|endofthought|>")
    max_thought_length: int = Field(default=64, ge=8, le=256)
    thought_probability: float = Field(default=0.5, ge=0.0, le=1.0)

    # Evaluation configuration
    eval_dataset: str = Field(default="gsm8k", description="Evaluation dataset")
    eval_samples: int = Field(default=100, ge=10, le=1000)
    eval_batch_size: int = Field(default=4, ge=1, le=32)

    # A/B testing configuration
    ab_test_rounds: int = Field(default=3, ge=1, le=10)
    significance_threshold: float = Field(default=0.05, ge=0.01, le=0.1)
    min_improvement: float = Field(default=0.02, ge=0.0, le=0.5)

    # Fine-tuning configuration
    learning_rate: float = Field(default=1e-5, ge=1e-7, le=1e-3)
    num_epochs: int = Field(default=3, ge=1, le=10)
    warmup_steps: int = Field(default=100, ge=0, le=1000)
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.1)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=16)

    # System configuration
    device: str = Field(default="auto")
    mixed_precision: bool = Field(default=True)
    seed: int = Field(default=42)

    # W&B configuration
    wandb_project: str = Field(default="agent-forge")
    wandb_entity: str | None = None
    wandb_tags: list[str] = Field(default_factory=lambda: ["quietstar", "reasoning"])

    @validator("device")
    def validate_device(self, v):
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v


# ============================================================================
# Thought Token Injection
# ============================================================================


class ThoughtInjector(nn.Module):
    """Injects thought tokens into model forward pass."""

    def __init__(self, model: nn.Module, tokenizer, config: QuietSTaRConfig) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Add special tokens
        self.add_thought_tokens()

        # Thought token IDs
        self.start_thought_id = self.tokenizer.convert_tokens_to_ids(
            config.start_thought_token
        )
        self.end_thought_id = self.tokenizer.convert_tokens_to_ids(
            config.end_thought_token
        )

        logger.info(
            f"Thought tokens initialized: {config.start_thought_token} ({self.start_thought_id}), "
            f"{config.end_thought_token} ({self.end_thought_id})"
        )

    def add_thought_tokens(self) -> None:
        """Add thought tokens to tokenizer."""
        special_tokens = {
            "additional_special_tokens": [
                self.config.start_thought_token,
                self.config.end_thought_token,
            ]
        }

        num_added = self.tokenizer.add_special_tokens(special_tokens)

        if num_added > 0:
            # Resize model embeddings
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {num_added} special tokens and resized embeddings")

    def inject_thoughts(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject thought tokens into input sequences."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Determine injection points (after each sentence or at random intervals)
        injection_points = self.find_injection_points(input_ids)

        new_sequences = []
        new_masks = []

        for b in range(batch_size):
            seq = input_ids[b]
            mask = attention_mask[b]
            points = injection_points[b]

            if len(points) == 0 or np.random.random() > self.config.thought_probability:
                # No injection for this sequence
                new_sequences.append(seq)
                new_masks.append(mask)
                continue

            # Build new sequence with thought tokens
            new_seq = []
            new_mask = []
            last_idx = 0

            for point in points:
                # Add original tokens up to injection point
                new_seq.extend(seq[last_idx:point].tolist())
                new_mask.extend(mask[last_idx:point].tolist())

                # Add thought tokens
                thought_length = np.random.randint(
                    8, min(self.config.max_thought_length, 32)
                )
                new_seq.append(self.start_thought_id)
                new_mask.append(1)

                # Add placeholder thought (will be generated by model)
                new_seq.extend([self.tokenizer.pad_token_id] * thought_length)
                new_mask.extend([0] * thought_length)  # Mask out during training

                new_seq.append(self.end_thought_id)
                new_mask.append(1)

                last_idx = point

            # Add remaining tokens
            new_seq.extend(seq[last_idx:].tolist())
            new_mask.extend(mask[last_idx:].tolist())

            new_sequences.append(torch.tensor(new_seq, device=device))
            new_masks.append(torch.tensor(new_mask, device=device))

        # Pad sequences to same length
        max_len = max(len(seq) for seq in new_sequences)
        padded_sequences = []
        padded_masks = []

        for seq, mask in zip(new_sequences, new_masks, strict=False):
            pad_len = max_len - len(seq)
            if pad_len > 0:
                seq = F.pad(seq, (0, pad_len), value=self.tokenizer.pad_token_id)
                mask = F.pad(mask, (0, pad_len), value=0)
            padded_sequences.append(seq)
            padded_masks.append(mask)

        return torch.stack(padded_sequences), torch.stack(padded_masks)

    def find_injection_points(self, input_ids: torch.Tensor) -> list[list[int]]:
        """Find suitable points for thought injection."""
        batch_size = input_ids.shape[0]
        injection_points = []

        for b in range(batch_size):
            seq = input_ids[b]
            points = []

            # Find sentence boundaries (periods, questions, etc.)
            for i in range(1, len(seq) - 1):
                token_text = self.tokenizer.decode([seq[i].item()])
                if any(punct in token_text for punct in [".", "?", "!", "\n"]):
                    points.append(i + 1)

            # Limit injection points
            if len(points) > 3:
                # Sample random subset
                points = sorted(
                    np.random.choice(points, size=3, replace=False).tolist()
                )

            injection_points.append(points)

        return injection_points

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """Forward pass with thought injection."""
        # Inject thoughts into sequences
        thought_input_ids, thought_attention_mask = self.inject_thoughts(
            input_ids, attention_mask
        )

        # Forward through model
        outputs = self.model(
            input_ids=thought_input_ids, attention_mask=thought_attention_mask, **kwargs
        )

        return outputs, thought_input_ids, thought_attention_mask

    def extract_thoughts(self, generated_ids: torch.Tensor) -> list[str]:
        """Extract generated thoughts from output."""
        thoughts = []

        for seq in generated_ids:
            seq_thoughts = []
            in_thought = False
            current_thought = []

            for token_id in seq:
                if token_id == self.start_thought_id:
                    in_thought = True
                    current_thought = []
                elif token_id == self.end_thought_id and in_thought:
                    in_thought = False
                    if current_thought:
                        thought_text = self.tokenizer.decode(
                            current_thought, skip_special_tokens=True
                        )
                        seq_thoughts.append(thought_text)
                elif in_thought:
                    current_thought.append(token_id.item())

            thoughts.append(seq_thoughts)

        return thoughts


# ============================================================================
# Evaluation Dataset
# ============================================================================


class ReasoningEvalDataset(Dataset):
    """Dataset for evaluating reasoning capabilities."""

    def __init__(
        self, dataset_name: str, num_samples: int, tokenizer, max_length: int = 512
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load dataset
        if dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="test")
            self.examples = self.prepare_gsm8k(dataset, num_samples)
        elif dataset_name == "math":
            dataset = load_dataset("hendrycks/math", split="test")
            self.examples = self.prepare_math(dataset, num_samples)
        else:
            msg = f"Unknown dataset: {dataset_name}"
            raise ValueError(msg)

        logger.info(f"Loaded {len(self.examples)} examples from {dataset_name}")

    def prepare_gsm8k(self, dataset, num_samples: int) -> list[dict]:
        """Prepare GSM8K examples."""
        examples = []

        for i, item in enumerate(dataset):
            if i >= num_samples:
                break

            # Extract question and answer
            question = item["question"]
            answer = item["answer"]

            # Extract numerical answer
            answer_parts = answer.split("####")
            if len(answer_parts) >= 2:
                numerical_answer = answer_parts[1].strip()
            else:
                numerical_answer = answer.strip()

            examples.append(
                {
                    "input": f"Question: {question}\nAnswer:",
                    "target": answer,
                    "numerical_answer": numerical_answer,
                }
            )

        return examples

    def prepare_math(self, dataset, num_samples: int) -> list[dict]:
        """Prepare MATH dataset examples."""
        examples = []

        for i, item in enumerate(dataset):
            if i >= num_samples:
                break

            problem = item["problem"]
            solution = item["solution"]

            examples.append(
                {
                    "input": f"Problem: {problem}\nSolution:",
                    "target": solution,
                    "numerical_answer": self.extract_math_answer(solution),
                }
            )

        return examples

    def extract_math_answer(self, solution: str) -> str:
        """Extract final answer from MATH solution."""
        # Look for boxed answer
        import re

        boxed_pattern = r"\\boxed\{([^}]+)\}"
        match = re.search(boxed_pattern, solution)

        if match:
            return match.group(1)

        # Fallback: last number in solution
        numbers = re.findall(r"-?\d+\.?\d*", solution)
        return numbers[-1] if numbers else ""

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize input
        encoding = self.tokenizer(
            example["input"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "target": example["target"],
            "numerical_answer": example["numerical_answer"],
        }


# ============================================================================
# A/B Testing Harness
# ============================================================================


class ABTestHarness:
    """A/B testing harness for comparing with/without thought tokens."""

    def __init__(
        self,
        model: nn.Module,
        thought_model: ThoughtInjector,
        tokenizer,
        config: QuietSTaRConfig,
    ) -> None:
        self.model = model
        self.thought_model = thought_model
        self.tokenizer = tokenizer
        self.config = config

        # Move models to device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.thought_model.to(self.device)

        # Evaluation metrics
        self.metrics = {
            "baseline": {"accuracy": [], "perplexity": [], "time": []},
            "with_thoughts": {"accuracy": [], "perplexity": [], "time": []},
        }

    async def run_ab_test(self, eval_dataset: ReasoningEvalDataset) -> dict[str, Any]:
        """Run A/B test comparing baseline vs thought-injected model."""
        logger.info(f"Starting A/B test with {len(eval_dataset)} examples")

        # Create dataloader
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        # Run multiple rounds
        for round_idx in range(self.config.ab_test_rounds):
            logger.info(f"A/B Test Round {round_idx + 1}/{self.config.ab_test_rounds}")

            # Test baseline model
            baseline_results = await self.evaluate_model(
                self.model,
                dataloader,
                use_thoughts=False,
                desc=f"Baseline R{round_idx + 1}",
            )

            # Test with thoughts
            thought_results = await self.evaluate_model(
                self.thought_model,
                dataloader,
                use_thoughts=True,
                desc=f"Thoughts R{round_idx + 1}",
            )

            # Record metrics
            self.metrics["baseline"]["accuracy"].append(baseline_results["accuracy"])
            self.metrics["baseline"]["perplexity"].append(
                baseline_results["perplexity"]
            )
            self.metrics["baseline"]["time"].append(baseline_results["avg_time"])

            self.metrics["with_thoughts"]["accuracy"].append(
                thought_results["accuracy"]
            )
            self.metrics["with_thoughts"]["perplexity"].append(
                thought_results["perplexity"]
            )
            self.metrics["with_thoughts"]["time"].append(thought_results["avg_time"])

            # Log round results
            logger.info(
                f"Round {round_idx + 1} - Baseline Accuracy: {baseline_results['accuracy']:.3f}"
            )
            logger.info(
                f"Round {round_idx + 1} - Thoughts Accuracy: {thought_results['accuracy']:.3f}"
            )

        # Analyze results
        analysis = self.analyze_results()

        return analysis

    async def evaluate_model(
        self, model: nn.Module, dataloader: DataLoader, use_thoughts: bool, desc: str
    ) -> dict[str, float]:
        """Evaluate model on dataset."""
        model.eval()

        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        total_time = 0.0
        thought_traces = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["target"]
                numerical_answers = batch["numerical_answer"]

                start_time = time.time()

                if use_thoughts:
                    # Generate with thought injection
                    outputs, thought_ids, thought_mask = model(
                        input_ids, attention_mask
                    )

                    # Generate response
                    generated = model.model.generate(
                        input_ids=thought_ids,
                        attention_mask=thought_mask,
                        max_new_tokens=100,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                    # Extract thoughts
                    thoughts = model.extract_thoughts(generated)
                    thought_traces.extend(thoughts)
                else:
                    # Generate without thoughts
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=100,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                end_time = time.time()
                total_time += end_time - start_time

                # Decode and evaluate
                for _i, (gen, _target, num_answer) in enumerate(
                    zip(generated, targets, numerical_answers, strict=False)
                ):
                    generated_text = self.tokenizer.decode(
                        gen, skip_special_tokens=True
                    )

                    # Check if answer is correct
                    if self.check_answer(generated_text, num_answer):
                        total_correct += 1

                    total_samples += 1

        # Calculate metrics
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_time = total_time / len(dataloader) if len(dataloader) > 0 else 0.0
        perplexity = (
            np.exp(total_loss / total_samples) if total_samples > 0 else float("inf")
        )

        results = {
            "accuracy": accuracy,
            "perplexity": perplexity,
            "avg_time": avg_time,
            "total_samples": total_samples,
            "thought_traces": thought_traces if use_thoughts else [],
        }

        return results

    def check_answer(self, generated_text: str, target_answer: str) -> bool:
        """Check if generated answer matches target."""
        # Extract numbers from generated text
        import re

        generated_numbers = re.findall(r"-?\d+\.?\d*", generated_text)

        # Check if target answer appears in generated numbers
        return target_answer in generated_numbers

    def collate_fn(self, batch):
        """Custom collate function for dataloader."""
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "target": [item["target"] for item in batch],
            "numerical_answer": [item["numerical_answer"] for item in batch],
        }

    def analyze_results(self) -> dict[str, Any]:
        """Analyze A/B test results."""
        # Calculate statistics
        baseline_acc_mean = np.mean(self.metrics["baseline"]["accuracy"])
        baseline_acc_std = np.std(self.metrics["baseline"]["accuracy"])

        thoughts_acc_mean = np.mean(self.metrics["with_thoughts"]["accuracy"])
        thoughts_acc_std = np.std(self.metrics["with_thoughts"]["accuracy"])

        improvement = thoughts_acc_mean - baseline_acc_mean

        # Statistical significance test (paired t-test)
        from scipy import stats

        t_stat, p_value = stats.ttest_rel(
            self.metrics["with_thoughts"]["accuracy"],
            self.metrics["baseline"]["accuracy"],
        )

        # Determine winner
        is_significant = p_value < self.config.significance_threshold
        is_improvement = improvement >= self.config.min_improvement
        winner = "thoughts" if (is_significant and is_improvement) else "baseline"

        analysis = {
            "baseline_accuracy": baseline_acc_mean,
            "baseline_std": baseline_acc_std,
            "thoughts_accuracy": thoughts_acc_mean,
            "thoughts_std": thoughts_acc_std,
            "improvement": improvement,
            "improvement_percent": (improvement / baseline_acc_mean * 100)
            if baseline_acc_mean > 0
            else 0,
            "p_value": p_value,
            "is_significant": is_significant,
            "winner": winner,
            "baseline_perplexity": np.mean(self.metrics["baseline"]["perplexity"]),
            "thoughts_perplexity": np.mean(self.metrics["with_thoughts"]["perplexity"]),
            "baseline_time": np.mean(self.metrics["baseline"]["time"]),
            "thoughts_time": np.mean(self.metrics["with_thoughts"]["time"]),
        }

        logger.info("A/B Test Results:")
        logger.info(
            f"  Baseline Accuracy: {baseline_acc_mean:.3f} ± {baseline_acc_std:.3f}"
        )
        logger.info(
            f"  Thoughts Accuracy: {thoughts_acc_mean:.3f} ± {thoughts_acc_std:.3f}"
        )
        logger.info(
            f"  Improvement: {improvement:.3f} ({analysis['improvement_percent']:.1f}%)"
        )
        logger.info(f"  P-value: {p_value:.4f}")
        logger.info(f"  Winner: {winner}")

        return analysis


# ============================================================================
# Weight Baking
# ============================================================================


class WeightBaker:
    """Bakes reasoning patterns into model weights via fine-tuning."""

    def __init__(self, model: nn.Module, tokenizer, config: QuietSTaRConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)

    def prepare_baking_dataset(
        self, examples: list[dict], thought_traces: list[list[str]]
    ) -> Dataset:
        """Prepare dataset for baking thoughts into weights."""
        baking_examples = []

        for example, thoughts in zip(examples, thought_traces, strict=False):
            if not thoughts:
                continue

            # Create training example with injected thoughts
            input_text = example["input"]
            target_text = example["target"]

            # Insert thoughts at appropriate positions
            thought_augmented = self.insert_thoughts_in_text(input_text, thoughts)

            baking_examples.append({"input": thought_augmented, "target": target_text})

        return BakingDataset(baking_examples, self.tokenizer, max_length=512)

    def insert_thoughts_in_text(self, text: str, thoughts: list[str]) -> str:
        """Insert thought tokens into text."""
        # Simple insertion after sentences
        sentences = text.split(". ")
        augmented = []

        for i, sentence in enumerate(sentences):
            augmented.append(sentence)

            if i < len(thoughts) and thoughts[i]:
                thought_text = f" {self.config.start_thought_token} {thoughts[i]} {self.config.end_thought_token}"
                augmented.append(thought_text)

        return ". ".join(augmented)

    async def bake_weights(self, baking_dataset: Dataset) -> nn.Module:
        """Fine-tune model to internalize reasoning patterns."""
        logger.info("Starting weight baking via fine-tuning")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(Path(self.config.output_path).parent / "baking_checkpoints"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=str(Path(self.config.output_path).parent / "baking_logs"),
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=["wandb"] if wandb.run else [],
            fp16=self.config.mixed_precision and self.device.type == "cuda",
            dataloader_drop_last=False,
            seed=self.config.seed,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=baking_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Fine-tune
        trainer.train()

        # Save baked model
        logger.info(f"Saving baked model to {self.config.output_path}")
        trainer.save_model(self.config.output_path)
        self.tokenizer.save_pretrained(self.config.output_path)

        return self.model


class BakingDataset(Dataset):
    """Dataset for weight baking."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Combine input and target for language modeling
        full_text = f"{example['input']} {example['target']}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Labels are same as input_ids for language modeling
        encoding["labels"] = encoding["input_ids"].clone()

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["labels"].squeeze(),
        }


# ============================================================================
# Main Pipeline
# ============================================================================


class QuietSTaRBaker:
    """Main Quiet-STaR baking pipeline."""

    def __init__(self, config: QuietSTaRConfig) -> None:
        self.config = config
        self.wandb_run = None

        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        logger.info(f"QuietSTaR Baker initialized with model: {config.model_path}")

    def initialize_wandb(self) -> None:
        """Initialize W&B tracking."""
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                job_type="quietstar",
                tags=self.config.wandb_tags,
                config=self.config.dict(),
            )

            logger.info(f"W&B initialized: {self.wandb_run.url}")

        except Exception as e:
            logger.exception(f"W&B initialization failed: {e}")
            self.wandb_run = None

    async def run_baking_pipeline(self) -> dict[str, Any]:
        """Run complete Quiet-STaR baking pipeline."""
        try:
            # Initialize W&B
            self.initialize_wandb()

            # Load champion model
            logger.info(f"Loading champion model from {self.config.model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_path or self.config.model_path
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16
                if self.config.device == "cuda"
                else torch.float32,
            )

            # Create thought injector
            thought_model = ThoughtInjector(model, tokenizer, self.config)

            # Load evaluation dataset
            eval_dataset = ReasoningEvalDataset(
                self.config.eval_dataset, self.config.eval_samples, tokenizer
            )

            # Run A/B test
            ab_harness = ABTestHarness(model, thought_model, tokenizer, self.config)
            ab_results = await ab_harness.run_ab_test(eval_dataset)

            # Log A/B results to W&B
            if self.wandb_run:
                self.wandb_run.log(
                    {
                        "baseline_accuracy": ab_results["baseline_accuracy"],
                        "thoughts_accuracy": ab_results["thoughts_accuracy"],
                        "improvement": ab_results["improvement"],
                        "improvement_percent": ab_results["improvement_percent"],
                        "p_value": ab_results["p_value"],
                        "winner": ab_results["winner"],
                    }
                )

                # Create accuracy comparison chart
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 6))
                x = range(self.config.ab_test_rounds)
                ax.plot(
                    x,
                    ab_harness.metrics["baseline"]["accuracy"],
                    "b-",
                    label="Baseline",
                    marker="o",
                )
                ax.plot(
                    x,
                    ab_harness.metrics["with_thoughts"]["accuracy"],
                    "r-",
                    label="With Thoughts",
                    marker="s",
                )
                ax.set_xlabel("Round")
                ax.set_ylabel("Accuracy")
                ax.set_title("Reasoning Accuracy: Baseline vs Thoughts")
                ax.legend()
                ax.grid(True)

                self.wandb_run.log({"reasoning_accuracy_chart": wandb.Image(fig)})
                plt.close()

            # Bake weights if thoughts improved performance
            baked_model = None

            if ab_results["winner"] == "thoughts":
                logger.info(
                    "Thought injection improved performance - proceeding with weight baking"
                )

                # Prepare baking dataset
                weight_baker = WeightBaker(thought_model.model, tokenizer, self.config)

                # Get thought traces from best round
                np.argmax(ab_harness.metrics["with_thoughts"]["accuracy"])

                # Re-run best round to get thought traces
                dataloader = DataLoader(
                    eval_dataset,
                    batch_size=self.config.eval_batch_size,
                    shuffle=False,
                    collate_fn=ab_harness.collate_fn,
                )

                best_round_results = await ab_harness.evaluate_model(
                    thought_model,
                    dataloader,
                    use_thoughts=True,
                    desc="Collecting thought traces",
                )

                # Create baking dataset
                baking_dataset = weight_baker.prepare_baking_dataset(
                    eval_dataset.examples, best_round_results["thought_traces"]
                )

                # Bake weights
                baked_model = await weight_baker.bake_weights(baking_dataset)

                # Save baked checkpoint as W&B artifact
                if self.wandb_run:
                    artifact = wandb.Artifact(
                        "baked_quietstar_model",
                        type="model",
                        description=f"QuietSTaR baked model with {ab_results['improvement_percent']:.1f}% improvement",
                    )
                    artifact.add_dir(self.config.output_path)
                    self.wandb_run.log_artifact(artifact)

                    # Log trace quality
                    trace_quality_scores = self.evaluate_trace_quality(
                        best_round_results["thought_traces"]
                    )
                    self.wandb_run.log(
                        {
                            "trace_quality_mean": np.mean(trace_quality_scores),
                            "trace_quality_std": np.std(trace_quality_scores),
                        }
                    )

                    # Create trace quality histogram
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(trace_quality_scores, bins=20, alpha=0.7, color="green")
                    ax.set_xlabel("Trace Quality Score")
                    ax.set_ylabel("Count")
                    ax.set_title("Distribution of Thought Trace Quality")
                    ax.axvline(
                        np.mean(trace_quality_scores),
                        color="red",
                        linestyle="--",
                        label=f"Mean: {np.mean(trace_quality_scores):.2f}",
                    )
                    ax.legend()

                    self.wandb_run.log({"trace_quality_distribution": wandb.Image(fig)})
                    plt.close()

                logger.info(f"Baked model saved to {self.config.output_path}")
            else:
                logger.info(
                    "Thought injection did not improve performance - skipping weight baking"
                )
                # Save original model
                model.save_pretrained(self.config.output_path)
                tokenizer.save_pretrained(self.config.output_path)

            # Final results
            results = {
                "ab_test_results": ab_results,
                "baked_model_path": self.config.output_path if baked_model else None,
                "improvement": ab_results["improvement_percent"],
                "winner": ab_results["winner"],
            }

            logger.info("QuietSTaR baking pipeline completed successfully")
            return results

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            raise

        finally:
            if self.wandb_run:
                self.wandb_run.finish()

    def evaluate_trace_quality(self, thought_traces: list[list[str]]) -> list[float]:
        """Evaluate quality of generated thought traces."""
        quality_scores = []

        for traces in thought_traces:
            if not traces:
                quality_scores.append(0.0)
                continue

            # Simple heuristics for trace quality
            trace_score = 0.0

            for trace in traces:
                # Length score (not too short, not too long)
                length = len(trace.split())
                if 5 <= length <= 30:
                    trace_score += 0.3

                # Contains reasoning indicators
                reasoning_words = [
                    "because",
                    "therefore",
                    "since",
                    "if",
                    "then",
                    "so",
                    "thus",
                ]
                if any(word in trace.lower() for word in reasoning_words):
                    trace_score += 0.4

                # Contains mathematical/logical operators
                math_indicators = ["+", "-", "*", "/", "=", ">", "<", "step"]
                if any(indicator in trace for indicator in math_indicators):
                    trace_score += 0.3

            # Normalize by number of traces
            quality_scores.append(min(trace_score / max(len(traces), 1), 1.0))

        return quality_scores


# ============================================================================
# CLI Interface
# ============================================================================


@click.group()
def forge() -> None:
    """Agent Forge CLI."""


@forge.command()
@click.option("--model", required=True, help="Path to champion model from EvoMerge")
@click.option("--out", required=True, help="Output path for baked model")
@click.option("--tokenizer", help="Custom tokenizer path (defaults to model path)")
@click.option(
    "--eval-dataset", default="gsm8k", help="Evaluation dataset (gsm8k, math)"
)
@click.option("--eval-samples", default=100, help="Number of evaluation samples")
@click.option("--device", default="auto", help="Device to use (auto, cuda, cpu)")
@click.option("--config", help="Configuration JSON file")
def bake_quietstar(
    model, out, tokenizer, eval_dataset, eval_samples, device, config
) -> None:
    """Bake Quiet-STaR reasoning into model weights."""
    try:
        # Load configuration
        if config and Path(config).exists():
            with open(config) as f:
                config_data = json.load(f)
            quietstar_config = QuietSTaRConfig(**config_data)
        else:
            # Create configuration from CLI args
            quietstar_config = QuietSTaRConfig(
                model_path=model,
                output_path=out,
                tokenizer_path=tokenizer,
                eval_dataset=eval_dataset,
                eval_samples=eval_samples,
                device=device,
            )

        # Run baking pipeline
        baker = QuietSTaRBaker(quietstar_config)

        logger.info("Starting Quiet-STaR baking pipeline...")
        results = asyncio.run(baker.run_baking_pipeline())

        # Print results
        print("\n" + "=" * 60)
        print("QUIET-STAR BAKING RESULTS")
        print("=" * 60)
        print(f"Winner: {results['winner']}")
        print(f"Improvement: {results['improvement']:.1f}%")

        if results["baked_model_path"]:
            print(f"Baked model saved to: {results['baked_model_path']}")
        else:
            print("Original model saved (no improvement from thoughts)")

        print("=" * 60)

    except Exception as e:
        logger.exception(f"Quiet-STaR baking failed: {e}")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    forge()
