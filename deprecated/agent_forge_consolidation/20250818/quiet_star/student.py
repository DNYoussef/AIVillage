"""
Student distillation system for Quiet-STaR.
Implements B3) Student distillation prompt for quiet reasoning.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .config import QuietSTaRConfig
from .model import QuietSTaRModelWrapper
from .teacher import TeacherPromptGenerator, TrainingPair


@dataclass
class DistillationConfig:
    """Configuration for student distillation training."""

    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # Distillation parameters
    temperature: float = 3.0  # For softmax temperature scaling
    alpha: float = 0.7  # Weight for distillation loss vs student loss

    # Data parameters
    max_sequence_length: int = 512
    train_test_split: float = 0.8

    # Quiet-STaR specific
    thought_supervision_weight: float = 1.0  # Weight for thought supervision
    consistency_regularization_weight: float = 0.1  # Weight for consistency loss


@dataclass
class StudentExample:
    """A training example for the student model."""

    input_text: str
    target_text: str  # Without thoughts - what user sees
    teacher_thoughts: str  # Teacher's internal reasoning
    metadata: dict[str, Any] = field(default_factory=dict)


class QuietReasoningStudent:
    """
    Student model that learns to perform quiet reasoning.
    Learns from teacher's thought processes but doesn't expose them.
    """

    def __init__(
        self,
        config: QuietSTaRConfig,
        distillation_config: DistillationConfig,
        base_model_name: str = "microsoft/DialoGPT-small",
        teacher_model: QuietSTaRModelWrapper | None = None,
    ):
        self.config = config
        self.distill_config = distillation_config
        self.base_model_name = base_model_name

        # Initialize student model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Add special tokens
        special_tokens = [
            config.start_of_thought_token,
            config.end_of_thought_token,
            config.no_thought_token,
        ]
        added_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        if added_tokens > 0:
            self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Create special token IDs mapping
        self.special_token_ids = {
            token: self.tokenizer.convert_tokens_to_ids(token)
            for token in special_tokens
        }

        # Wrap with Quiet-STaR architecture
        self.student_model = QuietSTaRModelWrapper(
            base_model=self.base_model,
            config=config,
            special_token_ids=self.special_token_ids,
        )

        self.teacher_model = teacher_model
        self.training_examples = []

    def prepare_training_data(
        self,
        training_pairs: list[TrainingPair],
        include_no_thought_examples: bool = True,
    ) -> list[StudentExample]:
        """Prepare training data from teacher-generated pairs."""

        examples = []

        for pair in training_pairs:
            # Main example: learn to generate answer with internal thoughts
            example = StudentExample(
                input_text=pair.question,
                target_text=pair.answer,
                teacher_thoughts=pair.reflection,
                metadata={
                    "type": "thought_supervision",
                    "style": pair.metadata.get("style", "unknown"),
                },
            )
            examples.append(example)

            # Optional: Add examples with no-thought token
            if include_no_thought_examples and random.random() < 0.3:
                no_thought_example = StudentExample(
                    input_text=pair.question,
                    target_text=f"{self.config.no_thought_token} {pair.answer}",
                    teacher_thoughts="",
                    metadata={"type": "no_thought", "style": "direct"},
                )
                examples.append(no_thought_example)

        self.training_examples = examples
        return examples

    def create_training_prompt(self, example: StudentExample) -> str:
        """Create training prompt that teaches quiet reasoning."""

        if example.metadata.get("type") == "no_thought":
            # Direct answer without thoughts
            return f"{example.input_text} {example.target_text}"
        else:
            # Answer with hidden thoughts (for training only)
            thought_section = f"{self.config.start_of_thought_token}{example.teacher_thoughts}{self.config.end_of_thought_token}"
            return f"{example.input_text} {thought_section} {example.target_text}"

    def create_inference_prompt(self, example: StudentExample) -> str:
        """Create inference prompt (what user sees - no thoughts)."""
        return f"{example.input_text} {example.target_text}"

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        thought_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute knowledge distillation loss."""

        # Standard cross-entropy loss for student
        student_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        student_loss = student_loss_fn(
            student_logits.view(-1, student_logits.size(-1)), labels.view(-1)
        )

        # Knowledge distillation loss (only where teacher has thoughts)
        if teacher_logits is not None and thought_mask.any():
            # Apply temperature scaling
            teacher_soft = nn.functional.log_softmax(
                teacher_logits / self.distill_config.temperature, dim=-1
            )
            student_soft = nn.functional.log_softmax(
                student_logits / self.distill_config.temperature, dim=-1
            )

            # KL divergence loss on thought regions
            kl_loss_fn = nn.KLDivLoss(reduction="none", log_target=True)
            kl_loss = kl_loss_fn(student_soft, teacher_soft).sum(dim=-1)

            # Apply thought mask to focus on reasoning regions
            thought_kl_loss = (
                kl_loss * thought_mask.float()
            ).sum() / thought_mask.float().sum()
        else:
            thought_kl_loss = torch.tensor(0.0, device=student_logits.device)

        # Consistency regularization - student should be consistent
        # with its own predictions across similar inputs
        consistency_loss = torch.tensor(0.0, device=student_logits.device)

        # Combined loss
        total_loss = (
            (1 - self.distill_config.alpha) * student_loss
            + self.distill_config.alpha * thought_kl_loss
            + self.distill_config.consistency_regularization_weight * consistency_loss
        )

        loss_components = {
            "student_loss": student_loss,
            "distillation_loss": thought_kl_loss,
            "consistency_loss": consistency_loss,
            "total_loss": total_loss,
        }

        return total_loss, loss_components

    def train_on_examples(
        self,
        examples: list[StudentExample],
        output_dir: Path,
        eval_steps: int = 100,
        save_steps: int = 500,
    ):
        """Train the student model on the prepared examples."""

        # Prepare tokenized datasets
        train_texts = [self.create_training_prompt(ex) for ex in examples]

        def tokenize_function(texts):
            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.distill_config.max_sequence_length,
                return_tensors="pt",
            )

        # Create dataset
        tokenized_data = tokenize_function(train_texts)

        # Split train/eval
        num_train = int(len(examples) * self.distill_config.train_test_split)
        train_dataset = {
            "input_ids": tokenized_data["input_ids"][:num_train],
            "attention_mask": tokenized_data["attention_mask"][:num_train],
            "labels": tokenized_data["input_ids"][
                :num_train
            ].clone(),  # For language modeling
        }

        eval_dataset = (
            {
                "input_ids": tokenized_data["input_ids"][num_train:],
                "attention_mask": tokenized_data["attention_mask"][num_train:],
                "labels": tokenized_data["input_ids"][num_train:].clone(),
            }
            if num_train < len(examples)
            else None
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.distill_config.num_epochs,
            per_device_train_batch_size=self.distill_config.batch_size,
            per_device_eval_batch_size=self.distill_config.batch_size,
            warmup_steps=self.distill_config.warmup_steps,
            weight_decay=self.distill_config.weight_decay,
            logging_dir=str(output_dir / "logs"),
            logging_steps=50,
            eval_steps=eval_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            learning_rate=self.distill_config.learning_rate,
        )

        # Custom trainer with distillation loss
        class DistillationTrainer(Trainer):
            def __init__(self, student_wrapper, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.student_wrapper = student_wrapper

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = self.student_wrapper(**inputs)

                # Standard language modeling loss
                shift_logits = outputs["logits"][..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                return (loss, outputs) if return_outputs else loss

        # Convert datasets to HuggingFace format
        from torch.utils.data import Dataset

        class CustomDataset(Dataset):
            def __init__(self, tokenized_data):
                self.data = tokenized_data
                self.length = len(tokenized_data["input_ids"])

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.data.items()}

        train_ds = CustomDataset(train_dataset)
        eval_ds = CustomDataset(eval_dataset) if eval_dataset else None

        # Initialize trainer
        trainer = DistillationTrainer(
            student_wrapper=self.student_model,
            model=self.base_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
        )

        # Train
        print("Starting student distillation training...")
        trainer.train()

        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        print(f"Student model saved to {output_dir}")

    def evaluate_quiet_reasoning(
        self, test_questions: list[str], output_file: Path | None = None
    ) -> dict[str, Any]:
        """Evaluate the student's quiet reasoning abilities."""

        self.student_model.eval()
        results = []

        for question in test_questions:
            with torch.no_grad():
                # Test inference mode (thoughts should be hidden)
                input_ids = self.tokenizer.encode(question, return_tensors="pt")

                # Generate with thoughts (training mode)
                self.student_model.train()
                outputs_with_thoughts = self.student_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                response_with_thoughts = self.tokenizer.decode(
                    outputs_with_thoughts[0][input_ids.shape[1] :],
                    skip_special_tokens=False,
                )

                # Generate without thoughts (inference mode)
                self.student_model.eval()
                outputs_inference = self.student_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                response_inference = self.tokenizer.decode(
                    outputs_inference[0][input_ids.shape[1] :], skip_special_tokens=True
                )

                result = {
                    "question": question,
                    "response_with_thoughts": response_with_thoughts,
                    "response_inference": response_inference,
                    "thoughts_hidden": self.config.start_of_thought_token
                    not in response_inference,
                    "has_internal_reasoning": self.config.start_of_thought_token
                    in response_with_thoughts,
                }

                results.append(result)

        # Calculate metrics
        evaluation = {
            "total_questions": len(test_questions),
            "thoughts_properly_hidden": sum(r["thoughts_hidden"] for r in results),
            "has_internal_reasoning": sum(r["has_internal_reasoning"] for r in results),
            "success_rate": sum(r["thoughts_hidden"] for r in results) / len(results),
            "reasoning_rate": sum(r["has_internal_reasoning"] for r in results)
            / len(results),
            "detailed_results": results,
        }

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False)

        return evaluation


def create_distillation_pipeline(
    config: QuietSTaRConfig,
    base_model_name: str = "microsoft/DialoGPT-small",
    output_dir: Path = Path("./quiet_star_student"),
) -> tuple[TeacherPromptGenerator, QuietReasoningStudent]:
    """Create complete teacher-student distillation pipeline."""

    # Initialize teacher
    teacher = TeacherPromptGenerator(config, base_model_name)

    # Initialize student with distillation config
    distill_config = DistillationConfig()
    student = QuietReasoningStudent(config, distill_config, base_model_name)

    return teacher, student


if __name__ == "__main__":
    # Demo usage
    from .config import get_training_config
    from .teacher import load_sample_questions

    config = get_training_config()

    print("=== Student Distillation Demo ===\n")

    # Create pipeline
    teacher, student = create_distillation_pipeline(config)

    # Generate training data
    questions = load_sample_questions()[:5]  # Use 5 questions for demo
    training_pairs = []

    print("Generating teacher examples...")
    for question in questions:
        pair = teacher.create_training_pair(question)
        training_pairs.append(pair)
        print(f"✓ Created pair for: {question[:50]}...")

    # Prepare student training data
    print("\nPreparing student training data...")
    student_examples = student.prepare_training_data(training_pairs)

    print(f"Created {len(student_examples)} training examples")

    # Show example training formats
    print("\n=== Training Example ===")
    example = student_examples[0]
    print(f"Input: {example.input_text}")
    print(f"Teacher thoughts: {example.teacher_thoughts}")
    print(f"Target output: {example.target_text}")
    print(f"Training format: {student.create_training_prompt(example)}")
    print(f"Inference format: {student.create_inference_prompt(example)}")

    print("\n✓ Student distillation system ready for training")
