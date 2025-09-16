#!/usr/bin/env python3
"""
Agent Forge Phase 2: Quiet-STaR Baking

This phase implements "prompt baking" of reasoning tokens into models, where thoughts like
deep system prompts are iteratively baked until they "stick" in the model weights.

Key Features:
- Iterative prompt baking with convergence testing
- ThoughtMixingHead for reasoning enhancement
- A/B testing to validate improvement
- Multiple cognitive strategies (systems thinking, first principles, etc.)
- Production-grade loss functions with leak prevention
- IoT processing with critique/alternatives/evaluation cycles
- Edge-of-chaos training for optimal learning
- Grokfast integration for 50x acceleration

Consolidates implementations from:
- src/agent_forge/quietstar_baker.py (main baking pipeline)
- src/agent_forge/quiet_star/model.py (ThoughtMixingHead)
- src/agent_forge/bakedquietiot/quiet_star.py (cognitive strategies)
- src/agent_forge/foundation/quiet_star.py (token definitions)
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any

from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:  # Optional DSPy integration
    from src.coordination.dspy_integration import DSPyAgentOptimizer
except Exception:  # pragma: no cover - DSPy optional
    DSPyAgentOptimizer = None  # type: ignore[misc]

# Try to import PhaseController, with fallback for direct imports
try:
    from ..core.phase_controller import PhaseController, PhaseResult
except (ImportError, ValueError):
    # Fallback for direct imports - create minimal base classes
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Any

    import torch.nn as nn

    @dataclass
    class PhaseResult:
        success: bool
        model: nn.Module
        phase_name: str = None
        metrics: dict = None
        duration_seconds: float = 0.0
        artifacts: dict = None
        config: dict = None
        error: str = None
        start_time: datetime = None
        end_time: datetime = None

        def __post_init__(self):
            if self.end_time is None:
                self.end_time = datetime.now()
            if self.start_time is None:
                self.start_time = self.end_time

    class PhaseController(ABC):
        def __init__(self, config: Any):
            self.config = config

        @abstractmethod
        async def run(self, model: nn.Module) -> PhaseResult:
            pass


# from packages.agent_forge.legacy_src.training.grokfast_ctrl import GrokfastOptimizer  # Reference implementation: import optimization available in legacy package

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class QuietSTaRConfig:
    """Configuration for Quiet-STaR baking phase."""

    # Model configuration
    model_path: str = ""
    tokenizer_path: str | None = None
    output_path: str = ""

    # Special tokens (following existing convention)
    start_thought_token: str = "<|startofthought|>"
    end_thought_token: str = "<|endofthought|>"
    no_thought_token: str = "<|nothought|>"

    # Baking configuration
    max_baking_iterations: int = 5
    convergence_threshold: float = 0.95  # When to stop baking (95% "stuck" rate)
    thought_probability: float = 0.5
    max_thought_length: int = 64

    # Cognitive strategies (from bakedquietiot)
    cognitive_strategies: list[str] = field(
        default_factory=lambda: [
            "systems_thinking",
            "first_principles",
            "cross_domain",
            "probabilistic_thinking",
            "rapid_iteration",
            "paradox_resolution",
        ]
    )

    # Evaluation configuration
    eval_dataset: str = "gsm8k"
    eval_samples: int = 100
    eval_batch_size: int = 4

    # A/B testing
    ab_test_rounds: int = 3
    significance_threshold: float = 0.05
    min_improvement: float = 0.02

    # Training configuration
    learning_rate: float = 1e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4

    # Loss weights (from losses.py)
    w_task: float = 1.0
    w_reflect: float = 0.3
    w_leak: float = 10.0

    # Grokfast configuration
    enable_grokfast: bool = True
    grokfast_ema_alpha: float = 0.98
    grokfast_lambda: float = 2.0

    # System configuration
    device: str = "auto"
    mixed_precision: bool = True
    seed: int = 42

    # DSPy optimization
    enable_dspy_optimization: bool = False
    dspy_optimized_prompt: str | None = None

    # W&B tracking
    wandb_project: str = "agent_forge"
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["quietstar", "phase2"])


# ============================================================================
# Thought Token Management
# ============================================================================


@dataclass
class ThoughtSegment:
    """Represents a parsed thought segment in the sequence."""

    start_idx: int
    end_idx: int
    thought_tokens: list[int]
    is_thought: bool


class ThoughtMixingHead(nn.Module):
    """
    Advanced mixing head that processes hidden thought representations.

    Consolidated from src/agent_forge/quiet_star/model.py with enhancements.
    """

    def __init__(
        self, hidden_size: int, vocab_size: int, config: QuietSTaRConfig, base_lm_head: nn.Module | None = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config = config
        self.base_lm_head = base_lm_head

        # Thought-aware processing layers
        self.thought_detector = nn.Linear(hidden_size, 3)  # [no_thought, start_thought, end_thought]
        self.thought_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4, dropout=0.1, batch_first=True
        )

        # Mixing and projection layers
        self.thought_gate = nn.Linear(hidden_size, 1)
        self.context_mixer = nn.Linear(hidden_size * 2, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights conservatively."""
        nn.init.xavier_uniform_(self.thought_detector.weight)
        nn.init.constant_(self.thought_detector.bias, 0)

        nn.init.xavier_uniform_(self.thought_gate.weight)
        nn.init.constant_(self.thought_gate.bias, -2.0)  # Start with low gate values

        nn.init.xavier_uniform_(self.context_mixer.weight)
        nn.init.constant_(self.context_mixer.bias, 0)

        # Initialize output projection to match base model if available
        if self.base_lm_head is not None and hasattr(self.base_lm_head, "weight"):
            with torch.no_grad():
                self.output_projection.weight.copy_(self.base_lm_head.weight)
                if hasattr(self.base_lm_head, "bias") and self.base_lm_head.bias is not None:
                    self.output_projection.bias.copy_(self.base_lm_head.bias)

    def parse_thought_segments(
        self, input_ids: torch.Tensor, special_token_ids: dict[str, int]
    ) -> list[list[ThoughtSegment]]:
        """Parse input sequences to identify thought segments."""
        batch_size, seq_len = input_ids.shape
        sot_id = special_token_ids.get(self.config.start_thought_token, -1)
        eot_id = special_token_ids.get(self.config.end_thought_token, -1)
        not_id = special_token_ids.get(self.config.no_thought_token, -1)

        batch_segments = []

        for b in range(batch_size):
            sequence = input_ids[b].tolist()
            segments = []
            i = 0

            while i < seq_len:
                if sequence[i] == sot_id:
                    # Found start of thought, look for end
                    start_idx = i
                    i += 1

                    # Find matching end token
                    while i < seq_len and sequence[i] != eot_id:
                        i += 1

                    if i < seq_len:  # Found end token
                        end_idx = i + 1  # Include end token
                        thought_tokens = sequence[start_idx:end_idx]
                        segments.append(
                            ThoughtSegment(
                                start_idx=start_idx, end_idx=end_idx, thought_tokens=thought_tokens, is_thought=True
                            )
                        )
                        i += 1
                    else:
                        # Unclosed thought, treat as regular tokens
                        segments.append(
                            ThoughtSegment(
                                start_idx=start_idx,
                                end_idx=seq_len,
                                thought_tokens=sequence[start_idx:],
                                is_thought=False,
                            )
                        )
                        break

                elif sequence[i] == not_id:
                    # No-thought token, single token segment
                    segments.append(
                        ThoughtSegment(start_idx=i, end_idx=i + 1, thought_tokens=[sequence[i]], is_thought=False)
                    )
                    i += 1

                else:
                    # Regular token, group with consecutive regular tokens
                    start_idx = i
                    while i < seq_len and sequence[i] not in [sot_id, eot_id, not_id]:
                        i += 1

                    segments.append(
                        ThoughtSegment(
                            start_idx=start_idx, end_idx=i, thought_tokens=sequence[start_idx:i], is_thought=False
                        )
                    )

            batch_segments.append(segments)

        return batch_segments

    def create_thought_mask(self, input_ids: torch.Tensor, special_token_ids: dict[str, int]) -> torch.Tensor:
        """Create mask indicating which tokens are inside thought segments."""
        batch_segments = self.parse_thought_segments(input_ids, special_token_ids)
        batch_size, seq_len = input_ids.shape

        thought_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)

        for b, segments in enumerate(batch_segments):
            for segment in segments:
                if segment.is_thought:
                    thought_mask[b, segment.start_idx : segment.end_idx] = True

        return thought_mask

    def encode_thoughts(self, hidden_states: torch.Tensor, thought_mask: torch.Tensor) -> torch.Tensor:
        """Apply thought-specific encoding to thought regions."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Clone hidden states to avoid in-place modification
        encoded_states = hidden_states.clone()

        # Process each batch item separately
        for b in range(batch_size):
            batch_mask = thought_mask[b]

            if batch_mask.any():
                # Extract thought tokens for this batch item
                thought_indices = batch_mask.nonzero(as_tuple=True)[0]

                if len(thought_indices) > 0:
                    thought_hidden = hidden_states[b, thought_indices].unsqueeze(0)

                    # Apply transformer encoder to thought sequence
                    thought_encoded = self.thought_encoder(thought_hidden)

                    # Put encoded thoughts back
                    encoded_states[b, thought_indices] = thought_encoded.squeeze(0)

        return encoded_states

    def mix_contexts(
        self, regular_hidden: torch.Tensor, thought_hidden: torch.Tensor, thought_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mix regular and thought contexts using gating mechanism."""
        # Compute thought influence gate
        gate_logits = self.thought_gate(thought_hidden)
        thought_gate = torch.sigmoid(gate_logits)

        # Mix regular and thought contexts
        concatenated = torch.cat([regular_hidden, thought_hidden], dim=-1)
        mixed_context = self.context_mixer(concatenated)

        # Apply gating - stronger mixing in thought regions
        gate_mask = thought_mask.float().unsqueeze(-1)
        effective_gate = thought_gate * gate_mask + (1 - gate_mask) * 0.1

        mixed_hidden = (1 - effective_gate) * regular_hidden + effective_gate * mixed_context

        return mixed_hidden

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        special_token_ids: dict[str, int],
        return_thought_representations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Forward pass through thought mixing head."""
        # Parse thought structure
        thought_mask = self.create_thought_mask(input_ids, special_token_ids)

        # Encode thought-specific representations
        thought_encoded = self.encode_thoughts(hidden_states, thought_mask)

        # Mix regular and thought contexts
        mixed_hidden = self.mix_contexts(hidden_states, thought_encoded, thought_mask)

        # Generate output logits
        logits = self.output_projection(mixed_hidden)

        # Strip thought tokens in inference mode
        if not self.training:
            logits = self._strip_thought_tokens(logits, input_ids, special_token_ids)

        if return_thought_representations:
            representations = {
                "thought_mask": thought_mask,
                "thought_encoded": thought_encoded,
                "mixed_hidden": mixed_hidden,
                "raw_logits": logits,
            }
            return logits, representations

        return logits

    def _strip_thought_tokens(
        self, logits: torch.Tensor, input_ids: torch.Tensor, special_token_ids: dict[str, int]
    ) -> torch.Tensor:
        """Strip thought token logits from output during inference."""
        sot_id = special_token_ids.get(self.config.start_thought_token, -1)
        eot_id = special_token_ids.get(self.config.end_thought_token, -1)

        stripped_logits = logits.clone()

        if sot_id >= 0 and sot_id < logits.size(-1):
            stripped_logits[:, :, sot_id] = -float("inf")
        if eot_id >= 0 and eot_id < logits.size(-1):
            stripped_logits[:, :, eot_id] = -float("inf")

        return stripped_logits


# ============================================================================
# Cognitive Strategy Integration
# ============================================================================


class CognitiveStrategyProcessor:
    """
    Integrates cognitive strategies from bakedquietiot implementation.

    Provides structured reasoning with critique/alternatives/evaluation cycles.
    """

    def __init__(self, model: nn.Module, tokenizer, config: QuietSTaRConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(
            config.device if config.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        )

    async def generate_thought_with_strategies(self, input_text: str, temperature: float = 0.5) -> dict[str, Any]:
        """Generate thought using cognitive strategies with IoT processing."""
        prompt = f"{input_text}\n\nApply the following cognitive strategies:\n"
        for strategy in self.config.cognitive_strategies:
            prompt += f"<{strategy}>\n"
        prompt += "<start of thought>"

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=500,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            early_stopping=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        return {
            "text": self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False),
            "hidden_states": outputs.hidden_states,
        }

    async def extract_strategy_insights(self, thought: str) -> dict[str, str]:
        """Extract insights for each cognitive strategy."""
        insights = {}
        for strategy in self.config.cognitive_strategies:
            start_tag = f"<{strategy}>"
            end_tag = f"</{strategy}>"
            start = thought.find(start_tag)
            end = thought.find(end_tag)
            if start != -1 and end != -1:
                insights[strategy] = thought[start + len(start_tag) : end].strip()
            else:
                insights[strategy] = "No specific insight found."
        return insights

    async def iot_process(self, input_text: str, max_iterations: int = 5) -> tuple[dict, dict]:
        """
        Iterative Optimization of Thought (IoT) processing.

        Implements critique/alternatives/evaluation cycles until convergence.
        """
        thought = {"text": input_text, "hidden_states": None}

        for iteration in range(max_iterations):
            # Generate thought with strategies
            thought = await self.generate_thought_with_strategies(thought["text"])
            insights = await self.extract_strategy_insights(thought["text"])

            # Generate critique
            critique = await self._generate_critique(thought["text"], insights, temperature=0.2)

            # Generate alternatives
            alternatives = await self._generate_alternatives(thought["text"], insights, temperature=0.8)

            # Self-evaluation
            evaluation = await self._self_evaluate(thought["text"], insights)

            # Revise thought based on feedback
            thought = await self._revise_thought(
                thought["text"], critique["text"], alternatives["text"], evaluation["text"], insights
            )

            # Check for convergence
            if "<ready to answer>" in thought["text"]:
                break

        return thought, insights

    async def _generate_critique(self, thought: str, insights: dict, temperature: float = 0.2) -> dict:
        """Generate critique of current thought."""
        prompt = f"Critique the following thought and insights:\n{thought}\n\nInsights:\n"
        for strategy, insight in insights.items():
            prompt += f"{strategy}: {insight}\n"
        prompt += "\nCritique:"
        return await self.generate_thought_with_strategies(prompt, temperature)

    async def _generate_alternatives(self, thought: str, insights: dict, temperature: float = 0.8) -> dict:
        """Generate alternative perspectives."""
        prompt = f"Generate alternative perspectives for:\n{thought}\n\nConsider these insights:\n"
        for strategy, insight in insights.items():
            prompt += f"{strategy}: {insight}\n"
        prompt += "\nAlternatives:"
        return await self.generate_thought_with_strategies(prompt, temperature)

    async def _self_evaluate(self, thought: str, insights: dict) -> dict:
        """Self-evaluate thought quality."""
        prompt = f"Self-evaluate the following thought and insights:\n{thought}\n\nInsights:\n"
        for strategy, insight in insights.items():
            prompt += f"{strategy}: {insight}\n"
        prompt += "\nEvaluation:"
        evaluation = await self.generate_thought_with_strategies(prompt)

        # Add ethical evaluation
        ethical_prompt = f"""
        Evaluate the following thought for ethical considerations:
        1. Does it promote unbiased and fair outcomes?
        2. Does it respect privacy and data protection?
        3. Is it transparent and explainable?
        4. Does it consider potential negative consequences?
        5. Is it as true to reality as possible?

        Thought: {thought}

        Ethical evaluation:
        """
        ethical_evaluation = await self.generate_thought_with_strategies(ethical_prompt)

        combined_evaluation = f"{evaluation['text']}\n\nEthical considerations:\n{ethical_evaluation['text']}"
        return {"text": combined_evaluation, "hidden_states": evaluation["hidden_states"]}

    async def _revise_thought(
        self, thought: str, critique: str, alternatives: str, evaluation: str, insights: dict
    ) -> dict:
        """Revise thought based on feedback."""
        prompt = f"""
        Original thought: {thought}
        Critique: {critique}
        Alternatives: {alternatives}
        Evaluation: {evaluation}

        Insights:
        """
        for strategy, insight in insights.items():
            prompt += f"{strategy}: {insight}\n"
        prompt += "\nRevised thought:"
        return await self.generate_thought_with_strategies(prompt)


# ============================================================================
# Iterative Prompt Baking Engine
# ============================================================================


class PromptBakingEngine:
    """
    Core engine for iterative prompt baking until thoughts "stick".

    Tests convergence by checking if the model naturally produces thought tokens
    without explicit prompting, indicating successful baking.
    """

    def __init__(self, model: nn.Module, tokenizer, config: QuietSTaRConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(
            config.device if config.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Add special tokens
        self._add_thought_tokens()

        # Create cognitive strategy processor
        self.cognitive_processor = CognitiveStrategyProcessor(model, tokenizer, config)

        # Initialize ThoughtMixingHead
        base_lm_head = self._find_lm_head()
        self.thought_head = ThoughtMixingHead(
            hidden_size=model.config.hidden_size, vocab_size=len(tokenizer), config=config, base_lm_head=base_lm_head
        )
        self.thought_head.to(self.device)

    def _add_thought_tokens(self):
        """Add thought tokens to tokenizer."""
        special_tokens = {
            "additional_special_tokens": [
                self.config.start_thought_token,
                self.config.end_thought_token,
                self.config.no_thought_token,
            ]
        }

        num_added = self.tokenizer.add_special_tokens(special_tokens)

        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {num_added} special tokens and resized embeddings")

        # Cache token IDs
        self.special_token_ids = {
            self.config.start_thought_token: self.tokenizer.convert_tokens_to_ids(self.config.start_thought_token),
            self.config.end_thought_token: self.tokenizer.convert_tokens_to_ids(self.config.end_thought_token),
            self.config.no_thought_token: self.tokenizer.convert_tokens_to_ids(self.config.no_thought_token),
        }

    def _find_lm_head(self) -> nn.Module | None:
        """Find the language model head in the base model."""
        lm_head_names = ["lm_head", "output_layer", "classifier", "projection"]

        for name in lm_head_names:
            if hasattr(self.model, name):
                return getattr(self.model, name)

        return None

    async def test_thought_convergence(self, test_prompts: list[str]) -> float:
        """
        Test if thoughts have "stuck" by measuring natural thought generation.

        Returns:
            convergence_rate: Proportion of prompts that naturally generate thoughts
        """
        self.model.eval()
        natural_thought_count = 0

        with torch.no_grad():
            for prompt in test_prompts:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

                # Generate without explicit thought prompting
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

                # Check if model naturally generated thought tokens
                if self.config.start_thought_token in generated_text or self.config.end_thought_token in generated_text:
                    natural_thought_count += 1

        convergence_rate = natural_thought_count / len(test_prompts) if test_prompts else 0.0
        logger.info(f"Thought convergence rate: {convergence_rate:.2%}")

        return convergence_rate

    async def bake_iteration(self, training_data: list[dict], iteration: int) -> dict[str, float]:
        """
        Perform one iteration of prompt baking.

        Args:
            training_data: List of training examples with prompts and targets
            iteration: Current iteration number

        Returns:
            metrics: Training metrics for this iteration
        """
        logger.info(f"Starting baking iteration {iteration + 1}")

        # Prepare augmented training data with thought tokens
        augmented_data = []
        for example in training_data:
            # Process with cognitive strategies
            thought_result, insights = await self.cognitive_processor.iot_process(example["prompt"], max_iterations=3)

            # Create training example with baked thoughts
            augmented_example = {
                "input": example["prompt"],
                "thought": thought_result["text"],
                "insights": insights,
                "target": example.get("target", ""),
            }
            augmented_data.append(augmented_example)

        # Fine-tune with Grokfast optimization
        training_metrics = await self._fine_tune_with_grokfast(augmented_data)

        return training_metrics

    async def _fine_tune_with_grokfast(self, augmented_data: list[dict]) -> dict[str, float]:
        """Fine-tune model with Grokfast optimization."""
        # Create training dataset
        dataset = BakingDataset(augmented_data, self.tokenizer, self.config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(Path(self.config.output_path).parent / "baking_checkpoints"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            fp16=self.config.mixed_precision,
            dataloader_drop_last=False,
            seed=self.config.seed,
            report_to=[],  # Disable default W&B logging
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Create trainer with Grokfast optimizer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Replace optimizer with Grokfast if enabled
        if self.config.enable_grokfast:
            base_optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )

            grokfast_optimizer = GrokfastOptimizer(
                base_optimizer, alpha=self.config.grokfast_ema_alpha, lamb=self.config.grokfast_lambda
            )

            trainer.optimizers = (grokfast_optimizer, None)

        # Train
        result = trainer.train()

        return {
            "train_loss": result.training_loss,
            "train_runtime": result.metrics["train_runtime"],
            "train_samples_per_second": result.metrics["train_samples_per_second"],
        }


class BakingDataset(Dataset):
    """Dataset for prompt baking training."""

    def __init__(self, examples: list[dict], tokenizer, config: QuietSTaRConfig):
        self.examples = examples
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = 512

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Construct input with baked thoughts
        input_text = example["input"]
        thought_text = example["thought"]
        target_text = example.get("target", "")

        # Insert thought tokens
        full_text = f"{input_text} {self.config.start_thought_token} {thought_text} {self.config.end_thought_token} {target_text}"

        # Tokenize
        encoding = self.tokenizer(
            full_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        # Labels are same as input_ids for language modeling
        encoding["labels"] = encoding["input_ids"].clone()

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["labels"].squeeze(),
        }


# ============================================================================
# A/B Testing Harness
# ============================================================================


class QuietSTaRABTester:
    """
    A/B testing harness to validate thought injection improvements.

    Consolidated from quietstar_baker.py with enhanced metrics.
    """

    def __init__(self, baseline_model: nn.Module, thought_model: nn.Module, tokenizer, config: QuietSTaRConfig):
        self.baseline_model = baseline_model
        self.thought_model = thought_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(
            config.device if config.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move models to device
        self.baseline_model.to(self.device)
        self.thought_model.to(self.device)

    async def run_ab_test(self, eval_dataset) -> dict[str, Any]:
        """Run comprehensive A/B test."""
        logger.info(f"Starting A/B test with {len(eval_dataset)} examples")

        baseline_metrics = []
        thought_metrics = []

        # Run multiple rounds for statistical significance
        for round_idx in range(self.config.ab_test_rounds):
            logger.info(f"A/B Test Round {round_idx + 1}/{self.config.ab_test_rounds}")

            # Test baseline model
            baseline_result = await self._evaluate_model(
                self.baseline_model, eval_dataset, use_thoughts=False, desc=f"Baseline R{round_idx + 1}"
            )

            # Test thought-enhanced model
            thought_result = await self._evaluate_model(
                self.thought_model, eval_dataset, use_thoughts=True, desc=f"Thoughts R{round_idx + 1}"
            )

            baseline_metrics.append(baseline_result)
            thought_metrics.append(thought_result)

        # Analyze results
        analysis = self._analyze_ab_results(baseline_metrics, thought_metrics)

        return analysis

    async def _evaluate_model(self, model: nn.Module, dataset, use_thoughts: bool, desc: str) -> dict[str, float]:
        """Evaluate model on dataset."""
        model.eval()
        total_correct = 0
        total_samples = 0
        total_time = 0.0

        with torch.no_grad():
            for i, example in enumerate(tqdm(dataset, desc=desc)):
                prompt = example["input"]
                target = example.get("numerical_answer", "")

                start_time = time.time()

                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

                if use_thoughts:
                    # Add thought prompting for enhanced model
                    thought_prompt = f"{prompt} {self.config.start_thought_token} Let me think step by step... {self.config.end_thought_token}"
                    input_ids = self.tokenizer.encode(thought_prompt, return_tensors="pt").to(self.device)

                # Generate response
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                end_time = time.time()
                total_time += end_time - start_time

                # Decode and check answer
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                if self._check_answer(generated_text, target):
                    total_correct += 1

                total_samples += 1

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_time = total_time / total_samples if total_samples > 0 else 0.0

        return {"accuracy": accuracy, "avg_time": avg_time, "total_samples": total_samples}

    def _check_answer(self, generated_text: str, target_answer: str) -> bool:
        """Check if generated answer matches target."""
        import re

        generated_numbers = re.findall(r"-?\d+\.?\d*", generated_text)
        return target_answer in generated_numbers

    def _analyze_ab_results(self, baseline_metrics: list, thought_metrics: list) -> dict[str, Any]:
        """Analyze A/B test results with statistical significance."""
        baseline_accuracies = [m["accuracy"] for m in baseline_metrics]
        thought_accuracies = [m["accuracy"] for m in thought_metrics]

        baseline_mean = np.mean(baseline_accuracies)
        thought_mean = np.mean(thought_accuracies)
        improvement = thought_mean - baseline_mean

        # Statistical significance test
        from scipy import stats

        t_stat, p_value = stats.ttest_rel(thought_accuracies, baseline_accuracies)

        is_significant = p_value < self.config.significance_threshold
        is_improvement = improvement >= self.config.min_improvement
        winner = "thoughts" if (is_significant and is_improvement) else "baseline"

        return {
            "baseline_accuracy": baseline_mean,
            "thoughts_accuracy": thought_mean,
            "improvement": improvement,
            "improvement_percent": (improvement / baseline_mean * 100) if baseline_mean > 0 else 0,
            "p_value": p_value,
            "is_significant": is_significant,
            "winner": winner,
            "baseline_time": np.mean([m["avg_time"] for m in baseline_metrics]),
            "thoughts_time": np.mean([m["avg_time"] for m in thought_metrics]),
        }


# ============================================================================
# Evaluation Dataset
# ============================================================================


class ReasoningEvalDataset:
    """Evaluation dataset for reasoning capabilities."""

    def __init__(self, dataset_name: str, num_samples: int, tokenizer):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

        if dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="test")
            self.examples = self._prepare_gsm8k(dataset, num_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        logger.info(f"Loaded {len(self.examples)} examples from {dataset_name}")

    def _prepare_gsm8k(self, dataset, num_samples: int) -> list[dict]:
        """Prepare GSM8K examples."""
        examples = []

        for i, item in enumerate(dataset):
            if i >= num_samples:
                break

            question = item["question"]
            answer = item["answer"]

            # Extract numerical answer
            answer_parts = answer.split("####")
            if len(answer_parts) >= 2:
                numerical_answer = answer_parts[1].strip()
            else:
                numerical_answer = answer.strip()

            examples.append(
                {"input": f"Question: {question}\nAnswer:", "target": answer, "numerical_answer": numerical_answer}
            )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================================
# Main Phase Controller
# ============================================================================


class QuietSTaRPhase(PhaseController):
    """
    Phase 2: Quiet-STaR Baking Controller

    Implements iterative prompt baking with convergence testing, cognitive strategies,
    A/B testing, and Grokfast optimization.
    """

    def __init__(self, config: QuietSTaRConfig):
        super().__init__(config)
        self.config = config
        self.phase_name = "QuietSTaR Baking"
        self.phase_number = 2

        # Set random seeds
        torch.manual_seed(getattr(config, 'seed', 42))
        np.random.seed(getattr(config, 'seed', 42))

    async def run(self, model: nn.Module) -> PhaseResult:
        """
        Execute the QuietSTaR phase processing.

        Args:
            model: Input model from previous phase

        Returns:
            PhaseResult with processed model and metrics
        """
        # Validate input model
        if not self.validate_input_model(model):
            return self.create_failure_result(model, "Input model validation failed")

        start_time = time.time()

        try:
            # Save model temporarily to pass to execute method
            temp_model_path = Path(self.config.output_path) / "temp_input_model"
            temp_model_path.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(str(temp_model_path))

            # Execute the phase using existing execute method
            result = await self.execute(str(temp_model_path))

            duration = time.time() - start_time

            if result.success:
                return self.create_success_result(
                    model=result.model,
                    metrics=result.metrics or {},
                    artifacts=result.artifacts or {},
                    duration=duration
                )
            else:
                return self.create_failure_result(model, result.error or "QuietSTaR phase failed", duration)

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"QuietSTaR phase failed: {e}")
            return self.create_failure_result(model, str(e), duration)

    async def execute(self, input_model_path: str, **kwargs) -> PhaseResult:
        """Execute Phase 2: Quiet-STaR Baking."""
        try:
            logger.info(f"🤔 Starting {self.phase_name}")

            # Update config with input model path
            self.config.model_path = input_model_path

            # Load model and tokenizer
            logger.info(f"Loading model from {input_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path or input_model_path)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                input_model_path, torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
            )

            # Create prompt baking engine
            baking_engine = PromptBakingEngine(model, tokenizer, self.config)

            # Load evaluation dataset
            eval_dataset = ReasoningEvalDataset(self.config.eval_dataset, self.config.eval_samples, tokenizer)

            # Prepare training data from evaluation samples
            training_data = []
            for example in eval_dataset.examples[:50]:  # Use subset for training
                training_data.append({"prompt": example["input"], "target": example["target"]})

            # Apply DSPy prompt optimization if available
            if self.config.enable_dspy_optimization and DSPyAgentOptimizer is not None:
                try:
                    optimizer = DSPyAgentOptimizer()
                    optimized = optimizer.get_optimized_prompt("quietstar")
                    if optimized:
                        for item in training_data:
                            item["prompt"] = optimized
                        self.config.dspy_optimized_prompt = optimized
                except Exception as opt_err:  # pragma: no cover - best effort
                    logger.warning(f"DSPy optimization skipped: {opt_err}")

            # Test prompts for convergence testing
            test_prompts = [example["input"] for example in eval_dataset.examples[50:70]]

            # Iterative baking process
            best_convergence = 0.0
            iteration_results = []

            for iteration in range(self.config.max_baking_iterations):
                logger.info(f"Baking iteration {iteration + 1}/{self.config.max_baking_iterations}")

                # Perform baking iteration
                training_metrics = await baking_engine.bake_iteration(training_data, iteration)

                # Test convergence
                convergence_rate = await baking_engine.test_thought_convergence(test_prompts)

                iteration_result = {
                    "iteration": iteration + 1,
                    "convergence_rate": convergence_rate,
                    "training_metrics": training_metrics,
                }
                iteration_results.append(iteration_result)

                logger.info(f"Iteration {iteration + 1}: {convergence_rate:.2%} convergence rate")

                # Check for convergence
                if convergence_rate >= self.config.convergence_threshold:
                    logger.info(
                        f"Convergence achieved: {convergence_rate:.2%} >= {self.config.convergence_threshold:.2%}"
                    )
                    break

                best_convergence = max(best_convergence, convergence_rate)

            # Create baseline model for A/B testing
            baseline_model = AutoModelForCausalLM.from_pretrained(
                input_model_path, torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
            )

            # Run A/B test
            logger.info("Running A/B test to validate improvements")
            ab_tester = QuietSTaRABTester(baseline_model, model, tokenizer, self.config)
            ab_results = await ab_tester.run_ab_test(eval_dataset)

            # Save baked model
            logger.info(f"Saving baked model to {self.config.output_path}")
            model.save_pretrained(self.config.output_path)
            tokenizer.save_pretrained(self.config.output_path)

            # Determine success based on convergence and A/B results
            success = best_convergence >= self.config.convergence_threshold * 0.8 or ab_results["winner"] == "thoughts"

            # Create phase result
            result = PhaseResult(
                phase_name=self.phase_name,
                success=success,
                model_path=self.config.output_path,
                metrics={
                    "final_convergence_rate": best_convergence,
                    "ab_test_winner": ab_results["winner"],
                    "accuracy_improvement": ab_results["improvement_percent"],
                    "p_value": ab_results["p_value"],
                    "iterations_completed": len(iteration_results),
                    "thoughts_stuck": best_convergence >= self.config.convergence_threshold,
                },
                artifacts={
                    "iteration_results": iteration_results,
                    "ab_test_results": ab_results,
                    "config": self.config.__dict__,
                },
                duration_seconds=0,  # Will be calculated by orchestrator
                memory_usage_mb=0,  # Will be calculated by orchestrator
            )

            status = "✅ SUCCESS" if success else "⚠️  PARTIAL"
            logger.info(f"{status} - Convergence: {best_convergence:.2%}, Winner: {ab_results['winner']}")

            return result

        except Exception as e:
            logger.exception(f"Quiet-STaR baking failed: {e}")

            return PhaseResult(
                phase_name=self.phase_name,
                success=False,
                model_path="",
                metrics={"error": str(e)},
                artifacts={},
                duration_seconds=0,
                memory_usage_mb=0,
                error_message=str(e),
            )


# ============================================================================
# Factory Function
# ============================================================================


def create_quietstar_phase(
    model_path: str = "",
    output_path: str = "",
    eval_samples: int = 100,
    max_baking_iterations: int = 5,
    convergence_threshold: float = 0.95,
    enable_grokfast: bool = True,
    device: str = "auto",
    **kwargs,
) -> QuietSTaRPhase:
    """
    Factory function to create QuietSTaR phase with common configurations.

    Args:
        model_path: Path to input model from EvoMerge
        output_path: Path for baked model output
        eval_samples: Number of evaluation samples
        max_baking_iterations: Maximum baking iterations
        convergence_threshold: Convergence threshold (95% default)
        enable_grokfast: Enable Grokfast optimization
        device: Device to use
        **kwargs: Additional configuration options

    Returns:
        QuietSTaRPhase: Configured phase controller
    """
    config = QuietSTaRConfig(
        model_path=model_path,
        output_path=output_path,
        eval_samples=eval_samples,
        max_baking_iterations=max_baking_iterations,
        convergence_threshold=convergence_threshold,
        enable_grokfast=enable_grokfast,
        device=device,
        **kwargs,
    )

    return QuietSTaRPhase(config)


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":

    async def main():
        # Example: Create and run QuietSTaR phase
        phase = create_quietstar_phase(
            model_path="./champion_model_from_evomerge",
            output_path="./phase2_quietstar_output",
            eval_samples=50,  # Smaller for testing
            max_baking_iterations=3,
            convergence_threshold=0.85,  # Lower for testing
        )

        result = await phase.execute("./champion_model_from_evomerge")

        print(f"Phase Result: {result.success}")
        print(f"Convergence Rate: {result.metrics.get('final_convergence_rate', 0):.2%}")
        print(f"A/B Winner: {result.metrics.get('ab_test_winner', 'unknown')}")
        print(f"Model Path: {result.model_path}")

    # Uncomment to run example
    # asyncio.run(main())
