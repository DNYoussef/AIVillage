"""
Quiet-STaR Training Losses
Implements L = w_task * L_task + w_reflect * L_reflect + w_leak * L_leak
"""

import logging
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QuietSTaRConfig

logger = logging.getLogger(__name__)


@dataclass
class LossComponents:
    """Container for individual loss components."""

    total_loss: torch.Tensor
    task_loss: torch.Tensor
    reflection_loss: torch.Tensor
    leak_loss: torch.Tensor

    # Additional metrics
    leak_count: int = 0
    reflection_quality_score: float = 0.0
    thought_token_ratio: float = 0.0


class QuietSTaRLoss(nn.Module):
    """
    Combined loss function for Quiet-STaR training.

    Components:
    - L_task: Standard language modeling loss on non-thought tokens
    - L_reflect: Supervised loss on thought quality (teacher or heuristic)
    - L_leak: Penalty when thought content appears in user-visible output
    """

    def __init__(self, config: QuietSTaRConfig):
        super().__init__()
        self.config = config

        # Loss components
        self.task_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.reflection_loss_fn = nn.CrossEntropyLoss(reduction="none")

        # Precompile regex patterns for leak detection
        self.thought_leak_patterns = [
            re.compile(r"<SoT>.*?</SoT>", re.DOTALL),  # Complete thought blocks
            re.compile(r"<SoT>"),  # Dangling start tokens
            re.compile(r"</SoT>"),  # Dangling end tokens
            re.compile(r"<NoT>"),  # No-thought tokens in output
        ]

        logger.info(
            f"Initialized QuietSTaRLoss with weights: task={config.w_task}, "
            f"reflect={config.w_reflect}, leak={config.w_leak}"
        )

    def compute_task_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, thought_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard language modeling loss on non-thought tokens.

        Args:
            logits: [batch_size, seq_len, vocab_size] - model predictions
            labels: [batch_size, seq_len] - target tokens
            thought_mask: [batch_size, seq_len] - mask for thought regions

        Returns:
            task_loss: Scalar tensor with task loss
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for loss computation
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        flat_thought_mask = thought_mask.view(-1)

        # Compute per-token losses
        token_losses = self.task_loss_fn(flat_logits, flat_labels)

        # Mask out thought regions - only compute loss on regular content
        non_thought_mask = ~flat_thought_mask
        valid_losses = token_losses[non_thought_mask]

        if len(valid_losses) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return valid_losses.mean()

    def compute_reflection_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        thought_mask: torch.Tensor,
        reflection_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute supervised loss on thought/reflection quality.

        Args:
            logits: [batch_size, seq_len, vocab_size] - model predictions
            labels: [batch_size, seq_len] - target tokens
            thought_mask: [batch_size, seq_len] - mask for thought regions
            reflection_targets: Optional pre-computed quality targets

        Returns:
            reflection_loss: Scalar tensor with reflection loss
            quality_score: Average reflection quality score
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten tensors
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        flat_thought_mask = thought_mask.view(-1)

        # Extract thought regions
        thought_indices = flat_thought_mask.nonzero(as_tuple=True)[0]

        if len(thought_indices) == 0:
            # No thoughts in batch
            return torch.tensor(0.0, device=logits.device, requires_grad=True), 0.0

        thought_logits = flat_logits[thought_indices]
        thought_labels = flat_labels[thought_indices]

        if reflection_targets is not None:
            # Use provided reflection targets (from teacher model)
            flat_targets = reflection_targets.view(-1)
            thought_targets = flat_targets[thought_indices]
            reflection_losses = self.reflection_loss_fn(thought_logits, thought_targets)
        else:
            # Use heuristic quality assessment
            reflection_losses = self.reflection_loss_fn(thought_logits, thought_labels)

            # Apply quality-based weighting
            quality_weights = self._compute_heuristic_quality_weights(
                thought_logits, thought_labels, batch_size, seq_len, thought_mask
            )

            if quality_weights is not None:
                reflection_losses = reflection_losses * quality_weights

        reflection_loss = reflection_losses.mean()

        # Compute quality score for monitoring
        with torch.no_grad():
            thought_probs = F.softmax(thought_logits, dim=-1)
            predicted_tokens = torch.argmax(thought_probs, dim=-1)
            accuracy = (predicted_tokens == thought_labels).float().mean().item()

            # Entropy-based quality measure (lower entropy = more confident)
            entropy = -torch.sum(
                thought_probs * torch.log(thought_probs + 1e-10), dim=-1
            )
            avg_entropy = entropy.mean().item()
            quality_score = accuracy * (
                1 - avg_entropy / torch.log(torch.tensor(vocab_size))
            )

        return reflection_loss, quality_score

    def _compute_heuristic_quality_weights(
        self,
        thought_logits: torch.Tensor,
        thought_labels: torch.Tensor,
        batch_size: int,
        seq_len: int,
        thought_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Compute quality-based weights for reflection loss using heuristics.

        This encourages higher-quality thoughts based on:
        - Confidence of predictions
        - Diversity of token usage
        - Consistency with surrounding context
        """
        if not self.config.enable_training:
            return None

        with torch.no_grad():
            # Confidence weighting: higher weight for more confident predictions
            thought_probs = F.softmax(thought_logits, dim=-1)
            confidence = torch.max(thought_probs, dim=-1)[
                0
            ]  # Max probability per token
            confidence_weight = (confidence > 0.5).float()  # Binary threshold for now

            # Diversity weighting: encourage diverse vocabulary in thoughts
            vocab_usage = torch.zeros(
                thought_logits.size(0), device=thought_logits.device
            )
            for i in range(thought_logits.size(0)):
                unique_tokens = torch.unique(thought_labels[i : i + 1])
                vocab_usage[i] = len(unique_tokens) / max(
                    1, thought_labels[i : i + 1].size(0)
                )

            diversity_weight = torch.clamp(vocab_usage * 2, 0, 1)  # Scale to [0, 1]

            # Combine weights
            quality_weights = 0.6 * confidence_weight + 0.4 * diversity_weight

            # Apply threshold from config
            quality_weights = (
                quality_weights > self.config.reflection_heuristic_threshold
            ).float()

        return quality_weights

    def compute_leak_loss(
        self,
        generated_text: list[str],
        input_ids: torch.Tensor,
        thought_mask: torch.Tensor,
        special_token_ids: dict[str, int],
    ) -> tuple[torch.Tensor, int]:
        """
        Compute penalty when thought content appears in user-visible output.

        Args:
            generated_text: List of generated text strings (decoded)
            input_ids: [batch_size, seq_len] - input token IDs
            thought_mask: [batch_size, seq_len] - thought region mask
            special_token_ids: Special token ID mapping

        Returns:
            leak_loss: Scalar tensor with leak penalty
            leak_count: Number of detected leaks
        """
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Get special tokens for detection
        sot_token = self.config.start_of_thought_token
        eot_token = self.config.end_of_thought_token
        not_token = self.config.no_thought_token

        total_leak_penalty = torch.tensor(0.0, device=device, requires_grad=True)
        total_leaks = 0

        for i, text in enumerate(generated_text):
            if i >= batch_size:
                break

            # Count different types of leaks
            leak_penalty = 0.0
            leak_count = 0

            # 1. Complete thought blocks leaked
            thought_blocks = self.thought_leak_patterns[0].findall(text)
            if thought_blocks:
                leak_count += len(thought_blocks)
                if self.config.leak_penalty_type == "exponential":
                    leak_penalty += len(thought_blocks) ** 2
                else:
                    leak_penalty += len(thought_blocks)

            # 2. Dangling start/end tokens
            for pattern_idx in [1, 2, 3]:  # SoT, EoT, NoT patterns
                matches = self.thought_leak_patterns[pattern_idx].findall(text)
                if matches:
                    leak_count += len(matches)
                    leak_penalty += (
                        len(matches) * 0.5
                    )  # Lower penalty for dangling tokens

            # 3. Check for leaked thought content based on input
            if i < len(generated_text) and torch.any(thought_mask[i]):
                # Extract actual thought content from input
                thought_indices = thought_mask[i].nonzero(as_tuple=True)[0]
                if len(thought_indices) > 0:
                    # This is a simplified check - in practice you'd decode thought regions
                    # and check for semantic similarity with output
                    pass

            if leak_count > 0:
                total_leaks += leak_count
                penalty_tensor = torch.tensor(
                    leak_penalty, device=device, requires_grad=True
                )
                total_leak_penalty = total_leak_penalty + penalty_tensor

        # Normalize by batch size
        if batch_size > 0:
            total_leak_penalty = total_leak_penalty / batch_size

        return total_leak_penalty, total_leaks

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        thought_mask: torch.Tensor,
        special_token_ids: dict[str, int],
        generated_texts: list[str] | None = None,
        reflection_targets: torch.Tensor | None = None,
    ) -> LossComponents:
        """
        Compute combined Quiet-STaR loss.

        Args:
            logits: [batch_size, seq_len, vocab_size] - model predictions
            labels: [batch_size, seq_len] - target tokens
            thought_mask: [batch_size, seq_len] - thought region mask
            special_token_ids: Special token ID mapping
            generated_texts: Decoded text for leak detection (optional)
            reflection_targets: Teacher targets for reflection loss (optional)

        Returns:
            LossComponents with total loss and individual components
        """
        # Compute individual loss components
        task_loss = self.compute_task_loss(logits, labels, thought_mask)

        reflection_loss, quality_score = self.compute_reflection_loss(
            logits, labels, thought_mask, reflection_targets
        )

        # Compute leak loss if generated texts provided
        if generated_texts is not None:
            leak_loss, leak_count = self.compute_leak_loss(
                generated_texts,
                labels,  # Use labels as proxy for input_ids
                thought_mask,
                special_token_ids,
            )
        else:
            leak_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            leak_count = 0

        # Combine losses with configured weights
        total_loss = (
            self.config.w_task * task_loss
            + self.config.w_reflect * reflection_loss
            + self.config.w_leak * leak_loss
        )

        # Compute thought token ratio for monitoring
        thought_ratio = thought_mask.float().mean().item()

        return LossComponents(
            total_loss=total_loss,
            task_loss=task_loss,
            reflection_loss=reflection_loss,
            leak_loss=leak_loss,
            leak_count=leak_count,
            reflection_quality_score=quality_score,
            thought_token_ratio=thought_ratio,
        )


class ReflectionQualityAssessor:
    """
    Heuristic assessment of reflection quality for training feedback.
    """

    def __init__(self, config: QuietSTaRConfig):
        self.config = config

        # Quality assessment criteria
        self.quality_patterns = {
            "structured_thinking": [
                re.compile(r"\d+\)", re.MULTILINE),  # Numbered steps: "1) ..."
                re.compile(r"step \d+", re.IGNORECASE),
                re.compile(r"first,|second,|then,|next,|finally", re.IGNORECASE),
            ],
            "test_driven_reasoning": [
                re.compile(r"test|unit|case|example", re.IGNORECASE),
                re.compile(r"assert|expect|should", re.IGNORECASE),
                re.compile(r"input.*output", re.IGNORECASE),
            ],
            "edge_conditions": [
                re.compile(r"edge|corner|boundary", re.IGNORECASE),
                re.compile(r"empty|null|zero|negative", re.IGNORECASE),
                re.compile(r"overflow|underflow|limit", re.IGNORECASE),
            ],
            "invariant_checks": [
                re.compile(r"invariant|constraint|condition", re.IGNORECASE),
                re.compile(r"always|never|must|should", re.IGNORECASE),
                re.compile(r"valid|invalid|check", re.IGNORECASE),
            ],
        }

    def assess_reflection_quality(self, reflection_text: str) -> dict[str, float]:
        """
        Assess quality of a reflection text using heuristics.

        Args:
            reflection_text: Text content of the reflection

        Returns:
            Quality scores for different criteria
        """
        scores = {}

        for criteria, patterns in self.quality_patterns.items():
            matches = 0
            for pattern in patterns:
                matches += len(pattern.findall(reflection_text))

            # Normalize by text length and number of patterns
            text_words = len(reflection_text.split())
            normalized_score = min(
                1.0, matches / (text_words / 10 + 1)
            )  # Scale by text length
            scores[criteria] = normalized_score

        # Overall quality score (weighted average)
        overall_score = (
            0.3 * scores.get("structured_thinking", 0)
            + 0.25 * scores.get("test_driven_reasoning", 0)
            + 0.25 * scores.get("edge_conditions", 0)
            + 0.2 * scores.get("invariant_checks", 0)
        )

        scores["overall"] = overall_score
        return scores

    def create_quality_targets(
        self, reflection_texts: list[str], tokenizer
    ) -> torch.Tensor:
        """
        Create quality-weighted targets for reflection loss.

        Args:
            reflection_texts: List of reflection text strings
            tokenizer: Tokenizer for encoding

        Returns:
            Quality weights tensor
        """
        quality_weights = []

        for text in reflection_texts:
            quality_scores = self.assess_reflection_quality(text)
            overall_quality = quality_scores["overall"]

            # Convert to weight (higher quality = higher weight)
            weight = max(0.1, overall_quality)  # Minimum weight to avoid zero gradients
            quality_weights.append(weight)

        return torch.tensor(quality_weights, dtype=torch.float)


def create_training_batch_with_thoughts(
    prompts: list[str],
    responses: list[str],
    config: QuietSTaRConfig,
    tokenizer,
    include_thoughts: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Create training batch with thought annotations.

    Args:
        prompts: List of input prompts
        responses: List of target responses
        config: QuietSTaR configuration
        tokenizer: Tokenizer for encoding
        include_thoughts: Whether to include thought segments

    Returns:
        Batch dictionary with input_ids, labels, thought_mask, etc.
    """
    batch_inputs = []
    batch_labels = []
    batch_thought_masks = []

    sot_token = config.start_of_thought_token
    eot_token = config.end_of_thought_token
    not_token = config.no_thought_token

    for prompt, response in zip(prompts, responses, strict=False):
        if include_thoughts and torch.rand(1).item() < config.thought_ratio:
            # Add thought segment to response
            thought_content = f" {sot_token} Let me think step by step about this problem... {eot_token} "
            full_text = prompt + thought_content + response
        else:
            # Add no-thought token
            full_text = prompt + f" {not_token} " + response

        # Tokenize
        encoded = tokenizer.encode(full_text, return_tensors="pt", truncation=True)
        input_ids = encoded.squeeze(0)

        # Create labels (shifted for language modeling)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = tokenizer.pad_token_id or -100

        # Create thought mask
        thought_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # Find thought regions in tokenized text
        tokens_str = tokenizer.decode(input_ids, skip_special_tokens=False)
        if sot_token in tokens_str and eot_token in tokens_str:
            # Mark thought regions (simplified - would need proper token alignment)
            sot_pos = tokens_str.find(sot_token)
            eot_pos = tokens_str.find(eot_token) + len(eot_token)
            if sot_pos >= 0 and eot_pos > sot_pos:
                # This is a simplification - proper implementation would align tokens
                thought_start = len(tokenizer.encode(tokens_str[:sot_pos])) - 1
                thought_end = len(tokenizer.encode(tokens_str[:eot_pos])) - 1
                thought_start = max(0, min(thought_start, len(input_ids) - 1))
                thought_end = max(0, min(thought_end, len(input_ids)))
                thought_mask[thought_start:thought_end] = True

        batch_inputs.append(input_ids)
        batch_labels.append(labels)
        batch_thought_masks.append(thought_mask)

    # Pad sequences to same length
    max_len = max(len(seq) for seq in batch_inputs)
    padded_inputs = []
    padded_labels = []
    padded_masks = []

    for inputs, labels, mask in zip(
        batch_inputs, batch_labels, batch_thought_masks, strict=False
    ):
        pad_len = max_len - len(inputs)
        if pad_len > 0:
            pad_token_id = tokenizer.pad_token_id or 0
            inputs = F.pad(inputs, (0, pad_len), value=pad_token_id)
            labels = F.pad(labels, (0, pad_len), value=-100)
            mask = F.pad(mask, (0, pad_len), value=False)

        padded_inputs.append(inputs)
        padded_labels.append(labels)
        padded_masks.append(mask)

    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels),
        "thought_mask": torch.stack(padded_masks),
        "attention_mask": (torch.stack(padded_inputs) != (tokenizer.pad_token_id or 0)),
    }
