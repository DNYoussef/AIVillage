"""
Quiet-STaR Sampling Policy and Inference Stripping
Handles thought generation during training and stripping during inference.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .config import QuietSTaRConfig

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    """Result from thought-aware sampling."""

    generated_ids: torch.Tensor
    thought_segments: list[dict[str, Any]]
    stripped_ids: torch.Tensor | None = None  # IDs with thoughts stripped
    generation_stats: dict[str, Any] = None


class ThoughtSampler:
    """
    Sampling policy for Quiet-STaR that controls thought generation.

    During training: sample p(thoughts) = thought_ratio
    During inference: default to quiet mode (strip thoughts)
    """

    def __init__(self, config: QuietSTaRConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Get special token IDs
        self.special_token_ids = self._get_special_token_ids()

        logger.info(f"Initialized ThoughtSampler with thought_ratio={config.thought_ratio}")

    def _get_special_token_ids(self) -> dict[str, int]:
        """Get special token IDs from tokenizer."""
        special_tokens = {}

        for token in [
            self.config.start_of_thought_token,
            self.config.end_of_thought_token,
            self.config.no_thought_token,
        ]:
            try:
                token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
                special_tokens[token] = token_id
            except (IndexError, KeyError):
                logger.warning(f"Could not find token ID for {token}")
                special_tokens[token] = -1

        return special_tokens

    def should_generate_thoughts(self, training_mode: bool = True, step: int | None = None) -> bool:
        """
        Decide whether to generate thoughts based on policy.

        Args:
            training_mode: Whether model is in training mode
            step: Current training step (for curriculum)

        Returns:
            Boolean indicating whether to generate thoughts
        """
        if not training_mode:
            # Inference mode: only generate thoughts if explicitly enabled
            return not self.config.strip_thoughts_in_inference

        # Training mode: sample based on thought_ratio
        return torch.rand(1).item() < self.config.thought_ratio

    def sample_thought_insertion_points(self, input_ids: torch.Tensor, max_insertions: int = 3) -> list[int]:
        """
        Sample positions where thoughts should be inserted.

        Args:
            input_ids: [seq_len] input token sequence
            max_insertions: Maximum number of thought insertions

        Returns:
            List of positions where thoughts should be inserted
        """
        seq_len = input_ids.size(0)

        # Avoid inserting thoughts at the very beginning or end
        valid_positions = list(range(1, seq_len - 1))

        # Sample insertion points
        num_insertions = torch.randint(1, max_insertions + 1, (1,)).item()
        num_insertions = min(num_insertions, len(valid_positions))

        if num_insertions == 0:
            return []

        # Sample positions without replacement
        insertion_indices = torch.randperm(len(valid_positions))[:num_insertions]
        insertion_positions = [valid_positions[i] for i in insertion_indices]

        return sorted(insertion_positions)

    def generate_thought_segment(
        self,
        model,
        context_ids: torch.Tensor,
        max_thought_length: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Generate a single thought segment.

        Args:
            model: Language model for generation
            context_ids: [batch_size, seq_len] context tokens
            max_thought_length: Maximum thought length (overrides config)
            temperature: Sampling temperature (overrides config)

        Returns:
            Dictionary with generated thought information
        """
        device = context_ids.device
        batch_size = context_ids.size(0)

        max_length = max_thought_length or self.config.max_thought_tokens
        temp = temperature or self.config.thought_temperature

        sot_id = self.special_token_ids.get(self.config.start_of_thought_token, -1)
        eot_id = self.special_token_ids.get(self.config.end_of_thought_token, -1)

        if sot_id == -1 or eot_id == -1:
            logger.warning("Missing special token IDs, cannot generate thoughts")
            return {"thought_tokens": [], "thought_text": "", "success": False}

        # Start with SoT token
        current_ids = torch.cat([context_ids, torch.full((batch_size, 1), sot_id, device=device)], dim=1)

        generated_tokens = [sot_id]

        # Generate thought content
        for step in range(max_length):
            with torch.no_grad():
                outputs = model(current_ids)

                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temp

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated_tokens.append(next_token.item())
                current_ids = torch.cat([current_ids, next_token], dim=1)

                # Stop if we generate end-of-thought
                if next_token.item() == eot_id:
                    break

                # Stop at EOS or other stopping criteria
                if hasattr(self.tokenizer, "eos_token_id") and next_token.item() == self.tokenizer.eos_token_id:
                    # Add explicit end-of-thought
                    current_ids = torch.cat(
                        [
                            current_ids,
                            torch.full((batch_size, 1), eot_id, device=device),
                        ],
                        dim=1,
                    )
                    generated_tokens.append(eot_id)
                    break

        # Ensure we end with EoT token
        if generated_tokens[-1] != eot_id:
            current_ids = torch.cat([current_ids, torch.full((batch_size, 1), eot_id, device=device)], dim=1)
            generated_tokens.append(eot_id)

        # Decode thought text
        thought_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

        return {
            "thought_tokens": generated_tokens,
            "thought_text": thought_text,
            "thought_ids": current_ids[:, context_ids.size(1) :],  # Just the thought part
            "success": True,
            "length": len(generated_tokens),
        }

    def sample_with_thoughts(
        self,
        model,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        force_thoughts: bool | None = None,
    ) -> SamplingResult:
        """
        Sample text generation with thought insertion policy.

        Args:
            model: Language model to use
            input_ids: [batch_size, seq_len] input tokens
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling vs greedy
            force_thoughts: Override thought generation decision

        Returns:
            SamplingResult with generated tokens and thought information
        """
        device = input_ids.device
        batch_size, initial_length = input_ids.shape

        # Decide whether to generate thoughts
        generate_thoughts = (
            force_thoughts if force_thoughts is not None else self.should_generate_thoughts(model.training)
        )

        current_ids = input_ids.clone()
        thought_segments = []
        generation_stats = {
            "thoughts_generated": 0,
            "total_thought_tokens": 0,
            "thought_positions": [],
        }

        not_id = self.special_token_ids.get(self.config.no_thought_token, -1)

        # If not generating thoughts, add NoT token
        if not generate_thoughts and not_id >= 0:
            current_ids = torch.cat([current_ids, torch.full((batch_size, 1), not_id, device=device)], dim=1)

        # Generate tokens
        for step in range(max_new_tokens):
            # Decide if we should insert a thought at this position
            should_insert_thought = (
                generate_thoughts
                and step > 5  # Don't insert thoughts immediately
                and step < max_new_tokens - 10  # Leave room after thoughts
                and torch.rand(1).item() < 0.1  # 10% chance per step
            )

            if should_insert_thought:
                # Generate thought segment
                thought_info = self.generate_thought_segment(
                    model=model,
                    context_ids=current_ids,
                    temperature=self.config.thought_temperature,
                )

                if thought_info["success"]:
                    # Add thought to sequence
                    thought_ids = thought_info["thought_ids"]
                    current_ids = torch.cat([current_ids, thought_ids], dim=1)

                    # Track thought segment
                    thought_segments.append(
                        {
                            "position": current_ids.size(1) - thought_ids.size(1),
                            "length": thought_ids.size(1),
                            "tokens": thought_info["thought_tokens"],
                            "text": thought_info["thought_text"],
                        }
                    )

                    generation_stats["thoughts_generated"] += 1
                    generation_stats["total_thought_tokens"] += thought_ids.size(1)
                    generation_stats["thought_positions"].append(step)

                    # Skip regular token generation this step
                    continue

            # Regular token generation
            with torch.no_grad():
                outputs = model(current_ids)

                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    logits = outputs

                next_token_logits = logits[:, -1, :] / temperature

                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                current_ids = torch.cat([current_ids, next_token], dim=1)

                # Check for early stopping
                if hasattr(self.tokenizer, "eos_token_id") and next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Check max length
                if current_ids.size(1) >= initial_length + max_new_tokens:
                    break

        # Create stripped version for inference
        stripped_ids = self.strip_thoughts_from_sequence(current_ids) if not model.training else None

        return SamplingResult(
            generated_ids=current_ids,
            thought_segments=thought_segments,
            stripped_ids=stripped_ids,
            generation_stats=generation_stats,
        )

    def strip_thoughts_from_sequence(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Strip thought tokens from generated sequence.

        Args:
            input_ids: [batch_size, seq_len] token sequence with thoughts

        Returns:
            stripped_ids: [batch_size, new_seq_len] sequence with thoughts removed
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        sot_id = self.special_token_ids.get(self.config.start_of_thought_token, -1)
        eot_id = self.special_token_ids.get(self.config.end_of_thought_token, -1)
        not_id = self.special_token_ids.get(self.config.no_thought_token, -1)

        stripped_sequences = []

        for b in range(batch_size):
            sequence = input_ids[b].tolist()
            stripped_sequence = []
            i = 0

            while i < len(sequence):
                token = sequence[i]

                if token == sot_id:
                    # Skip everything until EoT
                    i += 1
                    while i < len(sequence) and sequence[i] != eot_id:
                        i += 1
                    if i < len(sequence):  # Skip the EoT token too
                        i += 1
                    continue

                elif token == not_id:
                    # Skip no-thought tokens
                    i += 1
                    continue

                else:
                    # Keep regular tokens
                    stripped_sequence.append(token)
                    i += 1

            stripped_sequences.append(stripped_sequence)

        # Pad to same length
        if stripped_sequences:
            max_stripped_len = max(len(seq) for seq in stripped_sequences)

            padded_sequences = []
            for seq in stripped_sequences:
                pad_len = max_stripped_len - len(seq)
                if pad_len > 0:
                    pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
                    seq = seq + [pad_token_id] * pad_len
                padded_sequences.append(seq)

            return torch.tensor(padded_sequences, device=device)

        else:
            # All sequences were empty after stripping
            return torch.empty((batch_size, 0), device=device, dtype=input_ids.dtype)

    def decode_with_thought_annotation(self, input_ids: torch.Tensor, skip_special_tokens: bool = False) -> list[str]:
        """
        Decode sequences with thought annotation preserved or stripped.

        Args:
            input_ids: [batch_size, seq_len] token sequences
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            List of decoded strings
        """
        batch_size = input_ids.size(0)
        decoded_texts = []

        for b in range(batch_size):
            sequence = input_ids[b]

            # Remove padding tokens
            if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                non_pad_mask = sequence != self.tokenizer.pad_token_id
                sequence = sequence[non_pad_mask]

            # Decode
            if skip_special_tokens:
                # Strip thoughts before decoding
                stripped = self.strip_thoughts_from_sequence(sequence.unsqueeze(0))
                text = self.tokenizer.decode(stripped[0], skip_special_tokens=True)
            else:
                text = self.tokenizer.decode(sequence, skip_special_tokens=False)

            decoded_texts.append(text)

        return decoded_texts


class ThoughtLeakDetector:
    """
    Utility class for detecting thought leakage in generated text.
    """

    def __init__(self, config: QuietSTaRConfig):
        self.config = config

        # Compile leak detection patterns
        self.leak_patterns = [
            re.compile(
                rf"{re.escape(config.start_of_thought_token)}.*?{re.escape(config.end_of_thought_token)}",
                re.DOTALL,
            ),
            re.compile(rf"{re.escape(config.start_of_thought_token)}"),
            re.compile(rf"{re.escape(config.end_of_thought_token)}"),
            re.compile(rf"{re.escape(config.no_thought_token)}"),
        ]

        # Semantic leak patterns (thoughts that might leak without explicit tokens)
        self.semantic_leak_patterns = [
            re.compile(r"let me think|thinking about|my thought|I think", re.IGNORECASE),
            re.compile(r"step \d+|first,|second,|third,", re.IGNORECASE),
            re.compile(r"reasoning|analysis|conclusion", re.IGNORECASE),
        ]

    def detect_leaks(self, text: str, check_semantic: bool = True) -> dict[str, Any]:
        """
        Detect various types of thought leakage in text.

        Args:
            text: Text to check for leaks
            check_semantic: Whether to check for semantic leaks

        Returns:
            Dictionary with leak detection results
        """
        results = {
            "has_leaks": False,
            "token_leaks": [],
            "semantic_leaks": [],
            "leak_count": 0,
            "severity": 0.0,
        }

        # Check for explicit token leaks
        for i, pattern in enumerate(self.leak_patterns):
            matches = pattern.findall(text)
            if matches:
                results["has_leaks"] = True
                results["token_leaks"].extend(matches)
                results["leak_count"] += len(matches)

                # Weight different leak types
                if i == 0:  # Complete thought blocks
                    results["severity"] += len(matches) * 1.0
                else:  # Dangling tokens
                    results["severity"] += len(matches) * 0.5

        # Check for semantic leaks
        if check_semantic:
            for pattern in self.semantic_leak_patterns:
                matches = pattern.findall(text)
                if matches:
                    results["semantic_leaks"].extend(matches)
                    results["severity"] += len(matches) * 0.2  # Lower severity

        return results

    def is_safe_output(self, text: str, threshold: float = 0.1) -> bool:
        """
        Check if output is safe (no significant leaks).

        Args:
            text: Text to check
            threshold: Severity threshold for safety

        Returns:
            Boolean indicating if output is safe
        """
        leak_results = self.detect_leaks(text)

        # Any explicit token leaks are unsafe
        if leak_results["token_leaks"]:
            return False

        # Check semantic leak severity
        return leak_results["severity"] <= threshold


def create_inference_sampler(config: QuietSTaRConfig, tokenizer) -> ThoughtSampler:
    """
    Create sampler configured for inference (thoughts stripped).
    """
    inference_config = config
    inference_config.strip_thoughts_in_inference = True
    inference_config.thought_ratio = 0.0

    return ThoughtSampler(inference_config, tokenizer)


def create_training_sampler(config: QuietSTaRConfig, tokenizer) -> ThoughtSampler:
    """
    Create sampler configured for training (thoughts enabled).
    """
    training_config = config
    training_config.strip_thoughts_in_inference = False

    return ThoughtSampler(training_config, tokenizer)
