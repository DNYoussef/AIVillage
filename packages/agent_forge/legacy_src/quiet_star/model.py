"""
Quiet-STaR Model Architecture Extensions
Adds thought-aware mixing head and thought stripping capabilities.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QuietSTaRConfig

logger = logging.getLogger(__name__)


@dataclass
class ThoughtSegment:
    """Represents a parsed thought segment in the sequence."""

    start_idx: int
    end_idx: int
    thought_tokens: list[int]
    is_thought: bool  # True for <SoT>...</SoT>, False for regular content


class ThoughtMixingHead(nn.Module):
    """
    Mixing head that processes hidden thought representations but strips them from output.

    The key insight is that during forward pass, we:
    1. Process the full sequence (including thoughts) through the base model
    2. Use a mixing head to blend thought-aware representations
    3. Strip thought tokens from the final output logits
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        config: QuietSTaRConfig,
        base_lm_head: nn.Module | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config = config

        # Store reference to base language model head
        self.base_lm_head = base_lm_head

        # Thought-aware processing layers
        self.thought_detector = nn.Linear(hidden_size, 3)  # [no_thought, start_thought, end_thought]
        self.thought_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )

        # Mixing and projection layers
        self.thought_gate = nn.Linear(hidden_size, 1)  # Controls thought influence
        self.context_mixer = nn.Linear(hidden_size * 2, hidden_size)  # Mix thought + regular context
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        # Initialize with small weights to start conservative
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
        """
        Parse input sequences to identify thought segments.

        Args:
            input_ids: [batch_size, seq_len] tensor of token IDs
            special_token_ids: Mapping from special tokens to IDs

        Returns:
            List of ThoughtSegment lists, one per batch item
        """
        batch_size, seq_len = input_ids.shape
        sot_id = special_token_ids.get(self.config.start_of_thought_token, -1)
        eot_id = special_token_ids.get(self.config.end_of_thought_token, -1)
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
                                start_idx=start_idx,
                                end_idx=end_idx,
                                thought_tokens=thought_tokens,
                                is_thought=True,
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
                        ThoughtSegment(
                            start_idx=i,
                            end_idx=i + 1,
                            thought_tokens=[sequence[i]],
                            is_thought=False,
                        )
                    )
                    i += 1

                else:
                    # Regular token, group with consecutive regular tokens
                    start_idx = i
                    while i < seq_len and sequence[i] not in [sot_id, eot_id, not_id]:
                        i += 1

                    segments.append(
                        ThoughtSegment(
                            start_idx=start_idx,
                            end_idx=i,
                            thought_tokens=sequence[start_idx:i],
                            is_thought=False,
                        )
                    )

            batch_segments.append(segments)

        return batch_segments

    def create_thought_mask(self, input_ids: torch.Tensor, special_token_ids: dict[str, int]) -> torch.Tensor:
        """
        Create mask indicating which tokens are inside thought segments.

        Returns:
            thought_mask: [batch_size, seq_len] boolean tensor
        """
        batch_segments = self.parse_thought_segments(input_ids, special_token_ids)
        batch_size, seq_len = input_ids.shape

        thought_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)

        for b, segments in enumerate(batch_segments):
            for segment in segments:
                if segment.is_thought:
                    thought_mask[b, segment.start_idx : segment.end_idx] = True

        return thought_mask

    def encode_thoughts(self, hidden_states: torch.Tensor, thought_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply thought-specific encoding to thought regions.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            thought_mask: [batch_size, seq_len] boolean mask

        Returns:
            encoded_states: [batch_size, seq_len, hidden_size] with thought regions processed
        """
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
                    thought_hidden = hidden_states[b, thought_indices].unsqueeze(0)  # [1, n_thought, hidden]

                    # Apply transformer encoder to thought sequence
                    thought_encoded = self.thought_encoder(thought_hidden)  # [1, n_thought, hidden]

                    # Put encoded thoughts back
                    encoded_states[b, thought_indices] = thought_encoded.squeeze(0)

        return encoded_states

    def mix_contexts(
        self,
        regular_hidden: torch.Tensor,
        thought_hidden: torch.Tensor,
        thought_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mix regular and thought contexts using gating mechanism.

        Args:
            regular_hidden: [batch_size, seq_len, hidden_size] - regular context
            thought_hidden: [batch_size, seq_len, hidden_size] - thought-aware context
            thought_mask: [batch_size, seq_len] - thought regions

        Returns:
            mixed_hidden: [batch_size, seq_len, hidden_size] - mixed representations
        """
        # Compute thought influence gate
        gate_logits = self.thought_gate(thought_hidden)  # [batch_size, seq_len, 1]
        thought_gate = torch.sigmoid(gate_logits)

        # Mix regular and thought contexts
        concatenated = torch.cat([regular_hidden, thought_hidden], dim=-1)  # [..., 2*hidden_size]
        mixed_context = self.context_mixer(concatenated)  # [..., hidden_size]

        # Apply gating - stronger mixing in thought regions
        gate_mask = thought_mask.float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        effective_gate = thought_gate * gate_mask + (1 - gate_mask) * 0.1  # Low mixing outside thoughts

        mixed_hidden = (1 - effective_gate) * regular_hidden + effective_gate * mixed_context

        return mixed_hidden

    def strip_thought_tokens(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        special_token_ids: dict[str, int],
    ) -> torch.Tensor:
        """
        Strip thought token logits from output when in inference mode.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            input_ids: [batch_size, seq_len] - for context
            special_token_ids: Special token ID mapping

        Returns:
            stripped_logits: Logits with thought tokens suppressed
        """
        if not self.config.strip_thoughts_in_inference:
            return logits

        # Suppress thought token probabilities
        sot_id = special_token_ids.get(self.config.start_of_thought_token, -1)
        eot_id = special_token_ids.get(self.config.end_of_thought_token, -1)

        stripped_logits = logits.clone()

        if sot_id >= 0 and sot_id < logits.size(-1):
            stripped_logits[:, :, sot_id] = -float("inf")
        if eot_id >= 0 and eot_id < logits.size(-1):
            stripped_logits[:, :, eot_id] = -float("inf")

        return stripped_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        special_token_ids: dict[str, int],
        return_thought_representations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Forward pass through thought mixing head.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - base model hidden states
            input_ids: [batch_size, seq_len] - input token IDs
            special_token_ids: Special token ID mapping
            return_thought_representations: Whether to return intermediate representations

        Returns:
            logits: [batch_size, seq_len, vocab_size] - output logits (thoughts stripped in inference)
            representations: Dict with intermediate states (if return_thought_representations=True)
        """
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
            logits = self.strip_thought_tokens(logits, input_ids, special_token_ids)

        if return_thought_representations:
            representations = {
                "thought_mask": thought_mask,
                "thought_encoded": thought_encoded,
                "mixed_hidden": mixed_hidden,
                "raw_logits": logits,
            }
            return logits, representations

        return logits


class QuietSTaRModelWrapper(nn.Module):
    """
    Wrapper that adds Quiet-STaR capabilities to existing language models.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: QuietSTaRConfig,
        special_token_ids: dict[str, int] | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.special_token_ids = special_token_ids or {}

        # Extract model dimensions
        if hasattr(base_model, "config"):
            self.hidden_size = base_model.config.hidden_size
            self.vocab_size = base_model.config.vocab_size
        else:
            # Try to infer from model structure
            self.hidden_size = self._infer_hidden_size()
            self.vocab_size = self._infer_vocab_size()

        # Get reference to base LM head
        base_lm_head = self._find_lm_head()

        # Create thought mixing head
        self.thought_head = ThoughtMixingHead(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            config=config,
            base_lm_head=base_lm_head,
        )

        logger.info(
            f"Initialized QuietSTaRModelWrapper with hidden_size={self.hidden_size}, vocab_size={self.vocab_size}"
        )

    def _infer_hidden_size(self) -> int:
        """Infer hidden size from base model."""
        # Look for common hidden size attributes
        for attr in ["hidden_size", "d_model", "embed_dim"]:
            if hasattr(self.base_model.config, attr):
                return getattr(self.base_model.config, attr)

        # Fallback: inspect model parameters
        for param in self.base_model.parameters():
            if param.dim() >= 2:
                return param.size(-1)

        raise ValueError("Could not infer hidden size from base model")

    def _infer_vocab_size(self) -> int:
        """Infer vocabulary size from base model."""
        if hasattr(self.base_model.config, "vocab_size"):
            return self.base_model.config.vocab_size

        # Look for language model head
        lm_head = self._find_lm_head()
        if lm_head is not None:
            return lm_head.weight.size(0)

        raise ValueError("Could not infer vocab size from base model")

    def _find_lm_head(self) -> nn.Module | None:
        """Find the language model head in the base model."""
        # Common LM head names
        lm_head_names = ["lm_head", "output_layer", "classifier", "projection"]

        for name in lm_head_names:
            if hasattr(self.base_model, name):
                return getattr(self.base_model, name)

        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with Quiet-STaR processing.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            **kwargs: Additional arguments for base model

        Returns:
            outputs: Dictionary with logits and optional intermediate representations
        """
        # Run base model to get hidden states
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        # Extract hidden states (last layer)
        if hasattr(base_outputs, "hidden_states"):
            hidden_states = base_outputs.hidden_states[-1]
        elif hasattr(base_outputs, "last_hidden_state"):
            hidden_states = base_outputs.last_hidden_state
        else:
            # Fallback for different model types
            hidden_states = base_outputs[0] if isinstance(base_outputs, tuple) else base_outputs

        # Apply thought mixing head
        if self.config.enable_quiet_star:
            logits = self.thought_head(
                hidden_states=hidden_states,
                input_ids=input_ids,
                special_token_ids=self.special_token_ids,
            )
        else:
            # Use base model logits
            if hasattr(base_outputs, "logits"):
                logits = base_outputs.logits
            else:
                # Apply original LM head
                lm_head = self._find_lm_head()
                if lm_head is not None:
                    logits = lm_head(hidden_states)
                else:
                    raise ValueError("Cannot generate logits without LM head")

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "base_outputs": base_outputs,
        }

    def generate_with_thoughts(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        thought_probability: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Generate text with controlled thought insertion.

        Args:
            input_ids: [batch_size, seq_len] starting tokens
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            thought_probability: Override config thought_ratio

        Returns:
            Dictionary with generated tokens and thought information
        """
        input_ids.size(0)

        thought_prob = thought_probability if thought_probability is not None else self.config.thought_ratio
        sot_id = self.special_token_ids.get(self.config.start_of_thought_token, -1)
        eot_id = self.special_token_ids.get(self.config.end_of_thought_token, -1)
        self.special_token_ids.get(self.config.no_thought_token, -1)

        # Track generation state
        current_ids = input_ids.clone()
        thought_segments = []

        for step in range(max_length - input_ids.size(1)):
            # Forward pass
            outputs = self.forward(current_ids)
            next_token_logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Sample next tokens
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Decide whether to insert thoughts (during training)
            if self.training and torch.rand(1).item() < thought_prob and sot_id >= 0:
                # Insert start of thought
                current_ids = torch.cat([current_ids, torch.full_like(next_tokens, sot_id)], dim=1)

                # Generate thought content (limited length)
                thought_start = current_ids.size(1)
                for thought_step in range(self.config.max_thought_tokens):
                    thought_outputs = self.forward(current_ids)
                    thought_logits = thought_outputs["logits"][:, -1, :]

                    if temperature != 1.0:
                        thought_logits = thought_logits / self.config.thought_temperature

                    if do_sample:
                        thought_probs = F.softmax(thought_logits, dim=-1)
                        thought_token = torch.multinomial(thought_probs, num_samples=1)
                    else:
                        thought_token = torch.argmax(thought_logits, dim=-1, keepdim=True)

                    current_ids = torch.cat([current_ids, thought_token], dim=1)

                    # Stop if we generate end of thought or hit max length
                    if thought_token.item() == eot_id or current_ids.size(1) >= max_length:
                        break

                # Add end of thought if not already present
                if current_ids[:, -1].item() != eot_id and eot_id >= 0:
                    current_ids = torch.cat([current_ids, torch.full_like(next_tokens, eot_id)], dim=1)

                thought_end = current_ids.size(1)
                thought_segments.append(
                    {
                        "start": thought_start - 1,  # Include SoT token
                        "end": thought_end,
                        "tokens": current_ids[0, thought_start - 1 : thought_end].tolist(),
                    }
                )

            # Add the original next token
            current_ids = torch.cat([current_ids, next_tokens], dim=1)

            # Check for early termination
            if current_ids.size(1) >= max_length:
                break

        return {
            "generated_ids": current_ids,
            "thought_segments": thought_segments,
            "original_length": input_ids.size(1),
        }

    def update_special_tokens(self, special_token_ids: dict[str, int]) -> None:
        """Update special token IDs."""
        self.special_token_ids.update(special_token_ids)
        self.config.update_token_ids(None)  # Update config if needed
