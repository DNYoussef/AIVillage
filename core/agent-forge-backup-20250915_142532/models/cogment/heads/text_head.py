"""Text input heads for tokenization and text processing."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class TextHead(nn.Module):
    """Base text head for converting text to model representations."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Convert tokenized text to d_model representations."""
        # Base implementation - should be overridden by subclasses
        # For now, return zero tensor with correct shape
        B, seq_len = input_ids.shape
        return torch.zeros(B, seq_len, self.d_model)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CogmentTextHead(TextHead):
    """
    Cogment-specific text head with efficient token embedding.

    Optimized for parameter budget compliance:
    - Reuses HRRM tokenizer patterns
    - Lightweight embedding layer
    - Optional positional encoding
    - Target: <500K parameters
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 16000,  # Reduced from 32000 for efficiency
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        tie_embeddings: bool = True,
    ):
        super().__init__(d_model, vocab_size)

        self.max_seq_len = max_seq_len
        self.use_positional_encoding = use_positional_encoding
        self.tie_embeddings = tie_embeddings

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        if use_positional_encoding:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            self.register_buffer("position_ids", torch.arange(max_seq_len).expand((1, -1)))

        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Special token mappings (compatible with HRRM tokenizer)
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "<PLAN>": 4,
            "<SUBGOAL>": 5,
            "<ACTION>": 6,
            "<CHECK>": 7,
            "<ENDPLAN>": 8,
            "<SoT>": 9,  # Start of Thought
            "<EoT>": 10,  # End of Thought
        }

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with appropriate scaling."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.use_positional_encoding:
            nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert token IDs to embeddings.

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len] (optional)
            position_ids: Position IDs [B, seq_len] (optional)

        Returns:
            embeddings: Token embeddings [B, seq_len, d_model]
        """
        B, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # [B, seq_len, d_model]

        # Add positional embeddings
        if self.use_positional_encoding:
            if position_ids is None:
                position_ids = self.position_ids[:, :seq_len].expand(B, -1)
            pos_embeds = self.pos_embedding(position_ids)  # [B, seq_len, d_model]
            embeddings = token_embeds + pos_embeds
        else:
            embeddings = token_embeds

        # Apply dropout and normalization
        embeddings = self.dropout(embeddings)
        embeddings = self.norm(embeddings)

        # Apply attention mask if provided
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)

        return embeddings

    def get_special_token_id(self, token: str) -> int:
        """Get ID for special token."""
        return self.special_tokens.get(token, self.special_tokens["<unk>"])

    def create_attention_mask(self, input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """Create attention mask from input IDs."""
        return (input_ids != pad_token_id).long()


class SharedEmbeddingTextHead(TextHead):
    """
    Text head that shares embeddings with vocabulary heads for parameter efficiency.

    This approach ties input embeddings with output vocabulary heads to save parameters,
    following the pattern used in many language models.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        shared_embedding_weight: torch.Tensor,  # From vocabulary heads
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        scale_embeddings: bool = True,
    ):
        super().__init__(d_model, vocab_size)

        self.max_seq_len = max_seq_len
        self.scale_embeddings = scale_embeddings

        # Shared embedding (no additional parameters!)
        self.token_embedding = nn.Embedding.from_pretrained(shared_embedding_weight, freeze=False)

        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.register_buffer("position_ids", torch.arange(max_seq_len).expand((1, -1)))

        # Processing layers
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Embedding scaling factor
        if scale_embeddings:
            self.embed_scale = d_model**0.5
        else:
            self.embed_scale = 1.0

        self._init_weights()

    def _init_weights(self):
        """Initialize only non-shared parameters."""
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with shared embeddings."""
        B, seq_len = input_ids.shape

        # Token embeddings (shared with output heads)
        token_embeds = self.token_embedding(input_ids) * self.embed_scale

        # Positional embeddings
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len].expand(B, -1)
        pos_embeds = self.pos_embedding(position_ids)

        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        embeddings = self.norm(embeddings)

        # Apply attention mask
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)

        return embeddings

    def count_parameters(self) -> int:
        """Count parameters excluding shared embedding."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        shared_params = self.token_embedding.weight.numel()
        return total - shared_params  # Don't double-count shared parameters


class SimpleTokenizer:
    """
    Simple tokenizer for Cogment, compatible with HRRM patterns.

    Provides basic tokenization functionality when full tokenizers library
    is not available or for lightweight deployment.
    """

    def __init__(self, vocab_path: Path | None = None, vocab_size: int = 16000):
        self.vocab_size = vocab_size

        # Special tokens (must match TextHead)
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "<PLAN>": 4,
            "<SUBGOAL>": 5,
            "<ACTION>": 6,
            "<CHECK>": 7,
            "<ENDPLAN>": 8,
            "<SoT>": 9,
            "<EoT>": 10,
        }

        # Create vocabulary
        if vocab_path and vocab_path.exists():
            self.vocab = self._load_vocab(vocab_path)
        else:
            self.vocab = self._create_default_vocab()

        # Reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Token IDs
        self.pad_token_id = self.special_tokens["<pad>"]
        self.unk_token_id = self.special_tokens["<unk>"]
        self.bos_token_id = self.special_tokens["<s>"]
        self.eos_token_id = self.special_tokens["</s>"]

    def _load_vocab(self, vocab_path: Path) -> dict[str, int]:
        """Load vocabulary from JSON file."""
        with open(vocab_path) as f:
            data = json.load(f)
        return data.get("vocab", {})

    def _create_default_vocab(self) -> dict[str, int]:
        """Create default vocabulary with special tokens and common subwords."""
        vocab = self.special_tokens.copy()

        # Add common ASCII characters
        for i in range(32, 127):  # Printable ASCII
            char = chr(i)
            if char not in vocab:
                vocab[char] = len(vocab)

        # Add common subwords/byte-level tokens
        common_subwords = [
            "##a",
            "##e",
            "##i",
            "##o",
            "##u",
            "##n",
            "##r",
            "##s",
            "##t",
            "##l",
            "ing",
            "ed",
            "er",
            "ly",
            "tion",
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
        ]

        for subword in common_subwords:
            if subword not in vocab and len(vocab) < self.vocab_size:
                vocab[subword] = len(vocab)

        # Fill remaining slots with generated tokens
        while len(vocab) < self.vocab_size:
            token = f"<token_{len(vocab)}>"
            vocab[token] = len(vocab)

        return vocab

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """
        Encode text to token IDs.

        Simple character-level tokenization with subword lookup.
        """
        tokens = []

        # Add BOS token
        tokens.append(self.bos_token_id)

        # Simple tokenization
        i = 0
        while i < len(text):
            # Try longer subwords first
            found = False
            for length in range(min(10, len(text) - i), 0, -1):
                substr = text[i : i + length]
                if substr in self.vocab:
                    tokens.append(self.vocab[substr])
                    i += length
                    found = True
                    break

            if not found:
                # Fall back to character level
                char = text[i]
                tokens.append(self.vocab.get(char, self.unk_token_id))
                i += 1

        # Add EOS token
        tokens.append(self.eos_token_id)

        # Truncate if necessary
        if max_length and len(tokens) > max_length:
            tokens = tokens[: max_length - 1] + [self.eos_token_id]

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            token = self.id_to_token.get(token_id, "<unk>")
            if token.startswith("##"):
                tokens.append(token[2:])  # Remove subword prefix
            else:
                tokens.append(token)

        return "".join(tokens)

    def batch_encode(
        self,
        texts: list[str],
        max_length: int | None = None,
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Batch encode texts with padding."""
        encoded = [self.encode(text, max_length) for text in texts]

        if padding:
            max_len = max(len(seq) for seq in encoded)
            if max_length:
                max_len = min(max_len, max_length)

            # Pad sequences
            padded = []
            masks = []
            for seq in encoded:
                if len(seq) > max_len:
                    seq = seq[:max_len]

                mask = [1] * len(seq) + [0] * (max_len - len(seq))
                seq = seq + [self.pad_token_id] * (max_len - len(seq))

                padded.append(seq)
                masks.append(mask)

            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(padded, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long),
                }
            else:
                return {"input_ids": padded, "attention_mask": masks}

        return {"input_ids": encoded}


def create_text_head(head_type: str, d_model: int, vocab_size: int, **kwargs) -> TextHead:
    """
    Factory function for creating text heads.

    Args:
        head_type: Type of head ("standard", "shared")
        d_model: Model dimension
        vocab_size: Vocabulary size
        **kwargs: Additional arguments

    Returns:
        TextHead: Configured text head
    """
    if head_type == "standard":
        return CogmentTextHead(d_model, vocab_size, **kwargs)
    elif head_type == "shared":
        return SharedEmbeddingTextHead(d_model, vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown text head type: {head_type}")


if __name__ == "__main__":
    # Test text head
    d_model = 320
    vocab_size = 16000

    text_head = CogmentTextHead(d_model, vocab_size)
    print(f"Text Head parameters: {text_head.count_parameters():,}")

    # Test tokenizer
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)

    test_texts = [
        "Hello world!",
        "<PLAN> Solve the problem <SUBGOAL> Step 1 <ENDPLAN>",
        "<SoT> Let me think about this carefully <EoT>",
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: {text}")
        print(f"Encoded: {encoded[:10]}...")  # First 10 tokens
        print(f"Decoded: {decoded}")
        print()

    # Test batch encoding
    batch_result = tokenizer.batch_encode(test_texts, max_length=32, padding=True)
    print(f"Batch input_ids shape: {batch_result['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch_result['attention_mask'].shape}")

    # Test with text head
    embeddings = text_head(batch_result["input_ids"], batch_result["attention_mask"])
    print(f"Text embeddings shape: {embeddings.shape}")
