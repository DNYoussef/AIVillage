#!/usr/bin/env python3
"""Build BPE tokenizer for HRRM models."""

import argparse
import json
import logging
from pathlib import Path

# Try to import tokenizers, fall back to simple stub
try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_tokenizer(vocab_size: int = 32000) -> dict:
    """Create a simple tokenizer stub when tokenizers library is not available."""
    logger.warning("tokenizers library not available, creating stub tokenizer")

    # Create basic vocab mapping
    vocab = {}

    # Special tokens
    special_tokens = [
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",  # Standard tokens
        "<PLAN>",
        "<SUBGOAL>",
        "<ACTION>",
        "<CHECK>",
        "<ENDPLAN>",  # Planner tokens
        "<SoT>",
        "<EoT>",  # Reasoner tokens
    ]

    for i, token in enumerate(special_tokens):
        vocab[token] = i

    # Fill remaining vocab with dummy tokens
    for i in range(len(special_tokens), vocab_size):
        vocab[f"<token_{i}>"] = i

    return {"vocab": vocab, "type": "stub", "vocab_size": vocab_size, "special_tokens": special_tokens}


def create_bpe_tokenizer(vocab_size: int = 32000, sample_size: int = 2000000) -> Tokenizer:
    """Create BPE tokenizer using tokenizers library."""
    if not TOKENIZERS_AVAILABLE:
        raise ImportError("tokenizers library required for BPE tokenizer")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))  # nosec B106 - tokenizer special token

    # Set pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Set post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Define special tokens
    special_tokens = [
        "<pad>",
        "<unk>",
        "<s>",
        "</s>",  # Standard tokens
        "<PLAN>",
        "<SUBGOAL>",
        "<ACTION>",
        "<CHECK>",
        "<ENDPLAN>",  # Planner tokens
        "<SoT>",
        "<EoT>",  # Reasoner tokens
    ]

    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=2, show_progress=True
    )

    # For demo purposes, create synthetic training data
    # In production, would use real text data
    logger.info(f"Creating synthetic training data ({sample_size} samples)")

    training_data = []
    for i in range(sample_size):
        # Generate synthetic text with special tokens
        text = f"This is sample text {i}. "
        if i % 100 == 0:
            text += "<PLAN> Do something <SUBGOAL> Step 1 <ACTION> Execute <CHECK> Verify <ENDPLAN> "
        if i % 150 == 0:
            text += "<SoT> Let me think about this <EoT> "
        training_data.append(text)

    # Train tokenizer
    logger.info("Training BPE tokenizer...")
    tokenizer.train_from_iterator(training_data, trainer)

    return tokenizer


def save_tokenizer(tokenizer_data, output_path: Path):
    """Save tokenizer to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(tokenizer_data, dict):
        # Stub tokenizer
        with open(output_path, "w") as f:
            json.dump(tokenizer_data, f, indent=2)
    else:
        # Real tokenizer
        tokenizer_data.save(str(output_path))

    logger.info(f"Tokenizer saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build HRRM tokenizer")
    parser.add_argument("--out", required=True, help="Output tokenizer path")
    parser.add_argument("--vocab", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--sample", type=int, default=2000000, help="Training sample size")
    parser.add_argument("--force-stub", action="store_true", help="Force stub tokenizer creation")

    args = parser.parse_args()

    output_path = Path(args.out)

    try:
        if args.force_stub or not TOKENIZERS_AVAILABLE:
            tokenizer_data = create_simple_tokenizer(args.vocab)
        else:
            tokenizer_data = create_bpe_tokenizer(args.vocab, args.sample)

        save_tokenizer(tokenizer_data, output_path)

        # Print summary
        if isinstance(tokenizer_data, dict):
            logger.info(f"Created stub tokenizer with {tokenizer_data['vocab_size']} tokens")
            logger.info(f"Special tokens: {tokenizer_data['special_tokens']}")
        else:
            logger.info(f"Created BPE tokenizer with {tokenizer_data.get_vocab_size()} tokens")

    except Exception as e:
        logger.error(f"Failed to create tokenizer: {e}")
        # Fallback to stub
        logger.info("Falling back to stub tokenizer")
        tokenizer_data = create_simple_tokenizer(args.vocab)
        save_tokenizer(tokenizer_data, output_path)


if __name__ == "__main__":
    main()
