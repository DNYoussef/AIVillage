#!/usr/bin/env python3
"""Export HRRM models to HuggingFace compatible format."""

import argparse
import json
import logging
from pathlib import Path
import shutil
from typing import Any

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_hf_config(model_type: str, config: Any, param_count: int) -> dict[str, Any]:
    """Create HuggingFace model configuration."""

    base_config = {
        "architectures": [f"HRRM{model_type.title()}"],
        "model_type": f"hrrm_{model_type}",
        "vocab_size": config.vocab_size,
        "hidden_size": config.d_model,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_head,
        "intermediate_size": config.d_model * 4,
        "max_position_embeddings": config.max_seq_len,
        "rms_norm_eps": 1e-5,
        "rope_theta": config.rope_base,
        "tie_word_embeddings": config.tie_embeddings,
        "use_cache": True,
        "torch_dtype": "float32",
        "transformers_version": "4.21.0",
        "param_count": param_count,
    }

    # Model-specific configurations
    if model_type == "planner":
        base_config.update(
            {
                "control_tokens": config.control_tokens,
                "max_H": config.max_H,
                "inner_T": config.inner_T,
                "lambda_ctrl": config.lambda_ctrl,
            }
        )
    elif model_type == "reasoner":
        base_config.update(
            {
                "max_H": config.max_H,
                "inner_T": config.inner_T,
                "self_consistency_k": config.self_consistency_k,
                "start_thought_token": config.start_thought_token,
                "end_thought_token": config.end_thought_token,
            }
        )
    elif model_type == "memory":
        base_config.update(
            {
                "mem_dim": config.mem_dim,
                "mem_tokens": config.mem_tokens,
                "mem_slots": config.mem_slots,
                "titans_alpha": config.alpha,
                "titans_beta": config.beta,
                "titans_eta": config.eta,
                "titans_eta_decay": config.eta_decay,
            }
        )

    return base_config


def export_model(checkpoint_path: Path, output_dir: Path, model_type: str):
    """Export a single model to HuggingFace format."""

    logger.info(f"Exporting {model_type} model from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    # Count parameters
    param_count = sum(p.numel() for p in state_dict.values())

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(state_dict, output_dir / "pytorch_model.bin")

    # Create HF config
    hf_config = create_hf_config(model_type, config, param_count)
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # Create model card
    model_card = f"""---
language: en
license: mit
datasets:
- synthetic
tags:
- hrrm
- {model_type}
- transformers
- pytorch
---

# HRRM {model_type.title()} Model

This is a {model_type} model from the HRRM (Hierarchical Recurrent Reasoning Memory) family.

## Model Details

- **Model Type**: {model_type.title()}
- **Parameters**: {param_count:,}
- **Architecture**: {"HRM two-timescale loop" if model_type != "memory" else "Base transformer + Titans memory"}
- **Vocabulary Size**: {config.vocab_size:,}
- **Hidden Size**: {config.d_model}
- **Layers**: {config.n_layers}
- **Attention Heads**: {config.n_head}

## Features

"""

    if model_type == "planner":
        model_card += """- HRM two-timescale loop with deep supervision
- ControllerHead for DSL planning tokens: <PLAN>, <SUBGOAL>, <ACTION>, <CHECK>, <ENDPLAN>
- Approximate 1-step gradients for memory efficiency
"""
    elif model_type == "reasoner":
        model_card += """- HRM two-timescale loop with scratchpad supervision
- ScratchpadSupervisor for reasoning spans: <SoT>...<EoT>
- Self-consistency with k=5 reasoning chains
- Thought detection and gating mechanisms
"""
    elif model_type == "memory":
        model_card += """- Base transformer with Titans test-time learning memory
- Memory-as-Context (MAC) integration pattern
- Surprise-based memory updates with momentum and decay
- Neural memory with {config.mem_slots} slots
""".format(
            config=config
        )

    model_card += f"""
## Usage

```python
import torch
from transformers import AutoModel, AutoConfig

# Load model
config = AutoConfig.from_pretrained("path/to/model")
model = AutoModel.from_pretrained("path/to/model")

# Example inference
input_ids = torch.randint(0, {config.vocab_size}, (1, 128))
outputs = model(input_ids)
```

## Training

This model was trained using the HRRM bootstrap pipeline on synthetic data.
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(model_card)

    # Copy tokenizer if available
    tokenizer_path = Path("artifacts/tokenizer/hrrm_bpe_32k.json")
    if tokenizer_path.exists():
        shutil.copy(tokenizer_path, output_dir / "tokenizer.json")

        # Create tokenizer config
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }

        with open(output_dir / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)

    logger.info(f"Model exported to {output_dir}")
    logger.info(f"Parameters: {param_count:,}")


def main():
    parser = argparse.ArgumentParser(description="Export HRRM models to HuggingFace format")
    parser.add_argument("--src", required=True, help="Source checkpoint directory")
    parser.add_argument("--dst", required=True, help="Destination HF export directory")

    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    # Define model mappings
    models = {
        "planner": src_dir / "planner" / "latest.pt",
        "reasoner": src_dir / "reasoner" / "latest.pt",
        "memory": src_dir / "memory" / "latest.pt",
    }

    # Export each model
    for model_type, checkpoint_path in models.items():
        if checkpoint_path.exists():
            output_dir = dst_dir / model_type
            try:
                export_model(checkpoint_path, output_dir, model_type)
            except Exception as e:
                logger.error(f"Failed to export {model_type}: {e}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Export completed")
    logger.info(f"HuggingFace models available in: {dst_dir}")


if __name__ == "__main__":
    main()
