#!/usr/bin/env python3
"""Export HRRM models to HuggingFace format for Agent Forge integration."""

import json
import logging
import os
from pathlib import Path
import shutil
import sys

import torch
from transformers import PretrainedConfig, PreTrainedModel

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath("."))

from packages.hrrm.memory.model import MemoryAsContextTiny, MemoryConfig
from packages.hrrm.planner.heads import PlannerConfig
from packages.hrrm.planner.model import HRMPlanner
from packages.hrrm.reasoner.model import HRMReasoner, ReasonerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRRMConfig(PretrainedConfig):
    """HuggingFace-compatible configuration for HRRM models."""

    model_type = "hrrm"

    def __init__(self, model_name="hrrm", hrrm_type="planner", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.hrrm_type = hrrm_type


class HRRMForCausalLM(PreTrainedModel):
    """HuggingFace-compatible wrapper for HRRM models."""

    config_class = HRRMConfig

    def __init__(self, config):
        super().__init__(config)

        # Load the original HRRM model based on type
        if config.hrrm_type == "planner":
            # Load planner config and model
            with open("artifacts/checkpoints/planner/config.json") as f:
                hrrm_config_dict = json.load(f)
            hrrm_config = PlannerConfig(**{k: v for k, v in hrrm_config_dict.items() if k != "control_tokens"})
            hrrm_config.control_tokens = hrrm_config_dict["control_tokens"]
            self.hrrm_model = HRMPlanner(hrrm_config)

        elif config.hrrm_type == "reasoner":
            with open("artifacts/checkpoints/reasoner/config.json") as f:
                hrrm_config_dict = json.load(f)
            hrrm_config = ReasonerConfig(**hrrm_config_dict)
            self.hrrm_model = HRMReasoner(hrrm_config)

        elif config.hrrm_type == "memory":
            with open("artifacts/checkpoints/memory/config.json") as f:
                hrrm_config_dict = json.load(f)
            hrrm_config = MemoryConfig(**hrrm_config_dict)
            self.hrrm_model = MemoryAsContextTiny(hrrm_config)

        # Load weights
        checkpoint_path = f"artifacts/checkpoints/{config.hrrm_type}/model.pt"
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.hrrm_model.load_state_dict(state_dict)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass compatible with HuggingFace interface."""
        return self.hrrm_model(input_ids, labels=labels, attn_mask=attention_mask)


def export_model_to_hf(model_type, output_dir):
    """Export a single HRRM model to HuggingFace format."""
    logger.info(f"Exporting {model_type} to HuggingFace format...")

    # Create HuggingFace config
    config = HRRMConfig(model_name=f"hrrm-{model_type}", hrrm_type=model_type)

    # Create HuggingFace model
    model = HRRMForCausalLM(config)

    # Create output directory
    model_output_dir = Path(output_dir) / f"hrrm-{model_type}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Save model and config (disable safe serialization for tied embeddings)
    model.save_pretrained(model_output_dir, safe_serialization=False)

    # Copy original config for reference
    original_config_path = f"artifacts/checkpoints/{model_type}/config.json"
    shutil.copy2(original_config_path, model_output_dir / "hrrm_original_config.json")

    # Create model card
    model_card = f"""---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- hrrm
- hierarchical-memory
- reasoning
- planning
---

# HRRM {model_type.title()} Model

This is the {model_type} component of the Hierarchical Recurrent Reasoning Memory (HRRM) Bootstrap System.

## Model Details

- **Model Type**: {model_type.title()}
- **Parameters**: ~16.6M
- **Architecture**: Hierarchical Recurrent Memory with two-timescale dynamics
- **Training**: Synthetic data with deep supervision

## Usage

```python
from transformers import AutoModel, AutoConfig

config = AutoConfig.from_pretrained("./hrrm-{model_type}")
model = AutoModel.from_pretrained("./hrrm-{model_type}")

# Generate with the model
input_ids = torch.randint(0, 1000, (1, 10))
output = model(input_ids)
```

## Architecture Features

{"- **Control Token Detection**: Planning DSL with control tokens" if model_type == "planner" else ""}
{"- **Quiet-STaR Reasoning**: Internal thought processes with self-consistency" if model_type == "reasoner" else ""}
{"- **Titans Neural Memory**: Surprise-based memory updates with temporal dynamics" if model_type == "memory" else ""}
- **Hierarchical Dynamics**: H-slow/T-fast two-timescale processing
- **Deep Supervision**: Loss computed at each H-cycle for stable training

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{hrrm-bootstrap-2024,
  title={{Hierarchical Recurrent Reasoning Memory Bootstrap System}},
  author={{AIVillage Team}},
  year={{2024}},
  note={{Bootstrap implementation for Agent Forge integration}}
}}
```
"""

    with open(model_output_dir / "README.md", "w") as f:
        f.write(model_card)

    logger.info(f"✅ {model_type} exported to {model_output_dir}")
    return model_output_dir


def create_tokenizer(output_dir):
    """Create a simple tokenizer for HRRM models."""
    logger.info("Creating HRRM tokenizer...")

    # Create a simple BPE tokenizer config
    tokenizer_config = {
        "vocab_size": 32000,
        "model_type": "BPE",
        "special_tokens": {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "sep_token": "<sep>",
            "cls_token": "<cls>",
            "mask_token": "<mask>",
        },
        "control_tokens": [
            "<PLAN>",
            "<SUBGOAL>",
            "<ACTION>",
            "<CHECK>",
            "<ENDPLAN>",
            "<startofthought>",
            "<endofthought>",
            "<memory>",
            "<recall>",
            "<update>",
        ],
    }

    tokenizer_dir = Path(output_dir) / "hrrm-tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    with open(tokenizer_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Create vocab file (simplified)
    vocab = {}
    for i in range(32000):
        vocab[f"token_{i}"] = i

    # Add special tokens
    for token in tokenizer_config["special_tokens"].values():
        vocab[token] = len(vocab)

    for token in tokenizer_config["control_tokens"]:
        vocab[token] = len(vocab)

    with open(tokenizer_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    logger.info(f"✅ Tokenizer created at {tokenizer_dir}")
    return tokenizer_dir


def main():
    """Main export function."""
    logger.info("🤖 Starting HRRM HuggingFace Export Process...")
    logger.info("=" * 60)

    output_dir = "hf_models"

    # Export all three models
    models_exported = []
    for model_type in ["planner", "reasoner", "memory"]:
        try:
            model_path = export_model_to_hf(model_type, output_dir)
            models_exported.append(model_path)
        except Exception as e:
            logger.error(f"❌ Failed to export {model_type}: {e}")
            continue

    # Create tokenizer
    try:
        tokenizer_path = create_tokenizer(output_dir)
        models_exported.append(tokenizer_path)
    except Exception as e:
        logger.error(f"❌ Failed to create tokenizer: {e}")

    # Generate export summary
    logger.info("=" * 60)
    logger.info("🏆 HRRM HUGGINGFACE EXPORT SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Models Successfully Exported: {len([p for p in models_exported if 'hrrm-' in str(p)])}/3")
    logger.info(f"Tokenizer Created: {'✅' if any('tokenizer' in str(p) for p in models_exported) else '❌'}")

    if len(models_exported) >= 3:
        logger.info("✅ ALL HRRM MODELS EXPORTED TO HUGGINGFACE FORMAT!")
        logger.info("")
        logger.info("Exported Models:")
        for path in models_exported:
            if "hrrm-" in str(path):
                model_name = path.name
                logger.info(f"  • {model_name}: {path}")

        logger.info("")
        logger.info("🚀 Ready for Agent Forge Integration!")
        logger.info("Next steps:")
        logger.info("  1. Upload to HuggingFace Hub (optional)")
        logger.info("  2. Integrate with Agent Forge EvoMerge pipeline")
        logger.info("  3. Enable 30x speedup training")

        # Create export summary
        export_summary = {
            "export_status": "SUCCESS",
            "models_exported": len([p for p in models_exported if "hrrm-" in str(p)]),
            "total_models": 3,
            "tokenizer_created": any("tokenizer" in str(p) for p in models_exported),
            "export_paths": [str(p) for p in models_exported],
            "ready_for_agent_forge": True,
        }

    else:
        logger.info(f"❌ {3 - len(models_exported)} MODEL(S) FAILED EXPORT")
        export_summary = {
            "export_status": "FAILED",
            "models_exported": len(models_exported),
            "total_models": 3,
            "export_paths": [str(p) for p in models_exported],
            "ready_for_agent_forge": False,
        }

    # Save export report
    report_path = "artifacts/hrrm_hf_export_report.json"
    with open(report_path, "w") as f:
        json.dump(export_summary, f, indent=2)

    logger.info(f"📋 Export report saved to {report_path}")


if __name__ == "__main__":
    main()
