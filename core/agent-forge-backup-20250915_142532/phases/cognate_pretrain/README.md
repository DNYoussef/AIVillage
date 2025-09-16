# Cognate Pre-training Phase

**Dedicated pipeline for creating exactly 3x 25M parameter Cognate models that feed into EvoMerge.**

## Overview

This package consolidates all Cognate model creation functionality that was previously scattered across multiple files. It provides a clean, organized pipeline for Phase 1 of the Agent Forge system.

**Pipeline Flow**: 3x Cognate Models (25M each) → EvoMerge → Quiet-STaR → BitNet → etc.

## Key Features

- **Exact Parameter Targeting**: 25,069,534 parameters per model
- **ACT Halting**: Adaptive computation with train-many (8 steps) / infer-few (2 steps)
- **Titans-style LTM**: Long-term memory with surprise×novelty gating
- **Memory Cross-Attention**: Integrated memory-augmented generation
- **3 Model Variants**: Different focuses (reasoning, memory, computation)

## Architecture

```
cognate-pretrain/
├── __init__.py                 # Package exports
├── model_factory.py           # Main entry point - creates 3 models
├── cognate_creator.py         # Core model creation logic
├── pretrain_pipeline.py       # Optional pre-training pipeline
├── phase_integration.py       # Agent Forge phase integration
├── models/                    # Output directory for created models
└── README.md                  # This file
```

## Usage

### Simple Model Creation

```python
from core.agent_forge.phases.cognate_pretrain import create_three_cognate_models

# Create 3 Cognate models ready for EvoMerge
models = create_three_cognate_models()
print(f"Created {len(models)} models with {sum(m['parameter_count'] for m in models):,} total parameters")
```

### With Pre-training

```python
from core.agent_forge.phases.cognate_pretrain import create_three_cognate_models, run_pretraining_pipeline

# Create models
models = create_three_cognate_models()

# Optional pre-training
pretrained_models = run_pretraining_pipeline(models, steps=1000)
```

### Agent Forge Integration

The phase integration automatically handles the full pipeline:

```python
from core.agent_forge.phases.cognate_pretrain.phase_integration import CognatePhase

phase = CognatePhase()
result = phase.execute()  # Creates 3 models and hands off to EvoMerge
```

## Model Specifications

### Architecture (All 3 Models)
- **Parameters**: ~25,069,534 each
- **d_model**: 216 (hidden dimension)
- **n_layers**: 11 (transformer layers)
- **n_heads**: 4 (attention heads, 54 dim each)
- **ffn_mult**: 4 (FFN expansion to 864)
- **vocab_size**: 32,000
- **max_seq_len**: 2,048

### Model Variants

1. **Cognate Foundation 1** - Reasoning Focus
   - ACT threshold: 0.95
   - Memory capacity: 4,096
   - Surprise weight: 0.7, Novelty: 0.3

2. **Cognate Foundation 2** - Memory Integration Focus
   - ACT threshold: 0.90
   - Memory capacity: 8,192
   - Surprise weight: 0.5, Novelty: 0.5

3. **Cognate Foundation 3** - Adaptive Computation Focus
   - ACT threshold: 0.99
   - Memory capacity: 2,048
   - Surprise weight: 0.3, Novelty: 0.7

## Output Structure

```
models/
├── cognate_foundation_1/
│   ├── pytorch_model.bin       # Model weights
│   └── metadata.json          # Model configuration & stats
├── cognate_foundation_2/
│   ├── pytorch_model.bin
│   └── metadata.json
├── cognate_foundation_3/
│   ├── pytorch_model.bin
│   └── metadata.json
├── cognate_models_summary.json # Overall summary
└── pretraining_summary.json   # Pre-training results (if used)
```

## Validation

Models are automatically validated for:
- Parameter count accuracy (within 10% of 25M target)
- EvoMerge compatibility
- Required features (ACT, LTM, memory cross-attention)

## Replacing Scattered Files

This package **replaces and consolidates**:
- `cognate.py` - Main phase integration
- `optimal_25m_training.py` - 25M training logic
- `train_and_save_25m_models.py` - Model saving logic
- `cognate_evomerge*.py` - EvoMerge preparation
- Various training files with "cognate" or "25m" in the name

All functionality is now centralized in this single, organized package.

## Next Phase

The 3 created models automatically integrate with:
- **EvoMerge (Phase 2)**: Evolutionary model merging and selection
- **Agent Forge Pipeline**: Continues through remaining 6 phases
- **Production Deployment**: Ready for serving infrastructure
