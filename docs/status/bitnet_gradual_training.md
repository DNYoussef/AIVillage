# BitNet 1.58-bit Gradual Training

## Overview

This implementation converts BitNet from "all-at-once" ternarization to a gradual, training-driven schedule using self-generated data. The approach improves training stability while preserving compression efficiency.

## Key Features

### 1. Gradual λ Schedule

Instead of immediate quantization, weights are interpolated between full precision and ternary:

```python
effective_weights = (1 - λ) * weight_fp + λ * quantize(weight_fp)
```

- **λ = 0**: Pure full precision (training start)
- **λ = 1**: Pure ternary quantization (training end)
- **0 < λ < 1**: Smooth interpolation (during warmup)

### 2. Training Schedule

- **Warmup Phase**: λ ramps from 0 → 1 over first 40% of training steps (configurable)
- **Stable Phase**: λ = 1.0 for remaining 60% of training (pure ternary)
- **Evaluation**: Always uses λ = 1.0 (pure ternary inference)

### 3. RMSNorm Stabilization

- Optional RMSNorm insertion after attention layers (`--rmsnorm_post_attn`)
- Improves training stability during quantization transition
- BF16/CUDA safe implementation

## Implementation Components

### Core Files

- `stage1_bitnet.py`: BitNetLinear layer with λ interpolation
- `train_bitnet.py`: Training script with gradual schedule
- `selfgen/generate.py`: Self-generated training data
- `run_bitnet158.ps1`: Complete pipeline wrapper

### Training Pipeline

1. **Data Generation**: Self-generate diverse instruction-following data
2. **Model Conversion**: Replace Linear layers with BitNetLinear
3. **Gradual Training**: Train with λ warmup schedule
4. **Compression Ready**: Final model ready for `bitnet.py::compress()`

## Usage

### Quick Start

```powershell
.\scripts\run_bitnet158.ps1 -ModelPath "D:\AIVillage\models\Qwen2.5-1.5B-Instruct"
```

### Manual Steps

```bash
# 1. Generate training data
python selfgen/generate.py \
    --model_path "D:\AIVillage\models\Qwen2.5-1.5B-Instruct" \
    --out "D:\AIVillage\artifacts\bitnet158\training_data.jsonl" \
    --num 500

# 2. Train with gradual λ schedule
python train_bitnet.py \
    --base_model "D:\AIVillage\models\Qwen2.5-1.5B-Instruct" \
    --dataset "D:\AIVillage\artifacts\bitnet158\training_data.jsonl" \
    --out_dir "D:\AIVillage\artifacts\bitnet158\trained_model" \
    --steps 1000 \
    --lambda_warmup_frac 0.4
```

## Configuration Options

### Training Parameters

- `--steps`: Total training steps (default: 1000)
- `--lambda_warmup_frac`: Fraction of steps for λ warmup (default: 0.4)
- `--rmsnorm_post_attn`: Add RMSNorm after attention (default: 1)
- `--bsz`: Per-device batch size (default: 2)
- `--grad_accum`: Gradient accumulation steps (default: 8)
- `--lr`: Learning rate (default: 5e-5)

### Data Generation

- `--num`: Number of samples to generate (default: 100)
- `--max_new_tokens`: Max tokens per response (default: 512)
- `--seed`: Random seed for reproducibility (default: 42)

## Quality Filters

### Data Generation Filters

- **Length**: 32-2048 characters
- **AI Disclaimers**: Removes responses with "as an AI", etc.
- **Repetition**: Basic deduplication via content hashing
- **Diversity**: Minimum word diversity in responses

### Training Stability

- **RMSNorm**: Optional post-attention normalization
- **Gradient Checkpointing**: Memory-efficient training
- **FP16**: Mixed precision for faster training
- **Cosine LR Schedule**: Smooth learning rate decay

## Benefits vs. All-at-Once

### Improved Stability

- Gradual transition reduces training instability
- RMSNorm provides additional numerical stability
- Smooth λ schedule prevents gradient spikes

### Better Convergence

- Model adapts progressively to quantization
- Final weights optimized for ternary representation
- Maintains full precision capabilities during early training

### Preserved Compression

- Final λ=1.0 provides same compression as immediate quantization
- Compatible with existing `bitnet.py::compress()` interface
- Maintains 4-ternaries-per-byte packing efficiency

## Environment Setup

The system defaults to offline-first operation:

```powershell
$env:AIV_ROOT = "D:\AIVillage"
$env:AIV_MODELS_DIR = "D:\AIVillage\models"
$env:AIV_ARTIFACTS_DIR = "D:\AIVillage\artifacts"
$env:WANDB_DIR = "D:\AIVillage\wandb"
$env:WANDB_MODE = "offline"
```

## Monitoring

### W&B Integration

- **Project**: `AIVillage-BitNet158`
- **Tags**: `stage:bitnet158`, `source:selfgen`, `model:<name>`
- **Metrics**: Training loss, λ schedule, gradient norms
- **Offline Mode**: Logs saved locally for later sync

### Training Manifest

Each trained model includes a `training_manifest.json`:

```json
{
  "base_model": "/path/to/base/model",
  "training_steps": 1000,
  "lambda_warmup_frac": 0.4,
  "final_lambda": 1.0,
  "compression_ready": true,
  "model_type": "bitnet158"
}
```

## Testing

Run the test suite to verify implementation:

```bash
python -m pytest tests/test_bitnet_gradual.py -v
```

Tests cover:
- λ interpolation correctness
- Gradual schedule implementation
- RMSNorm functionality
- Data generation quality
- Training pipeline integration

## Next Steps

After training completion:

1. **Compression**: Use `bitnet.py::compress()` on trained weights
2. **Evaluation**: Benchmark against original model
3. **Deployment**: Deploy compressed model for inference
4. **Iteration**: Adjust hyperparameters based on results
