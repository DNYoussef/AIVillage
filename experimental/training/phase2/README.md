# Phase 2: Core Training Loop

This directory contains the implementation of Phase 2 of the Agent Forge training pipeline, focusing on the core training loop with geometric grokking detection and curriculum advancement.

## Overview

Phase 2 implements the foundational training methodology that uses geometric analysis to detect grokking events and advance through curriculum levels. This phase bridges the gap between basic model training and advanced self-modeling capabilities.

## Components

### `train_level.py`
**Purpose:** Core training function that runs a single curriculum level with geometric monitoring and grokking detection.

**Key Features:**
- **Geometric Monitoring:** Uses `snapshot()` from the geometry module to capture intrinsic dimensionality and topological properties
- **SVF Integration:** Applies Sparse Vector Format operations to modify model weights based on geometric features
- **Grokking Detection:** Monitors three key conditions for curriculum advancement:
  - `slow > tau`: Optimizer slow power exceeds threshold (gradient dynamics)
  - `id_drop > delta`: Intrinsic dimensionality drop indicates compression
  - `abs(task.score - 0.5) < eps`: Task performance near critical boundary

**Function Signature:**
```python
def run_level(model: torch.nn.Module, dataset: Sequence, config, state: dict) -> None
```

**Parameters:**
- `model`: PyTorch model being trained
- `dataset`: Sequence of training tasks for this curriculum level
- `config`: Configuration object with hyperparameters (lr, tau, delta, eps)
- `state`: Training state dictionary containing geo2z policy and tracking variables

**Training Process:**
1. Initialize Adam optimizer with specified learning rate
2. Create EdgePID controller for adaptive hyperparameter management
3. For each task in the dataset:
   - Forward pass through model with hidden state capture
   - Geometric analysis using `snapshot(H)`
   - Extract geometric features: ID_nl (intrinsic dimensionality), ratio, entropy
   - Apply geo2z policy to map geometry to latent space modifications
   - Use SVF to update specific model layers based on geometric insights
   - Check grokking conditions and break if level is complete

### `pid.py`
**Purpose:** EdgePID controller for adaptive hyperparameter management during training.

**Functionality:**
- Implements PID control for edge-of-chaos training dynamics
- Monitors gradient variance and learning rate adaptation
- Provides stability during geometric transitions

## Configuration Parameters

### Training Thresholds
- **`tau`**: Slow power threshold for gradient dynamics detection
- **`delta`**: Minimum intrinsic dimensionality drop required for advancement
- **`eps`**: Task score tolerance around critical boundary (0.5)

### Geometric Features
- **`ID_nl`**: Non-linear intrinsic dimensionality
- **`ratio`**: Geometric ratio metric
- **`entropy`**: Information-theoretic measure of hidden state complexity

## Integration with Pipeline

Phase 2 connects to:
- **Phase 1**: Receives foundation models and initial curriculum
- **Phase 3**: Advances to self-modeling gate when grokking is detected
- **Geometry Module**: Uses `snapshot()` for real-time geometric analysis
- **SVF Operations**: Applies sparse vector modifications based on geometric state

## Usage Example

```python
from agent_forge.phase2.train_level import run_level
from agent_forge.phase2.pid import EdgePID

# Configure training parameters
config = TrainingConfig(
    lr=1e-4,
    tau=0.7,      # Slow power threshold
    delta=0.1,    # ID drop threshold
    eps=0.05      # Score boundary tolerance
)

# Initialize training state
state = {
    "geo2z": geo2z_policy,  # Geometry-to-latent mapping
    "level_grok": False,    # Grokking detection flag
    "id_drop": 0.0         # Tracked ID changes
}

# Run curriculum level
run_level(model, curriculum_dataset, config, state)

# Check if ready for Phase 3
if state["level_grok"]:
    print("Grokking detected - advancing to Phase 3")
```

## Mathematical Foundation

The geometric grokking detection is based on:

1. **Intrinsic Dimensionality Monitoring**: Tracking ID_nl as the model compresses representations
2. **Gradient Dynamics**: Using slow power to detect optimization phase transitions
3. **Critical Boundary Detection**: Monitoring task performance near the 0.5 decision boundary

The combination of these three signals indicates when the model has achieved sufficient compression and understanding to advance to the next curriculum level.

## Future Enhancements

- Integration with Quiet-STaR thought generation
- Dynamic curriculum difficulty adjustment
- Multi-objective optimization for competing training goals
- Enhanced geometric feature extraction beyond current triplet

## Dependencies

- `torch`: Core PyTorch functionality
- `agent_forge.geometry.snapshot`: Geometric analysis functions
- `agent_forge.optim.Adam`: Augmented Adam optimizer with slow power tracking
- `agent_forge.svf.svf_ops`: Sparse Vector Format operations
