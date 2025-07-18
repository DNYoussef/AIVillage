# Geometry-Aware Training Components

This directory contains the core geometric analysis tools used throughout the Agent Forge training pipeline for monitoring neural network representational geometry and detecting phase transitions.

## Overview

The geometry module provides real-time analysis of neural network hidden states using intrinsic dimensionality estimation, principal component analysis, and entropy calculations. These geometric measurements are crucial for detecting grokking events, phase transitions, and optimal training dynamics.

## Core Components

### `snapshot.py` - Real-time Geometric Analysis
**Purpose:** Fast geometric measurement utilities for sensing representational geometry during training.

**Key Features:**
- **Nonlinear Intrinsic Dimensionality:** Uses Two-NN estimator for true geometric complexity
- **Linear Effective Dimension:** PCA-based analysis for linear compression characteristics
- **Entropy Measurement:** Token-wise softmax entropy for information-theoretic analysis
- **Efficient Implementation:** Optimized for mini-batch frequency measurements

### `id_twonn.py` - Two-NN Intrinsic Dimensionality Estimator
**Purpose:** Implementation of the Facco et al. Two-NN estimator for nonlinear intrinsic dimensionality.

**Key Features:**
- **Theoretical Foundation:** Based on rigorous mathematical framework from Facco et al.
- **Robust Estimation:** Handles high-dimensional data with nonlinear manifold structure
- **Fast Computation:** Efficient k-nearest neighbor distance ratio calculations
- **Fallback Implementation:** Local implementation when external twonn library unavailable

## Mathematical Foundation

### Two-NN Intrinsic Dimensionality

The Two-NN estimator computes intrinsic dimensionality using:

```
μᵢ = r₁(i) / r₂(i)
ID_nl = <log μ> / <log(1 - μ)>
```

Where:
- `r₁(i)` = distance to 1st nearest neighbor of point i
- `r₂(i)` = distance to 2nd nearest neighbor of point i
- `<·>` denotes average over all points

**Key Properties:**
- **Scale Invariant:** Robust to data scaling and normalization
- **Manifold Aware:** Detects true geometric complexity beyond linear approximations
- **Local Estimation:** Uses local neighborhood structure for global dimensionality

### Linear Effective Dimension

PCA-based linear dimensionality using:

```
ID_lin = min{k : Σⱼ₌₁ᵏ σⱼ² / Σⱼ₌₁ᴰ σⱼ² ≥ 0.99}
```

Where `σⱼ` are the singular values from PCA decomposition.

**Purpose:**
- Measures linear compression capability
- Compares to nonlinear complexity via ratio metric
- Indicates when linear methods are sufficient

### Entropy Measurement

Token-wise softmax entropy:

```
H = -Σᵢ Σⱼ p(xᵢⱼ) log p(xᵢⱼ)
```

Where `p(xᵢⱼ)` is the softmax probability for token i, dimension j.

**Interpretation:**
- High entropy: Uniform, uncertain representations
- Low entropy: Concentrated, confident representations
- Critical transitions: Entropy changes indicate phase shifts

## Core Functions

### `snapshot(hidden: Tensor, pca_q: int = 128) -> GeomState`

**Purpose:** Comprehensive geometric analysis of hidden states.

**Parameters:**
- `hidden`: (B, L, D) tensor of hidden states from model forward pass
- `pca_q`: Number of eigenvectors for PCA computation (default: 128)

**Returns:** `GeomState` dictionary containing:
- `ID_nl`: Nonlinear intrinsic dimensionality (float)
- `ID_lin`: Linear effective dimension (int)
- `ratio`: Compression ratio ID_nl / ID_lin (float)
- `entropy`: Token-wise entropy (float)

**Computational Complexity:**
- Two-NN: O(N² D) for distance computation, O(N log N) for k-selection
- PCA: O(N D q) using low-rank approximation
- Entropy: O(N D) for softmax and logarithm

**Usage Example:**
```python
import torch
from agent_forge.geometry.snapshot import snapshot

# Forward pass through model
outputs = model(input_ids, output_hidden_states=True)
hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

# Analyze geometry
geom = snapshot(hidden_states)

print(f"Nonlinear ID: {geom['ID_nl']:.2f}")
print(f"Linear ID: {geom['ID_lin']}")
print(f"Compression ratio: {geom['ratio']:.2f}")
print(f"Entropy: {geom['entropy']:.2f}")
```

### `twonn(x: torch.Tensor, k1: int = 2, k2: int = 3) -> float`

**Purpose:** Direct Two-NN intrinsic dimensionality estimation.

**Parameters:**
- `x`: (N, D) tensor of data points
- `k1`: First neighbor rank (default: 2, excludes self)
- `k2`: Second neighbor rank (default: 3)

**Returns:** Estimated intrinsic dimensionality (float)

**Implementation Details:**
- Uses `torch.cdist` for efficient pairwise distance computation
- `kthvalue` for exact k-nearest neighbor selection
- Numerical stability via epsilon clamping and minimum bounds

## Integration with Training Pipeline

### Phase 2: Core Training Loop
```python
from agent_forge.geometry.snapshot import snapshot

def run_level(model, dataset, config, state):
    for task in dataset:
        logits, H = model(task.prompt, return_h=True)
        G = snapshot(H)  # Real-time geometric analysis

        # Extract geometric features for geo2z policy
        z_geo = geo2z([G["ID_nl"], G["ratio"], G["entropy"]])

        # Check grokking conditions
        if slow > config.tau and state.get("id_drop", 0) > config.delta:
            state["level_grok"] = True
            break
```

### Phase 3: Self-Modeling Gate
```python
from agent_forge.geometry.snapshot import snapshot

def self_model_cycle(model, tokenizer, tasks, opt, thresholds, state):
    for step in range(max_iter):
        # Training step...

        # Geometric monitoring
        G = snapshot(H)
        drop = max(0, state["G"]["ID_nl"] - G["ID_nl"])

        # Internal grokking detection
        if slow > τ and drop > δ and abs(chaos - 0.5) < ε:
            state["self_grok"] = True
            break
```

## Geometric Interpretation

### Grokking Signatures

**Classic Grokking Pattern:**
1. **High Initial ID_nl:** Complex, unstructured representations
2. **Gradual Compression:** Steady decrease in intrinsic dimensionality
3. **Sudden Drop:** Sharp ID_nl reduction indicating grokking
4. **Stable Low ID:** Compressed, structured final representations

**Ratio Analysis:**
- `ratio > 1`: Nonlinear structure dominates (early training)
- `ratio ≈ 1`: Linear and nonlinear dimensions aligned (transition)
- `ratio < 1`: Linear structure sufficient (post-grokking)

### Training Dynamics

**Entropy Patterns:**
- **High Entropy → Low Entropy:** Learning progression
- **Entropy Oscillations:** Instability or exploration phases
- **Critical Entropy (~0.5):** Edge-of-chaos optimal learning

**Dimensionality Trends:**
- **ID_nl Decrease:** Compression and generalization
- **ID_lin Stability:** Consistent linear structure
- **Ratio Convergence:** Geometric optimization completion

## Optimization and Performance

### Computational Efficiency

**Memory Optimization:**
- Moves tensors to CPU for SVD computation when needed
- Uses low-rank PCA approximation (q=128) instead of full decomposition
- Reshapes to 2D for efficient distance computation

**Speed Optimizations:**
- `@torch.inference_mode()` for faster computation
- Batched distance computation via `torch.cdist`
- Early termination in PCA energy calculation

### Hyperparameter Tuning

**PCA Rank (pca_q):**
- Default: 128 (sufficient for dimensions 1024-4096)
- Increase for very high-dimensional models (>8192)
- Decrease for faster computation if accuracy permits

**Two-NN Parameters:**
- Default k1=2, k2=3 work well for most cases
- Increase for noisy data or small datasets
- Theoretical optimum depends on data manifold properties

## Research Connections

### Theoretical Foundations
- **Facco et al. (2017):** "Estimating the intrinsic dimension of datasets by a minimal neighborhood information"
- **Grokking Literature:** Connections to phase transitions in neural networks
- **Geometric Deep Learning:** Manifold structure in neural representations

### Applications in AI
- **Representation Learning:** Understanding learned feature geometry
- **Training Dynamics:** Detecting optimization phase transitions
- **Model Compression:** Identifying redundant representational dimensions
- **Continual Learning:** Monitoring catastrophic forgetting via geometry

## Usage Patterns

### Basic Geometric Monitoring
```python
# Simple monitoring during training
for epoch in training_loop:
    hidden = model.get_hidden_states(batch)
    geom = snapshot(hidden)

    # Log geometric properties
    logger.info(f"Epoch {epoch}: ID_nl={geom['ID_nl']:.2f}, ratio={geom['ratio']:.2f}")
```

### Grokking Detection
```python
# Detect grokking events
previous_id = float('inf')
grok_threshold = 0.5

for step in training_steps:
    geom = snapshot(hidden_states)
    id_drop = previous_id - geom['ID_nl']

    if id_drop > grok_threshold:
        print(f"Grokking detected at step {step}!")

    previous_id = geom['ID_nl']
```

### Advanced Analysis
```python
# Comprehensive geometric analysis
geometric_history = []

for batch in dataset:
    geom = snapshot(model_forward(batch))
    geometric_history.append(geom)

# Analyze trends
id_trend = [g['ID_nl'] for g in geometric_history]
ratio_trend = [g['ratio'] for g in geometric_history]

print(f"ID compression: {id_trend[0]:.2f} → {id_trend[-1]:.2f}")
print(f"Ratio evolution: {ratio_trend[0]:.2f} → {ratio_trend[-1]:.2f}")
```

## Future Extensions

### Planned Enhancements
- **Multi-Scale Analysis:** Geometric properties at different layer depths
- **Temporal Dynamics:** Time-series analysis of geometric evolution
- **Comparative Geometry:** Cross-model geometric comparisons
- **Adaptive Thresholds:** Dynamic grokking detection criteria

### Research Directions
- **Geometric Priors:** Using geometric constraints in training
- **Manifold Regularization:** Explicit geometric structure promotion
- **Multi-Agent Geometry:** Geometric analysis across agent ensembles
- **Real-time Adaptation:** Geometric feedback for hyperparameter tuning

## Dependencies

### Core Requirements
- `torch`: Core tensor operations and neural network functionality
- `twonn` (optional): External Two-NN library for enhanced performance

### Integration Points
- `agent_forge.phase2`: Grokking detection in core training
- `agent_forge.phase3`: Internal grokking for self-modeling
- `agent_forge.svf`: Geometric-guided sparse vector operations
- `agent_forge.meta`: Geometric policy learning (geo2z)

## Troubleshooting

### Common Issues

**Memory Errors:**
- Reduce `pca_q` parameter for large models
- Batch process large sequences
- Move computation to CPU if GPU memory limited

**Numerical Instability:**
- Check for NaN values in hidden states
- Ensure proper model initialization
- Verify input data normalization

**Performance Issues:**
- Use smaller batches for frequent monitoring
- Cache geometric computations when possible
- Profile Two-NN computation for bottlenecks

### Debug Configuration
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed geometric logging
logger = logging.getLogger("AF-Geometry")
logger.debug("Geometric analysis configuration loaded")
```

This provides insights into computation time, memory usage, and numerical stability during geometric analysis.
