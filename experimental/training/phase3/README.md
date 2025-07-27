# Phase 3: Self-Modeling & Expert Vectors

This directory implements Phase 3 of the Agent Forge training pipeline, focusing on self-modeling cycles and internal grokking detection for advanced reasoning development.

## Overview

Phase 3 introduces sophisticated self-modeling capabilities where the model learns to predict its own hidden states and undergoes internal grokking cycles. This phase is crucial for developing metacognitive abilities and preparing for expert vector creation.

## Components

### `self_modeling_gate.py`
**Purpose:** Implements the self-modeling cycle that trains the model to predict its own internal representations while monitoring for internal grokking signatures.

**Key Features:**
- **Self-Prediction Training:** Model learns to predict its own hidden states using a secondary prediction network
- **Internal Grokking Detection:** Monitors geometric signatures that indicate internal understanding breakthrough
- **Geometry-Aware Prompting:** Incorporates current geometric state into training prompts
- **Dual Loss Function:** Combines masked language modeling with self-prediction objectives

## Core Function: `self_model_cycle()`

**Purpose:** Runs iterative self-modeling training until internal grok signature is detected or maximum iterations reached.

**Signature:**
```python
def self_model_cycle(model, tokenizer, tasks: Sequence[str], opt, thresholds, state):
```

**Parameters:**
- `model`: Main PyTorch model being trained
- `tokenizer`: Tokenizer for text processing
- `tasks`: Collection of training texts for self-modeling
- `opt`: Optimizer with `slow_power()` and `step(filter=True)` capabilities
- `thresholds`: Dictionary containing `{slow, id_drop, chaos, max_iter}`
- `state`: Training state with geometric tracking and prediction network

## Training Process

### 1. Dynamic Prompt Generation
Each training step creates geometry-aware prompts:
```
<geom id={ID_nl:.2f} t={temperature:.2f}/>{task_text}
```

This embeds current geometric state directly into the input, allowing the model to condition its responses on its internal geometric properties.

### 2. Dual Objective Training
- **L_mask**: Standard masked language modeling loss for next-token prediction
- **L_pred**: MSE loss between predicted and actual hidden states
- **Combined Loss**: `L_mask + 0.1 * L_pred`

### 3. Geometric Monitoring
Real-time tracking of three key metrics:
- **`slow`**: Optimizer slow power indicating gradient dynamics phase
- **`drop`**: Intrinsic dimensionality reduction from previous step
- **`chaos`**: Complexity measure relative to current rule set

### 4. Internal Grokking Detection
Advancement occurs when all three conditions are met:
- `slow > τ` (tau): Sufficient gradient dynamics activity
- `drop > δ` (delta): Meaningful dimensionality compression
- `abs(chaos - 0.5) < ε` (epsilon): Optimal complexity at critical boundary

## Configuration Parameters

### Thresholds Dictionary
```python
thresholds = {
    'slow': 0.7,        # Slow power threshold for gradient dynamics
    'id_drop': 0.1,     # Minimum ID drop for advancement
    'chaos': 0.05,      # Chaos tolerance around 0.5 critical point
    'max_iter': 8000    # Maximum training steps before timeout
}
```

### State Dictionary
```python
state = {
    'hidden_pred': prediction_network,    # Neural network for hidden state prediction
    'G': {'ID_nl': current_id},          # Current geometric measurements
    'complexity': complexity_function,    # Function mapping geometry to complexity
    'rule_id': current_rule_set,         # Active rule set identifier
    'self_grok': False                   # Internal grokking detection flag
}
```

## Mathematical Foundation

### Self-Prediction Objective
The model learns the mapping `H → H'` where:
- `H`: Current hidden states from forward pass
- `H'`: Predicted hidden states from separate prediction network

This creates a self-referential loop that enhances the model's understanding of its own internal representations.

### Geometric Signature Detection
Internal grokking is characterized by:

1. **Slow Power Rise**: `∇²L` variance indicating optimization phase transition
2. **Dimensionality Drop**: Compression in hidden representation space
3. **Critical Complexity**: Chaos measure converging to 0.5 (edge of chaos)

The simultaneous occurrence of these three phenomena indicates the model has achieved internal breakthrough in understanding.

## Integration with Pipeline

### Input from Phase 2
- Foundation model with basic curriculum training
- Established geometric monitoring capabilities
- Initial grokking detection from curriculum advancement

### Output to Phase 4
- Self-aware model capable of internal state prediction
- Enhanced geometric understanding
- Readiness for prompt baking and ADAS optimization

### Dependencies
- **Geometry Module**: For real-time `snapshot()` analysis
- **Optimizer Extensions**: Requires `slow_power()` and filtered stepping
- **Phase 2**: Builds upon geometric grokking detection framework

## Usage Example

```python
from agent_forge.phase3.self_modeling_gate import self_model_cycle
import torch.nn.functional as F

# Create prediction network for hidden states
hidden_pred = torch.nn.Sequential(
    torch.nn.Linear(hidden_dim, hidden_dim * 2),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim * 2, hidden_dim)
)

# Configure thresholds for internal grokking
thresholds = {
    'slow': 0.7,
    'id_drop': 0.1,
    'chaos': 0.05,
    'max_iter': 8000
}

# Initialize state tracking
state = {
    'hidden_pred': hidden_pred,
    'G': {'ID_nl': 2.5},  # Starting intrinsic dimensionality
    'complexity': lambda g, r: compute_complexity(g, r),
    'rule_id': 0,
    'self_grok': False
}

# Run self-modeling cycle
self_model_cycle(
    model=main_model,
    tokenizer=tokenizer,
    tasks=training_texts,
    opt=augmented_optimizer,
    thresholds=thresholds,
    state=state
)

# Check for advancement
if state['self_grok']:
    print("Internal grokking achieved - ready for Phase 4")
```

## Research Connections

This implementation draws from:
- **Grokking Research**: Detecting phase transitions in neural network training
- **Self-Modeling**: Meta-learning approaches for internal state prediction
- **Geometric Deep Learning**: Using intrinsic dimensionality for training dynamics
- **Edge of Chaos**: Optimal complexity theory for learning systems

## Future Enhancements

- **Expert Vector Extraction**: Convert successful self-modeling patterns to expert vectors
- **Multi-Scale Grokking**: Detect grokking at different temporal scales
- **Adaptive Complexity**: Dynamic adjustment of complexity targets
- **Thought Generation**: Integration with Quiet-STaR for explicit reasoning traces

## Debugging and Monitoring

The module includes comprehensive logging via the `AF-SelfGrokk` logger:
- Step-by-step geometric measurements
- Internal grokking detection events
- Training convergence diagnostics
- Prediction network performance metrics

Monitor the logs to understand training dynamics and tune thresholds for optimal performance.
