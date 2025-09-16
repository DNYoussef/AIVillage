---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- hrrm
- hierarchical-memory
- reasoning
- hrmplanner
---

# HRRM HRMPlanner Model

This is the hrmplanner component of the Hierarchical Recurrent Reasoning Memory (HRRM) Bootstrap System.

## Model Details

- **Model Type**: HRMPlanner
- **Parameters**: ~86,039,045
- **Architecture**: Hierarchical Recurrent Memory with two-timescale dynamics
- **Training**: Synthetic data + benchmark datasets (GSM8K, ARC, HumanEval)

## Usage

```python
from transformers import AutoModel, AutoConfig

config = AutoConfig.from_pretrained("./hrrm-planner")
model = AutoModel.from_pretrained("./hrrm-planner")

# Generate with the model
import torch
input_ids = torch.randint(0, 1000, (1, 10))
output = model(input_ids)
```

## Architecture Features

- **HRMPlanner Specialization**: Optimized for specific reasoning tasks
- **Hierarchical Dynamics**: H-slow/T-fast two-timescale processing
- **Deep Supervision**: Loss computed at each H-cycle for stable training

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{hrrm-bootstrap-2024,
  title={Hierarchical Recurrent Reasoning Memory Bootstrap System},
  author={AIVillage Team},
  year={2024},
  note={Bootstrap implementation for Agent Forge integration}
}
```
