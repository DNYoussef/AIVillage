# Deprecated Duplicate Files

These files have been consolidated into the new `cognate-pretrain/` package structure.

## Moved Files

- `optimal_25m_training.py` → Replaced by `cognate-pretrain/cognate_creator.py`
- `train_and_save_25m_models.py` → Replaced by `cognate-pretrain/model_factory.py`
- `cognate_evomerge_50gen.py` → Functionality moved to `cognate-pretrain/`
- `cognate_25m_results.json` → Old results, replaced by new metadata system

## New Structure

All Cognate model creation is now handled by:
```
cognate-pretrain/
├── model_factory.py       # Main entry point (replaces train_and_save_25m_models.py)
├── cognate_creator.py     # Core creation logic (replaces optimal_25m_training.py)
├── pretrain_pipeline.py   # Optional pre-training
└── phase_integration.py   # Agent Forge integration
```

## Migration

If you were using any of these deprecated files directly, update your imports:

```python
# OLD
from core.agent_forge.phases.optimal_25m_training import create_25m_model

# NEW
from core.agent_forge.phases.cognate_pretrain import create_three_cognate_models
```

These files are safe to delete after testing the new system.
