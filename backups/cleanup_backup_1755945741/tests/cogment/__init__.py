"""
Cogment System Test Suite

Comprehensive testing for the unified Cogment model (23.7M parameters) that replaces 
the 3-model HRRM approach (150M total parameters).

Test coverage includes:
- Core model components (Agent 1): RefinementCore, ACT halting, parameter counting
- Memory system (Agent 2): GatedLTM read/write, surprise gating, memory decay  
- Heads optimization (Agent 3): Image, text, task adapters, vocabulary optimization
- Training curriculum (Agent 4): 4-stage pipeline, GrokFast integration, loss functions
- Data pipeline (Agent 5): All stages, ~300 ARC augmentations, batch generation
- Integration layer (Agent 6): EvoMerge adapter, HF export, deployment pipeline
- Configuration system (Agent 7): Parameter budget, validation, Option A configs
- Performance validation: HRRM vs Cogment benchmarking
- End-to-end integration: Full model instantiation through training
"""

__version__ = "1.0.0"
__description__ = "Cogment System Validation Test Suite"