#!/usr/bin/env python3
"""
Cognate Pre-training Phase - Dedicated Model Creation Pipeline

This module provides the complete pipeline for creating 3 Cognate models that feed into EvoMerge.
Consolidates all Cognate-related functionality into a single, organized package.

Pipeline: 3x Cognate Models → EvoMerge (Phase 2) → Quiet-STaR → BitNet → etc.
"""

from .cognate_creator import CognateCreatorConfig, CognateModelCreator
from .model_factory import create_three_cognate_models
from .pretrain_pipeline import CognatePretrainPipeline

__all__ = ["CognateModelCreator", "CognateCreatorConfig", "CognatePretrainPipeline", "create_three_cognate_models"]
