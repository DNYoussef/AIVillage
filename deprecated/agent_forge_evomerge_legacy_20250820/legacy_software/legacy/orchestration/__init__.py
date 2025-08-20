"""Multi-Model Orchestration for Agent Forge.

This module provides intelligent routing of training tasks to optimal models
via OpenRouter API integration.
"""

from .model_config import MODEL_ROUTING_CONFIG
from .openrouter_client import OpenRouterClient
from .task_router import TaskRouter, TaskType

__all__ = ["MODEL_ROUTING_CONFIG", "OpenRouterClient", "TaskRouter", "TaskType"]
