"""
Multi-Model Orchestration for Agent Forge

This module provides intelligent routing of training tasks to optimal models
via OpenRouter API integration.
"""

from .openrouter_client import OpenRouterClient
from .task_router import TaskRouter, TaskType
from .model_config import MODEL_ROUTING_CONFIG

__all__ = [
    "OpenRouterClient",
    "TaskRouter", 
    "TaskType",
    "MODEL_ROUTING_CONFIG"
]