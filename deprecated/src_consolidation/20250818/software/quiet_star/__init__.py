"""
Quiet Star Self-Learning System

Provides self-learning capabilities for meta-agents:
- Thought bubble generation (encrypted except King's public thoughts)
- Self-reasoning and reflection
- Continuous learning from interactions
- Internal monologue processing
- Special token handling for thought separation
"""

from .reflection import ReflectionSystem
from .self_reasoning import SelfReasoningEngine
from .thought_bubbles import ThoughtBubbleProcessor

__all__ = ["ThoughtBubbleProcessor", "SelfReasoningEngine", "ReflectionSystem"]
