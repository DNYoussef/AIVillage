"""WhatsApp Wave Bridge for AI Village
Sprint R-3+AF4: WhatsApp Integration with W&B Prompt Tuning.

A high-performance WhatsApp tutoring service with:
- Sub-5 second response times
- Multi-language support (10 languages)
- W&B-powered prompt optimization
- A/B testing for interaction optimization
- Comprehensive performance monitoring
"""

__version__ = "1.0.0"
__author__ = "AI Village Team"

from .app import app
from .language_support import SUPPORTED_LANGUAGES, auto_translate_flow, detect_language
from .metrics import ResponseMetrics
from .prompt_tuning import ABTestManager, PromptTuner
from .tutor_engine import AITutor

__all__ = [
    "SUPPORTED_LANGUAGES",
    "ABTestManager",
    "AITutor",
    "PromptTuner",
    "ResponseMetrics",
    "app",
    "auto_translate_flow",
    "detect_language",
]
