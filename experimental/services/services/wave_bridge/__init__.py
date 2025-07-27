"""
WhatsApp Wave Bridge for AI Village
Sprint R-3+AF4: WhatsApp Integration with W&B Prompt Tuning

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
from .language_support import detect_language, auto_translate_flow, SUPPORTED_LANGUAGES
from .prompt_tuning import PromptTuner, ABTestManager
from .tutor_engine import AITutor
from .metrics import ResponseMetrics

__all__ = [
    "app",
    "detect_language",
    "auto_translate_flow",
    "SUPPORTED_LANGUAGES",
    "PromptTuner",
    "ABTestManager",
    "AITutor",
    "ResponseMetrics"
]
