"""Agent Forge - Advanced Prompt Engineering for AI Village
Part B: Agent Forge Phase 4 - Prompt Engineering Package.
"""

from .prompt_engineering.ab_testing import PromptABTest, prompt_ab_test
from .prompt_engineering.prompt_baker import PromptBaker, prompt_baker
from .prompt_engineering.tutor_prompts import TutorPromptEngineer, tutor_prompt_engineer

__version__ = "1.0.0"
__author__ = "AI Village Team"

__all__ = [
    "PromptABTest",
    "PromptBaker",
    "TutorPromptEngineer",
    "prompt_ab_test",
    "prompt_baker",
    "tutor_prompt_engineer",
]
