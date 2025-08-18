"""
Quiet-STaR: Thought-Token Reasoning System
Implements internal reasoning with thought stripping for production use.
"""

from .alignment import AlignmentPrelude, EudaimoniaVirtue, MoralComplexity, create_aligned_training_pair
from .config import QuietSTaRConfig, get_default_config, get_inference_config, get_training_config
from .losses import LossComponents, QuietSTaRLoss, ReflectionQualityAssessor
from .mcp_tools import JournalEntry, MCPIntegration, MemoryPattern, SurpriseLevel, SurpriseMemoryJournal
from .model import QuietSTaRModelWrapper, ThoughtMixingHead, ThoughtSegment
from .sampler import SamplingResult, ThoughtLeakDetector, ThoughtSampler
from .temperature_recognition import (
    AdaptiveTemperatureSampler,
    ConfidenceLevel,
    ContextComplexity,
    TemperatureSelfRecognition,
    TemperatureStrategy,
    create_temperature_recognizer,
)

# Optional CLI import - only load if transformers is available
try:
    from .cli import quiet_star_cli

    _CLI_AVAILABLE = True
except ImportError:
    # CLI not available due to missing dependencies (transformers, etc)
    quiet_star_cli = None
    _CLI_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "QuietSTaRConfig",
    "get_default_config",
    "get_training_config",
    "get_inference_config",
    # Model components
    "QuietSTaRModelWrapper",
    "ThoughtMixingHead",
    "ThoughtSegment",
    # Training
    "QuietSTaRLoss",
    "LossComponents",
    "ReflectionQualityAssessor",
    # Inference
    "ThoughtSampler",
    "SamplingResult",
    "ThoughtLeakDetector",
    # Temperature Recognition
    "TemperatureSelfRecognition",
    "AdaptiveTemperatureSampler",
    "TemperatureStrategy",
    "ConfidenceLevel",
    "ContextComplexity",
    "create_temperature_recognizer",
    # Alignment
    "AlignmentPrelude",
    "EudaimoniaVirtue",
    "MoralComplexity",
    "create_aligned_training_pair",
    # MCP Tools and Journaling
    "SurpriseMemoryJournal",
    "MCPIntegration",
    "SurpriseLevel",
    "JournalEntry",
    "MemoryPattern",
]

# Add CLI to exports if available
if _CLI_AVAILABLE:
    __all__.append("quiet_star_cli")
