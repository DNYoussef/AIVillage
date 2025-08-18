"""
Temperature-Alternating Self-Modeling Fast-Grokking Training System

A comprehensive system that implements:
- Temperature curriculum learning with non-overlapping → overlapping bin progression
- Multi-head self-modeling (activation prediction, temperature inference, grok stage classification)
- Grokfast optimization with telemetry-based lambda gating (ID↓ + S_slow↑)
- Comprehensive telemetry tracking and encoding
- OpenRouter integration for prompt template generation
- Complete training loop with checkpointing and analysis
"""

from .grokfast_ctrl import (
    GrokfastConfig,
    GrokfastMode,
    GrokfastOptimizer,
    GrokSignalDetector,
    TelemetryState,
    TelemetryTracker,
    create_grokfast_optimizer,
    create_telemetry_tracker,
)
from .openrouter_integration import (
    OpenRouterClient,
    OpenRouterTempAltSystem,
    PromptCategory,
    PromptComplexity,
    PromptSuiteManager,
    PromptTemplate,
    create_openrouter_system,
    create_prompt_suite_manager,
)
from .self_model import MultiHeadSelfModel, SelfModelHead, StageClassifier, StageHead, TempCurriculum, TempInferHead
from .telemetry_encode import (
    EncodedTelemetry,
    TelemetryAnomalyDetector,
    TelemetryEncoder,
    TelemetryEncoding,
    TelemetryPredictor,
    create_anomaly_detector,
    create_telemetry_encoder,
    create_telemetry_predictor,
)

# CLI interface
from .temp_alt_cli import temp_alt_cli
from .temp_alt_loop import TempAltConfig, TempAlternationTrainer, TrainingState, create_temp_alt_trainer
from .temp_curriculum import (
    GeneratedSnippet,
    GrokStage,
    SnippetDataset,
    TeacherConsistency,
    TempBin,
    TempBinScheduler,
    TempBinType,
    TempRound,
    create_nonoverlap_scheduler,
    create_overlap_scheduler,
)

# Testing suite
from .test_temp_alt_system import run_tests

__version__ = "1.0.0"

__all__ = [
    # Temperature Curriculum
    "TempBinScheduler",
    "TempRound",
    "TempBinType",
    "GrokStage",
    "TempBin",
    "GeneratedSnippet",
    "SnippetDataset",
    "TeacherConsistency",
    "create_nonoverlap_scheduler",
    "create_overlap_scheduler",
    # Self-Modeling
    "SelfModelHead",
    "TempInferHead",
    "StageHead",
    "MultiHeadSelfModel",
    "TempCurriculum",
    "StageClassifier",
    # Grokfast & Telemetry
    "GrokfastOptimizer",
    "TelemetryTracker",
    "TelemetryState",
    "GrokSignalDetector",
    "GrokfastConfig",
    "GrokfastMode",
    "create_grokfast_optimizer",
    "create_telemetry_tracker",
    # Telemetry Encoding
    "TelemetryEncoder",
    "TelemetryPredictor",
    "TelemetryAnomalyDetector",
    "TelemetryEncoding",
    "EncodedTelemetry",
    "create_telemetry_encoder",
    "create_telemetry_predictor",
    "create_anomaly_detector",
    # Training Loop
    "TempAlternationTrainer",
    "TempAltConfig",
    "TrainingState",
    "create_temp_alt_trainer",
    # OpenRouter Integration
    "OpenRouterTempAltSystem",
    "PromptSuiteManager",
    "PromptTemplate",
    "PromptCategory",
    "PromptComplexity",
    "OpenRouterClient",
    "create_openrouter_system",
    "create_prompt_suite_manager",
    # CLI & Testing
    "temp_alt_cli",
    "run_tests",
]


def get_system_info():
    """Get information about the Temperature-Alternating system."""
    return {
        "name": "Temperature-Alternating Self-Modeling Fast-Grokking System",
        "version": __version__,
        "description": __doc__.strip(),
        "components": {
            "temperature_curriculum": "Non-overlapping → overlapping temperature bin scheduling",
            "self_modeling": "Multi-head prediction: activations, temperature, grok stage",
            "grokfast": "Lambda-gated optimization with telemetry (ID↓ + S_slow↑)",
            "telemetry": "Comprehensive training dynamics tracking and encoding",
            "openrouter": "Prompt template generation and consistency evaluation",
            "training_loop": "Complete training system with checkpointing",
            "cli": "Command-line interface for all operations",
            "testing": "Comprehensive test suite with integration tests",
        },
        "key_features": [
            "🌡️ Temperature-aware curriculum learning",
            "🧠 Multi-head self-modeling architecture",
            "⚡ Grokfast optimization with automatic gating",
            "📊 Real-time telemetry tracking and analysis",
            "🌐 OpenRouter API integration for prompt generation",
            "🎯 Grok onset detection (ID↓ + S_slow↑)",
            "🔄 Automatic curriculum round advancement",
            "💾 Complete checkpointing and resume capability",
            "🧪 Comprehensive testing and validation",
            "⌨️ Full CLI interface for all operations",
        ],
        "research_papers": [
            "Intelligence at the Edge of Chaos - Optimal complexity training",
            "Grokfast - 50x acceleration via slow gradient amplification",
            "Quiet-STaR - Internal reasoning with encrypted thoughts",
            "Self-Modeling Networks - Predictive efficiency improvements",
        ],
    }


# Maintain backward compatibility
def expert_vectors():
    """Legacy compatibility function."""
    return {
        "status": "implemented",
        "module": "temperature_alternating_training",
        "version": __version__,
    }


if __name__ == "__main__":
    import json

    info = get_system_info()
    print("🌡️⚡ Temperature-Alternating Self-Modeling Fast-Grokking System")
    print("=" * 70)
    print(f"Version: {info['version']}")
    print(f"\n{info['description']}")
    print("\n🎯 Key Features:")
    for feature in info["key_features"]:
        print(f"   {feature}")
    print("\n🔬 Research Integration:")
    for paper in info["research_papers"]:
        print(f"   • {paper}")
    print(f"\n📦 Components: {len(info['components'])} major systems")
    print("\n🚀 Ready for production deployment!")
