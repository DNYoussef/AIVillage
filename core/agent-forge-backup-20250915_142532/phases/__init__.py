#!/usr/bin/env python3
"""
Agent Forge Phases Module

Consolidated implementations of all Agent Forge phases:
1. Cognate: Model creation and initialization (NEWLY IMPLEMENTED)
2. EvoMerge: Evolutionary model merging and optimization
3. Quiet-STaR: Reasoning enhancement with thought baking
4. BitNet Compression: 1.58-bit quantization compression
5. Forge Training: Main training loop with Grokfast acceleration
6. Tool & Persona Baking: Tool and persona specialization
7. ADAS: Architecture Discovery and Search with vector composition
8. Final Compression: SeedLM + VPTQ + Hypercompression stack

All phases implement the PhaseController interface for consistent
model passing and result reporting. Now COMPLETE 8-phase pipeline!
"""

# Import all phase controllers and configurations
try:
    from .cognate import CognateConfig, CognatePhase
except ImportError as e:
    print(f"Warning: Cognate phase not available: {e}")
    CognatePhase = CognateConfig = None

try:
    from .evomerge import EvoMergeConfig, EvoMergePhase, MergeOperators
except ImportError as e:
    print(f"Warning: EvoMerge phase not available: {e}")
    EvoMergePhase = EvoMergeConfig = MergeOperators = None

try:
    from .quietstar import PromptBakingEngine, QuietSTaRConfig, QuietSTaRPhase
except ImportError as e:
    print(f"Warning: Quiet-STaR phase not available: {e}")
    QuietSTaRPhase = QuietSTaRConfig = PromptBakingEngine = None

try:
    from .bitnet_compression import BitNetCompressionPhase, BitNetConfig, BitNetQuantizer
except ImportError as e:
    print(f"Warning: BitNet compression phase not available: {e}")
    BitNetCompressionPhase = BitNetConfig = BitNetQuantizer = None

try:
    from .forge_training import (
        DreamCycleManager,
        EdgeController,
        ForgeTrainingConfig,
        ForgeTrainingPhase,
        GrokfastAdamW,
        SelfModelingModule,
    )
except ImportError as e:
    print(f"Warning: Forge training phase not available: {e}")
    ForgeTrainingPhase = ForgeTrainingConfig = None
    GrokfastAdamW = EdgeController = SelfModelingModule = DreamCycleManager = None

try:
    from .tool_persona_baking import (
        PersonaOptimizationSystem,
        ToolIntegrationSystem,
        ToolPersonaBakingConfig,
        ToolPersonaBakingPhase,
    )
except ImportError as e:
    print(f"Warning: Tool & Persona baking phase not available: {e}")
    ToolPersonaBakingPhase = ToolPersonaBakingConfig = None
    ToolIntegrationSystem = PersonaOptimizationSystem = None

try:
    from .adas import ADASConfig, ADASPhase, ArchitectureConfig, NSGAIIOptimizer, VectorCompositionOperator
except ImportError as e:
    print(f"Warning: ADAS phase not available: {e}")
    ADASPhase = ADASConfig = None
    VectorCompositionOperator = NSGAIIOptimizer = ArchitectureConfig = None

try:
    from .final_compression import (
        FinalCompressionConfig,
        FinalCompressionPhase,
        HyperCompressionEncoder,
        SEEDLMCompressor,
        VPTQCompressor,
    )
except ImportError as e:
    print(f"Warning: Final compression phase not available: {e}")
    FinalCompressionPhase = FinalCompressionConfig = None
    SEEDLMCompressor = VPTQCompressor = HyperCompressionEncoder = None

# Export all available phases
__all__ = [
    # Phase Controllers
    "CognatePhase",
    "EvoMergePhase",
    "QuietSTaRPhase",
    "BitNetCompressionPhase",
    "ForgeTrainingPhase",
    "ToolPersonaBakingPhase",
    "ADASPhase",
    "FinalCompressionPhase",
    # Phase Configurations
    "CognateConfig",
    "EvoMergeConfig",
    "QuietSTaRConfig",
    "BitNetConfig",
    "ForgeTrainingConfig",
    "ToolPersonaBakingConfig",
    "ADASConfig",
    "FinalCompressionConfig",
    # Core Components
    "MergeOperators",
    "PromptBakingEngine",
    "BitNetQuantizer",
    "GrokfastAdamW",
    "EdgeController",
    "SelfModelingModule",
    "DreamCycleManager",
    "ToolIntegrationSystem",
    "PersonaOptimizationSystem",
    "VectorCompositionOperator",
    "NSGAIIOptimizer",
    "ArchitectureConfig",
    "SEEDLMCompressor",
    "VPTQCompressor",
    "HyperCompressionEncoder",
]

# Remove None values from __all__
__all__ = [name for name in __all__ if globals().get(name) is not None]


def get_available_phases():
    """Get list of available phase controllers in execution order."""
    phases = []

    if CognatePhase:
        phases.append(("Cognate", CognatePhase))
    if EvoMergePhase:
        phases.append(("EvoMerge", EvoMergePhase))
    if QuietSTaRPhase:
        phases.append(("Quiet-STaR", QuietSTaRPhase))
    if BitNetCompressionPhase:
        phases.append(("BitNet Compression", BitNetCompressionPhase))
    if ForgeTrainingPhase:
        phases.append(("Forge Training", ForgeTrainingPhase))
    if ToolPersonaBakingPhase:
        phases.append(("Tool & Persona Baking", ToolPersonaBakingPhase))
    if ADASPhase:
        phases.append(("ADAS", ADASPhase))
    if FinalCompressionPhase:
        phases.append(("Final Compression", FinalCompressionPhase))

    return phases
