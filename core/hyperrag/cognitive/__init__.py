"""
HyperRAG Cognitive Subsystems

Advanced reasoning and analysis:
- CognitiveNexus: Multi-perspective analysis and reasoning
- InsightEngine: Creative pattern discovery
- GraphFixer: Knowledge gap detection and repair
"""

try:
    from .cognitive_nexus import AnalysisResult, AnalysisType, CognitiveNexus, ConfidenceLevel, ReasoningStrategy
except ImportError:
    CognitiveNexus = None
    AnalysisType = None
    ReasoningStrategy = None
    ConfidenceLevel = None
    AnalysisResult = None

try:
    from .insight_engine import CreativityEngine
except ImportError:
    CreativityEngine = None

try:
    from .graph_fixer import GraphFixer
except ImportError:
    GraphFixer = None

__all__ = [
    "CognitiveNexus",
    "AnalysisType",
    "ReasoningStrategy",
    "ConfidenceLevel",
    "AnalysisResult",
    "CreativityEngine",
    "GraphFixer",
]
