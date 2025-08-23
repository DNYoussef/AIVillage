"""
AIVillage Mobile Integration Package
Unified mobile interface consolidating all mobile functionality.

This package consolidates:
- Digital Twin Concierge (on-device AI assistant)
- Mini-RAG System (privacy-focused knowledge management)
- Resource Manager (battery/thermal optimization)
- Mobile P2P integration
- Cross-platform mobile support
"""

from .shared.digital_twin_concierge import DigitalTwinConcierge, DataPoint, SurpriseBasedLearning
from .shared.mini_rag_system import MiniRAGSystem, KnowledgePiece, KnowledgeRelevance
from .shared.resource_manager import MobileResourceManager, MobileDeviceProfile, PowerMode

__version__ = "2.0.0"
__all__ = [
    "DigitalTwinConcierge",
    "DataPoint", 
    "SurpriseBasedLearning",
    "MiniRAGSystem",
    "KnowledgePiece",
    "KnowledgeRelevance",
    "MobileResourceManager",
    "MobileDeviceProfile",
    "PowerMode"
]