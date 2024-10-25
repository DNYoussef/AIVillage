"""Sage Agent package."""

from .sage_agent import SageAgent
from .core.config import SageAgentConfig

# Core components
from .core import (
    QueryProcessor,
    ReasoningAgent,
    ResponseGenerator,
    UserIntentInterpreter,
    CollaborationManager,
    ContinuousLearningLayer,
    FoundationalLayer,
    SelfEvolvingSystem,
    TaskExecutor
)

# Knowledge management
from .knowledge_management import (
    KnowledgeGraphAgent,
    DynamicKnowledgeIntegrationAgent,
    KnowledgeSynthesizer
)

# RAG management
from .rag_management.unified_manager import UnifiedRAGManager

# Research capabilities
from .research import (
    ResearchCapabilities,
    WebScraper,
    OnlineSearchEngine,
    ReportWriter
)

__all__ = [
    # Main components
    "SageAgent",
    "SageAgentConfig",
    
    # Core components
    "QueryProcessor",
    "ReasoningAgent",
    "ResponseGenerator",
    "UserIntentInterpreter",
    "CollaborationManager",
    "ContinuousLearningLayer",
    "FoundationalLayer",
    "SelfEvolvingSystem",
    "TaskExecutor",
    
    # Knowledge management
    "KnowledgeGraphAgent",
    "DynamicKnowledgeIntegrationAgent",
    "KnowledgeSynthesizer",
    
    # RAG management
    "UnifiedRAGManager",
    
    # Research capabilities
    "ResearchCapabilities",
    "WebScraper",
    "OnlineSearchEngine",
    "ReportWriter"
]
