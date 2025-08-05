"""Standardized Interfaces for AIVillage Components

This module defines consistent interfaces that all AIVillage components
must implement to ensure interoperability and standardization across
the entire platform.
"""

from .agent_interface import AgentCapability, AgentInterface, AgentStatus, MessageInterface, TaskInterface
from .communication_interface import CommunicationInterface, MessageProtocol, ProtocolCapability
from .processing_interface import ProcessingInterface, ProcessorCapability, ProcessorStatus
from .rag_interface import DocumentInterface, EmbeddingInterface, QueryInterface, RAGInterface
from .training_interface import ModelInterface, TrainingInterface, TrainingMetrics, TrainingStatus

__all__ = [
    # Agent interfaces
    "AgentInterface",
    "AgentCapability",
    "AgentStatus",
    "TaskInterface",
    "MessageInterface",
    # Communication interfaces
    "CommunicationInterface",
    "MessageProtocol",
    "ProtocolCapability",
    # RAG interfaces
    "RAGInterface",
    "QueryInterface",
    "DocumentInterface",
    "EmbeddingInterface",
    # Processing interfaces
    "ProcessingInterface",
    "ProcessorCapability",
    "ProcessorStatus",
    # Training interfaces
    "TrainingInterface",
    "ModelInterface",
    "TrainingStatus",
    "TrainingMetrics",
]
