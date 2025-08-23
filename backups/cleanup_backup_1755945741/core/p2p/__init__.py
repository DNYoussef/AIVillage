#!/usr/bin/env python3
"""
Core P2P Mesh Protocol Package
Agent 4: Network Communication Specialist

Unified P2P mesh networking system with >90% message delivery reliability.
Consolidates 105+ P2P files into ONE reliable mesh network system.

Main exports:
- UnifiedMeshProtocol: Core mesh protocol with reliability guarantees
- MeshMessage: Unified message format
- ReliabilityConfig: Configuration for reliability mechanisms
- Factory functions for easy instantiation
"""

from .mesh_protocol import (
    # Core protocol
    UnifiedMeshProtocol,
    create_mesh_protocol,
    
    # Message types
    MeshMessage,
    MessagePriority,
    MessageStatus,
    
    # Transport types
    TransportType,
    
    # Configuration
    ReliabilityConfig,
    
    # Network entities
    PeerInfo,
    NodeStatus,
    
    # Utility classes
    CircuitBreaker,
    ConnectionPool
)

__version__ = "1.0.0"
__all__ = [
    # Core protocol
    "UnifiedMeshProtocol",
    "create_mesh_protocol",
    
    # Message types
    "MeshMessage", 
    "MessagePriority",
    "MessageStatus",
    
    # Transport types
    "TransportType",
    
    # Configuration
    "ReliabilityConfig",
    
    # Network entities
    "PeerInfo",
    "NodeStatus",
    
    # Utility classes
    "CircuitBreaker",
    "ConnectionPool"
]

# Package metadata
__description__ = "Unified P2P Mesh Protocol with >90% Message Delivery Reliability"
__author__ = "Agent 4: Network Communication Specialist"
__reliability_target__ = ">90% message delivery, <50ms latency, >1000 msg/sec throughput"