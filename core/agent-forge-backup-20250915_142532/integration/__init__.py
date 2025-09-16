"""
Agent Forge Integration Module

Provides integration capabilities for the Agent Forge pipeline with:
- P2P federated training across distributed devices
- Fog compute orchestration and load balancing
- Edge device resource management
- Distributed coordination and fault tolerance
"""

# Import integration components
try:
    FEDERATED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Federated training integration not available: {e}")
    FEDERATED_AVAILABLE = False

try:
    FOG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Fog compute integration not available: {e}")
    FOG_AVAILABLE = False

# Export available integrations
__all__ = []

if FEDERATED_AVAILABLE:
    __all__.extend(
        ["FederatedTrainingConfig", "FederatedAgentForge", "create_federated_pipeline", "run_federated_agent_forge"]
    )

if FOG_AVAILABLE:
    __all__.extend(
        ["FogComputeConfig", "FogComputeOrchestrator", "create_fog_compute_pipeline", "run_fog_distributed_agent_forge"]
    )


def get_available_integrations():
    """Get list of available integration capabilities."""
    integrations = []

    if FEDERATED_AVAILABLE:
        integrations.append("federated_training")
    if FOG_AVAILABLE:
        integrations.append("fog_compute")

    return integrations


def get_integration_status():
    """Get detailed status of integration capabilities."""
    return {
        "federated_training": {
            "available": FEDERATED_AVAILABLE,
            "description": "P2P federated training across distributed devices",
        },
        "fog_compute": {"available": FOG_AVAILABLE, "description": "Fog compute orchestration and load balancing"},
    }
