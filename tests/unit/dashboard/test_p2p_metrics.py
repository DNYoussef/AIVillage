#!/usr/bin/env python3
"""Tests for P2P metric collection in EnhancedOptimizationDashboard."""

from pathlib import Path
import sys

import pytest

# Ensure project root is on sys.path for direct imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import importlib.util
import types

# Create minimal stubs for required modules to avoid importing full package
analytics_stub = types.ModuleType("infrastructure.optimization.analytics")
class _PA:  # type: ignore
    pass
analytics_stub.PerformanceAnalytics = _PA
sys.modules["infrastructure.optimization.analytics"] = analytics_stub

monitoring_stub = types.ModuleType("infrastructure.optimization.monitoring")
class _PM:  # type: ignore
    pass
monitoring_stub.PerformanceMonitor = _PM
sys.modules["infrastructure.optimization.monitoring"] = monitoring_stub

netopt_stub = types.ModuleType("infrastructure.optimization.network_optimizer")
class _SENO:  # type: ignore
    pass
netopt_stub.SecurityEnhancedNetworkOptimizer = _SENO
sys.modules["infrastructure.optimization.network_optimizer"] = netopt_stub

resource_stub = types.ModuleType("infrastructure.optimization.resource_manager")
class _RM:  # type: ignore
    pass
resource_stub.ResourceManager = _RM
sys.modules["infrastructure.optimization.resource_manager"] = resource_stub

# Create package structure so relative imports resolve
infra_pkg = types.ModuleType("infrastructure")
infra_pkg.__path__ = [str(project_root / "infrastructure")]
sys.modules["infrastructure"] = infra_pkg

opt_pkg = types.ModuleType("infrastructure.optimization")
opt_pkg.__path__ = [str(project_root / "infrastructure" / "optimization")]
sys.modules["infrastructure.optimization"] = opt_pkg

# Load dashboard_integration module
module_path = project_root / "infrastructure" / "optimization" / "dashboard_integration.py"
spec = importlib.util.spec_from_file_location(
    "infrastructure.optimization.dashboard_integration", module_path
)
dashboard_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(dashboard_module)

EnhancedOptimizationDashboard = dashboard_module.EnhancedOptimizationDashboard


class DummyNatTraversalOptimizer:
    """Simple NAT traversal optimizer exposing BitChat peer information."""

    def __init__(self, peer_count: int):
        # Using dict to emulate structures commonly used in production
        self.active_peers = {f"peer_{i}": {} for i in range(peer_count)}


class DummyProtocolMultiplexer:
    """Simple protocol multiplexer exposing BetaNet circuit information."""

    def __init__(self, circuit_count: int):
        self.active_circuits = {f"circuit_{i}": {} for i in range(circuit_count)}


class DummyNetworkOptimizer:
    """Network optimizer exposing P2P components for dashboard metrics."""

    def __init__(self, peers: int, circuits: int):
        self.nat_traversal_optimizer = DummyNatTraversalOptimizer(peers)
        self.protocol_multiplexer = DummyProtocolMultiplexer(circuits)


@pytest.mark.asyncio
async def test_p2p_metrics_report_real_values():
    """Ensure P2P metrics reflect underlying optimizer data."""
    dashboard = EnhancedOptimizationDashboard()
    dashboard.network_optimizer = DummyNetworkOptimizer(peers=2, circuits=3)

    stats = await dashboard._collect_p2p_infrastructure_metrics()

    assert stats["bitchat_peers"] == 2
    assert stats["betanet_circuits"] == 3
