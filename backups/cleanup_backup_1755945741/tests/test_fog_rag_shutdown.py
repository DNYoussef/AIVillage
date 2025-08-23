import asyncio
import enum
import importlib.util
import sys
import types
from pathlib import Path

import pytest

pkg_root = Path(__file__).resolve().parent.parent / "packages"
packages_mod = types.ModuleType("packages")
packages_mod.__path__ = [str(pkg_root)]
sys.modules.setdefault("packages", packages_mod)

rag_mod = types.ModuleType("packages.rag")
rag_mod.__path__ = [str(pkg_root / "rag")]
sys.modules.setdefault("packages.rag", rag_mod)

# Stub core module to satisfy imports without loading full package
core_mod = types.ModuleType("packages.rag.core")
sys.modules.setdefault("packages.rag.core", core_mod)
hyper_stub = types.ModuleType("packages.rag.core.hyper_rag")


class HyperRAGOrchestrator:
    pass


class QueryMode(enum.Enum):
    BALANCED = "balanced"


hyper_stub.HyperRAGOrchestrator = HyperRAGOrchestrator
hyper_stub.QueryMode = QueryMode
sys.modules.setdefault("packages.rag.core.hyper_rag", hyper_stub)

spec = importlib.util.spec_from_file_location(
    "packages.rag.integration.fog_rag_bridge",
    pkg_root / "rag" / "integration" / "fog_rag_bridge.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules["packages.rag.integration.fog_rag_bridge"] = module
spec.loader.exec_module(module)

FogRAGCoordinator = module.FogRAGCoordinator


@pytest.mark.asyncio
async def test_shutdown_terminates_background_tasks():
    coord = FogRAGCoordinator()
    coord._node_discovery_task = asyncio.create_task(asyncio.sleep(10))
    coord._health_monitor_task = asyncio.create_task(asyncio.sleep(10))

    await coord.shutdown()
    assert coord._node_discovery_task.cancelled()
    assert coord._node_discovery_task.done()
    assert coord._health_monitor_task.cancelled()
    assert coord._health_monitor_task.done()
