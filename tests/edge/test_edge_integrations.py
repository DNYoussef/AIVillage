import asyncio
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import sys
import types

# Stub optional distributed RAG module to satisfy imports
stub_module = types.ModuleType("distributed_rag_coordinator")


class DummyCoordinator:  # pragma: no cover - simple stub
    pass


stub_module.DistributedRAGCoordinator = DummyCoordinator
sys.modules.setdefault("packages", types.ModuleType("packages"))
sys.modules.setdefault("packages.rag", types.ModuleType("packages.rag"))
sys.modules.setdefault("packages.rag.distributed", types.ModuleType("packages.rag.distributed"))
sys.modules["packages.rag.distributed.distributed_rag_coordinator"] = stub_module

from infrastructure.edge import create_edge_system, DataSource
from infrastructure.edge.communication.chat_engine import ChatMode


@pytest.mark.asyncio
async def test_health_and_knowledge_integrations():
    with tempfile.TemporaryDirectory() as temp_dir:
        system = await create_edge_system(
            device_name="TestDevice",
            data_dir=Path(temp_dir),
            enable_digital_twin=True,
            enable_mobile_bridge=False,
            enable_chat_engine=True,
        )

        # stop background tasks started during initialization
        system.initialized = False

        # Verify health monitoring registry contains chat engine
        assert "chat_engine" in system.health_checks

        # Add knowledge to knowledge system
        await system.knowledge_system.add_knowledge(
            "edge integration knowledge", DataSource.CONVERSATION, {}
        )

        # Force chat engine to local mode to avoid network access
        system.chat_engine._mode = ChatMode.LOCAL

        # Chat processing should include knowledge hits
        response = await system.process_chat(
            "edge integration knowledge", "conv1"
        )
        assert "knowledge" in response
        assert any("edge integration knowledge" in k for k in response["knowledge"])

        # Prepare health monitoring
        system.chat_engine.get_system_status = MagicMock(return_value={"ok": True})
        system.health_checks["chat_engine"] = system.chat_engine.get_system_status

        from asyncio import sleep as real_sleep

        async def fast_sleep(_):
            await real_sleep(0)

        # Run health monitoring task once
        system.initialized = True
        with patch("infrastructure.edge.asyncio.sleep", new=fast_sleep):
            task = asyncio.create_task(system._health_monitoring_task())
            await asyncio.sleep(0)
            system.initialized = False
            await asyncio.sleep(0)
            await task

        system.chat_engine.get_system_status.assert_called()

        # Run knowledge sync task once
        system.initialized = True
        with patch("infrastructure.edge.asyncio.sleep", new=fast_sleep):
            task = asyncio.create_task(system._knowledge_sync_task())
            await asyncio.sleep(0)
            system.initialized = False
            await asyncio.sleep(0)
            await task

        assert system.system_metrics["knowledge_pieces"] > 0

        await system.shutdown()
