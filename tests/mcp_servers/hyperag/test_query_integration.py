from datetime import datetime, timedelta
from pathlib import Path
import sys

import pytest
import pytest_asyncio

# Ensure src is on path and modules reloaded
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.modules.pop("mcp_servers", None)
sys.modules.pop("mcp_servers.hyperag", None)

from mcp_servers.hyperag.auth import AuthContext, HypeRAGPermissions, PermissionManager
from mcp_servers.hyperag.models import ModelRegistry
from mcp_servers.hyperag.protocol import MCPProtocolHandler
from mcp_servers.hyperag.storage import SQLiteStorage


@pytest_asyncio.fixture
async def protocol_handler():
    permission_manager = PermissionManager(jwt_secret="test", enable_audit=False)
    storage = SQLiteStorage()
    model_registry = ModelRegistry()
    handler = MCPProtocolHandler(
        permission_manager=permission_manager,
        model_registry=model_registry,
        storage_backend=storage,
    )
    yield handler, storage
    await storage.close()


@pytest.fixture
def auth_context():
    return AuthContext(
        user_id="user",
        agent_id="user",
        role="default",
        permissions={HypeRAGPermissions.READ, HypeRAGPermissions.WRITE},
        session_id="sess",
        expires_at=datetime.now() + timedelta(hours=1),
    )


@pytest.mark.asyncio
async def test_handle_query_end_to_end(protocol_handler, auth_context):
    handler, _ = protocol_handler

    await handler.handle_add_knowledge(
        auth_context,
        content="Paris is the capital of France",
        content_type="text",
    )

    result = await handler.handle_query(
        auth_context,
        query="capital of France",
    )

    assert result["status"] == "success"
    assert "Paris" in result["result"]["answer"]
    assert result["result"]["sources"]
