import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

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
        role="admin",
        permissions={HypeRAGPermissions.READ, HypeRAGPermissions.WRITE},
        session_id="sess",
        expires_at=datetime.now() + timedelta(hours=1),
    )


@pytest.mark.asyncio
async def test_add_and_search(protocol_handler, auth_context):
    handler, storage = protocol_handler

    add_result = await handler.handle_add_knowledge(auth_context, content="hello world", content_type="text")
    node_id = add_result["node_id"]

    search_result = await handler.handle_search_knowledge(auth_context, query="hello")

    assert search_result["total_count"] == 1
    assert search_result["results"][0]["id"] == node_id


@pytest.mark.asyncio
async def test_update(protocol_handler, auth_context):
    handler, storage = protocol_handler

    add_result = await handler.handle_add_knowledge(auth_context, content="foo", content_type="text")
    node_id = add_result["node_id"]

    await handler.handle_update_knowledge(auth_context, node_id=node_id, content="bar")

    search_result = await handler.handle_search_knowledge(auth_context, query="bar")

    assert search_result["total_count"] == 1
    assert search_result["results"][0]["content"] == "bar"


@pytest.mark.asyncio
async def test_delete(protocol_handler, auth_context):
    handler, storage = protocol_handler

    add_result = await handler.handle_add_knowledge(auth_context, content="to delete", content_type="text")
    node_id = add_result["node_id"]

    await handler.handle_delete_knowledge(auth_context, node_id=node_id)

    search_result = await handler.handle_search_knowledge(auth_context, query="to delete")

    assert search_result["total_count"] == 0
