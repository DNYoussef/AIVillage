import sys
from pathlib import Path
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from mcp_servers.hyperag.auth import AuthContext, PermissionManager
from mcp_servers.hyperag.models import ModelRegistry, Node
from mcp_servers.hyperag.protocol import MCPProtocolHandler


class InMemoryStorage:
    """Simple in-memory storage backend for testing."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}

    async def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    async def search_nodes(self, query: str, limit: int = 10, **kwargs):
        q = query.lower()
        results = [n for n in self.nodes.values() if q in n.content.lower()]
        return results[:limit]

    async def update_node(self, node_id: str, content: str | None = None, metadata: dict | None = None):
        node = self.nodes[node_id]
        if content:
            node.content = content
        if metadata:
            node.metadata.update(metadata)

    async def delete_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)


@pytest.fixture
def permission_manager():
    return PermissionManager(jwt_secret="test-secret", enable_audit=False)


@pytest.fixture
def auth_context(permission_manager):
    perms = set(permission_manager.permissions_config["admin"])
    return AuthContext(
        user_id="user1",
        agent_id="agent1",
        role="admin",
        permissions=perms,
        session_id="sess1",
        expires_at=datetime.now() + timedelta(hours=1),
    )


@pytest.fixture
def protocol_handler(permission_manager):
    registry = ModelRegistry()
    storage = InMemoryStorage()
    return MCPProtocolHandler(permission_manager, registry, storage)


@pytest.mark.asyncio
async def test_end_to_end_query_handling(protocol_handler, auth_context):
    await protocol_handler.handle_add_knowledge(auth_context, content="The sky is blue.")
    await protocol_handler.handle_add_knowledge(auth_context, content="Grass is green.")

    result = await protocol_handler.handle_query(auth_context, query="sky")

    assert result["status"] == "success"
    answer = result["result"]["answer"].lower()
    assert "sky" in answer and "blue" in answer
    sources = [s["content"].lower() for s in result["result"]["sources"]]
    assert any("sky is blue" in s for s in sources)


@pytest.mark.asyncio
async def test_knowledge_management_cycle(protocol_handler, auth_context):
    add_resp = await protocol_handler.handle_add_knowledge(auth_context, content="Apples are red.")
    node_id = add_resp["node_id"]

    search1 = await protocol_handler.handle_search_knowledge(auth_context, query="Apples")
    assert search1["total_count"] == 1

    await protocol_handler.handle_update_knowledge(auth_context, node_id=node_id, content="Apples are typically red.")
    search2 = await protocol_handler.handle_search_knowledge(auth_context, query="typically")
    assert search2["total_count"] == 1

    await protocol_handler.handle_delete_knowledge(auth_context, node_id=node_id)
    search3 = await protocol_handler.handle_search_knowledge(auth_context, query="Apples")
    assert search3["total_count"] == 0
