# isort: skip_file

import json
from datetime import datetime, timedelta

import pytest

from mcp_servers.hyperag.auth import AuthContext, HypeRAGPermissions
from mcp_servers.hyperag.models import ModelRegistry
from mcp_servers.hyperag.protocol import (
    MCPProtocolHandler,
    InvalidRequest,
)


class DummyPermissionManager:
    async def require_permission(self, context, permission, resource=None):
        return None


def make_context():
    return AuthContext(
        user_id="u1",
        agent_id="a1",
        role="admin",
        permissions={HypeRAGPermissions.ADMIN},
        session_id="s1",
        expires_at=datetime.now() + timedelta(hours=1),
    )


@pytest.mark.asyncio
async def test_model_registration_persists(tmp_path):
    handler = MCPProtocolHandler(DummyPermissionManager(), ModelRegistry())
    handler.model_registry_path = tmp_path / "registry.json"

    config = {"model_name": "test", "model_type": "transformer"}
    result = await handler.handle_register_model(make_context(), "agent1", config)

    assert result["status"] == "registered"
    assert result["model_metadata"]["config"] == config
    data = json.loads(handler.model_registry_path.read_text())
    assert "agent1" in data


@pytest.mark.asyncio
async def test_duplicate_model_registration_error(tmp_path):
    handler = MCPProtocolHandler(DummyPermissionManager(), ModelRegistry())
    handler.model_registry_path = tmp_path / "registry.json"

    config = {"model_name": "test", "model_type": "transformer"}
    await handler.handle_register_model(make_context(), "agent1", config)

    with pytest.raises(InvalidRequest):
        await handler.handle_register_model(make_context(), "agent1", config)
