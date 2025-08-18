import json
import sqlite3
from contextlib import contextmanager

import pytest

from core.security.secure_api_server import SecureAPIServer


class DummyConnection:
    def execute(self, _query):
        return 1


class DummyDB:
    def __init__(self, fail: bool = False):
        self.fail = fail

    @contextmanager
    def get_connection(self):
        if self.fail:
            raise sqlite3.OperationalError("db failure")
        yield DummyConnection()


@pytest.fixture()
def server_stub():
    server = SecureAPIServer.__new__(SecureAPIServer)
    server.tls_enabled = False
    server.cors_enabled = False
    server.profile_db = DummyDB()
    return server


@pytest.mark.asyncio
async def test_health_check_reports_latency(server_stub):
    resp = await server_stub._health_check(None)
    data = json.loads(resp.text)
    assert data["services"]["database"]["status"] == "operational"
    assert data["services"]["database"]["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_health_check_degraded_on_db_failure(server_stub):
    server_stub.profile_db = DummyDB(fail=True)
    resp = await server_stub._health_check(None)
    data = json.loads(resp.text)
    assert data["services"]["database"]["status"] == "degraded"
    assert "error" in data["services"]["database"]
