import importlib
import socket
import sys
import time

import pytest

if "requests" in sys.modules:
    del sys.modules["requests"]
requests = importlib.import_module("requests")


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait(url):
    for _ in range(20):
        try:
            if requests.get(url, timeout=1).ok:
                return True
        except requests.ConnectionError:
            time.sleep(0.5)
    return False


def test_chat_roundtrip():
    if not _port_open("localhost", 8000):
        pytest.skip("gateway not running")
    assert wait("http://localhost:8000/healthz")
    payload = {"message": "hi", "user_id": "testUser"}
    r = requests.post("http://localhost:8000/v1/chat", json=payload, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["response"].startswith("Echo")
    assert data["conversation_id"]
