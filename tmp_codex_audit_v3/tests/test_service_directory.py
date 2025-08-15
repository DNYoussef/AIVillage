import json

from communications.service_directory import ServiceDirectory


def test_register_and_lookup_persistence(tmp_path, monkeypatch):
    tmp_cache = tmp_path / "agents.json"
    monkeypatch.setattr("communications.service_directory._CACHE_PATH", tmp_cache)

    sd = ServiceDirectory()
    sd.register("agent1", "ws://localhost:9000/ws")
    assert json.loads(tmp_cache.read_text()) == {"agent1": "ws://localhost:9000/ws"}

    # New instance should load existing registration
    sd2 = ServiceDirectory()
    assert sd2.lookup("agent1") == "ws://localhost:9000/ws"


def test_lookup_fallback_env(tmp_path, monkeypatch):
    tmp_cache = tmp_path / "agents.json"
    monkeypatch.setattr("communications.service_directory._CACHE_PATH", tmp_cache)
    sd = ServiceDirectory()

    monkeypatch.delenv("COMM_DEFAULT_HOST", raising=False)
    monkeypatch.delenv("COMM_DEFAULT_PORT", raising=False)
    monkeypatch.setenv("COMM_DEFAULT_HOST", "example.com")
    monkeypatch.setenv("COMM_DEFAULT_PORT", "7777")

    assert sd.lookup("missing") == "ws://example.com:7777/ws"
