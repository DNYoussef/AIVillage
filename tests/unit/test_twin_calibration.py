import asyncio
import importlib
import sys
import types
from unittest.mock import patch

from services.twin.schemas import ChatRequest


def test_calibrated_prob_feature(monkeypatch):
    monkeypatch.setenv("CALIBRATION_ENABLED", "1")
    if "services.twin.app" in importlib.sys.modules:
        del importlib.sys.modules["services.twin.app"]

    fake_cachetools = types.ModuleType("cachetools")
    fake_cachetools.LRUCache = lambda *a, **k: {}
    fake_nx = types.ModuleType("networkx")
    fake_prom = types.ModuleType("prometheus_client")
    fake_prom.Counter = lambda *a, **k: lambda: None
    fake_prom.Histogram = lambda *a, **k: lambda: None
    fake_prom.generate_latest = lambda: b""

    class _Graph:
        def __init__(self):
            self.edges = {}

        def add_edge(self, s, t, relation=None):
            self.edges.setdefault(s, {})[t] = {"relation": relation}

        def __getitem__(self, item):
            return self.edges[item]

        def number_of_nodes(self):
            return len(self.edges)

    fake_nx.DiGraph = _Graph

    def shortest_path(g, s, t):
        return [s, t]

    fake_nx.shortest_path = shortest_path

    with patch.dict(
        sys.modules,
        {
            "cachetools": fake_cachetools,
            "networkx": fake_nx,
            "prometheus_client": fake_prom,
        },
    ):
        mod = importlib.import_module("services.twin.app")
    agent = mod.TwinAgent(mod.settings.model_path)
    req = ChatRequest(message="hello", user_id="u1")
    out = asyncio.run(agent.chat(req))
    assert out.calibrated_prob is not None
    assert 0.0 <= out.calibrated_prob <= 1.0
