import importlib
import inspect
import sys
import types
import unittest

if importlib.util.find_spec("httpx") is None:
    msg = "Required dependency not installed"
    raise unittest.SkipTest(msg)

import httpx

if "app" not in inspect.signature(httpx.Client).parameters:
    msg = "httpx lacks TestClient app support"
    raise unittest.SkipTest(msg)

from unittest.mock import patch

from fastapi.testclient import TestClient


class DummyMetric:
    def labels(self, **_):
        return self

    def inc(self, *_, **__):
        pass

    def observe(self, *_, **__):
        pass


def test_explain_endpoint(monkeypatch):
    monkeypatch.setenv("CALIBRATION_ENABLED", "0")
    fake_cachetools = types.ModuleType("cachetools")
    fake_cachetools.LRUCache = lambda *a, **k: {}
    fake_prom = types.ModuleType("prometheus_client")
    fake_prom.Counter = lambda *a, **k: DummyMetric()
    fake_prom.Histogram = lambda *a, **k: DummyMetric()
    fake_prom.generate_latest = lambda: b""
    fake_nx = types.ModuleType("networkx")

    class G:
        def __init__(self):
            self.edges = {"A": {"B": {"relation": "r"}}}

        def add_edge(self, s, t, relation=None):
            self.edges.setdefault(s, {})[t] = {"relation": relation}

        def __contains__(self, item):
            return item in self.edges

        def __getitem__(self, item):
            return self.edges[item]

        def number_of_nodes(self):
            return len(self.edges)

    fake_nx.DiGraph = G

    def shortest_path(g, s, t):
        return [s, t]

    fake_nx.shortest_path = shortest_path
    fake_nx.NetworkXNoPath = Exception

    modules = {
        "cachetools": fake_cachetools,
        "networkx": fake_nx,
        "prometheus_client": fake_prom,
    }
    with patch.dict(sys.modules, modules):
        if "services.twin.app" in sys.modules:
            del sys.modules["services.twin.app"]
        mod = importlib.import_module("services.twin.app")
        client = TestClient(mod.app)
        resp = client.post("/explain", json={"src": "A", "dst": "B"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["found"] is True
        assert body["hops"] == 1
        assert body["path"][0]["s"] == "A"
        assert body["path"][0]["t"] == "B"
