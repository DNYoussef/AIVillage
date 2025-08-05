import builtins
import importlib.util
from pathlib import Path
import sys

# Path to metrics module
ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "src" / "mcp_servers" / "hyperag" / "guardian" / "metrics.py"


def test_mock_metric_records_calls(monkeypatch):
    """Ensure MockMetric tracks metric operations when Prometheus is unavailable."""
    # Force prometheus_client import to fail
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "prometheus_client":
            raise ImportError("prometheus_client not installed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "prometheus_client", raising=False)

    spec = importlib.util.spec_from_file_location("guardian_metrics", METRICS_PATH)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    mock_cls = metrics.MockMetric

    metric = mock_cls("test_metric")
    metric.inc()
    metric.observe(1.0)
    metric.set(5)
    metric.info({"foo": "bar"})

    assert metric.calls["inc"] == 1
    assert metric.calls["observe"] == 1
    assert metric.calls["set"] == 1
    assert metric.calls["info"] == 1

    assert metric.counters[frozenset()] == 1
    assert metric.values == [1.0]
    assert metric.gauge_value == 5.0
    assert metric.info_data["foo"] == "bar"
