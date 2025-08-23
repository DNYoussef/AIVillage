import importlib
import sys
import warnings

from packages.core.legacy.compatibility.bridge_system import CompatibilityBridge


def test_import_bridge_redirects_and_warns(monkeypatch):
    bridge = CompatibilityBridge()
    bridge.allowed_modules.update({"legacy_math", "math"})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert bridge.create_import_bridge("legacy_math", "math")
        legacy_math = importlib.import_module("legacy_math")
        assert legacy_math.sqrt(4) == 2
        assert any(item.category is DeprecationWarning for item in w)

    monkeypatch.delitem(sys.modules, "legacy_math", raising=False)


def test_import_bridge_rejects_unlisted_modules():
    bridge = CompatibilityBridge()

    result = bridge.create_import_bridge("bad_mod", "math")
    assert result is False
    assert "bad_mod" not in sys.modules
