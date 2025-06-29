import importlib.util
import pytest

if importlib.util.find_spec('plyer') is None:
    pytest.skip('plyer not installed', allow_module_level=True)

from twin_runtime.guard import risk_gate


def test_risk_gate():
    assert risk_gate({}, 0.1) == "allow"
