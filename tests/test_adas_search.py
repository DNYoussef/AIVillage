import importlib.util
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("adas_system", repo_root / "agent_forge" / "adas" / "system.py")
adas_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adas_module)
adaptive_search = adas_module.adaptive_search


def test_adaptive_search_returns_best():
    space = [1, 2, 3]
    best = adaptive_search(lambda x: -x, space)
    assert best == 1
