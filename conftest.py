import sys
import types
import importlib.util
import importlib.machinery
import pytest

_STUBBED_MODULES = set()


def _ensure_module(name: str, attrs: dict | None = None):
    spec = importlib.util.find_spec(name)
    if spec is None and name not in sys.modules:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__path__ = []
        mod._is_stub = True
        sys.modules[name] = mod
        _STUBBED_MODULES.add(name)
        return mod
    return sys.modules.get(name)


# Minimal stubs for heavy optional dependencies
_ensure_module("faiss", {"IndexFlatL2": lambda *a, **k: object()})

torch_mod = _ensure_module(
    "torch",
    {
        "Tensor": object,
        "randn": lambda *a, **k: 0,
    },
)
if torch_mod is not None:
    torch_mod.__spec__.submodule_search_locations = []
    torch_mod.__path__ = []
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object
    torch_mod.nn = nn_mod
    sys.modules.setdefault("torch.nn", nn_mod)

_ensure_module("sklearn")
_ensure_module("sklearn.feature_extraction")
_ensure_module("sklearn.feature_extraction.text", {"TfidfVectorizer": object})
_ensure_module("sklearn.metrics")
_ensure_module("sklearn.metrics.pairwise", {"cosine_similarity": lambda a, b: []})

_real_find_spec = importlib.util.find_spec


def _patched_find_spec(name, *args, **kwargs):
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__spec__", None) is None:
        return None
    if name in _STUBBED_MODULES:
        return None
    return _real_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _patched_find_spec


def _deps_missing(*deps):
    missing = []
    for dep in deps:
        try:
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)
        except ValueError:
            missing.append(dep)
    return missing


def pytest_collection_modifyitems(config, items):
    missing = _deps_missing("torch", "sklearn", "faiss")
    if not missing:
        return
    skip_marker = pytest.mark.skip(reason=f"Missing dependencies: {', '.join(missing)}")
    for item in items:
        path = str(item.fspath)
        if (
            "agent_forge/evomerge" in path
            or "agents/king/tests" in path
            or path.endswith("test_king_agent.py")
        ):
            item.add_marker(skip_marker)


def pytest_ignore_collect(path, config):
    heavy_dirs = [
        "agent_forge/evomerge",
        "agents/king/tests",
        "tests/test_king_agent.py",
        "tests/test_train_level.py",
    ]
    special_tests = {
        "tests/test_twin_api.py": ["requests"],
        "tests/test_rate_limit.py": ["requests"],
        "tests/privacy/test_property_deletion.py": ["requests", "hypothesis"],
        "tests/agents/test_evidence_flow.py": ["bs4", "transformers"],
    }
    pstr = str(path)
    if any(h in pstr for h in heavy_dirs):
        if _deps_missing("torch", "sklearn", "faiss"):
            print(f"Skipping {pstr} due to missing dependencies")
            return True
    for tpath, deps in special_tests.items():
        if pstr.endswith(tpath):
            if _deps_missing(*deps):
                print(f"Skipping {pstr} due to missing dependencies")
                return True
    return False
