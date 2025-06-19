import sys
import types
import importlib.util
import importlib.machinery
import pytest

_STUBBED_MODULES = set()

# Provide lightweight stubs for heavy optional dependencies so test collection
# succeeds even if they are not installed in the environment.

def _ensure_module(name: str, attrs: dict | None = None):
    spec = importlib.util.find_spec(name)
    if spec is None and name not in sys.modules:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        mod._is_stub = True
        sys.modules[name] = mod
        _STUBBED_MODULES.add(name)
        return mod
    return sys.modules.get(name)

# Stub faiss if missing
_ensure_module('faiss', {'IndexFlatL2': lambda *args, **kwargs: object()})
_ensure_module('numpy', {'zeros': lambda *args, **kwargs: [0] * (args[0] if args else 0)})
_ensure_module('httpx')

# Provide simple stubs for optional dependencies used in tests.  These
# allow the test suite to be imported even when the real packages are not
# installed.  The actual tests that rely on them will be skipped if the
# dependencies are missing.
_ensure_module('numpy', {
    '__version__': '0.0',
    'array': lambda *a, **k: [],
    'zeros': lambda *a, **k: []
})
_ensure_module('httpx')

# Ensure `importlib.util.find_spec` reports these stubs as missing so tests
# can detect optional dependencies correctly.
_real_find_spec = importlib.util.find_spec

def _patched_find_spec(name, *args, **kwargs):
    if name in _STUBBED_MODULES:
        return None
    return _real_find_spec(name, *args, **kwargs)

importlib.util.find_spec = _patched_find_spec



def pytest_collection_modifyitems(config, items):
    """Skip tests requiring heavy ML dependencies if they aren't available."""
    missing = []
    for dep in ("torch", "sklearn", "psutil", "numpy", "httpx"):
        try:
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)
        except ValueError:
            missing.append(dep)
    if not missing:
        return
    skip_marker = pytest.mark.skip(reason=f"Missing dependencies: {', '.join(missing)}")
    for item in items:
        path = str(item.fspath)
        if "agent_forge/evomerge" in path or "agents/king/tests" in path or path.endswith("test_king_agent.py"):
            item.add_marker(skip_marker)


def pytest_ignore_collect(path, config):
    heavy_dirs = ["agent_forge/evomerge", "agents/king/tests", "tests/test_king_agent.py"]
    pstr = str(path)
    if any(h in pstr for h in heavy_dirs):
        missing = []
        for dep in ("torch", "sklearn", "psutil", "numpy", "httpx"):
            try:
                if importlib.util.find_spec(dep) is None:
                    missing.append(dep)
            except ValueError:
                missing.append(dep)
        if missing:
            print(f"Skipping {pstr} due to missing dependencies: {', '.join(missing)}")
            return True
    return False
