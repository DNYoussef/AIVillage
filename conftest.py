import sys
import sys
import types
import importlib.util
import importlib.machinery
import pytest

# Provide lightweight stubs for heavy optional dependencies so test collection
# succeeds even if they are not installed in the environment.

def _ensure_module(name: str, attrs: dict | None = None):
    if importlib.util.find_spec(name) is None and name not in sys.modules:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = mod
        return mod
    return sys.modules.get(name)

# Stub faiss if missing
_ensure_module('faiss', {'IndexFlatL2': lambda *args, **kwargs: object()})



def pytest_collection_modifyitems(config, items):
    """Skip tests requiring heavy ML dependencies if they aren't available."""
    missing = []
    for dep in ("torch", "sklearn", "psutil"):
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
        print('ignore_collect check', pstr)
        for dep in ("torch", "sklearn", "psutil"):
            try:
                if importlib.util.find_spec(dep) is None:
                    print('missing', dep)
                    return True
            except ValueError:
                print('error spec', dep)
                return True
    return False
