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
        mod.__path__ = []
        mod._is_stub = True
        sys.modules[name] = mod
        _STUBBED_MODULES.add(name)
        return mod
    return sys.modules.get(name)

# Stub faiss if missing
_ensure_module('faiss', {'IndexFlatL2': lambda *args, **kwargs: object()})
_ensure_module('numpy', {'zeros': lambda *args, **kwargs: [0] * (args[0] if args else 0)})
_ensure_module('httpx')
_ensure_module('torch', {
    'Tensor': object,
    'randn': lambda *a, **k: 0,
    '__getattr__': lambda name: object,
})
_ensure_module('sklearn')
_ensure_module('sklearn.feature_extraction')
_ensure_module('sklearn.feature_extraction.text', {'TfidfVectorizer': object})
_ensure_module('sklearn.metrics')
_ensure_module('sklearn.metrics.pairwise', {'cosine_similarity': lambda *a, **k: [[1.0]]})
_ensure_module('psutil')
_ensure_module('networkx', {'Graph': object})
parent = _ensure_module('langroid', {
    'ChatAgent': object,
    'ChatAgentConfig': object,
    'Task': object,
})
if parent is not None:
    parent.__spec__.submodule_search_locations = []
    parent.__path__ = []
_ensure_module('langroid.agent')
mod_agent = _ensure_module('langroid.agent')
if mod_agent is not None:
    mod_agent.__spec__.submodule_search_locations = []
    mod_agent.__path__ = []
_ensure_module('langroid.agent.tool_message', {'ToolMessage': object})
_ensure_module('langroid.language_models')
mod_lm = _ensure_module('langroid.language_models')
if mod_lm is not None:
    mod_lm.__spec__.submodule_search_locations = []
    mod_lm.__path__ = []
_ensure_module('langroid.language_models.openai_gpt', {'OpenAIGPTConfig': object})
tok = _ensure_module('tiktoken')
if tok is not None:
    tok.__spec__.submodule_search_locations = []
    tok.__path__ = []
_ensure_module('tiktoken.load', {'load_tiktoken_bpe': lambda p: {}})

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
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__spec__", None) is None:
        return None
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
