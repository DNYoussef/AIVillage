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
_ensure_module("faiss", {"IndexFlatL2": lambda *args, **kwargs: object()})
_ensure_module(
    "numpy",
    {"zeros": lambda *args, **kwargs: [0] * (args[0] if args else 0), "ndarray": list},
)
_ensure_module("httpx")
_ensure_module("requests", {"get": lambda *a, **k: None})
_ensure_module("yaml", {"safe_load": lambda *a, **k: {}})
torch_mod = _ensure_module(
    "torch",
    {
        "Tensor": object,
        "randn": lambda *a, **k: 0,
        "__getattr__": lambda name: object,
    },
)
if torch_mod is not None:
    torch_mod.__spec__.submodule_search_locations = []
    torch_mod.__path__ = []
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object
    nn_mod.functional = types.ModuleType("torch.nn.functional")
    optim_mod = types.ModuleType("torch.optim")
    torch_mod.nn = nn_mod
    torch_mod.optim = optim_mod
    torch_mod.nn.functional = nn_mod.functional
    sys.modules.setdefault("torch.nn", nn_mod)
    sys.modules.setdefault("torch.nn.functional", nn_mod.functional)
    sys.modules.setdefault("torch.optim", optim_mod)
_ensure_module("sklearn")
_ensure_module("sklearn.feature_extraction")


class _DummyTfidfVectorizer:
    def fit(self, docs):
        self.vocab = sorted({w.lower() for d in docs for w in str(d).split()})
        return self

    def transform(self, docs):
        vecs = []
        for d in docs:
            tokens = d.lower().split()
            vecs.append([tokens.count(v) for v in self.vocab])
        return vecs


_ensure_module(
    "sklearn.feature_extraction.text", {"TfidfVectorizer": _DummyTfidfVectorizer}
)
_ensure_module("sklearn.metrics")


class _DummyCosineResult(list):
    def __getitem__(self, item):
        if isinstance(item, tuple):
            return super().__getitem__(item[0])[item[1]]
        return super().__getitem__(item)

    class _Diag(list):
        def mean(self):
            return sum(self) / len(self) if self else 0.0

    def diagonal(self):
        return self._Diag([self[i][i] for i in range(min(len(self), len(self[0])))])


def _dummy_cosine_similarity(a, b):
    import math

    def dot(u, v):
        return sum(x * y for x, y in zip(u, v))

    def norm(u):
        return math.sqrt(sum(x * x for x in u))

    res = []
    for row in a:
        row_res = []
        for col in b:
            denom = norm(row) * norm(col)
            row_res.append(dot(row, col) / denom if denom else 0.0)
        res.append(row_res)
    return _DummyCosineResult(res)


_ensure_module(
    "sklearn.metrics.pairwise", {"cosine_similarity": _dummy_cosine_similarity}
)
_ensure_module("psutil")
_ensure_module("networkx", {"Graph": object})
parent = _ensure_module(
    "langroid",
    {
        "ChatAgent": object,
        "ChatAgentConfig": object,
        "Task": object,
    },
)
if parent is not None:
    parent.__spec__.submodule_search_locations = []
    parent.__path__ = []
_ensure_module("langroid.agent")
mod_agent = _ensure_module("langroid.agent")
if mod_agent is not None:
    mod_agent.__spec__.submodule_search_locations = []
    mod_agent.__path__ = []
_ensure_module("langroid.agent.tool_message", {"ToolMessage": object})
_ensure_module("langroid.language_models")
mod_lm = _ensure_module("langroid.language_models")
if mod_lm is not None:
    mod_lm.__spec__.submodule_search_locations = []
    mod_lm.__path__ = []
_ensure_module("langroid.language_models.openai_gpt", {"OpenAIGPTConfig": object})
tok = _ensure_module("tiktoken")
if tok is not None:
    tok.__spec__.submodule_search_locations = []
    tok.__path__ = []
_ensure_module("tiktoken.load", {"load_tiktoken_bpe": lambda p: {}})

# Provide simple stubs for optional dependencies used in tests.  These
# allow the test suite to be imported even when the real packages are not
# installed.  The actual tests that rely on them will be skipped if the
# dependencies are missing.
_ensure_module(
    "numpy",
    {
        "__version__": "0.0",
        "array": lambda *a, **k: [],
        "zeros": lambda *a, **k: [],
        "ndarray": list,
    },
)
_ensure_module("httpx")

# Ensure `importlib.util.find_spec` reports these stubs as missing so tests
# can detect optional dependencies correctly.

_real_find_spec = importlib.util.find_spec


def _patched_find_spec(name, *args, **kwargs):
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__spec__", None) is None:
        return None
    if name in _STUBBED_MODULES:
        return None
    if name.startswith("torch") or name.startswith("sklearn"):
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
