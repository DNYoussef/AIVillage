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

# Create a more complete torch stub
class MockTensor:
    def __init__(self, *args, **kwargs):
        self.shape = (10, 10) if not args else args
    def __getattr__(self, name):
        return lambda *a, **k: self
    def item(self):
        return 0.5
    def numpy(self):
        import numpy as np
        return np.zeros(self.shape)
    def to(self, *args, **kwargs):
        return self
    def cpu(self):
        return self
    def detach(self):
        return self

torch_mod = _ensure_module(
    "torch",
    {
        "Tensor": MockTensor,
        "randn": lambda *a, **k: MockTensor(*a, **k),
        "zeros": lambda *a, **k: MockTensor(*a, **k),
        "ones": lambda *a, **k: MockTensor(*a, **k),
        "tensor": lambda *a, **k: MockTensor(*a, **k),
        "rand": lambda *a, **k: MockTensor(*a, **k),
        "randn_like": lambda *a, **k: MockTensor(*a, **k),
        "float32": "float32",
        "int8": "int8",
        "norm": lambda *a, **k: MockTensor(),
        "mean": lambda *a, **k: MockTensor(),
        "stack": lambda *a, **k: MockTensor(),
        "cat": lambda *a, **k: MockTensor(),
    },
)
if torch_mod is not None:
    torch_mod.__spec__.submodule_search_locations = []
    torch_mod.__path__ = []
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object
    torch_mod.nn = nn_mod
    sys.modules.setdefault("torch.nn", nn_mod)

    # Add cuda module with is_available function
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: False  # Stub returns False for no GPU
    torch_mod.cuda = cuda_mod
    sys.modules.setdefault("torch.cuda", cuda_mod)

_ensure_module("sklearn")
_ensure_module("sklearn.feature_extraction")
_ensure_module("sklearn.feature_extraction.text", {"TfidfVectorizer": object})
_ensure_module("sklearn.metrics")
_ensure_module("sklearn.metrics.pairwise", {"cosine_similarity": lambda a, b: []})
_ensure_module("sklearn.linear_model", {"LogisticRegression": object})
_ensure_module("sklearn.ensemble", {"RandomForestClassifier": object})
_ensure_module("sklearn.model_selection", {"train_test_split": lambda *args, **kwargs: ([], [], [], [])})
_ensure_module("sklearn.preprocessing", {"StandardScaler": object})

# Add grokfast stub
class AugmentedAdam:
    def __init__(self, params, lr=1e-3, slow_freq=0.08, boost=1.5, **kwargs):
        self.params = list(params)
        self.lr = lr
        self.slow_freq = slow_freq
        self.boost = boost
        self._slow_cache = {}
    def step(self): pass
    def zero_grad(self): pass

_ensure_module("grokfast", {"AugmentedAdam": AugmentedAdam})

# Add numba stub
def jit(*args, **kwargs):
    """Stub jit decorator that returns the function unchanged."""
    if len(args) == 1 and callable(args[0]):
        return args[0]  # Function was passed directly
    return lambda func: func  # Decorator with arguments

_ensure_module("numba", {"jit": jit})

# Add tiktoken stub
class MockEncoding:
    def encode(self, text, *args, **kwargs):
        return [1, 2, 3]  # Mock token ids
    def decode(self, tokens, *args, **kwargs):
        return "mock decoded text"

_ensure_module("tiktoken", {
    "encoding_for_model": lambda model: MockEncoding(),
    "get_encoding": lambda encoding: MockEncoding(),
})

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
