import sys
import types
import importlib.util
import importlib.machinery
import pytest
import contextlib

_STUBBED_MODULES = set()

# Provide lightweight stubs for heavy optional dependencies so test collection
# succeeds even if they are not installed in the environment.

def _ensure_module(name: str, attrs: dict | None = None):
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
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
_ensure_module('httpx', {'BaseTransport': object})
_ensure_module('yaml', {'safe_load': lambda *a, **k: {}, 'safe_dump': lambda *a, **k: ''})
class _SimpleGraph:
    def __init__(self):
        self._nodes = {}
        self._edges = []

    class _NodeView(list):
        def __init__(self, nodes):
            self._nodes = nodes

        def __iter__(self):
            return iter(self._nodes)

        def __call__(self, data=False):
            if data:
                return [(n, a) for n, a in self._nodes.items()]
            return list(self._nodes)

    class _EdgeView(list):
        def __init__(self, edges):
            self._edges = edges

        def __iter__(self):
            return iter((u, v) for u, v, _ in self._edges)

        def __call__(self, data=False):
            if data:
                return [(u, v, a) for u, v, a in self._edges]
            return [(u, v) for u, v, _ in self._edges]

    def add_node(self, node, **attrs):
        self._nodes[node] = attrs

    @property
    def nodes(self):
        return self._NodeView(self._nodes)

    def add_edge(self, u, v, **attrs):
        self._edges.append((u, v, attrs))

    @property
    def edges(self):
        return self._EdgeView(self._edges)

    def has_node(self, node):
        return node in self._nodes

    def neighbors(self, node):
        return [v for u, v, _ in self._edges if u == node] + [u for u, v, _ in self._edges if v == node]

    def number_of_nodes(self):
        return len(self._nodes)

_ensure_module('networkx', {'Graph': _SimpleGraph})
_ensure_module('tiktoken.load', {'load_tiktoken_bpe': lambda *a, **k: {}})
_ensure_module('tiktoken', {'Encoding': type('Encoding', (), {})})

# Provide simple stubs for optional dependencies used in tests.  These
# allow the test suite to be imported even when the real packages are not
# installed.  The actual tests that rely on them will be skipped if the
# dependencies are missing.
_ensure_module('numpy', {
    '__version__': '0.0',
    'array': lambda *a, **k: [],
    'zeros': lambda *a, **k: []
})
_ensure_module('httpx', {'BaseTransport': object})

# Ensure `importlib.util.find_spec` reports these stubs as missing so tests
# can detect optional dependencies correctly.
_real_find_spec = importlib.util.find_spec

def _patched_find_spec(name, *args, **kwargs):
    if name in _STUBBED_MODULES:
        return None
    return _real_find_spec(name, *args, **kwargs)

importlib.util.find_spec = _patched_find_spec



def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked as requires_gpu",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_gpu: tests that need heavy GPU dependent packages"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_gpu = pytest.mark.skip(reason="requires GPU-heavy dependencies")
    for item in items:
        if "requires_gpu" in item.keywords:
            item.add_marker(skip_gpu)


def pytest_ignore_collect(path, config):
    if config.getoption("--runslow"):
        return False
    pstr = str(path)
    heavy = ["agent_forge/evomerge", "agents/king/tests", "tests/test_king_agent.py"]
    if any(h in pstr for h in heavy):
        return True
    return False


