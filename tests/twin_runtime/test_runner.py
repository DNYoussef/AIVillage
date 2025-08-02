import sys
import types
from pathlib import Path

import pytest


class DummyPersistentClient:
    def __init__(self, path):
        self.path = path

    def get_or_create_collection(self, name):
        class _Coll:
            def query(self, query_texts, n_results):
                return {"documents": [[]]}

        return _Coll()


class DummyLoraModel:
    @staticmethod
    def from_pretrained(*_, **__):
        class _Tmp:
            def merge_and_unload(self):
                return None

        return _Tmp()


@pytest.fixture(autouse=True)
def patch_modules(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "src"))
    monkeypatch.setitem(sys.modules, "chromadb", types.SimpleNamespace(PersistentClient=DummyPersistentClient))
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(LoraModel=DummyLoraModel))
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=lambda *a, **k: None))

    class DummyConfig:
        pass

    class DummyModel:
        def __init__(self, config=None):
            self.config = config

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(LlamaConfig=DummyConfig, LlamaForCausalLM=DummyModel),
    )

    # ensure module is re-imported each test
    sys.modules.pop("twin_runtime.runner", None)


def test_load_compressed_success(monkeypatch, tmp_path):
    cmp = tmp_path / "model.cmp"
    cmp.write_text("x")

    sentinel = object()

    class DummyLoader:
        def __init__(self, factory, path):
            assert path == str(cmp)

        def assemble_model(self):
            return sentinel

    monkeypatch.setitem(
        sys.modules,
        "twin_runtime.compressed_loader",
        types.SimpleNamespace(CompressedModelLoader=DummyLoader),
    )

    called = False

    def fake_llama(*args, **kwargs):  # pragma: no cover - should not be called
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr("llama_cpp.Llama", fake_llama)

    monkeypatch.setenv("TWIN_COMPRESSED", str(cmp))
    monkeypatch.setenv("TWIN_MODEL", str(tmp_path / "model.gguf"))
    monkeypatch.setenv("TWIN_HOME", str(tmp_path))

    import twin_runtime.runner as runner

    assert runner.LLM is sentinel
    assert not called


def test_load_compressed_failure(monkeypatch, tmp_path, caplog):
    cmp = tmp_path / "model.cmp"
    cmp.write_text("x")

    class DummyLoader:
        def __init__(self, *_, **__):
            raise RuntimeError("boom")

    monkeypatch.setitem(
        sys.modules,
        "twin_runtime.compressed_loader",
        types.SimpleNamespace(CompressedModelLoader=DummyLoader),
    )

    sentinel = object()

    def fake_llama(model_path, n_ctx, n_threads):
        return sentinel

    monkeypatch.setattr("llama_cpp.Llama", fake_llama)

    monkeypatch.setenv("TWIN_COMPRESSED", str(cmp))
    monkeypatch.setenv("TWIN_MODEL", str(tmp_path / "model.gguf"))
    monkeypatch.setenv("TWIN_HOME", str(tmp_path))

    caplog.set_level("ERROR")
    import twin_runtime.runner as runner

    assert runner.LLM is sentinel
    assert any("Failed to load compressed model" in rec.message for rec in caplog.records)

