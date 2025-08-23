import importlib.util

import pytest
from chromadb import PersistentClient
from ingestion import run_all
from ingestion.vector_ds import personal_ds

torch_spec = importlib.util.find_spec("torch.utils")
if torch_spec is None:
    pytest.skip("torch not installed", allow_module_level=True)


def test_ingest_noop(tmp_path):
    cli = PersistentClient(path=tmp_path)
    assert run_all("dummy", cli) >= 0
    ds = personal_ds("dummy")
    assert len(ds) >= 0
