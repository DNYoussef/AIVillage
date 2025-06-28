import sys
import importlib.util
from pathlib import Path
from unittest.mock import patch
import pytest

try:
    import torch.nn  # noqa: F401
except Exception:
    pytest.skip("PyTorch not installed", allow_module_level=True)

repo_root = Path(__file__).resolve().parents[1]
path = repo_root / 'agent_forge' / 'training' / 'train_level.py'
spec = importlib.util.spec_from_file_location('agent_forge.training.train_level', path)
train_level = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_level
spec.loader.exec_module(train_level)

class TestTrainLevel:
    def test_invokes_components(self):
        with patch.object(train_level, 'run_level') as run, \
             patch.object(train_level, 'self_model_cycle') as cycle:
            train_level.train_level('ds', ['t'], 'm', {'s': 1})
            run.assert_called_once_with('ds')
            cycle.assert_called_once_with('m', ['t'], {'s': 1})
