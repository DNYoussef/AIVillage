import importlib.util
import sys
from pathlib import Path

import pytest


def test_pretrain_three_models_missing_dependencies():
    """The pretrain script should raise a helpful ImportError when deps are missing."""
    script_path = Path("core/agent_forge/phases/cognate_pretrain/pretrain_three_models.py")

    original_path = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("pretrain_three_models", script_path)
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(ImportError) as excinfo:
            spec.loader.exec_module(module)
        assert "install the agent-forge training components" in str(excinfo.value).lower()
    finally:
        sys.path[:] = original_path
