"""
Mock modules for testing when dependencies are unavailable.
"""

import importlib
import sys
from unittest.mock import MagicMock


def install_mocks() -> None:
    """Install mock modules into ``sys.modules`` when imports fail.

    The test suite patches optional dependencies so tests can run without
    the real implementations.  However, if the real packages are available
    we should prefer them over mocks to avoid hiding import errors.  We
    therefore attempt to import the modules first and only register mocks
    when the import fails.
    """

    # Mock rag_system only if it's truly unavailable
    try:
        importlib.import_module("rag_system")
    except Exception:  # pragma: no cover - import failure fallback
        if "rag_system" not in sys.modules:
            sys.modules["rag_system"] = MagicMock()

    # Ensure required rag_system submodules exist
    for mod in [
        "rag_system.core.config",
        "rag_system.core.pipeline",
        "rag_system.retrieval.vector_store",
        "rag_system.tracking.unified_knowledge_tracker",
        "rag_system.utils.error_handling",
    ]:
        try:
            importlib.import_module(mod)
        except Exception:  # pragma: no cover - import failure fallback
            sys.modules.setdefault(mod, MagicMock())

    # Mock services only if they're missing
    try:
        importlib.import_module("services")
    except Exception:  # pragma: no cover - import failure fallback
        if "services" not in sys.modules:
            sys.modules["services"] = MagicMock()
            sys.modules["services.gateway"] = MagicMock()
            sys.modules["services.twin"] = MagicMock()

    # Mock missing agent_forge submodules
    try:
        importlib.import_module("agent_forge.memory_manager")
    except Exception:  # pragma: no cover - import failure fallback
        sys.modules.setdefault("agent_forge.memory_manager", MagicMock())

    try:
        importlib.import_module("agent_forge.wandb_manager")
    except Exception:  # pragma: no cover - import failure fallback
        sys.modules.setdefault("agent_forge.wandb_manager", MagicMock())


# Auto-install mocks when imported
install_mocks()
