"""
Mock modules for testing when dependencies are unavailable.
"""

import sys
from unittest.mock import MagicMock


def install_mocks():
    """Install mock modules into sys.modules."""
    # Mock rag_system if not available
    if "rag_system" not in sys.modules:
        sys.modules["rag_system"] = MagicMock()
        sys.modules["rag_system.pipeline"] = MagicMock()

    # Mock services if not available
    if "services" not in sys.modules:
        sys.modules["services"] = MagicMock()
        sys.modules["services.gateway"] = MagicMock()
        sys.modules["services.twin"] = MagicMock()


# Auto-install mocks when imported
install_mocks()
