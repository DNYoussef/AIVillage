"""Stub implementation for rag_system logging.
This is a placeholder to fix test infrastructure.
"""

import logging
import warnings

warnings.warn(
    "rag_system.utils.logging is a stub implementation. "
    "Replace with actual implementation before production use.",
    UserWarning,
    stacklevel=2
)

def setup_logger(name: str = "rag_system", level: str = "INFO"):
    """Setup a basic logger for testing."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Alias for compatibility
get_logger = setup_logger
