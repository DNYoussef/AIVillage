"""Shared logging utilities for AIVillage.

Provides a configurable ``setup_logging`` helper that supports both
console and rotating file handlers. The implementation is based on the
most feature-rich configuration previously used in the project and is
intended to be the single source of truth for logging setup.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union


def setup_logging(
    level: Union[int, str] = logging.INFO,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: str | Path | None = None,
    include_console: bool = True,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure the root logger.

    Args:
        level: Logging level as ``int`` or string name.
        format_string: Format string for log messages.
        log_file: Optional path to a log file. If provided, a rotating file
            handler is configured.
        include_console: Whether to output logs to the console.
        max_bytes: Maximum file size before rotation occurs.
        backup_count: Number of rotated log files to keep.

    Returns:
        The configured root logger.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(format_string)

    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
