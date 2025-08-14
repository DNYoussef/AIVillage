import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def setup_logging(
    *,
    log_file: str | None = None,
    log_level: int | str = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure application-wide logging.

    Parameters
    ----------
    log_file:
        Optional path to a log file. If provided, a rotating file handler will be
        attached that rolls over at ``max_bytes`` with ``backup_count`` backups.
    log_level:
        Logging level or level name (e.g. ``logging.INFO`` or ``"INFO"``).
    fmt:
        Formatter string for log records.
    max_bytes:
        Maximum size in bytes before rotating the log file.
    backup_count:
        Number of backup files to keep when rotating.

    Returns
    -------
    logging.Logger
        The root logger configured with the specified handlers.
    """
    level = getattr(logging, log_level.upper()) if isinstance(log_level, str) else log_level

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


__all__ = ["setup_logging"]
