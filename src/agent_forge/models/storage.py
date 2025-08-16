"""Utilities for managing local model storage."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def cleanup_storage(models_dir: Path | str, max_models: int = 8) -> list[Path]:
    """Ensure that no more than ``max_models`` reside in ``models_dir``.

    The directories are sorted by modification time; the oldest ones beyond the
    limit are removed.  A list of deleted paths is returned for logging.
    """

    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    if len(dirs) <= max_models:
        return []

    # Sort by modification time (oldest first)
    dirs.sort(key=lambda d: d.stat().st_mtime)
    to_delete: Iterable[Path] = dirs[: len(dirs) - max_models]

    deleted: list[Path] = []
    for d in to_delete:
        try:
            shutil.rmtree(d)
            deleted.append(d)
            logger.info("Removed old model directory: %s", d)
        except OSError as exc:  # pragma: no cover - logging only
            logger.warning("Failed to remove %s: %s", d, exc)
    return deleted
