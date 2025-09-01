"""Model management utilities for Agent Forge.
This module provides the ``SimpleModelManager`` class originally defined in
``start_agent_forge.py`` so it can be reused across the codebase.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SimpleModelManager:
    """Simplified model manager for production pipeline."""

    def __init__(self, base_dir: str | Path = "D:/AIVillage/models", max_models: int = 8):
        self.base_dir = Path(base_dir)
        self.max_models = max_models
        self.models: dict[str, dict[str, Any]] = {}

        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Model storage: %s", self.base_dir)

        # Check available space
        self._check_storage_space()

        # Load existing models
        self._load_existing_models()

    def _check_storage_space(self) -> None:
        """Check available storage space."""
        try:
            free_bytes = shutil.disk_usage(self.base_dir).free
            free_gb = free_bytes / (1024**3)
            logger.info("Available storage: %.1f GB", free_gb)

            if free_gb < 15:
                logger.warning("Low storage space - models may not download properly")

        except Exception as e:  # pragma: no cover - best effort logging
            logger.warning("Could not check storage: %s", e)

    def _load_existing_models(self) -> None:
        """Load information about existing models."""
        try:
            for model_dir in self.base_dir.iterdir():
                if model_dir.is_dir():
                    # Extract model ID from directory name
                    model_id = model_dir.name.replace("_", "/")
                    if model_id.count("/") == 1:  # Valid model ID format
                        size_gb = self._get_directory_size(model_dir) / (1024**3)
                        self.models[model_id] = {
                            "path": str(model_dir),
                            "size_gb": size_gb,
                            "downloaded": True,
                        }
                        logger.info(
                            "Found existing model: %s (%.1f GB)",
                            model_id,
                            size_gb,
                        )

            logger.info("Loaded %d existing models", len(self.models))

        except Exception as e:  # pragma: no cover - best effort logging
            logger.error("Error loading existing models: %s", e)

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total += file_path.stat().st_size
        except Exception as e:  # pragma: no cover - best effort
            logger.debug(f"Failed to calculate cache size for path {self.cache_dir}: {e}")
        return total

    async def download_model(self, model_spec: dict[str, Any]) -> bool:
        """Download a model using ``huggingface-cli``."""
        model_id = model_spec["model_id"]

        # Check if already downloaded
        if model_id in self.models:
            logger.info("Model %s already downloaded", model_id)
            return True

        # Create safe directory name
        safe_name = model_id.replace("/", "_").replace(":", "_")
        model_dir = self.base_dir / safe_name

        logger.info("Downloading %s to %s", model_id, model_dir)

        try:
            # Use huggingface-cli for downloading
            cmd = [
                "huggingface-cli",
                "download",
                model_id,
                "--local-dir",
                str(model_dir),
                "--local-dir-use-symlinks",
                "False",
            ]

            logger.info("Running: %s", " ".join(cmd))

            # Run download command
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            stdout, stderr = process.communicate()

            if process.returncode == 0:
                # Calculate size
                size_gb = self._get_directory_size(model_dir) / (1024**3)

                # Update registry
                self.models[model_id] = {
                    "path": str(model_dir),
                    "size_gb": size_gb,
                    "downloaded": True,
                    "purpose": model_spec.get("purpose"),
                }

                logger.info("Successfully downloaded %s (%.1f GB)", model_id, size_gb)
                return True
            else:  # pragma: no cover - external command
                logger.error("Download failed for %s", model_id)
                logger.error("Error: %s", stderr)
                return False

        except FileNotFoundError:  # pragma: no cover - environment setup
            logger.error("huggingface-cli not found - installing huggingface_hub")

            # Try to install huggingface_hub
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "huggingface_hub"],
                    check=True,
                )
                logger.info("Installed huggingface_hub - please restart the script")
                return False
            except subprocess.CalledProcessError:  # pragma: no cover
                logger.error("Failed to install huggingface_hub")
                return False

        except Exception as e:  # pragma: no cover - best effort logging
            logger.error("Error downloading %s: %s", model_id, e)
            return False

    def cleanup_old_models(self) -> None:
        """Remove old models if we exceed ``max_models`` limit."""
        if len(self.models) <= self.max_models:
            return

        logger.info("Cleaning up models: %d > %d", len(self.models), self.max_models)

        # Sort by size (remove largest first to save space quickly)
        models_by_size = sorted(self.models.items(), key=lambda x: x[1].get("size_gb", 0), reverse=True)

        to_remove = models_by_size[self.max_models :]

        for model_id, model_info in to_remove:
            try:
                model_path = Path(model_info["path"])
                if model_path.exists():
                    shutil.rmtree(model_path)
                    logger.info("Removed %s", model_id)

                del self.models[model_id]

            except Exception as e:  # pragma: no cover - best effort
                logger.error("Failed to remove %s: %s", model_id, e)


__all__ = ["SimpleModelManager"]
