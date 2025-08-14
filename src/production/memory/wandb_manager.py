#!/usr/bin/env python3
"""W&B Manager for Agent Forge.

Provides robust W&B integration with:
- Automatic authentication setup
- Graceful fallback for offline operation
- Connection testing and validation
- Error handling for network issues
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class WandBManager:
    """Manages W&B connection and initialization with robust error handling."""

    def __init__(self) -> None:
        self.run = None
        self.is_initialized = False
        self.offline_mode = False
        self.connection_tested = False

    def setup_authentication(self) -> bool:
        """Set up W&B authentication using multiple methods."""
        if not WANDB_AVAILABLE:
            logger.warning("W&B not installed. Running in offline mode.")
            self.offline_mode = True
            return False

        # Method 1: Check environment variable
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            logger.info("Found W&B API key in environment")
            return True

        # Method 2: Check existing login
        try:
            current_user = wandb.api.default_entity
            if current_user:
                logger.info(f"Already logged in to W&B as: {current_user}")
                return True
        except Exception:
            pass

        # Method 3: Check .netrc file
        netrc_path = Path.home() / ".netrc"
        if netrc_path.exists():
            try:
                with open(netrc_path) as f:
                    content = f.read()
                    if "api.wandb.ai" in content:
                        logger.info("Found W&B credentials in .netrc")
                        return True
            except Exception:
                pass

        logger.warning(
            "No W&B authentication found. See: https://docs.wandb.ai/quickstart"
        )
        logger.info("Options:")
        logger.info("1. Set WANDB_API_KEY environment variable")
        logger.info("2. Run 'wandb login' in terminal")
        logger.info("3. Continuing in offline mode...")

        self.offline_mode = True
        return False

    def test_connection(self) -> bool:
        """Test W&B connection."""
        if not WANDB_AVAILABLE or self.offline_mode:
            return False

        try:
            # Simple connection test
            api = wandb.Api(timeout=10)
            user = api.default_entity
            logger.info(f"W&B connection successful. User: {user}")
            self.connection_tested = True
            return True
        except Exception as e:
            logger.warning(f"W&B connection failed: {e}")
            logger.info("Switching to offline mode")
            self.offline_mode = True
            return False

    def initialize_run(
        self,
        project: str = "agent-forge",
        entity: str | None = None,
        job_type: str = "training",
        tags: list | None = None,
        config: dict | None = None,
        name: str | None = None,
    ) -> bool:
        """Initialize W&B run with error handling."""
        if not self.connection_tested:
            if not self.setup_authentication():
                return False
            if not self.test_connection():
                return False

        if self.offline_mode:
            logger.info("Running in offline mode - no W&B logging")
            return False

        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                job_type=job_type,
                tags=tags or [],
                config=config or {},
                name=name,
                reinit=True,  # Allow multiple runs
                settings=wandb.Settings(start_method="thread"),
            )

            self.is_initialized = True
            logger.info(f"W&B run initialized: {self.run.url}")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize W&B run: {e}")
            logger.info("Switching to offline mode")
            self.offline_mode = True
            return False

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to W&B with error handling."""
        if not self.is_initialized or self.offline_mode:
            # Log to local file instead
            self._log_to_file(metrics, step)
            return

        try:
            self.run.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to W&B: {e}")
            self._log_to_file(metrics, step)

    def log_artifact(
        self, artifact_path: str, artifact_name: str, artifact_type: str = "model"
    ) -> None:
        """Log artifact to W&B with error handling."""
        if not self.is_initialized or self.offline_mode:
            logger.info(
                f"Offline mode: Artifact {artifact_name} saved locally at {artifact_path}"
            )
            return

        try:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)
            logger.info(f"Artifact {artifact_name} logged to W&B")
        except Exception as e:
            logger.warning(f"Failed to log artifact to W&B: {e}")

    def _log_to_file(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to local file when W&B is unavailable."""
        log_dir = Path("./wandb_offline")
        log_dir.mkdir(exist_ok=True)

        log_entry = {"timestamp": time.time(), "step": step, "metrics": metrics}

        log_file = log_dir / "metrics.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def finish(self) -> None:
        """Finish W&B run safely."""
        if self.is_initialized and self.run:
            try:
                self.run.finish()
                logger.info("W&B run finished successfully")
            except Exception as e:
                logger.warning(f"Error finishing W&B run: {e}")
            finally:
                self.is_initialized = False
                self.run = None

    def is_available(self) -> bool:
        """Check if W&B is available and connected."""
        return WANDB_AVAILABLE and not self.offline_mode and self.is_initialized


# Global W&B manager instance
wandb_manager = WandBManager()


# Convenience functions
def init_wandb(project: str = "agent-forge", **kwargs) -> bool:
    """Initialize W&B with error handling."""
    return wandb_manager.initialize_run(project=project, **kwargs)


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics with W&B fallback."""
    wandb_manager.log_metrics(metrics, step)


def log_artifact(
    artifact_path: str, artifact_name: str, artifact_type: str = "model"
) -> None:
    """Log artifact with W&B fallback."""
    wandb_manager.log_artifact(artifact_path, artifact_name, artifact_type)


def finish_wandb() -> None:
    """Finish W&B run safely."""
    wandb_manager.finish()
