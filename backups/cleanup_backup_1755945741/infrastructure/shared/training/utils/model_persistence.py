"""
Model Persistence Utilities

Handles saving and loading of trained models and configurations.
Separated from training logic to follow single responsibility principle.
"""

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    """Protocol for models that can be saved."""

    def state_dict(self) -> dict[str, Any]:
        """Get model state dictionary."""
        ...


class ModelPersistenceManager:
    """
    Manages saving and loading of models and configurations.

    Provides consistent interface for model persistence operations.
    """

    def save_model_and_config(self, model: ModelProtocol, config: Any, model_name: str, checkpoint_dir: Path) -> bool:
        """
        Save model state and configuration to disk.

        Args:
            model: Model to save
            config: Configuration object to save
            model_name: Name for logging purposes
            checkpoint_dir: Directory to save to

        Returns:
            True if save was successful
        """
        try:
            # Create directory if it doesn't exist
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save model state dict
            model_path = checkpoint_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            logger.debug(f"Model state saved to {model_path}")

            # Save configuration
            config_path = checkpoint_dir / "config.json"
            config_dict = self._extract_config_dict(config)

            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            logger.debug(f"Configuration saved to {config_path}")

            logger.info(f"Saved {model_name} to {checkpoint_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to save {model_name}: {e}")
            return False

    def load_model_state(self, checkpoint_dir: Path) -> dict[str, Any] | None:
        """
        Load model state dict from checkpoint directory.

        Args:
            checkpoint_dir: Directory containing saved model

        Returns:
            Model state dict or None if loading failed
        """
        try:
            model_path = checkpoint_dir / "model.pt"

            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None

            state_dict = torch.load(model_path, map_location="cpu")
            logger.info(f"Loaded model state from {model_path}")
            return state_dict

        except Exception as e:
            logger.error(f"Failed to load model from {checkpoint_dir}: {e}")
            return None

    def load_config(self, checkpoint_dir: Path) -> dict[str, Any] | None:
        """
        Load configuration from checkpoint directory.

        Args:
            checkpoint_dir: Directory containing saved configuration

        Returns:
            Configuration dict or None if loading failed
        """
        try:
            config_path = checkpoint_dir / "config.json"

            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                return None

            with open(config_path) as f:
                config_dict = json.load(f)

            logger.info(f"Loaded configuration from {config_path}")
            return config_dict

        except Exception as e:
            logger.error(f"Failed to load config from {checkpoint_dir}: {e}")
            return None

    def save_training_summary(self, summary_data: dict[str, Any], summary_path: Path | str) -> bool:
        """
        Save training summary to JSON file.

        Args:
            summary_data: Dictionary containing training summary
            summary_path: Path to save summary file

        Returns:
            True if save was successful
        """
        try:
            summary_path = Path(summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            with open(summary_path, "w") as f:
                json.dump(summary_data, f, indent=2)

            logger.info(f"Training summary saved to {summary_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")
            return False

    def _extract_config_dict(self, config: Any) -> dict[str, Any]:
        """
        Extract dictionary representation from configuration object.

        Args:
            config: Configuration object

        Returns:
            Dictionary representation of configuration
        """
        if hasattr(config, "__dict__"):
            return config.__dict__
        elif isinstance(config, dict):
            return config
        else:
            # Try to convert to dict for basic types
            try:
                return dict(config)
            except (TypeError, ValueError):
                logger.warning(f"Could not convert config to dict: {type(config)}")
                return {"config_type": str(type(config))}

    def get_checkpoint_info(self, checkpoint_dir: Path) -> dict[str, Any]:
        """
        Get information about a checkpoint directory.

        Args:
            checkpoint_dir: Directory to inspect

        Returns:
            Dictionary with checkpoint information
        """
        info = {
            "directory": str(checkpoint_dir),
            "exists": checkpoint_dir.exists(),
            "model_file_exists": False,
            "config_file_exists": False,
            "file_count": 0,
        }

        if checkpoint_dir.exists():
            info["model_file_exists"] = (checkpoint_dir / "model.pt").exists()
            info["config_file_exists"] = (checkpoint_dir / "config.json").exists()
            info["file_count"] = len(list(checkpoint_dir.iterdir()))

        return info
