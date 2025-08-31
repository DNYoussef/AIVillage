"""
Persistence Service - Handles data serialization and storage.
Extracted from UnifiedManagement god class.
"""
import json
import logging
from typing import Any
from pathlib import Path

from core.error_handling import AIVillageException

logger = logging.getLogger(__name__)


class PersistenceService:
    """Service responsible for state persistence operations."""
    
    def __init__(self) -> None:
        """Initialize the persistence service."""
        self._backup_directory = Path("backups")
        self._backup_directory.mkdir(exist_ok=True)
        
    async def save_state(self, filename: str, state_data: dict[str, Any]) -> None:
        """Save system state to file."""
        try:
            filepath = Path(filename)
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists
            if filepath.exists():
                backup_path = self._backup_directory / f"{filepath.name}.backup"
                filepath.rename(backup_path)
                logger.info("Created backup at %s", backup_path)
            
            # Prepare state for serialization
            serializable_state = self._prepare_state_for_serialization(state_data)
            
            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_state, f, indent=2, default=str)
                
            logger.info("Saved state to %s", filepath)
            
        except Exception as e:
            logger.exception("Error saving state: %s", e)
            msg = f"Error saving state: {e!s}"
            raise AIVillageException(msg) from e

    async def load_state(self, filename: str) -> dict[str, Any]:
        """Load system state from file."""
        try:
            filepath = Path(filename)
            
            if not filepath.exists():
                msg = f"State file {filepath} does not exist"
                raise FileNotFoundError(msg)
            
            with open(filepath, encoding="utf-8") as f:
                state_data = json.load(f)
                
            # Post-process the loaded state
            processed_state = self._process_loaded_state(state_data)
            
            logger.info("Loaded state from %s", filepath)
            return processed_state
            
        except Exception as e:
            logger.exception("Error loading state: %s", e)
            msg = f"Error loading state: {e!s}"
            raise AIVillageException(msg) from e

    def _prepare_state_for_serialization(self, state_data: dict[str, Any]) -> dict[str, Any]:
        """Prepare state data for JSON serialization."""
        try:
            serializable_state = {}
            
            for key, value in state_data.items():
                if hasattr(value, '__dict__'):
                    # Convert objects to dictionaries
                    serializable_state[key] = self._object_to_dict(value)
                elif isinstance(value, (list, tuple)):
                    # Handle collections
                    serializable_state[key] = [
                        self._object_to_dict(item) if hasattr(item, '__dict__') else item
                        for item in value
                    ]
                elif isinstance(value, dict):
                    # Handle dictionaries recursively
                    serializable_state[key] = {
                        k: self._object_to_dict(v) if hasattr(v, '__dict__') else v
                        for k, v in value.items()
                    }
                else:
                    serializable_state[key] = value
                    
            return serializable_state
            
        except Exception as e:
            logger.exception("Error preparing state for serialization: %s", e)
            return state_data

    def _object_to_dict(self, obj: Any) -> dict[str, Any]:
        """Convert an object to a dictionary."""
        try:
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):  # Skip private attributes
                        if hasattr(value, '__dict__'):
                            result[key] = self._object_to_dict(value)
                        else:
                            result[key] = value
                return result
            return str(obj)
        except Exception:
            return str(obj)

    def _process_loaded_state(self, state_data: dict[str, Any]) -> dict[str, Any]:
        """Process loaded state data to restore proper types."""
        try:
            # This is where we would reconstruct objects from dictionaries
            # For now, we'll return the raw data and let the services handle reconstruction
            return state_data
        except Exception as e:
            logger.exception("Error processing loaded state: %s", e)
            return state_data

    async def create_checkpoint(self, checkpoint_name: str, state_data: dict[str, Any]) -> None:
        """Create a named checkpoint of the current state."""
        try:
            checkpoint_path = self._backup_directory / f"{checkpoint_name}.checkpoint.json"
            await self.save_state(str(checkpoint_path), state_data)
            logger.info("Created checkpoint: %s", checkpoint_name)
        except Exception as e:
            logger.exception("Error creating checkpoint: %s", e)
            msg = f"Error creating checkpoint: {e!s}"
            raise AIVillageException(msg) from e

    async def restore_checkpoint(self, checkpoint_name: str) -> dict[str, Any]:
        """Restore state from a named checkpoint."""
        try:
            checkpoint_path = self._backup_directory / f"{checkpoint_name}.checkpoint.json"
            state_data = await self.load_state(str(checkpoint_path))
            logger.info("Restored checkpoint: %s", checkpoint_name)
            return state_data
        except Exception as e:
            logger.exception("Error restoring checkpoint: %s", e)
            msg = f"Error restoring checkpoint: {e!s}"
            raise AIVillageException(msg) from e

    def list_checkpoints(self) -> list[str]:
        """List available checkpoints."""
        try:
            checkpoints = [
                f.stem.replace('.checkpoint', '') 
                for f in self._backup_directory.glob("*.checkpoint.json")
            ]
            return sorted(checkpoints)
        except Exception as e:
            logger.exception("Error listing checkpoints: %s", e)
            return []

    def get_file_info(self, filename: str) -> dict[str, Any]:
        """Get information about a state file."""
        try:
            filepath = Path(filename)
            if not filepath.exists():
                return {"exists": False}
                
            stat = filepath.stat()
            return {
                "exists": True,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "path": str(filepath.absolute()),
            }
        except Exception as e:
            logger.exception("Error getting file info: %s", e)
            return {"exists": False, "error": str(e)}