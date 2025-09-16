"""
Cross-Phase State Manager
Manages state transitions and persistence across Agent Forge phases.
"""

import asyncio
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib

class StateType(Enum):
    MODEL_STATE = "model_state"
    TRAINING_STATE = "training_state"
    CONFIGURATION = "configuration"
    METRICS = "metrics"
    CHECKPOINT = "checkpoint"

class StateStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    CORRUPTED = "corrupted"
    MIGRATING = "migrating"

@dataclass
class StateMetadata:
    """Metadata for state objects."""
    state_id: str
    phase: str
    state_type: StateType
    created_at: datetime
    updated_at: datetime
    version: str
    checksum: str
    size_bytes: int
    dependencies: List[str]

@dataclass
class StateSnapshot:
    """Complete state snapshot."""
    metadata: StateMetadata
    data: Dict[str, Any]
    status: StateStatus

class CrossPhaseStateManager:
    """
    Manages state persistence and transitions across Agent Forge phases.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.state_dir = state_dir or Path("state/cross_phase")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # State storage
        self.active_states = {}
        self.state_history = {}
        self.migration_queue = []

        # Initialize subdirectories
        self.phase_dirs = {
            "phase4": self.state_dir / "phase4",
            "phase5": self.state_dir / "phase5",
            "phase6": self.state_dir / "phase6",
            "shared": self.state_dir / "shared"
        }

        for phase_dir in self.phase_dirs.values():
            phase_dir.mkdir(exist_ok=True)

    async def initialize(self) -> bool:
        """Initialize state manager."""
        try:
            self.logger.info("Initializing cross-phase state manager")

            # Load existing states
            await self._load_existing_states()

            # Verify state integrity
            await self._verify_state_integrity()

            # Clean up corrupted states
            await self._cleanup_corrupted_states()

            self.logger.info(f"State manager initialized with {len(self.active_states)} active states")
            return True

        except Exception as e:
            self.logger.error(f"State manager initialization failed: {e}")
            return False

    async def save_state(self, phase: str, state_type: StateType, state_id: str, data: Dict[str, Any], dependencies: List[str] = None) -> bool:
        """Save state for a specific phase."""
        try:
            self.logger.info(f"Saving state: {phase}/{state_type.value}/{state_id}")

            # Create state metadata
            metadata = StateMetadata(
                state_id=state_id,
                phase=phase,
                state_type=state_type,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version="1.0",
                checksum=self._calculate_checksum(data),
                size_bytes=len(json.dumps(data, default=str)),
                dependencies=dependencies or []
            )

            # Create snapshot
            snapshot = StateSnapshot(
                metadata=metadata,
                data=data,
                status=StateStatus.ACTIVE
            )

            # Save to disk
            await self._save_snapshot_to_disk(snapshot)

            # Update in-memory cache
            self.active_states[f"{phase}:{state_id}"] = snapshot

            # Update history
            if f"{phase}:{state_id}" not in self.state_history:
                self.state_history[f"{phase}:{state_id}"] = []
            self.state_history[f"{phase}:{state_id}"].append(metadata)

            self.logger.info(f"State saved successfully: {phase}:{state_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save state {phase}:{state_id}: {e}")
            return False

    async def load_state(self, phase: str, state_id: str) -> Optional[Dict[str, Any]]:
        """Load state for a specific phase."""
        try:
            state_key = f"{phase}:{state_id}"
            self.logger.info(f"Loading state: {state_key}")

            # Check in-memory cache first
            if state_key in self.active_states:
                snapshot = self.active_states[state_key]
                if snapshot.status == StateStatus.ACTIVE:
                    return snapshot.data

            # Load from disk
            snapshot = await self._load_snapshot_from_disk(phase, state_id)
            if snapshot and snapshot.status == StateStatus.ACTIVE:
                # Update cache
                self.active_states[state_key] = snapshot
                return snapshot.data

            self.logger.warning(f"State not found or inactive: {state_key}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to load state {phase}:{state_id}: {e}")
            return None

    async def migrate_state(self, from_phase: str, to_phase: str, state_id: str, migration_config: Dict[str, Any] = None) -> bool:
        """Migrate state between phases."""
        try:
            self.logger.info(f"Migrating state {state_id}: {from_phase} -> {to_phase}")

            # Load source state
            source_data = await self.load_state(from_phase, state_id)
            if source_data is None:
                raise ValueError(f"Source state not found: {from_phase}:{state_id}")

            # Apply migration transformation
            migrated_data = await self._apply_migration_transform(
                source_data, from_phase, to_phase, migration_config or {}
            )

            # Save to target phase
            migration_success = await self.save_state(
                to_phase, StateType.MIGRATING, state_id, migrated_data
            )

            if migration_success:
                # Mark source as archived
                await self._archive_state(from_phase, state_id)

                # Update target state status to active
                await self._activate_migrated_state(to_phase, state_id)

                self.logger.info(f"State migration completed: {from_phase}:{state_id} -> {to_phase}:{state_id}")
                return True
            else:
                raise Exception("Failed to save migrated state")

        except Exception as e:
            self.logger.error(f"State migration failed: {e}")
            return False

    async def create_checkpoint(self, phase: str, checkpoint_name: str, include_states: List[str] = None) -> bool:
        """Create checkpoint of current phase state."""
        try:
            self.logger.info(f"Creating checkpoint: {phase}/{checkpoint_name}")

            # Determine states to include
            if include_states is None:
                # Include all active states for the phase
                include_states = [
                    key.split(':')[1] for key in self.active_states.keys()
                    if key.startswith(f"{phase}:")
                ]

            # Collect state data
            checkpoint_data = {
                "checkpoint_name": checkpoint_name,
                "phase": phase,
                "created_at": datetime.now().isoformat(),
                "states": {}
            }

            for state_id in include_states:
                state_data = await self.load_state(phase, state_id)
                if state_data:
                    checkpoint_data["states"][state_id] = state_data

            # Save checkpoint
            checkpoint_success = await self.save_state(
                phase, StateType.CHECKPOINT, checkpoint_name, checkpoint_data
            )

            if checkpoint_success:
                self.logger.info(f"Checkpoint created: {phase}/{checkpoint_name}")
                return True
            else:
                raise Exception("Failed to save checkpoint")

        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            return False

    async def restore_checkpoint(self, phase: str, checkpoint_name: str) -> bool:
        """Restore from checkpoint."""
        try:
            self.logger.info(f"Restoring checkpoint: {phase}/{checkpoint_name}")

            # Load checkpoint data
            checkpoint_data = await self.load_state(phase, checkpoint_name)
            if not checkpoint_data:
                raise ValueError(f"Checkpoint not found: {phase}/{checkpoint_name}")

            # Restore individual states
            restored_states = []
            failed_states = []

            states = checkpoint_data.get("states", {})
            for state_id, state_data in states.items():
                success = await self.save_state(phase, StateType.MODEL_STATE, state_id, state_data)
                if success:
                    restored_states.append(state_id)
                else:
                    failed_states.append(state_id)

            # Log results
            self.logger.info(f"Checkpoint restore: {len(restored_states)} restored, {len(failed_states)} failed")

            return len(failed_states) == 0

        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint: {e}")
            return False

    async def get_state_summary(self, phase: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of states."""
        try:
            summary = {
                "total_states": 0,
                "by_phase": {},
                "by_type": {},
                "by_status": {},
                "storage_size_mb": 0
            }

            for state_key, snapshot in self.active_states.items():
                phase_name, state_id = state_key.split(':', 1)

                # Filter by phase if specified
                if phase and phase_name != phase:
                    continue

                summary["total_states"] += 1
                summary["storage_size_mb"] += snapshot.metadata.size_bytes / (1024 * 1024)

                # Count by phase
                if phase_name not in summary["by_phase"]:
                    summary["by_phase"][phase_name] = 0
                summary["by_phase"][phase_name] += 1

                # Count by type
                state_type = snapshot.metadata.state_type.value
                if state_type not in summary["by_type"]:
                    summary["by_type"][state_type] = 0
                summary["by_type"][state_type] += 1

                # Count by status
                status = snapshot.status.value
                if status not in summary["by_status"]:
                    summary["by_status"][status] = 0
                summary["by_status"][status] += 1

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get state summary: {e}")
            return {"error": str(e)}

    async def cleanup_old_states(self, max_age_days: int = 30, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up old states."""
        try:
            self.logger.info(f"Cleaning up states older than {max_age_days} days (dry_run={dry_run})")

            cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)

            cleanup_summary = {
                "total_checked": 0,
                "to_cleanup": 0,
                "cleaned_up": 0,
                "errors": 0,
                "freed_mb": 0
            }

            states_to_cleanup = []

            for state_key, snapshot in self.active_states.items():
                cleanup_summary["total_checked"] += 1

                if snapshot.metadata.updated_at.timestamp() < cutoff_date:
                    if snapshot.status == StateStatus.ARCHIVED:
                        states_to_cleanup.append((state_key, snapshot))
                        cleanup_summary["to_cleanup"] += 1
                        cleanup_summary["freed_mb"] += snapshot.metadata.size_bytes / (1024 * 1024)

            # Perform cleanup if not dry run
            if not dry_run:
                for state_key, snapshot in states_to_cleanup:
                    try:
                        await self._delete_state(state_key)
                        cleanup_summary["cleaned_up"] += 1
                    except Exception as e:
                        self.logger.error(f"Failed to cleanup state {state_key}: {e}")
                        cleanup_summary["errors"] += 1

            self.logger.info(f"Cleanup summary: {cleanup_summary}")
            return cleanup_summary

        except Exception as e:
            self.logger.error(f"State cleanup failed: {e}")
            return {"error": str(e)}

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    async def _save_snapshot_to_disk(self, snapshot: StateSnapshot) -> bool:
        """Save snapshot to disk."""
        try:
            phase = snapshot.metadata.phase
            state_id = snapshot.metadata.state_id

            # Determine storage directory
            phase_dir = self.phase_dirs.get(phase, self.phase_dirs["shared"])

            # Create state file paths
            metadata_file = phase_dir / f"{state_id}_metadata.json"
            data_file = phase_dir / f"{state_id}_data.pkl"

            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(asdict(snapshot.metadata), f, indent=2, default=str)

            # Save data
            with open(data_file, 'wb') as f:
                pickle.dump(snapshot.data, f)

            return True

        except Exception as e:
            self.logger.error(f"Failed to save snapshot to disk: {e}")
            return False

    async def _load_snapshot_from_disk(self, phase: str, state_id: str) -> Optional[StateSnapshot]:
        """Load snapshot from disk."""
        try:
            phase_dir = self.phase_dirs.get(phase, self.phase_dirs["shared"])

            metadata_file = phase_dir / f"{state_id}_metadata.json"
            data_file = phase_dir / f"{state_id}_data.pkl"

            # Check if files exist
            if not metadata_file.exists() or not data_file.exists():
                return None

            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)

            # Convert datetime strings back to datetime objects
            metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
            metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
            metadata_dict['state_type'] = StateType(metadata_dict['state_type'])

            metadata = StateMetadata(**metadata_dict)

            # Load data
            with open(data_file, 'rb') as f:
                data = pickle.load(f)

            # Verify checksum
            expected_checksum = metadata.checksum
            actual_checksum = self._calculate_checksum(data)

            if expected_checksum != actual_checksum:
                self.logger.warning(f"Checksum mismatch for state {phase}:{state_id}")
                status = StateStatus.CORRUPTED
            else:
                status = StateStatus.ACTIVE

            return StateSnapshot(metadata=metadata, data=data, status=status)

        except Exception as e:
            self.logger.error(f"Failed to load snapshot from disk: {e}")
            return None

    async def _load_existing_states(self) -> None:
        """Load existing states from disk."""
        try:
            for phase, phase_dir in self.phase_dirs.items():
                if not phase_dir.exists():
                    continue

                # Find metadata files
                metadata_files = list(phase_dir.glob("*_metadata.json"))

                for metadata_file in metadata_files:
                    # Extract state_id from filename
                    state_id = metadata_file.stem.replace("_metadata", "")

                    # Load snapshot
                    snapshot = await self._load_snapshot_from_disk(phase, state_id)
                    if snapshot:
                        state_key = f"{phase}:{state_id}"
                        self.active_states[state_key] = snapshot

                        # Initialize history
                        if state_key not in self.state_history:
                            self.state_history[state_key] = []
                        self.state_history[state_key].append(snapshot.metadata)

            self.logger.info(f"Loaded {len(self.active_states)} existing states")

        except Exception as e:
            self.logger.error(f"Failed to load existing states: {e}")

    async def _verify_state_integrity(self) -> None:
        """Verify integrity of loaded states."""
        corrupted_states = []

        for state_key, snapshot in self.active_states.items():
            if snapshot.status == StateStatus.CORRUPTED:
                corrupted_states.append(state_key)

        if corrupted_states:
            self.logger.warning(f"Found {len(corrupted_states)} corrupted states: {corrupted_states}")

    async def _cleanup_corrupted_states(self) -> None:
        """Clean up corrupted states."""
        corrupted_states = [
            state_key for state_key, snapshot in self.active_states.items()
            if snapshot.status == StateStatus.CORRUPTED
        ]

        for state_key in corrupted_states:
            self.logger.info(f"Archiving corrupted state: {state_key}")
            phase, state_id = state_key.split(':', 1)
            await self._archive_state(phase, state_id)

    async def _apply_migration_transform(self, data: Dict[str, Any], from_phase: str, to_phase: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply migration transformation to data."""
        try:
            # Phase-specific transformations
            if from_phase == "phase4" and to_phase == "phase5":
                return await self._migrate_phase4_to_phase5(data, config)
            elif from_phase == "phase5" and to_phase == "phase6":
                return await self._migrate_phase5_to_phase6(data, config)
            else:
                # Default: no transformation
                return data.copy()

        except Exception as e:
            self.logger.error(f"Migration transform failed: {e}")
            raise

    async def _migrate_phase4_to_phase5(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate Phase 4 data to Phase 5 format."""
        migrated_data = data.copy()

        # Add Phase 5 specific fields
        migrated_data["phase5_enhancements"] = {
            "dynamic_quantization": True,
            "adaptive_training": True,
            "migration_timestamp": datetime.now().isoformat()
        }

        # Update configuration if present
        if "config" in migrated_data:
            migrated_data["config"]["phase"] = "phase5"
            migrated_data["config"]["migrated_from"] = "phase4"

        return migrated_data

    async def _migrate_phase5_to_phase6(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate Phase 5 data to Phase 6 format."""
        migrated_data = data.copy()

        # Add Phase 6 specific fields
        migrated_data["phase6_preparation"] = {
            "baking_ready": True,
            "export_format": "phase6_compatible",
            "migration_timestamp": datetime.now().isoformat()
        }

        # Update configuration if present
        if "config" in migrated_data:
            migrated_data["config"]["phase"] = "phase6"
            migrated_data["config"]["migrated_from"] = "phase5"

        return migrated_data

    async def _archive_state(self, phase: str, state_id: str) -> bool:
        """Archive a state."""
        try:
            state_key = f"{phase}:{state_id}"
            if state_key in self.active_states:
                snapshot = self.active_states[state_key]
                snapshot.status = StateStatus.ARCHIVED
                snapshot.metadata.updated_at = datetime.now()

                # Save updated snapshot
                await self._save_snapshot_to_disk(snapshot)

                self.logger.info(f"State archived: {state_key}")
                return True
            else:
                self.logger.warning(f"State not found for archiving: {state_key}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to archive state {phase}:{state_id}: {e}")
            return False

    async def _activate_migrated_state(self, phase: str, state_id: str) -> bool:
        """Activate a migrated state."""
        try:
            state_key = f"{phase}:{state_id}"
            if state_key in self.active_states:
                snapshot = self.active_states[state_key]
                snapshot.status = StateStatus.ACTIVE
                snapshot.metadata.updated_at = datetime.now()

                # Save updated snapshot
                await self._save_snapshot_to_disk(snapshot)

                self.logger.info(f"Migrated state activated: {state_key}")
                return True
            else:
                self.logger.warning(f"Migrated state not found: {state_key}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to activate migrated state {phase}:{state_id}: {e}")
            return False

    async def _delete_state(self, state_key: str) -> bool:
        """Delete a state completely."""
        try:
            phase, state_id = state_key.split(':', 1)
            phase_dir = self.phase_dirs.get(phase, self.phase_dirs["shared"])

            # Delete files
            metadata_file = phase_dir / f"{state_id}_metadata.json"
            data_file = phase_dir / f"{state_id}_data.pkl"

            if metadata_file.exists():
                metadata_file.unlink()
            if data_file.exists():
                data_file.unlink()

            # Remove from memory
            if state_key in self.active_states:
                del self.active_states[state_key]
            if state_key in self.state_history:
                del self.state_history[state_key]

            self.logger.info(f"State deleted: {state_key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete state {state_key}: {e}")
            return False