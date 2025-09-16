"""
Cross-Phase State Management System for Phase 6 Integration

This module manages state consistency across Phase 5, 6, and 7, ensuring
seamless data flow, state persistence, and recovery capabilities.
"""

import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class StateStatus(Enum):
    """State status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class Phase(Enum):
    """Phase enumeration"""
    PHASE5_TRAINING = "phase5_training"
    PHASE6_BAKING = "phase6_baking"
    PHASE7_ADAS = "phase7_adas"

@dataclass
class StateMetadata:
    """Metadata for state tracking"""
    state_id: str
    phase: Phase
    status: StateStatus
    created_at: datetime
    updated_at: datetime
    model_id: Optional[str]
    dependencies: List[str]
    checksum: str
    size_bytes: int
    tags: Dict[str, str]

@dataclass
class StateTransition:
    """State transition record"""
    from_state: str
    to_state: str
    phase: Phase
    timestamp: datetime
    success: bool
    duration_ms: float
    metadata: Dict[str, Any]

@dataclass
class StatePersistenceConfig:
    """Configuration for state persistence"""
    storage_dir: str
    max_state_age_days: int
    compression_enabled: bool
    encryption_enabled: bool
    backup_enabled: bool
    cleanup_interval_hours: int

class StateManager:
    """Cross-phase state management system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_dir = Path(config.get('storage_dir', '.claude/.artifacts/state'))
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.states = {}
        self.state_metadata = {}
        self.transitions = []
        self.dependencies = defaultdict(set)
        self.locks = defaultdict(threading.RLock)

        # Persistence configuration
        self.persistence_config = StatePersistenceConfig(
            storage_dir=str(self.storage_dir),
            max_state_age_days=config.get('max_state_age_days', 30),
            compression_enabled=config.get('compression_enabled', True),
            encryption_enabled=config.get('encryption_enabled', False),
            backup_enabled=config.get('backup_enabled', True),
            cleanup_interval_hours=config.get('cleanup_interval_hours', 24)
        )

        # Load existing states
        self._load_persisted_states()

    def create_state(self, state_id: str, phase: Phase, data: Any,
                    model_id: Optional[str] = None,
                    dependencies: Optional[List[str]] = None,
                    tags: Optional[Dict[str, str]] = None) -> bool:
        """Create a new state entry"""
        try:
            with self.locks[state_id]:
                if state_id in self.states:
                    logger.warning(f"State {state_id} already exists")
                    return False

                # Serialize data
                serialized_data = self._serialize_data(data)
                checksum = self._calculate_checksum(serialized_data)

                # Create metadata
                metadata = StateMetadata(
                    state_id=state_id,
                    phase=phase,
                    status=StateStatus.PENDING,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    model_id=model_id,
                    dependencies=dependencies or [],
                    checksum=checksum,
                    size_bytes=len(serialized_data),
                    tags=tags or {}
                )

                # Store state and metadata
                self.states[state_id] = data
                self.state_metadata[state_id] = metadata

                # Update dependencies
                for dep in dependencies or []:
                    self.dependencies[dep].add(state_id)

                # Persist state
                self._persist_state(state_id, data, metadata)

                logger.info(f"Created state {state_id} for {phase.value}")
                return True

        except Exception as e:
            logger.error(f"Failed to create state {state_id}: {e}")
            return False

    def update_state(self, state_id: str, data: Any,
                    status: Optional[StateStatus] = None,
                    tags: Optional[Dict[str, str]] = None) -> bool:
        """Update an existing state"""
        try:
            with self.locks[state_id]:
                if state_id not in self.states:
                    logger.error(f"State {state_id} not found")
                    return False

                old_metadata = self.state_metadata[state_id]

                # Update data if provided
                if data is not None:
                    serialized_data = self._serialize_data(data)
                    checksum = self._calculate_checksum(serialized_data)
                    self.states[state_id] = data
                    old_metadata.checksum = checksum
                    old_metadata.size_bytes = len(serialized_data)

                # Update status if provided
                if status is not None:
                    old_status = old_metadata.status
                    old_metadata.status = status

                    # Record transition
                    transition = StateTransition(
                        from_state=old_status.value,
                        to_state=status.value,
                        phase=old_metadata.phase,
                        timestamp=datetime.now(),
                        success=True,
                        duration_ms=0.0,
                        metadata={'state_id': state_id}
                    )
                    self.transitions.append(transition)

                # Update tags if provided
                if tags is not None:
                    old_metadata.tags.update(tags)

                old_metadata.updated_at = datetime.now()

                # Persist updated state
                self._persist_state(state_id, self.states[state_id], old_metadata)

                logger.info(f"Updated state {state_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to update state {state_id}: {e}")
            return False

    def get_state(self, state_id: str) -> Optional[Tuple[Any, StateMetadata]]:
        """Get state data and metadata"""
        try:
            with self.locks[state_id]:
                if state_id not in self.states:
                    return None

                return self.states[state_id], self.state_metadata[state_id]

        except Exception as e:
            logger.error(f"Failed to get state {state_id}: {e}")
            return None

    def delete_state(self, state_id: str) -> bool:
        """Delete a state and its dependencies"""
        try:
            with self.locks[state_id]:
                if state_id not in self.states:
                    logger.warning(f"State {state_id} not found")
                    return False

                # Check for dependents
                dependents = self.dependencies.get(state_id, set())
                if dependents:
                    logger.warning(f"State {state_id} has dependents: {dependents}")
                    return False

                # Remove from memory
                del self.states[state_id]
                del self.state_metadata[state_id]

                # Remove from dependencies
                for dep_id, deps in self.dependencies.items():
                    deps.discard(state_id)

                # Remove persisted state
                self._remove_persisted_state(state_id)

                logger.info(f"Deleted state {state_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete state {state_id}: {e}")
            return False

    def list_states(self, phase: Optional[Phase] = None,
                   status: Optional[StateStatus] = None,
                   model_id: Optional[str] = None) -> List[StateMetadata]:
        """List states with optional filtering"""
        results = []

        for state_id, metadata in self.state_metadata.items():
            if phase is not None and metadata.phase != phase:
                continue
            if status is not None and metadata.status != status:
                continue
            if model_id is not None and metadata.model_id != model_id:
                continue

            results.append(metadata)

        return sorted(results, key=lambda x: x.updated_at, reverse=True)

    def get_phase_transition(self, from_phase: Phase, to_phase: Phase,
                           model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get state transition between phases"""
        try:
            # Find states in source phase
            source_states = self.list_states(phase=from_phase, model_id=model_id)
            if not source_states:
                return None

            # Find the most recent completed state
            completed_source = None
            for state in source_states:
                if state.status == StateStatus.COMPLETED:
                    completed_source = state
                    break

            if not completed_source:
                return None

            # Look for corresponding state in target phase
            target_states = self.list_states(phase=to_phase, model_id=model_id)
            target_state = target_states[0] if target_states else None

            transition_data = {
                'source_state_id': completed_source.state_id,
                'source_phase': from_phase.value,
                'target_state_id': target_state.state_id if target_state else None,
                'target_phase': to_phase.value,
                'transition_ready': completed_source.status == StateStatus.COMPLETED,
                'source_metadata': asdict(completed_source),
                'target_metadata': asdict(target_state) if target_state else None
            }

            return transition_data

        except Exception as e:
            logger.error(f"Failed to get phase transition {from_phase} -> {to_phase}: {e}")
            return None

    def validate_state_consistency(self) -> Dict[str, Any]:
        """Validate consistency across all states"""
        validation_results = {
            'consistent': True,
            'total_states': len(self.states),
            'issues': [],
            'warnings': [],
            'orphaned_states': [],
            'missing_dependencies': [],
            'checksum_mismatches': [],
            'statistics': {}
        }

        try:
            # Check for orphaned states
            for state_id, metadata in self.state_metadata.items():
                for dep_id in metadata.dependencies:
                    if dep_id not in self.states:
                        validation_results['missing_dependencies'].append({
                            'state_id': state_id,
                            'missing_dependency': dep_id
                        })
                        validation_results['consistent'] = False

            # Check for checksum mismatches
            for state_id, data in self.states.items():
                metadata = self.state_metadata[state_id]
                serialized_data = self._serialize_data(data)
                current_checksum = self._calculate_checksum(serialized_data)

                if current_checksum != metadata.checksum:
                    validation_results['checksum_mismatches'].append({
                        'state_id': state_id,
                        'expected': metadata.checksum,
                        'actual': current_checksum
                    })
                    validation_results['consistent'] = False

            # Generate statistics
            phase_counts = defaultdict(int)
            status_counts = defaultdict(int)

            for metadata in self.state_metadata.values():
                phase_counts[metadata.phase.value] += 1
                status_counts[metadata.status.value] += 1

            validation_results['statistics'] = {
                'by_phase': dict(phase_counts),
                'by_status': dict(status_counts),
                'total_size_bytes': sum(m.size_bytes for m in self.state_metadata.values()),
                'average_age_hours': self._calculate_average_age_hours()
            }

            # Generate warnings for old states
            max_age_hours = self.persistence_config.max_state_age_days * 24
            for state_id, metadata in self.state_metadata.items():
                age_hours = (datetime.now() - metadata.created_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    validation_results['warnings'].append(f"State {state_id} is {age_hours:.1f} hours old")

        except Exception as e:
            validation_results['issues'].append(f"Validation error: {e}")
            validation_results['consistent'] = False

        return validation_results

    def create_checkpoint(self, checkpoint_name: str,
                         phases: Optional[List[Phase]] = None) -> bool:
        """Create a checkpoint of current state"""
        try:
            checkpoint_dir = self.storage_dir / 'checkpoints' / checkpoint_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Filter states by phases if specified
            states_to_checkpoint = {}
            metadata_to_checkpoint = {}

            for state_id, data in self.states.items():
                metadata = self.state_metadata[state_id]
                if phases is None or metadata.phase in phases:
                    states_to_checkpoint[state_id] = data
                    metadata_to_checkpoint[state_id] = metadata

            # Save checkpoint data
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'phases': [p.value for p in phases] if phases else 'all',
                'state_count': len(states_to_checkpoint),
                'total_size_bytes': sum(m.size_bytes for m in metadata_to_checkpoint.values())
            }

            # Save states
            with open(checkpoint_dir / 'states.pkl', 'wb') as f:
                pickle.dump(states_to_checkpoint, f)

            # Save metadata
            serializable_metadata = {}
            for state_id, metadata in metadata_to_checkpoint.items():
                serializable_metadata[state_id] = asdict(metadata)
                # Convert datetime objects to strings
                serializable_metadata[state_id]['created_at'] = metadata.created_at.isoformat()
                serializable_metadata[state_id]['updated_at'] = metadata.updated_at.isoformat()
                serializable_metadata[state_id]['phase'] = metadata.phase.value
                serializable_metadata[state_id]['status'] = metadata.status.value

            with open(checkpoint_dir / 'metadata.json', 'w') as f:
                json.dump(serializable_metadata, f, indent=2)

            # Save checkpoint info
            with open(checkpoint_dir / 'checkpoint_info.json', 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.info(f"Created checkpoint {checkpoint_name} with {len(states_to_checkpoint)} states")
            return True

        except Exception as e:
            logger.error(f"Failed to create checkpoint {checkpoint_name}: {e}")
            return False

    def restore_checkpoint(self, checkpoint_name: str,
                          overwrite_existing: bool = False) -> bool:
        """Restore from a checkpoint"""
        try:
            checkpoint_dir = self.storage_dir / 'checkpoints' / checkpoint_name

            if not checkpoint_dir.exists():
                logger.error(f"Checkpoint {checkpoint_name} not found")
                return False

            # Load checkpoint info
            with open(checkpoint_dir / 'checkpoint_info.json', 'r') as f:
                checkpoint_info = json.load(f)

            # Load states
            with open(checkpoint_dir / 'states.pkl', 'rb') as f:
                checkpoint_states = pickle.load(f)

            # Load metadata
            with open(checkpoint_dir / 'metadata.json', 'r') as f:
                checkpoint_metadata = json.load(f)

            # Convert metadata back to proper types
            restored_metadata = {}
            for state_id, meta_dict in checkpoint_metadata.items():
                meta_dict['created_at'] = datetime.fromisoformat(meta_dict['created_at'])
                meta_dict['updated_at'] = datetime.fromisoformat(meta_dict['updated_at'])
                meta_dict['phase'] = Phase(meta_dict['phase'])
                meta_dict['status'] = StateStatus(meta_dict['status'])

                restored_metadata[state_id] = StateMetadata(**meta_dict)

            # Restore states
            conflicts = []
            for state_id in checkpoint_states.keys():
                if state_id in self.states and not overwrite_existing:
                    conflicts.append(state_id)

            if conflicts and not overwrite_existing:
                logger.error(f"Conflicts found, use overwrite_existing=True: {conflicts}")
                return False

            # Apply restoration
            for state_id, data in checkpoint_states.items():
                self.states[state_id] = data
                self.state_metadata[state_id] = restored_metadata[state_id]

                # Update dependencies
                metadata = restored_metadata[state_id]
                for dep in metadata.dependencies:
                    self.dependencies[dep].add(state_id)

            logger.info(f"Restored checkpoint {checkpoint_name} with {len(checkpoint_states)} states")
            return True

        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_name}: {e}")
            return False

    def cleanup_old_states(self, dry_run: bool = False) -> Dict[str, Any]:
        """Clean up old states based on configuration"""
        cleanup_results = {
            'states_examined': 0,
            'states_cleaned': 0,
            'space_freed_bytes': 0,
            'cleaned_states': [],
            'errors': []
        }

        try:
            max_age_hours = self.persistence_config.max_state_age_days * 24
            current_time = datetime.now()

            states_to_clean = []
            for state_id, metadata in self.state_metadata.items():
                cleanup_results['states_examined'] += 1

                age_hours = (current_time - metadata.created_at).total_seconds() / 3600

                # Clean if too old and not in use
                if (age_hours > max_age_hours and
                    metadata.status in [StateStatus.COMPLETED, StateStatus.FAILED] and
                    not self.dependencies.get(state_id, set())):

                    states_to_clean.append((state_id, metadata))

            # Perform cleanup
            for state_id, metadata in states_to_clean:
                try:
                    if not dry_run:
                        success = self.delete_state(state_id)
                        if success:
                            cleanup_results['states_cleaned'] += 1
                            cleanup_results['space_freed_bytes'] += metadata.size_bytes
                            cleanup_results['cleaned_states'].append(state_id)
                    else:
                        cleanup_results['states_cleaned'] += 1
                        cleanup_results['space_freed_bytes'] += metadata.size_bytes
                        cleanup_results['cleaned_states'].append(state_id)

                except Exception as e:
                    cleanup_results['errors'].append(f"Failed to clean {state_id}: {e}")

            logger.info(f"Cleanup completed: {cleanup_results['states_cleaned']} states cleaned, "
                       f"{cleanup_results['space_freed_bytes']} bytes freed")

        except Exception as e:
            cleanup_results['errors'].append(f"Cleanup error: {e}")

        return cleanup_results

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage"""
        try:
            return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum for data integrity"""
        import hashlib
        return hashlib.sha256(data).hexdigest()

    def _persist_state(self, state_id: str, data: Any, metadata: StateMetadata):
        """Persist state to storage"""
        try:
            state_dir = self.storage_dir / 'states' / state_id
            state_dir.mkdir(parents=True, exist_ok=True)

            # Save data
            serialized_data = self._serialize_data(data)
            with open(state_dir / 'data.pkl', 'wb') as f:
                f.write(serialized_data)

            # Save metadata
            metadata_dict = asdict(metadata)
            metadata_dict['created_at'] = metadata.created_at.isoformat()
            metadata_dict['updated_at'] = metadata.updated_at.isoformat()
            metadata_dict['phase'] = metadata.phase.value
            metadata_dict['status'] = metadata.status.value

            with open(state_dir / 'metadata.json', 'w') as f:
                json.dump(metadata_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to persist state {state_id}: {e}")

    def _remove_persisted_state(self, state_id: str):
        """Remove persisted state from storage"""
        try:
            state_dir = self.storage_dir / 'states' / state_id
            if state_dir.exists():
                import shutil
                shutil.rmtree(state_dir)
        except Exception as e:
            logger.error(f"Failed to remove persisted state {state_id}: {e}")

    def _load_persisted_states(self):
        """Load persisted states on initialization"""
        try:
            states_dir = self.storage_dir / 'states'
            if not states_dir.exists():
                return

            for state_dir in states_dir.iterdir():
                if not state_dir.is_dir():
                    continue

                state_id = state_dir.name

                try:
                    # Load metadata
                    with open(state_dir / 'metadata.json', 'r') as f:
                        metadata_dict = json.load(f)

                    # Convert back to proper types
                    metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                    metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
                    metadata_dict['phase'] = Phase(metadata_dict['phase'])
                    metadata_dict['status'] = StateStatus(metadata_dict['status'])

                    metadata = StateMetadata(**metadata_dict)

                    # Load data
                    with open(state_dir / 'data.pkl', 'rb') as f:
                        data = pickle.load(f)

                    # Store in memory
                    self.states[state_id] = data
                    self.state_metadata[state_id] = metadata

                    # Update dependencies
                    for dep in metadata.dependencies:
                        self.dependencies[dep].add(state_id)

                except Exception as e:
                    logger.error(f"Failed to load state {state_id}: {e}")

            logger.info(f"Loaded {len(self.states)} persisted states")

        except Exception as e:
            logger.error(f"Failed to load persisted states: {e}")

    def _calculate_average_age_hours(self) -> float:
        """Calculate average age of states in hours"""
        if not self.state_metadata:
            return 0.0

        current_time = datetime.now()
        total_age_hours = 0.0

        for metadata in self.state_metadata.values():
            age_hours = (current_time - metadata.created_at).total_seconds() / 3600
            total_age_hours += age_hours

        return total_age_hours / len(self.state_metadata)

    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary"""
        validation_results = self.validate_state_consistency()

        return {
            'total_states': len(self.states),
            'consistency_status': 'HEALTHY' if validation_results['consistent'] else 'ISSUES',
            'statistics': validation_results['statistics'],
            'storage_location': str(self.storage_dir),
            'persistence_config': asdict(self.persistence_config),
            'recent_transitions': len([t for t in self.transitions
                                     if (datetime.now() - t.timestamp).total_seconds() < 3600]),
            'validation_summary': {
                'consistent': validation_results['consistent'],
                'issues_count': len(validation_results['issues']),
                'warnings_count': len(validation_results['warnings'])
            }
        }

def create_state_manager(config: Dict[str, Any]) -> StateManager:
    """Factory function to create state manager"""
    return StateManager(config)

# Testing utilities
def test_state_management():
    """Test state management functionality"""
    config = {
        'storage_dir': '.claude/.artifacts/test_state',
        'max_state_age_days': 7,
        'compression_enabled': True,
        'backup_enabled': True
    }

    state_manager = StateManager(config)

    # Test state creation
    test_data = {'model': 'test_model', 'accuracy': 0.95}
    success = state_manager.create_state(
        'test_state_1',
        Phase.PHASE6_BAKING,
        test_data,
        model_id='model_123'
    )
    print(f"State creation success: {success}")

    # Test state retrieval
    retrieved = state_manager.get_state('test_state_1')
    print(f"State retrieved: {retrieved is not None}")

    # Test validation
    validation = state_manager.validate_state_consistency()
    print(f"Validation results: {validation}")

    # Test summary
    summary = state_manager.get_state_summary()
    print(f"State summary: {summary}")

    return state_manager

if __name__ == "__main__":
    test_state_management()