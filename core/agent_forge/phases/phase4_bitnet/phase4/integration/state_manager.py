#!/usr/bin/env python3
"""
BitNet Phase 4 - Cross-Phase State Management

Manages state synchronization across all BitNet phases:
- Phase 2 (EvoMerge) state tracking
- Phase 3 (Quiet-STaR) state management
- Phase 4 (BitNet) state coordination
- Phase 5 preparation state handling
"""

import torch
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

class PhaseState(Enum):
    """Phase state enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SYNCHRONIZED = "synchronized"

@dataclass
class StateSnapshot:
    """State snapshot for a specific phase"""
    phase_id: str
    timestamp: str
    state: PhaseState
    data: Dict[str, Any]
    metrics: Dict[str, float]
    errors: List[str]
    
class CrossPhaseStateManager:
    """Manages state across all BitNet phases"""
    
    def __init__(self, state_dir: str = "./.claude/.artifacts/bitnet-states"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._state_lock = threading.RLock()
        
        # Phase states
        self.phase_states = {
            'phase2': {
                'state': PhaseState.PENDING,
                'data': {},
                'metrics': {},
                'errors': [],
                'last_update': None
            },
            'phase3': {
                'state': PhaseState.PENDING,
                'data': {},
                'metrics': {},
                'errors': [],
                'last_update': None
            },
            'phase4': {
                'state': PhaseState.IN_PROGRESS,
                'data': {},
                'metrics': {},
                'errors': [],
                'last_update': datetime.now().isoformat()
            },
            'phase5': {
                'state': PhaseState.PENDING,
                'data': {},
                'metrics': {},
                'errors': [],
                'last_update': None
            }
        }
        
        # Global state tracking
        self.global_state = {
            'current_phase': 'phase4',
            'overall_progress': 0.0,
            'integration_status': 'in_progress',
            'quality_gates_passed': 0,
            'total_quality_gates': 16,  # Total across all phases
            'critical_errors': [],
            'performance_metrics': {}
        }
        
        # Load existing state if available
        self._load_persistent_state()
        
    def update_phase_state(self, phase_id: str, state_data: Dict[str, Any], 
                          metrics: Optional[Dict[str, float]] = None,
                          errors: Optional[List[str]] = None) -> bool:
        """Update state for a specific phase"""
        try:
            with self._state_lock:
                if phase_id not in self.phase_states:
                    self.logger.error(f"Invalid phase ID: {phase_id}")
                    return False
                    
                # Update phase state
                phase_state = self.phase_states[phase_id]
                phase_state['data'].update(state_data)
                
                if metrics:
                    phase_state['metrics'].update(metrics)
                    
                if errors:
                    phase_state['errors'].extend(errors)
                    
                phase_state['last_update'] = datetime.now().isoformat()
                
                # Determine phase status
                self._update_phase_status(phase_id)
                
                # Update global state
                self._update_global_state()
                
                # Persist state
                self._save_persistent_state()
                
                self.logger.info(f"Updated {phase_id} state")
                return True
                
        except Exception as e:
            self.logger.error(f"State update error for {phase_id}: {e}")
            return False
            
    def get_phase_state(self, phase_id: str) -> Optional[Dict[str, Any]]:
        """Get current state for a specific phase"""
        with self._state_lock:
            if phase_id in self.phase_states:
                return self.phase_states[phase_id].copy()
            return None
            
    def get_global_state(self) -> Dict[str, Any]:
        """Get global system state"""
        with self._state_lock:
            return {
                'global_state': self.global_state.copy(),
                'phase_states': {k: v.copy() for k, v in self.phase_states.items()},
                'timestamp': datetime.now().isoformat()
            }
            
    def synchronize_phases(self) -> Dict[str, Any]:
        """Synchronize state across all phases"""
        sync_results = {
            'timestamp': datetime.now().isoformat(),
            'synchronized_phases': [],
            'failed_phases': [],
            'overall_sync_status': False
        }
        
        try:
            with self._state_lock:
                self.logger.info("Starting cross-phase synchronization")
                
                # Check each phase for synchronization readiness
                for phase_id, phase_state in self.phase_states.items():
                    try:
                        if self._validate_phase_synchronization(phase_id, phase_state):
                            phase_state['state'] = PhaseState.SYNCHRONIZED
                            sync_results['synchronized_phases'].append(phase_id)
                            self.logger.info(f"Phase {phase_id} synchronized")
                        else:
                            sync_results['failed_phases'].append(phase_id)
                            self.logger.warning(f"Phase {phase_id} synchronization failed")
                            
                    except Exception as e:
                        self.logger.error(f"Synchronization error for {phase_id}: {e}")
                        sync_results['failed_phases'].append(phase_id)
                        
                # Determine overall sync status
                sync_results['overall_sync_status'] = len(sync_results['failed_phases']) == 0
                self.global_state['integration_status'] = (
                    'synchronized' if sync_results['overall_sync_status'] else 'partial_sync'
                )
                
                # Save synchronized state
                self._save_persistent_state()
                
                return sync_results
                
        except Exception as e:
            self.logger.error(f"Cross-phase synchronization error: {e}")
            sync_results['error'] = str(e)
            return sync_results
            
    def validate_integration_readiness(self) -> Dict[str, Any]:
        """Validate readiness for full integration"""
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'phase_readiness': {},
            'integration_score': 0.0,
            'ready_for_integration': False,
            'blocking_issues': []
        }
        
        try:
            with self._state_lock:
                total_score = 0
                phase_count = 0
                
                # Validate each phase readiness
                for phase_id, phase_state in self.phase_states.items():
                    readiness = self._validate_phase_readiness(phase_id, phase_state)
                    validation_results['phase_readiness'][phase_id] = readiness
                    
                    total_score += readiness['readiness_score']
                    phase_count += 1
                    
                    # Collect blocking issues
                    if readiness['blocking_issues']:
                        validation_results['blocking_issues'].extend(
                            [f"{phase_id}: {issue}" for issue in readiness['blocking_issues']]
                        )
                        
                # Calculate integration score
                validation_results['integration_score'] = total_score / phase_count if phase_count > 0 else 0.0
                validation_results['ready_for_integration'] = (
                    validation_results['integration_score'] >= 0.85 and
                    len(validation_results['blocking_issues']) == 0
                )
                
                return validation_results
                
        except Exception as e:
            self.logger.error(f"Integration readiness validation error: {e}")
            validation_results['error'] = str(e)
            return validation_results
            
    def export_state_snapshot(self) -> str:
        """Export complete state snapshot"""
        try:
            snapshot = {
                'export_timestamp': datetime.now().isoformat(),
                'global_state': self.global_state.copy(),
                'phase_states': {k: v.copy() for k, v in self.phase_states.items()},
                'system_info': {
                    'torch_version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
                }
            }
            
            # Save snapshot
            snapshot_path = self.state_dir / f"state_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot, f, indent=2)
                
            self.logger.info(f"State snapshot exported to {snapshot_path}")
            return str(snapshot_path)
            
        except Exception as e:
            self.logger.error(f"State snapshot export error: {e}")
            return ""
            
    def restore_state_snapshot(self, snapshot_path: str) -> bool:
        """Restore state from snapshot"""
        try:
            with open(snapshot_path, 'r') as f:
                snapshot = json.load(f)
                
            with self._state_lock:
                # Restore global state
                if 'global_state' in snapshot:
                    self.global_state.update(snapshot['global_state'])
                    
                # Restore phase states
                if 'phase_states' in snapshot:
                    for phase_id, state_data in snapshot['phase_states'].items():
                        if phase_id in self.phase_states:
                            self.phase_states[phase_id].update(state_data)
                            
                # Save restored state
                self._save_persistent_state()
                
            self.logger.info(f"State restored from {snapshot_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"State restoration error: {e}")
            return False
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all phases"""
        try:
            with self._state_lock:
                summary = {
                    'timestamp': datetime.now().isoformat(),
                    'overall_progress': self.global_state['overall_progress'],
                    'quality_gates_status': {
                        'passed': self.global_state['quality_gates_passed'],
                        'total': self.global_state['total_quality_gates'],
                        'percentage': (self.global_state['quality_gates_passed'] / 
                                     self.global_state['total_quality_gates'] * 100)
                    },
                    'phase_performance': {},
                    'critical_metrics': self.global_state['performance_metrics'].copy()
                }
                
                # Collect phase-specific performance
                for phase_id, phase_state in self.phase_states.items():
                    summary['phase_performance'][phase_id] = {
                        'state': phase_state['state'].value if isinstance(phase_state['state'], PhaseState) else str(phase_state['state']),
                        'metrics': phase_state['metrics'].copy(),
                        'error_count': len(phase_state['errors'])
                    }
                    
                return summary
                
        except Exception as e:
            self.logger.error(f"Performance summary error: {e}")
            return {'error': str(e)}
            
    def _update_phase_status(self, phase_id: str):
        """Update phase status based on current data"""
        phase_state = self.phase_states[phase_id]
        
        # Determine status based on data completeness and errors
        if phase_state['errors']:
            phase_state['state'] = PhaseState.FAILED
        elif phase_state['data']:
            # Check for completion indicators
            completion_indicators = [
                'model_exported',
                'quality_validated',
                'integration_complete',
                'ready_for_next_phase'
            ]
            
            if any(phase_state['data'].get(indicator, False) for indicator in completion_indicators):
                phase_state['state'] = PhaseState.COMPLETED
            else:
                phase_state['state'] = PhaseState.IN_PROGRESS
        else:
            phase_state['state'] = PhaseState.PENDING
            
    def _update_global_state(self):
        """Update global state based on phase states"""
        completed_phases = sum(
            1 for state in self.phase_states.values()
            if state['state'] in [PhaseState.COMPLETED, PhaseState.SYNCHRONIZED]
        )
        total_phases = len(self.phase_states)
        
        self.global_state['overall_progress'] = completed_phases / total_phases
        
        # Count quality gates passed
        quality_gates = 0
        for phase_state in self.phase_states.values():
            quality_gates += sum(
                1 for key, value in phase_state['data'].items()
                if 'quality' in key.lower() and value is True
            )
            
        self.global_state['quality_gates_passed'] = quality_gates
        
        # Update performance metrics
        all_metrics = {}
        for phase_state in self.phase_states.values():
            all_metrics.update(phase_state['metrics'])
            
        self.global_state['performance_metrics'] = all_metrics
        
    def _validate_phase_synchronization(self, phase_id: str, phase_state: Dict[str, Any]) -> bool:
        """Validate if a phase is ready for synchronization"""
        try:
            # Check basic requirements
            has_data = bool(phase_state['data'])
            no_critical_errors = not any('critical' in str(error).lower() for error in phase_state['errors'])
            recent_update = phase_state['last_update'] is not None
            
            return has_data and no_critical_errors and recent_update
            
        except Exception:
            return False
            
    def _validate_phase_readiness(self, phase_id: str, phase_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate phase readiness for integration"""
        readiness = {
            'readiness_score': 0.0,
            'blocking_issues': [],
            'quality_checks': {}
        }
        
        try:
            checks = []
            
            # Data completeness check
            data_complete = len(phase_state['data']) > 0
            checks.append(data_complete)
            readiness['quality_checks']['data_complete'] = data_complete
            
            if not data_complete:
                readiness['blocking_issues'].append("Missing phase data")
                
            # Error check
            no_errors = len(phase_state['errors']) == 0
            checks.append(no_errors)
            readiness['quality_checks']['no_errors'] = no_errors
            
            if not no_errors:
                readiness['blocking_issues'].append(f"Phase has {len(phase_state['errors'])} errors")
                
            # State check
            state_valid = phase_state['state'] in [PhaseState.COMPLETED, PhaseState.SYNCHRONIZED]
            checks.append(state_valid)
            readiness['quality_checks']['state_valid'] = state_valid
            
            if not state_valid:
                readiness['blocking_issues'].append(f"Phase state is {phase_state['state']}")
                
            # Metrics check
            has_metrics = len(phase_state['metrics']) > 0
            checks.append(has_metrics)
            readiness['quality_checks']['has_metrics'] = has_metrics
            
            if not has_metrics:
                readiness['blocking_issues'].append("Missing performance metrics")
                
            readiness['readiness_score'] = sum(checks) / len(checks) if checks else 0.0
            
        except Exception as e:
            readiness['blocking_issues'].append(f"Validation error: {str(e)}")
            
        return readiness
        
    def _save_persistent_state(self):
        """Save state to persistent storage"""
        try:
            state_file = self.state_dir / "persistent_state.json"
            state_data = {
                'global_state': self.global_state,
                'phase_states': {
                    k: {
                        **v,
                        'state': v['state'].value if isinstance(v['state'], PhaseState) else str(v['state'])
                    }
                    for k, v in self.phase_states.items()
                },
                'last_save': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"State persistence error: {e}")
            
    def _load_persistent_state(self):
        """Load state from persistent storage"""
        try:
            state_file = self.state_dir / "persistent_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    
                # Restore global state
                if 'global_state' in state_data:
                    self.global_state.update(state_data['global_state'])
                    
                # Restore phase states
                if 'phase_states' in state_data:
                    for phase_id, phase_data in state_data['phase_states'].items():
                        if phase_id in self.phase_states:
                            # Convert state string back to enum
                            if 'state' in phase_data:
                                try:
                                    phase_data['state'] = PhaseState(phase_data['state'])
                                except ValueError:
                                    phase_data['state'] = PhaseState.PENDING
                                    
                            self.phase_states[phase_id].update(phase_data)
                            
                self.logger.info("Persistent state loaded")
                
        except Exception as e:
            self.logger.warning(f"State loading error (starting fresh): {e}")

# Global state manager instance
_state_manager = None

def get_state_manager() -> CrossPhaseStateManager:
    """Get global state manager instance"""
    global _state_manager
    if _state_manager is None:
        _state_manager = CrossPhaseStateManager()
    return _state_manager

def update_phase_state(phase_id: str, state_data: Dict[str, Any], 
                      metrics: Optional[Dict[str, float]] = None,
                      errors: Optional[List[str]] = None) -> bool:
    """Convenience function to update phase state"""
    return get_state_manager().update_phase_state(phase_id, state_data, metrics, errors)

def get_integration_status() -> Dict[str, Any]:
    """Get current integration status across all phases"""
    state_manager = get_state_manager()
    return {
        'global_state': state_manager.get_global_state(),
        'performance_summary': state_manager.get_performance_summary(),
        'integration_readiness': state_manager.validate_integration_readiness()
    }
