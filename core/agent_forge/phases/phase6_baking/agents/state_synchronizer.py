"""
Phase 6 Baking - State Synchronizer Agent
Coordinates state across all 9 baking agents during model optimization
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"
    RECOVERING = "recovering"


@dataclass
class StateSnapshot:
    agent_id: str
    state: AgentState
    progress: float
    last_update: datetime
    metadata: Dict[str, Any]
    dependencies_met: bool
    errors: List[str]

    def to_dict(self):
        return {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'progress': self.progress,
            'last_update': self.last_update.isoformat(),
            'metadata': self.metadata,
            'dependencies_met': self.dependencies_met,
            'errors': self.errors
        }


class StateSynchronizer:
    """
    Synchronizes state across all Phase 6 baking agents
    Ensures consistency, handles conflicts, and maintains ordering
    """

    def __init__(self, num_agents: int = 9):
        self.num_agents = num_agents
        self.agent_states: Dict[str, StateSnapshot] = {}
        self.state_history: List[Dict] = []
        self.lock = threading.Lock()
        self.event_queue = asyncio.Queue()
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.checkpoint_dir = Path("checkpoints/state_sync")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize agent registry
        self.agent_registry = {
            'neural_optimizer': {'deps': [], 'priority': 1},
            'inference_accelerator': {'deps': ['neural_optimizer'], 'priority': 2},
            'quality_monitor': {'deps': [], 'priority': 1},
            'performance_profiler': {'deps': [], 'priority': 1},
            'orchestrator': {'deps': [], 'priority': 0},
            'state_synchronizer': {'deps': [], 'priority': 0},
            'deployment_validator': {'deps': ['inference_accelerator'], 'priority': 3},
            'integration_tester': {'deps': ['deployment_validator'], 'priority': 4},
            'completion_auditor': {'deps': ['integration_tester'], 'priority': 5}
        }

        self._build_dependency_graph()
        self._initialize_agents()

    def _build_dependency_graph(self):
        """Build dependency graph for agent coordination"""
        for agent_id, config in self.agent_registry.items():
            self.dependency_graph[agent_id] = set(config['deps'])

    def _initialize_agents(self):
        """Initialize all agent states"""
        for agent_id in self.agent_registry:
            self.agent_states[agent_id] = StateSnapshot(
                agent_id=agent_id,
                state=AgentState.IDLE,
                progress=0.0,
                last_update=datetime.now(),
                metadata={},
                dependencies_met=len(self.dependency_graph[agent_id]) == 0,
                errors=[]
            )

    def update_state(self, agent_id: str, state: AgentState,
                    progress: float = None, metadata: Dict = None) -> bool:
        """
        Update agent state with conflict resolution
        """
        with self.lock:
            if agent_id not in self.agent_states:
                logger.error(f"Unknown agent: {agent_id}")
                return False

            snapshot = self.agent_states[agent_id]

            # Validate state transition
            if not self._validate_transition(snapshot.state, state):
                logger.warning(f"Invalid state transition for {agent_id}: {snapshot.state} -> {state}")
                return False

            # Update state
            snapshot.state = state
            if progress is not None:
                snapshot.progress = min(max(progress, 0.0), 100.0)
            if metadata:
                snapshot.metadata.update(metadata)
            snapshot.last_update = datetime.now()

            # Record history
            self.state_history.append({
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'state': state.value,
                'progress': snapshot.progress
            })

            # Check and update dependencies
            self._update_dependencies(agent_id)

            # Checkpoint if needed
            if state in [AgentState.COMPLETED, AgentState.ERROR]:
                self._save_checkpoint()

            logger.info(f"State updated: {agent_id} -> {state.value} ({snapshot.progress:.1f}%)")
            return True

    def _validate_transition(self, current: AgentState, new: AgentState) -> bool:
        """Validate state transitions"""
        valid_transitions = {
            AgentState.IDLE: [AgentState.INITIALIZING, AgentState.ERROR],
            AgentState.INITIALIZING: [AgentState.PROCESSING, AgentState.WAITING, AgentState.ERROR],
            AgentState.PROCESSING: [AgentState.WAITING, AgentState.COMPLETED, AgentState.ERROR],
            AgentState.WAITING: [AgentState.PROCESSING, AgentState.ERROR],
            AgentState.COMPLETED: [AgentState.IDLE],  # Can restart
            AgentState.ERROR: [AgentState.RECOVERING, AgentState.IDLE],
            AgentState.RECOVERING: [AgentState.IDLE, AgentState.PROCESSING, AgentState.ERROR]
        }

        return new in valid_transitions.get(current, [])

    def _update_dependencies(self, completed_agent: str):
        """Update dependency status for dependent agents"""
        if self.agent_states[completed_agent].state != AgentState.COMPLETED:
            return

        for agent_id, deps in self.dependency_graph.items():
            if completed_agent in deps:
                # Check if all dependencies are met
                all_met = all(
                    self.agent_states[dep].state == AgentState.COMPLETED
                    for dep in deps
                )
                self.agent_states[agent_id].dependencies_met = all_met

                if all_met:
                    logger.info(f"Dependencies met for {agent_id}")

    def get_ready_agents(self) -> List[str]:
        """Get list of agents ready to process"""
        with self.lock:
            ready = []
            for agent_id, snapshot in self.agent_states.items():
                if (snapshot.state == AgentState.IDLE and
                    snapshot.dependencies_met and
                    not snapshot.errors):
                    ready.append(agent_id)

            # Sort by priority
            ready.sort(key=lambda x: self.agent_registry[x]['priority'])
            return ready

    def wait_for_agent(self, agent_id: str, timeout: float = 60.0) -> bool:
        """Wait for agent to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self.lock:
                if agent_id in self.agent_states:
                    state = self.agent_states[agent_id].state
                    if state == AgentState.COMPLETED:
                        return True
                    elif state == AgentState.ERROR:
                        return False

            time.sleep(0.5)

        return False

    def get_global_progress(self) -> float:
        """Calculate overall system progress"""
        with self.lock:
            if not self.agent_states:
                return 0.0

            total_progress = sum(s.progress for s in self.agent_states.values())
            return total_progress / len(self.agent_states)

    def handle_conflict(self, agent1: str, agent2: str, conflict_type: str) -> str:
        """
        Resolve state conflicts between agents
        Returns winning agent ID
        """
        with self.lock:
            state1 = self.agent_states.get(agent1)
            state2 = self.agent_states.get(agent2)

            if not state1 or not state2:
                return agent1 if state1 else agent2

            # Resolution strategies based on conflict type
            if conflict_type == "resource":
                # Priority-based resolution
                priority1 = self.agent_registry[agent1]['priority']
                priority2 = self.agent_registry[agent2]['priority']
                winner = agent1 if priority1 <= priority2 else agent2

            elif conflict_type == "state":
                # Progress-based resolution
                winner = agent1 if state1.progress >= state2.progress else agent2

            elif conflict_type == "dependency":
                # Dependency order resolution
                if agent2 in self.dependency_graph[agent1]:
                    winner = agent2  # agent1 depends on agent2
                elif agent1 in self.dependency_graph[agent2]:
                    winner = agent1  # agent2 depends on agent1
                else:
                    winner = agent1  # No dependency, use first

            else:
                # Default: timestamp-based
                winner = agent1 if state1.last_update <= state2.last_update else agent2

            logger.info(f"Conflict resolved: {conflict_type} between {agent1} and {agent2} -> {winner} wins")
            return winner

    def synchronize_checkpoints(self) -> bool:
        """Synchronize checkpoints across all agents"""
        with self.lock:
            try:
                checkpoint_data = {
                    'timestamp': datetime.now().isoformat(),
                    'global_progress': self.get_global_progress(),
                    'agent_states': {
                        agent_id: snapshot.to_dict()
                        for agent_id, snapshot in self.agent_states.items()
                    },
                    'history_length': len(self.state_history)
                }

                # Save synchronized checkpoint
                checkpoint_file = self.checkpoint_dir / f"sync_{int(time.time())}.pkl"
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)

                logger.info(f"Synchronized checkpoint saved: {checkpoint_file}")
                return True

            except Exception as e:
                logger.error(f"Checkpoint synchronization failed: {e}")
                return False

    def _save_checkpoint(self):
        """Save state checkpoint"""
        self.synchronize_checkpoints()

    def restore_from_checkpoint(self, checkpoint_file: Path) -> bool:
        """Restore state from checkpoint"""
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)

            # Restore agent states
            for agent_id, state_dict in data['agent_states'].items():
                self.agent_states[agent_id] = StateSnapshot(
                    agent_id=state_dict['agent_id'],
                    state=AgentState(state_dict['state']),
                    progress=state_dict['progress'],
                    last_update=datetime.fromisoformat(state_dict['last_update']),
                    metadata=state_dict['metadata'],
                    dependencies_met=state_dict['dependencies_met'],
                    errors=state_dict['errors']
                )

            logger.info(f"State restored from checkpoint: {checkpoint_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False

    def broadcast_event(self, event_type: str, data: Dict):
        """Broadcast event to all agents"""
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }

        # In real implementation, this would use message bus
        logger.info(f"Broadcasting event: {event_type}")

    def get_status_report(self) -> Dict:
        """Generate comprehensive status report"""
        with self.lock:
            return {
                'global_progress': self.get_global_progress(),
                'agents': {
                    agent_id: {
                        'state': snapshot.state.value,
                        'progress': snapshot.progress,
                        'dependencies_met': snapshot.dependencies_met,
                        'has_errors': len(snapshot.errors) > 0
                    }
                    for agent_id, snapshot in self.agent_states.items()
                },
                'ready_agents': self.get_ready_agents(),
                'completed_agents': [
                    aid for aid, s in self.agent_states.items()
                    if s.state == AgentState.COMPLETED
                ],
                'error_agents': [
                    aid for aid, s in self.agent_states.items()
                    if s.state == AgentState.ERROR
                ]
            }


if __name__ == "__main__":
    # Test state synchronizer
    sync = StateSynchronizer()

    # Simulate agent state updates
    sync.update_state('neural_optimizer', AgentState.INITIALIZING)
    sync.update_state('neural_optimizer', AgentState.PROCESSING, progress=50.0)
    sync.update_state('neural_optimizer', AgentState.COMPLETED, progress=100.0)

    # Check dependencies
    print("Ready agents:", sync.get_ready_agents())

    # Get status
    status = sync.get_status_report()
    print(f"Global progress: {status['global_progress']:.1f}%")
    print(f"Completed agents: {status['completed_agents']}")