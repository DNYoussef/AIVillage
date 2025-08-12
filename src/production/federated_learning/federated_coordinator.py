"""Distributed Federated Learning Coordinator.

Coordinates federated learning across distributed devices, integrating with
Sprint 6's evolution system and P2P infrastructure.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
import random
import time
from typing import Any
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ...core.p2p.p2p_node import P2PNode, PeerCapabilities
from ..evolution.infrastructure_aware_evolution import (
    InfrastructureAwareEvolution,
)

logger = logging.getLogger(__name__)


class TrainingRoundStatus(Enum):
    """Status of federated training round."""

    INITIALIZING = "initializing"
    PARTICIPANT_SELECTION = "participant_selection"
    MODEL_DISTRIBUTION = "model_distribution"
    LOCAL_TRAINING = "local_training"
    GRADIENT_COLLECTION = "gradient_collection"
    AGGREGATION = "aggregation"
    MODEL_UPDATE = "model_update"
    EVALUATION = "evaluation"
    COMPLETED = "completed"
    FAILED = "failed"


class ParticipantStatus(Enum):
    """Status of individual participant."""

    INVITED = "invited"
    ACCEPTED = "accepted"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    DROPPED = "dropped"


@dataclass
class TrainingParticipant:
    """Federated learning participant."""

    device_id: str
    capabilities: PeerCapabilities
    local_data_size: int = 0
    training_quality_score: float = 0.0
    participation_history: list[bool] = field(default_factory=list)
    contribution_score: float = 0.5
    privacy_budget_remaining: float = 1.0
    status: ParticipantStatus = ParticipantStatus.INVITED
    round_metrics: dict[str, float] = field(default_factory=dict)
    gradients: dict[str, torch.Tensor] | None = None


@dataclass
class FederatedTrainingRound:
    """Single round of federated training."""

    round_id: str
    round_number: int
    participants: list[TrainingParticipant]
    global_model_state: dict[str, torch.Tensor] | None = None
    aggregated_gradients: dict[str, torch.Tensor] | None = None
    status: TrainingRoundStatus = TrainingRoundStatus.INITIALIZING
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    target_participants: int = 10
    min_participants: int = 3
    training_config: dict[str, Any] = field(default_factory=dict)
    privacy_config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FederatedLearningConfig:
    """Configuration for federated learning system."""

    # Participant selection
    min_participants_per_round: int = 3
    max_participants_per_round: int = 20
    participation_fraction: float = 0.3  # 30% of available devices
    min_data_samples: int = 10
    min_device_resources: dict[str, float] = field(
        default_factory=lambda: {
            "ram_mb": 2048,
            "battery_percent": 50,
            "trust_score": 0.6,
            "network_stability": 0.7,
        }
    )

    # Training parameters
    local_epochs: int = 3
    local_batch_size: int = 8
    local_learning_rate: float = 0.001
    gradient_clipping: float = 1.0

    # Privacy settings
    differential_privacy_epsilon: float = 1.0
    differential_privacy_delta: float = 1e-5
    secure_aggregation_enabled: bool = True
    byzantine_robust: bool = True

    # System settings
    round_timeout_seconds: float = 300.0  # 5 minutes
    communication_timeout_seconds: float = 60.0
    max_rounds: int = 100
    convergence_threshold: float = 0.001

    # Hierarchical aggregation
    enable_hierarchical_aggregation: bool = True
    cluster_size_target: int = 5
    regional_aggregation: bool = True


class DistributedFederatedLearning:
    """Main coordinator for distributed federated learning."""

    def __init__(
        self,
        p2p_node: P2PNode,
        evolution_system: InfrastructureAwareEvolution | None = None,
        config: FederatedLearningConfig | None = None,
    ) -> None:
        self.p2p_node = p2p_node
        self.evolution_system = evolution_system
        self.config = config or FederatedLearningConfig()

        # Training state
        self.current_round: FederatedTrainingRound | None = None
        self.training_history: list[FederatedTrainingRound] = []
        self.global_model: nn.Module | None = None
        self.is_coordinator = False

        # Participant management
        self.available_participants: dict[str, TrainingParticipant] = {}
        self.participant_pool: set[str] = set()

        # Privacy and security
        self.privacy_budgets: dict[str, float] = {}
        self.secure_aggregation = None  # Will be initialized when needed

        # Performance tracking
        self.fl_stats = {
            "rounds_completed": 0,
            "total_participants": 0,
            "avg_round_time": 0.0,
            "convergence_rounds": 0,
            "privacy_budget_consumed": 0.0,
            "byzantine_attacks_detected": 0,
            "gradient_compression_ratio": 1.0,
        }

        # Evolution integration
        self.evolution_integration_enabled = evolution_system is not None

        # Register P2P handlers
        self._register_p2p_handlers()

        logger.info("DistributedFederatedLearning initialized")

    def _register_p2p_handlers(self) -> None:
        """Register P2P message handlers for federated learning."""
        # These would be registered with the P2P node's message handling system

    async def initialize_federated_learning(self, model: nn.Module, is_coordinator: bool = True) -> bool:
        """Initialize federated learning system."""
        logger.info(f"Initializing federated learning (coordinator: {is_coordinator})")

        try:
            self.global_model = model
            self.is_coordinator = is_coordinator

            # Discover potential participants
            await self._discover_participants()

            # Initialize privacy budgets
            self._initialize_privacy_budgets()

            # Setup secure aggregation if enabled
            if self.config.secure_aggregation_enabled:
                from .secure_aggregation import SecureAggregationProtocol

                self.secure_aggregation = SecureAggregationProtocol(self.p2p_node)
                await self.secure_aggregation.initialize()

            # Announce federated learning capability
            await self._announce_fl_capability()

            logger.info(
                f"Federated learning initialized with {len(self.available_participants)} potential participants"
            )
            return True

        except Exception as e:
            logger.exception(f"Federated learning initialization failed: {e}")
            return False

    async def _discover_participants(self) -> None:
        """Discover devices suitable for federated learning."""
        # Get suitable peers from P2P network
        suitable_peers = self.p2p_node.get_suitable_evolution_peers(min_count=1)

        # Add local device if suitable
        if self.p2p_node.local_capabilities and self._is_device_suitable_for_fl(self.p2p_node.local_capabilities):
            suitable_peers.insert(0, self.p2p_node.local_capabilities)

        # Create participant records
        for peer in suitable_peers:
            if self._is_device_suitable_for_fl(peer):
                participant = TrainingParticipant(
                    device_id=peer.device_id,
                    capabilities=peer,
                    contribution_score=self._calculate_initial_contribution_score(peer),
                )

                self.available_participants[peer.device_id] = participant
                self.participant_pool.add(peer.device_id)

        logger.info(f"Discovered {len(self.available_participants)} suitable FL participants")

    def _is_device_suitable_for_fl(self, capabilities: PeerCapabilities) -> bool:
        """Check if device is suitable for federated learning."""
        min_resources = self.config.min_device_resources

        # Memory constraint
        if capabilities.ram_mb < min_resources["ram_mb"]:
            return False

        # Battery constraint (for mobile devices)
        if capabilities.battery_percent is not None and capabilities.battery_percent < min_resources["battery_percent"]:
            return False

        # Trust score constraint
        if capabilities.trust_score < min_resources["trust_score"]:
            return False

        # Evolution capacity (proxy for network stability and compute)
        return not capabilities.evolution_capacity < min_resources["network_stability"]

    def _calculate_initial_contribution_score(self, capabilities: PeerCapabilities) -> float:
        """Calculate initial contribution score for participant."""
        score = 0.0

        # Resource contribution
        resource_score = (
            (capabilities.ram_mb / 8192) * 0.3  # Normalized to 8GB
            + (capabilities.cpu_cores / 8) * 0.2  # Normalized to 8 cores
            + capabilities.trust_score * 0.3
            + capabilities.evolution_capacity * 0.2
        )

        score += min(1.0, resource_score)

        # Network quality
        if capabilities.latency_ms < 50:
            score += 0.1
        elif capabilities.latency_ms > 200:
            score -= 0.1

        # Reliability bonus
        if capabilities.trust_score > 0.8:
            score += 0.1

        return max(0.1, min(1.0, score))

    def _initialize_privacy_budgets(self) -> None:
        """Initialize privacy budgets for all participants."""
        initial_budget = 10.0  # Total privacy budget

        for device_id in self.available_participants:
            self.privacy_budgets[device_id] = initial_budget
            self.available_participants[device_id].privacy_budget_remaining = initial_budget

    async def _announce_fl_capability(self) -> None:
        """Announce federated learning capability to network."""
        announcement = {
            "type": "FL_CAPABILITY_ANNOUNCEMENT",
            "coordinator": self.is_coordinator,
            "config": {
                "min_participants": self.config.min_participants_per_round,
                "max_participants": self.config.max_participants_per_round,
                "privacy_enabled": self.config.differential_privacy_epsilon > 0,
                "secure_aggregation": self.config.secure_aggregation_enabled,
            },
            "timestamp": time.time(),
        }

        await self.p2p_node.broadcast_to_peers("FL_CAPABILITY_ANNOUNCEMENT", announcement)

    async def run_distributed_training_round(self) -> FederatedTrainingRound:
        """Run a complete federated training round."""
        if not self.is_coordinator:
            msg = "Only coordinator can initiate training rounds"
            raise ValueError(msg)

        if not self.global_model:
            msg = "Global model not initialized"
            raise ValueError(msg)

        round_number = len(self.training_history) + 1
        round_id = f"fl_round_{round_number}_{uuid.uuid4().hex[:8]}"

        logger.info(f"Starting federated training round {round_number}")

        # Create training round
        training_round = FederatedTrainingRound(
            round_id=round_id,
            round_number=round_number,
            participants=[],
            global_model_state=self.global_model.state_dict(),
            training_config={
                "local_epochs": self.config.local_epochs,
                "local_batch_size": self.config.local_batch_size,
                "local_learning_rate": self.config.local_learning_rate,
                "gradient_clipping": self.config.gradient_clipping,
            },
            privacy_config={
                "epsilon": self.config.differential_privacy_epsilon,
                "delta": self.config.differential_privacy_delta,
            },
        )

        self.current_round = training_round

        try:
            # Phase 1: Participant Selection
            training_round.status = TrainingRoundStatus.PARTICIPANT_SELECTION
            selected_participants = await self._select_participants_for_round(training_round)

            if len(selected_participants) < self.config.min_participants_per_round:
                training_round.status = TrainingRoundStatus.FAILED
                msg = f"Insufficient participants: {len(selected_participants)} < {self.config.min_participants_per_round}"
                raise ValueError(msg)

            training_round.participants = selected_participants

            # Phase 2: Model Distribution
            training_round.status = TrainingRoundStatus.MODEL_DISTRIBUTION
            await self._distribute_global_model(training_round)

            # Phase 3: Local Training
            training_round.status = TrainingRoundStatus.LOCAL_TRAINING
            await self._coordinate_local_training(training_round)

            # Phase 4: Gradient Collection
            training_round.status = TrainingRoundStatus.GRADIENT_COLLECTION
            await self._collect_gradients(training_round)

            # Phase 5: Secure Aggregation
            training_round.status = TrainingRoundStatus.AGGREGATION
            await self._aggregate_gradients(training_round)

            # Phase 6: Model Update
            training_round.status = TrainingRoundStatus.MODEL_UPDATE
            await self._update_global_model(training_round)

            # Phase 7: Evaluation
            training_round.status = TrainingRoundStatus.EVALUATION
            await self._evaluate_round_results(training_round)

            # Complete round
            training_round.status = TrainingRoundStatus.COMPLETED
            training_round.end_time = time.time()

            # Update statistics
            self._update_fl_statistics(training_round)

            # Integration with evolution system
            if self.evolution_integration_enabled:
                await self._integrate_with_evolution(training_round)

            # Store round
            self.training_history.append(training_round)
            self.current_round = None

            logger.info(f"Federated training round {round_number} completed successfully")
            return training_round

        except Exception as e:
            logger.exception(f"Federated training round failed: {e}")
            training_round.status = TrainingRoundStatus.FAILED
            training_round.end_time = time.time()
            training_round.metrics["error"] = str(e)

            self.training_history.append(training_round)
            self.current_round = None

            raise

    async def _select_participants_for_round(self, training_round: FederatedTrainingRound) -> list[TrainingParticipant]:
        """Select participants for training round."""
        logger.info("Selecting participants for training round")

        # Calculate target number of participants
        available_count = len(self.available_participants)
        target_count = min(
            self.config.max_participants_per_round,
            max(
                self.config.min_participants_per_round,
                int(available_count * self.config.participation_fraction),
            ),
        )

        # Filter eligible participants
        eligible_participants = []
        for participant in self.available_participants.values():
            if self._is_participant_eligible(participant):
                eligible_participants.append(participant)

        if len(eligible_participants) < self.config.min_participants_per_round:
            msg = f"Insufficient eligible participants: {len(eligible_participants)}"
            raise ValueError(msg)

        # Select participants using contribution-based sampling
        selected_participants = self._sample_participants(eligible_participants, target_count)

        # Send invitations
        successful_participants = []
        for participant in selected_participants:
            success = await self._invite_participant(participant, training_round)
            if success:
                participant.status = ParticipantStatus.ACCEPTED
                successful_participants.append(participant)
            else:
                participant.status = ParticipantStatus.DROPPED

        logger.info(f"Selected {len(successful_participants)} participants for training")
        return successful_participants

    def _is_participant_eligible(self, participant: TrainingParticipant) -> bool:
        """Check if participant is eligible for current round."""
        # Privacy budget check
        if participant.privacy_budget_remaining < self.config.differential_privacy_epsilon:
            return False

        # Device availability check
        capabilities = participant.capabilities
        if not capabilities.is_suitable_for_evolution():
            return False

        # Historical performance check
        if len(participant.participation_history) >= 5:
            recent_success_rate = sum(participant.participation_history[-5:]) / 5
            if recent_success_rate < 0.3:  # Less than 30% success rate
                return False

        return True

    def _sample_participants(
        self, eligible_participants: list[TrainingParticipant], target_count: int
    ) -> list[TrainingParticipant]:
        """Sample participants based on contribution scores."""
        if len(eligible_participants) <= target_count:
            return eligible_participants

        # Probability-based sampling weighted by contribution score
        weights = [p.contribution_score for p in eligible_participants]
        total_weight = sum(weights)

        if total_weight == 0:
            # Uniform sampling fallback
            return random.sample(eligible_participants, target_count)

        # Weighted sampling without replacement
        selected = []
        remaining = eligible_participants.copy()
        remaining_weights = weights.copy()

        for _ in range(target_count):
            if not remaining:
                break

            # Normalize weights
            total = sum(remaining_weights)
            probabilities = [w / total for w in remaining_weights]

            # Sample one participant
            selected_idx = np.random.choice(len(remaining), p=probabilities)
            selected.append(remaining[selected_idx])

            # Remove from remaining
            remaining.pop(selected_idx)
            remaining_weights.pop(selected_idx)

        return selected

    async def _invite_participant(
        self, participant: TrainingParticipant, training_round: FederatedTrainingRound
    ) -> bool:
        """Invite participant to join training round."""
        invitation = {
            "type": "FL_TRAINING_INVITATION",
            "round_id": training_round.round_id,
            "round_number": training_round.round_number,
            "training_config": training_round.training_config,
            "privacy_config": training_round.privacy_config,
            "estimated_duration_minutes": 5,
            "data_requirements": {"min_samples": self.config.min_data_samples},
            "deadline": time.time() + self.config.communication_timeout_seconds,
        }

        if participant.device_id == self.p2p_node.node_id:
            # Local participation
            return await self._accept_local_participation(invitation)
        # Remote invitation
        return await self.p2p_node.send_to_peer(participant.device_id, invitation)

    async def _accept_local_participation(self, invitation: dict[str, Any]) -> bool:
        """Accept participation for local device."""
        # Check local resources and data availability
        # For now, simulate acceptance
        return True

    async def _distribute_global_model(self, training_round: FederatedTrainingRound) -> None:
        """Distribute global model to participants."""
        logger.info("Distributing global model to participants")

        # Serialize model state
        model_state_data = {}
        for name, param in training_round.global_model_state.items():
            model_state_data[name] = param.cpu().numpy().tolist()

        distribution_message = {
            "type": "FL_MODEL_DISTRIBUTION",
            "round_id": training_round.round_id,
            "model_state": model_state_data,
            "model_metadata": {
                "parameter_count": sum(p.numel() for p in self.global_model.parameters()),
                "model_architecture": self.global_model.__class__.__name__,
            },
        }

        # Send to all participants
        distribution_tasks = []
        for participant in training_round.participants:
            if participant.device_id != self.p2p_node.node_id:
                task = self.p2p_node.send_to_peer(participant.device_id, distribution_message)
                distribution_tasks.append(task)

        # Wait for distribution completion
        if distribution_tasks:
            results = await asyncio.gather(*distribution_tasks, return_exceptions=True)
            successful_distributions = sum(1 for r in results if r is True)
            logger.info(
                f"Model distributed to {successful_distributions}/{len(distribution_tasks)} remote participants"
            )

    async def _coordinate_local_training(self, training_round: FederatedTrainingRound) -> None:
        """Coordinate local training phase."""
        logger.info("Coordinating local training phase")

        # Send training start signal
        training_start_message = {
            "type": "FL_START_LOCAL_TRAINING",
            "round_id": training_round.round_id,
            "timeout_seconds": self.config.round_timeout_seconds,
        }

        # Broadcast to all participants
        await self.p2p_node.broadcast_to_peers("FL_START_LOCAL_TRAINING", training_start_message)

        # Start local training if participating
        local_participant = next(
            (p for p in training_round.participants if p.device_id == self.p2p_node.node_id),
            None,
        )

        if local_participant:
            await self._run_local_training(training_round, local_participant)

        # Wait for training completion
        await self._wait_for_training_completion(training_round)

    async def _run_local_training(
        self, training_round: FederatedTrainingRound, participant: TrainingParticipant
    ) -> None:
        """Run local training for this device."""
        logger.info("Running local training")

        try:
            participant.status = ParticipantStatus.TRAINING

            # Create local model copy
            local_model = self._create_local_model_copy()
            local_model.load_state_dict(training_round.global_model_state)

            # Get local dataset (simulated)
            local_dataset = self._get_local_dataset()

            # Train locally
            gradients = await self._train_model_locally(local_model, local_dataset, training_round.training_config)

            # Apply differential privacy noise
            if self.config.differential_privacy_epsilon > 0:
                gradients = self._add_differential_privacy_noise(
                    gradients,
                    self.config.differential_privacy_epsilon,
                    self.config.differential_privacy_delta,
                )

            # Store gradients
            participant.gradients = gradients
            participant.status = ParticipantStatus.COMPLETED

            # Update privacy budget
            participant.privacy_budget_remaining -= self.config.differential_privacy_epsilon

            logger.info("Local training completed successfully")

        except Exception as e:
            logger.exception(f"Local training failed: {e}")
            participant.status = ParticipantStatus.FAILED

    def _create_local_model_copy(self) -> nn.Module:
        """Create copy of global model for local training."""
        # For now, return reference to global model
        # In real implementation, this would create a proper deep copy
        return self.global_model

    def _get_local_dataset(self) -> Dataset:
        """Get local dataset for training."""

        # Simulate local dataset
        class MockDataset(Dataset):
            def __init__(self, size=100) -> None:
                self.size = size

            def __len__(self) -> int:
                return self.size

            def __getitem__(self, idx):
                # Return mock data
                return torch.randn(10), torch.randint(0, 2, (1,))

        return MockDataset()

    async def _train_model_locally(
        self, model: nn.Module, dataset: Dataset, config: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Train model locally and return gradients."""
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=config["local_learning_rate"])
        criterion = nn.CrossEntropyLoss()

        dataloader = DataLoader(dataset, batch_size=config["local_batch_size"], shuffle=True)

        # Store initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        # Local training loop
        for _epoch in range(config["local_epochs"]):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.squeeze())

                loss.backward()

                # Gradient clipping
                if config.get("gradient_clipping", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])

                optimizer.step()

        # Calculate gradients as parameter differences
        gradients = {}
        for name, param in model.named_parameters():
            gradients[name] = initial_params[name] - param

        return gradients

    def _add_differential_privacy_noise(
        self, gradients: dict[str, torch.Tensor], epsilon: float, delta: float
    ) -> dict[str, torch.Tensor]:
        """Add differential privacy noise to gradients."""
        # Simplified DP noise addition
        # In practice, this would use more sophisticated DP mechanisms

        noisy_gradients = {}
        sensitivity = 1.0  # L2 sensitivity bound

        for name, grad in gradients.items():
            # Gaussian mechanism
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            noise = torch.normal(0, sigma, size=grad.shape)

            noisy_gradients[name] = grad + noise

        return noisy_gradients

    async def _wait_for_training_completion(self, training_round: FederatedTrainingRound) -> None:
        """Wait for all participants to complete training."""
        logger.info("Waiting for training completion")

        timeout = training_round.start_time + self.config.round_timeout_seconds
        check_interval = 10.0  # Check every 10 seconds

        while time.time() < timeout:
            # Check completion status
            completed_count = sum(
                1
                for p in training_round.participants
                if p.status in [ParticipantStatus.COMPLETED, ParticipantStatus.FAILED]
            )

            if completed_count >= len(training_round.participants) * 0.8:  # 80% completion
                break

            await asyncio.sleep(check_interval)

        # Mark timed-out participants
        for participant in training_round.participants:
            if participant.status == ParticipantStatus.TRAINING:
                participant.status = ParticipantStatus.DROPPED

        completed_participants = [p for p in training_round.participants if p.status == ParticipantStatus.COMPLETED]

        logger.info(f"Training completed: {len(completed_participants)} participants")

    async def _collect_gradients(self, training_round: FederatedTrainingRound) -> None:
        """Collect gradients from all participants."""
        logger.info("Collecting gradients from participants")

        collection_message = {
            "type": "FL_GRADIENT_COLLECTION",
            "round_id": training_round.round_id,
            "deadline": time.time() + self.config.communication_timeout_seconds,
        }

        # Request gradients from remote participants
        for participant in training_round.participants:
            if participant.device_id != self.p2p_node.node_id and participant.status == ParticipantStatus.COMPLETED:
                await self.p2p_node.send_to_peer(participant.device_id, collection_message)

        # Wait for gradient collection
        await self._wait_for_gradient_collection(training_round)

    async def _wait_for_gradient_collection(self, training_round: FederatedTrainingRound) -> None:
        """Wait for gradient collection from participants."""
        timeout = time.time() + self.config.communication_timeout_seconds

        while time.time() < timeout:
            # Check how many gradients we have
            participants_with_gradients = sum(1 for p in training_round.participants if p.gradients is not None)

            # Need at least minimum participants with gradients
            if participants_with_gradients >= self.config.min_participants_per_round:
                break

            await asyncio.sleep(5.0)

        participants_with_gradients = sum(1 for p in training_round.participants if p.gradients is not None)

        logger.info(f"Collected gradients from {participants_with_gradients} participants")

    async def _aggregate_gradients(self, training_round: FederatedTrainingRound) -> None:
        """Aggregate gradients using secure aggregation."""
        logger.info("Aggregating gradients")

        # Get participants with gradients
        participants_with_gradients = [p for p in training_round.participants if p.gradients is not None]

        if len(participants_with_gradients) < self.config.min_participants_per_round:
            msg = f"Insufficient gradients for aggregation: {len(participants_with_gradients)}"
            raise ValueError(msg)

        # Perform aggregation
        if self.config.secure_aggregation_enabled and self.secure_aggregation:
            # Secure aggregation
            aggregated_gradients = await self._secure_aggregate_gradients(participants_with_gradients)
        else:
            # Simple averaging
            aggregated_gradients = self._simple_average_gradients(participants_with_gradients)

        # Byzantine robustness check
        if self.config.byzantine_robust:
            aggregated_gradients = self._apply_byzantine_robustness(participants_with_gradients, aggregated_gradients)

        training_round.aggregated_gradients = aggregated_gradients

        logger.info("Gradient aggregation completed")

    async def _secure_aggregate_gradients(self, participants: list[TrainingParticipant]) -> dict[str, torch.Tensor]:
        """Perform secure aggregation of gradients."""
        # Simplified secure aggregation
        # In practice, this would use cryptographic protocols

        return self._simple_average_gradients(participants)

    def _simple_average_gradients(self, participants: list[TrainingParticipant]) -> dict[str, torch.Tensor]:
        """Simple averaging of gradients."""
        if not participants:
            return {}

        # Initialize aggregated gradients
        aggregated = {}
        first_participant = participants[0]

        for name, grad in first_participant.gradients.items():
            aggregated[name] = torch.zeros_like(grad)

        # Sum all gradients
        for participant in participants:
            for name, grad in participant.gradients.items():
                aggregated[name] += grad

        # Average
        num_participants = len(participants)
        for name in aggregated:
            aggregated[name] /= num_participants

        return aggregated

    def _apply_byzantine_robustness(
        self,
        participants: list[TrainingParticipant],
        aggregated_gradients: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply Byzantine robustness to aggregated gradients."""
        # Simplified Byzantine robustness using gradient norms
        # In practice, this would use more sophisticated methods like Krum or trimmed mean

        gradient_norms = []
        for participant in participants:
            norm = 0.0
            for grad in participant.gradients.values():
                norm += torch.norm(grad).item() ** 2
            gradient_norms.append(np.sqrt(norm))

        # Detect outliers using interquartile range
        q1 = np.percentile(gradient_norms, 25)
        q3 = np.percentile(gradient_norms, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out Byzantine participants
        honest_participants = []
        byzantine_count = 0

        for i, participant in enumerate(participants):
            if lower_bound <= gradient_norms[i] <= upper_bound:
                honest_participants.append(participant)
            else:
                byzantine_count += 1

        if byzantine_count > 0:
            logger.warning(f"Detected {byzantine_count} potential Byzantine participants")
            self.fl_stats["byzantine_attacks_detected"] += byzantine_count

            # Re-aggregate with honest participants only
            if honest_participants:
                return self._simple_average_gradients(honest_participants)

        return aggregated_gradients

    async def _update_global_model(self, training_round: FederatedTrainingRound) -> None:
        """Update global model with aggregated gradients."""
        logger.info("Updating global model")

        if not training_round.aggregated_gradients:
            msg = "No aggregated gradients available"
            raise ValueError(msg)

        # Apply aggregated gradients to global model
        for name, param in self.global_model.named_parameters():
            if name in training_round.aggregated_gradients:
                param.data -= training_round.aggregated_gradients[name]

        # Update training round with new model state
        training_round.global_model_state = self.global_model.state_dict()

        logger.info("Global model updated successfully")

    async def _evaluate_round_results(self, training_round: FederatedTrainingRound) -> None:
        """Evaluate results of training round."""
        logger.info("Evaluating round results")

        # Calculate round metrics
        training_round.metrics = {
            "participants_invited": len(training_round.participants),
            "participants_completed": sum(
                1 for p in training_round.participants if p.status == ParticipantStatus.COMPLETED
            ),
            "completion_rate": (
                sum(1 for p in training_round.participants if p.status == ParticipantStatus.COMPLETED)
                / len(training_round.participants)
                if training_round.participants
                else 0
            ),
            "round_duration": (training_round.end_time or time.time()) - training_round.start_time,
            "privacy_budget_consumed": sum(
                self.config.differential_privacy_epsilon
                for p in training_round.participants
                if p.status == ParticipantStatus.COMPLETED
            ),
        }

        # Model evaluation (simplified)
        training_round.metrics["model_loss"] = self._evaluate_model_performance()

        # Update participant contribution scores
        self._update_contribution_scores(training_round)

        logger.info(f"Round evaluation completed: {training_round.metrics}")

    def _evaluate_model_performance(self) -> float:
        """Evaluate global model performance."""
        # Simplified model evaluation
        # In practice, this would evaluate on a test dataset
        return random.uniform(0.1, 0.5)  # Mock loss value

    def _update_contribution_scores(self, training_round: FederatedTrainingRound) -> None:
        """Update participant contribution scores based on round performance."""
        for participant in training_round.participants:
            if participant.status == ParticipantStatus.COMPLETED:
                # Positive contribution
                participant.contribution_score = min(1.0, participant.contribution_score + 0.05)
                participant.participation_history.append(True)
            else:
                # Negative contribution
                participant.contribution_score = max(0.1, participant.contribution_score - 0.02)
                participant.participation_history.append(False)

            # Keep only recent history
            if len(participant.participation_history) > 10:
                participant.participation_history.pop(0)

    def _update_fl_statistics(self, training_round: FederatedTrainingRound) -> None:
        """Update federated learning statistics."""
        self.fl_stats["rounds_completed"] += 1
        self.fl_stats["avg_round_time"] = (
            self.fl_stats["avg_round_time"] + training_round.metrics["round_duration"]
        ) / 2
        self.fl_stats["privacy_budget_consumed"] += training_round.metrics["privacy_budget_consumed"]

        # Check for convergence
        if len(self.training_history) >= 2:
            current_loss = training_round.metrics["model_loss"]
            previous_loss = self.training_history[-2].metrics.get("model_loss", float("inf"))

            if abs(current_loss - previous_loss) < self.config.convergence_threshold:
                self.fl_stats["convergence_rounds"] += 1

    async def _integrate_with_evolution(self, training_round: FederatedTrainingRound) -> None:
        """Integrate FL results with evolution system."""
        if not self.evolution_system:
            return

        logger.info("Integrating FL results with evolution system")

        # Trigger evolution based on FL results
        {
            "trigger_type": "federated_learning_round",
            "round_metrics": training_round.metrics,
            "model_performance": training_round.metrics.get("model_loss", 0.0),
            "participant_count": len(training_round.participants),
            "convergence_indicator": self.fl_stats["convergence_rounds"],
        }

        # This would trigger evolution in the evolution system
        # await self.evolution_system.trigger_evolution(evolution_trigger_data)

    def get_fl_status(self) -> dict[str, Any]:
        """Get current federated learning status."""
        current_round_info = None
        if self.current_round:
            current_round_info = {
                "round_id": self.current_round.round_id,
                "round_number": self.current_round.round_number,
                "status": self.current_round.status.value,
                "participants": len(self.current_round.participants),
                "elapsed_time": time.time() - self.current_round.start_time,
            }

        return {
            "initialized": self.global_model is not None,
            "is_coordinator": self.is_coordinator,
            "available_participants": len(self.available_participants),
            "current_round": current_round_info,
            "completed_rounds": len(self.training_history),
            "statistics": self.fl_stats.copy(),
            "config": {
                "min_participants": self.config.min_participants_per_round,
                "max_participants": self.config.max_participants_per_round,
                "privacy_epsilon": self.config.differential_privacy_epsilon,
                "secure_aggregation": self.config.secure_aggregation_enabled,
                "byzantine_robust": self.config.byzantine_robust,
            },
            "privacy_budgets": {device_id: budget for device_id, budget in self.privacy_budgets.items() if budget > 0},
        }

    async def implement_hierarchical_aggregation(self) -> dict[str, Any]:
        """Implement hierarchical aggregation for bandwidth efficiency."""
        if not self.config.enable_hierarchical_aggregation:
            return {"enabled": False, "reason": "hierarchical_aggregation_disabled"}

        logger.info("Implementing hierarchical aggregation")

        # Group participants into clusters
        clusters = self._create_participant_clusters()

        aggregation_results = {
            "clusters_created": len(clusters),
            "bandwidth_savings": 0.0,
            "aggregation_tiers": [],
        }

        # Local aggregation within clusters
        cluster_aggregates = []
        for cluster_id, cluster_participants in clusters.items():
            if len(cluster_participants) >= 2:
                cluster_aggregate = self._aggregate_cluster(cluster_participants)
                cluster_aggregates.append(cluster_aggregate)

                aggregation_results["aggregation_tiers"].append(
                    {
                        "tier": "local",
                        "cluster_id": cluster_id,
                        "participants": len(cluster_participants),
                    }
                )

        # Global aggregation of cluster aggregates
        if cluster_aggregates:
            self._aggregate_clusters(cluster_aggregates)
            aggregation_results["aggregation_tiers"].append({"tier": "global", "clusters": len(cluster_aggregates)})

        # Estimate bandwidth savings (60-80% typical)
        baseline_communications = len(self.available_participants)
        hierarchical_communications = len(clusters) + 1  # cluster heads + coordinator
        bandwidth_savings = 1.0 - (hierarchical_communications / baseline_communications)
        aggregation_results["bandwidth_savings"] = bandwidth_savings

        logger.info(f"Hierarchical aggregation: {bandwidth_savings:.1%} bandwidth savings")
        return aggregation_results

    def _create_participant_clusters(self) -> dict[str, list[TrainingParticipant]]:
        """Create participant clusters for hierarchical aggregation."""
        participants = list(self.available_participants.values())
        clusters = {}

        # Simple clustering by network latency and geographical proximity
        # In practice, this would use more sophisticated clustering algorithms

        cluster_id = 0
        current_cluster = []

        for participant in participants:
            current_cluster.append(participant)

            if len(current_cluster) >= self.config.cluster_size_target:
                clusters[f"cluster_{cluster_id}"] = current_cluster
                current_cluster = []
                cluster_id += 1

        # Add remaining participants to last cluster
        if current_cluster:
            if clusters:
                # Add to last cluster
                last_cluster_key = list(clusters.keys())[-1]
                clusters[last_cluster_key].extend(current_cluster)
            else:
                clusters[f"cluster_{cluster_id}"] = current_cluster

        return clusters

    def _aggregate_cluster(self, cluster_participants: list[TrainingParticipant]) -> dict[str, torch.Tensor]:
        """Aggregate gradients within a cluster."""
        # Filter participants with gradients
        participants_with_gradients = [p for p in cluster_participants if p.gradients is not None]

        if not participants_with_gradients:
            return {}

        return self._simple_average_gradients(participants_with_gradients)

    def _aggregate_clusters(self, cluster_aggregates: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Aggregate cluster-level aggregates into global aggregate."""
        if not cluster_aggregates:
            return {}

        # Initialize global aggregate
        global_aggregate = {}
        first_aggregate = cluster_aggregates[0]

        for name, grad in first_aggregate.items():
            global_aggregate[name] = torch.zeros_like(grad)

        # Sum cluster aggregates
        for cluster_aggregate in cluster_aggregates:
            for name, grad in cluster_aggregate.items():
                global_aggregate[name] += grad

        # Average
        num_clusters = len(cluster_aggregates)
        for name in global_aggregate:
            global_aggregate[name] /= num_clusters

        return global_aggregate
