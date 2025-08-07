"""Core federated learning infrastructure with privacy-preserving mechanisms.

This module implements the complete federated learning system including:
- FederatedTrainingCoordinator for distributed training management
- PrivacyEngine for differential privacy
- SecureAggregator for gradient combination
- ModelSynchronizer for distributed model management
- TrainingMonitor for progress tracking
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
import hashlib
import logging
import time
from typing import Any

import numpy as np
import torch
from torch import nn


class AggregationStrategy(Enum):
    """Federated learning aggregation strategies."""

    FEDAVG = auto()  # Standard federated averaging
    FEDPROX = auto()  # Proximal term for heterogeneity
    SCAFFOLD = auto()  # Control variates for drift
    FEDBN = auto()  # Batch norm adaptation
    CLUSTERED = auto()  # Cluster-based aggregation
    HIERARCHICAL = auto()  # Multi-tier aggregation


class PrivacyMechanism(Enum):
    """Privacy-preserving mechanisms."""

    NONE = auto()
    DIFFERENTIAL_PRIVACY = auto()
    SECURE_AGGREGATION = auto()
    HOMOMORPHIC_ENCRYPTION = auto()
    COMBINED = auto()


@dataclass
class PrivacyBudget:
    """Privacy budget tracking for differential privacy."""

    epsilon: float = 1.0
    delta: float = 1e-5
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    max_rounds: int = 100

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.epsilon - self.consumed_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.delta - self.consumed_delta)

    def consume(self, eps: float, delta: float) -> bool:
        """Consume privacy budget. Returns True if budget allows."""
        if (
            self.consumed_epsilon + eps <= self.epsilon
            and self.consumed_delta + delta <= self.delta
        ):
            self.consumed_epsilon += eps
            self.consumed_delta += delta
            return True
        return False


@dataclass
class ClientUpdate:
    """Update from a federated learning client."""

    client_id: str
    round_number: int
    model_version: str
    gradients: dict[str, torch.Tensor]
    num_samples: int
    metrics: dict[str, float]
    computation_time: float
    privacy_spent: tuple[float, float] = (0.0, 0.0)  # (epsilon, delta)
    client_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FederatedRound:
    """Information about a federated learning round."""

    round_number: int
    participating_clients: list[str]
    start_time: float
    end_time: float | None = None
    global_model_version: str = ""
    aggregated_metrics: dict[str, float] = field(default_factory=dict)
    privacy_consumed: tuple[float, float] = (0.0, 0.0)  # (epsilon, delta)
    round_config: dict[str, Any] = field(default_factory=dict)


class PrivacyEngine:
    """Privacy-preserving mechanisms for federated learning."""

    def __init__(
        self,
        mechanism: PrivacyMechanism = PrivacyMechanism.DIFFERENTIAL_PRIVACY,
        privacy_budget: PrivacyBudget | None = None,
    ):
        self.mechanism = mechanism
        self.privacy_budget = privacy_budget or PrivacyBudget()
        self.logger = logging.getLogger("PrivacyEngine")

    def add_noise_to_gradients(
        self,
        gradients: dict[str, torch.Tensor],
        sensitivity: float = 1.0,
        epsilon: float = 0.1,
    ) -> tuple[dict[str, torch.Tensor], tuple[float, float]]:
        """Add differential privacy noise to gradients."""
        if self.mechanism != PrivacyMechanism.DIFFERENTIAL_PRIVACY:
            return gradients, (0.0, 0.0)

        # Calculate noise scale using Gaussian mechanism
        delta = 1e-5
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon

        # Check privacy budget
        if not self.privacy_budget.consume(epsilon, delta):
            raise ValueError("Insufficient privacy budget")

        noisy_gradients = {}
        for param_name, gradient in gradients.items():
            # Add Gaussian noise
            noise = torch.normal(0, sigma, size=gradient.shape, device=gradient.device)
            noisy_gradients[param_name] = gradient + noise

        self.logger.info(f"Added DP noise: ε={epsilon:.3f}, σ={sigma:.3f}")
        return noisy_gradients, (epsilon, delta)

    def secure_aggregation_mask(
        self, gradients: dict[str, torch.Tensor], client_id: str, round_number: int
    ) -> dict[str, torch.Tensor]:
        """Apply secure aggregation masking (simplified implementation)."""
        if self.mechanism not in [
            PrivacyMechanism.SECURE_AGGREGATION,
            PrivacyMechanism.COMBINED,
        ]:
            return gradients

        # Generate deterministic mask based on client_id and round
        seed = hash(f"{client_id}_{round_number}") % (2**32)
        rng = torch.Generator()
        rng.manual_seed(seed)

        masked_gradients = {}
        for param_name, gradient in gradients.items():
            # Generate mask with same shape
            mask = (
                torch.randn(gradient.shape, generator=rng, device=gradient.device)
                * 0.01
            )
            masked_gradients[param_name] = gradient + mask

        return masked_gradients

    def validate_privacy_guarantees(self) -> dict[str, Any]:
        """Validate current privacy guarantees."""
        return {
            "mechanism": self.mechanism.name,
            "epsilon_budget": self.privacy_budget.epsilon,
            "delta_budget": self.privacy_budget.delta,
            "epsilon_consumed": self.privacy_budget.consumed_epsilon,
            "delta_consumed": self.privacy_budget.consumed_delta,
            "epsilon_remaining": self.privacy_budget.remaining_epsilon,
            "delta_remaining": self.privacy_budget.remaining_delta,
            "budget_exhausted": (
                self.privacy_budget.remaining_epsilon <= 0
                or self.privacy_budget.remaining_delta <= 0
            ),
        }


class SecureAggregator:
    """Secure gradient aggregation with Byzantine fault tolerance."""

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
        byzantine_threshold: float = 0.3,
    ):
        self.strategy = strategy
        self.byzantine_threshold = byzantine_threshold
        self.logger = logging.getLogger("SecureAggregator")

    def aggregate_gradients(
        self, updates: list[ClientUpdate], global_model_state: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Aggregate client gradients securely."""
        if not updates:
            return global_model_state, {}

        # Byzantine detection and filtering
        filtered_updates = self._detect_and_filter_byzantine(updates)

        if len(filtered_updates) == 0:
            self.logger.warning("All updates filtered as Byzantine")
            return global_model_state, {"byzantine_filtered": len(updates)}

        # Choose aggregation strategy
        if self.strategy == AggregationStrategy.FEDAVG:
            aggregated_state = self._federated_averaging(
                filtered_updates, global_model_state
            )
        elif self.strategy == AggregationStrategy.FEDPROX:
            aggregated_state = self._fedprox_aggregation(
                filtered_updates, global_model_state
            )
        elif self.strategy == AggregationStrategy.SCAFFOLD:
            aggregated_state = self._scaffold_aggregation(
                filtered_updates, global_model_state
            )
        else:
            aggregated_state = self._federated_averaging(
                filtered_updates, global_model_state
            )

        metrics = {
            "total_updates": len(updates),
            "filtered_updates": len(filtered_updates),
            "byzantine_detected": len(updates) - len(filtered_updates),
            "total_samples": sum(u.num_samples for u in filtered_updates),
        }

        return aggregated_state, metrics

    def _detect_and_filter_byzantine(
        self, updates: list[ClientUpdate]
    ) -> list[ClientUpdate]:
        """Detect and filter Byzantine updates using statistical methods."""
        if len(updates) <= 2:
            return updates  # Can't detect Byzantine with too few updates

        # Calculate gradient norms for each update
        gradient_norms = []
        for update in updates:
            total_norm = 0.0
            for param_name, gradient in update.gradients.items():
                total_norm += torch.norm(gradient).item() ** 2
            gradient_norms.append(np.sqrt(total_norm))

        # Use median-based outlier detection
        median_norm = np.median(gradient_norms)
        mad = np.median(np.abs(np.array(gradient_norms) - median_norm))

        # Filter outliers (simplified Byzantine detection)
        threshold = median_norm + 3 * mad  # 3-sigma rule

        filtered_updates = []
        for i, update in enumerate(updates):
            if gradient_norms[i] <= threshold:
                filtered_updates.append(update)
            else:
                self.logger.warning(
                    f"Filtered Byzantine update from {update.client_id}: "
                    f"norm={gradient_norms[i]:.3f} > threshold={threshold:.3f}"
                )

        return filtered_updates

    def _federated_averaging(
        self, updates: list[ClientUpdate], global_model_state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Standard federated averaging with sample weighting."""
        total_samples = sum(update.num_samples for update in updates)
        aggregated_state = {}

        for param_name in global_model_state:
            weighted_sum = torch.zeros_like(global_model_state[param_name])

            for update in updates:
                if param_name in update.gradients:
                    weight = update.num_samples / total_samples
                    # Apply gradient to get updated parameter
                    updated_param = (
                        global_model_state[param_name] + update.gradients[param_name]
                    )
                    weighted_sum += weight * updated_param

            aggregated_state[param_name] = weighted_sum

        return aggregated_state

    def _fedprox_aggregation(
        self, updates: list[ClientUpdate], global_model_state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """FedProx aggregation with proximal term."""
        # Start with standard averaging
        aggregated_state = self._federated_averaging(updates, global_model_state)

        # Add proximal regularization
        mu = 0.01  # Proximal term coefficient
        for param_name in aggregated_state:
            aggregated_state[param_name] = (
                aggregated_state[param_name] + mu * global_model_state[param_name]
            ) / (1 + mu)

        return aggregated_state

    def _scaffold_aggregation(
        self, updates: list[ClientUpdate], global_model_state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """SCAFFOLD aggregation with control variates (simplified)."""
        # Use standard averaging with drift correction
        aggregated_state = self._federated_averaging(updates, global_model_state)

        # Apply drift correction
        drift_correction = 0.1
        for param_name in aggregated_state:
            aggregated_state[param_name] = (1 - drift_correction) * aggregated_state[
                param_name
            ] + drift_correction * global_model_state[param_name]

        return aggregated_state


class ModelSynchronizer:
    """Manages model synchronization across distributed nodes."""

    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio
        self.model_versions = {}
        self.logger = logging.getLogger("ModelSynchronizer")

    def create_model_version(self, model_state: dict[str, torch.Tensor]) -> str:
        """Create a unique version identifier for model state."""
        # Hash model parameters for version ID
        param_bytes = b""
        for key in sorted(model_state.keys()):
            param_bytes += model_state[key].cpu().numpy().tobytes()

        version = hashlib.sha256(param_bytes).hexdigest()[:16]
        self.model_versions[version] = {k: v.clone() for k, v in model_state.items()}
        return version

    def compress_model_state(
        self, model_state: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Compress model state for efficient transmission."""
        compressed = {}

        for key, tensor in model_state.items():
            # Apply quantization and compression
            if tensor.dtype == torch.float32:
                # Quantize to half precision
                quantized = tensor.half()
                compressed[key] = {
                    "data": quantized.cpu().numpy().tobytes(),
                    "shape": list(tensor.shape),
                    "dtype": "float16",
                    "compression": "quantized",
                }
            else:
                # Keep original for non-float tensors
                compressed[key] = {
                    "data": tensor.cpu().numpy().tobytes(),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "compression": "none",
                }

        return compressed

    def decompress_model_state(
        self, compressed_state: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Decompress model state from compressed format."""
        decompressed = {}

        for key, compressed_tensor in compressed_state.items():
            data_bytes = compressed_tensor["data"]
            shape = compressed_tensor["shape"]
            dtype = compressed_tensor["dtype"]

            if dtype == "float16":
                # Decompress quantized tensor
                tensor = torch.frombuffer(data_bytes, dtype=torch.float16).reshape(
                    shape
                )
                decompressed[key] = tensor.float()  # Convert back to float32
            else:
                # Handle other dtypes
                if dtype.startswith("torch."):
                    torch_dtype = getattr(torch, dtype.split(".")[-1])
                else:
                    torch_dtype = torch.float32

                tensor = torch.frombuffer(data_bytes, dtype=torch_dtype).reshape(shape)
                decompressed[key] = tensor

        return decompressed

    def calculate_model_diff(
        self, old_state: dict[str, torch.Tensor], new_state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Calculate difference between model states."""
        diff = {}
        for key in old_state:
            if key in new_state:
                diff[key] = new_state[key] - old_state[key]
        return diff

    def apply_model_diff(
        self, base_state: dict[str, torch.Tensor], diff: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply model difference to base state."""
        updated_state = {}
        for key in base_state:
            if key in diff:
                updated_state[key] = base_state[key] + diff[key]
            else:
                updated_state[key] = base_state[key].clone()
        return updated_state


class TrainingMonitor:
    """Monitors federated training progress and health."""

    def __init__(self):
        self.round_history = []
        self.client_performance = {}
        self.privacy_metrics = {}
        self.convergence_metrics = {}
        self.logger = logging.getLogger("TrainingMonitor")

    def record_round(self, round_info: FederatedRound):
        """Record information about a completed round."""
        self.round_history.append(round_info)

        # Update privacy metrics
        eps_consumed, delta_consumed = round_info.privacy_consumed
        total_eps = self.privacy_metrics.get("total_epsilon_consumed", 0.0)
        total_delta = self.privacy_metrics.get("total_delta_consumed", 0.0)

        self.privacy_metrics.update(
            {
                "total_epsilon_consumed": total_eps + eps_consumed,
                "total_delta_consumed": total_delta + delta_consumed,
                "rounds_completed": len(self.round_history),
            }
        )

        # Update convergence metrics
        if round_info.aggregated_metrics:
            for metric_name, value in round_info.aggregated_metrics.items():
                if metric_name not in self.convergence_metrics:
                    self.convergence_metrics[metric_name] = []
                self.convergence_metrics[metric_name].append(
                    {
                        "round": round_info.round_number,
                        "value": value,
                        "timestamp": round_info.end_time or round_info.start_time,
                    }
                )

    def record_client_performance(
        self, client_id: str, round_number: int, metrics: dict[str, Any]
    ):
        """Record client performance metrics."""
        if client_id not in self.client_performance:
            self.client_performance[client_id] = []

        self.client_performance[client_id].append(
            {"round": round_number, "timestamp": time.time(), **metrics}
        )

    def get_training_health(self) -> dict[str, Any]:
        """Get overall training health assessment."""
        if not self.round_history:
            return {"status": "no_data", "health_score": 0.0}

        # Calculate participation rate
        total_clients = set()
        participating_clients = set()

        for round_info in self.round_history:
            for client_id in round_info.participating_clients:
                total_clients.add(client_id)
                participating_clients.add(client_id)

        participation_rate = (
            len(participating_clients) / len(total_clients) if total_clients else 0.0
        )

        # Calculate convergence stability
        convergence_score = 0.0
        if (
            "loss" in self.convergence_metrics
            and len(self.convergence_metrics["loss"]) >= 3
        ):
            recent_losses = [m["value"] for m in self.convergence_metrics["loss"][-3:]]
            convergence_score = (
                1.0
                if all(
                    recent_losses[i] >= recent_losses[i + 1]
                    for i in range(len(recent_losses) - 1)
                )
                else 0.5
            )

        # Calculate health score
        health_score = (participation_rate + convergence_score) / 2.0

        return {
            "status": (
                "healthy"
                if health_score > 0.7
                else "degraded" if health_score > 0.4 else "unhealthy"
            ),
            "health_score": health_score,
            "participation_rate": participation_rate,
            "convergence_score": convergence_score,
            "total_rounds": len(self.round_history),
            "active_clients": len(total_clients),
            "privacy_budget_consumed": {
                "epsilon": self.privacy_metrics.get("total_epsilon_consumed", 0.0),
                "delta": self.privacy_metrics.get("total_delta_consumed", 0.0),
            },
        }

    def get_convergence_analysis(self) -> dict[str, Any]:
        """Analyze training convergence patterns."""
        if not self.convergence_metrics:
            return {"status": "insufficient_data"}

        analysis = {}

        for metric_name, history in self.convergence_metrics.items():
            if len(history) < 3:
                continue

            values = [h["value"] for h in history]
            rounds = [h["round"] for h in history]

            # Calculate trend
            if len(values) >= 2:
                trend = "improving" if values[-1] < values[0] else "degrading"

                # Calculate rate of change
                rate_of_change = (values[-1] - values[0]) / (rounds[-1] - rounds[0])

                analysis[metric_name] = {
                    "trend": trend,
                    "rate_of_change": rate_of_change,
                    "current_value": values[-1],
                    "best_value": min(values) if metric_name == "loss" else max(values),
                    "stability": (
                        np.std(values[-5:]) if len(values) >= 5 else float("inf")
                    ),
                }

        return analysis

    def export_metrics(self) -> dict[str, Any]:
        """Export all collected metrics."""
        return {
            "round_history": [
                {
                    "round_number": r.round_number,
                    "participating_clients": r.participating_clients,
                    "duration": (r.end_time - r.start_time) if r.end_time else None,
                    "metrics": r.aggregated_metrics,
                    "privacy_consumed": r.privacy_consumed,
                }
                for r in self.round_history
            ],
            "client_performance": self.client_performance,
            "privacy_metrics": self.privacy_metrics,
            "convergence_metrics": self.convergence_metrics,
            "training_health": self.get_training_health(),
            "convergence_analysis": self.get_convergence_analysis(),
        }


class FederatedTrainingCoordinator:
    """Main coordinator for federated learning training."""

    def __init__(
        self,
        model: nn.Module,
        privacy_engine: PrivacyEngine | None = None,
        aggregator: SecureAggregator | None = None,
        synchronizer: ModelSynchronizer | None = None,
        monitor: TrainingMonitor | None = None,
        min_clients: int = 3,
        max_rounds: int = 100,
        round_timeout: int = 300,
    ):
        self.model = model
        self.privacy_engine = privacy_engine or PrivacyEngine()
        self.aggregator = aggregator or SecureAggregator()
        self.synchronizer = synchronizer or ModelSynchronizer()
        self.monitor = monitor or TrainingMonitor()

        self.min_clients = min_clients
        self.max_rounds = max_rounds
        self.round_timeout = round_timeout

        # Training state
        self.current_round = 0
        self.registered_clients = {}
        self.client_updates = {}
        self.current_model_version = self.synchronizer.create_model_version(
            self.model.state_dict()
        )

        # Configuration
        self.client_selection_fraction = 0.3
        self.learning_rate = 0.01

        self.logger = logging.getLogger("FederatedCoordinator")

    async def start_training(self) -> dict[str, Any]:
        """Start federated training process."""
        self.logger.info(f"Starting federated training for {self.max_rounds} rounds")

        training_results = {"rounds": [], "final_metrics": {}}

        for round_num in range(1, self.max_rounds + 1):
            self.logger.info(f"Starting round {round_num}/{self.max_rounds}")

            round_result = await self.execute_round()
            training_results["rounds"].append(round_result)

            if round_result.get("status") == "insufficient_clients":
                self.logger.warning("Stopping training due to insufficient clients")
                break

            # Check convergence
            if self._check_convergence():
                self.logger.info("Training converged early")
                break

        # Generate final results
        training_results["final_metrics"] = self.monitor.get_training_health()
        training_results["convergence_analysis"] = (
            self.monitor.get_convergence_analysis()
        )
        training_results["privacy_report"] = (
            self.privacy_engine.validate_privacy_guarantees()
        )

        return training_results

    async def execute_round(self) -> dict[str, Any]:
        """Execute a single federated learning round."""
        self.current_round += 1

        # Select clients
        selected_clients = self._select_clients()

        if len(selected_clients) < self.min_clients:
            return {
                "status": "insufficient_clients",
                "round_number": self.current_round,
                "available_clients": len(selected_clients),
                "required_clients": self.min_clients,
            }

        # Create round info
        round_info = FederatedRound(
            round_number=self.current_round,
            participating_clients=[c["client_id"] for c in selected_clients],
            start_time=time.time(),
            global_model_version=self.current_model_version,
            round_config={
                "learning_rate": self.learning_rate,
                "batch_size": 32,
                "local_epochs": 5,
                "aggregation_strategy": self.aggregator.strategy.name,
                "privacy_mechanism": self.privacy_engine.mechanism.name,
            },
        )

        # Prepare client updates storage
        self.client_updates[self.current_round] = []

        # Simulate client training (in real implementation, this would be distributed)
        for client_info in selected_clients:
            client_id = client_info["client_id"]
            update = await self._simulate_client_training(client_id, round_info)

            if update:
                await self.receive_client_update(update)

        # Wait for any remaining updates
        await asyncio.sleep(1.0)

        # Aggregate updates
        await self._aggregate_round_updates(round_info)

        # Record round completion
        round_info.end_time = time.time()
        self.monitor.record_round(round_info)

        return {
            "status": "completed",
            "round_number": self.current_round,
            "participating_clients": len(selected_clients),
            "duration": round_info.end_time - round_info.start_time,
            "aggregated_metrics": round_info.aggregated_metrics,
            "privacy_consumed": round_info.privacy_consumed,
        }

    async def receive_client_update(self, update: ClientUpdate) -> dict[str, Any]:
        """Receive and validate client update."""
        if update.round_number != self.current_round:
            return {"status": "wrong_round", "expected": self.current_round}

        # Validate update integrity
        if not self._validate_update(update):
            return {"status": "invalid_update"}

        # Apply privacy mechanisms
        if self.privacy_engine.mechanism != PrivacyMechanism.NONE:
            try:
                noisy_gradients, privacy_cost = (
                    self.privacy_engine.add_noise_to_gradients(
                        update.gradients, epsilon=0.01
                    )
                )
                update.gradients = noisy_gradients
                update.privacy_spent = privacy_cost
            except ValueError as e:
                self.logger.error(f"Privacy budget exceeded: {e}")
                return {"status": "privacy_budget_exceeded"}

        # Store update
        self.client_updates[self.current_round].append(update)

        # Record client performance
        self.monitor.record_client_performance(
            update.client_id,
            update.round_number,
            {
                "computation_time": update.computation_time,
                "num_samples": update.num_samples,
                "metrics": update.metrics,
                "privacy_spent": update.privacy_spent,
            },
        )

        self.logger.info(f"Received update from {update.client_id}")
        return {"status": "accepted"}

    async def _aggregate_round_updates(self, round_info: FederatedRound):
        """Aggregate all updates for the current round."""
        updates = self.client_updates[self.current_round]

        if not updates:
            self.logger.warning("No updates to aggregate")
            return

        # Secure aggregation
        aggregated_state, aggregation_metrics = self.aggregator.aggregate_gradients(
            updates, self.model.state_dict()
        )

        # Update global model
        self.model.load_state_dict(aggregated_state)
        self.current_model_version = self.synchronizer.create_model_version(
            aggregated_state
        )

        # Calculate aggregated metrics
        round_metrics = self._aggregate_client_metrics(updates)
        round_info.aggregated_metrics = {**round_metrics, **aggregation_metrics}

        # Calculate privacy consumption
        total_eps = sum(u.privacy_spent[0] for u in updates)
        total_delta = sum(u.privacy_spent[1] for u in updates)
        round_info.privacy_consumed = (total_eps, total_delta)

        self.logger.info(
            f"Aggregated {len(updates)} updates, "
            f"privacy cost: ε={total_eps:.3f}, δ={total_delta:.6f}"
        )

    def _select_clients(self) -> list[dict[str, Any]]:
        """Select clients for participation in current round."""
        # Filter eligible clients
        eligible_clients = [
            client_info
            for client_info in self.registered_clients.values()
            if client_info.get("active", False)
            and client_info.get("battery_level", 0) > 0.2
        ]

        if len(eligible_clients) <= self.min_clients:
            return eligible_clients

        # Select based on fraction and capabilities
        num_to_select = max(
            self.min_clients,
            int(len(eligible_clients) * self.client_selection_fraction),
        )

        # Weight-based selection
        weights = []
        for client in eligible_clients:
            weight = (
                client.get("reliability_score", 0.5)
                * client.get("battery_level", 0.5)
                * (1 + client.get("compute_power", 1))
            )
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(eligible_clients)) / len(eligible_clients)

        # Random selection with weights
        rng = np.random.default_rng()
        indices = rng.choice(
            len(eligible_clients), size=num_to_select, replace=False, p=weights
        )

        return [eligible_clients[i] for i in indices]

    def _validate_update(self, update: ClientUpdate) -> bool:
        """Validate client update for integrity and correctness."""
        # Check required fields
        if not all(
            [
                update.client_id,
                update.gradients,
                update.num_samples > 0,
                update.computation_time > 0,
            ]
        ):
            return False

        # Check gradient shapes match model
        model_state = self.model.state_dict()
        for param_name, gradient in update.gradients.items():
            if param_name not in model_state:
                return False
            if gradient.shape != model_state[param_name].shape:
                return False

        # Check for NaN or infinite values
        for gradient in update.gradients.values():
            if torch.isnan(gradient).any() or torch.isinf(gradient).any():
                return False

        return True

    def _aggregate_client_metrics(
        self, updates: list[ClientUpdate]
    ) -> dict[str, float]:
        """Aggregate metrics from client updates."""
        if not updates:
            return {}

        # Collect all metric names
        all_metrics = set()
        for update in updates:
            all_metrics.update(update.metrics.keys())

        # Aggregate each metric using weighted average
        aggregated = {}
        total_samples = sum(u.num_samples for u in updates)

        for metric_name in all_metrics:
            weighted_sum = 0.0
            for update in updates:
                if metric_name in update.metrics:
                    weight = update.num_samples / total_samples
                    weighted_sum += weight * update.metrics[metric_name]
            aggregated[metric_name] = weighted_sum

        return aggregated

    def _check_convergence(self) -> bool:
        """Check if training has converged."""
        convergence_analysis = self.monitor.get_convergence_analysis()

        if "loss" not in convergence_analysis:
            return False

        loss_analysis = convergence_analysis["loss"]

        # Consider converged if loss is stable and improving slowly
        return (
            loss_analysis.get("stability", float("inf")) < 0.01
            and abs(loss_analysis.get("rate_of_change", float("inf"))) < 0.001
        )

    async def _simulate_client_training(
        self, client_id: str, round_info: FederatedRound
    ) -> ClientUpdate | None:
        """Simulate client training (for testing purposes)."""
        # This is a placeholder - in real implementation,
        # clients would train independently and send updates

        # Simulate training time
        await asyncio.sleep(np.random.uniform(0.1, 1.0))

        # Generate fake gradients
        model_state = self.model.state_dict()
        fake_gradients = {}

        for param_name, param in model_state.items():
            # Generate small random gradients
            gradient = torch.randn_like(param) * 0.01
            fake_gradients[param_name] = gradient

        # Create update
        update = ClientUpdate(
            client_id=client_id,
            round_number=round_info.round_number,
            model_version=round_info.global_model_version,
            gradients=fake_gradients,
            num_samples=np.random.randint(50, 200),
            metrics={
                "loss": np.random.uniform(0.1, 1.0),
                "accuracy": np.random.uniform(0.7, 0.95),
            },
            computation_time=np.random.uniform(1.0, 5.0),
            client_metadata={
                "device": "mobile",
                "battery_level": np.random.uniform(0.3, 1.0),
            },
        )

        return update

    def register_client(self, client_id: str, client_info: dict[str, Any]):
        """Register a new client for federated learning."""
        self.registered_clients[client_id] = {
            "client_id": client_id,
            "registration_time": time.time(),
            "active": True,
            "reliability_score": 0.8,
            "battery_level": 1.0,
            "compute_power": 1.0,
            **client_info,
        }

        self.logger.info(f"Registered client {client_id}")

    def get_training_status(self) -> dict[str, Any]:
        """Get current training status."""
        return {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "registered_clients": len(self.registered_clients),
            "active_clients": sum(
                1 for c in self.registered_clients.values() if c.get("active", False)
            ),
            "current_model_version": self.current_model_version,
            "training_health": self.monitor.get_training_health(),
            "privacy_status": self.privacy_engine.validate_privacy_guarantees(),
        }
