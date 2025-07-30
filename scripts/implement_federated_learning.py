#!/usr/bin/env python3
"""Implement federated learning infrastructure for distributed training across mesh.

Supports heterogeneous devices and intermittent connectivity.
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

# Use the global random number generator
rng = np.random.default_rng()


class AggregationStrategy(Enum):
    """Federated learning aggregation strategies."""

    FEDAVG = auto()  # Standard federated averaging
    FEDPROX = auto()  # Proximal term for heterogeneity
    SCAFFOLD = auto()  # Control variates for drift
    FEDBN = auto()  # Batch norm adaptation
    CLUSTERED = auto()  # Cluster-based aggregation
    HIERARCHICAL = auto()  # Multi-tier aggregation


@dataclass
class ClientUpdate:
    """Update from a federated learning client."""

    client_id: str
    round_number: int
    model_id: str
    gradients: dict[str, torch.Tensor]
    num_samples: int
    metrics: dict[str, float]
    computation_time: float
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
    round_config: dict[str, Any] = field(default_factory=dict)


class FederatedLearningServer:
    """Central server for federated learning coordination."""

    def __init__(
        self,
        model: nn.Module,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
        min_clients: int = 3,
        max_rounds: int = 100,
    ) -> None:
        """Initialize federated learning server."""
        self.model = model
        self.aggregation_strategy = aggregation_strategy
        self.min_clients = min_clients
        self.max_rounds = max_rounds

        # State management
        self.current_round = 0
        self.client_updates: dict[int, list[ClientUpdate]] = {}
        self.round_history: list[FederatedRound] = []
        self.registered_clients: dict[str, dict[str, Any]] = {}

        # Model versioning
        self.model_versions: dict[str, dict[str, torch.Tensor]] = {}
        self.current_model_version = self._create_model_version()

        # Configuration
        self.round_timeout = 300  # 5 minutes
        self.client_selection_fraction = 0.3
        self.learning_rate = 0.01

        self.logger = logging.getLogger("FedServer")

    def _create_model_version(self) -> str:
        """Create a version identifier for current model state."""
        state_dict = self.model.state_dict()
        # Hash the model parameters
        param_bytes = b""
        for key in sorted(state_dict.keys()):
            param_bytes += state_dict[key].cpu().numpy().tobytes()

        version = hashlib.sha256(param_bytes).hexdigest()[:16]
        self.model_versions[version] = {k: v.clone() for k, v in state_dict.items()}
        return version

    async def start_round(self) -> dict[str, Any]:
        """Start a new federated learning round."""
        self.current_round += 1

        # Select clients for this round
        selected_clients = self._select_clients()

        if len(selected_clients) < self.min_clients:
            self.logger.warning(
                f"Not enough clients: {len(selected_clients)} < {self.min_clients}"
            )
            return {
                "status": "insufficient_clients", 
                "round_number": self.current_round,  # Provide for consistency
                "error": f"Need {self.min_clients} clients, got {len(selected_clients)}"
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
                "aggregation_strategy": self.aggregation_strategy.name,
            },
        )

        self.round_history.append(round_info)
        self.client_updates[self.current_round] = []

        # Prepare round configuration for clients
        round_config = {
            "round_number": self.current_round,
            "model_version": self.current_model_version,
            "model_state": self._compress_model_state(self.model.state_dict()),
            "round_config": round_info.round_config,
            "deadline": time.time() + self.round_timeout,
            "participating_clients": round_info.participating_clients,
        }

        self.logger.info(
            f"Started round {self.current_round} with {len(selected_clients)} clients"
        )

        return round_config

    def _select_clients(self) -> list[dict[str, Any]]:
        """Select clients for participation in current round."""
        min_battery_level = 0.2
        eligible_clients = [
            client_info
            for client_info in self.registered_clients.values()
            # Check client eligibility
            if (
                client_info.get("active", False)
                and client_info.get("battery_level", 0) > min_battery_level
            )
        ]

        # Random selection based on fraction
        num_to_select = max(
            self.min_clients,
            int(len(eligible_clients) * self.client_selection_fraction),
        )

        # Weighted selection based on reliability and resources
        if len(eligible_clients) <= num_to_select:
            return eligible_clients

        # Calculate selection weights
        weights = []
        for client in eligible_clients:
            weight = (
                client.get("reliability_score", 0.5)
                * client.get("battery_level", 0.5)
                * (1 + client.get("computation_power", 1))
            )
            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(eligible_clients)) / len(eligible_clients)

        # Select clients
        indices = rng.choice(
            len(eligible_clients), size=num_to_select, replace=False, p=weights
        )

        return [eligible_clients[i] for i in indices]

    async def receive_update(self, update: ClientUpdate) -> dict[str, Any]:
        """Receive update from a client."""
        # Validate update
        if update.round_number != self.current_round:
            return {"status": "wrong_round", "current_round": self.current_round}

        # Store update
        self.client_updates[self.current_round].append(update)

        self.logger.info(
            f"Received update from {update.client_id} for round {update.round_number}"
        )

        # Check if we have enough updates to aggregate
        current_round_info = self.round_history[-1]
        expected_clients = set(current_round_info.participating_clients)
        received_clients = {
            u.client_id for u in self.client_updates[self.current_round]
        }

        if len(received_clients) >= self.min_clients:
            # Check if timeout reached or all clients reported
            if (
                time.time() - current_round_info.start_time > self.round_timeout
                or expected_clients.issubset(received_clients)
            ):
                # Trigger aggregation
                await self._aggregate_updates()

        return {"status": "received", "round": self.current_round}

    async def _aggregate_updates(self) -> None:
        """Aggregate client updates into global model."""
        updates = self.client_updates[self.current_round]

        if not updates:
            return

        self.logger.info(
            f"Aggregating {len(updates)} updates for round {self.current_round}"
        )

        # Choose aggregation method
        if self.aggregation_strategy == AggregationStrategy.FEDAVG:
            aggregated_state = self._federated_averaging(updates)
        elif self.aggregation_strategy == AggregationStrategy.FEDPROX:
            aggregated_state = self._fedprox_aggregation(updates)
        elif self.aggregation_strategy == AggregationStrategy.SCAFFOLD:
            aggregated_state = self._scaffold_aggregation(updates)
        else:
            aggregated_state = self._federated_averaging(updates)  # Default

        # Update global model
        self.model.load_state_dict(aggregated_state)
        self.current_model_version = self._create_model_version()

        # Update round info
        current_round_info = self.round_history[-1]
        current_round_info.end_time = time.time()
        current_round_info.aggregated_metrics = self._aggregate_metrics(updates)

        self.logger.info(f"Round {self.current_round} complete")

    def _federated_averaging(
        self, updates: list[ClientUpdate]
    ) -> dict[str, torch.Tensor]:
        """Standard federated averaging algorithm."""
        # Calculate total samples
        total_samples = sum(update.num_samples for update in updates)

        # Initialize aggregated state
        aggregated_state = {}

        # Get model architecture
        model_state = self.model.state_dict()

        for param_name in model_state:
            # Weighted average of parameters
            aggregated_param = None

            for update in updates:
                if param_name in update.gradients:
                    weight = update.num_samples / total_samples

                    # Apply gradient to current parameter
                    updated_param = (
                        model_state[param_name]
                        + self.learning_rate * update.gradients[param_name]
                    )

                    if aggregated_param is None:
                        aggregated_param = weight * updated_param
                    else:
                        aggregated_param += weight * updated_param

            if aggregated_param is not None:
                aggregated_state[param_name] = aggregated_param
            else:
                # Keep original parameter if no updates
                aggregated_state[param_name] = model_state[param_name]

        return aggregated_state

    def _fedprox_aggregation(
        self, updates: list[ClientUpdate]
    ) -> dict[str, torch.Tensor]:
        """FedProx aggregation with proximal term."""
        # Start with standard averaging
        aggregated_state = self._federated_averaging(updates)

        # Add proximal term to handle heterogeneity
        mu = 0.01  # Proximal term weight
        current_state = self.model.state_dict()

        for param_name in aggregated_state:
            # Add proximal regularization
            aggregated_state[param_name] = (
                aggregated_state[param_name] + mu * current_state[param_name]
            ) / (1 + mu)

        return aggregated_state

    def _scaffold_aggregation(
        self, updates: list[ClientUpdate]
    ) -> dict[str, torch.Tensor]:
        """SCAFFOLD aggregation with control variates."""
        # This is a simplified version - full SCAFFOLD requires control variates
        # For now, use standard averaging with drift correction

        aggregated_state = self._federated_averaging(updates)

        # Estimate drift
        drift_correction = 0.1
        current_state = self.model.state_dict()

        for param_name in aggregated_state:
            # Apply drift correction
            aggregated_state[param_name] = (1 - drift_correction) * aggregated_state[
                param_name
            ] + drift_correction * current_state[param_name]

        return aggregated_state

    def _aggregate_metrics(self, updates: list[ClientUpdate]) -> dict[str, float]:
        """Aggregate metrics from client updates."""
        aggregated = {}

        # Collect all metric names
        all_metrics = set()
        for update in updates:
            all_metrics.update(update.metrics.keys())

        # Aggregate each metric
        for metric_name in all_metrics:
            values = []
            weights = []

            for update in updates:
                if metric_name in update.metrics:
                    values.append(update.metrics[metric_name])
                    weights.append(update.num_samples)

            if values:
                # Weighted average
                aggregated[metric_name] = np.average(values, weights=weights)

        return aggregated

    def _compress_model_state(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Compress model state for efficient transmission."""
        compressed = {}

        for key, tensor in state_dict.items():
            # Simple compression - convert to half precision
            compressed[key] = {
                "data": tensor.half().cpu().numpy().tolist(),
                "shape": list(tensor.shape),
                "dtype": "float16",
            }

        return compressed

    def register_client(self, client_id: str, client_info: dict[str, Any]) -> None:
        """Register a new client."""
        self.registered_clients[client_id] = {
            "client_id": client_id,
            "registration_time": time.time(),
            "active": True,
            **client_info,
        }

        self.logger.info(f"Registered client {client_id}")

    def get_current_model(self) -> tuple[nn.Module, str]:
        """Get current global model and version."""
        return self.model, self.current_model_version

    def get_round_history(self) -> list[dict[str, Any]]:
        """Get history of all rounds."""
        history = []

        for round_info in self.round_history:
            history.append(
                {
                    "round_number": round_info.round_number,
                    "num_clients": len(round_info.participating_clients),
                    "duration": (
                        (round_info.end_time - round_info.start_time)
                        if round_info.end_time
                        else None
                    ),
                    "metrics": round_info.aggregated_metrics,
                }
            )

        return history


class FederatedLearningClient:
    """Client for federated learning."""

    def __init__(
        self,
        client_id: str,
        local_model: nn.Module,
        local_data_loader: torch.utils.data.DataLoader,
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id
        self.local_model = local_model
        self.local_data_loader = local_data_loader
        self.device = device

        # Move model to device
        self.local_model.to(device)

        # Training configuration
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        # Client state
        self.current_round = 0
        self.training_history = []

        self.logger = logging.getLogger(f"FedClient-{client_id[:8]}")

    async def participate_in_round(self, round_config: dict[str, Any]) -> ClientUpdate:
        """Participate in a federated learning round."""
        # Validate round config before proceeding
        if "status" in round_config:
            if round_config["status"] == "insufficient_clients":
                raise ValueError("Server rejected round: insufficient clients")
            
        if "round_number" not in round_config:
            raise ValueError(f"Invalid round config: missing 'round_number'. Got: {list(round_config.keys())}")
            
        self.current_round = round_config["round_number"]

        # Update local model with global model
        self._load_compressed_state(round_config["model_state"])

        # Get training configuration
        config = round_config["round_config"]
        learning_rate = config.get("learning_rate", 0.01)
        local_epochs = config.get("local_epochs", 5)

        # Setup optimizer
        self.optimizer = torch.optim.SGD(
            self.local_model.parameters(), lr=learning_rate, momentum=0.9
        )

        # Save initial state for gradient calculation
        initial_state = {k: v.clone() for k, v in self.local_model.state_dict().items()}

        # Local training
        start_time = time.time()
        metrics = await self._train_local_model(local_epochs)
        computation_time = time.time() - start_time

        # Calculate gradients (parameter updates)
        gradients = {}
        final_state = self.local_model.state_dict()

        for param_name in initial_state:
            gradients[param_name] = final_state[param_name] - initial_state[param_name]

        # Create update
        update = ClientUpdate(
            client_id=self.client_id,
            round_number=self.current_round,
            model_id=round_config["model_version"],
            gradients=gradients,
            num_samples=len(self.local_data_loader.dataset),
            metrics=metrics,
            computation_time=computation_time,
            client_metadata={
                "device": self.device,
                "battery_level": self._get_battery_level(),
                "network_quality": self._get_network_quality(),
            },
        )

        self.logger.info(f"Completed round {self.current_round} training")

        return update

    async def _train_local_model(self, epochs: int) -> dict[str, float]:
        """Train model on local data."""
        self.local_model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for _epoch in range(epochs):
            epoch_loss = 0
            epoch_correct = 0

            for _batch_idx, (data, target) in enumerate(self.local_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += len(self.local_data_loader.dataset)

            # Allow other tasks to run
            await asyncio.sleep(0)

        # Calculate average metrics
        metrics = {
            "loss": total_loss / (epochs * len(self.local_data_loader)),
            "accuracy": total_correct / total_samples,
            "num_epochs": epochs,
            "num_batches": len(self.local_data_loader) * epochs,
        }

        return metrics

    def _load_compressed_state(self, compressed_state: dict[str, Any]) -> None:
        """Load compressed model state."""
        state_dict = {}

        for key, compressed_tensor in compressed_state.items():
            # Decompress tensor
            data = torch.tensor(compressed_tensor["data"], dtype=torch.float32).reshape(
                compressed_tensor["shape"]
            )

            state_dict[key] = data

        self.local_model.load_state_dict(state_dict)

    def _get_battery_level(self) -> float:
        """Get current battery level."""
        # In real implementation, query system
        return 0.8

    def _get_network_quality(self) -> float:
        """Get network quality metric."""
        # In real implementation, measure network
        return 0.9


async def test_federated_learning() -> bool:
    """Test federated learning implementation."""
    print("Testing Federated Learning Implementation...")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Create server
    global_model = SimpleModel()
    server = FederatedLearningServer(
        global_model, aggregation_strategy=AggregationStrategy.FEDAVG, min_clients=2
    )

    # Create clients with synthetic data
    clients = []
    for i in range(3):
        # Create synthetic dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10), torch.randint(0, 2, (100,))
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

        client = FederatedLearningClient(
            client_id=f"client_{i}",
            local_model=SimpleModel(),
            local_data_loader=dataloader,
        )

        clients.append(client)

        # Register client with server
        server.register_client(
            client.client_id,
            {
                "compute_power": rng.uniform(0.5, 1.0),
                "battery_level": rng.uniform(0.5, 1.0),
                "reliability_score": rng.uniform(0.7, 1.0),
            },
        )

    print(f"\nCreated {len(clients)} federated learning clients")

    # Simulate federated learning rounds
    num_rounds = 3

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1} ===")

        # Start round
        round_config = await server.start_round()

        if round_config.get("status") == "insufficient_clients":
            print("Not enough clients available")
            continue

        # Simulate client training
        participating_clients = round_config.get("participating_clients", [])

        for client in clients:
            if client.client_id in participating_clients:
                print(f"Client {client.client_id} training...")

                # Simulate network delay
                await asyncio.sleep(rng.uniform(0.1, 0.5))

                # Train and create update
                update = await client.participate_in_round(round_config)

                # Send update to server
                await server.receive_update(update)

                print(
                    f"Client {client.client_id} completed: "
                    f"loss={update.metrics['loss']:.4f}, "
                    f"acc={update.metrics['accuracy']:.4f}"
                )

        # Wait for aggregation
        await asyncio.sleep(0.5)

    # Get round history
    history = server.get_round_history()

    print("\n=== Federated Learning Summary ===")
    for round_info in history:
        if round_info["duration"]:
            print(
                f"Round {round_info['round_number']}: "
                f"{round_info['num_clients']} clients, "
                f"duration={round_info['duration']:.1f}s"
            )

            if round_info["metrics"]:
                print(f"  Aggregated metrics: {round_info['metrics']}")

    print("\n[CHECK] Federated learning testing complete!")
    return True


if __name__ == "__main__":
    # Test implementation
    asyncio.run(test_federated_learning())

    print("\n[CHECK] Federated learning implementation complete!")
