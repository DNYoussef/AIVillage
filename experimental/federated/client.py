"""Federated learning client implementation.

This module provides a complete federated learning client that can:
- Train models locally with privacy preservation
- Communicate with federated coordinators
- Handle intermittent connectivity
- Optimize for mobile and resource-constrained devices
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .core import ClientUpdate, PrivacyEngine, ModelSynchronizer


class FederatedLearningClient:
    """Advanced federated learning client with privacy and efficiency optimizations."""

    def __init__(
        self,
        client_id: str,
        local_model: nn.Module,
        local_data_loader: DataLoader,
        device: str = "cpu",
        privacy_engine: Optional[PrivacyEngine] = None,
        model_synchronizer: Optional[ModelSynchronizer] = None
    ):
        self.client_id = client_id
        self.local_model = local_model
        self.local_data_loader = local_data_loader
        self.device = device

        # Move model to device
        self.local_model.to(device)

        # Components
        self.privacy_engine = privacy_engine or PrivacyEngine()
        self.model_synchronizer = model_synchronizer or ModelSynchronizer()

        # Training configuration
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 0.01

        # Client state
        self.current_round = 0
        self.training_history = []
        self.battery_level = 1.0
        self.network_quality = 1.0
        self.compute_power = 1.0

        # Performance tracking
        self.local_training_times = []
        self.communication_times = []
        self.energy_consumption = []

        self.logger = logging.getLogger(f"FedClient-{client_id[:8]}")

    async def participate_in_round(
        self,
        round_config: Dict[str, Any],
        privacy_budget: Optional[Tuple[float, float]] = None
    ) -> ClientUpdate:
        """Participate in a federated learning round."""
        start_time = time.time()

        # Validate round configuration
        self._validate_round_config(round_config)

        self.current_round = round_config["round_number"]
        self.logger.info(f"Starting participation in round {self.current_round}")

        # Check if client should participate based on resources
        if not self._should_participate():
            raise ValueError("Client cannot participate due to resource constraints")

        # Update local model with global model
        await self._update_local_model(round_config)

        # Extract training configuration
        config = round_config["round_config"]
        local_epochs = config.get("local_epochs", 5)
        batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.01)

        # Setup optimizer
        self.optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )

        # Save initial model state for gradient calculation
        initial_state = {k: v.clone() for k, v in self.local_model.state_dict().items()}

        # Perform local training
        training_start = time.time()
        training_metrics = await self._train_local_model(local_epochs)
        training_time = time.time() - training_start

        self.local_training_times.append(training_time)

        # Calculate model updates (gradients)
        gradients = self._calculate_gradients(initial_state)

        # Apply privacy mechanisms if configured
        privacy_cost = (0.0, 0.0)
        if privacy_budget and self.privacy_engine:
            try:
                gradients, privacy_cost = self.privacy_engine.add_noise_to_gradients(
                    gradients,
                    epsilon=privacy_budget[0],
                    sensitivity=1.0
                )
            except ValueError as e:
                self.logger.warning(f"Privacy mechanism failed: {e}")

        # Create client update
        update = ClientUpdate(
            client_id=self.client_id,
            round_number=self.current_round,
            model_version=round_config["model_version"],
            gradients=gradients,
            num_samples=len(self.local_data_loader.dataset),
            metrics=training_metrics,
            computation_time=training_time,
            privacy_spent=privacy_cost,
            client_metadata=self._get_client_metadata()
        )

        # Update client state
        self._update_client_state(training_time)

        total_time = time.time() - start_time
        self.logger.info(
            f"Round {self.current_round} complete: "
            f"total_time={total_time:.2f}s, "
            f"training_time={training_time:.2f}s, "
            f"loss={training_metrics.get('loss', 0):.4f}"
        )

        return update

    async def _train_local_model(self, epochs: int) -> Dict[str, float]:
        """Train model on local data with efficiency optimizations."""
        self.local_model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = 0

        # Track energy consumption
        energy_start = self._estimate_energy_consumption()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (data, target) in enumerate(self.local_data_loader):
                # Check if we should continue training (battery/resource management)
                if not self._can_continue_training(epoch, batch_idx):
                    self.logger.info("Stopping training early due to resource constraints")
                    break

                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                batch_size = len(data)
                epoch_samples += batch_size

                # Calculate accuracy
                with torch.no_grad():
                    pred = output.argmax(dim=1, keepdim=True)
                    epoch_correct += pred.eq(target.view_as(pred)).sum().item()

                batch_count += 1

                # Periodic yielding for async operations
                if batch_idx % 10 == 0:
                    await asyncio.sleep(0.001)

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples

            # Allow other tasks to run between epochs
            await asyncio.sleep(0.01)

        # Calculate energy consumption
        energy_end = self._estimate_energy_consumption()
        energy_used = energy_end - energy_start
        self.energy_consumption.append(energy_used)

        # Calculate final metrics
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "epochs_completed": epochs,
            "batches_processed": batch_count,
            "samples_processed": total_samples,
            "energy_consumed": energy_used
        }

        # Record training history
        self.training_history.append({
            "round": self.current_round,
            "metrics": metrics,
            "timestamp": time.time()
        })

        return metrics

    def _calculate_gradients(
        self,
        initial_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate model gradients from initial to final state."""
        gradients = {}
        final_state = self.local_model.state_dict()

        for param_name in initial_state:
            if param_name in final_state:
                # Calculate parameter update as gradient
                gradients[param_name] = final_state[param_name] - initial_state[param_name]

        return gradients

    async def _update_local_model(self, round_config: Dict[str, Any]):
        """Update local model with global model state."""
        communication_start = time.time()

        try:
            # Decompress model state
            compressed_state = round_config["model_state"]
            model_state = self.model_synchronizer.decompress_model_state(compressed_state)

            # Load into local model
            self.local_model.load_state_dict(model_state)

            communication_time = time.time() - communication_start
            self.communication_times.append(communication_time)

            self.logger.debug(f"Model updated in {communication_time:.3f}s")

        except Exception as e:
            self.logger.error(f"Failed to update local model: {e}")
            raise

    def _validate_round_config(self, round_config: Dict[str, Any]):
        """Validate round configuration."""
        required_fields = ["round_number", "model_version", "model_state", "round_config"]

        for field in required_fields:
            if field not in round_config:
                raise ValueError(f"Missing required field: {field}")

    def _should_participate(self) -> bool:
        """Determine if client should participate based on current resources."""
        # Check battery level
        if self.battery_level < 0.2:
            self.logger.info("Skipping round due to low battery")
            return False

        # Check network quality
        if self.network_quality < 0.3:
            self.logger.info("Skipping round due to poor network")
            return False

        # Check compute resources
        if self.compute_power < 0.1:
            self.logger.info("Skipping round due to low compute power")
            return False

        return True

    def _can_continue_training(self, epoch: int, batch_idx: int) -> bool:
        """Check if training can continue based on current resources."""
        # Simplified resource check during training
        if self.battery_level < 0.15:  # Even lower threshold during training
            return False

        # Check if we're taking too long (mobile optimization)
        if hasattr(self, '_training_start_time'):
            elapsed = time.time() - self._training_start_time
            if elapsed > 30.0:  # Max 30 seconds of training
                return False

        return True

    def _get_client_metadata(self) -> Dict[str, Any]:
        """Get current client metadata."""
        return {
            "device": self.device,
            "battery_level": self.battery_level,
            "network_quality": self.network_quality,
            "compute_power": self.compute_power,
            "memory_usage": self._get_memory_usage(),
            "avg_training_time": np.mean(self.local_training_times) if self.local_training_times else 0,
            "avg_communication_time": np.mean(self.communication_times) if self.communication_times else 0,
            "total_energy_consumed": sum(self.energy_consumption),
            "rounds_participated": len(self.training_history)
        }

    def _update_client_state(self, training_time: float):
        """Update client state based on training performance."""
        # Simulate battery drain
        energy_factor = training_time / 10.0  # Normalize by expected training time
        battery_drain = 0.01 * energy_factor
        self.battery_level = max(0.0, self.battery_level - battery_drain)

        # Simulate network quality fluctuation
        self.network_quality = max(0.1, min(1.0,
            self.network_quality + np.random.normal(0, 0.05)
        ))

        # Simulate compute power (thermal throttling, etc.)
        if training_time > 5.0:  # Long training reduces compute power
            self.compute_power = max(0.5, self.compute_power - 0.05)
        else:
            self.compute_power = min(1.0, self.compute_power + 0.01)

    def _estimate_energy_consumption(self) -> float:
        """Estimate current energy consumption (simplified)."""
        # In real implementation, this would query system APIs
        return time.time() * 0.1  # Placeholder

    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        # In real implementation, query system memory
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            return torch.cuda.memory_allocated(self.device) / 1024**2  # MB
        return 0.0

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get client performance statistics."""
        return {
            "client_id": self.client_id,
            "rounds_participated": len(self.training_history),
            "avg_training_time": np.mean(self.local_training_times) if self.local_training_times else 0,
            "avg_communication_time": np.mean(self.communication_times) if self.communication_times else 0,
            "total_energy_consumed": sum(self.energy_consumption),
            "current_battery_level": self.battery_level,
            "current_network_quality": self.network_quality,
            "current_compute_power": self.compute_power,
            "training_history": self.training_history[-10:],  # Last 10 rounds
            "resource_efficiency": self._calculate_efficiency_score()
        }

    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score for this client."""
        if not self.training_history:
            return 0.0

        # Factors: training time, energy consumption, accuracy
        time_score = 1.0 / (1.0 + np.mean(self.local_training_times)) if self.local_training_times else 0.0
        energy_score = 1.0 / (1.0 + np.mean(self.energy_consumption)) if self.energy_consumption else 0.0

        # Accuracy score from recent training
        recent_accuracies = [
            h["metrics"].get("accuracy", 0.0)
            for h in self.training_history[-5:]
        ]
        accuracy_score = np.mean(recent_accuracies) if recent_accuracies else 0.0

        # Combined efficiency score
        efficiency = (time_score + energy_score + accuracy_score) / 3.0
        return min(1.0, max(0.0, efficiency))

    def simulate_mobile_constraints(
        self,
        initial_battery: float = 1.0,
        network_stability: float = 0.8,
        compute_variation: float = 0.1
    ):
        """Simulate mobile device constraints for testing."""
        self.battery_level = initial_battery
        self.network_quality = network_stability
        self.compute_power = 1.0 - compute_variation

        # Add some randomness
        self.network_quality += np.random.normal(0, 0.1)
        self.network_quality = max(0.1, min(1.0, self.network_quality))

    async def handle_connection_interruption(self, duration: float = 5.0):
        """Simulate handling of connection interruption."""
        self.logger.warning(f"Connection interrupted for {duration}s")

        # Set network quality to zero during interruption
        original_quality = self.network_quality
        self.network_quality = 0.0

        # Wait for reconnection
        await asyncio.sleep(duration)

        # Restore network quality (possibly degraded)
        self.network_quality = min(original_quality, 0.8)
        self.logger.info("Connection restored")

    def save_state(self, filepath: str):
        """Save client state for persistence."""
        state = {
            "client_id": self.client_id,
            "current_round": self.current_round,
            "battery_level": self.battery_level,
            "network_quality": self.network_quality,
            "compute_power": self.compute_power,
            "training_history": self.training_history,
            "performance_stats": self.get_performance_stats()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load client state from persistence."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.current_round = state.get("current_round", 0)
            self.battery_level = state.get("battery_level", 1.0)
            self.network_quality = state.get("network_quality", 1.0)
            self.compute_power = state.get("compute_power", 1.0)
            self.training_history = state.get("training_history", [])

            self.logger.info(f"Loaded state from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")


class MobileOptimizedClient(FederatedLearningClient):
    """Specialized client optimized for mobile devices."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Mobile-specific configurations
        self.max_training_time = 15.0  # Max 15 seconds
        self.battery_threshold = 0.25  # Higher threshold for mobile
        self.memory_limit = 256  # MB

    def _should_participate(self) -> bool:
        """Mobile-specific participation decision."""
        # More conservative participation for mobile
        if self.battery_level < self.battery_threshold:
            return False

        # Check memory constraints
        if self._get_memory_usage() > self.memory_limit:
            return False

        # Check if device is charging (prefer training when charging)
        charging = self._is_device_charging()
        if not charging and self.battery_level < 0.5:
            return False

        return super()._should_participate()

    def _can_continue_training(self, epoch: int, batch_idx: int) -> bool:
        """Mobile-specific training continuation check."""
        # Strict time limit for mobile
        if hasattr(self, '_training_start_time'):
            elapsed = time.time() - self._training_start_time
            if elapsed > self.max_training_time:
                return False

        # Check memory pressure
        if self._get_memory_usage() > self.memory_limit * 0.9:
            return False

        return super()._can_continue_training(epoch, batch_idx)

    def _is_device_charging(self) -> bool:
        """Check if mobile device is charging."""
        # In real implementation, query system power status
        return np.random.random() > 0.7  # 30% chance of charging

    async def adaptive_training(
        self,
        round_config: Dict[str, Any],
        privacy_budget: Optional[Tuple[float, float]] = None
    ) -> ClientUpdate:
        """Adaptive training that adjusts based on device conditions."""
        # Adjust training parameters based on device state
        config = round_config["round_config"].copy()

        # Reduce epochs if battery is low
        if self.battery_level < 0.4:
            config["local_epochs"] = max(1, config.get("local_epochs", 5) // 2)

        # Reduce batch size if memory is constrained
        if self._get_memory_usage() > self.memory_limit * 0.7:
            config["batch_size"] = max(16, config.get("batch_size", 32) // 2)

        # Update round config
        round_config["round_config"] = config

        return await self.participate_in_round(round_config, privacy_budget)


class EdgeOptimizedClient(FederatedLearningClient):
    """Client optimized for edge computing environments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Edge-specific configurations
        self.bandwidth_limit = 1024 * 1024  # 1MB/s
        self.latency_threshold = 100  # ms
        self.reliability_threshold = 0.95

    async def participate_in_round(
        self,
        round_config: Dict[str, Any],
        privacy_budget: Optional[Tuple[float, float]] = None
    ) -> ClientUpdate:
        """Edge-optimized participation with bandwidth management."""
        # Measure network conditions
        network_conditions = await self._measure_network_conditions()

        if not self._network_suitable(network_conditions):
            raise ValueError("Network conditions unsuitable for participation")

        # Compress model updates more aggressively for edge
        original_compression = self.model_synchronizer.compression_ratio
        self.model_synchronizer.compression_ratio = 0.8  # Higher compression

        try:
            update = await super().participate_in_round(round_config, privacy_budget)

            # Apply additional compression to gradients
            update.gradients = self._compress_gradients(update.gradients)

            return update

        finally:
            # Restore original compression ratio
            self.model_synchronizer.compression_ratio = original_compression

    async def _measure_network_conditions(self) -> Dict[str, float]:
        """Measure current network conditions."""
        # Simulate network measurement
        await asyncio.sleep(0.1)

        return {
            "bandwidth": np.random.uniform(0.5, 2.0) * 1024 * 1024,  # MB/s
            "latency": np.random.uniform(20, 200),  # ms
            "reliability": np.random.uniform(0.8, 1.0),
            "jitter": np.random.uniform(5, 50)  # ms
        }

    def _network_suitable(self, conditions: Dict[str, float]) -> bool:
        """Check if network conditions are suitable for training."""
        return (
            conditions["bandwidth"] > self.bandwidth_limit * 0.5 and
            conditions["latency"] < self.latency_threshold and
            conditions["reliability"] > self.reliability_threshold
        )

    def _compress_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply gradient compression for edge transmission."""
        compressed_gradients = {}

        for param_name, gradient in gradients.items():
            # Apply sparsification (keep only top-k elements)
            k = max(1, int(gradient.numel() * 0.1))  # Keep top 10%

            flat_grad = gradient.flatten()
            _, indices = torch.topk(torch.abs(flat_grad), k)

            # Create sparse gradient
            sparse_grad = torch.zeros_like(flat_grad)
            sparse_grad[indices] = flat_grad[indices]

            compressed_gradients[param_name] = sparse_grad.reshape(gradient.shape)

        return compressed_gradients
