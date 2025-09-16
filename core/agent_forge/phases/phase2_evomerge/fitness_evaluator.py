"""
Fitness evaluation for merged models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time
import psutil
from dataclasses import dataclass

@dataclass
class FitnessMetrics:
    """Container for fitness metrics."""
    perplexity: float
    accuracy: float
    inference_speed: float
    memory_usage: float
    composite_fitness: float
    individual_scores: Dict[str, float]

class FitnessEvaluator:
    """Evaluates fitness of merged models."""

    def __init__(self, config: Dict[str, Any], validation_data: Optional[Any] = None):
        self.config = config
        self.validation_data = validation_data
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Fitness weights
        self.weights = config.get('fitness_weights', {
            'perplexity': 0.4,
            'accuracy': 0.3,
            'speed': 0.2,
            'memory': 0.1
        })

        # Cache for fitness scores
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def evaluate(self, model: nn.Module) -> FitnessMetrics:
        """Comprehensive fitness evaluation."""
        # Check cache first
        model_hash = self._get_model_hash(model)
        if model_hash in self.cache:
            self.cache_hits += 1
            return self.cache[model_hash]

        self.cache_misses += 1

        # Move model to device
        model = model.to(self.device)
        model.eval()

        # Evaluate different aspects
        perplexity = self._evaluate_perplexity(model)
        accuracy = self._evaluate_accuracy(model)
        inference_speed = self._evaluate_speed(model)
        memory_usage = self._evaluate_memory(model)

        # Calculate composite fitness
        individual_scores = {
            'perplexity': 1.0 / (1.0 + perplexity),  # Lower is better
            'accuracy': accuracy,
            'speed': 1.0 / (1.0 + inference_speed),  # Lower is better
            'memory': 1.0 / (1.0 + memory_usage)  # Lower is better
        }

        composite_fitness = sum(
            self.weights.get(key.replace('_score', ''), 0) * score
            for key, score in individual_scores.items()
        )

        metrics = FitnessMetrics(
            perplexity=perplexity,
            accuracy=accuracy,
            inference_speed=inference_speed,
            memory_usage=memory_usage,
            composite_fitness=composite_fitness,
            individual_scores=individual_scores
        )

        # Cache the result
        self.cache[model_hash] = metrics

        return metrics

    def _evaluate_perplexity(self, model: nn.Module) -> float:
        """Evaluate model perplexity."""
        if self.validation_data is None:
            # Use synthetic data for testing
            return self._synthetic_perplexity(model)

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.validation_data:
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids', batch.get('inputs'))
                    labels = batch.get('labels', input_ids)
                else:
                    input_ids = labels = batch

                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(input_ids, labels=labels)

                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    # Calculate cross-entropy loss
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1)
                    )

                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.numel()

        # Calculate perplexity
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = np.exp(avg_loss)

        return perplexity

    def _evaluate_accuracy(self, model: nn.Module) -> float:
        """Evaluate model accuracy."""
        if self.validation_data is None:
            # Use synthetic evaluation
            return self._synthetic_accuracy(model)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.validation_data:
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids', batch.get('inputs'))
                    labels = batch.get('labels', input_ids)
                else:
                    input_ids = labels = batch

                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(input_ids)

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                # Get predictions
                predictions = torch.argmax(logits, dim=-1)

                # Calculate accuracy
                correct += (predictions == labels).sum().item()
                total += labels.numel()

        accuracy = correct / max(total, 1)
        return accuracy

    def _evaluate_speed(self, model: nn.Module) -> float:
        """Evaluate inference speed (ms per sample)."""
        model.eval()

        # Create dummy input
        batch_size = 1
        seq_length = 128

        # Try to get input dimensions from model
        if hasattr(model, 'config'):
            if hasattr(model.config, 'vocab_size'):
                vocab_size = model.config.vocab_size
            else:
                vocab_size = 32000
        else:
            vocab_size = 32000

        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(dummy_input)

        # Measure inference time
        num_iterations = 10
        times = []

        for _ in range(num_iterations):
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            start_time = time.perf_counter()

            with torch.no_grad():
                _ = model(dummy_input)

            torch.cuda.synchronize() if torch.cuda.is_available() else None

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        # Return average time
        avg_time = np.mean(times)
        return avg_time

    def _evaluate_memory(self, model: nn.Module) -> float:
        """Evaluate memory usage (MB)."""
        # Calculate model size
        param_memory = 0
        buffer_memory = 0

        for param in model.parameters():
            param_memory += param.numel() * param.element_size()

        for buffer in model.buffers():
            buffer_memory += buffer.numel() * buffer.element_size()

        total_memory_mb = (param_memory + buffer_memory) / (1024 * 1024)

        # If on GPU, also measure actual GPU memory
        if torch.cuda.is_available() and self.device != 'cpu':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Measure memory before and after model allocation
            initial_memory = torch.cuda.memory_allocated(self.device)

            # Create dummy forward pass to measure activation memory
            dummy_input = torch.randint(0, 32000, (1, 128)).to(self.device)

            with torch.no_grad():
                _ = model(dummy_input)

            final_memory = torch.cuda.memory_allocated(self.device)
            activation_memory_mb = (final_memory - initial_memory) / (1024 * 1024)

            total_memory_mb += activation_memory_mb

        return total_memory_mb

    def _synthetic_perplexity(self, model: nn.Module) -> float:
        """Calculate synthetic perplexity for testing."""
        # Create random input
        batch_size = 4
        seq_length = 128
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)

        with torch.no_grad():
            outputs = model(input_ids)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            # Calculate cross-entropy loss
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1)
            )

            perplexity = torch.exp(loss).item()

        return perplexity

    def _synthetic_accuracy(self, model: nn.Module) -> float:
        """Calculate synthetic accuracy for testing."""
        # Create random input and targets
        batch_size = 4
        seq_length = 128
        vocab_size = 32000

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)

        with torch.no_grad():
            outputs = model(input_ids)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == targets).float().mean().item()

        return accuracy

    def _get_model_hash(self, model: nn.Module) -> str:
        """Generate a hash for model caching."""
        # Simple hash based on parameter sum
        param_sum = sum(p.sum().item() for p in model.parameters())
        return f"{model.__class__.__name__}_{param_sum:.6f}"

    def batch_evaluate(self, models: list) -> list:
        """Evaluate multiple models in batch."""
        results = []

        for i, model in enumerate(models):
            print(f"Evaluating model {i+1}/{len(models)}")
            metrics = self.evaluate(model)
            results.append(metrics)

        return results

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }