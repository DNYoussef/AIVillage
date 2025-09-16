"""
Agent Forge Phase 5 - Training Validation System
Real-time monitoring and validation with comprehensive metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import numpy as np
import time
import logging
import threading
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
from pathlib import Path
import psutil
import gc

class ValidationMode(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"

class MetricType(Enum):
    ACCURACY = "accuracy"
    LOSS = "loss"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUC = "auc"
    PERPLEXITY = "perplexity"
    BLEU = "bleu"
    CUSTOM = "custom"

@dataclass
class ValidationConfig:
    """Validation configuration"""
    # Validation frequency
    validation_frequency: int = 1000  # steps
    validation_steps: int = 100
    validation_patience: int = 10

    # Metrics to track
    primary_metric: str = "accuracy"
    metrics_to_track: List[str] = field(default_factory=lambda: ["accuracy", "loss", "f1_score"])

    # Performance monitoring
    memory_monitoring: bool = True
    gpu_monitoring: bool = True
    throughput_monitoring: bool = True

    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4

    # Model checkpointing
    save_best_model: bool = True
    checkpoint_frequency: int = 5000

    # Real-time monitoring
    realtime_plotting: bool = False
    log_frequency: int = 100

    # Distributed validation
    distributed_validation: bool = False
    validation_world_size: int = 1

class MetricCalculator:
    """Calculate various training and validation metrics"""

    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate classification accuracy"""
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)
        correct = (predictions == targets).float()
        return correct.mean().item()

    @staticmethod
    def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """Calculate top-k accuracy"""
        _, top_k_preds = predictions.topk(k, dim=-1)
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=-1).float()
        return correct.mean().item()

    @staticmethod
    def f1_score(predictions: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> float:
        """Calculate F1 score"""
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)

        # Convert to numpy for sklearn
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        try:
            from sklearn.metrics import f1_score as sklearn_f1
            return sklearn_f1(targets_np, preds_np, average=average, zero_division=0)
        except ImportError:
            # Fallback implementation for binary classification
            tp = ((predictions == 1) & (targets == 1)).sum().float()
            fp = ((predictions == 1) & (targets == 0)).sum().float()
            fn = ((predictions == 0) & (targets == 1)).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            return f1.item()

    @staticmethod
    def precision_recall(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        """Calculate precision and recall"""
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=-1)

        tp = ((predictions == 1) & (targets == 1)).sum().float()
        fp = ((predictions == 1) & (targets == 0)).sum().float()
        fn = ((predictions == 0) & (targets == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        return precision.item(), recall.item()

    @staticmethod
    def perplexity(loss: float) -> float:
        """Calculate perplexity from cross-entropy loss"""
        return math.exp(min(loss, 100))  # Cap to prevent overflow

    @staticmethod
    def bleu_score(predictions: List[str], targets: List[str]) -> float:
        """Calculate BLEU score for text generation"""
        try:
            from nltk.translate.bleu_score import corpus_bleu
            # Convert strings to token lists
            pred_tokens = [pred.split() for pred in predictions]
            target_tokens = [[target.split()] for target in targets]
            return corpus_bleu(target_tokens, pred_tokens)
        except ImportError:
            logging.warning("NLTK not available for BLEU score calculation")
            return 0.0

class PerformanceMonitor:
    """Monitor system performance during training"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.gpu_available = torch.cuda.is_available()
        self.memory_history = deque(maxlen=1000)
        self.gpu_memory_history = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)

    def get_system_stats(self) -> Dict[str, float]:
        """Get current system performance statistics"""
        stats = {}

        # CPU and RAM
        if self.config.memory_monitoring:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            stats.update({
                'cpu_percent': cpu_percent,
                'ram_used_gb': memory.used / (1024**3),
                'ram_available_gb': memory.available / (1024**3),
                'ram_percent': memory.percent
            })

        # GPU memory
        if self.config.gpu_monitoring and self.gpu_available:
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            gpu_cached = torch.cuda.memory_reserved() / (1024**3)
            gpu_max = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            stats.update({
                'gpu_memory_used_gb': gpu_memory,
                'gpu_memory_cached_gb': gpu_cached,
                'gpu_memory_max_gb': gpu_max,
                'gpu_memory_percent': (gpu_memory / gpu_max) * 100
            })

            self.gpu_memory_history.append(gpu_memory)

        self.memory_history.append(stats.get('ram_used_gb', 0))

        return stats

    def get_throughput_stats(self, samples_processed: int, time_elapsed: float) -> Dict[str, float]:
        """Calculate throughput statistics"""
        if time_elapsed > 0:
            samples_per_second = samples_processed / time_elapsed
            self.throughput_history.append(samples_per_second)

            return {
                'samples_per_second': samples_per_second,
                'avg_samples_per_second': np.mean(list(self.throughput_history)),
                'max_samples_per_second': np.max(list(self.throughput_history)),
                'min_samples_per_second': np.min(list(self.throughput_history))
            }
        return {}

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        if len(self.memory_history) < 10:
            return False

        recent_memory = list(self.memory_history)[-10:]
        memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]

        # Check if memory usage is increasing rapidly
        if memory_trend > 0.1:  # 100MB per measurement
            return True

        # Check GPU memory if available
        if self.gpu_available and len(self.gpu_memory_history) >= 10:
            recent_gpu_memory = list(self.gpu_memory_history)[-10:]
            gpu_memory_trend = np.polyfit(range(len(recent_gpu_memory)), recent_gpu_memory, 1)[0]

            if gpu_memory_trend > 0.05:  # 50MB per measurement
                return True

        return False

class RealTimeValidator:
    """Real-time validation system with continuous monitoring"""

    def __init__(
        self,
        model: nn.Module,
        config: ValidationConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device

        # Monitoring components
        self.performance_monitor = PerformanceMonitor(config)
        self.metric_calculator = MetricCalculator()

        # Validation state
        self.step_count = 0
        self.last_validation_step = 0
        self.best_metric_value = float('-inf') if config.primary_metric != 'loss' else float('inf')
        self.no_improvement_count = 0

        # Metrics history
        self.metrics_history = defaultdict(list)
        self.performance_history = defaultdict(list)

        # Threading for real-time monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()

        # Results storage
        self.validation_results = []

    def start_monitoring(self):
        """Start real-time monitoring thread"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()

    def stop_monitoring_thread(self):
        """Stop real-time monitoring thread"""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while not self.stop_monitoring.is_set():
            try:
                # Collect system statistics
                system_stats = self.performance_monitor.get_system_stats()
                self.performance_history['timestamp'].append(time.time())

                for key, value in system_stats.items():
                    self.performance_history[key].append(value)

                # Check for memory pressure
                if self.performance_monitor.check_memory_pressure():
                    logging.warning("Memory pressure detected - considering cleanup")
                    self._cleanup_memory()

                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)

    def _cleanup_memory(self):
        """Perform memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validate_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable,
        mode: ValidationMode = ValidationMode.VALIDATION
    ) -> Dict[str, float]:
        """Perform single validation step"""
        self.model.eval()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        start_time = time.time()

        with torch.no_grad():
            # Forward pass
            if 'input' in batch:
                predictions = self.model(batch['input'])
            else:
                # Handle different batch formats
                predictions = self.model(**batch)

            # Calculate loss
            if 'target' in batch:
                loss = loss_fn(predictions, batch['target'])
            else:
                loss = loss_fn(batch)

        step_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, batch.get('target'), loss)

        # Add performance metrics
        batch_size = batch['input'].size(0) if 'input' in batch else 32
        throughput_stats = self.performance_monitor.get_throughput_stats(batch_size, step_time)
        metrics.update(throughput_stats)

        # Record metrics
        self._record_metrics(metrics, mode)

        return metrics

    def _calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor],
        loss: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate all configured metrics"""
        metrics = {'loss': loss.item()}

        if targets is not None:
            # Classification metrics
            if len(targets.shape) == 1 or targets.shape[-1] == 1:
                if 'accuracy' in self.config.metrics_to_track:
                    metrics['accuracy'] = self.metric_calculator.accuracy(predictions, targets)

                if 'top_5_accuracy' in self.config.metrics_to_track:
                    metrics['top_5_accuracy'] = self.metric_calculator.top_k_accuracy(
                        predictions, targets, k=5
                    )

                if 'f1_score' in self.config.metrics_to_track:
                    metrics['f1_score'] = self.metric_calculator.f1_score(predictions, targets)

                if 'precision' in self.config.metrics_to_track or 'recall' in self.config.metrics_to_track:
                    precision, recall = self.metric_calculator.precision_recall(predictions, targets)
                    metrics['precision'] = precision
                    metrics['recall'] = recall

            # Language modeling metrics
            if 'perplexity' in self.config.metrics_to_track:
                metrics['perplexity'] = self.metric_calculator.perplexity(loss.item())

        return metrics

    def _record_metrics(self, metrics: Dict[str, float], mode: ValidationMode):
        """Record metrics in history"""
        timestamp = time.time()

        for metric_name, value in metrics.items():
            key = f"{mode.value}_{metric_name}"
            self.metrics_history[key].append((timestamp, value))

        # Keep only recent history to manage memory
        max_history_length = 10000
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > max_history_length:
                self.metrics_history[key] = self.metrics_history[key][-max_history_length:]

    def run_validation(
        self,
        val_loader,
        loss_fn: Callable,
        num_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """Run complete validation"""
        if num_steps is None:
            num_steps = self.config.validation_steps

        all_metrics = defaultdict(list)
        total_samples = 0
        start_time = time.time()

        for step, batch in enumerate(val_loader):
            if step >= num_steps:
                break

            step_metrics = self.validate_step(batch, loss_fn, ValidationMode.VALIDATION)

            # Accumulate metrics
            for key, value in step_metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)

            batch_size = batch['input'].size(0) if 'input' in batch else 32
            total_samples += batch_size

        # Calculate averages
        avg_metrics = {}
        for key, values in all_metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)

        # Add validation summary
        total_time = time.time() - start_time
        avg_metrics['validation_time'] = total_time
        avg_metrics['total_samples'] = total_samples

        # Check for improvement
        self._check_improvement(avg_metrics)

        # Store results
        self.validation_results.append({
            'step': self.step_count,
            'timestamp': time.time(),
            'metrics': avg_metrics.copy()
        })

        return avg_metrics

    def _check_improvement(self, metrics: Dict[str, float]):
        """Check if validation metrics improved"""
        if self.config.primary_metric not in metrics:
            return

        current_value = metrics[self.config.primary_metric]
        is_better = (
            current_value > self.best_metric_value
            if self.config.primary_metric != 'loss'
            else current_value < self.best_metric_value
        )

        improvement = abs(current_value - self.best_metric_value)

        if is_better and improvement >= self.config.early_stopping_min_delta:
            self.best_metric_value = current_value
            self.no_improvement_count = 0
            logging.info(f"New best {self.config.primary_metric}: {current_value:.6f}")
        else:
            self.no_improvement_count += 1

    def should_stop_early(self) -> bool:
        """Check if early stopping criteria are met"""
        if not self.config.early_stopping_enabled:
            return False

        return self.no_improvement_count >= self.config.early_stopping_patience

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        summary = {
            'step_count': self.step_count,
            'best_metric_value': self.best_metric_value,
            'no_improvement_count': self.no_improvement_count,
            'should_stop_early': self.should_stop_early(),
            'total_validations': len(self.validation_results)
        }

        # Add recent metrics
        if self.validation_results:
            latest_result = self.validation_results[-1]
            summary['latest_metrics'] = latest_result['metrics']
            summary['latest_timestamp'] = latest_result['timestamp']

        # Add performance summary
        if self.performance_history:
            summary['system_performance'] = {
                'avg_cpu_percent': np.mean(self.performance_history.get('cpu_percent', [0])),
                'avg_ram_used_gb': np.mean(self.performance_history.get('ram_used_gb', [0])),
                'avg_gpu_memory_used_gb': np.mean(self.performance_history.get('gpu_memory_used_gb', [0]))
            }

        return summary

    def export_results(self, save_path: str):
        """Export validation results to file"""
        export_data = {
            'config': {
                'validation_frequency': self.config.validation_frequency,
                'metrics_to_track': self.config.metrics_to_track,
                'primary_metric': self.config.primary_metric
            },
            'validation_results': self.validation_results,
            'metrics_history': {
                key: list(values) for key, values in self.metrics_history.items()
            },
            'performance_history': {
                key: list(values) for key, values in self.performance_history.items()
            },
            'summary': self.get_validation_summary()
        }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logging.info(f"Validation results exported to {save_path}")

    def step(self):
        """Increment step counter"""
        self.step_count += 1

if __name__ == "__main__":
    # Example usage and testing
    import torch.nn as nn

    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

        def forward(self, x):
            return self.layers(x)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TestModel().to(device)

    config = ValidationConfig(
        validation_frequency=10,
        validation_steps=5,
        metrics_to_track=['accuracy', 'loss', 'f1_score'],
        primary_metric='accuracy'
    )

    validator = RealTimeValidator(model, config, device)

    # Start monitoring
    validator.start_monitoring()

    # Test loss function
    def test_loss_fn(predictions, targets):
        return nn.CrossEntropyLoss()(predictions, targets)

    # Create test data
    test_batches = []
    for _ in range(20):
        batch = {
            'input': torch.randn(32, 128).to(device),
            'target': torch.randint(0, 10, (32,)).to(device)
        }
        test_batches.append(batch)

    # Test validation steps
    print("Testing validation steps...")
    for i, batch in enumerate(test_batches[:10]):
        metrics = validator.validate_step(batch, test_loss_fn)
        validator.step()

        if i % 3 == 0:
            print(f"Step {i}: Accuracy = {metrics.get('accuracy', 0):.4f}, "
                  f"Loss = {metrics.get('loss', 0):.4f}")

    # Test full validation
    print("\nTesting full validation...")
    val_metrics = validator.run_validation(test_batches[10:], test_loss_fn, num_steps=5)
    print(f"Validation results: {val_metrics}")

    # Get summary
    summary = validator.get_validation_summary()
    print(f"\nValidation summary: {summary}")

    # Stop monitoring
    validator.stop_monitoring_thread()

    # Export results
    validator.export_results("test_validation_results.json")

    print("Validation system test completed successfully!")