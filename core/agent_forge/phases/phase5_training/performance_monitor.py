"""
Agent Forge Phase 5: Performance Monitor
========================================

Comprehensive performance monitoring system for distributed BitNet training
with real-time metrics, bottleneck detection, and optimization recommendations.

Key Features:
- Real-time training metrics
- GPU utilization monitoring
- Memory efficiency tracking
- Bottleneck detection
- Performance optimization recommendations
- NASA POT10 compliance monitoring
"""

import torch
import torch.distributed as dist
import psutil
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import threading
import queue
from datetime import datetime, timedelta


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""
    timestamp: float
    epoch: int
    step: int
    loss: float
    learning_rate: float
    gpu_memory_used: float
    gpu_memory_total: float
    cpu_usage: float
    ram_usage: float
    throughput: float
    batch_time: float


@dataclass
class TrainingMetrics:
    """Comprehensive training performance metrics."""
    total_steps: int = 0
    total_epochs: int = 0
    training_time: float = 0.0
    average_loss: float = 0.0
    best_loss: float = float('inf')
    throughput_samples_per_sec: float = 0.0
    gpu_utilization_avg: float = 0.0
    memory_efficiency: float = 0.0
    convergence_speed: float = 0.0
    bottlenecks_detected: List[str] = None

    def __post_init__(self):
        if self.bottlenecks_detected is None:
            self.bottlenecks_detected = []


class GPUMonitor:
    """GPU performance monitoring utility."""

    def __init__(self, device_ids: List[int]):
        self.device_ids = device_ids
        self.monitoring_active = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None

    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous GPU monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop GPU monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

    def _monitor_loop(self, interval: float) -> None:
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self.get_gpu_metrics()
                self.metrics_queue.put(metrics)
                time.sleep(interval)
            except Exception as e:
                logging.error(f"GPU monitoring error: {e}")

    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get current GPU metrics for all devices."""
        if not torch.cuda.is_available():
            return {}

        metrics = {}

        for device_id in self.device_ids:
            try:
                # Memory metrics
                memory_allocated = torch.cuda.memory_allocated(device_id)
                memory_reserved = torch.cuda.memory_reserved(device_id)
                memory_total = torch.cuda.get_device_properties(device_id).total_memory

                # Utilization metrics
                utilization_pct = (memory_allocated / memory_total) * 100

                metrics[f'gpu_{device_id}'] = {
                    'memory_allocated_gb': memory_allocated / (1024**3),
                    'memory_reserved_gb': memory_reserved / (1024**3),
                    'memory_total_gb': memory_total / (1024**3),
                    'utilization_pct': utilization_pct,
                    'temperature': self._get_gpu_temperature(device_id),
                    'power_usage': self._get_gpu_power(device_id)
                }

            except Exception as e:
                logging.error(f"Error monitoring GPU {device_id}: {e}")
                metrics[f'gpu_{device_id}'] = {'error': str(e)}

        return metrics

    def _get_gpu_temperature(self, device_id: int) -> Optional[float]:
        """Get GPU temperature (placeholder - requires nvidia-ml-py)."""
        # In production, use nvidia-ml-py for actual temperature monitoring
        return None

    def _get_gpu_power(self, device_id: int) -> Optional[float]:
        """Get GPU power usage (placeholder - requires nvidia-ml-py)."""
        # In production, use nvidia-ml-py for actual power monitoring
        return None

    def get_recent_metrics(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent GPU metrics from queue."""
        recent_metrics = []
        try:
            while not self.metrics_queue.empty() and len(recent_metrics) < count:
                recent_metrics.append(self.metrics_queue.get_nowait())
        except queue.Empty:
            pass

        return recent_metrics


class BottleneckDetector:
    """Detects performance bottlenecks in training pipeline."""

    def __init__(self, threshold_config: Dict[str, float] = None):
        self.thresholds = threshold_config or {
            'gpu_utilization_low': 50.0,  # GPU utilization below 50%
            'memory_utilization_high': 90.0,  # Memory above 90%
            'cpu_utilization_high': 80.0,  # CPU above 80%
            'throughput_variance_high': 0.3,  # Throughput variance above 30%
            'loss_stagnation_steps': 100,  # Loss not improving for 100 steps
            'gradient_norm_explosion': 10.0,  # Gradient norm above 10
            'learning_rate_too_low': 1e-6  # Learning rate below 1e-6
        }

        self.detection_history = defaultdict(list)
        self.active_bottlenecks = set()

    def detect_bottlenecks(
        self,
        performance_data: List[PerformanceSnapshot],
        gpu_metrics: Dict[str, Any]
    ) -> List[str]:
        """Detect current performance bottlenecks."""

        bottlenecks = []

        if len(performance_data) < 10:
            return bottlenecks  # Need sufficient data for detection

        recent_data = performance_data[-10:]

        # GPU utilization bottleneck
        if gpu_metrics:
            avg_gpu_util = np.mean([
                gpu_data.get('utilization_pct', 0)
                for gpu_data in gpu_metrics.values()
                if isinstance(gpu_data, dict) and 'utilization_pct' in gpu_data
            ])

            if avg_gpu_util < self.thresholds['gpu_utilization_low']:
                bottlenecks.append('gpu_underutilization')

        # Memory utilization bottleneck
        avg_memory_util = np.mean([snap.gpu_memory_used / snap.gpu_memory_total * 100
                                   for snap in recent_data if snap.gpu_memory_total > 0])

        if avg_memory_util > self.thresholds['memory_utilization_high']:
            bottlenecks.append('memory_pressure')

        # CPU utilization bottleneck
        avg_cpu_util = np.mean([snap.cpu_usage for snap in recent_data])
        if avg_cpu_util > self.thresholds['cpu_utilization_high']:
            bottlenecks.append('cpu_bottleneck')

        # Throughput variance bottleneck
        throughputs = [snap.throughput for snap in recent_data if snap.throughput > 0]
        if len(throughputs) > 1:
            throughput_cv = np.std(throughputs) / np.mean(throughputs)
            if throughput_cv > self.thresholds['throughput_variance_high']:
                bottlenecks.append('unstable_throughput')

        # Loss stagnation bottleneck
        if len(performance_data) >= self.thresholds['loss_stagnation_steps']:
            recent_losses = [snap.loss for snap in performance_data[-int(self.thresholds['loss_stagnation_steps']):]]
            loss_improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]

            if abs(loss_improvement) < 0.01:  # Less than 1% improvement
                bottlenecks.append('loss_stagnation')

        # Learning rate bottleneck
        current_lr = recent_data[-1].learning_rate
        if current_lr < self.thresholds['learning_rate_too_low']:
            bottlenecks.append('learning_rate_too_low')

        # Update detection history
        timestamp = time.time()
        for bottleneck in bottlenecks:
            self.detection_history[bottleneck].append(timestamp)
            self.active_bottlenecks.add(bottleneck)

        # Clean old detections (older than 10 minutes)
        cutoff_time = timestamp - 600
        for bottleneck in list(self.active_bottlenecks):
            self.detection_history[bottleneck] = [
                t for t in self.detection_history[bottleneck] if t > cutoff_time
            ]
            if not self.detection_history[bottleneck]:
                self.active_bottlenecks.discard(bottleneck)

        return bottlenecks

    def get_optimization_recommendations(
        self,
        bottlenecks: List[str]
    ) -> Dict[str, List[str]]:
        """Get optimization recommendations for detected bottlenecks."""

        recommendations = {}

        for bottleneck in bottlenecks:
            if bottleneck == 'gpu_underutilization':
                recommendations[bottleneck] = [
                    "Increase batch size to improve GPU utilization",
                    "Check for CPU preprocessing bottlenecks",
                    "Consider using mixed precision training",
                    "Optimize data loading pipeline"
                ]

            elif bottleneck == 'memory_pressure':
                recommendations[bottleneck] = [
                    "Reduce batch size to lower memory usage",
                    "Enable gradient checkpointing",
                    "Use gradient accumulation instead of larger batches",
                    "Consider model parallel training"
                ]

            elif bottleneck == 'cpu_bottleneck':
                recommendations[bottleneck] = [
                    "Increase number of data loader workers",
                    "Optimize data preprocessing pipeline",
                    "Use faster storage (SSD) for dataset",
                    "Consider CPU-optimized preprocessing"
                ]

            elif bottleneck == 'unstable_throughput':
                recommendations[bottleneck] = [
                    "Check for memory leaks",
                    "Stabilize batch sizes across devices",
                    "Monitor system resource contention",
                    "Review data loading consistency"
                ]

            elif bottleneck == 'loss_stagnation':
                recommendations[bottleneck] = [
                    "Increase learning rate",
                    "Apply learning rate scheduling",
                    "Check for gradient clipping issues",
                    "Consider different optimization algorithm"
                ]

            elif bottleneck == 'learning_rate_too_low':
                recommendations[bottleneck] = [
                    "Increase learning rate gradually",
                    "Use learning rate finder to optimize",
                    "Consider adaptive learning rate methods",
                    "Check for numerical precision issues"
                ]

        return recommendations


class TrainingMonitor:
    """
    Comprehensive training performance monitor for Agent Forge Phase 5.

    Provides real-time monitoring, bottleneck detection, and optimization
    recommendations for distributed BitNet training.
    """

    def __init__(self, config, device_ids: Optional[List[int]] = None):
        self.config = config
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))

        # Setup logging
        self.logger = logging.getLogger('training_monitor')

        # Performance tracking
        self.performance_history: List[PerformanceSnapshot] = []
        self.current_metrics = TrainingMetrics()

        # Monitoring components
        self.gpu_monitor = GPUMonitor(self.device_ids)
        self.bottleneck_detector = BottleneckDetector()

        # Timing and state
        self.epoch_start_time = 0.0
        self.step_start_time = 0.0
        self.training_start_time = time.time()

        # Early stopping tracking
        self.early_stopping_patience = getattr(config, 'early_stopping_patience', None)
        self.early_stopping_counter = 0
        self.best_validation_loss = float('inf')

        # Report generation
        self.report_interval = 100  # Report every 100 steps
        self.last_report_step = 0

        self.logger.info(f"Training monitor initialized for {len(self.device_ids)} GPUs")

    def start_epoch(self, epoch: int, num_batches: int) -> None:
        """Start monitoring for new epoch."""
        self.epoch_start_time = time.time()
        self.current_metrics.total_epochs = epoch
        self.logger.info(f"Starting epoch {epoch} monitoring ({num_batches} batches)")

    def end_epoch(self, epoch_loss: float) -> None:
        """End epoch monitoring and update metrics."""
        epoch_time = time.time() - self.epoch_start_time
        self.current_metrics.training_time += epoch_time

        # Update loss metrics
        if epoch_loss < self.current_metrics.best_loss:
            self.current_metrics.best_loss = epoch_loss

        self.logger.info(f"Epoch completed in {epoch_time:.2f}s, Loss: {epoch_loss:.6f}")

    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        step_time: float
    ) -> None:
        """Log performance metrics for training step."""

        # Get system metrics
        gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()

        # Calculate throughput (assuming batch size from config)
        batch_size = getattr(self.config, 'batch_size', 1)
        throughput = batch_size / step_time if step_time > 0 else 0.0

        # GPU memory metrics
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0

        if gpu_metrics:
            for gpu_data in gpu_metrics.values():
                if isinstance(gpu_data, dict) and 'memory_allocated_gb' in gpu_data:
                    gpu_memory_used += gpu_data['memory_allocated_gb']
                    gpu_memory_total += gpu_data['memory_total_gb']

        # Create performance snapshot
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            epoch=self.current_metrics.total_epochs,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            cpu_usage=cpu_usage,
            ram_usage=memory_info.percent,
            throughput=throughput,
            batch_time=step_time
        )

        self.performance_history.append(snapshot)
        self.current_metrics.total_steps += 1

        # Update running averages
        self._update_running_averages(snapshot)

        # Detect bottlenecks periodically
        if step % 10 == 0:  # Check every 10 steps
            bottlenecks = self.bottleneck_detector.detect_bottlenecks(
                self.performance_history[-100:],  # Last 100 snapshots
                gpu_metrics
            )

            if bottlenecks:
                self.current_metrics.bottlenecks_detected.extend(bottlenecks)
                self.logger.warning(f"Bottlenecks detected: {bottlenecks}")

        # Generate periodic reports
        if step - self.last_report_step >= self.report_interval:
            self._generate_performance_report(step)
            self.last_report_step = step

    def _update_running_averages(self, snapshot: PerformanceSnapshot) -> None:
        """Update running average metrics."""
        alpha = 0.1  # EMA factor

        # Update averages
        if self.current_metrics.average_loss == 0.0:
            self.current_metrics.average_loss = snapshot.loss
        else:
            self.current_metrics.average_loss = (
                (1 - alpha) * self.current_metrics.average_loss +
                alpha * snapshot.loss
            )

        if self.current_metrics.throughput_samples_per_sec == 0.0:
            self.current_metrics.throughput_samples_per_sec = snapshot.throughput
        else:
            self.current_metrics.throughput_samples_per_sec = (
                (1 - alpha) * self.current_metrics.throughput_samples_per_sec +
                alpha * snapshot.throughput
            )

        # GPU utilization average
        if snapshot.gpu_memory_total > 0:
            current_util = (snapshot.gpu_memory_used / snapshot.gpu_memory_total) * 100
            if self.current_metrics.gpu_utilization_avg == 0.0:
                self.current_metrics.gpu_utilization_avg = current_util
            else:
                self.current_metrics.gpu_utilization_avg = (
                    (1 - alpha) * self.current_metrics.gpu_utilization_avg +
                    alpha * current_util
                )

    def should_early_stop(self, validation_loss: float) -> bool:
        """Check if early stopping criteria are met."""
        if self.early_stopping_patience is None:
            return False

        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {self.early_stopping_counter} epochs without improvement")
                return True

        return False

    def get_epoch_time(self) -> float:
        """Get current epoch time."""
        return time.time() - self.epoch_start_time

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage metrics."""
        gpu_metrics = self.gpu_monitor.get_gpu_metrics()
        memory_info = psutil.virtual_memory()

        memory_usage = {
            'ram_usage_pct': memory_info.percent,
            'ram_available_gb': memory_info.available / (1024**3)
        }

        # Add GPU memory metrics
        for gpu_id, metrics in gpu_metrics.items():
            if isinstance(metrics, dict) and 'memory_allocated_gb' in metrics:
                memory_usage[f'{gpu_id}_memory_gb'] = metrics['memory_allocated_gb']
                memory_usage[f'{gpu_id}_utilization_pct'] = metrics['utilization_pct']

        return memory_usage

    def _generate_performance_report(self, current_step: int) -> None:
        """Generate periodic performance report."""
        if len(self.performance_history) < 10:
            return

        recent_snapshots = self.performance_history[-100:]

        # Calculate statistics
        avg_loss = np.mean([s.loss for s in recent_snapshots])
        avg_throughput = np.mean([s.throughput for s in recent_snapshots])
        avg_batch_time = np.mean([s.batch_time for s in recent_snapshots])
        avg_gpu_util = np.mean([
            (s.gpu_memory_used / s.gpu_memory_total) * 100
            for s in recent_snapshots if s.gpu_memory_total > 0
        ])

        # Bottleneck summary
        active_bottlenecks = list(self.bottleneck_detector.active_bottlenecks)

        self.logger.info(
            f"Step {current_step} Performance Report:\n"
            f"  Average Loss: {avg_loss:.6f}\n"
            f"  Throughput: {avg_throughput:.1f} samples/sec\n"
            f"  Batch Time: {avg_batch_time:.3f}s\n"
            f"  GPU Utilization: {avg_gpu_util:.1f}%\n"
            f"  Active Bottlenecks: {active_bottlenecks}"
        )

    def save_metrics(self, checkpoint_dir: str, epoch: int) -> None:
        """Save performance metrics to disk."""
        metrics_dir = Path(checkpoint_dir) / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Save current metrics
        metrics_path = metrics_dir / f'training_metrics_epoch_{epoch}.json'
        with open(metrics_path, 'w') as f:
            json.dump(asdict(self.current_metrics), f, indent=2)

        # Save performance history (recent 1000 snapshots)
        history_path = metrics_dir / f'performance_history_epoch_{epoch}.json'
        recent_history = self.performance_history[-1000:]
        history_data = [asdict(snapshot) for snapshot in recent_history]

        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)

        self.logger.info(f"Performance metrics saved: {metrics_path}")

    def get_nasa_compliance_metrics(self) -> Dict[str, Any]:
        """Generate NASA POT10 compliance metrics."""
        if not self.performance_history:
            return {'compliance_score': 0.0, 'issues': ['No performance data available']}

        recent_data = self.performance_history[-100:]
        issues = []
        compliance_score = 100.0

        # Check performance stability (NASA requirement)
        throughputs = [s.throughput for s in recent_data if s.throughput > 0]
        if len(throughputs) > 1:
            throughput_cv = np.std(throughputs) / np.mean(throughputs)
            if throughput_cv > 0.2:  # More than 20% variance
                issues.append('High throughput variance detected')
                compliance_score -= 15.0

        # Check memory efficiency
        gpu_utilizations = [
            (s.gpu_memory_used / s.gpu_memory_total) * 100
            for s in recent_data if s.gpu_memory_total > 0
        ]

        if gpu_utilizations:
            avg_util = np.mean(gpu_utilizations)
            if avg_util > 95:  # Too high memory usage
                issues.append('Memory usage exceeds safe limits')
                compliance_score -= 20.0
            elif avg_util < 30:  # Too low utilization
                issues.append('Low GPU memory utilization')
                compliance_score -= 10.0

        # Check for error conditions
        if self.current_metrics.bottlenecks_detected:
            unique_bottlenecks = set(self.current_metrics.bottlenecks_detected)
            for bottleneck in unique_bottlenecks:
                issues.append(f'Performance bottleneck: {bottleneck}')
                compliance_score -= 5.0

        # Ensure minimum score
        compliance_score = max(0.0, compliance_score)

        return {
            'compliance_score': compliance_score,
            'issues': issues,
            'performance_stability': 100.0 - (throughput_cv * 100 if throughputs else 0),
            'resource_efficiency': avg_util if gpu_utilizations else 0.0,
            'bottleneck_count': len(set(self.current_metrics.bottlenecks_detected))
        }

    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final training report."""
        total_time = time.time() - self.training_start_time

        # Performance summary
        if self.performance_history:
            all_throughputs = [s.throughput for s in self.performance_history if s.throughput > 0]
            avg_throughput = np.mean(all_throughputs) if all_throughputs else 0.0
            max_throughput = np.max(all_throughputs) if all_throughputs else 0.0

            all_losses = [s.loss for s in self.performance_history]
            final_loss = all_losses[-1] if all_losses else float('inf')
            loss_improvement = ((all_losses[0] - final_loss) / all_losses[0] * 100) if len(all_losses) > 1 else 0.0
        else:
            avg_throughput = max_throughput = 0.0
            final_loss = float('inf')
            loss_improvement = 0.0

        # Bottleneck analysis
        bottleneck_summary = {}
        for bottleneck in set(self.current_metrics.bottlenecks_detected):
            bottleneck_summary[bottleneck] = self.current_metrics.bottlenecks_detected.count(bottleneck)

        # NASA compliance
        nasa_metrics = self.get_nasa_compliance_metrics()

        return {
            'training_summary': {
                'total_time_hours': total_time / 3600,
                'total_epochs': self.current_metrics.total_epochs,
                'total_steps': self.current_metrics.total_steps,
                'final_loss': final_loss,
                'best_loss': self.current_metrics.best_loss,
                'loss_improvement_pct': loss_improvement
            },
            'performance_metrics': {
                'average_throughput': avg_throughput,
                'peak_throughput': max_throughput,
                'average_gpu_utilization': self.current_metrics.gpu_utilization_avg,
                'memory_efficiency': self.current_metrics.memory_efficiency
            },
            'bottleneck_analysis': {
                'bottlenecks_detected': bottleneck_summary,
                'optimization_opportunities': self.bottleneck_detector.get_optimization_recommendations(
                    list(bottleneck_summary.keys())
                )
            },
            'nasa_pot10_compliance': nasa_metrics,
            'recommendations': self._generate_final_recommendations()
        }

    def _generate_final_recommendations(self) -> List[str]:
        """Generate final optimization recommendations."""
        recommendations = []

        # Based on throughput
        if self.current_metrics.throughput_samples_per_sec < 100:
            recommendations.append("Consider increasing batch size or optimizing data pipeline for better throughput")

        # Based on GPU utilization
        if self.current_metrics.gpu_utilization_avg < 60:
            recommendations.append("GPU utilization is low - consider increasing model size or batch size")
        elif self.current_metrics.gpu_utilization_avg > 95:
            recommendations.append("GPU utilization is very high - consider reducing batch size to prevent OOM")

        # Based on bottlenecks
        if 'loss_stagnation' in self.current_metrics.bottlenecks_detected:
            recommendations.append("Loss stagnation detected - consider adjusting learning rate or optimization strategy")

        # Based on convergence
        if len(self.performance_history) > 100:
            recent_losses = [s.loss for s in self.performance_history[-50:]]
            if len(set([round(l, 4) for l in recent_losses])) == 1:
                recommendations.append("Training may have converged - consider stopping or reducing learning rate")

        return recommendations

    def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        self.gpu_monitor.stop_monitoring()
        self.logger.info("Training monitor cleanup completed")


if __name__ == "__main__":
    # Example usage and testing
    def test_performance_monitor():
        """Test performance monitoring system."""
        from training_config import TrainingConfig

        config = TrainingConfig()
        device_ids = list(range(min(2, torch.cuda.device_count())))

        # Create performance monitor
        monitor = TrainingMonitor(config, device_ids)

        # Test epoch monitoring
        monitor.start_epoch(0, 100)

        # Simulate training steps
        for step in range(10):
            monitor.log_step(
                step=step,
                loss=1.0 - (step * 0.05),  # Decreasing loss
                learning_rate=1e-4,
                step_time=0.5
            )

        monitor.end_epoch(0.5)

        # Test final report generation
        final_report = monitor.generate_final_report()
        print(f"✓ Final report generated: {len(final_report)} sections")

        # Test NASA compliance metrics
        nasa_metrics = monitor.get_nasa_compliance_metrics()
        print(f"✓ NASA compliance score: {nasa_metrics['compliance_score']:.1f}")

        # Cleanup
        monitor.cleanup()

        print("Performance monitor test completed successfully")

    # Run test
    test_performance_monitor()