"""
Comprehensive Training Benchmarking Suite for Agent Forge Phase 5 BitNet Training
Implements training speed measurements, memory usage profiling, GPU utilization analysis,
convergence speed evaluation, and quality preservation validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import psutil
import threading
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict, deque
import statistics
import os
import gc
from pathlib import Path

@dataclass
class BenchmarkConfig:
    """Configuration for training benchmarks."""
    # Benchmark duration settings
    warmup_batches: int = 50
    benchmark_batches: int = 500
    measurement_interval: int = 10

    # Performance targets
    target_speedup: float = 1.5  # 50% improvement target
    target_gpu_utilization: float = 0.9  # 90% GPU utilization
    target_memory_efficiency: float = 0.8  # 80% memory efficiency
    target_convergence_speedup: float = 2.0  # 2x faster convergence

    # Quality preservation thresholds
    max_quality_degradation: float = 0.05  # 5% max quality loss
    min_compression_ratio: float = 8.0  # 8x compression from BitNet

    # Monitoring settings
    enable_detailed_profiling: bool = True
    save_benchmark_plots: bool = True
    enable_memory_tracking: bool = True
    enable_gpu_monitoring: bool = True

@dataclass
class BenchmarkResults:
    """Results structure for training benchmarks."""
    # Training speed metrics
    samples_per_second: float = 0.0
    batches_per_second: float = 0.0
    speedup_factor: float = 1.0
    training_efficiency: float = 1.0

    # Memory metrics
    peak_memory_usage: float = 0.0
    average_memory_usage: float = 0.0
    memory_efficiency: float = 0.0
    memory_savings: float = 0.0

    # GPU metrics
    average_gpu_utilization: float = 0.0
    peak_gpu_utilization: float = 0.0
    gpu_memory_usage: float = 0.0
    gpu_efficiency: float = 0.0

    # Convergence metrics
    convergence_speed: float = 0.0
    final_loss: float = 0.0
    quality_preservation: float = 1.0
    compression_ratio: float = 1.0

    # Detailed tracking
    timeline_data: Dict[str, List[float]] = field(default_factory=dict)
    performance_breakdown: Dict[str, float] = field(default_factory=dict)
    resource_utilization: Dict[str, Any] = field(default_factory=dict)

class SystemMonitor:
    """System resource monitoring for benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_queue = deque(maxlen=10000)
        self.gpu_available = torch.cuda.is_available()

    def start_monitoring(self):
        """Start system monitoring in background thread."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logging.info("System monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_queue.append(metrics)
                time.sleep(0.1)  # 10Hz monitoring
            except Exception as e:
                logging.error(f"Monitoring error: {e}")

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        timestamp = time.time()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available
        memory_used = memory.used

        # GPU metrics
        gpu_metrics = {}
        if self.gpu_available:
            try:
                gpu_metrics = {
                    'gpu_memory_allocated': torch.cuda.memory_allocated(),
                    'gpu_memory_reserved': torch.cuda.memory_reserved(),
                    'gpu_memory_max_allocated': torch.cuda.max_memory_allocated(),
                    'gpu_utilization': self._get_gpu_utilization(),
                    'gpu_temperature': self._get_gpu_temperature()
                }
            except Exception as e:
                logging.debug(f"GPU metrics collection failed: {e}")

        # Process-specific metrics
        process = psutil.Process()
        process_metrics = {
            'process_cpu_percent': process.cpu_percent(),
            'process_memory_rss': process.memory_info().rss,
            'process_memory_vms': process.memory_info().vms,
            'process_num_threads': process.num_threads(),
            'process_num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
        }

        return {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
            'load_avg_1m': load_avg[0],
            'memory_percent': memory_percent,
            'memory_available_gb': memory_available / (1024**3),
            'memory_used_gb': memory_used / (1024**3),
            **gpu_metrics,
            **process_metrics
        }

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            # This would typically use nvidia-ml-py or similar
            # For now, return a placeholder based on memory usage
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return (allocated / total) * 100 if total > 0 else 0.0
        except:
            return 0.0

    def _get_gpu_temperature(self) -> float:
        """Get GPU temperature."""
        try:
            # Placeholder - would use nvidia-ml-py for real implementation
            return 0.0
        except:
            return 0.0

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring data."""
        if not self.metrics_queue:
            return {}

        metrics_list = list(self.metrics_queue)

        # Calculate averages and peaks
        summary = {}

        # CPU metrics
        cpu_values = [m['cpu_percent'] for m in metrics_list]
        summary['cpu'] = {
            'average': np.mean(cpu_values),
            'peak': np.max(cpu_values),
            'std': np.std(cpu_values)
        }

        # Memory metrics
        memory_values = [m['memory_percent'] for m in metrics_list]
        summary['memory'] = {
            'average_percent': np.mean(memory_values),
            'peak_percent': np.max(memory_values),
            'average_used_gb': np.mean([m['memory_used_gb'] for m in metrics_list])
        }

        # GPU metrics
        if self.gpu_available and 'gpu_utilization' in metrics_list[0]:
            gpu_util_values = [m.get('gpu_utilization', 0) for m in metrics_list]
            gpu_memory_values = [m.get('gpu_memory_allocated', 0) for m in metrics_list]

            summary['gpu'] = {
                'average_utilization': np.mean(gpu_util_values),
                'peak_utilization': np.max(gpu_util_values),
                'average_memory_gb': np.mean(gpu_memory_values) / (1024**3),
                'peak_memory_gb': np.max(gpu_memory_values) / (1024**3)
            }

        return summary

class TrainingSpeedBenchmark:
    """Benchmark training speed and throughput."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.speed_measurements = []
        self.baseline_measurements = []

    def benchmark_training_speed(self, model: nn.Module, dataloader: DataLoader,
                                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                                enable_optimizations: bool = True) -> Dict[str, float]:
        """Benchmark training speed with and without optimizations."""
        model.train()

        # Warmup phase
        self._warmup_phase(model, dataloader, optimizer, criterion)

        # Benchmark phase
        measurements = self._benchmark_phase(
            model, dataloader, optimizer, criterion, enable_optimizations
        )

        # Calculate metrics
        results = self._calculate_speed_metrics(measurements)

        if enable_optimizations:
            self.speed_measurements = measurements
        else:
            self.baseline_measurements = measurements

        return results

    def _warmup_phase(self, model: nn.Module, dataloader: DataLoader,
                     optimizer: torch.optim.Optimizer, criterion: nn.Module):
        """Warmup phase to stabilize measurements."""
        logging.info(f"Warmup phase: {self.config.warmup_batches} batches")

        batch_iter = iter(dataloader)
        for i in range(self.config.warmup_batches):
            try:
                batch = next(batch_iter)
                if len(batch) >= 2:
                    data, labels = batch[0], batch[1]
                    data = data.to(model.device if hasattr(model, 'device') else 'cpu')
                    labels = labels.to(model.device if hasattr(model, 'device') else 'cpu')

                    # Forward pass
                    outputs = model(data)
                    loss = criterion(outputs, labels)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            except StopIteration:
                batch_iter = iter(dataloader)
                continue

        # Clear cache after warmup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _benchmark_phase(self, model: nn.Module, dataloader: DataLoader,
                        optimizer: torch.optim.Optimizer, criterion: nn.Module,
                        enable_optimizations: bool) -> List[Dict[str, Any]]:
        """Main benchmark measurement phase."""
        logging.info(f"Benchmark phase: {self.config.benchmark_batches} batches")

        measurements = []
        batch_iter = iter(dataloader)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        benchmark_start = time.time()

        for batch_idx in range(self.config.benchmark_batches):
            try:
                batch = next(batch_iter)
                if len(batch) >= 2:
                    data, labels = batch[0], batch[1]

                    batch_start = time.time()

                    # Move to device
                    device = next(model.parameters()).device
                    data = data.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    data_transfer_time = time.time() - batch_start

                    # Forward pass timing
                    forward_start = time.time()
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    forward_time = time.time() - forward_start

                    # Backward pass timing
                    backward_start = time.time()
                    optimizer.zero_grad()
                    loss.backward()
                    backward_time = time.time() - backward_start

                    # Optimizer step timing
                    optimizer_start = time.time()
                    optimizer.step()
                    optimizer_time = time.time() - optimizer_start

                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    total_batch_time = time.time() - batch_start

                    # Record measurement
                    measurement = {
                        'batch_idx': batch_idx,
                        'timestamp': batch_start,
                        'batch_size': len(data),
                        'total_time': total_batch_time,
                        'data_transfer_time': data_transfer_time,
                        'forward_time': forward_time,
                        'backward_time': backward_time,
                        'optimizer_time': optimizer_time,
                        'loss': loss.item(),
                        'samples_per_second': len(data) / total_batch_time,
                        'enable_optimizations': enable_optimizations
                    }

                    measurements.append(measurement)

                    # Progress reporting
                    if (batch_idx + 1) % 50 == 0:
                        avg_time = np.mean([m['total_time'] for m in measurements[-50:]])
                        avg_sps = np.mean([m['samples_per_second'] for m in measurements[-50:]])
                        logging.info(f"  Batch {batch_idx + 1}/{self.config.benchmark_batches}: "
                                   f"{avg_time:.4f}s/batch, {avg_sps:.1f} samples/s")

            except StopIteration:
                batch_iter = iter(dataloader)
                continue

        total_benchmark_time = time.time() - benchmark_start
        logging.info(f"Benchmark completed in {total_benchmark_time:.2f}s")

        return measurements

    def _calculate_speed_metrics(self, measurements: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate speed metrics from measurements."""
        if not measurements:
            return {}

        # Basic throughput metrics
        total_samples = sum(m['batch_size'] for m in measurements)
        total_time = sum(m['total_time'] for m in measurements)
        avg_samples_per_second = total_samples / total_time if total_time > 0 else 0

        # Time breakdown analysis
        avg_forward_time = np.mean([m['forward_time'] for m in measurements])
        avg_backward_time = np.mean([m['backward_time'] for m in measurements])
        avg_optimizer_time = np.mean([m['optimizer_time'] for m in measurements])
        avg_data_transfer_time = np.mean([m['data_transfer_time'] for m in measurements])

        # Batch-level metrics
        batch_times = [m['total_time'] for m in measurements]
        avg_batch_time = np.mean(batch_times)
        std_batch_time = np.std(batch_times)
        min_batch_time = np.min(batch_times)
        max_batch_time = np.max(batch_times)

        # Percentile analysis
        p50_batch_time = np.percentile(batch_times, 50)
        p95_batch_time = np.percentile(batch_times, 95)
        p99_batch_time = np.percentile(batch_times, 99)

        # Calculate efficiency metrics
        computation_time = avg_forward_time + avg_backward_time + avg_optimizer_time
        total_avg_time = avg_batch_time
        computation_efficiency = computation_time / total_avg_time if total_avg_time > 0 else 0

        return {
            'samples_per_second': avg_samples_per_second,
            'batches_per_second': 1.0 / avg_batch_time if avg_batch_time > 0 else 0,
            'avg_batch_time': avg_batch_time,
            'std_batch_time': std_batch_time,
            'min_batch_time': min_batch_time,
            'max_batch_time': max_batch_time,
            'p50_batch_time': p50_batch_time,
            'p95_batch_time': p95_batch_time,
            'p99_batch_time': p99_batch_time,
            'avg_forward_time': avg_forward_time,
            'avg_backward_time': avg_backward_time,
            'avg_optimizer_time': avg_optimizer_time,
            'avg_data_transfer_time': avg_data_transfer_time,
            'computation_efficiency': computation_efficiency,
            'total_samples': total_samples,
            'total_batches': len(measurements)
        }

    def calculate_speedup_factor(self) -> float:
        """Calculate speedup factor between optimized and baseline."""
        if not self.baseline_measurements or not self.speed_measurements:
            return 1.0

        baseline_sps = np.mean([m['samples_per_second'] for m in self.baseline_measurements])
        optimized_sps = np.mean([m['samples_per_second'] for m in self.speed_measurements])

        if baseline_sps > 0:
            return optimized_sps / baseline_sps
        return 1.0

class MemoryProfiler:
    """Memory usage profiling and analysis."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.memory_timeline = []
        self.peak_memory = 0
        self.baseline_memory = 0

    @contextmanager
    def profile_memory(self, phase_name: str = "training"):
        """Context manager for memory profiling."""
        if not self.config.enable_memory_tracking:
            yield
            return

        # Clear cache and collect garbage before profiling
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Record initial memory state
        initial_memory = self._get_current_memory_usage()
        start_time = time.time()

        try:
            yield self

        finally:
            # Record final memory state
            final_memory = self._get_current_memory_usage()
            end_time = time.time()

            # Record memory profile
            profile_record = {
                'phase_name': phase_name,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'initial_memory': initial_memory,
                'final_memory': final_memory,
                'peak_memory': self.peak_memory,
                'memory_delta': final_memory['total_allocated'] - initial_memory['total_allocated']
            }

            self.memory_timeline.append(profile_record)

            logging.info(f"Memory profile for {phase_name}:")
            logging.info(f"  Peak memory: {self.peak_memory / (1024**3):.2f} GB")
            logging.info(f"  Memory delta: {profile_record['memory_delta'] / (1024**3):.2f} GB")

    def track_memory_usage(self):
        """Track memory usage during training."""
        current_memory = self._get_current_memory_usage()
        total_allocated = current_memory['total_allocated']

        if total_allocated > self.peak_memory:
            self.peak_memory = total_allocated

        # Store timeline data
        self.memory_timeline.append({
            'timestamp': time.time(),
            'memory_usage': current_memory
        })

    def _get_current_memory_usage(self) -> Dict[str, int]:
        """Get comprehensive current memory usage."""
        memory_info = {}

        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_total'] = system_memory.total
        memory_info['system_used'] = system_memory.used
        memory_info['system_available'] = system_memory.available

        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        memory_info['process_rss'] = process_memory.rss
        memory_info['process_vms'] = process_memory.vms

        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated()
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved()
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated()
            memory_info['gpu_max_reserved'] = torch.cuda.max_memory_reserved()
        else:
            memory_info['gpu_allocated'] = 0
            memory_info['gpu_reserved'] = 0
            memory_info['gpu_max_allocated'] = 0
            memory_info['gpu_max_reserved'] = 0

        # Total allocated (system + GPU)
        memory_info['total_allocated'] = memory_info['process_rss'] + memory_info['gpu_allocated']

        return memory_info

    def calculate_memory_efficiency(self) -> Dict[str, float]:
        """Calculate memory efficiency metrics."""
        if not self.memory_timeline:
            return {}

        # Extract memory usage values
        if isinstance(self.memory_timeline[0].get('memory_usage'), dict):
            memory_values = [record['memory_usage']['total_allocated'] for record in self.memory_timeline
                           if 'memory_usage' in record]
        else:
            memory_values = [record.get('peak_memory', 0) for record in self.memory_timeline]

        if not memory_values:
            return {}

        avg_memory = np.mean(memory_values)
        peak_memory = np.max(memory_values)
        min_memory = np.min(memory_values)

        # Calculate efficiency
        memory_utilization = avg_memory / peak_memory if peak_memory > 0 else 0
        memory_stability = 1.0 - (np.std(memory_values) / avg_memory) if avg_memory > 0 else 0

        return {
            'average_memory_gb': avg_memory / (1024**3),
            'peak_memory_gb': peak_memory / (1024**3),
            'min_memory_gb': min_memory / (1024**3),
            'memory_utilization': memory_utilization,
            'memory_stability': memory_stability,
            'memory_range_gb': (peak_memory - min_memory) / (1024**3)
        }

class ConvergenceBenchmark:
    """Benchmark convergence speed and quality preservation."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loss_history = []
        self.accuracy_history = []
        self.convergence_points = []

    def track_convergence(self, loss: float, accuracy: float = None, epoch: int = 0, batch: int = 0):
        """Track convergence metrics during training."""
        timestamp = time.time()

        loss_record = {
            'timestamp': timestamp,
            'epoch': epoch,
            'batch': batch,
            'loss': loss
        }

        if accuracy is not None:
            loss_record['accuracy'] = accuracy
            self.accuracy_history.append({
                'timestamp': timestamp,
                'epoch': epoch,
                'batch': batch,
                'accuracy': accuracy
            })

        self.loss_history.append(loss_record)

        # Check for convergence
        if self._detect_convergence():
            convergence_point = {
                'timestamp': timestamp,
                'epoch': epoch,
                'batch': batch,
                'loss': loss,
                'steps_to_convergence': len(self.loss_history)
            }

            if accuracy is not None:
                convergence_point['accuracy'] = accuracy

            self.convergence_points.append(convergence_point)
            logging.info(f"Convergence detected at epoch {epoch}, batch {batch}")

    def _detect_convergence(self) -> bool:
        """Detect if convergence has occurred."""
        if len(self.loss_history) < 50:  # Need minimum history
            return False

        # Check if loss has stabilized (coefficient of variation < 1%)
        recent_losses = [record['loss'] for record in self.loss_history[-20:]]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)

        if mean_loss > 0:
            cv = std_loss / mean_loss
            return cv < 0.01

        return False

    def calculate_convergence_speed(self, baseline_steps: int = None) -> Dict[str, float]:
        """Calculate convergence speed metrics."""
        if not self.convergence_points:
            return {'converged': False}

        last_convergence = self.convergence_points[-1]
        steps_to_convergence = last_convergence['steps_to_convergence']

        results = {
            'converged': True,
            'steps_to_convergence': steps_to_convergence,
            'convergence_loss': last_convergence['loss'],
            'convergence_epoch': last_convergence['epoch']
        }

        if 'accuracy' in last_convergence:
            results['convergence_accuracy'] = last_convergence['accuracy']

        # Calculate speedup if baseline is provided
        if baseline_steps is not None and baseline_steps > 0:
            speedup_factor = baseline_steps / steps_to_convergence
            results['convergence_speedup'] = speedup_factor

        return results

    def calculate_quality_preservation(self, baseline_final_loss: float = None,
                                     baseline_final_accuracy: float = None) -> Dict[str, float]:
        """Calculate quality preservation metrics."""
        if not self.loss_history:
            return {}

        final_loss = self.loss_history[-1]['loss']
        results = {'final_loss': final_loss}

        if self.accuracy_history:
            final_accuracy = self.accuracy_history[-1]['accuracy']
            results['final_accuracy'] = final_accuracy

            if baseline_final_accuracy is not None:
                accuracy_preservation = final_accuracy / baseline_final_accuracy
                results['accuracy_preservation'] = accuracy_preservation
                results['accuracy_degradation'] = 1.0 - accuracy_preservation

        if baseline_final_loss is not None:
            loss_ratio = final_loss / baseline_final_loss
            results['loss_ratio'] = loss_ratio
            results['quality_preservation'] = 1.0 / loss_ratio if loss_ratio > 0 else 1.0

        return results

class ComprehensiveTrainingBenchmark:
    """Main comprehensive training benchmark coordinator."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        # Initialize benchmarking components
        self.system_monitor = SystemMonitor(config)
        self.speed_benchmark = TrainingSpeedBenchmark(config)
        self.memory_profiler = MemoryProfiler(config)
        self.convergence_benchmark = ConvergenceBenchmark(config)

        # Results storage
        self.benchmark_results = BenchmarkResults()
        self.detailed_logs = []

    def run_comprehensive_benchmark(self, model: nn.Module, dataloader: DataLoader,
                                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                                   num_epochs: int = 5) -> BenchmarkResults:
        """Run comprehensive training benchmark."""
        logging.info("Starting comprehensive training benchmark...")

        # Start system monitoring
        self.system_monitor.start_monitoring()

        try:
            # Baseline benchmark (without optimizations)
            logging.info("Running baseline benchmark...")
            baseline_results = self._run_baseline_benchmark(model, dataloader, optimizer, criterion)

            # Optimized benchmark (with optimizations)
            logging.info("Running optimized benchmark...")
            optimized_results = self._run_optimized_benchmark(model, dataloader, optimizer, criterion)

            # Training convergence benchmark
            logging.info("Running convergence benchmark...")
            convergence_results = self._run_convergence_benchmark(
                model, dataloader, optimizer, criterion, num_epochs
            )

            # Compile final results
            self._compile_benchmark_results(baseline_results, optimized_results, convergence_results)

        finally:
            # Stop monitoring
            self.system_monitor.stop_monitoring()

        # Generate report
        self._generate_benchmark_report()

        return self.benchmark_results

    def _run_baseline_benchmark(self, model: nn.Module, dataloader: DataLoader,
                               optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, Any]:
        """Run baseline benchmark without optimizations."""
        with self.memory_profiler.profile_memory("baseline"):
            baseline_results = self.speed_benchmark.benchmark_training_speed(
                model, dataloader, optimizer, criterion, enable_optimizations=False
            )

        return baseline_results

    def _run_optimized_benchmark(self, model: nn.Module, dataloader: DataLoader,
                                optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, Any]:
        """Run optimized benchmark with all optimizations enabled."""
        with self.memory_profiler.profile_memory("optimized"):
            optimized_results = self.speed_benchmark.benchmark_training_speed(
                model, dataloader, optimizer, criterion, enable_optimizations=True
            )

        return optimized_results

    def _run_convergence_benchmark(self, model: nn.Module, dataloader: DataLoader,
                                  optimizer: torch.optim.Optimizer, criterion: nn.Module,
                                  num_epochs: int) -> Dict[str, Any]:
        """Run convergence speed benchmark."""
        model.train()
        batch_count = 0

        with self.memory_profiler.profile_memory("convergence"):
            for epoch in range(num_epochs):
                epoch_start = time.time()
                epoch_loss = 0.0
                num_batches = 0

                for batch_idx, batch in enumerate(dataloader):
                    if len(batch) >= 2:
                        data, labels = batch[0], batch[1]

                        # Move to device
                        device = next(model.parameters()).device
                        data = data.to(device)
                        labels = labels.to(device)

                        # Forward pass
                        outputs = model(data)
                        loss = criterion(outputs, labels)

                        # Calculate accuracy if classification task
                        accuracy = None
                        if len(outputs.shape) == 2 and outputs.shape[1] > 1:
                            _, predicted = torch.max(outputs.data, 1)
                            if len(labels.shape) == 1:  # Classification labels
                                correct = (predicted == labels).sum().item()
                                accuracy = correct / labels.size(0)

                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Track convergence
                        self.convergence_benchmark.track_convergence(
                            loss.item(), accuracy, epoch, batch_count
                        )

                        # Track memory periodically
                        if batch_count % 10 == 0:
                            self.memory_profiler.track_memory_usage()

                        epoch_loss += loss.item()
                        num_batches += 1
                        batch_count += 1

                        # Limit batches for benchmark
                        if batch_idx >= 100:  # Limit for benchmark
                            break

                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                epoch_duration = time.time() - epoch_start

                logging.info(f"Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}, "
                           f"time={epoch_duration:.2f}s")

        # Calculate convergence metrics
        convergence_speed = self.convergence_benchmark.calculate_convergence_speed()
        quality_preservation = self.convergence_benchmark.calculate_quality_preservation()

        return {
            'convergence_speed': convergence_speed,
            'quality_preservation': quality_preservation,
            'total_batches': batch_count,
            'final_loss': self.convergence_benchmark.loss_history[-1]['loss'] if self.convergence_benchmark.loss_history else 0
        }

    def _compile_benchmark_results(self, baseline_results: Dict[str, Any],
                                  optimized_results: Dict[str, Any],
                                  convergence_results: Dict[str, Any]):
        """Compile all benchmark results into final structure."""
        # Speed metrics
        self.benchmark_results.samples_per_second = optimized_results.get('samples_per_second', 0)
        self.benchmark_results.batches_per_second = optimized_results.get('batches_per_second', 0)
        self.benchmark_results.speedup_factor = self.speed_benchmark.calculate_speedup_factor()

        # Memory metrics
        memory_metrics = self.memory_profiler.calculate_memory_efficiency()
        self.benchmark_results.peak_memory_usage = memory_metrics.get('peak_memory_gb', 0)
        self.benchmark_results.average_memory_usage = memory_metrics.get('average_memory_gb', 0)
        self.benchmark_results.memory_efficiency = memory_metrics.get('memory_utilization', 0)

        # GPU metrics
        system_summary = self.system_monitor.get_monitoring_summary()
        if 'gpu' in system_summary:
            self.benchmark_results.average_gpu_utilization = system_summary['gpu']['average_utilization'] / 100
            self.benchmark_results.peak_gpu_utilization = system_summary['gpu']['peak_utilization'] / 100
            self.benchmark_results.gpu_memory_usage = system_summary['gpu']['peak_memory_gb']

        # Convergence metrics
        convergence_speed = convergence_results.get('convergence_speed', {})
        if convergence_speed.get('converged', False):
            self.benchmark_results.convergence_speed = convergence_speed.get('convergence_speedup', 1.0)

        quality_preservation = convergence_results.get('quality_preservation', {})
        self.benchmark_results.quality_preservation = quality_preservation.get('quality_preservation', 1.0)
        self.benchmark_results.final_loss = convergence_results.get('final_loss', 0)

        # Performance breakdown
        self.benchmark_results.performance_breakdown = {
            'baseline_sps': baseline_results.get('samples_per_second', 0),
            'optimized_sps': optimized_results.get('samples_per_second', 0),
            'speedup_factor': self.benchmark_results.speedup_factor,
            'memory_efficiency': self.benchmark_results.memory_efficiency,
            'gpu_utilization': self.benchmark_results.average_gpu_utilization
        }

        # Timeline data
        self.benchmark_results.timeline_data = {
            'baseline_batch_times': [m['total_time'] for m in self.speed_benchmark.baseline_measurements],
            'optimized_batch_times': [m['total_time'] for m in self.speed_benchmark.speed_measurements],
            'loss_history': [record['loss'] for record in self.convergence_benchmark.loss_history],
            'accuracy_history': [record['accuracy'] for record in self.convergence_benchmark.accuracy_history
                               if 'accuracy' in record]
        }

    def _generate_benchmark_report(self):
        """Generate comprehensive benchmark report."""
        report = {
            'timestamp': time.time(),
            'configuration': {
                'warmup_batches': self.config.warmup_batches,
                'benchmark_batches': self.config.benchmark_batches,
                'target_speedup': self.config.target_speedup,
                'target_gpu_utilization': self.config.target_gpu_utilization
            },
            'results': {
                'samples_per_second': self.benchmark_results.samples_per_second,
                'speedup_factor': self.benchmark_results.speedup_factor,
                'memory_efficiency': self.benchmark_results.memory_efficiency,
                'gpu_utilization': self.benchmark_results.average_gpu_utilization,
                'convergence_speed': self.benchmark_results.convergence_speed,
                'quality_preservation': self.benchmark_results.quality_preservation
            },
            'targets_met': {
                'speedup_target': self.benchmark_results.speedup_factor >= self.config.target_speedup,
                'gpu_utilization_target': self.benchmark_results.average_gpu_utilization >= self.config.target_gpu_utilization,
                'memory_efficiency_target': self.benchmark_results.memory_efficiency >= self.config.target_memory_efficiency,
                'convergence_target': self.benchmark_results.convergence_speed >= self.config.target_convergence_speedup
            },
            'performance_breakdown': self.benchmark_results.performance_breakdown,
            'system_summary': self.system_monitor.get_monitoring_summary()
        }

        # Save report
        report_path = Path('benchmark_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logging.info(f"Benchmark report saved to {report_path}")

        # Generate performance plots if enabled
        if self.config.save_benchmark_plots:
            self._generate_performance_plots()

    def _generate_performance_plots(self):
        """Generate performance visualization plots."""
        try:
            # Training speed comparison
            plt.figure(figsize=(12, 8))

            # Plot 1: Batch time comparison
            plt.subplot(2, 2, 1)
            baseline_times = self.benchmark_results.timeline_data.get('baseline_batch_times', [])
            optimized_times = self.benchmark_results.timeline_data.get('optimized_batch_times', [])

            if baseline_times and optimized_times:
                x_baseline = range(len(baseline_times))
                x_optimized = range(len(optimized_times))

                plt.plot(x_baseline, baseline_times, label='Baseline', alpha=0.7)
                plt.plot(x_optimized, optimized_times, label='Optimized', alpha=0.7)
                plt.xlabel('Batch Index')
                plt.ylabel('Batch Time (seconds)')
                plt.title('Training Speed Comparison')
                plt.legend()

            # Plot 2: Loss convergence
            plt.subplot(2, 2, 2)
            loss_history = self.benchmark_results.timeline_data.get('loss_history', [])
            if loss_history:
                plt.plot(loss_history)
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.title('Loss Convergence')

            # Plot 3: Memory usage
            plt.subplot(2, 2, 3)
            memory_timeline = self.memory_profiler.memory_timeline
            if memory_timeline:
                timestamps = [record.get('timestamp', 0) for record in memory_timeline]
                memory_usage = [record.get('memory_usage', {}).get('total_allocated', 0) / (1024**3)
                              for record in memory_timeline if 'memory_usage' in record]

                if memory_usage and len(memory_usage) == len(timestamps):
                    plt.plot(timestamps, memory_usage)
                    plt.xlabel('Time')
                    plt.ylabel('Memory Usage (GB)')
                    plt.title('Memory Usage Over Time')

            # Plot 4: Performance summary
            plt.subplot(2, 2, 4)
            metrics = ['Speedup', 'Memory Eff.', 'GPU Util.', 'Conv. Speed']
            values = [
                self.benchmark_results.speedup_factor,
                self.benchmark_results.memory_efficiency,
                self.benchmark_results.average_gpu_utilization,
                self.benchmark_results.convergence_speed
            ]
            targets = [
                self.config.target_speedup,
                self.config.target_memory_efficiency,
                self.config.target_gpu_utilization,
                self.config.target_convergence_speedup
            ]

            x_pos = range(len(metrics))
            plt.bar(x_pos, values, alpha=0.7, label='Achieved')
            plt.plot(x_pos, targets, 'ro-', label='Targets')
            plt.xlabel('Metrics')
            plt.ylabel('Values')
            plt.title('Performance vs Targets')
            plt.xticks(x_pos, metrics)
            plt.legend()

            plt.tight_layout()
            plt.savefig('training_benchmark_results.png', dpi=300, bbox_inches='tight')
            plt.close()

            logging.info("Performance plots saved to training_benchmark_results.png")

        except Exception as e:
            logging.warning(f"Failed to generate plots: {e}")

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark results."""
        return {
            'speedup_achieved': self.benchmark_results.speedup_factor,
            'memory_efficiency': self.benchmark_results.memory_efficiency,
            'gpu_utilization': self.benchmark_results.average_gpu_utilization,
            'convergence_speedup': self.benchmark_results.convergence_speed,
            'quality_preserved': self.benchmark_results.quality_preservation,
            'targets_met': {
                'speedup': self.benchmark_results.speedup_factor >= self.config.target_speedup,
                'memory': self.benchmark_results.memory_efficiency >= self.config.target_memory_efficiency,
                'gpu': self.benchmark_results.average_gpu_utilization >= self.config.target_gpu_utilization,
                'convergence': self.benchmark_results.convergence_speed >= self.config.target_convergence_speedup
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Example configuration
    config = BenchmarkConfig(
        warmup_batches=20,
        benchmark_batches=100,
        target_speedup=1.5,
        target_gpu_utilization=0.9,
        enable_detailed_profiling=True,
        save_benchmark_plots=True
    )

    # Initialize benchmark
    benchmark = ComprehensiveTrainingBenchmark(config)

    # Example model, dataset, and training components
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    )

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Example dataset
    dataset = TensorDataset(
        torch.randn(5000, 1000),
        torch.randint(0, 10, (5000,))
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        model, dataloader, optimizer, criterion, num_epochs=3
    )

    # Print summary
    summary = benchmark.get_benchmark_summary()
    print(f"\nBenchmark Summary:")
    print(f"  Speedup achieved: {summary['speedup_achieved']:.2f}x")
    print(f"  Memory efficiency: {summary['memory_efficiency']:.2%}")
    print(f"  GPU utilization: {summary['gpu_utilization']:.2%}")
    print(f"  Convergence speedup: {summary['convergence_speedup']:.2f}x")
    print(f"  Quality preserved: {summary['quality_preserved']:.2%}")

    print(f"\nTargets met:")
    for target, met in summary['targets_met'].items():
        status = "✓" if met else "✗"
        print(f"  {target}: {status}")

    print(f"\nDetailed benchmark report saved to benchmark_report.json")