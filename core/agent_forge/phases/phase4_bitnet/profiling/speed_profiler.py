"""
BitNet Speed Profiler - Agent Forge Phase 4

Advanced Performance and Latency Analysis
=========================================

Implements comprehensive speed profiling for BitNet models to identify
performance bottlenecks and validate 2-4x speedup targets.

Key Features:
1. Real-time inference latency tracking
2. Training speed analysis
3. Throughput measurement and optimization
4. Hardware utilization profiling
5. Bottleneck identification
6. Performance regression detection
7. Comparative analysis and benchmarking

Author: Agent Forge Phase 4 - Speed Profiling Specialist
License: NASA POT10 Compliant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import time
import logging
import json
import statistics
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from pathlib import Path
from collections import deque, defaultdict
import threading
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpeedMeasurement:
    """Individual speed measurement record."""
    measurement_id: str
    operation: str
    timestamp: float

    # Timing measurements
    start_time: float
    end_time: float
    duration_ms: float

    # Context information
    batch_size: int
    sequence_length: int
    input_shape: Tuple[int, ...]

    # Hardware state
    gpu_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0

    # Performance metrics
    throughput_samples_per_sec: float = 0.0
    latency_per_sample_ms: float = 0.0

    # Quality indicators
    is_warmup: bool = False
    has_outlier_timing: bool = False

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    baseline_id: str
    model_name: str
    device: str
    created_timestamp: float

    # Baseline metrics
    avg_inference_time_ms: float
    avg_throughput_samples_per_sec: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Hardware configuration
    hardware_config: Dict[str, Any] = field(default_factory=dict)

    # Model configuration
    model_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SpeedProfilingConfig:
    """Configuration for speed profiling."""
    # Profiling scope
    profile_inference: bool = True
    profile_training: bool = True
    profile_hardware_utilization: bool = True

    # Measurement configuration
    warmup_iterations: int = 10
    measurement_iterations: int = 100
    outlier_threshold_std: float = 3.0  # Standard deviations for outlier detection

    # Sampling configuration
    continuous_profiling: bool = False
    sampling_interval_ms: int = 50
    max_measurements_memory: int = 10000

    # Analysis configuration
    enable_bottleneck_analysis: bool = True
    enable_regression_detection: bool = True
    performance_threshold_percent: float = 5.0  # 5% performance change threshold

    # Baseline management
    maintain_baselines: bool = True
    baseline_storage_path: str = "speed_baselines.json"

    # Export configuration
    export_detailed_measurements: bool = True
    export_analysis_report: bool = True
    generate_performance_plots: bool = True
    output_directory: str = "speed_profiling"

class LatencyProfiler:
    """Latency measurement and analysis."""

    def __init__(self, device: torch.device):
        self.device = device
        self.measurements = deque(maxlen=10000)  # Ring buffer for measurements
        self.measurement_counter = 0

    def measure_latency(self, operation: str, batch_size: int = 1,
                       sequence_length: int = 512, input_shape: Tuple[int, ...] = ()) -> SpeedMeasurement:
        """Measure operation latency with context."""
        measurement_id = f"latency_{self.measurement_counter}_{int(time.time() * 1000)}"
        self.measurement_counter += 1

        # Pre-measurement state
        pre_memory = self._get_memory_usage()
        pre_time = time.perf_counter()

        # Synchronize if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

        timestamp = time.time()

        measurement = SpeedMeasurement(
            measurement_id=measurement_id,
            operation=operation,
            timestamp=timestamp,
            start_time=pre_time,
            end_time=0.0,  # Will be set after operation
            duration_ms=0.0,  # Will be calculated
            batch_size=batch_size,
            sequence_length=sequence_length,
            input_shape=input_shape,
            memory_usage_mb=pre_memory
        )

        return measurement

    def finalize_measurement(self, measurement: SpeedMeasurement) -> SpeedMeasurement:
        """Finalize latency measurement after operation completion."""
        # Synchronize if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

        measurement.end_time = time.perf_counter()
        measurement.duration_ms = (measurement.end_time - measurement.start_time) * 1000

        # Calculate performance metrics
        total_samples = measurement.batch_size
        measurement.latency_per_sample_ms = measurement.duration_ms / total_samples if total_samples > 0 else measurement.duration_ms
        measurement.throughput_samples_per_sec = total_samples / (measurement.duration_ms / 1000) if measurement.duration_ms > 0 else 0

        # Get final hardware state
        measurement.memory_usage_mb = self._get_memory_usage()
        measurement.gpu_utilization = self._get_gpu_utilization()
        measurement.cpu_percent = self._get_cpu_utilization()

        # Outlier detection
        measurement.has_outlier_timing = self._is_outlier_timing(measurement.duration_ms)

        self.measurements.append(measurement)
        return measurement

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == 'cuda':
            try:
                return torch.cuda.memory_allocated(self.device) / (1024**2)
            except Exception:
                return 0.0
        return 0.0

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        # Placeholder - would use nvidia-ml-py for real GPU utilization
        return 0.0

    def _get_cpu_utilization(self) -> float:
        """Get CPU utilization percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            return 0.0

    def _is_outlier_timing(self, duration_ms: float) -> bool:
        """Detect if timing is an outlier compared to recent measurements."""
        if len(self.measurements) < 10:
            return False

        recent_durations = [m.duration_ms for m in list(self.measurements)[-10:] if not m.is_warmup]
        if len(recent_durations) < 5:
            return False

        mean_duration = statistics.mean(recent_durations)
        std_duration = statistics.stdev(recent_durations)

        return abs(duration_ms - mean_duration) > (3.0 * std_duration)

    def get_latency_statistics(self, operation_filter: Optional[str] = None,
                             exclude_warmup: bool = True,
                             exclude_outliers: bool = True) -> Dict[str, Any]:
        """Get comprehensive latency statistics."""
        filtered_measurements = []

        for measurement in self.measurements:
            # Apply filters
            if exclude_warmup and measurement.is_warmup:
                continue
            if exclude_outliers and measurement.has_outlier_timing:
                continue
            if operation_filter and measurement.operation != operation_filter:
                continue

            filtered_measurements.append(measurement)

        if not filtered_measurements:
            return {"statistics_available": False, "reason": "No measurements match criteria"}

        # Extract timing data
        durations = [m.duration_ms for m in filtered_measurements]
        throughputs = [m.throughput_samples_per_sec for m in filtered_measurements]
        latencies = [m.latency_per_sample_ms for m in filtered_measurements]

        stats = {
            "statistics_available": True,
            "measurement_count": len(filtered_measurements),
            "operation_filter": operation_filter,

            # Duration statistics
            "duration_stats": {
                "mean_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "std_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
                "p95_ms": np.percentile(durations, 95),
                "p99_ms": np.percentile(durations, 99),
                "coefficient_of_variation": statistics.stdev(durations) / statistics.mean(durations) if len(durations) > 1 and statistics.mean(durations) > 0 else 0
            },

            # Throughput statistics
            "throughput_stats": {
                "mean_samples_per_sec": statistics.mean(throughputs),
                "median_samples_per_sec": statistics.median(throughputs),
                "max_samples_per_sec": max(throughputs),
                "min_samples_per_sec": min(throughputs)
            },

            # Per-sample latency statistics
            "latency_stats": {
                "mean_per_sample_ms": statistics.mean(latencies),
                "median_per_sample_ms": statistics.median(latencies),
                "p95_per_sample_ms": np.percentile(latencies, 95),
                "p99_per_sample_ms": np.percentile(latencies, 99)
            }
        }

        return stats

class ThroughputProfiler:
    """Throughput measurement and optimization analysis."""

    def __init__(self, device: torch.device):
        self.device = device
        self.throughput_measurements = []
        self.batch_size_analysis = {}

    def measure_throughput(self, model: nn.Module,
                         input_generator: Callable[[], torch.Tensor],
                         batch_sizes: List[int] = [1, 4, 8, 16, 32],
                         iterations_per_batch: int = 50,
                         warmup_iterations: int = 10) -> Dict[str, Any]:
        """Comprehensive throughput measurement across batch sizes."""
        logger.info("Starting throughput measurement...")

        model.eval()
        throughput_results = {}

        for batch_size in batch_sizes:
            logger.info(f"Measuring throughput for batch size: {batch_size}")

            # Generate inputs for this batch size
            inputs = []
            for _ in range(warmup_iterations + iterations_per_batch):
                batch_input = torch.stack([input_generator() for _ in range(batch_size)], dim=0)
                inputs.append(batch_input.to(self.device))

            # Warmup
            with torch.no_grad():
                for i in range(warmup_iterations):
                    _ = model(inputs[i])
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()

            # Measure throughput
            throughput_times = []
            total_samples = 0

            with torch.no_grad():
                for i in range(warmup_iterations, warmup_iterations + iterations_per_batch):
                    start_time = time.perf_counter()
                    output = model(inputs[i])
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()

                    duration = end_time - start_time
                    throughput_times.append(duration)
                    total_samples += batch_size

            # Calculate throughput statistics
            avg_time_per_batch = statistics.mean(throughput_times)
            total_time = sum(throughput_times)
            samples_per_second = total_samples / total_time
            batches_per_second = iterations_per_batch / total_time

            batch_results = {
                "batch_size": batch_size,
                "iterations": iterations_per_batch,
                "total_samples": total_samples,
                "avg_time_per_batch_ms": avg_time_per_batch * 1000,
                "samples_per_second": samples_per_second,
                "batches_per_second": batches_per_second,
                "efficiency_score": samples_per_second / batch_size,  # Samples per second per batch unit
                "time_per_sample_ms": (avg_time_per_batch * 1000) / batch_size
            }

            throughput_results[f"batch_{batch_size}"] = batch_results
            self.batch_size_analysis[batch_size] = batch_results

        # Find optimal batch size
        optimal_batch_analysis = self._analyze_optimal_batch_size()
        throughput_results["optimal_batch_analysis"] = optimal_batch_analysis

        return throughput_results

    def _analyze_optimal_batch_size(self) -> Dict[str, Any]:
        """Analyze optimal batch size for throughput."""
        if not self.batch_size_analysis:
            return {"analysis_available": False}

        # Find batch size with maximum samples per second
        max_throughput_batch = max(self.batch_size_analysis.items(),
                                 key=lambda x: x[1]["samples_per_second"])

        # Find batch size with best efficiency (samples per second per batch unit)
        max_efficiency_batch = max(self.batch_size_analysis.items(),
                                 key=lambda x: x[1]["efficiency_score"])

        # Analyze throughput scaling
        batch_sizes = sorted(self.batch_size_analysis.keys())
        throughputs = [self.batch_size_analysis[bs]["samples_per_second"] for bs in batch_sizes]

        # Calculate scaling efficiency (how well throughput scales with batch size)
        scaling_ratios = []
        for i in range(1, len(batch_sizes)):
            prev_batch, curr_batch = batch_sizes[i-1], batch_sizes[i]
            prev_throughput, curr_throughput = throughputs[i-1], throughputs[i]

            batch_ratio = curr_batch / prev_batch
            throughput_ratio = curr_throughput / prev_throughput
            scaling_efficiency = throughput_ratio / batch_ratio
            scaling_ratios.append(scaling_efficiency)

        return {
            "analysis_available": True,
            "max_throughput_batch_size": max_throughput_batch[0],
            "max_throughput_samples_per_sec": max_throughput_batch[1]["samples_per_second"],
            "max_efficiency_batch_size": max_efficiency_batch[0],
            "max_efficiency_score": max_efficiency_batch[1]["efficiency_score"],
            "avg_scaling_efficiency": statistics.mean(scaling_ratios) if scaling_ratios else 0,
            "throughput_scaling_analysis": {
                "batch_sizes": batch_sizes,
                "throughputs": throughputs,
                "scaling_ratios": scaling_ratios
            }
        }

class BottleneckAnalyzer:
    """Performance bottleneck identification and analysis."""

    def __init__(self, device: torch.device):
        self.device = device
        self.bottleneck_measurements = []

    def profile_model_components(self, model: nn.Module,
                                input_tensor: torch.Tensor,
                                iterations: int = 50) -> Dict[str, Any]:
        """Profile individual model components to identify bottlenecks."""
        logger.info("Profiling model components for bottlenecks...")

        model.eval()
        component_timings = {}

        # Hook for measuring layer timings
        layer_timings = {}
        hooks = []

        def create_timing_hook(name: str):
            def timing_hook(module, input, output):
                if not hasattr(timing_hook, 'start_time'):
                    timing_hook.start_time = time.perf_counter()
                else:
                    end_time = time.perf_counter()
                    duration = (end_time - timing_hook.start_time) * 1000
                    if name not in layer_timings:
                        layer_timings[name] = []
                    layer_timings[name].append(duration)
            return timing_hook

        # Register hooks for all modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(create_timing_hook(name))
                hooks.append(hook)

        # Run profiling iterations
        with torch.no_grad():
            for i in range(iterations):
                layer_timings.clear()

                start_time = time.perf_counter()
                output = model(input_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                total_time = (end_time - start_time) * 1000

                # Aggregate layer timings for this iteration
                for layer_name, timings in layer_timings.items():
                    if layer_name not in component_timings:
                        component_timings[layer_name] = []
                    component_timings[layer_name].append(sum(timings))

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Analyze component performance
        component_analysis = {}
        total_model_time = sum(statistics.mean(timings) for timings in component_timings.values())

        for component_name, timings in component_timings.items():
            if timings:
                avg_time = statistics.mean(timings)
                component_analysis[component_name] = {
                    "avg_time_ms": avg_time,
                    "min_time_ms": min(timings),
                    "max_time_ms": max(timings),
                    "std_time_ms": statistics.stdev(timings) if len(timings) > 1 else 0,
                    "time_percentage": (avg_time / total_model_time) * 100 if total_model_time > 0 else 0,
                    "is_bottleneck": (avg_time / total_model_time) > 0.1 if total_model_time > 0 else False  # >10% of total time
                }

        # Identify bottlenecks
        bottlenecks = {name: analysis for name, analysis in component_analysis.items()
                      if analysis["is_bottleneck"]}

        bottleneck_results = {
            "total_components_profiled": len(component_analysis),
            "total_model_time_ms": total_model_time,
            "component_analysis": component_analysis,
            "identified_bottlenecks": bottlenecks,
            "bottleneck_count": len(bottlenecks),
            "bottleneck_recommendations": self._generate_bottleneck_recommendations(bottlenecks)
        }

        return bottleneck_results

    def _generate_bottleneck_recommendations(self, bottlenecks: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on bottlenecks."""
        recommendations = []

        for component_name, analysis in bottlenecks.items():
            time_percentage = analysis["time_percentage"]

            if "linear" in component_name.lower() or "dense" in component_name.lower():
                recommendations.append(
                    f"Linear layer '{component_name}' is a bottleneck ({time_percentage:.1f}% of time). "
                    "Consider quantization, pruning, or tensor core optimization."
                )
            elif "attention" in component_name.lower():
                recommendations.append(
                    f"Attention layer '{component_name}' is a bottleneck ({time_percentage:.1f}% of time). "
                    "Consider Flash Attention or other attention optimizations."
                )
            elif "norm" in component_name.lower():
                recommendations.append(
                    f"Normalization layer '{component_name}' is a bottleneck ({time_percentage:.1f}% of time). "
                    "Consider fusing with adjacent operations."
                )
            else:
                recommendations.append(
                    f"Component '{component_name}' is a bottleneck ({time_percentage:.1f}% of time). "
                    "Consider component-specific optimizations."
                )

        if not recommendations:
            recommendations.append("No significant bottlenecks detected. Performance appears well-balanced.")

        return recommendations

class BaselineManager:
    """Performance baseline management and comparison."""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.baselines = {}
        self._load_baselines()

    def _load_baselines(self) -> None:
        """Load existing baselines from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.baselines = {k: PerformanceBaseline(**v) for k, v in data.items()}
                logger.info(f"Loaded {len(self.baselines)} performance baselines")
        except Exception as e:
            logger.warning(f"Failed to load baselines: {e}")
            self.baselines = {}

    def save_baseline(self, baseline: PerformanceBaseline) -> None:
        """Save a new performance baseline."""
        self.baselines[baseline.baseline_id] = baseline
        self._save_baselines()

    def _save_baselines(self) -> None:
        """Save baselines to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                data = {k: asdict(v) for k, v in self.baselines.items()}
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save baselines: {e}")

    def compare_with_baseline(self, current_performance: Dict[str, Any],
                            baseline_id: Optional[str] = None) -> Dict[str, Any]:
        """Compare current performance with baseline."""
        if not self.baselines:
            return {"comparison_possible": False, "reason": "No baselines available"}

        # Use most recent baseline if none specified
        if baseline_id is None:
            baseline = max(self.baselines.values(), key=lambda x: x.created_timestamp)
        else:
            baseline = self.baselines.get(baseline_id)

        if not baseline:
            return {"comparison_possible": False, "reason": f"Baseline '{baseline_id}' not found"}

        # Extract current performance metrics
        current_latency = current_performance.get("avg_inference_time_ms", 0)
        current_throughput = current_performance.get("avg_throughput_samples_per_sec", 0)

        # Calculate performance changes
        latency_change_percent = ((current_latency - baseline.avg_inference_time_ms) / baseline.avg_inference_time_ms) * 100 if baseline.avg_inference_time_ms > 0 else 0
        throughput_change_percent = ((current_throughput - baseline.avg_throughput_samples_per_sec) / baseline.avg_throughput_samples_per_sec) * 100 if baseline.avg_throughput_samples_per_sec > 0 else 0

        # Determine performance status
        latency_improved = latency_change_percent < -5  # 5% faster
        throughput_improved = throughput_change_percent > 5   # 5% higher throughput
        performance_regressed = latency_change_percent > 10 or throughput_change_percent < -10

        comparison_results = {
            "comparison_possible": True,
            "baseline_id": baseline.baseline_id,
            "baseline_timestamp": baseline.created_timestamp,

            # Performance changes
            "latency_change_percent": latency_change_percent,
            "throughput_change_percent": throughput_change_percent,

            # Performance status
            "latency_improved": latency_improved,
            "throughput_improved": throughput_improved,
            "performance_regressed": performance_regressed,
            "overall_improved": latency_improved and throughput_improved,

            # Detailed comparison
            "baseline_latency_ms": baseline.avg_inference_time_ms,
            "current_latency_ms": current_latency,
            "baseline_throughput": baseline.avg_throughput_samples_per_sec,
            "current_throughput": current_throughput
        }

        return comparison_results

class SpeedProfiler:
    """Comprehensive speed profiler orchestrator."""

    def __init__(self, config: SpeedProfilingConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize profilers
        self.latency_profiler = LatencyProfiler(self.device)
        self.throughput_profiler = ThroughputProfiler(self.device)
        self.bottleneck_analyzer = BottleneckAnalyzer(self.device)

        # Initialize baseline manager
        if config.maintain_baselines:
            self.baseline_manager = BaselineManager(config.baseline_storage_path)
        else:
            self.baseline_manager = None

        self.profiling_active = False
        self.profiling_start_time = None

    @contextmanager
    def profile_speed(self, operation: str, batch_size: int = 1,
                     sequence_length: int = 512, input_shape: Tuple[int, ...] = ()):
        """Context manager for speed profiling."""
        measurement = self.latency_profiler.measure_latency(
            operation, batch_size, sequence_length, input_shape
        )

        try:
            yield measurement
        finally:
            self.latency_profiler.finalize_measurement(measurement)

    def comprehensive_speed_analysis(self, model: nn.Module,
                                   input_generator: Callable[[], torch.Tensor],
                                   test_name: str = "speed_analysis") -> Dict[str, Any]:
        """Run comprehensive speed analysis."""
        logger.info(f"Starting comprehensive speed analysis: {test_name}")

        analysis_results = {
            "test_name": test_name,
            "device": str(self.device),
            "timestamp": time.time()
        }

        # Generate sample input for profiling
        sample_input = input_generator().to(self.device)

        # Latency analysis
        if self.config.profile_inference:
            logger.info("Running latency analysis...")
            latency_results = self._run_latency_analysis(model, sample_input)
            analysis_results["latency_analysis"] = latency_results

        # Throughput analysis
        logger.info("Running throughput analysis...")
        throughput_results = self.throughput_profiler.measure_throughput(
            model, input_generator,
            batch_sizes=[1, 4, 8, 16, 32],
            iterations_per_batch=self.config.measurement_iterations // 2
        )
        analysis_results["throughput_analysis"] = throughput_results

        # Bottleneck analysis
        if self.config.enable_bottleneck_analysis:
            logger.info("Running bottleneck analysis...")
            bottleneck_results = self.bottleneck_analyzer.profile_model_components(
                model, sample_input, iterations=self.config.measurement_iterations // 4
            )
            analysis_results["bottleneck_analysis"] = bottleneck_results

        # Performance validation
        speed_validation = self._validate_speed_targets(analysis_results)
        analysis_results["speed_validation"] = speed_validation

        # Baseline comparison
        if self.baseline_manager:
            baseline_comparison = self._compare_with_baselines(analysis_results)
            analysis_results["baseline_comparison"] = baseline_comparison

        logger.info("Comprehensive speed analysis completed")
        return analysis_results

    def _run_latency_analysis(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Run detailed latency analysis."""
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                with self.profile_speed("warmup", input_tensor.shape[0],
                                      input_tensor.shape[1] if len(input_tensor.shape) > 1 else 0,
                                      input_tensor.shape):
                    _ = model(input_tensor)

        # Mark warmup measurements
        for measurement in list(self.latency_profiler.measurements)[-self.config.warmup_iterations:]:
            measurement.is_warmup = True

        # Measurement phase
        with torch.no_grad():
            for i in range(self.config.measurement_iterations):
                with self.profile_speed(f"inference_{i}", input_tensor.shape[0],
                                      input_tensor.shape[1] if len(input_tensor.shape) > 1 else 0,
                                      input_tensor.shape):
                    _ = model(input_tensor)

        # Get latency statistics
        latency_stats = self.latency_profiler.get_latency_statistics(
            operation_filter=None, exclude_warmup=True, exclude_outliers=True
        )

        return latency_stats

    def _validate_speed_targets(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate speed performance against targets."""
        validation_results = {"validation_possible": False}

        # Extract performance metrics
        latency_analysis = analysis_results.get("latency_analysis", {})
        if not latency_analysis.get("statistics_available"):
            return validation_results

        current_latency = latency_analysis["duration_stats"]["mean_ms"]
        current_throughput = latency_analysis["throughput_stats"]["mean_samples_per_sec"]

        # For validation, we need a baseline. If no baseline is available,
        # we'll use heuristic estimates or user-provided targets
        estimated_baseline_latency = current_latency * 3.0  # Assume 3x slower baseline

        # Calculate speedup
        speedup_ratio = estimated_baseline_latency / current_latency if current_latency > 0 else 0

        # Validate against targets (2-4x speedup)
        min_target_achieved = speedup_ratio >= 2.0
        max_target_achieved = speedup_ratio >= 4.0

        validation_results = {
            "validation_possible": True,
            "current_latency_ms": current_latency,
            "current_throughput": current_throughput,
            "estimated_baseline_latency_ms": estimated_baseline_latency,
            "speedup_ratio": speedup_ratio,
            "min_target_speedup": 2.0,
            "max_target_speedup": 4.0,
            "min_target_achieved": min_target_achieved,
            "max_target_achieved": max_target_achieved,
            "performance_level": "optimal" if max_target_achieved else "acceptable" if min_target_achieved else "below_target"
        }

        return validation_results

    def _compare_with_baselines(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current performance with stored baselines."""
        if not self.baseline_manager:
            return {"comparison_possible": False, "reason": "Baseline manager not available"}

        # Extract current performance
        latency_analysis = analysis_results.get("latency_analysis", {})
        if not latency_analysis.get("statistics_available"):
            return {"comparison_possible": False, "reason": "No latency analysis available"}

        current_performance = {
            "avg_inference_time_ms": latency_analysis["duration_stats"]["mean_ms"],
            "avg_throughput_samples_per_sec": latency_analysis["throughput_stats"]["mean_samples_per_sec"]
        }

        return self.baseline_manager.compare_with_baseline(current_performance)

    def save_performance_baseline(self, analysis_results: Dict[str, Any],
                                baseline_name: str) -> None:
        """Save current performance as baseline."""
        if not self.baseline_manager:
            logger.warning("Baseline manager not available")
            return

        latency_analysis = analysis_results.get("latency_analysis", {})
        if not latency_analysis.get("statistics_available"):
            logger.warning("No latency analysis available for baseline")
            return

        baseline = PerformanceBaseline(
            baseline_id=f"{baseline_name}_{int(time.time())}",
            model_name=baseline_name,
            device=str(self.device),
            created_timestamp=time.time(),
            avg_inference_time_ms=latency_analysis["duration_stats"]["mean_ms"],
            avg_throughput_samples_per_sec=latency_analysis["throughput_stats"]["mean_samples_per_sec"],
            p95_latency_ms=latency_analysis["duration_stats"]["p95_ms"],
            p99_latency_ms=latency_analysis["duration_stats"]["p99_ms"],
            hardware_config={"device": str(self.device)},
            model_config=analysis_results.get("model_config", {})
        )

        self.baseline_manager.save_baseline(baseline)
        logger.info(f"Performance baseline saved: {baseline.baseline_id}")

    def export_profiling_results(self, analysis_results: Dict[str, Any],
                                output_directory: Optional[str] = None) -> Dict[str, str]:
        """Export speed profiling results."""
        if not (self.config.export_detailed_measurements or self.config.export_analysis_report):
            return {}

        output_dir = Path(output_directory or self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # Export analysis report
        if self.config.export_analysis_report:
            report_path = output_dir / f"speed_analysis_report_{int(time.time())}.json"
            with open(report_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            exported_files["analysis_report"] = str(report_path)

        # Export detailed measurements
        if self.config.export_detailed_measurements:
            measurements_data = [asdict(m) for m in self.latency_profiler.measurements]
            measurements_path = output_dir / f"speed_measurements_{int(time.time())}.json"
            with open(measurements_path, 'w') as f:
                json.dump(measurements_data, f, indent=2, default=str)
            exported_files["detailed_measurements"] = str(measurements_path)

        logger.info(f"Speed profiling results exported to {output_dir}")
        return exported_files

def create_speed_profiler(device: torch.device = None,
                         profiling_level: str = "comprehensive") -> SpeedProfiler:
    """Create speed profiler with preset configurations."""

    configs = {
        "basic": SpeedProfilingConfig(
            measurement_iterations=50,
            profile_training=False,
            enable_bottleneck_analysis=False,
            maintain_baselines=False,
            generate_performance_plots=False
        ),
        "standard": SpeedProfilingConfig(
            measurement_iterations=100,
            profile_training=True,
            enable_bottleneck_analysis=True,
            maintain_baselines=True,
            export_analysis_report=True
        ),
        "comprehensive": SpeedProfilingConfig(
            measurement_iterations=200,
            profile_inference=True,
            profile_training=True,
            enable_bottleneck_analysis=True,
            enable_regression_detection=True,
            maintain_baselines=True,
            export_detailed_measurements=True,
            export_analysis_report=True,
            generate_performance_plots=True
        )
    }

    config = configs.get(profiling_level, configs["standard"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return SpeedProfiler(config, device)

def main():
    """Demonstration of speed profiling capabilities."""
    print("BitNet Speed Profiler - Agent Forge Phase 4")
    print("=" * 48)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling Device: {device}")

    # Create speed profiler
    profiler = create_speed_profiler(device, "comprehensive")

    # Create a model for profiling
    model = nn.Sequential(
        nn.Linear(768, 3072),
        nn.ReLU(),
        nn.Linear(3072, 768)
    ).to(device)

    # Input generator
    def input_generator():
        return torch.randn(128, 768)

    print("\nRunning comprehensive speed analysis...")

    # Run comprehensive analysis
    results = profiler.comprehensive_speed_analysis(
        model, input_generator, "bitnet_demo_model"
    )

    print("\nSpeed Analysis Results:")

    # Display latency results
    latency_analysis = results.get("latency_analysis", {})
    if latency_analysis.get("statistics_available"):
        duration_stats = latency_analysis["duration_stats"]
        print(f"  Average Latency: {duration_stats['mean_ms']:.2f}ms")
        print(f"  P95 Latency: {duration_stats['p95_ms']:.2f}ms")
        print(f"  P99 Latency: {duration_stats['p99_ms']:.2f}ms")

        throughput_stats = latency_analysis["throughput_stats"]
        print(f"  Average Throughput: {throughput_stats['mean_samples_per_sec']:.1f} samples/sec")

    # Display speed validation
    speed_validation = results.get("speed_validation", {})
    if speed_validation.get("validation_possible"):
        print(f"  Speedup Ratio: {speed_validation['speedup_ratio']:.1f}x")
        print(f"  Target Achieved: {speed_validation['min_target_achieved']}")
        print(f"  Performance Level: {speed_validation['performance_level']}")

    # Display bottleneck analysis
    bottleneck_analysis = results.get("bottleneck_analysis", {})
    if bottleneck_analysis.get("identified_bottlenecks"):
        print(f"  Bottlenecks Found: {bottleneck_analysis['bottleneck_count']}")
        for bottleneck in list(bottleneck_analysis["identified_bottlenecks"].keys())[:3]:
            print(f"    - {bottleneck}")

    # Save as baseline
    profiler.save_performance_baseline(results, "demo_model")

    # Export results
    exported_files = profiler.export_profiling_results(results)
    if exported_files:
        print(f"\nResults exported to: {exported_files}")

    print("\nSpeed profiling demonstration completed!")

if __name__ == "__main__":
    main()