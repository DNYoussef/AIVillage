#!/usr/bin/env python3
"""
Agent Forge Phase 6: Performance Profiler
=========================================

Comprehensive performance profiling system that measures, analyzes, and reports
model performance metrics including latency, throughput, memory usage, and
hardware utilization with detailed bottleneck analysis.
"""

import torch
import torch.nn as nn
import torch.profiler as profiler
import numpy as np
import logging
import time
import json
import gc
import threading
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import warnings

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Latency metrics (milliseconds)
    latency_mean: float = 0.0
    latency_median: float = 0.0
    latency_std: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # Throughput metrics
    throughput_samples_per_sec: float = 0.0
    throughput_batches_per_sec: float = 0.0

    # Memory metrics (MB)
    memory_allocated_mb: float = 0.0
    memory_cached_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    memory_peak_mb: float = 0.0

    # Compute metrics
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    flops_per_second: float = 0.0

    # Model metrics
    model_size_mb: float = 0.0
    parameter_count: int = 0
    activation_memory_mb: float = 0.0

    # Efficiency metrics
    memory_bandwidth_utilization: float = 0.0
    compute_efficiency: float = 0.0
    energy_efficiency: float = 0.0

@dataclass
class ProfilingConfig:
    """Configuration for performance profiling"""
    # Benchmarking settings
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    enable_detailed_profiling: bool = True

    # Memory profiling
    enable_memory_profiling: bool = True
    track_memory_timeline: bool = False

    # Hardware profiling
    enable_gpu_profiling: bool = True
    enable_cpu_profiling: bool = True
    profile_kernels: bool = False

    # Analysis settings
    enable_bottleneck_analysis: bool = True
    enable_operator_analysis: bool = True
    profile_batch_sizes: List[int] = None

    # Output settings
    generate_trace: bool = False
    trace_format: str = "json"  # json, chrome

    def __post_init__(self):
        if self.profile_batch_sizes is None:
            self.profile_batch_sizes = [1, 4, 8, 16, 32]

@dataclass
class BottleneckAnalysis:
    """Bottleneck analysis results"""
    bottleneck_type: str = "unknown"  # memory, compute, bandwidth
    bottleneck_severity: float = 0.0  # 0-1 scale
    bottleneck_location: str = ""
    optimization_suggestions: List[str] = None
    performance_impact: float = 0.0

    def __post_init__(self):
        if self.optimization_suggestions is None:
            self.optimization_suggestions = []

class PerformanceProfiler:
    """
    Comprehensive performance profiling system that provides detailed
    analysis of model performance, identifies bottlenecks, and generates
    optimization recommendations.
    """

    def __init__(self, config, device: torch.device, logger: logging.Logger):
        self.config = config
        self.device = device
        self.logger = logger
        self.profiling_config = ProfilingConfig()

        # Profiling state
        self.profiling_history: List[Dict[str, Any]] = []
        self.performance_baselines: Dict[str, PerformanceMetrics] = {}

        # Hardware monitoring
        self.enable_gpu_monitoring = (
            self.device.type == "cuda" and torch.cuda.is_available()
        )

        self.logger.info(f"PerformanceProfiler initialized for {self.device}")

    def profile_model(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        model_name: str = "model",
        profiling_config: Optional[ProfilingConfig] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model performance profiling.

        Args:
            model: Model to profile
            sample_inputs: Sample inputs for profiling
            model_name: Model name for tracking
            profiling_config: Optional custom profiling configuration

        Returns:
            Dictionary containing comprehensive performance metrics
        """
        profiling_config = profiling_config or self.profiling_config
        self.logger.info(f"Starting performance profiling for {model_name}")

        # Move model and inputs to device
        model = model.to(self.device)
        sample_inputs = sample_inputs.to(self.device)
        model.eval()

        profiling_results = {
            "model_name": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(self.device),
            "profiling_config": asdict(profiling_config)
        }

        try:
            # Phase 1: Basic performance metrics
            self.logger.info("Phase 1: Basic performance benchmarking")
            basic_metrics = self._benchmark_basic_performance(
                model, sample_inputs, profiling_config
            )
            profiling_results["basic_metrics"] = asdict(basic_metrics)

            # Phase 2: Detailed profiling
            if profiling_config.enable_detailed_profiling:
                self.logger.info("Phase 2: Detailed profiling")
                detailed_metrics = self._profile_detailed_performance(
                    model, sample_inputs, profiling_config
                )
                profiling_results["detailed_metrics"] = detailed_metrics

            # Phase 3: Memory profiling
            if profiling_config.enable_memory_profiling:
                self.logger.info("Phase 3: Memory profiling")
                memory_metrics = self._profile_memory_usage(
                    model, sample_inputs, profiling_config
                )
                profiling_results["memory_metrics"] = memory_metrics

            # Phase 4: Multi-batch profiling
            self.logger.info("Phase 4: Multi-batch profiling")
            batch_metrics = self._profile_batch_performance(
                model, sample_inputs, profiling_config
            )
            profiling_results["batch_metrics"] = batch_metrics

            # Phase 5: Bottleneck analysis
            if profiling_config.enable_bottleneck_analysis:
                self.logger.info("Phase 5: Bottleneck analysis")
                bottleneck_analysis = self._analyze_bottlenecks(
                    model, sample_inputs, basic_metrics, profiling_config
                )
                profiling_results["bottleneck_analysis"] = asdict(bottleneck_analysis)

            # Phase 6: Operator analysis
            if profiling_config.enable_operator_analysis:
                self.logger.info("Phase 6: Operator analysis")
                operator_analysis = self._analyze_operators(
                    model, sample_inputs, profiling_config
                )
                profiling_results["operator_analysis"] = operator_analysis

            # Store profiling history
            self.profiling_history.append(profiling_results)

            self.logger.info(f"Performance profiling completed for {model_name}")
            self.logger.info(f"Avg latency: {basic_metrics.latency_mean:.2f}ms")
            self.logger.info(f"Throughput: {basic_metrics.throughput_samples_per_sec:.1f} samples/sec")
            self.logger.info(f"Memory usage: {basic_metrics.memory_peak_mb:.1f}MB")

            return profiling_results

        except Exception as e:
            self.logger.error(f"Performance profiling failed: {str(e)}")
            profiling_results["error"] = str(e)
            raise

    def _benchmark_basic_performance(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        config: ProfilingConfig
    ) -> PerformanceMetrics:
        """Benchmark basic performance metrics"""
        metrics = PerformanceMetrics()

        # Model info
        metrics.model_size_mb = self._calculate_model_size(model)
        metrics.parameter_count = sum(p.numel() for p in model.parameters())

        # Clear cache and reset memory stats
        if self.enable_gpu_monitoring:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        for _ in range(config.warmup_iterations):
            with torch.no_grad():
                _ = model(sample_inputs)

        # Synchronization
        self._synchronize_device()

        # Benchmark latency
        latencies = []
        for _ in range(config.benchmark_iterations):
            start_time = time.perf_counter()

            with torch.no_grad():
                _ = model(sample_inputs)

            self._synchronize_device()
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate latency statistics
        latencies = np.array(latencies)
        metrics.latency_mean = float(np.mean(latencies))
        metrics.latency_median = float(np.median(latencies))
        metrics.latency_std = float(np.std(latencies))
        metrics.latency_min = float(np.min(latencies))
        metrics.latency_max = float(np.max(latencies))
        metrics.latency_p95 = float(np.percentile(latencies, 95))
        metrics.latency_p99 = float(np.percentile(latencies, 99))

        # Calculate throughput
        batch_size = sample_inputs.size(0)
        metrics.throughput_samples_per_sec = (batch_size * 1000) / metrics.latency_mean
        metrics.throughput_batches_per_sec = 1000 / metrics.latency_mean

        # Memory metrics
        if self.enable_gpu_monitoring:
            metrics.memory_allocated_mb = torch.cuda.memory_allocated() / (1024**2)
            metrics.memory_cached_mb = torch.cuda.memory_reserved() / (1024**2)
            metrics.memory_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

        # Estimate FLOPs per second
        estimated_flops = self._estimate_model_flops(model, sample_inputs)
        metrics.flops_per_second = estimated_flops * metrics.throughput_samples_per_sec

        return metrics

    def _profile_detailed_performance(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        config: ProfilingConfig
    ) -> Dict[str, Any]:
        """Detailed performance profiling with PyTorch profiler"""
        detailed_metrics = {}

        # Setup profiler
        profiler_config = {
            "activities": [torch.profiler.ProfilerActivity.CPU],
            "record_shapes": True,
            "profile_memory": config.enable_memory_profiling,
            "with_stack": True
        }

        if self.enable_gpu_monitoring:
            profiler_config["activities"].append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(**profiler_config) as prof:
            # Profile multiple iterations
            for _ in range(min(config.benchmark_iterations, 10)):  # Limit for profiler overhead
                with torch.no_grad():
                    _ = model(sample_inputs)

        # Extract profiler statistics
        detailed_metrics["profiler_stats"] = self._extract_profiler_stats(prof)

        # Generate trace if requested
        if config.generate_trace:
            trace_path = f"/tmp/{model.__class__.__name__}_trace.json"
            prof.export_chrome_trace(trace_path)
            detailed_metrics["trace_path"] = trace_path

        return detailed_metrics

    def _extract_profiler_stats(self, prof: torch.profiler.profile) -> Dict[str, Any]:
        """Extract statistics from PyTorch profiler"""
        stats = {}

        # Key averages
        key_averages = prof.key_averages()

        # Top operations by time
        top_ops = sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)[:10]
        stats["top_operations"] = []

        for op in top_ops:
            stats["top_operations"].append({
                "name": op.key,
                "cpu_time_total": op.cpu_time_total,
                "cuda_time_total": op.cuda_time_total,
                "count": op.count,
                "cpu_memory_usage": op.cpu_memory_usage,
                "cuda_memory_usage": op.cuda_memory_usage
            })

        # Memory usage summary
        if hasattr(prof, 'profiler'):
            memory_stats = prof.profiler.kineto_results.memory_profile()
            if memory_stats:
                stats["memory_timeline"] = memory_stats

        return stats

    def _profile_memory_usage(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        config: ProfilingConfig
    ) -> Dict[str, Any]:
        """Detailed memory usage profiling"""
        memory_metrics = {}

        if not self.enable_gpu_monitoring:
            return {"note": "GPU memory profiling not available"}

        # Memory profiling with different batch sizes
        batch_memory_usage = {}

        for batch_size in [1, 2, 4, 8, 16]:
            if batch_size > sample_inputs.size(0):
                continue

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create batch
            batch_input = sample_inputs[:batch_size] if batch_size <= sample_inputs.size(0) else \
                         sample_inputs.repeat(batch_size // sample_inputs.size(0) + 1, *([1] * (len(sample_inputs.shape) - 1)))[:batch_size]

            # Measure memory
            with torch.no_grad():
                _ = model(batch_input)

            batch_memory_usage[batch_size] = {
                "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "peak_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "reserved_mb": torch.cuda.memory_reserved() / (1024**2)
            }

        memory_metrics["batch_memory_usage"] = batch_memory_usage

        # Memory efficiency analysis
        memory_metrics["memory_efficiency"] = self._analyze_memory_efficiency(
            model, sample_inputs, batch_memory_usage
        )

        return memory_metrics

    def _analyze_memory_efficiency(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        batch_memory_usage: Dict[int, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Analyze memory usage efficiency"""
        efficiency_analysis = {}

        # Calculate memory scaling with batch size
        batch_sizes = sorted(batch_memory_usage.keys())
        memory_values = [batch_memory_usage[bs]["peak_mb"] for bs in batch_sizes]

        if len(batch_sizes) > 1:
            # Linear regression to find memory scaling
            coeffs = np.polyfit(batch_sizes, memory_values, 1)
            slope = coeffs[0]  # MB per sample
            intercept = coeffs[1]  # Fixed memory overhead

            efficiency_analysis["memory_per_sample_mb"] = slope
            efficiency_analysis["fixed_overhead_mb"] = intercept
            efficiency_analysis["memory_scaling_efficiency"] = (
                slope / (intercept + slope) if (intercept + slope) > 0 else 0.0
            )

        # Model memory breakdown
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024**2)

        efficiency_analysis["parameter_memory_mb"] = param_memory
        efficiency_analysis["buffer_memory_mb"] = buffer_memory
        efficiency_analysis["activation_memory_mb"] = (
            batch_memory_usage.get(1, {}).get("peak_mb", 0) - param_memory - buffer_memory
        )

        return efficiency_analysis

    def _profile_batch_performance(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        config: ProfilingConfig
    ) -> Dict[str, Any]:
        """Profile performance across different batch sizes"""
        batch_metrics = {}

        for batch_size in config.profile_batch_sizes:
            if batch_size > sample_inputs.size(0) * 4:  # Skip very large batches
                continue

            try:
                # Create batch
                if batch_size <= sample_inputs.size(0):
                    batch_input = sample_inputs[:batch_size]
                else:
                    repeats = batch_size // sample_inputs.size(0) + 1
                    batch_input = sample_inputs.repeat(repeats, *([1] * (len(sample_inputs.shape) - 1)))[:batch_size]

                batch_input = batch_input.to(self.device)

                # Quick benchmark
                latencies = []
                for _ in range(min(config.benchmark_iterations, 50)):
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        _ = model(batch_input)
                    self._synchronize_device()
                    latencies.append((time.perf_counter() - start_time) * 1000)

                batch_metrics[batch_size] = {
                    "latency_mean": float(np.mean(latencies)),
                    "latency_std": float(np.std(latencies)),
                    "throughput_samples_per_sec": (batch_size * 1000) / np.mean(latencies),
                    "latency_per_sample": np.mean(latencies) / batch_size
                }

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    batch_metrics[batch_size] = {"error": "Out of memory"}
                else:
                    batch_metrics[batch_size] = {"error": str(e)}

        # Analyze batch efficiency
        batch_metrics["batch_efficiency_analysis"] = self._analyze_batch_efficiency(batch_metrics)

        return batch_metrics

    def _analyze_batch_efficiency(self, batch_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze batch processing efficiency"""
        analysis = {}

        # Extract successful batch results
        valid_batches = {
            bs: metrics for bs, metrics in batch_metrics.items()
            if isinstance(metrics, dict) and "error" not in metrics and isinstance(bs, int)
        }

        if len(valid_batches) < 2:
            return {"note": "Insufficient data for batch efficiency analysis"}

        batch_sizes = sorted(valid_batches.keys())
        throughputs = [valid_batches[bs]["throughput_samples_per_sec"] for bs in batch_sizes]

        # Find optimal batch size
        max_throughput_idx = np.argmax(throughputs)
        optimal_batch_size = batch_sizes[max_throughput_idx]
        max_throughput = throughputs[max_throughput_idx]

        analysis["optimal_batch_size"] = optimal_batch_size
        analysis["max_throughput"] = max_throughput

        # Calculate efficiency curve
        analysis["batch_efficiency_curve"] = {
            bs: valid_batches[bs]["throughput_samples_per_sec"] / max_throughput
            for bs in batch_sizes
        }

        # Memory limits
        memory_limited_batches = [
            bs for bs, metrics in batch_metrics.items()
            if isinstance(metrics, dict) and metrics.get("error") == "Out of memory"
        ]

        if memory_limited_batches:
            analysis["memory_limit_batch_size"] = min(memory_limited_batches)

        return analysis

    def _analyze_bottlenecks(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        metrics: PerformanceMetrics,
        config: ProfilingConfig
    ) -> BottleneckAnalysis:
        """Analyze performance bottlenecks"""
        bottleneck = BottleneckAnalysis()

        # Memory bottleneck analysis
        if self.enable_gpu_monitoring:
            gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / (1024**2)
            memory_utilization = metrics.memory_peak_mb / gpu_memory_total

            if memory_utilization > 0.9:
                bottleneck.bottleneck_type = "memory"
                bottleneck.bottleneck_severity = min((memory_utilization - 0.7) / 0.3, 1.0)
                bottleneck.bottleneck_location = "GPU memory"
                bottleneck.optimization_suggestions.extend([
                    "Consider model pruning or quantization",
                    "Reduce batch size",
                    "Use gradient checkpointing",
                    "Enable memory-efficient attention"
                ])

        # Compute bottleneck analysis
        theoretical_peak_flops = self._estimate_theoretical_peak_flops()
        if theoretical_peak_flops > 0:
            compute_utilization = metrics.flops_per_second / theoretical_peak_flops

            if compute_utilization < 0.3:  # Low compute utilization
                if bottleneck.bottleneck_type == "unknown":
                    bottleneck.bottleneck_type = "compute"
                    bottleneck.bottleneck_severity = 1.0 - compute_utilization
                    bottleneck.bottleneck_location = "GPU compute"
                    bottleneck.optimization_suggestions.extend([
                        "Optimize operator fusion",
                        "Use mixed precision training",
                        "Increase batch size if memory allows",
                        "Consider kernel optimization"
                    ])

        # Bandwidth bottleneck analysis
        memory_bandwidth_utilization = self._estimate_memory_bandwidth_utilization(metrics)
        if memory_bandwidth_utilization < 0.5:
            if bottleneck.bottleneck_type in ["unknown", "compute"]:
                bottleneck.bottleneck_type = "bandwidth"
                bottleneck.bottleneck_severity = 1.0 - memory_bandwidth_utilization
                bottleneck.bottleneck_location = "Memory bandwidth"
                bottleneck.optimization_suggestions.extend([
                    "Optimize data layout",
                    "Reduce memory transfers",
                    "Use tensor fusion",
                    "Optimize memory access patterns"
                ])

        return bottleneck

    def _analyze_operators(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        config: ProfilingConfig
    ) -> Dict[str, Any]:
        """Analyze individual operator performance"""
        operator_analysis = {}

        # Hook-based operator timing
        operator_times = defaultdict(list)
        operator_memory = defaultdict(list)

        def create_hook(name):
            def hook(module, input, output):
                start_time = time.perf_counter()

                if self.enable_gpu_monitoring:
                    torch.cuda.synchronize()
                    start_memory = torch.cuda.memory_allocated()

                # The actual computation happens during forward pass
                # This hook runs after, so we measure the next operation indirectly

                if self.enable_gpu_monitoring:
                    torch.cuda.synchronize()
                    end_memory = torch.cuda.memory_allocated()
                    memory_diff = (end_memory - start_memory) / (1024**2)
                    operator_memory[name].append(memory_diff)

                end_time = time.perf_counter()
                operator_times[name].append((end_time - start_time) * 1000)

            return hook

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU)):
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)

        # Run profiling
        for _ in range(10):
            with torch.no_grad():
                _ = model(sample_inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Analyze operator performance
        for op_name, times in operator_times.items():
            if times:
                operator_analysis[op_name] = {
                    "avg_time_ms": float(np.mean(times)),
                    "std_time_ms": float(np.std(times)),
                    "total_time_ms": float(np.sum(times)),
                    "call_count": len(times)
                }

                if op_name in operator_memory and operator_memory[op_name]:
                    operator_analysis[op_name]["avg_memory_mb"] = float(np.mean(operator_memory[op_name]))

        return operator_analysis

    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024**2)

    def _estimate_model_flops(self, model: nn.Module, sample_inputs: torch.Tensor) -> int:
        """Estimate FLOPs for a single forward pass"""
        total_flops = 0

        def flop_count_hook(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Conv2d):
                # Convolution FLOPs
                output_dims = output.shape[2:]
                kernel_flops = np.prod(module.kernel_size) * module.in_channels // module.groups
                output_elements = np.prod(output_dims)
                total_flops += kernel_flops * output_elements * module.out_channels

            elif isinstance(module, nn.Linear):
                # Linear layer FLOPs
                total_flops += module.in_features * module.out_features

        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(flop_count_hook)
                hooks.append(hook)

        # Count FLOPs
        with torch.no_grad():
            _ = model(sample_inputs[:1])  # Single sample

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return total_flops

    def _estimate_theoretical_peak_flops(self) -> float:
        """Estimate theoretical peak FLOPs for the device"""
        if not self.enable_gpu_monitoring:
            return 0.0

        # This is a rough estimation based on common GPU architectures
        props = torch.cuda.get_device_properties(self.device)
        compute_capability = f"{props.major}.{props.minor}"

        # Rough FLOPS estimates (TFLOPS) for common architectures
        flops_estimates = {
            "7.5": 65.0,   # RTX 2080 Ti (Turing)
            "8.0": 156.0,  # A100 (Ampere)
            "8.6": 83.0,   # RTX 3090 (Ampere)
            "8.9": 165.0,  # RTX 4090 (Ada Lovelace)
            "9.0": 300.0   # H100 (Hopper)
        }

        peak_tflops = flops_estimates.get(compute_capability, 50.0)  # Default estimate
        return peak_tflops * 1e12  # Convert to FLOPS

    def _estimate_memory_bandwidth_utilization(self, metrics: PerformanceMetrics) -> float:
        """Estimate memory bandwidth utilization"""
        if not self.enable_gpu_monitoring:
            return 0.0

        # Rough memory bandwidth estimates (GB/s) for common GPUs
        props = torch.cuda.get_device_properties(self.device)
        compute_capability = f"{props.major}.{props.minor}"

        bandwidth_estimates = {
            "7.5": 616,   # RTX 2080 Ti
            "8.0": 1555,  # A100
            "8.6": 936,   # RTX 3090
            "8.9": 1008,  # RTX 4090
            "9.0": 2000   # H100
        }

        peak_bandwidth_gb_s = bandwidth_estimates.get(compute_capability, 500)  # Default

        # Estimate actual memory bandwidth usage
        # This is a rough approximation
        bytes_per_sample = metrics.model_size_mb * 1024 * 1024 * 2  # Rough estimate
        actual_bandwidth_gb_s = (bytes_per_sample * metrics.throughput_samples_per_sec) / (1024**3)

        return min(actual_bandwidth_gb_s / peak_bandwidth_gb_s, 1.0)

    def _synchronize_device(self):
        """Synchronize device operations"""
        if self.enable_gpu_monitoring:
            torch.cuda.synchronize()

    def compare_performance(
        self,
        baseline_results: Dict[str, Any],
        current_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare performance between two profiling results"""
        comparison = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_model": baseline_results.get("model_name", "baseline"),
            "current_model": current_results.get("model_name", "current"),
            "improvements": {},
            "regressions": {},
            "summary": {}
        }

        baseline_metrics = baseline_results.get("basic_metrics", {})
        current_metrics = current_results.get("basic_metrics", {})

        # Compare key metrics
        key_metrics = [
            "latency_mean", "throughput_samples_per_sec", "memory_peak_mb",
            "flops_per_second"
        ]

        for metric in key_metrics:
            if metric in baseline_metrics and metric in current_metrics:
                baseline_val = baseline_metrics[metric]
                current_val = current_metrics[metric]

                if baseline_val > 0:
                    improvement = (current_val - baseline_val) / baseline_val

                    if metric == "latency_mean" or metric == "memory_peak_mb":
                        # Lower is better for latency and memory
                        improvement = -improvement

                    comparison["improvements"][metric] = {
                        "baseline": baseline_val,
                        "current": current_val,
                        "improvement_percent": improvement * 100
                    }

                    if improvement > 0.05:  # 5% improvement threshold
                        comparison["summary"][f"{metric}_improved"] = True
                    elif improvement < -0.05:  # 5% regression threshold
                        comparison["regressions"][metric] = improvement * 100

        return comparison

    def generate_performance_report(
        self,
        profiling_results: List[Dict[str, Any]],
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_models": len(profiling_results),
                "device": str(self.device),
                "profiling_config": asdict(self.profiling_config)
            },
            "models": {},
            "comparative_analysis": {},
            "recommendations": []
        }

        # Process individual model results
        for result in profiling_results:
            model_name = result.get("model_name", "unknown")
            report["models"][model_name] = result

        # Generate recommendations
        report["recommendations"] = self._generate_performance_recommendations(profiling_results)

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Performance report saved to {output_path}")
        return report

    def _generate_performance_recommendations(
        self,
        profiling_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        # Analyze common patterns across models
        latencies = []
        memory_usages = []
        bottleneck_types = []

        for result in profiling_results:
            basic_metrics = result.get("basic_metrics", {})
            if basic_metrics:
                latencies.append(basic_metrics.get("latency_mean", 0))
                memory_usages.append(basic_metrics.get("memory_peak_mb", 0))

            bottleneck = result.get("bottleneck_analysis", {})
            if bottleneck:
                bottleneck_types.append(bottleneck.get("bottleneck_type", "unknown"))

        # General recommendations based on patterns
        if latencies and np.mean(latencies) > 100:  # High latency
            recommendations.append(
                "High inference latency detected. Consider model optimization techniques."
            )

        if memory_usages and np.mean(memory_usages) > 8000:  # High memory usage
            recommendations.append(
                "High memory usage detected. Consider quantization or pruning."
            )

        # Bottleneck-specific recommendations
        most_common_bottleneck = max(set(bottleneck_types), key=bottleneck_types.count) \
                                 if bottleneck_types else None

        if most_common_bottleneck == "memory":
            recommendations.append(
                "Memory bottleneck is common. Focus on memory optimization techniques."
            )
        elif most_common_bottleneck == "compute":
            recommendations.append(
                "Compute bottleneck detected. Consider operator fusion and mixed precision."
            )

        return recommendations


def main():
    """Example usage of PerformanceProfiler"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("PerformanceProfiler")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    # Mock configuration
    class MockConfig:
        pass

    config = MockConfig()

    # Initialize profiler
    profiler = PerformanceProfiler(config, device, logger)

    # Example model
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = ExampleModel()
    sample_inputs = torch.randn(4, 3, 32, 32)

    # Profile model
    try:
        results = profiler.profile_model(model, sample_inputs, "example_model")

        print(f"Performance profiling completed!")
        print(f"Average latency: {results['basic_metrics']['latency_mean']:.2f}ms")
        print(f"Throughput: {results['basic_metrics']['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"Memory usage: {results['basic_metrics']['memory_peak_mb']:.1f}MB")

        if "bottleneck_analysis" in results:
            bottleneck = results["bottleneck_analysis"]
            print(f"Bottleneck: {bottleneck['bottleneck_type']} (severity: {bottleneck['bottleneck_severity']:.2f})")

    except Exception as e:
        print(f"Performance profiling failed: {e}")


if __name__ == "__main__":
    main()