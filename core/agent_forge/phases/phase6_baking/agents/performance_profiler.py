"""
Phase 6 Baking - Performance Profiler Agent
Profiles and analyzes model performance during tool/persona baking optimization
"""

import torch
import torch.nn as nn
import torch.profiler as profiler
import numpy as np
import logging
import time
import asyncio
import psutil
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import pickle
from collections import defaultdict, deque
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfilerMode(Enum):
    CPU_ONLY = "cpu_only"
    GPU_ONLY = "gpu_only"
    CPU_GPU = "cpu_gpu"
    MEMORY = "memory"
    COMPREHENSIVE = "comprehensive"


class ProfilingMetric(Enum):
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"
    CPU_UTILIZATION = "cpu_utilization"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    FLOPS = "flops"
    BANDWIDTH = "bandwidth"


@dataclass
class ProfilingConfig:
    mode: ProfilerMode
    metrics: List[ProfilingMetric]
    profiling_duration: int
    sample_frequency: int
    enable_detailed_trace: bool
    memory_profiling: bool
    gpu_profiling: bool
    export_chrome_trace: bool
    export_tensorboard: bool
    profile_memory_allocations: bool
    record_shapes: bool
    with_stack: bool


@dataclass
class PerformanceSnapshot:
    timestamp: datetime
    execution_time_ms: float
    memory_usage_mb: float
    gpu_memory_mb: float
    cpu_utilization: float
    gpu_utilization: float
    throughput_ops_per_sec: float
    latency_ms: float
    flops: float
    memory_bandwidth_gbps: float
    bottleneck_analysis: Dict[str, Any]


@dataclass
class ProfilingMetrics:
    snapshots: List[PerformanceSnapshot]
    aggregated_stats: Dict[str, Any]
    bottlenecks: List[str]
    optimization_suggestions: List[str]
    profiling_overhead: float
    monitoring_active: bool
    last_update: datetime


class PerformanceProfiler:
    """Advanced performance profiler with comprehensive metrics"""

    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.metrics = ProfilingMetrics(
            snapshots=[],
            aggregated_stats={},
            bottlenecks=[],
            optimization_suggestions=[],
            profiling_overhead=0.0,
            monitoring_active=False,
            last_update=datetime.now()
        )
        self.profiler = None
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.performance_history = deque(maxlen=1000)

    def start_profiling(self, model: nn.Module) -> bool:
        """Start performance profiling"""
        try:
            if self.config.mode == ProfilerMode.COMPREHENSIVE:
                # Use PyTorch profiler for comprehensive analysis
                self.profiler = profiler.profile(
                    activities=[
                        profiler.ProfilerActivity.CPU,
                        profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None
                    ],
                    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                    on_trace_ready=self._handle_trace_ready,
                    record_shapes=self.config.record_shapes,
                    with_stack=self.config.with_stack,
                    profile_memory=self.config.memory_profiling
                )
                self.profiler.start()

            # Start continuous monitoring
            self.metrics.monitoring_active = True
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._continuous_monitoring)
            self.monitoring_thread.start()

            logger.info(f"Performance profiling started in {self.config.mode.value} mode")
            return True

        except Exception as e:
            logger.error(f"Failed to start profiling: {e}")
            return False

    def stop_profiling(self) -> Dict[str, Any]:
        """Stop performance profiling and return results"""
        try:
            # Stop continuous monitoring
            self.stop_monitoring.set()
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)

            # Stop PyTorch profiler
            if self.profiler:
                self.profiler.stop()

            self.metrics.monitoring_active = False

            # Generate final report
            return self._generate_profiling_report()

        except Exception as e:
            logger.error(f"Failed to stop profiling: {e}")
            return {'error': str(e)}

    def _continuous_monitoring(self):
        """Continuous performance monitoring thread"""
        while not self.stop_monitoring.is_set():
            try:
                snapshot = self._capture_performance_snapshot()
                if snapshot:
                    self.metrics.snapshots.append(snapshot)
                    self.performance_history.append(snapshot)
                    self._analyze_bottlenecks(snapshot)

                time.sleep(1.0 / self.config.sample_frequency)

            except Exception as e:
                logger.warning(f"Monitoring iteration failed: {e}")

    def _capture_performance_snapshot(self) -> Optional[PerformanceSnapshot]:
        """Capture current performance snapshot"""
        try:
            timestamp = datetime.now()

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB

            # GPU metrics
            gpu_memory = 0.0
            gpu_utilization = 0.0

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = gpu_info.gpu
                except ImportError:
                    gpu_utilization = 0.0

            # Compute derived metrics
            throughput = self._estimate_throughput()
            latency = self._estimate_latency()
            flops = self._estimate_flops()
            bandwidth = self._estimate_memory_bandwidth()

            # Bottleneck analysis
            bottleneck_analysis = self._analyze_current_bottlenecks(
                cpu_percent, memory_usage, gpu_memory, gpu_utilization
            )

            return PerformanceSnapshot(
                timestamp=timestamp,
                execution_time_ms=0.0,  # Will be filled by specific operations
                memory_usage_mb=memory_usage,
                gpu_memory_mb=gpu_memory,
                cpu_utilization=cpu_percent,
                gpu_utilization=gpu_utilization,
                throughput_ops_per_sec=throughput,
                latency_ms=latency,
                flops=flops,
                memory_bandwidth_gbps=bandwidth,
                bottleneck_analysis=bottleneck_analysis
            )

        except Exception as e:
            logger.warning(f"Failed to capture performance snapshot: {e}")
            return None

    def _estimate_throughput(self) -> float:
        """Estimate current throughput"""
        # Simplified estimation based on recent performance
        if len(self.performance_history) >= 2:
            recent_snapshots = list(self.performance_history)[-10:]
            if len(recent_snapshots) >= 2:
                time_diff = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds()
                if time_diff > 0:
                    return len(recent_snapshots) / time_diff

        return 0.0

    def _estimate_latency(self) -> float:
        """Estimate current latency"""
        # Simplified estimation
        if len(self.performance_history) >= 2:
            recent_snapshots = list(self.performance_history)[-5:]
            avg_cpu = np.mean([s.cpu_utilization for s in recent_snapshots])
            return max(1.0, avg_cpu / 10.0)  # Rough approximation

        return 1.0

    def _estimate_flops(self) -> float:
        """Estimate floating point operations per second"""
        # This would require more sophisticated profiling
        # Simplified estimation based on GPU utilization
        if torch.cuda.is_available() and len(self.performance_history) > 0:
            latest = self.performance_history[-1]
            # Rough estimation: assume modern GPU can do ~10 TFLOPS at 100% util
            return latest.gpu_utilization * 10e12 / 100.0

        return 0.0

    def _estimate_memory_bandwidth(self) -> float:
        """Estimate memory bandwidth utilization"""
        # Simplified estimation
        if len(self.performance_history) >= 2:
            recent_snapshots = list(self.performance_history)[-5:]
            memory_changes = [
                abs(recent_snapshots[i].memory_usage_mb - recent_snapshots[i-1].memory_usage_mb)
                for i in range(1, len(recent_snapshots))
            ]
            if memory_changes:
                avg_change = np.mean(memory_changes)
                return avg_change / 1024.0  # GB/s approximation

        return 0.0

    def _analyze_current_bottlenecks(self, cpu_util: float, memory_mb: float,
                                   gpu_memory_mb: float, gpu_util: float) -> Dict[str, Any]:
        """Analyze current performance bottlenecks"""
        bottlenecks = {}

        # CPU bottleneck
        if cpu_util > 80.0:
            bottlenecks['cpu'] = {
                'severity': 'high' if cpu_util > 95.0 else 'medium',
                'utilization': cpu_util,
                'recommendation': 'Consider CPU optimization or more cores'
            }

        # Memory bottleneck
        total_memory = psutil.virtual_memory().total / 1024 / 1024  # MB
        memory_percent = (memory_mb / total_memory) * 100

        if memory_percent > 80.0:
            bottlenecks['memory'] = {
                'severity': 'high' if memory_percent > 95.0 else 'medium',
                'usage_percent': memory_percent,
                'usage_mb': memory_mb,
                'recommendation': 'Consider memory optimization or more RAM'
            }

        # GPU bottleneck
        if torch.cuda.is_available() and gpu_util > 80.0:
            bottlenecks['gpu'] = {
                'severity': 'high' if gpu_util > 95.0 else 'medium',
                'utilization': gpu_util,
                'memory_mb': gpu_memory_mb,
                'recommendation': 'Consider GPU optimization or better GPU'
            }

        return bottlenecks

    def _analyze_bottlenecks(self, snapshot: PerformanceSnapshot):
        """Analyze bottlenecks and generate suggestions"""
        # Update bottleneck list
        for bottleneck_type, info in snapshot.bottleneck_analysis.items():
            if info.get('severity') == 'high':
                bottleneck_msg = f"{bottleneck_type.upper()} bottleneck detected: {info.get('recommendation', '')}"
                if bottleneck_msg not in self.metrics.bottlenecks:
                    self.metrics.bottlenecks.append(bottleneck_msg)

        # Generate optimization suggestions
        self._generate_optimization_suggestions(snapshot)

    def _generate_optimization_suggestions(self, snapshot: PerformanceSnapshot):
        """Generate performance optimization suggestions"""
        suggestions = []

        # High memory usage
        if snapshot.memory_usage_mb > 1000:  # > 1GB
            suggestions.append("Consider model pruning or quantization to reduce memory usage")

        # Low GPU utilization
        if torch.cuda.is_available() and snapshot.gpu_utilization < 50.0:
            suggestions.append("GPU underutilized - consider increasing batch size or model complexity")

        # High latency
        if snapshot.latency_ms > 100.0:
            suggestions.append("High latency detected - consider model acceleration techniques")

        # Low throughput
        if snapshot.throughput_ops_per_sec < 10.0:
            suggestions.append("Low throughput - consider optimizing data loading or model architecture")

        # Add new suggestions
        for suggestion in suggestions:
            if suggestion not in self.metrics.optimization_suggestions:
                self.metrics.optimization_suggestions.append(suggestion)

    def _handle_trace_ready(self, prof):
        """Handle PyTorch profiler trace ready"""
        try:
            if self.config.export_chrome_trace:
                trace_path = Path("profiler_trace.json")
                prof.export_chrome_trace(str(trace_path))
                logger.info(f"Chrome trace exported to {trace_path}")

            if self.config.export_tensorboard:
                tb_path = Path("tensorboard_logs")
                tb_path.mkdir(exist_ok=True)
                prof.export_stacks(str(tb_path / "profiler_stacks.txt"), "self_cuda_time_total")

        except Exception as e:
            logger.warning(f"Failed to export profiler trace: {e}")

    def profile_operation(self, operation_func, *args, **kwargs) -> Dict[str, Any]:
        """Profile a specific operation"""
        try:
            # Warm up
            for _ in range(3):
                operation_func(*args, **kwargs)

            # Clear cache and garbage collect
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

            # Profile operation
            start_time = time.time()
            memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            with profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True
            ) as prof:
                result = operation_func(*args, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            # Analyze results
            execution_time = (end_time - start_time) * 1000  # ms
            memory_delta = (memory_after - memory_before) / 1024 / 1024  # MB

            # Get profiler stats
            prof_stats = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)

            return {
                'execution_time_ms': execution_time,
                'memory_delta_mb': memory_delta,
                'profiler_stats': prof_stats,
                'result': result
            }

        except Exception as e:
            logger.error(f"Operation profiling failed: {e}")
            return {'error': str(e)}

    def _generate_profiling_report(self) -> Dict[str, Any]:
        """Generate comprehensive profiling report"""
        if not self.metrics.snapshots:
            return {'error': 'No profiling data available'}

        # Aggregate statistics
        snapshots = self.metrics.snapshots

        avg_cpu = np.mean([s.cpu_utilization for s in snapshots])
        avg_memory = np.mean([s.memory_usage_mb for s in snapshots])
        avg_gpu_util = np.mean([s.gpu_utilization for s in snapshots])
        avg_throughput = np.mean([s.throughput_ops_per_sec for s in snapshots])

        self.metrics.aggregated_stats = {
            'avg_cpu_utilization': avg_cpu,
            'avg_memory_usage_mb': avg_memory,
            'avg_gpu_utilization': avg_gpu_util,
            'avg_throughput': avg_throughput,
            'peak_memory_mb': max([s.memory_usage_mb for s in snapshots]),
            'peak_gpu_memory_mb': max([s.gpu_memory_mb for s in snapshots]),
            'total_snapshots': len(snapshots),
            'profiling_duration_sec': (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds()
        }

        return {
            'aggregated_stats': self.metrics.aggregated_stats,
            'bottlenecks': self.metrics.bottlenecks,
            'optimization_suggestions': self.metrics.optimization_suggestions,
            'profiling_overhead': self.metrics.profiling_overhead,
            'config': asdict(self.config),
            'snapshot_count': len(self.metrics.snapshots)
        }

    async def profile_model_async(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Asynchronously profile model performance"""
        try:
            # Profile forward pass
            def forward_pass():
                with torch.no_grad():
                    return model(sample_input)

            forward_stats = self.profile_operation(forward_pass)

            # Start continuous profiling
            self.start_profiling(model)

            # Let it run for configured duration
            await asyncio.sleep(self.config.profiling_duration)

            # Stop and get report
            report = self.stop_profiling()

            return {
                'forward_pass_stats': forward_stats,
                'continuous_profiling': report,
                'model_profiled': True
            }

        except Exception as e:
            logger.error(f"Async model profiling failed: {e}")
            return {'error': str(e), 'model_profiled': False}


def create_default_profiling_config() -> ProfilingConfig:
    """Create default profiling configuration"""
    return ProfilingConfig(
        mode=ProfilerMode.COMPREHENSIVE,
        metrics=[
            ProfilingMetric.EXECUTION_TIME,
            ProfilingMetric.MEMORY_USAGE,
            ProfilingMetric.GPU_UTILIZATION,
            ProfilingMetric.THROUGHPUT
        ],
        profiling_duration=30,
        sample_frequency=10,
        enable_detailed_trace=True,
        memory_profiling=True,
        gpu_profiling=torch.cuda.is_available(),
        export_chrome_trace=True,
        export_tensorboard=False,
        profile_memory_allocations=True,
        record_shapes=True,
        with_stack=False
    )


# Agent Integration Interface
class PerformanceProfilerAgent:
    """Agent wrapper for performance profiler"""

    def __init__(self, config: Optional[ProfilingConfig] = None):
        self.config = config or create_default_profiling_config()
        self.profiler = PerformanceProfiler(self.config)
        self.agent_id = "performance_profiler"
        self.status = "idle"

    async def run(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """Run profiling agent"""
        self.status = "running"

        try:
            sample_input = kwargs.get('sample_input')
            if sample_input is None:
                # Create default sample input
                sample_input = torch.randn(1, 512, 768)  # Default transformer input

            result = await self.profiler.profile_model_async(model, sample_input)
            self.status = "completed"
            return result

        except Exception as e:
            self.status = "failed"
            logger.error(f"Performance profiler failed: {e}")
            return {'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'profiling_active': self.profiler.metrics.monitoring_active,
            'snapshot_count': len(self.profiler.metrics.snapshots)
        }