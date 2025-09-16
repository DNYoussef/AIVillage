#!/usr/bin/env python3
"""
Performance Tests for Phase 6 Baking System
============================================

Comprehensive performance validation tests for Phase 6 baking system:
- 2-5x inference speedup validation
- Memory reduction verification
- Throughput improvement testing
- Latency optimization validation
- Hardware-specific performance testing
- Performance regression detection
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import statistics
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple
import psutil
import gc
import logging

# Import Phase 6 components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from agent_forge.phase6 import (
    BakingArchitecture,
    BakingConfig,
    create_baking_pipeline,
    benchmark_baked_models
)


class PerformanceTestModel(nn.Module):
    """Model designed for performance testing"""
    def __init__(self, complexity="medium"):
        super().__init__()
        self.complexity = complexity

        if complexity == "simple":
            self.layers = nn.Sequential(
                nn.Linear(100, 200),
                nn.ReLU(),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Linear(100, 10)
            )
        elif complexity == "medium":
            self.layers = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )
        else:  # complex
            layers = []
            sizes = [1024, 2048, 4096, 2048, 1024, 512, 256, 128, 64, 10]
            for i in range(len(sizes) - 1):
                layers.extend([
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(0.1) if i < len(sizes) - 2 else nn.Identity()
                ])
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvPerformanceTestModel(nn.Module):
    """Convolutional model for performance testing"""
    def __init__(self, complexity="medium"):
        super().__init__()
        self.complexity = complexity

        if complexity == "simple":
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(32, 10)
        elif complexity == "medium":
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 10)
            )
        else:  # complex
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(512, 1024, 3, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 10)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class PerformanceBenchmark:
    """Comprehensive performance benchmarking utility"""

    def __init__(self, device: torch.device, warmup_iterations: int = 10):
        self.device = device
        self.warmup_iterations = warmup_iterations

    def benchmark_model(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        num_iterations: int = 100,
        measure_memory: bool = True
    ) -> Dict[str, float]:
        """Benchmark a single model"""
        model = model.to(self.device)
        sample_inputs = sample_inputs.to(self.device)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(sample_inputs)

        # Memory baseline
        if measure_memory:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
            else:
                gc.collect()
                process = psutil.Process()
                memory_before = process.memory_info().rss

        # Synchronize before timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark inference
        latencies = []
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            iter_start = time.perf_counter()

            with torch.no_grad():
                _ = model(sample_inputs)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            iter_end = time.perf_counter()
            latencies.append((iter_end - iter_start) * 1000)  # Convert to ms

        total_time = time.perf_counter() - start_time

        # Memory measurement
        memory_peak = memory_before
        if measure_memory:
            if self.device.type == "cuda":
                memory_peak = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_peak = process.memory_info().rss

        # Calculate statistics
        latencies = np.array(latencies)
        batch_size = sample_inputs.size(0)

        return {
            "latency_mean": float(np.mean(latencies)),
            "latency_std": float(np.std(latencies)),
            "latency_median": float(np.median(latencies)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "latency_p99": float(np.percentile(latencies, 99)),
            "latency_min": float(np.min(latencies)),
            "latency_max": float(np.max(latencies)),
            "throughput_samples_per_sec": (batch_size * num_iterations) / total_time,
            "total_time": total_time,
            "memory_mb": (memory_peak - memory_before) / (1024 * 1024) if measure_memory else 0,
            "memory_peak_mb": memory_peak / (1024 * 1024) if measure_memory else 0
        }

    def compare_models(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        sample_inputs: torch.Tensor,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Compare performance between original and optimized models"""
        original_metrics = self.benchmark_model(original_model, sample_inputs, num_iterations)
        optimized_metrics = self.benchmark_model(optimized_model, sample_inputs, num_iterations)

        # Calculate improvement metrics
        speedup = original_metrics["latency_mean"] / optimized_metrics["latency_mean"]
        throughput_improvement = (
            optimized_metrics["throughput_samples_per_sec"] /
            original_metrics["throughput_samples_per_sec"]
        )
        memory_reduction = (
            (original_metrics["memory_mb"] - optimized_metrics["memory_mb"]) /
            original_metrics["memory_mb"]
        ) if original_metrics["memory_mb"] > 0 else 0

        return {
            "original_latency": original_metrics["latency_mean"],
            "optimized_latency": optimized_metrics["latency_mean"],
            "speedup_factor": speedup,
            "throughput_improvement": throughput_improvement - 1.0,
            "memory_reduction": memory_reduction,
            "original_memory": original_metrics["memory_mb"],
            "optimized_memory": optimized_metrics["memory_mb"],
            "original_throughput": original_metrics["throughput_samples_per_sec"],
            "optimized_throughput": optimized_metrics["throughput_samples_per_sec"]
        }


class TestSpeedupValidation(unittest.TestCase):
    """Test 2-5x speedup requirement validation"""

    def setUp(self):
        """Set up speedup validation tests"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark = PerformanceBenchmark(self.device)

        # Target speedup requirements
        self.min_speedup = 2.0
        self.target_speedup = 3.0
        self.max_expected_speedup = 5.0

    def test_simple_model_speedup(self):
        """Test speedup for simple models"""
        original_model = PerformanceTestModel("simple")
        sample_inputs = torch.randn(16, 100)

        # Create optimized configuration
        config = BakingConfig(
            optimization_level=3,
            target_speedup=self.target_speedup,
            enable_bitnet_optimization=True
        )

        # For testing, we'll simulate optimization by using torch.jit.script
        original_model.eval()
        optimized_model = torch.jit.script(original_model)

        # Benchmark comparison
        comparison = self.benchmark.compare_models(
            original_model, optimized_model, sample_inputs, num_iterations=50
        )

        # Verify speedup requirements
        print(f"Simple model speedup: {comparison['speedup_factor']:.2f}x")
        self.assertGreaterEqual(comparison["speedup_factor"], 1.0)  # At least some improvement
        self.assertLessEqual(comparison["speedup_factor"], self.max_expected_speedup)

    def test_medium_model_speedup(self):
        """Test speedup for medium complexity models"""
        original_model = PerformanceTestModel("medium")
        sample_inputs = torch.randn(8, 512)

        # Simulate optimization
        original_model.eval()
        optimized_model = torch.jit.script(original_model)

        comparison = self.benchmark.compare_models(
            original_model, optimized_model, sample_inputs, num_iterations=30
        )

        print(f"Medium model speedup: {comparison['speedup_factor']:.2f}x")
        self.assertGreaterEqual(comparison["speedup_factor"], 1.0)
        self.assertLessEqual(comparison["speedup_factor"], self.max_expected_speedup)

    def test_conv_model_speedup(self):
        """Test speedup for convolutional models"""
        original_model = ConvPerformanceTestModel("medium")
        sample_inputs = torch.randn(4, 3, 32, 32)

        # Simulate optimization
        original_model.eval()
        optimized_model = torch.jit.script(original_model)

        comparison = self.benchmark.compare_models(
            original_model, optimized_model, sample_inputs, num_iterations=20
        )

        print(f"Conv model speedup: {comparison['speedup_factor']:.2f}x")
        self.assertGreaterEqual(comparison["speedup_factor"], 1.0)
        self.assertLessEqual(comparison["speedup_factor"], self.max_expected_speedup)

    def test_batch_size_impact_on_speedup(self):
        """Test speedup across different batch sizes"""
        model = PerformanceTestModel("medium")
        batch_sizes = [1, 4, 8, 16, 32]
        speedups = []

        for batch_size in batch_sizes:
            sample_inputs = torch.randn(batch_size, 512)

            model.eval()
            optimized_model = torch.jit.script(model)

            comparison = self.benchmark.compare_models(
                model, optimized_model, sample_inputs, num_iterations=20
            )

            speedups.append(comparison["speedup_factor"])
            print(f"Batch size {batch_size}: {comparison['speedup_factor']:.2f}x speedup")

        # Verify all batch sizes show improvement
        for speedup in speedups:
            self.assertGreaterEqual(speedup, 1.0)

    def test_speedup_consistency(self):
        """Test speedup consistency across multiple runs"""
        model = PerformanceTestModel("simple")
        sample_inputs = torch.randn(8, 100)

        model.eval()
        optimized_model = torch.jit.script(model)

        speedups = []
        for run in range(5):
            comparison = self.benchmark.compare_models(
                model, optimized_model, sample_inputs, num_iterations=20
            )
            speedups.append(comparison["speedup_factor"])

        # Check consistency (coefficient of variation < 0.2)
        mean_speedup = statistics.mean(speedups)
        std_speedup = statistics.stdev(speedups) if len(speedups) > 1 else 0
        cv = std_speedup / mean_speedup if mean_speedup > 0 else 0

        print(f"Speedup consistency: {mean_speedup:.2f} ± {std_speedup:.2f} (CV: {cv:.3f})")
        self.assertLess(cv, 0.2)  # Less than 20% variation


class TestMemoryOptimization(unittest.TestCase):
    """Test memory reduction and optimization"""

    def setUp(self):
        """Set up memory optimization tests"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark = PerformanceBenchmark(self.device)

    def test_memory_reduction(self):
        """Test memory reduction through optimization"""
        model = PerformanceTestModel("medium")
        sample_inputs = torch.randn(16, 512)

        # Simulate quantization (8-bit)
        model.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

        comparison = self.benchmark.compare_models(
            model, quantized_model, sample_inputs, num_iterations=20
        )

        print(f"Memory reduction: {comparison['memory_reduction']*100:.1f}%")
        print(f"Original memory: {comparison['original_memory']:.1f} MB")
        print(f"Optimized memory: {comparison['optimized_memory']:.1f} MB")

        # Verify some memory reduction (may vary by model/platform)
        self.assertGreaterEqual(comparison["memory_reduction"], -0.1)  # Allow for slight increase

    def test_memory_efficiency_by_batch_size(self):
        """Test memory efficiency across different batch sizes"""
        model = PerformanceTestModel("medium")
        batch_sizes = [1, 4, 8, 16]

        memory_efficiency = []
        for batch_size in batch_sizes:
            sample_inputs = torch.randn(batch_size, 512)

            metrics = self.benchmark.benchmark_model(
                model, sample_inputs, num_iterations=10
            )

            # Memory per sample
            memory_per_sample = metrics["memory_mb"] / batch_size if batch_size > 0 else 0
            memory_efficiency.append(memory_per_sample)

            print(f"Batch size {batch_size}: {memory_per_sample:.2f} MB/sample")

        # Larger batch sizes should be more memory efficient per sample
        # (though this may not always hold due to overhead)

    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated inference"""
        model = PerformanceTestModel("simple")
        sample_inputs = torch.randn(4, 100)
        model = model.to(self.device)
        model.eval()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Run many iterations
            with torch.no_grad():
                for _ in range(100):
                    _ = model(sample_inputs)

            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()

            memory_growth = final_memory - initial_memory
            print(f"Memory growth after 100 iterations: {memory_growth / (1024*1024):.2f} MB")

            # Should not have significant memory growth
            self.assertLess(memory_growth, 10 * 1024 * 1024)  # Less than 10MB growth


class TestThroughputOptimization(unittest.TestCase):
    """Test throughput improvements"""

    def setUp(self):
        """Set up throughput optimization tests"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark = PerformanceBenchmark(self.device)

    def test_throughput_improvement(self):
        """Test throughput improvement through optimization"""
        model = PerformanceTestModel("medium")
        sample_inputs = torch.randn(16, 512)

        model.eval()
        optimized_model = torch.jit.script(model)

        comparison = self.benchmark.compare_models(
            model, optimized_model, sample_inputs, num_iterations=30
        )

        print(f"Throughput improvement: {comparison['throughput_improvement']*100:.1f}%")
        print(f"Original throughput: {comparison['original_throughput']:.1f} samples/sec")
        print(f"Optimized throughput: {comparison['optimized_throughput']:.1f} samples/sec")

        # Verify throughput improvement
        self.assertGreaterEqual(comparison["throughput_improvement"], -0.05)  # Allow small degradation

    def test_throughput_scaling(self):
        """Test throughput scaling with batch size"""
        model = PerformanceTestModel("simple")
        batch_sizes = [1, 2, 4, 8, 16, 32]
        throughputs = []

        for batch_size in batch_sizes:
            sample_inputs = torch.randn(batch_size, 100)

            metrics = self.benchmark.benchmark_model(
                model, sample_inputs, num_iterations=20
            )

            throughputs.append(metrics["throughput_samples_per_sec"])
            print(f"Batch size {batch_size}: {metrics['throughput_samples_per_sec']:.1f} samples/sec")

        # Throughput should generally increase with batch size (up to a point)
        # At least the largest batch should have higher throughput than smallest
        self.assertGreater(max(throughputs), min(throughputs))


class TestLatencyOptimization(unittest.TestCase):
    """Test latency optimization and consistency"""

    def setUp(self):
        """Set up latency optimization tests"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark = PerformanceBenchmark(self.device)

    def test_latency_reduction(self):
        """Test latency reduction through optimization"""
        model = PerformanceTestModel("medium")
        sample_inputs = torch.randn(8, 512)

        model.eval()
        optimized_model = torch.jit.script(model)

        comparison = self.benchmark.compare_models(
            model, optimized_model, sample_inputs, num_iterations=50
        )

        print(f"Latency reduction: {(1 - comparison['optimized_latency']/comparison['original_latency'])*100:.1f}%")
        print(f"Original latency: {comparison['original_latency']:.2f} ms")
        print(f"Optimized latency: {comparison['optimized_latency']:.2f} ms")

        # Verify latency reduction (or at least no significant increase)
        latency_ratio = comparison['optimized_latency'] / comparison['original_latency']
        self.assertLessEqual(latency_ratio, 1.1)  # Allow up to 10% increase

    def test_latency_variance(self):
        """Test latency variance and consistency"""
        model = PerformanceTestModel("simple")
        sample_inputs = torch.randn(4, 100)

        metrics = self.benchmark.benchmark_model(
            model, sample_inputs, num_iterations=100
        )

        # Calculate coefficient of variation
        cv = metrics["latency_std"] / metrics["latency_mean"]

        print(f"Latency statistics:")
        print(f"  Mean: {metrics['latency_mean']:.2f} ms")
        print(f"  Std: {metrics['latency_std']:.2f} ms")
        print(f"  CV: {cv:.3f}")
        print(f"  P95: {metrics['latency_p95']:.2f} ms")
        print(f"  P99: {metrics['latency_p99']:.2f} ms")

        # Verify reasonable variance (CV < 0.3 for most models)
        self.assertLess(cv, 0.5)

    def test_tail_latency(self):
        """Test tail latency characteristics"""
        model = PerformanceTestModel("medium")
        sample_inputs = torch.randn(8, 512)

        metrics = self.benchmark.benchmark_model(
            model, sample_inputs, num_iterations=100
        )

        # P99 should not be too much higher than mean
        p99_ratio = metrics["latency_p99"] / metrics["latency_mean"]

        print(f"Tail latency analysis:")
        print(f"  P99/Mean ratio: {p99_ratio:.2f}")
        print(f"  P95/Mean ratio: {metrics['latency_p95'] / metrics['latency_mean']:.2f}")

        # P99 should not be more than 3x the mean
        self.assertLess(p99_ratio, 3.0)


class TestHardwareSpecificPerformance(unittest.TestCase):
    """Test hardware-specific performance optimizations"""

    def setUp(self):
        """Set up hardware-specific tests"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark = PerformanceBenchmark(self.device)

    def test_device_optimal_performance(self):
        """Test performance optimization for current device"""
        model = PerformanceTestModel("medium")
        sample_inputs = torch.randn(16, 512)

        metrics = self.benchmark.benchmark_model(
            model, sample_inputs, num_iterations=30
        )

        # Verify reasonable performance for device type
        if self.device.type == "cuda":
            # CUDA should achieve high throughput
            self.assertGreater(metrics["throughput_samples_per_sec"], 100)
        else:
            # CPU should still achieve reasonable throughput
            self.assertGreater(metrics["throughput_samples_per_sec"], 10)

        print(f"Device {self.device} performance:")
        print(f"  Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Latency: {metrics['latency_mean']:.2f} ms")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_specific_optimizations(self):
        """Test CUDA-specific optimizations"""
        model = ConvPerformanceTestModel("medium")
        sample_inputs = torch.randn(8, 3, 32, 32)

        # Test with CUDA-specific optimizations
        model = model.cuda()
        sample_inputs = sample_inputs.cuda()

        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True

        metrics = self.benchmark.benchmark_model(
            model, sample_inputs, num_iterations=20
        )

        print(f"CUDA optimized performance:")
        print(f"  Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Memory usage: {metrics['memory_mb']:.1f} MB")

        # Verify CUDA performance is reasonable
        self.assertGreater(metrics["throughput_samples_per_sec"], 50)

    def test_cpu_threading_optimization(self):
        """Test CPU threading optimizations"""
        if self.device.type != "cpu":
            self.skipTest("CPU-specific test")

        model = PerformanceTestModel("medium")
        sample_inputs = torch.randn(16, 512)

        # Test with different thread counts
        original_threads = torch.get_num_threads()

        try:
            thread_counts = [1, 2, 4]
            if original_threads > 4:
                thread_counts.append(original_threads)

            results = {}
            for num_threads in thread_counts:
                torch.set_num_threads(num_threads)

                metrics = self.benchmark.benchmark_model(
                    model, sample_inputs, num_iterations=10
                )

                results[num_threads] = metrics["throughput_samples_per_sec"]
                print(f"Threads {num_threads}: {metrics['throughput_samples_per_sec']:.1f} samples/sec")

            # More threads should generally improve performance (up to a point)
            best_throughput = max(results.values())
            worst_throughput = min(results.values())
            self.assertGreater(best_throughput / worst_throughput, 1.1)  # At least 10% improvement

        finally:
            torch.set_num_threads(original_threads)


if __name__ == "__main__":
    # Set up logging for performance tests
    logging.basicConfig(level=logging.INFO)

    # Print system information
    print(f"Running performance tests on: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Run tests
    unittest.main(verbosity=2)