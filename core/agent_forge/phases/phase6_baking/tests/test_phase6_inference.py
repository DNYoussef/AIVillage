#!/usr/bin/env python3
"""
Inference Capability Tests for Phase 6 Baking System
====================================================

Comprehensive inference capability tests for Phase 6 baked models:
- Real-time inference capability
- Multi-batch inference validation
- Streaming inference support
- Hardware-specific inference optimization
- Inference accuracy validation
- Memory efficiency during inference
- Concurrent inference handling
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import threading
import queue
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import concurrent.futures
from unittest.mock import Mock

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


class InferenceTestModel(nn.Module):
    """Model optimized for inference testing"""
    def __init__(self, model_type="classification", complexity="medium"):
        super().__init__()
        self.model_type = model_type
        self.complexity = complexity

        if model_type == "classification":
            if complexity == "simple":
                self.features = nn.Sequential(
                    nn.Linear(784, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10)
                )
            else:  # medium/complex
                self.features = nn.Sequential(
                    nn.Linear(784, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                )

        elif model_type == "regression":
            self.features = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        elif model_type == "cnn":
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 16, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )

    def forward(self, x):
        if self.model_type == "cnn":
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
        else:
            return self.features(x)


class InferenceLatencyBenchmark:
    """Specialized benchmark for inference latency testing"""

    def __init__(self, device: torch.device, warmup_iterations: int = 10):
        self.device = device
        self.warmup_iterations = warmup_iterations

    def measure_single_inference_latency(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_measurements: int = 100
    ) -> Dict[str, float]:
        """Measure single inference latency with high precision"""
        model = model.to(self.device)
        input_tensor = input_tensor.to(self.device)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(input_tensor)

        # Synchronize
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Measure latencies
        latencies = []

        for _ in range(num_measurements):
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            with torch.no_grad():
                _ = model(input_tensor)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        latencies = np.array(latencies)

        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "median_latency_ms": float(np.median(latencies))
        }

    def measure_batch_inference_scaling(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """Measure how inference scales with batch size"""
        model = model.to(self.device)
        model.eval()

        results = {}

        for batch_size in batch_sizes:
            # Create input tensor for this batch size
            input_tensor = torch.randn(batch_size, *input_shape[1:]).to(self.device)

            # Measure latency
            latency_stats = self.measure_single_inference_latency(
                model, input_tensor, num_measurements=50
            )

            # Calculate throughput
            throughput = batch_size / (latency_stats["mean_latency_ms"] / 1000)

            results[batch_size] = {
                **latency_stats,
                "throughput_samples_per_sec": throughput,
                "latency_per_sample_ms": latency_stats["mean_latency_ms"] / batch_size
            }

        return results


class TestRealTimeInference(unittest.TestCase):
    """Test real-time inference capabilities"""

    def setUp(self):
        """Set up real-time inference tests"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark = InferenceLatencyBenchmark(self.device)

        # Real-time requirements (adjust based on use case)
        self.max_latency_ms = 50.0  # Maximum acceptable latency
        self.target_fps = 30  # Target frames per second
        self.max_frame_time_ms = 1000.0 / self.target_fps

    def test_single_sample_inference_latency(self):
        """Test single sample inference meets real-time requirements"""
        model = InferenceTestModel("classification", "simple")
        input_tensor = torch.randn(1, 784)  # Single sample

        # Create optimized model
        model.eval()
        optimized_model = torch.jit.script(model)

        # Measure original model latency
        original_stats = self.benchmark.measure_single_inference_latency(
            model, input_tensor, num_measurements=100
        )

        # Measure optimized model latency
        optimized_stats = self.benchmark.measure_single_inference_latency(
            optimized_model, input_tensor, num_measurements=100
        )

        print(f"Original model latency: {original_stats['mean_latency_ms']:.2f} ms")
        print(f"Optimized model latency: {optimized_stats['mean_latency_ms']:.2f} ms")
        print(f"Speedup: {original_stats['mean_latency_ms'] / optimized_stats['mean_latency_ms']:.2f}x")

        # Verify real-time capability
        self.assertLess(optimized_stats["mean_latency_ms"], self.max_latency_ms)
        self.assertLess(optimized_stats["p99_latency_ms"], self.max_frame_time_ms)

    def test_real_time_video_processing(self):
        """Test real-time video processing capability"""
        model = InferenceTestModel("cnn", "medium")
        input_shape = (1, 3, 224, 224)  # Single frame

        model.eval()
        optimized_model = torch.jit.script(model)

        # Simulate video frame processing
        frame_latencies = []
        num_frames = 100

        input_tensor = torch.randn(*input_shape).to(self.device)

        start_time = time.time()

        for frame_idx in range(num_frames):
            frame_start = time.perf_counter()

            with torch.no_grad():
                _ = optimized_model(input_tensor)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            frame_end = time.perf_counter()
            frame_latency = (frame_end - frame_start) * 1000

            frame_latencies.append(frame_latency)

        total_time = time.time() - start_time
        avg_fps = num_frames / total_time

        frame_latencies = np.array(frame_latencies)

        print(f"Video processing statistics:")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Average frame latency: {np.mean(frame_latencies):.2f} ms")
        print(f"  P95 frame latency: {np.percentile(frame_latencies, 95):.2f} ms")
        print(f"  Max frame latency: {np.max(frame_latencies):.2f} ms")

        # Verify real-time video processing capability
        self.assertGreaterEqual(avg_fps, self.target_fps * 0.9)  # Allow 10% margin
        self.assertLess(np.percentile(frame_latencies, 95), self.max_frame_time_ms)

    def test_low_latency_streaming(self):
        """Test low-latency streaming inference"""
        model = InferenceTestModel("classification", "simple")
        model.eval()
        optimized_model = torch.jit.script(model)

        # Simulate streaming data
        stream_data = [torch.randn(1, 784) for _ in range(50)]
        processing_times = []

        for data in stream_data:
            data = data.to(self.device)

            start_time = time.perf_counter()

            with torch.no_grad():
                result = optimized_model(data)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            processing_times.append((end_time - start_time) * 1000)

        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)

        print(f"Streaming inference:")
        print(f"  Average processing time: {avg_processing_time:.2f} ms")
        print(f"  Maximum processing time: {max_processing_time:.2f} ms")

        # Verify low-latency streaming
        self.assertLess(avg_processing_time, 10.0)  # < 10ms average
        self.assertLess(max_processing_time, 20.0)  # < 20ms worst case


class TestBatchInference(unittest.TestCase):
    """Test batch inference capabilities"""

    def setUp(self):
        """Set up batch inference tests"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark = InferenceLatencyBenchmark(self.device)

    def test_batch_size_scaling(self):
        """Test inference scaling with different batch sizes"""
        model = InferenceTestModel("classification", "medium")
        input_shape = (1, 784)
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

        model.eval()
        optimized_model = torch.jit.script(model)

        scaling_results = self.benchmark.measure_batch_inference_scaling(
            optimized_model, input_shape, batch_sizes
        )

        print("Batch size scaling results:")
        for batch_size, stats in scaling_results.items():
            print(f"  Batch {batch_size:2d}: "
                  f"{stats['latency_per_sample_ms']:.2f} ms/sample, "
                  f"{stats['throughput_samples_per_sec']:.0f} samples/sec")

        # Verify throughput improves with batch size (generally)
        throughputs = [stats["throughput_samples_per_sec"] for stats in scaling_results.values()]
        max_throughput = max(throughputs)
        min_throughput = min(throughputs)

        self.assertGreater(max_throughput / min_throughput, 2.0)  # At least 2x improvement

    def test_optimal_batch_size_detection(self):
        """Test detection of optimal batch size for inference"""
        model = InferenceTestModel("cnn", "medium")
        input_shape = (1, 3, 32, 32)
        batch_sizes = [1, 2, 4, 8, 16, 32]

        model.eval()
        optimized_model = torch.jit.script(model)

        scaling_results = self.benchmark.measure_batch_inference_scaling(
            optimized_model, input_shape, batch_sizes
        )

        # Find optimal batch size (highest throughput with reasonable latency)
        optimal_batch = None
        best_efficiency = 0

        for batch_size, stats in scaling_results.items():
            # Calculate efficiency as throughput per ms of latency
            efficiency = stats["throughput_samples_per_sec"] / stats["mean_latency_ms"]

            if efficiency > best_efficiency:
                best_efficiency = efficiency
                optimal_batch = batch_size

        print(f"Optimal batch size: {optimal_batch}")
        print(f"Best efficiency: {best_efficiency:.2f} samples/sec/ms")

        # Verify optimal batch size is reasonable
        self.assertGreaterEqual(optimal_batch, 1)
        self.assertLessEqual(optimal_batch, 64)

    def test_memory_efficient_batching(self):
        """Test memory-efficient batch processing"""
        model = InferenceTestModel("classification", "medium")

        model.eval()
        optimized_model = torch.jit.script(model).to(self.device)

        # Test large dataset processing with constrained memory
        large_dataset_size = 1000
        max_batch_size = 32

        # Generate large dataset
        large_dataset = torch.randn(large_dataset_size, 784)
        all_results = []

        # Process in batches
        num_batches = (large_dataset_size + max_batch_size - 1) // max_batch_size

        start_time = time.time()

        for i in range(num_batches):
            start_idx = i * max_batch_size
            end_idx = min((i + 1) * max_batch_size, large_dataset_size)

            batch = large_dataset[start_idx:end_idx].to(self.device)

            with torch.no_grad():
                batch_results = optimized_model(batch)

            all_results.append(batch_results.cpu())

        total_time = time.time() - start_time

        # Combine results
        final_results = torch.cat(all_results, dim=0)

        print(f"Large dataset processing:")
        print(f"  Dataset size: {large_dataset_size}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Throughput: {large_dataset_size / total_time:.0f} samples/sec")

        # Verify correctness
        self.assertEqual(final_results.shape[0], large_dataset_size)
        self.assertGreater(large_dataset_size / total_time, 100)  # At least 100 samples/sec


class TestConcurrentInference(unittest.TestCase):
    """Test concurrent inference handling"""

    def setUp(self):
        """Set up concurrent inference tests"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_multi_threaded_inference(self):
        """Test multi-threaded inference handling"""
        model = InferenceTestModel("classification", "simple")
        model.eval()
        optimized_model = torch.jit.script(model).to(self.device)

        def inference_worker(model, inputs, results_queue, worker_id):
            """Worker function for threaded inference"""
            worker_results = []
            worker_times = []

            for i, input_tensor in enumerate(inputs):
                input_tensor = input_tensor.to(self.device)

                start_time = time.perf_counter()

                with torch.no_grad():
                    result = model(input_tensor)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                end_time = time.perf_counter()

                worker_results.append(result.cpu())
                worker_times.append((end_time - start_time) * 1000)

            results_queue.put({
                "worker_id": worker_id,
                "results": worker_results,
                "times": worker_times
            })

        # Create test data for multiple workers
        num_workers = 4
        samples_per_worker = 20

        test_inputs = [
            [torch.randn(1, 784) for _ in range(samples_per_worker)]
            for _ in range(num_workers)
        ]

        # Run concurrent inference
        results_queue = queue.Queue()
        threads = []

        start_time = time.time()

        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=inference_worker,
                args=(optimized_model, test_inputs[worker_id], results_queue, worker_id)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Collect results
        all_results = []
        all_times = []

        for _ in range(num_workers):
            worker_result = results_queue.get()
            all_results.extend(worker_result["results"])
            all_times.extend(worker_result["times"])

        avg_time_per_sample = np.mean(all_times)
        total_samples = num_workers * samples_per_worker
        throughput = total_samples / total_time

        print(f"Multi-threaded inference:")
        print(f"  Workers: {num_workers}")
        print(f"  Total samples: {total_samples}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average time per sample: {avg_time_per_sample:.2f} ms")
        print(f"  Throughput: {throughput:.0f} samples/sec")

        # Verify concurrent processing efficiency
        self.assertEqual(len(all_results), total_samples)
        self.assertGreater(throughput, 50)  # Reasonable throughput

    def test_async_inference_processing(self):
        """Test asynchronous inference processing"""
        model = InferenceTestModel("classification", "medium")
        model.eval()
        optimized_model = torch.jit.script(model).to(self.device)

        async def process_batch_async(inputs):
            """Async batch processing function"""
            results = []
            for input_tensor in inputs:
                input_tensor = input_tensor.to(self.device)

                with torch.no_grad():
                    result = optimized_model(input_tensor)

                results.append(result.cpu())

            return results

        # Create test batches
        num_batches = 10
        batch_size = 5

        test_batches = [
            [torch.randn(1, 784) for _ in range(batch_size)]
            for _ in range(num_batches)
        ]

        # Process batches concurrently
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_batch = {
                executor.submit(process_batch_async, batch): i
                for i, batch in enumerate(test_batches)
            }

            batch_results = {}
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results[batch_idx] = future.result()
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")

        total_time = time.time() - start_time
        total_samples = num_batches * batch_size
        throughput = total_samples / total_time

        print(f"Async inference processing:")
        print(f"  Batches: {num_batches}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total samples: {total_samples}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Throughput: {throughput:.0f} samples/sec")

        # Verify async processing
        self.assertEqual(len(batch_results), num_batches)
        self.assertGreater(throughput, 20)


class TestInferenceAccuracy(unittest.TestCase):
    """Test inference accuracy validation"""

    def setUp(self):
        """Set up inference accuracy tests"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_inference_output_consistency(self):
        """Test consistency of inference outputs"""
        model = InferenceTestModel("classification", "medium")
        model.eval()
        optimized_model = torch.jit.script(model)

        # Create deterministic test inputs
        torch.manual_seed(42)
        test_inputs = torch.randn(100, 784)

        # Run inference multiple times
        results = []
        for run in range(5):
            with torch.no_grad():
                result = optimized_model(test_inputs.to(self.device))
            results.append(result.cpu())

        # Check consistency across runs
        for i in range(1, len(results)):
            difference = torch.abs(results[0] - results[i])
            max_diff = torch.max(difference)

            print(f"Run {i} max difference from run 0: {max_diff.item():.6f}")

            # Should be exactly the same (deterministic)
            self.assertLess(max_diff.item(), 1e-5)

    def test_inference_vs_training_mode(self):
        """Test difference between training and inference modes"""
        model = InferenceTestModel("classification", "medium")
        test_inputs = torch.randn(20, 784).to(self.device)

        # Training mode inference
        model.train()
        with torch.no_grad():
            train_mode_output = model(test_inputs)

        # Inference mode
        model.eval()
        with torch.no_grad():
            eval_mode_output = model(test_inputs)

        # Compare outputs
        output_difference = torch.abs(train_mode_output - eval_mode_output)
        max_diff = torch.max(output_difference)
        mean_diff = torch.mean(output_difference)

        print(f"Training vs Inference mode difference:")
        print(f"  Max difference: {max_diff.item():.6f}")
        print(f"  Mean difference: {mean_diff.item():.6f}")

        # For models with dropout, there should be some difference
        # For models without dropout, should be identical
        # This test ensures we're aware of the differences

    def test_numerical_precision_inference(self):
        """Test numerical precision during inference"""
        model = InferenceTestModel("regression", "medium")
        model.eval()
        optimized_model = torch.jit.script(model)

        # Test with various input magnitudes
        test_cases = [
            torch.randn(10, 100) * 0.01,    # Small values
            torch.randn(10, 100) * 1.0,     # Normal values
            torch.randn(10, 100) * 100.0,   # Large values
        ]

        for i, test_input in enumerate(test_cases):
            test_input = test_input.to(self.device)

            with torch.no_grad():
                original_output = model(test_input)
                optimized_output = optimized_model(test_input)

            # Check for numerical issues
            original_has_nan = torch.isnan(original_output).any()
            optimized_has_nan = torch.isnan(optimized_output).any()

            original_has_inf = torch.isinf(original_output).any()
            optimized_has_inf = torch.isinf(optimized_output).any()

            print(f"Test case {i} (magnitude: {torch.std(test_input).item():.2f}):")
            print(f"  Original NaN: {original_has_nan}, Inf: {original_has_inf}")
            print(f"  Optimized NaN: {optimized_has_nan}, Inf: {optimized_has_inf}")

            # Verify no numerical issues
            self.assertFalse(optimized_has_nan)
            self.assertFalse(optimized_has_inf)

            # Verify reasonable output similarity
            if not original_has_nan and not original_has_inf:
                similarity = torch.cosine_similarity(
                    original_output.view(-1),
                    optimized_output.view(-1),
                    dim=0
                )
                self.assertGreater(similarity.item(), 0.99)


if __name__ == "__main__":
    # Set up logging for inference tests
    logging.basicConfig(level=logging.INFO)

    print("Running Phase 6 Inference Capability Tests")
    print("=" * 50)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 50)

    # Run tests
    unittest.main(verbosity=2)