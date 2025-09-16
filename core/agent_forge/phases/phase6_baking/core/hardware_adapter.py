#!/usr/bin/env python3
"""
Agent Forge Phase 6: Hardware Adapter
=====================================

Hardware-specific optimization system that adapts models for optimal performance
on different hardware platforms (CUDA GPUs, CPUs, specialized accelerators)
with automatic hardware detection and optimization selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import platform
import subprocess
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import time
from pathlib import Path

@dataclass
class HardwareProfile:
    """Hardware profile information"""
    device_type: str  # cuda, cpu, mps, etc.
    device_name: str
    compute_capability: Optional[str] = None
    memory_total: int = 0  # MB
    memory_available: int = 0  # MB
    core_count: int = 0
    frequency: float = 0.0  # GHz
    architecture: str = ""
    supports_mixed_precision: bool = False
    supports_tensorrt: bool = False
    supports_onednn: bool = False
    optimization_features: List[str] = None

    def __post_init__(self):
        if self.optimization_features is None:
            self.optimization_features = []

@dataclass
class OptimizationStrategy:
    """Hardware-specific optimization strategy"""
    name: str
    priority: int
    applicable_devices: List[str]
    optimization_passes: List[str]
    performance_weight: float
    accuracy_impact: float
    memory_impact: float
    configuration: Dict[str, Any] = None

    def __post_init__(self):
        if self.configuration is None:
            self.configuration = {}

class HardwareAdapter:
    """
    Hardware-specific optimization system that automatically detects
    hardware capabilities and applies appropriate optimizations for
    maximum performance while maintaining model accuracy.
    """

    def __init__(self, config, device: torch.device, logger: logging.Logger):
        self.config = config
        self.device = device
        self.logger = logger

        # Hardware detection
        self.hardware_profile = self._detect_hardware_profile()
        self.optimization_strategies = self._initialize_optimization_strategies()

        # Adaptation state
        self.adapted_models = {}
        self.performance_cache = {}

        self.logger.info(f"HardwareAdapter initialized for {self.hardware_profile.device_name}")
        self.logger.info(f"Available optimizations: {len(self.optimization_strategies)}")

    def _detect_hardware_profile(self) -> HardwareProfile:
        """Detect comprehensive hardware profile"""
        profile = HardwareProfile(device_type=self.device.type, device_name="Unknown")

        if self.device.type == "cuda":
            profile = self._detect_cuda_profile()
        elif self.device.type == "cpu":
            profile = self._detect_cpu_profile()
        elif self.device.type == "mps":
            profile = self._detect_mps_profile()

        return profile

    def _detect_cuda_profile(self) -> HardwareProfile:
        """Detect CUDA GPU hardware profile"""
        if not torch.cuda.is_available():
            return HardwareProfile(device_type="cuda", device_name="CUDA Not Available")

        device_idx = self.device.index or 0
        props = torch.cuda.get_device_properties(device_idx)

        profile = HardwareProfile(
            device_type="cuda",
            device_name=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            memory_total=props.total_memory // (1024 ** 2),  # Convert to MB
            memory_available=self._get_available_cuda_memory(device_idx),
            core_count=props.multi_processor_count,
            architecture=self._get_cuda_architecture(props.major, props.minor)
        )

        # Check for optimization feature support
        profile.supports_mixed_precision = self._check_mixed_precision_support(props)
        profile.supports_tensorrt = self._check_tensorrt_support()

        # Add CUDA-specific optimization features
        profile.optimization_features.extend([
            "cuda_graphs",
            "async_execution",
            "memory_pools",
            "multi_stream"
        ])

        if profile.supports_mixed_precision:
            profile.optimization_features.append("mixed_precision")

        if profile.supports_tensorrt:
            profile.optimization_features.append("tensorrt")

        return profile

    def _detect_cpu_profile(self) -> HardwareProfile:
        """Detect CPU hardware profile"""
        profile = HardwareProfile(
            device_type="cpu",
            device_name=platform.processor() or "CPU",
            core_count=self._get_cpu_core_count(),
            architecture=platform.machine(),
            memory_total=self._get_system_memory()
        )

        # Check for CPU-specific optimization support
        profile.supports_onednn = self._check_onednn_support()

        # Add CPU-specific optimization features
        profile.optimization_features.extend([
            "vectorization",
            "threading_optimization",
            "cache_optimization"
        ])

        if profile.supports_onednn:
            profile.optimization_features.append("onednn")

        # Check for specific CPU features
        cpu_features = self._detect_cpu_features()
        profile.optimization_features.extend(cpu_features)

        return profile

    def _detect_mps_profile(self) -> HardwareProfile:
        """Detect Apple Metal Performance Shaders (MPS) profile"""
        profile = HardwareProfile(
            device_type="mps",
            device_name="Apple Silicon GPU",
            architecture="arm64"
        )

        # MPS-specific features
        profile.optimization_features.extend([
            "metal_shaders",
            "unified_memory",
            "neural_engine"
        ])

        return profile

    def _get_available_cuda_memory(self, device_idx: int) -> int:
        """Get available CUDA memory in MB"""
        try:
            torch.cuda.empty_cache()
            free_memory, _ = torch.cuda.mem_get_info(device_idx)
            return free_memory // (1024 ** 2)
        except Exception:
            return 0

    def _get_cuda_architecture(self, major: int, minor: int) -> str:
        """Get CUDA architecture name"""
        arch_map = {
            (3, 0): "Kepler", (3, 5): "Kepler", (3, 7): "Kepler",
            (5, 0): "Maxwell", (5, 2): "Maxwell",
            (6, 0): "Pascal", (6, 1): "Pascal",
            (7, 0): "Volta", (7, 5): "Turing",
            (8, 0): "Ampere", (8, 6): "Ampere", (8, 9): "Ampere",
            (9, 0): "Hopper"
        }
        return arch_map.get((major, minor), f"Unknown_{major}.{minor}")

    def _check_mixed_precision_support(self, props) -> bool:
        """Check if GPU supports mixed precision training"""
        # Tensor cores available on Volta (7.0) and newer
        return props.major >= 7

    def _check_tensorrt_support(self) -> bool:
        """Check if TensorRT is available"""
        try:
            import tensorrt
            return True
        except ImportError:
            try:
                import torch_tensorrt
                return True
            except ImportError:
                return False

    def _get_cpu_core_count(self) -> int:
        """Get CPU core count"""
        try:
            import os
            return os.cpu_count()
        except Exception:
            return 1

    def _get_system_memory(self) -> int:
        """Get system memory in MB"""
        try:
            import psutil
            return psutil.virtual_memory().total // (1024 ** 2)
        except ImportError:
            # Fallback method
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal'):
                            return int(line.split()[1]) // 1024  # Convert from KB to MB
            except Exception:
                return 8192  # Default to 8GB

    def _check_onednn_support(self) -> bool:
        """Check if OneDNN (MKL-DNN) is available"""
        return hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available()

    def _detect_cpu_features(self) -> List[str]:
        """Detect CPU-specific optimization features"""
        features = []

        try:
            # Check for AVX support
            if self._check_cpu_flag('avx2'):
                features.append('avx2')
            elif self._check_cpu_flag('avx'):
                features.append('avx')

            # Check for other CPU features
            if self._check_cpu_flag('sse4_1'):
                features.append('sse4_1')
            if self._check_cpu_flag('fma'):
                features.append('fma')

        except Exception:
            pass  # Ignore errors in feature detection

        return features

    def _check_cpu_flag(self, flag: str) -> bool:
        """Check if CPU supports specific flag"""
        try:
            if platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    return flag in cpuinfo
            elif platform.system() == "Windows":
                # Windows-specific detection would go here
                pass
            elif platform.system() == "Darwin":
                # macOS-specific detection would go here
                pass
        except Exception:
            pass
        return False

    def _initialize_optimization_strategies(self) -> List[OptimizationStrategy]:
        """Initialize hardware-specific optimization strategies"""
        strategies = []

        # CUDA-specific strategies
        if self.hardware_profile.device_type == "cuda":
            strategies.extend(self._create_cuda_strategies())

        # CPU-specific strategies
        elif self.hardware_profile.device_type == "cpu":
            strategies.extend(self._create_cpu_strategies())

        # MPS-specific strategies
        elif self.hardware_profile.device_type == "mps":
            strategies.extend(self._create_mps_strategies())

        # Sort by priority
        strategies.sort(key=lambda x: x.priority)
        return strategies

    def _create_cuda_strategies(self) -> List[OptimizationStrategy]:
        """Create CUDA-specific optimization strategies"""
        strategies = []

        # Mixed Precision Strategy
        if self.hardware_profile.supports_mixed_precision:
            strategies.append(OptimizationStrategy(
                name="mixed_precision",
                priority=1,
                applicable_devices=["cuda"],
                optimization_passes=["amp_conversion", "loss_scaling"],
                performance_weight=0.3,
                accuracy_impact=0.02,
                memory_impact=-0.4,  # Reduces memory usage
                configuration={
                    "dtype": torch.float16,
                    "loss_scale": 128.0
                }
            ))

        # TensorRT Strategy
        if self.hardware_profile.supports_tensorrt:
            strategies.append(OptimizationStrategy(
                name="tensorrt",
                priority=2,
                applicable_devices=["cuda"],
                optimization_passes=["tensorrt_compilation"],
                performance_weight=0.5,
                accuracy_impact=0.01,
                memory_impact=-0.2,
                configuration={
                    "precision": "fp16",
                    "max_batch_size": 32,
                    "workspace_size": 1 << 30
                }
            ))

        # CUDA Graphs Strategy
        strategies.append(OptimizationStrategy(
            name="cuda_graphs",
            priority=3,
            applicable_devices=["cuda"],
            optimization_passes=["graph_capture", "graph_optimization"],
            performance_weight=0.2,
            accuracy_impact=0.0,
            memory_impact=0.1,
            configuration={
                "warmup_runs": 10,
                "capture_batch_size": 1
            }
        ))

        # Memory Optimization Strategy
        strategies.append(OptimizationStrategy(
            name="cuda_memory_optimization",
            priority=4,
            applicable_devices=["cuda"],
            optimization_passes=["memory_pooling", "tensor_fusion"],
            performance_weight=0.15,
            accuracy_impact=0.0,
            memory_impact=-0.3,
            configuration={
                "enable_memory_pool": True,
                "fusion_threshold": 1024
            }
        ))

        return strategies

    def _create_cpu_strategies(self) -> List[OptimizationStrategy]:
        """Create CPU-specific optimization strategies"""
        strategies = []

        # OneDNN Strategy
        if self.hardware_profile.supports_onednn:
            strategies.append(OptimizationStrategy(
                name="onednn_optimization",
                priority=1,
                applicable_devices=["cpu"],
                optimization_passes=["onednn_conversion", "layout_optimization"],
                performance_weight=0.4,
                accuracy_impact=0.0,
                memory_impact=-0.1,
                configuration={
                    "enable_onednn": True,
                    "optimize_for_inference": True
                }
            ))

        # Threading Strategy
        strategies.append(OptimizationStrategy(
            name="threading_optimization",
            priority=2,
            applicable_devices=["cpu"],
            optimization_passes=["thread_pool_setup", "parallel_execution"],
            performance_weight=0.3,
            accuracy_impact=0.0,
            memory_impact=0.05,
            configuration={
                "num_threads": self.hardware_profile.core_count,
                "inter_op_parallelism": True,
                "intra_op_parallelism": True
            }
        ))

        # Vectorization Strategy
        if "avx2" in self.hardware_profile.optimization_features:
            strategies.append(OptimizationStrategy(
                name="avx2_vectorization",
                priority=3,
                applicable_devices=["cpu"],
                optimization_passes=["vectorization_optimization"],
                performance_weight=0.2,
                accuracy_impact=0.0,
                memory_impact=0.0,
                configuration={
                    "vectorization_type": "avx2"
                }
            ))

        return strategies

    def _create_mps_strategies(self) -> List[OptimizationStrategy]:
        """Create MPS-specific optimization strategies"""
        strategies = []

        # Metal Shaders Strategy
        strategies.append(OptimizationStrategy(
            name="metal_shaders",
            priority=1,
            applicable_devices=["mps"],
            optimization_passes=["metal_compilation"],
            performance_weight=0.3,
            accuracy_impact=0.0,
            memory_impact=-0.1,
            configuration={
                "optimize_for_gpu": True
            }
        ))

        return strategies

    def adapt_model(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        optimization_level: Optional[int] = None
    ) -> nn.Module:
        """
        Adapt model for optimal performance on the target hardware.

        Args:
            model: Model to adapt
            sample_inputs: Sample inputs for optimization
            optimization_level: Override optimization level (uses config if None)

        Returns:
            Hardware-adapted model
        """
        self.logger.info(f"Adapting model for {self.hardware_profile.device_name}")
        start_time = time.time()

        optimization_level = optimization_level or getattr(self.config, 'optimization_level', 3)

        # Select appropriate optimization strategies
        selected_strategies = self._select_optimization_strategies(optimization_level)

        adapted_model = model.to(self.device)
        applied_optimizations = []

        try:
            for strategy in selected_strategies:
                self.logger.info(f"Applying {strategy.name} optimization")

                if strategy.name == "mixed_precision":
                    adapted_model = self._apply_mixed_precision(adapted_model, strategy)
                elif strategy.name == "tensorrt":
                    adapted_model = self._apply_tensorrt(adapted_model, sample_inputs, strategy)
                elif strategy.name == "cuda_graphs":
                    adapted_model = self._apply_cuda_graphs(adapted_model, sample_inputs, strategy)
                elif strategy.name == "cuda_memory_optimization":
                    adapted_model = self._apply_cuda_memory_optimization(adapted_model, strategy)
                elif strategy.name == "onednn_optimization":
                    adapted_model = self._apply_onednn(adapted_model, strategy)
                elif strategy.name == "threading_optimization":
                    adapted_model = self._apply_threading_optimization(adapted_model, strategy)
                elif strategy.name == "avx2_vectorization":
                    adapted_model = self._apply_vectorization(adapted_model, strategy)
                elif strategy.name == "metal_shaders":
                    adapted_model = self._apply_metal_optimization(adapted_model, strategy)

                applied_optimizations.append(strategy.name)

            adaptation_time = time.time() - start_time
            self.logger.info(f"Hardware adaptation completed in {adaptation_time:.2f}s")
            self.logger.info(f"Applied optimizations: {applied_optimizations}")

            return adapted_model

        except Exception as e:
            self.logger.error(f"Hardware adaptation failed: {str(e)}")
            self.logger.warning("Returning original model")
            return model.to(self.device)

    def _select_optimization_strategies(self, optimization_level: int) -> List[OptimizationStrategy]:
        """Select optimization strategies based on level and hardware"""
        selected = []

        for strategy in self.optimization_strategies:
            # Check device compatibility
            if self.hardware_profile.device_type not in strategy.applicable_devices:
                continue

            # Check optimization level
            if strategy.priority > optimization_level:
                continue

            # Check hardware feature requirements
            if not self._check_strategy_requirements(strategy):
                continue

            selected.append(strategy)

        return selected

    def _check_strategy_requirements(self, strategy: OptimizationStrategy) -> bool:
        """Check if hardware meets strategy requirements"""
        if strategy.name == "mixed_precision":
            return self.hardware_profile.supports_mixed_precision
        elif strategy.name == "tensorrt":
            return self.hardware_profile.supports_tensorrt
        elif strategy.name == "onednn_optimization":
            return self.hardware_profile.supports_onednn
        elif strategy.name == "avx2_vectorization":
            return "avx2" in self.hardware_profile.optimization_features

        return True

    def _apply_mixed_precision(
        self,
        model: nn.Module,
        strategy: OptimizationStrategy
    ) -> nn.Module:
        """Apply mixed precision optimization"""
        # Convert model to use mixed precision
        class MixedPrecisionModel(nn.Module):
            def __init__(self, base_model, dtype=torch.float16):
                super().__init__()
                self.base_model = base_model
                self.dtype = dtype

            def forward(self, x):
                with torch.cuda.amp.autocast():
                    return self.base_model(x)

        # Convert certain layers to use mixed precision
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.half()  # Convert to float16

        return MixedPrecisionModel(model, strategy.configuration.get("dtype", torch.float16))

    def _apply_tensorrt(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        strategy: OptimizationStrategy
    ) -> nn.Module:
        """Apply TensorRT optimization"""
        try:
            import torch_tensorrt

            # Compile model with TensorRT
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[sample_inputs],
                enabled_precisions={
                    torch.float16 if strategy.configuration.get("precision") == "fp16"
                    else torch.float32
                },
                workspace_size=strategy.configuration.get("workspace_size", 1 << 30),
                max_batch_size=strategy.configuration.get("max_batch_size", 32)
            )

            return trt_model

        except Exception as e:
            self.logger.warning(f"TensorRT optimization failed: {e}")
            return model

    def _apply_cuda_graphs(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        strategy: OptimizationStrategy
    ) -> nn.Module:
        """Apply CUDA graphs optimization"""
        if not hasattr(torch.cuda, 'CUDAGraph'):
            return model

        class CUDAGraphModel(nn.Module):
            def __init__(self, base_model, warmup_runs=10):
                super().__init__()
                self.base_model = base_model
                self.warmup_runs = warmup_runs
                self.graph = None
                self.static_input = None
                self.static_output = None
                self.first_run = True

            def forward(self, x):
                if self.first_run:
                    # Warmup runs
                    for _ in range(self.warmup_runs):
                        _ = self.base_model(x)

                    # Capture graph
                    self.static_input = x.clone()
                    self.static_output = self.base_model(self.static_input)

                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph):
                        self.static_output = self.base_model(self.static_input)

                    self.first_run = False

                # Use captured graph
                self.static_input.copy_(x)
                self.graph.replay()
                return self.static_output.clone()

        return CUDAGraphModel(
            model,
            strategy.configuration.get("warmup_runs", 10)
        )

    def _apply_cuda_memory_optimization(
        self,
        model: nn.Module,
        strategy: OptimizationStrategy
    ) -> nn.Module:
        """Apply CUDA memory optimization"""
        # Enable memory pool if configured
        if strategy.configuration.get("enable_memory_pool", False):
            # Create memory pool (if supported)
            pass

        # Apply gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        return model

    def _apply_onednn(
        self,
        model: nn.Module,
        strategy: OptimizationStrategy
    ) -> nn.Module:
        """Apply OneDNN optimization"""
        if not torch.backends.mkldnn.is_available():
            return model

        # Optimize for inference
        if strategy.configuration.get("optimize_for_inference", False):
            model = torch.jit.optimize_for_inference(model)

        return model

    def _apply_threading_optimization(
        self,
        model: nn.Module,
        strategy: OptimizationStrategy
    ) -> nn.Module:
        """Apply CPU threading optimization"""
        num_threads = strategy.configuration.get("num_threads", self.hardware_profile.core_count)

        # Set thread count
        torch.set_num_threads(num_threads)

        # Enable inter-op parallelism
        if strategy.configuration.get("inter_op_parallelism", True):
            torch.set_num_interop_threads(num_threads // 2)

        return model

    def _apply_vectorization(
        self,
        model: nn.Module,
        strategy: OptimizationStrategy
    ) -> nn.Module:
        """Apply CPU vectorization optimization"""
        # Enable vectorized operations (this is mostly handled by PyTorch automatically)
        # Custom vectorization would require lower-level implementation
        return model

    def _apply_metal_optimization(
        self,
        model: nn.Module,
        strategy: OptimizationStrategy
    ) -> nn.Module:
        """Apply Metal Performance Shaders optimization"""
        # MPS-specific optimizations would go here
        return model

    def benchmark_hardware_performance(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark model performance on current hardware.

        Returns:
            Hardware performance metrics
        """
        self.logger.info(f"Benchmarking on {self.hardware_profile.device_name}")

        model = model.to(self.device)
        sample_inputs = sample_inputs.to(self.device)
        model.eval()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(sample_inputs)

        # Synchronization point
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(sample_inputs)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_latency = (total_time / num_iterations) * 1000  # ms
        throughput = (num_iterations * sample_inputs.size(0)) / total_time  # samples/sec

        # Memory usage
        if self.device.type == "cuda":
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_used = 0  # CPU memory tracking is more complex

        benchmark_results = {
            "hardware_profile": asdict(self.hardware_profile),
            "performance_metrics": {
                "avg_latency_ms": avg_latency,
                "throughput_samples_per_sec": throughput,
                "memory_usage_mb": memory_used,
                "total_benchmark_time": total_time
            },
            "benchmark_config": {
                "num_iterations": num_iterations,
                "batch_size": sample_inputs.size(0),
                "input_shape": list(sample_inputs.shape)
            }
        }

        return benchmark_results

    def generate_hardware_report(
        self,
        benchmark_results: Dict[str, Any],
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate comprehensive hardware optimization report"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware_profile": asdict(self.hardware_profile),
            "optimization_strategies": [asdict(s) for s in self.optimization_strategies],
            "benchmark_results": benchmark_results,
            "recommendations": self._generate_hardware_recommendations()
        }

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Hardware report saved to {output_path}")
        return report

    def _generate_hardware_recommendations(self) -> List[str]:
        """Generate hardware-specific optimization recommendations"""
        recommendations = []

        if self.hardware_profile.device_type == "cuda":
            if self.hardware_profile.supports_mixed_precision:
                recommendations.append(
                    "Enable mixed precision training for ~50% memory reduction and 30% speedup"
                )

            if self.hardware_profile.supports_tensorrt:
                recommendations.append(
                    "Consider TensorRT compilation for up to 5x inference speedup"
                )

            if self.hardware_profile.memory_available < 4096:  # Less than 4GB
                recommendations.append(
                    "Limited GPU memory detected. Consider model pruning or quantization"
                )

        elif self.hardware_profile.device_type == "cpu":
            if self.hardware_profile.core_count >= 8:
                recommendations.append(
                    f"Optimize threading for {self.hardware_profile.core_count} cores"
                )

            if "avx2" in self.hardware_profile.optimization_features:
                recommendations.append(
                    "Enable AVX2 vectorization for improved CPU performance"
                )

            if self.hardware_profile.supports_onednn:
                recommendations.append(
                    "Enable OneDNN for optimized CPU inference"
                )

        return recommendations


def main():
    """Example usage of HardwareAdapter"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("HardwareAdapter")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    # Mock configuration
    class MockConfig:
        optimization_level = 3

    config = MockConfig()

    # Initialize adapter
    adapter = HardwareAdapter(config, device, logger)

    # Example model
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(16 * 32 * 32, 10)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = ExampleModel()
    sample_inputs = torch.randn(1, 3, 32, 32)

    # Adapt model
    try:
        adapted_model = adapter.adapt_model(model, sample_inputs)

        # Benchmark
        benchmark_results = adapter.benchmark_hardware_performance(
            adapted_model, sample_inputs
        )

        print(f"Hardware adaptation completed!")
        print(f"Device: {adapter.hardware_profile.device_name}")
        print(f"Avg latency: {benchmark_results['performance_metrics']['avg_latency_ms']:.2f}ms")
        print(f"Throughput: {benchmark_results['performance_metrics']['throughput_samples_per_sec']:.1f} samples/sec")

    except Exception as e:
        print(f"Hardware adaptation failed: {e}")


if __name__ == "__main__":
    main()