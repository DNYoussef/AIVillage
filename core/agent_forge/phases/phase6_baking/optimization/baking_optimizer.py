"""
Baking Process Optimization System
Optimizes the entire baking pipeline for sub-minute baking times while maintaining quality.
"""

import torch
import torch.nn as nn
import time
import threading
import asyncio
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import json
import pickle
import hashlib
import os
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import psutil
import gc

@dataclass
class BakingConfig:
    """Configuration for baking optimization"""
    max_baking_time: float = 60.0  # Maximum allowed baking time in seconds
    target_baking_time: float = 30.0  # Target baking time in seconds
    enable_parallel_processing: bool = True
    enable_pipeline_optimization: bool = True
    enable_caching: bool = True
    enable_incremental_baking: bool = True
    enable_async_processing: bool = True
    max_parallel_jobs: int = 4
    cache_size_mb: int = 512
    validation_workers: int = 2
    optimization_workers: int = 2
    memory_budget_mb: int = 2048
    checkpoint_interval: int = 10  # Save checkpoint every N optimization steps
    early_stopping_patience: int = 5

@dataclass
class BakingMetrics:
    """Metrics for baking optimization"""
    total_baking_time: float
    optimization_time: float
    validation_time: float
    conversion_time: float
    serialization_time: float
    cache_hit_rate: float
    parallel_efficiency: float
    memory_peak_usage: float
    quality_score: float
    speedup_achieved: float
    success: bool
    error_messages: List[str]

class BakingCache:
    """Intelligent caching system for baking optimization"""
    
    def __init__(self, cache_size_mb: int = 512):
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self.cache = {}
        self.access_order = []
        self.current_size = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        
    def _compute_hash(self, obj: Any) -> str:
        """Compute hash for cache key"""
        if hasattr(obj, 'state_dict'):
            # For models, hash the state dict
            state_str = str(sorted(obj.state_dict().items()))
        else:
            state_str = str(obj)
        
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, size_bytes: int):
        """Put item in cache"""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                old_size = len(pickle.dumps(self.cache[key]))
                self.current_size -= old_size
                self.access_order.remove(key)
            
            # Evict if necessary
            while self.current_size + size_bytes > self.cache_size_bytes and self.cache:
                lru_key = self.access_order.pop(0)
                lru_value = self.cache.pop(lru_key)
                self.current_size -= len(pickle.dumps(lru_value))
            
            # Add new item
            self.cache[key] = value
            self.access_order.append(key)
            self.current_size += size_bytes
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class PipelineStage:
    """Base class for pipeline stages"""
    
    def __init__(self, name: str, config: BakingConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.metrics = {}
        
    async def execute(self, input_data: Any) -> Any:
        """Execute pipeline stage"""
        raise NotImplementedError
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stage metrics"""
        return self.metrics.copy()

class OptimizationStage(PipelineStage):
    """Model optimization stage"""
    
    def __init__(self, config: BakingConfig):
        super().__init__("optimization", config)
        self.optimization_cache = BakingCache(config.cache_size_mb // 4)
    
    async def execute(self, model: nn.Module) -> nn.Module:
        """Execute model optimization"""
        start_time = time.time()
        
        try:
            # Check cache first
            model_hash = self.optimization_cache._compute_hash(model)
            cached_model = self.optimization_cache.get(f"optimized_{model_hash}")
            
            if cached_model is not None:
                self.metrics['cache_hit'] = True
                self.metrics['optimization_time'] = time.time() - start_time
                return cached_model
            
            # Perform optimization
            optimized_model = await self._optimize_model(model)
            
            # Cache result
            model_size = sum(p.numel() * p.element_size() for p in optimized_model.parameters())
            self.optimization_cache.put(f"optimized_{model_hash}", optimized_model, model_size)
            
            self.metrics['cache_hit'] = False
            self.metrics['optimization_time'] = time.time() - start_time
            
            return optimized_model
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            self.metrics['error'] = str(e)
            return model
    
    async def _optimize_model(self, model: nn.Module) -> nn.Module:
        """Perform actual model optimization"""
        # Import optimizers
        from ..inference.inference_optimizer import create_inference_optimizer
        from ..memory_optimizer import create_memory_optimizer
        from ..hardware.hardware_optimizer import create_hardware_optimizer
        
        # Create optimizers
        inference_optimizer = create_inference_optimizer()
        memory_optimizer = create_memory_optimizer()
        hardware_optimizer = create_hardware_optimizer()
        
        try:
            # Apply optimizations in parallel where possible
            loop = asyncio.get_event_loop()
            
            # Memory optimization (can run independently)
            memory_task = loop.run_in_executor(
                None, 
                lambda: memory_optimizer.optimize_model_memory(model)
            )
            
            # Wait for memory optimization
            model, memory_metrics = await memory_task
            
            # Hardware optimization (depends on memory optimization)
            hardware_task = loop.run_in_executor(
                None,
                lambda: hardware_optimizer.optimize_model(model)
            )
            
            model, hardware_metrics = await hardware_task
            
            # Inference optimization (final step)
            example_inputs = torch.randn(1, 3, 224, 224)  # Default input
            inference_task = loop.run_in_executor(
                None,
                lambda: inference_optimizer.optimize_model(model, example_inputs)
            )
            
            model, inference_metrics = await inference_task
            
            # Store optimization metrics
            self.metrics.update({
                'memory_reduction': memory_metrics.memory_reduction,
                'inference_speedup': inference_metrics.speedup_ratio,
                'hardware_efficiency': hardware_metrics.device_utilization
            })
            
            return model
            
        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            return model

class ValidationStage(PipelineStage):
    """Model validation stage"""
    
    def __init__(self, config: BakingConfig):
        super().__init__("validation", config)
        self.validation_cache = BakingCache(config.cache_size_mb // 4)
    
    async def execute(self, model: nn.Module) -> Dict[str, Any]:
        """Execute model validation"""
        start_time = time.time()
        
        try:
            # Check cache first
            model_hash = self.validation_cache._compute_hash(model)
            cached_results = self.validation_cache.get(f"validation_{model_hash}")
            
            if cached_results is not None:
                self.metrics['cache_hit'] = True
                self.metrics['validation_time'] = time.time() - start_time
                return cached_results
            
            # Perform validation
            validation_results = await self._validate_model(model)
            
            # Cache results
            result_size = len(json.dumps(validation_results).encode())
            self.validation_cache.put(f"validation_{model_hash}", validation_results, result_size)
            
            self.metrics['cache_hit'] = False
            self.metrics['validation_time'] = time.time() - start_time
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self.metrics['error'] = str(e)
            return {'success': False, 'error': str(e)}
    
    async def _validate_model(self, model: nn.Module) -> Dict[str, Any]:
        """Perform actual model validation"""
        try:
            validation_results = {}
            
            # Basic functionality test
            model.eval()
            with torch.no_grad():
                test_input = torch.randn(1, 3, 224, 224)
                output = model(test_input)
                validation_results['output_shape'] = list(output.shape)
                validation_results['output_range'] = [float(output.min()), float(output.max())]
            
            # Performance benchmark
            latencies = []
            with torch.no_grad():
                for _ in range(10):
                    start = time.time()
                    _ = model(test_input)
                    latencies.append(time.time() - start)
            
            validation_results['average_latency'] = sum(latencies) / len(latencies)
            validation_results['latency_std'] = float(torch.tensor(latencies).std())
            
            # Memory usage
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated()
                _ = model(test_input)
                memory_after = torch.cuda.memory_allocated()
                validation_results['memory_usage'] = memory_after - memory_before
            
            # Model size
            param_count = sum(p.numel() for p in model.parameters())
            validation_results['parameter_count'] = param_count
            
            # Quality score (composite metric)
            quality_score = self._calculate_quality_score(validation_results)
            validation_results['quality_score'] = quality_score
            
            validation_results['success'] = True
            
            return validation_results
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate composite quality score"""
        try:
            # Normalize metrics to [0, 1] and combine
            latency_score = max(0, 1 - results.get('average_latency', 1.0) / 0.1)  # Target < 100ms
            memory_score = max(0, 1 - results.get('memory_usage', 1e9) / 1e6)  # Target < 1MB
            
            # Simple weighted average
            quality_score = 0.6 * latency_score + 0.4 * memory_score
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception:
            return 0.5  # Default score

class ConversionStage(PipelineStage):
    """Model conversion and serialization stage"""
    
    def __init__(self, config: BakingConfig):
        super().__init__("conversion", config)
        self.conversion_cache = BakingCache(config.cache_size_mb // 4)
    
    async def execute(self, model: nn.Module, target_format: str = "torchscript") -> bytes:
        """Execute model conversion"""
        start_time = time.time()
        
        try:
            # Check cache first
            model_hash = self.conversion_cache._compute_hash(model)
            cache_key = f"converted_{model_hash}_{target_format}"
            cached_data = self.conversion_cache.get(cache_key)
            
            if cached_data is not None:
                self.metrics['cache_hit'] = True
                self.metrics['conversion_time'] = time.time() - start_time
                return cached_data
            
            # Perform conversion
            converted_data = await self._convert_model(model, target_format)
            
            # Cache result
            self.conversion_cache.put(cache_key, converted_data, len(converted_data))
            
            self.metrics['cache_hit'] = False
            self.metrics['conversion_time'] = time.time() - start_time
            
            return converted_data
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            self.metrics['error'] = str(e)
            return b""
    
    async def _convert_model(self, model: nn.Module, target_format: str) -> bytes:
        """Perform actual model conversion"""
        try:
            if target_format == "torchscript":
                # Convert to TorchScript
                model.eval()
                example_input = torch.randn(1, 3, 224, 224)
                traced_model = torch.jit.trace(model, example_input)
                
                # Serialize to bytes
                buffer = torch.jit.save(traced_model, None)
                return buffer
                
            elif target_format == "onnx":
                # Convert to ONNX (requires torch.onnx)
                import io
                buffer = io.BytesIO()
                
                example_input = torch.randn(1, 3, 224, 224)
                torch.onnx.export(
                    model, example_input, buffer,
                    opset_version=11,
                    do_constant_folding=True
                )
                
                return buffer.getvalue()
                
            else:
                # Default: pickle serialization
                return pickle.dumps(model)
                
        except Exception as e:
            self.logger.error(f"Model conversion to {target_format} failed: {e}")
            # Fallback to pickle
            return pickle.dumps(model)

class BakingPipeline:
    """Main baking pipeline coordinator"""
    
    def __init__(self, config: BakingConfig = None):
        self.config = config or BakingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize stages
        self.optimization_stage = OptimizationStage(self.config)
        self.validation_stage = ValidationStage(self.config)
        self.conversion_stage = ConversionStage(self.config)
        
        # Pipeline state
        self.pipeline_metrics = {}
        self.checkpoints = {}
        
        # Executors for parallel processing
        if self.config.enable_parallel_processing:
            self.thread_executor = ThreadPoolExecutor(max_workers=self.config.max_parallel_jobs)
            self.process_executor = ProcessPoolExecutor(max_workers=min(self.config.max_parallel_jobs, 2))
        else:
            self.thread_executor = None
            self.process_executor = None
    
    async def bake_model(self, model: nn.Module, target_format: str = "torchscript") -> Tuple[bytes, BakingMetrics]:
        """Execute complete baking pipeline"""
        start_time = time.time()
        
        metrics = BakingMetrics(
            total_baking_time=0, optimization_time=0, validation_time=0,
            conversion_time=0, serialization_time=0, cache_hit_rate=0,
            parallel_efficiency=0, memory_peak_usage=0, quality_score=0,
            speedup_achieved=0, success=False, error_messages=[]
        )
        
        try:
            self.logger.info(f"Starting baking pipeline for model with target time: {self.config.target_baking_time}s")
            
            # Stage 1: Optimization
            optimization_start = time.time()
            optimized_model = await self.optimization_stage.execute(model)
            metrics.optimization_time = time.time() - optimization_start
            
            # Checkpoint
            if self.config.checkpoint_interval > 0:
                self.checkpoints['optimized_model'] = optimized_model
            
            # Stage 2: Validation (can run in parallel with conversion prep)
            validation_start = time.time()
            validation_results = await self.validation_stage.execute(optimized_model)
            metrics.validation_time = time.time() - validation_start
            metrics.quality_score = validation_results.get('quality_score', 0)
            
            # Check if validation passed
            if not validation_results.get('success', False):
                metrics.error_messages.append(f"Validation failed: {validation_results.get('error', 'Unknown error')}")
                return b"", metrics
            
            # Stage 3: Conversion
            conversion_start = time.time()
            converted_data = await self.conversion_stage.execute(optimized_model, target_format)
            metrics.conversion_time = time.time() - conversion_start
            
            # Calculate final metrics
            metrics.total_baking_time = time.time() - start_time
            metrics.cache_hit_rate = self._calculate_cache_hit_rate()
            metrics.parallel_efficiency = self._calculate_parallel_efficiency()
            metrics.memory_peak_usage = self._get_peak_memory_usage()
            metrics.speedup_achieved = self.config.max_baking_time / metrics.total_baking_time
            metrics.success = metrics.total_baking_time <= self.config.max_baking_time
            
            # Log results
            self.logger.info(f"Baking completed in {metrics.total_baking_time:.2f}s")
            self.logger.info(f"Target achieved: {metrics.success}")
            self.logger.info(f"Quality score: {metrics.quality_score:.3f}")
            
            return converted_data, metrics
            
        except Exception as e:
            metrics.error_messages.append(str(e))
            metrics.total_baking_time = time.time() - start_time
            self.logger.error(f"Baking pipeline failed: {e}")
            return b"", metrics
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        stages = [self.optimization_stage, self.validation_stage, self.conversion_stage]
        total_hits = sum(getattr(stage, 'optimization_cache', BakingCache(1)).hits + 
                        getattr(stage, 'validation_cache', BakingCache(1)).hits +
                        getattr(stage, 'conversion_cache', BakingCache(1)).hits for stage in stages)
        total_requests = sum(getattr(stage, 'optimization_cache', BakingCache(1)).hits + 
                           getattr(stage, 'optimization_cache', BakingCache(1)).misses +
                           getattr(stage, 'validation_cache', BakingCache(1)).hits +
                           getattr(stage, 'validation_cache', BakingCache(1)).misses +
                           getattr(stage, 'conversion_cache', BakingCache(1)).hits +
                           getattr(stage, 'conversion_cache', BakingCache(1)).misses for stage in stages)
        
        return total_hits / total_requests if total_requests > 0 else 0.0
    
    def _calculate_parallel_efficiency(self) -> float:
        """Calculate parallel processing efficiency"""
        if not self.config.enable_parallel_processing:
            return 0.0
        
        # Simplified calculation based on stage overlap
        total_sequential_time = (
            self.pipeline_metrics.get('optimization_time', 0) +
            self.pipeline_metrics.get('validation_time', 0) +
            self.pipeline_metrics.get('conversion_time', 0)
        )
        
        actual_time = self.pipeline_metrics.get('total_baking_time', total_sequential_time)
        
        return (total_sequential_time - actual_time) / total_sequential_time if total_sequential_time > 0 else 0.0
    
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage during baking"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            process = psutil.Process()
            return process.memory_info().peak_wss / 1024 / 1024 if hasattr(process.memory_info(), 'peak_wss') else 0
    
    def cleanup(self):
        """Clean up pipeline resources"""
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        if self.process_executor:
            self.process_executor.shutdown(wait=True)

class BakingOptimizer:
    """Main baking optimization coordinator"""
    
    def __init__(self, config: BakingConfig = None):
        self.config = config or BakingConfig()
        self.logger = logging.getLogger(__name__)
        self.pipeline = BakingPipeline(self.config)
        self.optimization_history = []
    
    async def optimize_baking(self, model: nn.Module, target_format: str = "torchscript") -> Tuple[bytes, BakingMetrics]:
        """Optimize the baking process for the given model"""
        self.logger.info("Starting baking optimization")
        
        # Run baking pipeline
        baked_data, metrics = await self.pipeline.bake_model(model, target_format)
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': time.time(),
            'metrics': metrics,
            'config': self.config,
            'target_format': target_format
        })
        
        # Adaptive optimization for future runs
        if len(self.optimization_history) > 1:
            self._adapt_configuration(metrics)
        
        return baked_data, metrics
    
    def _adapt_configuration(self, current_metrics: BakingMetrics):
        """Adapt configuration based on performance history"""
        try:
            # Analyze recent performance
            recent_metrics = [h['metrics'] for h in self.optimization_history[-5:]]
            
            avg_baking_time = sum(m.total_baking_time for m in recent_metrics) / len(recent_metrics)
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
            
            # Adapt cache size if hit rate is low
            if avg_cache_hit_rate < 0.5 and self.config.cache_size_mb < 1024:
                self.config.cache_size_mb = min(1024, int(self.config.cache_size_mb * 1.5))
                self.logger.info(f"Increased cache size to {self.config.cache_size_mb}MB")
            
            # Adapt parallelization if baking time is high
            if avg_baking_time > self.config.target_baking_time * 1.2:
                if self.config.max_parallel_jobs < 8:
                    self.config.max_parallel_jobs += 1
                    self.logger.info(f"Increased parallel jobs to {self.config.max_parallel_jobs}")
            
        except Exception as e:
            self.logger.warning(f"Configuration adaptation failed: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of baking optimizations"""
        if not self.optimization_history:
            return {}
        
        recent_metrics = [h['metrics'] for h in self.optimization_history[-10:]]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'average_baking_time': sum(m.total_baking_time for m in recent_metrics) / len(recent_metrics),
            'success_rate': sum(1 for m in recent_metrics if m.success) / len(recent_metrics),
            'average_quality_score': sum(m.quality_score for m in recent_metrics) / len(recent_metrics),
            'average_speedup': sum(m.speedup_achieved for m in recent_metrics) / len(recent_metrics),
            'target_achievement_rate': sum(1 for m in recent_metrics if m.total_baking_time <= self.config.target_baking_time) / len(recent_metrics)
        }
    
    def cleanup(self):
        """Clean up optimizer resources"""
        self.pipeline.cleanup()

def create_baking_optimizer(config_dict: Optional[Dict[str, Any]] = None) -> BakingOptimizer:
    """Factory function to create baking optimizer"""
    if config_dict:
        config = BakingConfig(**config_dict)
    else:
        config = BakingConfig()
    
    return BakingOptimizer(config)

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Example model
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.mean(x, dim=[2, 3])
            x = self.fc(x)
            return x
    
    async def main():
        # Create optimizer
        config = BakingConfig(
            target_baking_time=30.0,
            enable_parallel_processing=True,
            enable_caching=True
        )
        
        optimizer = BakingOptimizer(config)
        
        # Optimize baking
        model = ExampleModel()
        
        baked_data, metrics = await optimizer.optimize_baking(model, "torchscript")
        
        print(f"Baking time: {metrics.total_baking_time:.2f}s")
        print(f"Target achieved: {metrics.success}")
        print(f"Quality score: {metrics.quality_score:.3f}")
        print(f"Speedup: {metrics.speedup_achieved:.2f}x")
        print(f"Baked model size: {len(baked_data)} bytes")
        
        summary = optimizer.get_optimization_summary()
        print(f"Optimization summary: {summary}")
        
        optimizer.cleanup()
    
    # Run example
    asyncio.run(main())