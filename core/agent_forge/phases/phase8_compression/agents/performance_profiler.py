import torch
import torch.nn as nn
import torch.profiler
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import time
import psutil
import threading
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import os
from collections import defaultdict

@dataclass
class ProfilingConfig:
    """Configuration for performance profiling."""
    profile_cpu: bool = True
    profile_gpu: bool = True
    profile_memory: bool = True
    profile_detailed: bool = True
    warmup_iterations: int = 10
    profiling_iterations: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64])
    enable_autograd: bool = False
    export_traces: bool = True
    
@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    inference_metrics: Dict[str, Any]
    memory_metrics: Dict[str, Any]
    throughput_metrics: Dict[str, Any]
    latency_metrics: Dict[str, Any]
    resource_utilization: Dict[str, Any]
    bottleneck_analysis: Dict[str, Any]
    optimization_suggestions: List[str]
    
class ProfilerBase(ABC):
    """Abstract base class for profilers."""
    
    @abstractmethod
    def profile(self, model: nn.Module, 
               sample_input: torch.Tensor,
               config: ProfilingConfig) -> Dict[str, Any]:
        pass
        
class InferenceProfiler(ProfilerBase):
    """Profile inference performance."""
    
    def __init__(self, device: torch.device, logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def profile(self, model: nn.Module, 
               sample_input: torch.Tensor,
               config: ProfilingConfig) -> Dict[str, Any]:
        """Profile inference performance."""
        try:
            model.to(self.device)
            model.eval()
            sample_input = sample_input.to(self.device)
            
            results = {}
            
            # Basic inference timing
            timing_results = self._profile_inference_timing(model, sample_input, config)
            results['timing'] = timing_results
            
            # Batch size analysis
            batch_analysis = self._profile_batch_sizes(model, sample_input, config)
            results['batch_analysis'] = batch_analysis
            
            # Layer-wise profiling
            if config.profile_detailed:
                layer_profiling = self._profile_layers(model, sample_input, config)
                results['layer_profiling'] = layer_profiling
                
            # PyTorch profiler integration
            if config.export_traces:
                torch_profiling = self._pytorch_profiler_analysis(model, sample_input, config)
                results['torch_profiling'] = torch_profiling
                
            return results
            
        except Exception as e:
            self.logger.error(f"Inference profiling failed: {e}")
            raise
            
    def _profile_inference_timing(self, model: nn.Module, 
                                 sample_input: torch.Tensor,
                                 config: ProfilingConfig) -> Dict[str, Any]:
        """Profile basic inference timing."""
        # Warmup
        with torch.no_grad():
            for _ in range(config.warmup_iterations):
                model(sample_input)
                
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        # Measure inference times
        inference_times = []
        
        with torch.no_grad():
            for _ in range(config.profiling_iterations):
                start_time = time.perf_counter()
                
                output = model(sample_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms
                
        return {
            'mean_time_ms': float(np.mean(inference_times)),
            'std_time_ms': float(np.std(inference_times)),
            'min_time_ms': float(np.min(inference_times)),
            'max_time_ms': float(np.max(inference_times)),
            'p50_time_ms': float(np.percentile(inference_times, 50)),
            'p95_time_ms': float(np.percentile(inference_times, 95)),
            'p99_time_ms': float(np.percentile(inference_times, 99)),
            'all_times': inference_times
        }
        
    def _profile_batch_sizes(self, model: nn.Module, 
                           sample_input: torch.Tensor,
                           config: ProfilingConfig) -> Dict[str, Any]:
        """Profile performance across different batch sizes."""
        batch_results = {}
        
        for batch_size in config.batch_sizes:
            try:
                # Create batch
                if batch_size <= sample_input.size(0):
                    batch_input = sample_input[:batch_size]
                else:
                    # Repeat samples
                    repeats = (batch_size + sample_input.size(0) - 1) // sample_input.size(0)
                    expanded_input = sample_input.repeat(repeats, 1, 1, 1)
                    batch_input = expanded_input[:batch_size]
                    
                batch_input = batch_input.to(self.device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        model(batch_input)
                        
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                # Measure
                times = []
                with torch.no_grad():
                    for _ in range(20):
                        start_time = time.perf_counter()
                        model(batch_input)
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        end_time = time.perf_counter()
                        
                        batch_time = (end_time - start_time) * 1000
                        per_sample_time = batch_time / batch_size
                        times.append(per_sample_time)
                        
                batch_results[f'batch_{batch_size}'] = {
                    'mean_time_per_sample_ms': float(np.mean(times)),
                    'throughput_samples_per_sec': 1000.0 / np.mean(times),
                    'total_batch_time_ms': float(np.mean(times)) * batch_size,
                    'memory_usage_mb': self._get_memory_usage()
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to profile batch size {batch_size}: {e}")
                batch_results[f'batch_{batch_size}'] = {'error': str(e)}
                
        return batch_results
        
    def _profile_layers(self, model: nn.Module, 
                       sample_input: torch.Tensor,
                       config: ProfilingConfig) -> Dict[str, Any]:
        """Profile individual layers."""
        layer_times = {}
        layer_memory = {}
        
        def create_hook(name):
            def hook(module, input, output):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                if hasattr(hook, 'start_time'):
                    layer_time = (end_time - hook.start_time) * 1000
                    if name not in layer_times:
                        layer_times[name] = []
                    layer_times[name].append(layer_time)
                    
                    # Memory usage
                    memory_usage = self._get_memory_usage()
                    if name not in layer_memory:
                        layer_memory[name] = []
                    layer_memory[name].append(memory_usage)
                    
            def pre_hook(module, input):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                hook.start_time = time.perf_counter()
                
            return hook, pre_hook
            
        # Register hooks
        hooks = []
        pre_hooks = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook, pre_hook = create_hook(name)
                hooks.append(module.register_forward_hook(hook))
                pre_hooks.append(module.register_forward_pre_hook(pre_hook))
                
        # Run profiling
        with torch.no_grad():
            for _ in range(10):
                model(sample_input)
                
        # Clean up hooks
        for hook in hooks + pre_hooks:
            hook.remove()
            
        # Process results
        layer_results = {}
        for name in layer_times:
            if layer_times[name]:
                layer_results[name] = {
                    'mean_time_ms': float(np.mean(layer_times[name])),
                    'std_time_ms': float(np.std(layer_times[name])),
                    'mean_memory_mb': float(np.mean(layer_memory[name])) if layer_memory[name] else 0.0,
                    'parameter_count': sum(p.numel() for p in dict(model.named_modules())[name].parameters())
                }
                
        return layer_results
        
    def _pytorch_profiler_analysis(self, model: nn.Module, 
                                  sample_input: torch.Tensor,
                                  config: ProfilingConfig) -> Dict[str, Any]:
        """Use PyTorch profiler for detailed analysis."""
        try:
            # Configure profiler
            profiler_config = {
                'activities': [torch.profiler.ProfilerActivity.CPU],
                'record_shapes': True,
                'profile_memory': config.profile_memory,
                'with_stack': True,
                'with_flops': True
            }
            
            if config.profile_gpu and self.device.type == 'cuda':
                profiler_config['activities'].append(torch.profiler.ProfilerActivity.CUDA)
                
            # Run profiler
            with torch.profiler.profile(**profiler_config) as prof:
                with torch.no_grad():
                    for _ in range(10):
                        model(sample_input)
                        
            # Extract key metrics
            events = prof.key_averages()
            
            # Top operations by time
            top_ops = []
            for event in events[:10]:
                top_ops.append({
                    'name': event.key,
                    'cpu_time_ms': event.cpu_time_total / 1000,
                    'cuda_time_ms': event.cuda_time_total / 1000 if hasattr(event, 'cuda_time_total') else 0,
                    'count': event.count,
                    'input_shapes': str(event.input_shapes) if hasattr(event, 'input_shapes') else 'N/A'
                })
                
            # Total times
            total_cpu_time = sum(event.cpu_time_total for event in events) / 1000
            total_cuda_time = sum(getattr(event, 'cuda_time_total', 0) for event in events) / 1000
            
            return {
                'top_operations': top_ops,
                'total_cpu_time_ms': total_cpu_time,
                'total_cuda_time_ms': total_cuda_time,
                'profiler_trace': 'Available via prof.export_chrome_trace()'
            }
            
        except Exception as e:
            self.logger.warning(f"PyTorch profiler failed: {e}")
            return {'error': str(e)}
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
            
class MemoryProfiler(ProfilerBase):
    """Profile memory usage patterns."""
    
    def __init__(self, device: torch.device, logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def profile(self, model: nn.Module, 
               sample_input: torch.Tensor,
               config: ProfilingConfig) -> Dict[str, Any]:
        """Profile memory usage."""
        try:
            model.to(self.device)
            sample_input = sample_input.to(self.device)
            
            results = {}
            
            # Basic memory profiling
            basic_memory = self._profile_basic_memory(model, sample_input)
            results['basic_memory'] = basic_memory
            
            # Memory over time
            memory_timeline = self._profile_memory_timeline(model, sample_input)
            results['memory_timeline'] = memory_timeline
            
            # Peak memory analysis
            peak_memory = self._profile_peak_memory(model, sample_input, config)
            results['peak_memory'] = peak_memory
            
            # Memory fragmentation
            if self.device.type == 'cuda':
                fragmentation = self._profile_memory_fragmentation(model, sample_input)
                results['fragmentation'] = fragmentation
                
            return results
            
        except Exception as e:
            self.logger.error(f"Memory profiling failed: {e}")
            raise
            
    def _profile_basic_memory(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profile basic memory usage."""
        # Model memory
        model_memory = 0
        for param in model.parameters():
            model_memory += param.numel() * param.element_size()
        for buffer in model.buffers():
            buffer_memory += buffer.numel() * buffer.element_size()
            
        model_memory_mb = model_memory / (1024 * 1024)
        
        # Input memory
        input_memory_mb = (sample_input.numel() * sample_input.element_size()) / (1024 * 1024)
        
        # Activation memory (estimate)
        model.eval()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        with torch.no_grad():
            output = model(sample_input)
            
        if self.device.type == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            current_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            process = psutil.Process()
            current_memory_mb = process.memory_info().rss / (1024 * 1024)
            peak_memory_mb = current_memory_mb
            
        output_memory_mb = 0
        if hasattr(output, 'numel'):
            output_memory_mb = (output.numel() * output.element_size()) / (1024 * 1024)
        elif isinstance(output, (list, tuple)):
            for tensor in output:
                if hasattr(tensor, 'numel'):
                    output_memory_mb += (tensor.numel() * tensor.element_size()) / (1024 * 1024)
                    
        activation_memory_mb = peak_memory_mb - model_memory_mb - input_memory_mb
        
        return {
            'model_memory_mb': model_memory_mb,
            'input_memory_mb': input_memory_mb,
            'output_memory_mb': output_memory_mb,
            'activation_memory_mb': max(0, activation_memory_mb),
            'peak_memory_mb': peak_memory_mb,
            'current_memory_mb': current_memory_mb
        }
        
    def _profile_memory_timeline(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profile memory usage over time."""
        memory_timeline = []
        
        def memory_hook(name):
            def hook(module, input, output):
                current_memory = self._get_current_memory()
                memory_timeline.append({
                    'layer': name,
                    'memory_mb': current_memory,
                    'timestamp': time.time()
                })
            return hook
            
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hook = module.register_forward_hook(memory_hook(name))
                hooks.append(hook)
                
        # Record initial memory
        start_memory = self._get_current_memory()
        start_time = time.time()
        
        # Run inference
        model.eval()
        with torch.no_grad():
            model(sample_input)
            
        # Clean up
        for hook in hooks:
            hook.remove()
            
        # Process timeline
        if memory_timeline:
            base_time = memory_timeline[0]['timestamp']
            for entry in memory_timeline:
                entry['relative_time_ms'] = (entry['timestamp'] - base_time) * 1000
                
            max_memory = max(entry['memory_mb'] for entry in memory_timeline)
            memory_growth = max_memory - start_memory
            
        return {
            'timeline': memory_timeline,
            'start_memory_mb': start_memory,
            'max_memory_mb': max_memory if memory_timeline else start_memory,
            'memory_growth_mb': memory_growth if memory_timeline else 0
        }
        
    def _profile_peak_memory(self, model: nn.Module, 
                           sample_input: torch.Tensor,
                           config: ProfilingConfig) -> Dict[str, Any]:
        """Profile peak memory usage across different batch sizes."""
        peak_memory_results = {}
        
        for batch_size in config.batch_sizes:
            try:
                # Create batch
                if batch_size <= sample_input.size(0):
                    batch_input = sample_input[:batch_size]
                else:
                    repeats = (batch_size + sample_input.size(0) - 1) // sample_input.size(0)
                    expanded_input = sample_input.repeat(repeats, 1, 1, 1)
                    batch_input = expanded_input[:batch_size]
                    
                batch_input = batch_input.to(self.device)
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    
                # Run inference
                model.eval()
                with torch.no_grad():
                    output = model(batch_input)
                    
                if self.device.type == 'cuda':
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                else:
                    process = psutil.Process()
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    peak_memory = current_memory
                    
                peak_memory_results[f'batch_{batch_size}'] = {
                    'peak_memory_mb': peak_memory,
                    'current_memory_mb': current_memory,
                    'memory_per_sample_mb': peak_memory / batch_size
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to profile memory for batch {batch_size}: {e}")
                peak_memory_results[f'batch_{batch_size}'] = {'error': str(e)}
                
        return peak_memory_results
        
    def _profile_memory_fragmentation(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profile memory fragmentation (CUDA only)."""
        if self.device.type != 'cuda':
            return {'error': 'Fragmentation profiling only available for CUDA'}
            
        try:
            # Get memory stats before
            torch.cuda.empty_cache()
            memory_stats_before = torch.cuda.memory_stats()
            
            # Run multiple inferences
            model.eval()
            with torch.no_grad():
                for _ in range(10):
                    output = model(sample_input)
                    del output
                    
            # Get memory stats after
            memory_stats_after = torch.cuda.memory_stats()
            
            # Calculate fragmentation metrics
            allocated_bytes = memory_stats_after['allocated_bytes.all.current']
            reserved_bytes = memory_stats_after['reserved_bytes.all.current']
            
            fragmentation_ratio = (reserved_bytes - allocated_bytes) / reserved_bytes if reserved_bytes > 0 else 0
            
            return {
                'allocated_mb': allocated_bytes / (1024 * 1024),
                'reserved_mb': reserved_bytes / (1024 * 1024),
                'fragmentation_ratio': fragmentation_ratio,
                'num_alloc_retries': memory_stats_after['num_alloc_retries'],
                'num_ooms': memory_stats_after['num_ooms']
            }
            
        except Exception as e:
            return {'error': str(e)}
            
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
            
class ResourceUtilizationProfiler(ProfilerBase):
    """Profile CPU and GPU utilization."""
    
    def __init__(self, device: torch.device, logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
    def profile(self, model: nn.Module, 
               sample_input: torch.Tensor,
               config: ProfilingConfig) -> Dict[str, Any]:
        """Profile resource utilization."""
        try:
            model.to(self.device)
            sample_input = sample_input.to(self.device)
            
            results = {}
            
            # CPU utilization
            if config.profile_cpu:
                cpu_metrics = self._profile_cpu_utilization(model, sample_input)
                results['cpu'] = cpu_metrics
                
            # GPU utilization
            if config.profile_gpu and self.device.type == 'cuda':
                gpu_metrics = self._profile_gpu_utilization(model, sample_input)
                results['gpu'] = gpu_metrics
                
            # System resources
            system_metrics = self._profile_system_resources(model, sample_input)
            results['system'] = system_metrics
            
            return results
            
        except Exception as e:
            self.logger.error(f"Resource utilization profiling failed: {e}")
            raise
            
    def _profile_cpu_utilization(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profile CPU utilization."""
        cpu_percentages = []
        memory_usage = []
        
        def monitor_resources():
            for _ in range(50):  # Monitor for 5 seconds at 0.1s intervals
                cpu_percentages.append(psutil.cpu_percent())
                memory_usage.append(psutil.virtual_memory().percent)
                time.sleep(0.1)
                
        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Run inference
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(20):
                model(sample_input)
                
        end_time = time.time()
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        return {
            'avg_cpu_percent': float(np.mean(cpu_percentages)) if cpu_percentages else 0.0,
            'max_cpu_percent': float(np.max(cpu_percentages)) if cpu_percentages else 0.0,
            'avg_memory_percent': float(np.mean(memory_usage)) if memory_usage else 0.0,
            'cpu_cores': psutil.cpu_count(),
            'inference_duration_sec': end_time - start_time
        }
        
    def _profile_gpu_utilization(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profile GPU utilization."""
        try:
            import pynvml
            
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            gpu_percentages = []
            memory_percentages = []
            temperatures = []
            
            def monitor_gpu():
                for _ in range(50):
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        
                        gpu_percentages.append(util.gpu)
                        memory_percentages.append((memory_info.used / memory_info.total) * 100)
                        temperatures.append(temp)
                        
                    except Exception:
                        pass
                        
                    time.sleep(0.1)
                    
            # Start monitoring
            monitor_thread = threading.Thread(target=monitor_gpu)
            monitor_thread.start()
            
            # Run inference
            model.eval()
            with torch.no_grad():
                for _ in range(20):
                    model(sample_input)
                    torch.cuda.synchronize()
                    
            monitor_thread.join()
            
            # Get device info
            device_name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                'device_name': device_name.decode('utf-8') if isinstance(device_name, bytes) else str(device_name),
                'avg_gpu_percent': float(np.mean(gpu_percentages)) if gpu_percentages else 0.0,
                'max_gpu_percent': float(np.max(gpu_percentages)) if gpu_percentages else 0.0,
                'avg_memory_percent': float(np.mean(memory_percentages)) if memory_percentages else 0.0,
                'max_memory_percent': float(np.max(memory_percentages)) if memory_percentages else 0.0,
                'avg_temperature_c': float(np.mean(temperatures)) if temperatures else 0.0,
                'max_temperature_c': float(np.max(temperatures)) if temperatures else 0.0,
                'total_memory_gb': memory_info.total / (1024**3)
            }
            
        except ImportError:
            self.logger.warning("pynvml not available for GPU monitoring")
            return {'error': 'pynvml not available'}
        except Exception as e:
            return {'error': str(e)}
            
    def _profile_system_resources(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profile overall system resources."""
        # Get system info
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': psutil.uname().system,
            'architecture': psutil.uname().machine
        }
        
        # Monitor during inference
        process = psutil.Process()
        
        # Before inference
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # Run inference
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                model(sample_input)
                
        end_time = time.time()
        
        # After inference
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / (1024**2)  # MB
        
        system_info.update({
            'process_cpu_percent': cpu_after,
            'process_memory_mb': memory_after,
            'memory_increase_mb': memory_after - memory_before,
            'inference_time_sec': end_time - start_time
        })
        
        return system_info
        
class PerformanceProfiler:
    """Main performance profiling agent."""
    
    def __init__(self, device: torch.device = torch.device('cpu'), logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.profilers = {
            'inference': InferenceProfiler(device, logger),
            'memory': MemoryProfiler(device, logger),
            'resources': ResourceUtilizationProfiler(device, logger)
        }
        
    def comprehensive_profile(self, model: nn.Module,
                            sample_input: torch.Tensor,
                            config: ProfilingConfig) -> PerformanceMetrics:
        """Run comprehensive performance profiling."""
        try:
            results = {}
            
            # Run all profilers
            for profiler_name, profiler in self.profilers.items():
                self.logger.info(f"Running {profiler_name} profiler")
                profiler_results = profiler.profile(model, sample_input, config)
                results[profiler_name] = profiler_results
                
            # Analyze results
            metrics = self._analyze_profiling_results(results)
            
            self.logger.info("Performance profiling completed")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Comprehensive profiling failed: {e}")
            raise
            
    def _analyze_profiling_results(self, results: Dict[str, Any]) -> PerformanceMetrics:
        """Analyze profiling results and generate metrics."""
        # Extract key metrics
        inference_metrics = self._extract_inference_metrics(results.get('inference', {}))
        memory_metrics = self._extract_memory_metrics(results.get('memory', {}))
        resource_metrics = self._extract_resource_metrics(results.get('resources', {}))
        
        # Calculate derived metrics
        throughput_metrics = self._calculate_throughput_metrics(inference_metrics)
        latency_metrics = self._calculate_latency_metrics(inference_metrics)
        
        # Bottleneck analysis
        bottleneck_analysis = self._analyze_bottlenecks(results)
        
        # Optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(results)
        
        return PerformanceMetrics(
            inference_metrics=inference_metrics,
            memory_metrics=memory_metrics,
            throughput_metrics=throughput_metrics,
            latency_metrics=latency_metrics,
            resource_utilization=resource_metrics,
            bottleneck_analysis=bottleneck_analysis,
            optimization_suggestions=optimization_suggestions
        )
        
    def _extract_inference_metrics(self, inference_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract inference metrics."""
        timing = inference_results.get('timing', {})
        batch_analysis = inference_results.get('batch_analysis', {})
        
        metrics = {
            'mean_inference_time_ms': timing.get('mean_time_ms', 0.0),
            'p95_inference_time_ms': timing.get('p95_time_ms', 0.0),
            'inference_stability': timing.get('std_time_ms', 0.0) / max(timing.get('mean_time_ms', 1.0), 0.001)
        }
        
        # Add batch analysis
        if batch_analysis:
            optimal_batch = min(batch_analysis.keys(), 
                              key=lambda k: batch_analysis[k].get('mean_time_per_sample_ms', float('inf')))
            metrics['optimal_batch_size'] = int(optimal_batch.split('_')[1]) if '_' in optimal_batch else 1
            
        return metrics
        
    def _extract_memory_metrics(self, memory_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract memory metrics."""
        basic_memory = memory_results.get('basic_memory', {})
        peak_memory = memory_results.get('peak_memory', {})
        
        metrics = {
            'model_memory_mb': basic_memory.get('model_memory_mb', 0.0),
            'peak_memory_mb': basic_memory.get('peak_memory_mb', 0.0),
            'activation_memory_mb': basic_memory.get('activation_memory_mb', 0.0),
            'memory_efficiency': basic_memory.get('model_memory_mb', 0.0) / max(basic_memory.get('peak_memory_mb', 1.0), 0.001)
        }
        
        # Memory scaling with batch size
        if peak_memory:
            batch_memories = []
            for key, value in peak_memory.items():
                if 'batch_' in key and 'memory_per_sample_mb' in value:
                    batch_memories.append(value['memory_per_sample_mb'])
                    
            if batch_memories:
                metrics['memory_per_sample_mb'] = float(np.mean(batch_memories))
                
        return metrics
        
    def _extract_resource_metrics(self, resource_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract resource utilization metrics."""
        cpu_metrics = resource_results.get('cpu', {})
        gpu_metrics = resource_results.get('gpu', {})
        system_metrics = resource_results.get('system', {})
        
        return {
            'cpu_utilization': cpu_metrics.get('avg_cpu_percent', 0.0),
            'gpu_utilization': gpu_metrics.get('avg_gpu_percent', 0.0),
            'memory_utilization': cpu_metrics.get('avg_memory_percent', 0.0),
            'gpu_memory_utilization': gpu_metrics.get('avg_memory_percent', 0.0),
            'system_info': system_metrics
        }
        
    def _calculate_throughput_metrics(self, inference_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate throughput metrics."""
        mean_time_ms = inference_metrics.get('mean_inference_time_ms', 1.0)
        
        return {
            'samples_per_second': 1000.0 / max(mean_time_ms, 0.001),
            'batch_throughput': inference_metrics.get('optimal_batch_size', 1) * (1000.0 / max(mean_time_ms, 0.001)),
            'theoretical_max_throughput': 1000.0 / max(inference_metrics.get('min_inference_time_ms', mean_time_ms), 0.001)
        }
        
    def _calculate_latency_metrics(self, inference_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate latency metrics."""
        return {
            'mean_latency_ms': inference_metrics.get('mean_inference_time_ms', 0.0),
            'p95_latency_ms': inference_metrics.get('p95_inference_time_ms', 0.0),
            'latency_jitter_ms': inference_metrics.get('inference_stability', 0.0) * inference_metrics.get('mean_inference_time_ms', 0.0)
        }
        
    def _analyze_bottlenecks(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        bottlenecks = []
        
        # Memory bottleneck
        memory_results = results.get('memory', {})
        basic_memory = memory_results.get('basic_memory', {})
        if basic_memory.get('peak_memory_mb', 0) > 8000:  # > 8GB
            bottlenecks.append('High memory usage detected')
            
        # CPU bottleneck
        resource_results = results.get('resources', {})
        cpu_metrics = resource_results.get('cpu', {})
        if cpu_metrics.get('avg_cpu_percent', 0) > 80:
            bottlenecks.append('High CPU utilization detected')
            
        # GPU bottleneck
        gpu_metrics = resource_results.get('gpu', {})
        if gpu_metrics.get('avg_gpu_percent', 0) > 90:
            bottlenecks.append('High GPU utilization detected')
            
        # Inference time bottleneck
        inference_results = results.get('inference', {})
        timing = inference_results.get('timing', {})
        if timing.get('mean_time_ms', 0) > 100:  # > 100ms
            bottlenecks.append('High inference latency detected')
            
        return {
            'identified_bottlenecks': bottlenecks,
            'bottleneck_count': len(bottlenecks),
            'performance_limited_by': bottlenecks[0] if bottlenecks else 'No major bottlenecks detected'
        }
        
    def _generate_optimization_suggestions(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on profiling results."""
        suggestions = []
        
        # Memory optimizations
        memory_results = results.get('memory', {})
        basic_memory = memory_results.get('basic_memory', {})
        if basic_memory.get('activation_memory_mb', 0) > basic_memory.get('model_memory_mb', 0) * 2:
            suggestions.append('Consider gradient checkpointing to reduce activation memory')
            
        # Batch size optimizations
        inference_results = results.get('inference', {})
        batch_analysis = inference_results.get('batch_analysis', {})
        if len(batch_analysis) > 1:
            best_efficiency = 0
            best_batch = 1
            for key, metrics in batch_analysis.items():
                if isinstance(metrics, dict) and 'throughput_samples_per_sec' in metrics:
                    if metrics['throughput_samples_per_sec'] > best_efficiency:
                        best_efficiency = metrics['throughput_samples_per_sec']
                        best_batch = int(key.split('_')[1])
            suggestions.append(f'Optimal batch size appears to be {best_batch} for maximum throughput')
            
        # Device optimizations
        if self.device.type == 'cuda':
            gpu_metrics = results.get('resources', {}).get('gpu', {})
            if gpu_metrics.get('avg_gpu_percent', 0) < 50:
                suggestions.append('GPU utilization is low - consider increasing batch size or model complexity')
        else:
            suggestions.append('Consider using GPU acceleration if available')
            
        # General optimizations
        timing = inference_results.get('timing', {})
        if timing.get('std_time_ms', 0) / max(timing.get('mean_time_ms', 1.0), 0.001) > 0.1:
            suggestions.append('High inference time variance detected - consider model optimization or warm-up')
            
        return suggestions
        
    def create_config(self, **kwargs) -> ProfilingConfig:
        """Create profiling configuration."""
        return ProfilingConfig(**kwargs)
        
    def export_profiling_report(self, metrics: PerformanceMetrics, output_path: str) -> None:
        """Export comprehensive profiling report."""
        try:
            report = {
                'profiling_summary': {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'device': str(self.device)
                },
                'inference_metrics': metrics.inference_metrics,
                'memory_metrics': metrics.memory_metrics,
                'throughput_metrics': metrics.throughput_metrics,
                'latency_metrics': metrics.latency_metrics,
                'resource_utilization': metrics.resource_utilization,
                'bottleneck_analysis': metrics.bottleneck_analysis,
                'optimization_suggestions': metrics.optimization_suggestions
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            self.logger.info(f"Profiling report exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export profiling report: {e}")
            raise
            
    def quick_profile(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
        """Quick performance profiling for basic metrics."""
        try:
            model.to(self.device)
            model.eval()
            sample_input = sample_input.to(self.device)
            
            # Quick timing
            with torch.no_grad():
                # Warmup
                for _ in range(5):
                    model(sample_input)
                    
                # Measure
                times = []
                for _ in range(20):
                    start = time.perf_counter()
                    model(sample_input)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
                    
            # Quick memory
            if self.device.type == 'cuda':
                memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                
            return {
                'avg_inference_time_ms': float(np.mean(times)),
                'throughput_samples_per_sec': 1000.0 / np.mean(times),
                'memory_usage_mb': memory_mb,
                'model_parameters': sum(p.numel() for p in model.parameters())
            }
            
        except Exception as e:
            self.logger.error(f"Quick profiling failed: {e}")
            return {'error': str(e)}
