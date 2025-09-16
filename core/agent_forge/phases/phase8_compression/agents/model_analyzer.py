import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from dataclasses import dataclass
from collections import defaultdict
import logging

@dataclass
class ModelAnalysis:
    """Complete model analysis for compression planning."""
    total_params: int
    total_size_mb: float
    layer_analysis: Dict[str, Dict[str, Any]]
    bottlenecks: List[str]
    compression_opportunities: Dict[str, float]
    memory_footprint: Dict[str, float]
    computation_profile: Dict[str, Any]
    
class ModelAnalyzer:
    """Analyze model structure and identify compression opportunities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def analyze_model(self, model: nn.Module, sample_input: torch.Tensor) -> ModelAnalysis:
        """Comprehensive model analysis for compression planning."""
        try:
            # Basic model statistics
            total_params = sum(p.numel() for p in model.parameters())
            total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            # Layer-wise analysis
            layer_analysis = self._analyze_layers(model)
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(model, sample_input)
            
            # Compression opportunities
            compression_opportunities = self._identify_compression_opportunities(model)
            
            # Memory footprint analysis
            memory_footprint = self._analyze_memory_footprint(model, sample_input)
            
            # Computation profile
            computation_profile = self._profile_computation(model, sample_input)
            
            return ModelAnalysis(
                total_params=total_params,
                total_size_mb=total_size_mb,
                layer_analysis=layer_analysis,
                bottlenecks=bottlenecks,
                compression_opportunities=compression_opportunities,
                memory_footprint=memory_footprint,
                computation_profile=computation_profile
            )
            
        except Exception as e:
            self.logger.error(f"Model analysis failed: {e}")
            raise
            
    def _analyze_layers(self, model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """Analyze each layer for compression potential."""
        layer_analysis = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                analysis = {
                    'type': type(module).__name__,
                    'params': sum(p.numel() for p in module.parameters()),
                    'prunable': self._is_prunable(module),
                    'quantizable': self._is_quantizable(module),
                    'distillable': self._is_distillable(module)
                }
                
                # Add specific analysis based on layer type
                if isinstance(module, nn.Conv2d):
                    analysis.update(self._analyze_conv_layer(module))
                elif isinstance(module, nn.Linear):
                    analysis.update(self._analyze_linear_layer(module))
                elif isinstance(module, nn.BatchNorm2d):
                    analysis.update(self._analyze_batchnorm_layer(module))
                    
                layer_analysis[name] = analysis
                
        return layer_analysis
        
    def _identify_bottlenecks(self, model: nn.Module, sample_input: torch.Tensor) -> List[str]:
        """Identify computational and memory bottlenecks."""
        bottlenecks = []
        
        # Hook to capture layer statistics
        layer_stats = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    layer_stats[name] = {
                        'output_size': output.numel(),
                        'memory_mb': output.numel() * output.element_size() / (1024 * 1024),
                        'shape': list(output.shape)
                    }
            return hook
            
        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)
                
        # Forward pass to collect statistics
        with torch.no_grad():
            model(sample_input)
            
        # Clean up hooks
        for handle in handles:
            handle.remove()
            
        # Identify bottlenecks (top 20% memory consumers)
        if layer_stats:
            sorted_layers = sorted(layer_stats.items(), 
                                 key=lambda x: x[1]['memory_mb'], 
                                 reverse=True)
            bottleneck_count = max(1, len(sorted_layers) // 5)
            bottlenecks = [name for name, _ in sorted_layers[:bottleneck_count]]
            
        return bottlenecks
        
    def _identify_compression_opportunities(self, model: nn.Module) -> Dict[str, float]:
        """Identify and score compression opportunities."""
        opportunities = {
            'pruning_potential': 0.0,
            'quantization_potential': 0.0,
            'knowledge_distillation_potential': 0.0,
            'architecture_optimization_potential': 0.0
        }
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Pruning potential based on weight magnitudes
        prunable_params = 0
        small_weights = 0
        
        for param in model.parameters():
            if param.dim() >= 2:  # Only weight matrices/tensors
                prunable_params += param.numel()
                # Count weights below 1% of max magnitude
                threshold = 0.01 * param.abs().max()
                small_weights += (param.abs() < threshold).sum().item()
                
        if prunable_params > 0:
            opportunities['pruning_potential'] = small_weights / prunable_params
            
        # Quantization potential (assume 4x compression for 8-bit)
        quantizable_layers = sum(1 for m in model.modules() 
                               if isinstance(m, (nn.Conv2d, nn.Linear)))
        total_layers = sum(1 for _ in model.modules())
        
        if total_layers > 0:
            opportunities['quantization_potential'] = quantizable_layers / total_layers * 0.75
            
        # Knowledge distillation potential based on model complexity
        if total_params > 1000000:  # Large models benefit more
            opportunities['knowledge_distillation_potential'] = min(0.8, total_params / 10000000)
            
        # Architecture optimization based on redundant patterns
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if len(conv_layers) > 5:
            opportunities['architecture_optimization_potential'] = 0.3
            
        return opportunities
        
    def _analyze_memory_footprint(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
        """Analyze memory usage patterns."""
        # Parameters memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Activation memory (approximate)
        with torch.no_grad():
            output = model(sample_input)
            if isinstance(output, torch.Tensor):
                activation_memory = output.numel() * output.element_size() / (1024 * 1024)
            else:
                activation_memory = sum(o.numel() * o.element_size() for o in output) / (1024 * 1024)
                
        return {
            'parameters_mb': param_memory,
            'activations_mb': activation_memory,
            'total_mb': param_memory + activation_memory
        }
        
    def _profile_computation(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profile computational requirements."""
        # Count operations
        flops = self._count_flops(model, sample_input)
        
        # Measure actual inference time
        import time
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                model(sample_input)
                
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                model(sample_input)
                times.append(time.time() - start)
                
        return {
            'flops': flops,
            'avg_inference_time_ms': np.mean(times) * 1000,
            'std_inference_time_ms': np.std(times) * 1000,
            'throughput_samples_per_sec': 1.0 / np.mean(times)
        }
        
    def _count_flops(self, model: nn.Module, sample_input: torch.Tensor) -> int:
        """Estimate FLOPs for the model."""
        total_flops = 0
        
        def flop_hook(name):
            def hook(module, input, output):
                nonlocal total_flops
                if isinstance(module, nn.Conv2d):
                    # Conv2d FLOPs: output_elements * (kernel_size * in_channels + bias)
                    if isinstance(output, torch.Tensor):
                        kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                        output_elements = output.numel()
                        total_flops += output_elements * kernel_flops
                        if module.bias is not None:
                            total_flops += output_elements
                            
                elif isinstance(module, nn.Linear):
                    # Linear FLOPs: output_features * (in_features + bias)
                    if isinstance(output, torch.Tensor):
                        batch_size = output.shape[0]
                        total_flops += batch_size * module.in_features * module.out_features
                        if module.bias is not None:
                            total_flops += batch_size * module.out_features
            return hook
            
        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(flop_hook(name))
                handles.append(handle)
                
        # Forward pass
        with torch.no_grad():
            model(sample_input)
            
        # Clean up
        for handle in handles:
            handle.remove()
            
        return total_flops
        
    def _is_prunable(self, module: nn.Module) -> bool:
        """Check if layer can be pruned."""
        return isinstance(module, (nn.Conv2d, nn.Linear))
        
    def _is_quantizable(self, module: nn.Module) -> bool:
        """Check if layer can be quantized."""
        return isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))
        
    def _is_distillable(self, module: nn.Module) -> bool:
        """Check if layer benefits from knowledge distillation."""
        return isinstance(module, (nn.Conv2d, nn.Linear)) and \
               sum(p.numel() for p in module.parameters()) > 1000
               
    def _analyze_conv_layer(self, conv: nn.Conv2d) -> Dict[str, Any]:
        """Specific analysis for convolutional layers."""
        return {
            'in_channels': conv.in_channels,
            'out_channels': conv.out_channels,
            'kernel_size': conv.kernel_size,
            'stride': conv.stride,
            'padding': conv.padding,
            'groups': conv.groups,
            'separable': conv.groups == conv.in_channels,
            'depthwise': conv.groups == conv.in_channels and conv.out_channels == conv.in_channels
        }
        
    def _analyze_linear_layer(self, linear: nn.Linear) -> Dict[str, Any]:
        """Specific analysis for linear layers."""
        return {
            'in_features': linear.in_features,
            'out_features': linear.out_features,
            'bias': linear.bias is not None,
            'compression_ratio_potential': min(0.9, 1000000 / (linear.in_features * linear.out_features))
        }
        
    def _analyze_batchnorm_layer(self, bn: nn.BatchNorm2d) -> Dict[str, Any]:
        """Specific analysis for batch normalization layers."""
        return {
            'num_features': bn.num_features,
            'eps': bn.eps,
            'momentum': bn.momentum,
            'affine': bn.affine,
            'track_running_stats': bn.track_running_stats
        }
        
    def export_analysis(self, analysis: ModelAnalysis, filepath: str) -> None:
        """Export analysis results to JSON file."""
        try:
            # Convert to serializable format
            export_data = {
                'total_params': analysis.total_params,
                'total_size_mb': analysis.total_size_mb,
                'layer_analysis': analysis.layer_analysis,
                'bottlenecks': analysis.bottlenecks,
                'compression_opportunities': analysis.compression_opportunities,
                'memory_footprint': analysis.memory_footprint,
                'computation_profile': analysis.computation_profile
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            self.logger.info(f"Model analysis exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export analysis: {e}")
            raise
            
    def generate_compression_strategy(self, analysis: ModelAnalysis) -> Dict[str, Any]:
        """Generate compression strategy based on analysis."""
        strategy = {
            'recommended_techniques': [],
            'target_compression_ratio': 1.0,
            'priority_layers': [],
            'estimated_speedup': 1.0,
            'estimated_accuracy_loss': 0.0
        }
        
        opportunities = analysis.compression_opportunities
        
        # Recommend techniques based on potential
        if opportunities['pruning_potential'] > 0.3:
            strategy['recommended_techniques'].append('structured_pruning')
            strategy['target_compression_ratio'] *= (1 - opportunities['pruning_potential'] * 0.5)
            
        if opportunities['quantization_potential'] > 0.5:
            strategy['recommended_techniques'].append('int8_quantization')
            strategy['target_compression_ratio'] *= 0.25  # 4x compression
            
        if opportunities['knowledge_distillation_potential'] > 0.4:
            strategy['recommended_techniques'].append('knowledge_distillation')
            strategy['target_compression_ratio'] *= 0.5
            
        if opportunities['architecture_optimization_potential'] > 0.2:
            strategy['recommended_techniques'].append('architecture_search')
            
        # Priority layers (bottlenecks)
        strategy['priority_layers'] = analysis.bottlenecks[:5]
        
        # Estimate performance impact
        compression_ratio = strategy['target_compression_ratio']
        strategy['estimated_speedup'] = 1.0 / compression_ratio
        strategy['estimated_accuracy_loss'] = (1 - compression_ratio) * 0.1  # Rough estimate
        
        return strategy
