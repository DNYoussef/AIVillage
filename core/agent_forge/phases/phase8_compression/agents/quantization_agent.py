import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QConfig
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import copy
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class QuantizationConfig:
    """Configuration for quantization strategies."""
    quantization_type: str  # 'static', 'dynamic', 'qat', 'fx'
    backend: str = 'fbgemm'  # 'fbgemm', 'qnnpack'
    calibration_dataset: Optional[torch.utils.data.DataLoader] = None
    num_calibration_batches: int = 100
    qconfig_spec: Optional[Dict[str, Any]] = None
    preserve_sparsity: bool = False
    
@dataclass
class QuantizationResult:
    """Results from quantization operation."""
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    quantization_error: float
    layer_precisions: Dict[str, str]
    performance_metrics: Dict[str, float]
    
class QuantizationStrategy(ABC):
    """Abstract base class for quantization strategies."""
    
    @abstractmethod
    def quantize_model(self, model: nn.Module, config: QuantizationConfig) -> Tuple[nn.Module, QuantizationResult]:
        pass
        
class StaticQuantization(QuantizationStrategy):
    """Post-training static quantization."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def quantize_model(self, model: nn.Module, config: QuantizationConfig) -> Tuple[nn.Module, QuantizationResult]:
        """Apply static quantization."""
        try:
            # Prepare model for quantization
            model.eval()
            
            # Set quantization config
            if config.qconfig_spec:
                qconfig = self._create_custom_qconfig(config.qconfig_spec)
            else:
                qconfig = torch.quantization.get_default_qconfig(config.backend)
                
            model.qconfig = qconfig
            
            # Prepare model
            prepared_model = torch.quantization.prepare(model, inplace=False)
            
            # Calibration
            if config.calibration_dataset:
                self._calibrate_model(prepared_model, config)
            else:
                self.logger.warning("No calibration dataset provided for static quantization")
                
            # Convert to quantized model
            quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            
            # Calculate results
            result = self._calculate_quantization_result(model, quantized_model, config)
            
            return quantized_model, result
            
        except Exception as e:
            self.logger.error(f"Static quantization failed: {e}")
            raise
            
    def _calibrate_model(self, model: nn.Module, config: QuantizationConfig) -> None:
        """Calibrate model for static quantization."""
        model.eval()
        
        with torch.no_grad():
            for i, (data, _) in enumerate(config.calibration_dataset):
                if i >= config.num_calibration_batches:
                    break
                model(data)
                
        self.logger.info(f"Calibration completed with {i+1} batches")
        
    def _create_custom_qconfig(self, qconfig_spec: Dict[str, Any]) -> QConfig:
        """Create custom quantization configuration."""
        # Default quantization settings
        activation_observer = torch.quantization.MinMaxObserver.with_args(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        )
        
        weight_observer = torch.quantization.MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
        
        # Apply custom settings
        if 'activation_bits' in qconfig_spec:
            if qconfig_spec['activation_bits'] == 8:
                activation_observer = torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8
                )
            elif qconfig_spec['activation_bits'] == 16:
                activation_observer = torch.quantization.PlaceholderObserver.with_args(
                    dtype=torch.float16
                )
                
        if 'weight_bits' in qconfig_spec:
            if qconfig_spec['weight_bits'] == 8:
                weight_observer = torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8
                )
                
        return QConfig(
            activation=activation_observer,
            weight=weight_observer
        )
        
    def _calculate_quantization_result(self, original_model: nn.Module, 
                                     quantized_model: nn.Module,
                                     config: QuantizationConfig) -> QuantizationResult:
        """Calculate quantization results."""
        # Calculate model sizes
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        
        # Get layer precisions
        layer_precisions = self._get_layer_precisions(quantized_model)
        
        return QuantizationResult(
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=original_size / quantized_size,
            quantization_error=0.0,  # Would need validation dataset to compute
            layer_precisions=layer_precisions,
            performance_metrics={}
        )
        
class DynamicQuantization(QuantizationStrategy):
    """Dynamic quantization for inference."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def quantize_model(self, model: nn.Module, config: QuantizationConfig) -> Tuple[nn.Module, QuantizationResult]:
        """Apply dynamic quantization."""
        try:
            # Apply dynamic quantization to specific layer types
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=torch.qint8,
                inplace=False
            )
            
            # Calculate results
            result = self._calculate_quantization_result(model, quantized_model, config)
            
            return quantized_model, result
            
        except Exception as e:
            self.logger.error(f"Dynamic quantization failed: {e}")
            raise
            
    def _calculate_quantization_result(self, original_model: nn.Module, 
                                     quantized_model: nn.Module,
                                     config: QuantizationConfig) -> QuantizationResult:
        """Calculate quantization results."""
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        
        layer_precisions = self._get_dynamic_layer_precisions(quantized_model)
        
        return QuantizationResult(
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=original_size / quantized_size,
            quantization_error=0.0,
            layer_precisions=layer_precisions,
            performance_metrics={}
        )
        
    def _get_dynamic_layer_precisions(self, model: nn.Module) -> Dict[str, str]:
        """Get precision information for dynamically quantized layers."""
        precisions = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                if hasattr(module.weight, 'dtype'):
                    if module.weight.dtype == torch.qint8:
                        precisions[name] = 'int8'
                    elif module.weight.dtype == torch.float32:
                        precisions[name] = 'float32'
                    else:
                        precisions[name] = str(module.weight.dtype)
                else:
                    precisions[name] = 'float32'
                    
        return precisions
        
class QuantizationAwareTraining(QuantizationStrategy):
    """Quantization-aware training."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def quantize_model(self, model: nn.Module, config: QuantizationConfig) -> Tuple[nn.Module, QuantizationResult]:
        """Prepare model for quantization-aware training."""
        try:
            # Set quantization config
            qconfig = torch.quantization.get_default_qat_qconfig(config.backend)
            model.qconfig = qconfig
            
            # Prepare for QAT
            prepared_model = torch.quantization.prepare_qat(model, inplace=False)
            
            # Note: Actual training would happen here in a real scenario
            # For this implementation, we'll just convert directly
            prepared_model.eval()
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            
            # Calculate results
            result = self._calculate_quantization_result(model, quantized_model, config)
            
            return quantized_model, result
            
        except Exception as e:
            self.logger.error(f"QAT preparation failed: {e}")
            raise
            
    def _calculate_quantization_result(self, original_model: nn.Module, 
                                     quantized_model: nn.Module,
                                     config: QuantizationConfig) -> QuantizationResult:
        """Calculate QAT quantization results."""
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        
        layer_precisions = self._get_qat_layer_precisions(quantized_model)
        
        return QuantizationResult(
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=original_size / quantized_size,
            quantization_error=0.0,
            layer_precisions=layer_precisions,
            performance_metrics={'qat_enabled': True}
        )
        
    def _get_qat_layer_precisions(self, model: nn.Module) -> Dict[str, str]:
        """Get precision information for QAT model."""
        precisions = {}
        
        for name, module in model.named_modules():
            if 'quantized' in str(type(module)).lower():
                precisions[name] = 'int8'
            else:
                precisions[name] = 'float32'
                
        return precisions
        
class FXQuantization(QuantizationStrategy):
    """FX Graph Mode Quantization."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def quantize_model(self, model: nn.Module, config: QuantizationConfig) -> Tuple[nn.Module, QuantizationResult]:
        """Apply FX graph mode quantization."""
        try:
            from torch.quantization.quantize_fx import prepare_fx, convert_fx
            
            # Create example input for tracing
            example_input = torch.randn(1, 3, 224, 224)  # Adjust based on model
            
            # Prepare for quantization
            qconfig_dict = {
                "": torch.quantization.get_default_qconfig(config.backend)
            }
            
            prepared_model = prepare_fx(model, qconfig_dict, example_input)
            
            # Calibration
            if config.calibration_dataset:
                self._calibrate_fx_model(prepared_model, config)
                
            # Convert to quantized model
            quantized_model = convert_fx(prepared_model)
            
            # Calculate results
            result = self._calculate_quantization_result(model, quantized_model, config)
            
            return quantized_model, result
            
        except Exception as e:
            self.logger.error(f"FX quantization failed: {e}")
            # Fallback to static quantization
            self.logger.info("Falling back to static quantization")
            static_strategy = StaticQuantization(self.logger)
            return static_strategy.quantize_model(model, config)
            
    def _calibrate_fx_model(self, model: nn.Module, config: QuantizationConfig) -> None:
        """Calibrate FX model."""
        model.eval()
        
        with torch.no_grad():
            for i, (data, _) in enumerate(config.calibration_dataset):
                if i >= config.num_calibration_batches:
                    break
                model(data)
                
    def _calculate_quantization_result(self, original_model: nn.Module, 
                                     quantized_model: nn.Module,
                                     config: QuantizationConfig) -> QuantizationResult:
        """Calculate FX quantization results."""
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        
        layer_precisions = self._get_fx_layer_precisions(quantized_model)
        
        return QuantizationResult(
            original_size_mb=original_size,
            quantized_size_mb=quantized_size,
            compression_ratio=original_size / quantized_size,
            quantization_error=0.0,
            layer_precisions=layer_precisions,
            performance_metrics={'fx_mode': True}
        )
        
    def _get_fx_layer_precisions(self, model: nn.Module) -> Dict[str, str]:
        """Get precision information for FX quantized model."""
        precisions = {}
        
        for name, module in model.named_modules():
            module_str = str(type(module))
            if 'quantized' in module_str.lower():
                precisions[name] = 'int8'
            else:
                precisions[name] = 'float32'
                
        return precisions
        
class QuantizationAgent:
    """Main quantization agent coordinating different quantization strategies."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.strategies = {
            'static': StaticQuantization(logger),
            'dynamic': DynamicQuantization(logger),
            'qat': QuantizationAwareTraining(logger),
            'fx': FXQuantization(logger)
        }
        
    def quantize_model(self, model: nn.Module, config: QuantizationConfig) -> Tuple[nn.Module, QuantizationResult]:
        """Apply quantization using specified strategy."""
        try:
            if config.quantization_type not in self.strategies:
                raise ValueError(f"Unknown quantization type: {config.quantization_type}")
                
            strategy = self.strategies[config.quantization_type]
            quantized_model, result = strategy.quantize_model(model, config)
            
            self.logger.info(f"Quantization completed: {result.compression_ratio:.2f}x compression")
            return quantized_model, result
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            raise
            
    def evaluate_quantization_impact(self, 
                                    original_model: nn.Module,
                                    quantized_model: nn.Module,
                                    test_loader: torch.utils.data.DataLoader,
                                    device: torch.device = torch.device('cpu')) -> Dict[str, float]:
        """Evaluate the impact of quantization on model performance."""
        try:
            metrics = {}
            
            # Move models to device
            original_model.to(device)
            quantized_model.to(device)
            
            # Evaluate original model
            original_acc = self._evaluate_accuracy(original_model, test_loader, device)
            
            # Evaluate quantized model
            quantized_acc = self._evaluate_accuracy(quantized_model, test_loader, device)
            
            # Calculate metrics
            metrics['original_accuracy'] = original_acc
            metrics['quantized_accuracy'] = quantized_acc
            metrics['accuracy_drop'] = original_acc - quantized_acc
            metrics['relative_accuracy_drop'] = (original_acc - quantized_acc) / original_acc
            
            # Model size comparison
            original_size = self._get_model_size(original_model)
            quantized_size = self._get_model_size(quantized_model)
            
            metrics['compression_ratio'] = original_size / quantized_size
            metrics['size_reduction'] = 1 - (quantized_size / original_size)
            
            # Inference speed comparison
            original_speed = self._measure_inference_speed(original_model, test_loader, device)
            quantized_speed = self._measure_inference_speed(quantized_model, test_loader, device)
            
            metrics['original_speed_ms'] = original_speed
            metrics['quantized_speed_ms'] = quantized_speed
            metrics['speedup'] = original_speed / quantized_speed
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
            
    def _evaluate_accuracy(self, model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
        """Evaluate model accuracy on test set."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total
        
    def _measure_inference_speed(self, model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
        """Measure average inference time per sample."""
        import time
        
        model.eval()
        times = []
        
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= 100:  # Limit to 100 batches
                    break
                    
                data = data.to(device)
                
                start_time = time.time()
                model(data)
                end_time = time.time()
                
                batch_time = (end_time - start_time) * 1000  # Convert to ms
                per_sample_time = batch_time / data.size(0)
                times.append(per_sample_time)
                
        return np.mean(times) if times else 0.0
        
    def _get_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        total_size = 0
        
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
            
        # Add buffer sizes
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
            
        return total_size / (1024 * 1024)  # Convert to MB
        
    def _get_layer_precisions(self, model: nn.Module) -> Dict[str, str]:
        """Get precision information for each layer."""
        precisions = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                if hasattr(module.weight, 'dtype'):
                    dtype = module.weight.dtype
                    if dtype == torch.qint8:
                        precisions[name] = 'int8'
                    elif dtype == torch.quint8:
                        precisions[name] = 'uint8'
                    elif dtype == torch.float16:
                        precisions[name] = 'float16'
                    elif dtype == torch.float32:
                        precisions[name] = 'float32'
                    else:
                        precisions[name] = str(dtype)
                else:
                    precisions[name] = 'float32'
                    
        return precisions
        
    def create_quantization_config(self, 
                                 quantization_type: str,
                                 backend: str = 'fbgemm',
                                 calibration_dataset: Optional[torch.utils.data.DataLoader] = None,
                                 **kwargs) -> QuantizationConfig:
        """Create quantization configuration."""
        return QuantizationConfig(
            quantization_type=quantization_type,
            backend=backend,
            calibration_dataset=calibration_dataset,
            **kwargs
        )
        
    def benchmark_quantization_methods(self, 
                                     model: nn.Module,
                                     test_loader: torch.utils.data.DataLoader,
                                     device: torch.device = torch.device('cpu')) -> Dict[str, Dict[str, float]]:
        """Benchmark different quantization methods."""
        results = {}
        
        methods = ['dynamic', 'static', 'qat']
        
        for method in methods:
            try:
                self.logger.info(f"Benchmarking {method} quantization")
                
                # Create config
                config = QuantizationConfig(
                    quantization_type=method,
                    calibration_dataset=test_loader if method == 'static' else None,
                    num_calibration_batches=10
                )
                
                # Quantize model
                quantized_model, quant_result = self.quantize_model(copy.deepcopy(model), config)
                
                # Evaluate
                eval_metrics = self.evaluate_quantization_impact(
                    model, quantized_model, test_loader, device
                )
                
                # Combine results
                results[method] = {
                    'compression_ratio': quant_result.compression_ratio,
                    'accuracy_drop': eval_metrics.get('accuracy_drop', 0.0),
                    'speedup': eval_metrics.get('speedup', 1.0),
                    'original_size_mb': quant_result.original_size_mb,
                    'quantized_size_mb': quant_result.quantized_size_mb
                }
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark {method}: {e}")
                results[method] = {'error': str(e)}
                
        return results
