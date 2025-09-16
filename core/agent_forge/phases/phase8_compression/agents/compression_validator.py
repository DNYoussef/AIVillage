import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import copy

@dataclass
class ValidationConfig:
    """Configuration for compression validation."""
    accuracy_threshold: float = 0.95  # Minimum accuracy retention
    performance_threshold: float = 1.5  # Minimum speedup requirement
    memory_threshold: float = 0.8      # Maximum memory usage relative to original
    stability_tests: bool = True
    robustness_tests: bool = True
    cross_platform_tests: bool = True
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 64])
    
@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    accuracy_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    memory_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    robustness_metrics: Dict[str, float]
    overall_score: float
    passed_validation: bool
    detailed_results: Dict[str, Any]
    
class ValidationTest(ABC):
    """Abstract base class for validation tests."""
    
    @abstractmethod
    def run_test(self, original_model: nn.Module, 
                compressed_model: nn.Module,
                test_data: DataLoader,
                device: torch.device) -> Dict[str, Any]:
        pass
        
class AccuracyValidation(ValidationTest):
    """Validate accuracy preservation after compression."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def run_test(self, original_model: nn.Module, 
                compressed_model: nn.Module,
                test_data: DataLoader,
                device: torch.device) -> Dict[str, Any]:
        """Run accuracy validation tests."""
        try:
            results = {}
            
            # Move models to device
            original_model.to(device)
            compressed_model.to(device)
            
            # Basic accuracy comparison
            original_acc = self._evaluate_accuracy(original_model, test_data, device)
            compressed_acc = self._evaluate_accuracy(compressed_model, test_data, device)
            
            results['original_accuracy'] = original_acc
            results['compressed_accuracy'] = compressed_acc
            results['accuracy_retention'] = compressed_acc / original_acc if original_acc > 0 else 0.0
            results['accuracy_drop'] = original_acc - compressed_acc
            
            # Per-class accuracy analysis
            class_accuracies = self._evaluate_per_class_accuracy(compressed_model, test_data, device)
            results['per_class_accuracy'] = class_accuracies
            
            # Confidence analysis
            confidence_metrics = self._analyze_prediction_confidence(original_model, compressed_model, test_data, device)
            results['confidence_analysis'] = confidence_metrics
            
            # Loss analysis
            loss_metrics = self._analyze_loss_distribution(original_model, compressed_model, test_data, device)
            results['loss_analysis'] = loss_metrics
            
            # Top-k accuracy
            top_k_metrics = self._evaluate_top_k_accuracy(compressed_model, test_data, device)
            results['top_k_accuracy'] = top_k_metrics
            
            return results
            
        except Exception as e:
            self.logger.error(f"Accuracy validation failed: {e}")
            return {'error': str(e)}
            
    def _evaluate_accuracy(self, model: nn.Module, test_data: DataLoader, device: torch.device) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total
        
    def _evaluate_per_class_accuracy(self, model: nn.Module, test_data: DataLoader, device: torch.device) -> Dict[int, float]:
        """Evaluate per-class accuracy."""
        model.eval()
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                for i, (p, t) in enumerate(zip(pred, target)):
                    t_item = t.item()
                    if t_item not in class_total:
                        class_total[t_item] = 0
                        class_correct[t_item] = 0
                    
                    class_total[t_item] += 1
                    if p == t:
                        class_correct[t_item] += 1
                        
        return {cls: class_correct[cls] / class_total[cls] 
                for cls in class_total if class_total[cls] > 0}
                
    def _analyze_prediction_confidence(self, original_model: nn.Module, 
                                     compressed_model: nn.Module,
                                     test_data: DataLoader, 
                                     device: torch.device) -> Dict[str, float]:
        """Analyze prediction confidence differences."""
        original_model.eval()
        compressed_model.eval()
        
        original_confidences = []
        compressed_confidences = []
        confidence_diffs = []
        
        with torch.no_grad():
            for data, _ in test_data:
                data = data.to(device)
                
                original_output = F.softmax(original_model(data), dim=1)
                compressed_output = F.softmax(compressed_model(data), dim=1)
                
                orig_conf = original_output.max(dim=1)[0]
                comp_conf = compressed_output.max(dim=1)[0]
                
                original_confidences.extend(orig_conf.cpu().numpy())
                compressed_confidences.extend(comp_conf.cpu().numpy())
                confidence_diffs.extend((orig_conf - comp_conf).cpu().numpy())
                
        return {
            'original_avg_confidence': np.mean(original_confidences),
            'compressed_avg_confidence': np.mean(compressed_confidences),
            'confidence_drop': np.mean(confidence_diffs),
            'confidence_std_diff': np.std(confidence_diffs)
        }
        
    def _analyze_loss_distribution(self, original_model: nn.Module, 
                                 compressed_model: nn.Module,
                                 test_data: DataLoader, 
                                 device: torch.device) -> Dict[str, float]:
        """Analyze loss distribution differences."""
        original_model.eval()
        compressed_model.eval()
        
        original_losses = []
        compressed_losses = []
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(device), target.to(device)
                
                original_output = original_model(data)
                compressed_output = compressed_model(data)
                
                original_loss = F.cross_entropy(original_output, target, reduction='none')
                compressed_loss = F.cross_entropy(compressed_output, target, reduction='none')
                
                original_losses.extend(original_loss.cpu().numpy())
                compressed_losses.extend(compressed_loss.cpu().numpy())
                
        return {
            'original_avg_loss': np.mean(original_losses),
            'compressed_avg_loss': np.mean(compressed_losses),
            'loss_increase': np.mean(compressed_losses) - np.mean(original_losses),
            'loss_std_ratio': np.std(compressed_losses) / np.std(original_losses)
        }
        
    def _evaluate_top_k_accuracy(self, model: nn.Module, test_data: DataLoader, device: torch.device) -> Dict[str, float]:
        """Evaluate top-k accuracy."""
        model.eval()
        top1_correct = 0
        top5_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # Top-1 accuracy
                pred = output.argmax(dim=1)
                top1_correct += pred.eq(target).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = output.topk(5, dim=1)
                top5_correct += top5_pred.eq(target.view(-1, 1).expand_as(top5_pred)).any(dim=1).sum().item()
                
                total += target.size(0)
                
        return {
            'top1_accuracy': top1_correct / total,
            'top5_accuracy': top5_correct / total
        }
        
class PerformanceValidation(ValidationTest):
    """Validate performance improvements after compression."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def run_test(self, original_model: nn.Module, 
                compressed_model: nn.Module,
                test_data: DataLoader,
                device: torch.device) -> Dict[str, Any]:
        """Run performance validation tests."""
        try:
            results = {}
            
            # Move models to device
            original_model.to(device)
            compressed_model.to(device)
            
            # Inference time comparison
            timing_results = self._benchmark_inference_time(original_model, compressed_model, test_data, device)
            results['timing'] = timing_results
            
            # Memory usage comparison
            memory_results = self._benchmark_memory_usage(original_model, compressed_model, device)
            results['memory'] = memory_results
            
            # Throughput comparison
            throughput_results = self._benchmark_throughput(original_model, compressed_model, test_data, device)
            results['throughput'] = throughput_results
            
            # Model size comparison
            size_results = self._compare_model_sizes(original_model, compressed_model)
            results['model_size'] = size_results
            
            # FLOP comparison
            flop_results = self._compare_flops(original_model, compressed_model, next(iter(test_data))[0][:1], device)
            results['flops'] = flop_results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            return {'error': str(e)}
            
    def _benchmark_inference_time(self, original_model: nn.Module, 
                                 compressed_model: nn.Module,
                                 test_data: DataLoader, 
                                 device: torch.device) -> Dict[str, float]:
        """Benchmark inference time."""
        def measure_time(model, data_loader, num_batches=100):
            model.eval()
            times = []
            
            # Warm up
            with torch.no_grad():
                for i, (data, _) in enumerate(data_loader):
                    if i >= 5:
                        break
                    data = data.to(device)
                    model(data)
                    
            # Measure
            with torch.no_grad():
                for i, (data, _) in enumerate(data_loader):
                    if i >= num_batches:
                        break
                        
                    data = data.to(device)
                    
                    start_time = time.time()
                    model(data)
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    
            return times
            
        original_times = measure_time(original_model, test_data)
        compressed_times = measure_time(compressed_model, test_data)
        
        return {
            'original_avg_time': np.mean(original_times),
            'compressed_avg_time': np.mean(compressed_times),
            'speedup': np.mean(original_times) / np.mean(compressed_times),
            'original_std': np.std(original_times),
            'compressed_std': np.std(compressed_times)
        }
        
    def _benchmark_memory_usage(self, original_model: nn.Module, 
                              compressed_model: nn.Module,
                              device: torch.device) -> Dict[str, float]:
        """Benchmark memory usage."""
        def get_model_memory(model):
            total_memory = 0
            for param in model.parameters():
                total_memory += param.numel() * param.element_size()
            for buffer in model.buffers():
                total_memory += buffer.numel() * buffer.element_size()
            return total_memory / (1024 * 1024)  # MB
            
        original_memory = get_model_memory(original_model)
        compressed_memory = get_model_memory(compressed_model)
        
        return {
            'original_memory_mb': original_memory,
            'compressed_memory_mb': compressed_memory,
            'memory_reduction': 1 - (compressed_memory / original_memory),
            'compression_ratio': original_memory / compressed_memory
        }
        
    def _benchmark_throughput(self, original_model: nn.Module, 
                            compressed_model: nn.Module,
                            test_data: DataLoader, 
                            device: torch.device) -> Dict[str, float]:
        """Benchmark throughput (samples per second)."""
        def measure_throughput(model, data_loader, duration=10.0):
            model.eval()
            total_samples = 0
            start_time = time.time()
            
            with torch.no_grad():
                while time.time() - start_time < duration:
                    for data, _ in data_loader:
                        data = data.to(device)
                        model(data)
                        total_samples += data.size(0)
                        
                        if time.time() - start_time >= duration:
                            break
                            
            elapsed_time = time.time() - start_time
            return total_samples / elapsed_time
            
        original_throughput = measure_throughput(original_model, test_data)
        compressed_throughput = measure_throughput(compressed_model, test_data)
        
        return {
            'original_throughput': original_throughput,
            'compressed_throughput': compressed_throughput,
            'throughput_improvement': compressed_throughput / original_throughput
        }
        
    def _compare_model_sizes(self, original_model: nn.Module, compressed_model: nn.Module) -> Dict[str, Any]:
        """Compare model sizes."""
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        
        return {
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'parameter_reduction': 1 - (compressed_params / original_params),
            'compression_ratio': original_params / compressed_params
        }
        
    def _compare_flops(self, original_model: nn.Module, 
                      compressed_model: nn.Module,
                      sample_input: torch.Tensor, 
                      device: torch.device) -> Dict[str, Any]:
        """Compare FLOPs."""
        def count_flops(model, input_tensor):
            total_flops = 0
            
            def flop_hook(module, input, output):
                nonlocal total_flops
                if isinstance(module, nn.Conv2d):
                    if isinstance(output, torch.Tensor):
                        kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                        output_elements = output.numel()
                        total_flops += output_elements * kernel_flops
                elif isinstance(module, nn.Linear):
                    if isinstance(output, torch.Tensor):
                        batch_size = output.shape[0]
                        total_flops += batch_size * module.in_features * module.out_features
                        
            handles = []
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    handle = module.register_forward_hook(flop_hook)
                    handles.append(handle)
                    
            with torch.no_grad():
                model(input_tensor.to(device))
                
            for handle in handles:
                handle.remove()
                
            return total_flops
            
        original_flops = count_flops(original_model, sample_input)
        compressed_flops = count_flops(compressed_model, sample_input)
        
        return {
            'original_flops': original_flops,
            'compressed_flops': compressed_flops,
            'flop_reduction': 1 - (compressed_flops / original_flops),
            'flop_compression_ratio': original_flops / compressed_flops
        }
        
class StabilityValidation(ValidationTest):
    """Validate model stability across different conditions."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def run_test(self, original_model: nn.Module, 
                compressed_model: nn.Module,
                test_data: DataLoader,
                device: torch.device) -> Dict[str, Any]:
        """Run stability validation tests."""
        try:
            results = {}
            
            # Batch size stability
            batch_stability = self._test_batch_size_stability(compressed_model, test_data, device)
            results['batch_stability'] = batch_stability
            
            # Numerical stability
            numerical_stability = self._test_numerical_stability(compressed_model, test_data, device)
            results['numerical_stability'] = numerical_stability
            
            # Gradient stability (if training)
            gradient_stability = self._test_gradient_stability(compressed_model, test_data, device)
            results['gradient_stability'] = gradient_stability
            
            # Repeated inference stability
            inference_stability = self._test_inference_stability(compressed_model, test_data, device)
            results['inference_stability'] = inference_stability
            
            return results
            
        except Exception as e:
            self.logger.error(f"Stability validation failed: {e}")
            return {'error': str(e)}
            
    def _test_batch_size_stability(self, model: nn.Module, 
                                  test_data: DataLoader, 
                                  device: torch.device) -> Dict[str, Any]:
        """Test stability across different batch sizes."""
        model.eval()
        batch_sizes = [1, 4, 8, 16, 32]
        results = {}
        
        # Get a sample
        sample_data, sample_target = next(iter(test_data))
        
        for batch_size in batch_sizes:
            try:
                # Create batch of specified size
                if batch_size <= len(sample_data):
                    batch_data = sample_data[:batch_size].to(device)
                else:
                    # Repeat samples if needed
                    repeats = (batch_size + len(sample_data) - 1) // len(sample_data)
                    expanded_data = sample_data.repeat(repeats, 1, 1, 1)
                    batch_data = expanded_data[:batch_size].to(device)
                    
                with torch.no_grad():
                    output = model(batch_data)
                    
                results[f'batch_{batch_size}'] = {
                    'success': True,
                    'output_shape': list(output.shape),
                    'output_mean': output.mean().item(),
                    'output_std': output.std().item()
                }
                
            except Exception as e:
                results[f'batch_{batch_size}'] = {
                    'success': False,
                    'error': str(e)
                }
                
        return results
        
    def _test_numerical_stability(self, model: nn.Module, 
                                test_data: DataLoader, 
                                device: torch.device) -> Dict[str, Any]:
        """Test numerical stability."""
        model.eval()
        
        # Get a sample
        sample_data, _ = next(iter(test_data))
        sample_data = sample_data[:1].to(device)
        
        results = {}
        
        with torch.no_grad():
            # Baseline output
            baseline_output = model(sample_data)
            
            # Test with small perturbations
            perturbations = [1e-6, 1e-5, 1e-4, 1e-3]
            
            for eps in perturbations:
                perturbed_data = sample_data + torch.randn_like(sample_data) * eps
                perturbed_output = model(perturbed_data)
                
                output_diff = torch.abs(perturbed_output - baseline_output)
                
                results[f'perturbation_{eps}'] = {
                    'max_diff': output_diff.max().item(),
                    'mean_diff': output_diff.mean().item(),
                    'relative_change': (output_diff / torch.abs(baseline_output + 1e-8)).mean().item()
                }
                
        return results
        
    def _test_gradient_stability(self, model: nn.Module, 
                               test_data: DataLoader, 
                               device: torch.device) -> Dict[str, Any]:
        """Test gradient stability during training."""
        model.train()
        
        # Get a sample
        sample_data, sample_target = next(iter(test_data))
        sample_data, sample_target = sample_data[:4].to(device), sample_target[:4].to(device)
        
        results = {}
        
        try:
            # Compute gradients
            output = model(sample_data)
            loss = F.cross_entropy(output, sample_target)
            loss.backward()
            
            # Analyze gradients
            grad_norms = []
            grad_stats = {}
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    
                    grad_stats[name] = {
                        'norm': grad_norm,
                        'max': param.grad.max().item(),
                        'min': param.grad.min().item(),
                        'mean': param.grad.mean().item(),
                        'std': param.grad.std().item()
                    }
                    
            results['gradient_analysis'] = grad_stats
            results['overall_gradient_norm'] = np.mean(grad_norms)
            results['gradient_norm_std'] = np.std(grad_norms)
            
            # Clear gradients
            model.zero_grad()
            
        except Exception as e:
            results['error'] = str(e)
            
        model.eval()  # Reset to eval mode
        return results
        
    def _test_inference_stability(self, model: nn.Module, 
                                test_data: DataLoader, 
                                device: torch.device) -> Dict[str, Any]:
        """Test repeated inference stability."""
        model.eval()
        
        # Get a sample
        sample_data, _ = next(iter(test_data))
        sample_data = sample_data[:1].to(device)
        
        outputs = []
        
        with torch.no_grad():
            for _ in range(10):
                output = model(sample_data)
                outputs.append(output.clone())
                
        # Analyze consistency
        output_tensor = torch.stack(outputs)
        
        return {
            'output_variance': output_tensor.var(dim=0).mean().item(),
            'max_difference': (output_tensor.max(dim=0)[0] - output_tensor.min(dim=0)[0]).max().item(),
            'mean_consistency': (output_tensor.std(dim=0) / (output_tensor.mean(dim=0).abs() + 1e-8)).mean().item()
        }
        
class CompressionValidator:
    """Main compression validation agent."""
    
    def __init__(self, device: torch.device = torch.device('cpu'), logger: Optional[logging.Logger] = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.tests = {
            'accuracy': AccuracyValidation(logger),
            'performance': PerformanceValidation(logger),
            'stability': StabilityValidation(logger)
        }
        
    def validate_compression(self, original_model: nn.Module,
                           compressed_model: nn.Module,
                           test_loader: DataLoader,
                           config: ValidationConfig) -> ValidationMetrics:
        """Comprehensive compression validation."""
        try:
            results = {}
            
            # Run all validation tests
            for test_name, test_instance in self.tests.items():
                self.logger.info(f"Running {test_name} validation")
                test_results = test_instance.run_test(
                    original_model, compressed_model, test_loader, self.device
                )
                results[test_name] = test_results
                
            # Calculate validation metrics
            metrics = self._calculate_validation_metrics(results, config)
            
            self.logger.info(f"Validation completed. Overall score: {metrics.overall_score:.3f}")
            self.logger.info(f"Validation {'PASSED' if metrics.passed_validation else 'FAILED'}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Compression validation failed: {e}")
            raise
            
    def _calculate_validation_metrics(self, results: Dict[str, Any], config: ValidationConfig) -> ValidationMetrics:
        """Calculate comprehensive validation metrics."""
        # Extract key metrics
        accuracy_metrics = self._extract_accuracy_metrics(results.get('accuracy', {}))
        performance_metrics = self._extract_performance_metrics(results.get('performance', {}))
        memory_metrics = self._extract_memory_metrics(results.get('performance', {}))
        stability_metrics = self._extract_stability_metrics(results.get('stability', {}))
        
        # Calculate robustness metrics (placeholder)
        robustness_metrics = {'robustness_score': 0.8}  # Would need additional tests
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            accuracy_metrics, performance_metrics, memory_metrics, 
            stability_metrics, robustness_metrics
        )
        
        # Check validation thresholds
        passed_validation = self._check_validation_thresholds(
            accuracy_metrics, performance_metrics, memory_metrics, config
        )
        
        return ValidationMetrics(
            accuracy_metrics=accuracy_metrics,
            performance_metrics=performance_metrics,
            memory_metrics=memory_metrics,
            stability_metrics=stability_metrics,
            robustness_metrics=robustness_metrics,
            overall_score=overall_score,
            passed_validation=passed_validation,
            detailed_results=results
        )
        
    def _extract_accuracy_metrics(self, accuracy_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract accuracy metrics from test results."""
        return {
            'accuracy_retention': accuracy_results.get('accuracy_retention', 0.0),
            'top1_accuracy': accuracy_results.get('top_k_accuracy', {}).get('top1_accuracy', 0.0),
            'top5_accuracy': accuracy_results.get('top_k_accuracy', {}).get('top5_accuracy', 0.0),
            'confidence_drop': accuracy_results.get('confidence_analysis', {}).get('confidence_drop', 0.0),
            'loss_increase': accuracy_results.get('loss_analysis', {}).get('loss_increase', 0.0)
        }
        
    def _extract_performance_metrics(self, performance_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from test results."""
        timing = performance_results.get('timing', {})
        throughput = performance_results.get('throughput', {})
        flops = performance_results.get('flops', {})
        
        return {
            'speedup': timing.get('speedup', 1.0),
            'throughput_improvement': throughput.get('throughput_improvement', 1.0),
            'flop_reduction': flops.get('flop_reduction', 0.0),
            'inference_time_reduction': 1 - (1 / timing.get('speedup', 1.0)) if timing.get('speedup', 1.0) > 0 else 0.0
        }
        
    def _extract_memory_metrics(self, performance_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract memory metrics from test results."""
        memory = performance_results.get('memory', {})
        model_size = performance_results.get('model_size', {})
        
        return {
            'memory_reduction': memory.get('memory_reduction', 0.0),
            'parameter_reduction': model_size.get('parameter_reduction', 0.0),
            'compression_ratio': memory.get('compression_ratio', 1.0),
            'memory_efficiency': memory.get('memory_reduction', 0.0) * model_size.get('compression_ratio', 1.0)
        }
        
    def _extract_stability_metrics(self, stability_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract stability metrics from test results."""
        batch_stability = stability_results.get('batch_stability', {})
        numerical_stability = stability_results.get('numerical_stability', {})
        
        # Count successful batch size tests
        batch_success_rate = 0.0
        if batch_stability:
            successful_tests = sum(1 for test in batch_stability.values() 
                                 if isinstance(test, dict) and test.get('success', False))
            batch_success_rate = successful_tests / len(batch_stability)
            
        # Analyze numerical stability
        numerical_score = 1.0
        if numerical_stability:
            for test_results in numerical_stability.values():
                if isinstance(test_results, dict) and 'relative_change' in test_results:
                    # Penalize large relative changes
                    relative_change = test_results['relative_change']
                    numerical_score *= max(0.1, 1.0 - relative_change)
                    
        return {
            'batch_stability_score': batch_success_rate,
            'numerical_stability_score': numerical_score,
            'overall_stability_score': (batch_success_rate + numerical_score) / 2
        }
        
    def _calculate_overall_score(self, accuracy_metrics: Dict[str, float],
                                performance_metrics: Dict[str, float],
                                memory_metrics: Dict[str, float],
                                stability_metrics: Dict[str, float],
                                robustness_metrics: Dict[str, float]) -> float:
        """Calculate overall validation score."""
        # Weighted combination of different aspects
        weights = {
            'accuracy': 0.4,
            'performance': 0.25,
            'memory': 0.15,
            'stability': 0.15,
            'robustness': 0.05
        }
        
        # Normalize scores
        accuracy_score = accuracy_metrics.get('accuracy_retention', 0.0)
        performance_score = min(1.0, performance_metrics.get('speedup', 1.0) / 2.0)  # Normalize speedup
        memory_score = memory_metrics.get('memory_reduction', 0.0)
        stability_score = stability_metrics.get('overall_stability_score', 0.0)
        robustness_score = robustness_metrics.get('robustness_score', 0.0)
        
        overall_score = (
            weights['accuracy'] * accuracy_score +
            weights['performance'] * performance_score +
            weights['memory'] * memory_score +
            weights['stability'] * stability_score +
            weights['robustness'] * robustness_score
        )
        
        return overall_score
        
    def _check_validation_thresholds(self, accuracy_metrics: Dict[str, float],
                                   performance_metrics: Dict[str, float],
                                   memory_metrics: Dict[str, float],
                                   config: ValidationConfig) -> bool:
        """Check if validation passes all thresholds."""
        # Check accuracy threshold
        accuracy_ok = accuracy_metrics.get('accuracy_retention', 0.0) >= config.accuracy_threshold
        
        # Check performance threshold
        performance_ok = performance_metrics.get('speedup', 1.0) >= config.performance_threshold
        
        # Check memory threshold
        memory_ok = memory_metrics.get('compression_ratio', 1.0) >= (1.0 / config.memory_threshold)
        
        return accuracy_ok and performance_ok and memory_ok
        
    def create_validation_config(self, **kwargs) -> ValidationConfig:
        """Create validation configuration."""
        return ValidationConfig(**kwargs)
        
    def generate_validation_report(self, metrics: ValidationMetrics, output_path: str) -> None:
        """Generate comprehensive validation report."""
        try:
            report = {
                'validation_summary': {
                    'overall_score': metrics.overall_score,
                    'passed_validation': metrics.passed_validation,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'accuracy_metrics': metrics.accuracy_metrics,
                'performance_metrics': metrics.performance_metrics,
                'memory_metrics': metrics.memory_metrics,
                'stability_metrics': metrics.stability_metrics,
                'robustness_metrics': metrics.robustness_metrics,
                'detailed_results': metrics.detailed_results
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.logger.info(f"Validation report saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate validation report: {e}")
            raise
            
    def quick_validation(self, original_model: nn.Module,
                        compressed_model: nn.Module,
                        test_loader: DataLoader) -> bool:
        """Quick validation for basic compression checks."""
        try:
            # Quick accuracy check
            original_acc = self._quick_accuracy_check(original_model, test_loader)
            compressed_acc = self._quick_accuracy_check(compressed_model, test_loader)
            
            accuracy_retention = compressed_acc / original_acc if original_acc > 0 else 0.0
            
            # Quick size check
            original_params = sum(p.numel() for p in original_model.parameters())
            compressed_params = sum(p.numel() for p in compressed_model.parameters())
            compression_ratio = original_params / compressed_params
            
            # Basic validation: at least 90% accuracy retention and some compression
            return accuracy_retention >= 0.9 and compression_ratio > 1.1
            
        except Exception as e:
            self.logger.error(f"Quick validation failed: {e}")
            return False
            
    def _quick_accuracy_check(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Quick accuracy check on limited data."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if i >= 10:  # Limit to 10 batches for speed
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total if total > 0 else 0.0
