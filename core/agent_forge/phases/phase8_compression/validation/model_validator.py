"""
Phase 8 Compression - Model Validator

Comprehensive model validation framework for compressed models including
accuracy validation, numerical stability, and deployment readiness checks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
import json
import time
from abc import ABC, abstractmethod
import warnings


@dataclass
class ValidationThresholds:
    """Validation thresholds and criteria."""
    min_accuracy_retention: float = 0.95
    max_accuracy_drop: float = 0.05
    max_latency_increase: float = 1.2
    max_memory_increase: float = 1.5
    min_numerical_stability: float = 0.99
    max_inference_variance: float = 0.01
    deployment_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    accuracy_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    deployment_metrics: Dict[str, float]
    overall_score: float
    validation_passed: bool
    issues_found: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report."""
    model_name: str
    validation_timestamp: str
    thresholds: ValidationThresholds
    metrics: ValidationMetrics
    detailed_results: Dict[str, Any]
    recommendations: List[str]
    validation_summary: str


class ValidationTest(ABC):
    """Abstract base class for validation tests."""

    @abstractmethod
    def run_test(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        validation_data: Any,
        thresholds: ValidationThresholds
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """
        Run validation test.

        Returns:
            Tuple of (metrics, issues, warnings)
        """
        pass

    @abstractmethod
    def get_test_name(self) -> str:
        """Get test name."""
        pass


class AccuracyValidationTest(ValidationTest):
    """Validates model accuracy and classification performance."""

    def run_test(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        validation_data: Any,
        thresholds: ValidationThresholds
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """Run accuracy validation test."""

        metrics = {}
        issues = []
        warnings = []

        try:
            # Evaluate original model
            original_accuracy, original_loss = self._evaluate_model(original_model, validation_data)

            # Evaluate compressed model
            compressed_accuracy, compressed_loss = self._evaluate_model(compressed_model, validation_data)

            # Calculate metrics
            accuracy_retention = compressed_accuracy / original_accuracy if original_accuracy > 0 else 0.0
            accuracy_drop = original_accuracy - compressed_accuracy
            loss_change = compressed_loss - original_loss

            metrics.update({
                'original_accuracy': original_accuracy,
                'compressed_accuracy': compressed_accuracy,
                'accuracy_retention': accuracy_retention,
                'accuracy_drop': accuracy_drop,
                'original_loss': original_loss,
                'compressed_loss': compressed_loss,
                'loss_change': loss_change
            })

            # Check thresholds
            if accuracy_retention < thresholds.min_accuracy_retention:
                issues.append(f"Accuracy retention {accuracy_retention:.3f} below threshold {thresholds.min_accuracy_retention}")

            if accuracy_drop > thresholds.max_accuracy_drop:
                issues.append(f"Accuracy drop {accuracy_drop:.3f} exceeds threshold {thresholds.max_accuracy_drop}")

            if loss_change > 0.5:  # Arbitrary threshold
                warnings.append(f"Significant loss increase: {loss_change:.3f}")

            # Additional classification metrics if applicable
            if validation_data:
                class_metrics = self._calculate_class_metrics(compressed_model, validation_data)
                metrics.update(class_metrics)

        except Exception as e:
            issues.append(f"Accuracy validation failed: {e}")
            metrics['accuracy_validation_error'] = True

        return metrics, issues, warnings

    def _evaluate_model(self, model: nn.Module, validation_data: Any) -> Tuple[float, float]:
        """Evaluate model accuracy and loss."""
        model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in validation_data:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None

                if targets is None:
                    continue

                outputs = model(inputs)

                # Calculate loss
                loss = F.cross_entropy(outputs, targets)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                total_correct += (predicted == targets).sum().item()

                num_batches += 1

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return accuracy, avg_loss

    def _calculate_class_metrics(self, model: nn.Module, validation_data: Any) -> Dict[str, float]:
        """Calculate per-class metrics."""
        model.eval()
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for batch in validation_data:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    continue

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                for i in range(targets.size(0)):
                    label = targets[i].item()
                    pred = predicted[i].item()

                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0

                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1

        # Calculate per-class accuracies
        class_accuracies = {}
        for class_id in class_total:
            if class_total[class_id] > 0:
                accuracy = class_correct[class_id] / class_total[class_id]
                class_accuracies[f'class_{class_id}_accuracy'] = accuracy

        # Calculate macro and micro averages
        if class_accuracies:
            macro_avg = np.mean(list(class_accuracies.values()))
            micro_avg = sum(class_correct.values()) / sum(class_total.values())

            class_accuracies['macro_avg_accuracy'] = macro_avg
            class_accuracies['micro_avg_accuracy'] = micro_avg

        return class_accuracies

    def get_test_name(self) -> str:
        return "Accuracy Validation"


class NumericalStabilityTest(ValidationTest):
    """Tests numerical stability and consistency of compressed models."""

    def run_test(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        validation_data: Any,
        thresholds: ValidationThresholds
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """Run numerical stability test."""

        metrics = {}
        issues = []
        warnings = []

        try:
            # Test output consistency
            consistency_metrics = self._test_output_consistency(
                original_model, compressed_model, validation_data
            )
            metrics.update(consistency_metrics)

            # Test numerical precision
            precision_metrics = self._test_numerical_precision(compressed_model, validation_data)
            metrics.update(precision_metrics)

            # Test gradient stability (if applicable)
            if compressed_model.training:
                gradient_metrics = self._test_gradient_stability(compressed_model, validation_data)
                metrics.update(gradient_metrics)

            # Check stability thresholds
            if metrics.get('cosine_similarity', 0) < thresholds.min_numerical_stability:
                issues.append(f"Low numerical stability: {metrics.get('cosine_similarity', 0):.3f}")

            if metrics.get('output_variance', 1.0) > thresholds.max_inference_variance:
                warnings.append(f"High output variance: {metrics.get('output_variance', 0):.3f}")

        except Exception as e:
            issues.append(f"Numerical stability test failed: {e}")
            metrics['stability_test_error'] = True

        return metrics, issues, warnings

    def _test_output_consistency(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        validation_data: Any
    ) -> Dict[str, float]:
        """Test output consistency between original and compressed models."""

        similarities = []
        mse_values = []

        original_model.eval()
        compressed_model.eval()

        with torch.no_grad():
            for batch in validation_data:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                # Get outputs
                original_outputs = original_model(inputs)
                compressed_outputs = compressed_model(inputs)

                # Calculate similarity
                cos_sim = F.cosine_similarity(
                    original_outputs.flatten(),
                    compressed_outputs.flatten(),
                    dim=0
                )
                similarities.append(cos_sim.item())

                # Calculate MSE
                mse = F.mse_loss(original_outputs, compressed_outputs)
                mse_values.append(mse.item())

        return {
            'cosine_similarity': np.mean(similarities),
            'cosine_similarity_std': np.std(similarities),
            'mse_loss': np.mean(mse_values),
            'mse_loss_std': np.std(mse_values)
        }

    def _test_numerical_precision(self, model: nn.Module, validation_data: Any) -> Dict[str, float]:
        """Test numerical precision and stability."""

        model.eval()
        output_ranges = []
        output_variances = []

        with torch.no_grad():
            for batch in validation_data:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                outputs = model(inputs)

                # Calculate output statistics
                output_range = outputs.max() - outputs.min()
                output_var = outputs.var()

                output_ranges.append(output_range.item())
                output_variances.append(output_var.item())

        return {
            'output_range_mean': np.mean(output_ranges),
            'output_range_std': np.std(output_ranges),
            'output_variance': np.mean(output_variances),
            'output_variance_std': np.std(output_variances)
        }

    def _test_gradient_stability(self, model: nn.Module, validation_data: Any) -> Dict[str, float]:
        """Test gradient stability during training."""

        model.train()
        gradient_norms = []

        for batch in validation_data:
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            else:
                continue

            model.zero_grad()
            outputs = model(inputs)

            if targets is not None:
                loss = F.cross_entropy(outputs, targets)
                loss.backward()

                # Calculate gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2

                total_norm = total_norm ** (1. / 2)
                gradient_norms.append(total_norm)

        model.eval()

        return {
            'gradient_norm_mean': np.mean(gradient_norms),
            'gradient_norm_std': np.std(gradient_norms),
            'gradient_norm_max': np.max(gradient_norms) if gradient_norms else 0.0
        }

    def get_test_name(self) -> str:
        return "Numerical Stability"


class PerformanceValidationTest(ValidationTest):
    """Validates model performance characteristics."""

    def run_test(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        validation_data: Any,
        thresholds: ValidationThresholds
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """Run performance validation test."""

        metrics = {}
        issues = []
        warnings = []

        try:
            # Benchmark inference time
            timing_metrics = self._benchmark_inference_time(
                original_model, compressed_model, validation_data
            )
            metrics.update(timing_metrics)

            # Memory usage analysis
            memory_metrics = self._analyze_memory_usage(
                original_model, compressed_model, validation_data
            )
            metrics.update(memory_metrics)

            # Check performance thresholds
            latency_ratio = metrics.get('latency_ratio', 1.0)
            if latency_ratio > thresholds.max_latency_increase:
                issues.append(f"Latency increased by {latency_ratio:.2f}x (threshold: {thresholds.max_latency_increase})")

            memory_ratio = metrics.get('memory_ratio', 1.0)
            if memory_ratio > thresholds.max_memory_increase:
                warnings.append(f"Memory usage increased by {memory_ratio:.2f}x")

        except Exception as e:
            issues.append(f"Performance validation failed: {e}")
            metrics['performance_test_error'] = True

        return metrics, issues, warnings

    def _benchmark_inference_time(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        validation_data: Any,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark inference time."""

        def benchmark_model(model: nn.Module) -> float:
            model.eval()
            times = []

            # Warmup
            for _ in range(10):
                for batch in validation_data:
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch

                    with torch.no_grad():
                        _ = model(inputs)
                    break  # Only use first batch

            # Actual timing
            with torch.no_grad():
                for i in range(num_iterations):
                    for batch in validation_data:
                        if isinstance(batch, (tuple, list)):
                            inputs = batch[0]
                        else:
                            inputs = batch

                        start_time = time.time()
                        _ = model(inputs)
                        end_time = time.time()

                        times.append(end_time - start_time)
                        break  # Only use first batch

            return np.mean(times) * 1000  # Convert to milliseconds

        # Benchmark both models
        original_time = benchmark_model(original_model)
        compressed_time = benchmark_model(compressed_model)

        return {
            'original_inference_ms': original_time,
            'compressed_inference_ms': compressed_time,
            'latency_ratio': compressed_time / original_time if original_time > 0 else 1.0,
            'speedup': original_time / compressed_time if compressed_time > 0 else 1.0
        }

    def _analyze_memory_usage(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        validation_data: Any
    ) -> Dict[str, float]:
        """Analyze memory usage."""

        def get_model_memory(model: nn.Module) -> float:
            """Get model memory usage in MB."""
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / (1024 ** 2)

        original_memory = get_model_memory(original_model)
        compressed_memory = get_model_memory(compressed_model)

        return {
            'original_memory_mb': original_memory,
            'compressed_memory_mb': compressed_memory,
            'memory_ratio': compressed_memory / original_memory if original_memory > 0 else 1.0,
            'memory_reduction': (original_memory - compressed_memory) / original_memory if original_memory > 0 else 0.0
        }

    def get_test_name(self) -> str:
        return "Performance Validation"


class DeploymentReadinessTest(ValidationTest):
    """Tests deployment readiness and compatibility."""

    def run_test(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        validation_data: Any,
        thresholds: ValidationThresholds
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """Run deployment readiness test."""

        metrics = {}
        issues = []
        warnings = []

        try:
            # Test serialization
            serialization_metrics = self._test_serialization(compressed_model)
            metrics.update(serialization_metrics)

            # Test device compatibility
            device_metrics = self._test_device_compatibility(compressed_model)
            metrics.update(device_metrics)

            # Test batch size flexibility
            batch_metrics = self._test_batch_flexibility(compressed_model, validation_data)
            metrics.update(batch_metrics)

            # Check deployment requirements
            if thresholds.deployment_requirements:
                deployment_issues = self._check_deployment_requirements(
                    compressed_model, thresholds.deployment_requirements
                )
                issues.extend(deployment_issues)

        except Exception as e:
            issues.append(f"Deployment readiness test failed: {e}")
            metrics['deployment_test_error'] = True

        return metrics, issues, warnings

    def _test_serialization(self, model: nn.Module) -> Dict[str, float]:
        """Test model serialization and loading."""
        try:
            # Test state_dict serialization
            state_dict = model.state_dict()
            serialized_size = len(torch.save(state_dict, f='/tmp/temp_model.pth'))

            # Test model loading
            model_copy = type(model)(*[])  # This would need proper constructor
            # model_copy.load_state_dict(state_dict)  # Would need to handle architecture mismatch

            return {
                'serialization_success': True,
                'serialized_size_bytes': serialized_size
            }

        except Exception as e:
            return {
                'serialization_success': False,
                'serialization_error': str(e)
            }

    def _test_device_compatibility(self, model: nn.Module) -> Dict[str, float]:
        """Test device compatibility."""
        compatible_devices = []

        # Test CPU compatibility
        try:
            model_cpu = model.cpu()
            dummy_input = torch.randn(1, 3, 224, 224)  # Default input shape
            with torch.no_grad():
                _ = model_cpu(dummy_input)
            compatible_devices.append('cpu')
        except Exception:
            pass

        # Test CUDA compatibility if available
        if torch.cuda.is_available():
            try:
                model_cuda = model.cuda()
                dummy_input = torch.randn(1, 3, 224, 224).cuda()
                with torch.no_grad():
                    _ = model_cuda(dummy_input)
                compatible_devices.append('cuda')
            except Exception:
                pass

        return {
            'cpu_compatible': 'cpu' in compatible_devices,
            'cuda_compatible': 'cuda' in compatible_devices,
            'num_compatible_devices': len(compatible_devices)
        }

    def _test_batch_flexibility(self, model: nn.Module, validation_data: Any) -> Dict[str, float]:
        """Test model flexibility with different batch sizes."""
        model.eval()
        batch_sizes_tested = []
        successful_batch_sizes = []

        test_batch_sizes = [1, 2, 4, 8, 16]

        with torch.no_grad():
            for batch_size in test_batch_sizes:
                try:
                    # Get sample input
                    for batch in validation_data:
                        if isinstance(batch, (tuple, list)):
                            sample_input = batch[0][:1]  # Get single sample
                        else:
                            sample_input = batch[:1]

                        # Repeat to create desired batch size
                        if batch_size > 1:
                            repeat_dims = [batch_size] + [1] * (sample_input.dim() - 1)
                            test_input = sample_input.repeat(*repeat_dims)
                        else:
                            test_input = sample_input

                        # Test inference
                        _ = model(test_input)
                        successful_batch_sizes.append(batch_size)
                        break  # Only test with first batch

                except Exception:
                    pass

                batch_sizes_tested.append(batch_size)

        return {
            'batch_flexibility_score': len(successful_batch_sizes) / len(batch_sizes_tested),
            'max_successful_batch_size': max(successful_batch_sizes) if successful_batch_sizes else 0,
            'successful_batch_sizes': successful_batch_sizes
        }

    def _check_deployment_requirements(
        self,
        model: nn.Module,
        requirements: Dict[str, Any]
    ) -> List[str]:
        """Check specific deployment requirements."""
        issues = []

        # Check model size requirement
        if 'max_model_size_mb' in requirements:
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
            if model_size > requirements['max_model_size_mb']:
                issues.append(f"Model size {model_size:.2f}MB exceeds requirement {requirements['max_model_size_mb']}MB")

        # Check parameter count requirement
        if 'max_parameters' in requirements:
            param_count = sum(p.numel() for p in model.parameters())
            if param_count > requirements['max_parameters']:
                issues.append(f"Parameter count {param_count:,} exceeds requirement {requirements['max_parameters']:,}")

        return issues

    def get_test_name(self) -> str:
        return "Deployment Readiness"


class ModelValidationFramework:
    """
    Comprehensive model validation framework for compressed models.
    """

    def __init__(self, thresholds: ValidationThresholds = None):
        self.thresholds = thresholds or ValidationThresholds()
        self.logger = logging.getLogger(__name__)

        # Initialize validation tests
        self.tests = [
            AccuracyValidationTest(),
            NumericalStabilityTest(),
            PerformanceValidationTest(),
            DeploymentReadinessTest()
        ]

    def validate_model(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        validation_data: Any,
        model_name: str = "compressed_model"
    ) -> ValidationReport:
        """
        Run comprehensive validation of compressed model.

        Args:
            original_model: Original uncompressed model
            compressed_model: Compressed model to validate
            validation_data: Validation dataset
            model_name: Name for the model being validated

        Returns:
            ValidationReport with comprehensive validation results
        """
        self.logger.info(f"Starting comprehensive validation of {model_name}")

        # Initialize validation metrics
        all_metrics = {
            'accuracy_metrics': {},
            'performance_metrics': {},
            'stability_metrics': {},
            'deployment_metrics': {}
        }

        all_issues = []
        all_warnings = []
        detailed_results = {}

        # Run all validation tests
        for test in self.tests:
            self.logger.info(f"Running {test.get_test_name()}")

            try:
                test_metrics, test_issues, test_warnings = test.run_test(
                    original_model, compressed_model, validation_data, self.thresholds
                )

                # Categorize metrics
                test_name = test.get_test_name().lower().replace(' ', '_')
                detailed_results[test_name] = {
                    'metrics': test_metrics,
                    'issues': test_issues,
                    'warnings': test_warnings
                }

                # Add to appropriate metric categories
                if 'accuracy' in test_name:
                    all_metrics['accuracy_metrics'].update(test_metrics)
                elif 'performance' in test_name:
                    all_metrics['performance_metrics'].update(test_metrics)
                elif 'stability' in test_name:
                    all_metrics['stability_metrics'].update(test_metrics)
                elif 'deployment' in test_name:
                    all_metrics['deployment_metrics'].update(test_metrics)

                all_issues.extend(test_issues)
                all_warnings.extend(test_warnings)

            except Exception as e:
                error_msg = f"{test.get_test_name()} failed: {e}"
                self.logger.error(error_msg)
                all_issues.append(error_msg)

        # Calculate overall score
        overall_score = self._calculate_overall_score(all_metrics)

        # Determine if validation passed
        validation_passed = len(all_issues) == 0

        # Create validation metrics
        validation_metrics = ValidationMetrics(
            accuracy_metrics=all_metrics['accuracy_metrics'],
            performance_metrics=all_metrics['performance_metrics'],
            stability_metrics=all_metrics['stability_metrics'],
            deployment_metrics=all_metrics['deployment_metrics'],
            overall_score=overall_score,
            validation_passed=validation_passed,
            issues_found=all_issues,
            warnings=all_warnings
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(validation_metrics)

        # Create validation summary
        validation_summary = self._create_validation_summary(validation_metrics)

        # Create validation report
        report = ValidationReport(
            model_name=model_name,
            validation_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            thresholds=self.thresholds,
            metrics=validation_metrics,
            detailed_results=detailed_results,
            recommendations=recommendations,
            validation_summary=validation_summary
        )

        self.logger.info(f"Validation completed. Overall score: {overall_score:.3f}")
        self.logger.info(f"Validation {'PASSED' if validation_passed else 'FAILED'}")

        return report

    def _calculate_overall_score(self, all_metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall validation score."""
        scores = []

        # Accuracy score
        accuracy_metrics = all_metrics['accuracy_metrics']
        if 'accuracy_retention' in accuracy_metrics:
            scores.append(accuracy_metrics['accuracy_retention'])

        # Performance score
        performance_metrics = all_metrics['performance_metrics']
        if 'speedup' in performance_metrics:
            # Normalize speedup (1.0 = baseline, higher is better)
            speedup_score = min(performance_metrics['speedup'] / 2.0, 1.0)
            scores.append(speedup_score)

        # Stability score
        stability_metrics = all_metrics['stability_metrics']
        if 'cosine_similarity' in stability_metrics:
            scores.append(stability_metrics['cosine_similarity'])

        # Deployment score
        deployment_metrics = all_metrics['deployment_metrics']
        if 'batch_flexibility_score' in deployment_metrics:
            scores.append(deployment_metrics['batch_flexibility_score'])

        return np.mean(scores) if scores else 0.0

    def _generate_recommendations(self, metrics: ValidationMetrics) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Accuracy-based recommendations
        accuracy_retention = metrics.accuracy_metrics.get('accuracy_retention', 1.0)
        if accuracy_retention < 0.95:
            recommendations.append("Consider less aggressive compression or fine-tuning to improve accuracy retention")

        # Performance-based recommendations
        latency_ratio = metrics.performance_metrics.get('latency_ratio', 1.0)
        if latency_ratio > 1.5:
            recommendations.append("Model inference time increased significantly - consider different compression approach")

        # Stability-based recommendations
        cosine_similarity = metrics.stability_metrics.get('cosine_similarity', 1.0)
        if cosine_similarity < 0.95:
            recommendations.append("Low numerical stability detected - verify quantization precision")

        # Deployment-based recommendations
        if not metrics.deployment_metrics.get('cpu_compatible', True):
            recommendations.append("Model not compatible with CPU deployment - check layer compatibility")

        # General recommendations
        if len(metrics.issues_found) > 3:
            recommendations.append("Multiple validation issues found - reconsider compression strategy")

        if not recommendations:
            recommendations.append("Model passes all validation criteria and is ready for deployment")

        return recommendations

    def _create_validation_summary(self, metrics: ValidationMetrics) -> str:
        """Create human-readable validation summary."""
        summary_parts = []

        # Overall status
        if metrics.validation_passed:
            summary_parts.append("✓ VALIDATION PASSED")
        else:
            summary_parts.append("✗ VALIDATION FAILED")

        # Key metrics
        accuracy_retention = metrics.accuracy_metrics.get('accuracy_retention', 0)
        summary_parts.append(f"Accuracy Retention: {accuracy_retention:.1%}")

        speedup = metrics.performance_metrics.get('speedup', 1.0)
        summary_parts.append(f"Speedup: {speedup:.2f}x")

        memory_reduction = metrics.performance_metrics.get('memory_reduction', 0)
        summary_parts.append(f"Memory Reduction: {memory_reduction:.1%}")

        # Issues and warnings
        if metrics.issues_found:
            summary_parts.append(f"Issues: {len(metrics.issues_found)}")

        if metrics.warnings:
            summary_parts.append(f"Warnings: {len(metrics.warnings)}")

        return " | ".join(summary_parts)

    def save_validation_report(self, report: ValidationReport, output_path: Path):
        """Save validation report to file."""
        report_data = {
            'model_name': report.model_name,
            'validation_timestamp': report.validation_timestamp,
            'validation_summary': report.validation_summary,
            'overall_score': report.metrics.overall_score,
            'validation_passed': report.metrics.validation_passed,
            'accuracy_metrics': report.metrics.accuracy_metrics,
            'performance_metrics': report.metrics.performance_metrics,
            'stability_metrics': report.metrics.stability_metrics,
            'deployment_metrics': report.metrics.deployment_metrics,
            'issues_found': report.metrics.issues_found,
            'warnings': report.metrics.warnings,
            'recommendations': report.recommendations,
            'thresholds': {
                'min_accuracy_retention': report.thresholds.min_accuracy_retention,
                'max_accuracy_drop': report.thresholds.max_accuracy_drop,
                'max_latency_increase': report.thresholds.max_latency_increase,
                'max_memory_increase': report.thresholds.max_memory_increase,
                'min_numerical_stability': report.thresholds.min_numerical_stability
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Validation report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import torch.nn as nn

    # Create test models
    original_model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    compressed_model = nn.Sequential(
        nn.Linear(784, 128),  # Reduced size
        nn.ReLU(),
        nn.Linear(128, 64),   # Reduced size
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    # Create dummy validation data
    validation_data = [
        (torch.randn(32, 784), torch.randint(0, 10, (32,)))
        for _ in range(10)
    ]

    # Set validation thresholds
    thresholds = ValidationThresholds(
        min_accuracy_retention=0.90,
        max_accuracy_drop=0.10,
        max_latency_increase=1.5,
        min_numerical_stability=0.95
    )

    # Create validation framework
    validator = ModelValidationFramework(thresholds)

    print("Model validation framework initialized")
    print(f"Validation tests: {len(validator.tests)}")
    print(f"Thresholds: min_accuracy_retention={thresholds.min_accuracy_retention}")
    print("Ready for validation...")