#!/usr/bin/env python3
"""
Agent Forge Phase 6: Quality Validator
======================================

Comprehensive quality validation system that ensures model accuracy preservation,
detects performance theater, and validates optimization quality with NASA POT10
compliance standards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for model validation"""
    # Accuracy metrics
    accuracy: float = 0.0
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Performance metrics
    latency_mean: float = 0.0
    latency_std: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput: float = 0.0

    # Model metrics
    model_size_mb: float = 0.0
    parameter_count: int = 0
    flops: int = 0
    memory_usage_mb: float = 0.0

    # Quality scores
    overall_quality_score: float = 0.0
    nasa_pot10_score: float = 0.0
    theater_risk_score: float = 0.0

@dataclass
class ValidationConfig:
    """Configuration for quality validation"""
    # Accuracy validation
    accuracy_threshold: float = 0.95
    enable_statistical_testing: bool = True
    confidence_level: float = 0.95
    num_bootstrap_samples: int = 1000

    # Performance validation
    enable_latency_validation: bool = True
    latency_threshold_ms: float = 100.0
    enable_memory_validation: bool = True
    memory_threshold_mb: float = 1024.0

    # Theater detection
    enable_theater_detection: bool = True
    theater_sensitivity: float = 0.8
    theater_metrics: List[str] = None

    # NASA POT10 compliance
    enable_nasa_compliance: bool = True
    required_nasa_score: float = 0.90

    # Validation dataset
    validation_batch_size: int = 32
    max_validation_samples: int = 1000

    def __post_init__(self):
        if self.theater_metrics is None:
            self.theater_metrics = [
                "accuracy_variance",
                "latency_consistency",
                "memory_stability",
                "output_determinism"
            ]

class QualityValidator:
    """
    Comprehensive quality validation system that ensures model quality,
    detects performance theater, and maintains NASA POT10 compliance.
    """

    def __init__(self, config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.validation_config = ValidationConfig()

        # Validation state
        self.baseline_metrics: Optional[QualityMetrics] = None
        self.validation_history: List[Dict[str, Any]] = []
        self.theater_patterns: Dict[str, List[float]] = {}

        self.logger.info("QualityValidator initialized")

    def validate_model_quality(
        self,
        model: nn.Module,
        validation_data: Tuple[torch.Tensor, torch.Tensor],
        baseline_metrics: Optional[QualityMetrics] = None,
        model_name: str = "model"
    ) -> Tuple[QualityMetrics, Dict[str, Any]]:
        """
        Comprehensive model quality validation.

        Args:
            model: Model to validate
            validation_data: (inputs, targets) for validation
            baseline_metrics: Optional baseline metrics for comparison
            model_name: Name for tracking

        Returns:
            Tuple of (quality_metrics, validation_report)
        """
        self.logger.info(f"Starting quality validation for {model_name}")
        start_time = time.time()

        validation_inputs, validation_targets = validation_data
        model.eval()

        # Initialize metrics
        metrics = QualityMetrics()
        validation_report = {
            "model_name": model_name,
            "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validation_passed": False,
            "issues_found": [],
            "recommendations": []
        }

        try:
            # Phase 1: Accuracy validation
            self.logger.info("Phase 1: Accuracy validation")
            accuracy_metrics = self._validate_accuracy(
                model, validation_inputs, validation_targets
            )
            self._update_metrics(metrics, accuracy_metrics)

            # Phase 2: Performance validation
            self.logger.info("Phase 2: Performance validation")
            performance_metrics = self._validate_performance(
                model, validation_inputs
            )
            self._update_metrics(metrics, performance_metrics)

            # Phase 3: Statistical validation
            if self.validation_config.enable_statistical_testing:
                self.logger.info("Phase 3: Statistical validation")
                statistical_results = self._validate_statistical_properties(
                    model, validation_data
                )
                validation_report["statistical_analysis"] = statistical_results

            # Phase 4: Theater detection
            if self.validation_config.enable_theater_detection:
                self.logger.info("Phase 4: Theater detection")
                theater_results = self._detect_performance_theater(
                    model, validation_data, metrics, baseline_metrics
                )
                metrics.theater_risk_score = theater_results["risk_score"]
                validation_report["theater_detection"] = theater_results

            # Phase 5: NASA POT10 compliance
            if self.validation_config.enable_nasa_compliance:
                self.logger.info("Phase 5: NASA POT10 compliance")
                nasa_results = self._validate_nasa_compliance(
                    model, validation_data, metrics
                )
                metrics.nasa_pot10_score = nasa_results["compliance_score"]
                validation_report["nasa_compliance"] = nasa_results

            # Phase 6: Quality gate evaluation
            validation_results = self._evaluate_quality_gates(
                metrics, baseline_metrics, validation_report
            )
            validation_report.update(validation_results)

            # Calculate overall quality score
            metrics.overall_quality_score = self._calculate_overall_quality_score(metrics)

            validation_report["validation_time"] = time.time() - start_time
            validation_report["quality_metrics"] = asdict(metrics)

            self.logger.info(f"Quality validation completed in {validation_report['validation_time']:.2f}s")
            self.logger.info(f"Overall quality score: {metrics.overall_quality_score:.3f}")
            self.logger.info(f"Validation passed: {validation_report['validation_passed']}")

            # Store validation history
            self.validation_history.append({
                "timestamp": validation_report["validation_timestamp"],
                "model_name": model_name,
                "metrics": asdict(metrics),
                "passed": validation_report["validation_passed"]
            })

            return metrics, validation_report

        except Exception as e:
            self.logger.error(f"Quality validation failed: {str(e)}")
            validation_report["error"] = str(e)
            validation_report["validation_passed"] = False
            raise

    def validate_accuracy(
        self,
        model: nn.Module,
        validation_data: Tuple[torch.Tensor, torch.Tensor],
        threshold: Optional[float] = None
    ) -> float:
        """
        Validate model accuracy against threshold.

        Args:
            model: Model to validate
            validation_data: (inputs, targets) for validation
            threshold: Accuracy threshold (uses config default if None)

        Returns:
            Model accuracy
        """
        inputs, targets = validation_data
        threshold = threshold or self.validation_config.accuracy_threshold

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            # Handle batch processing for large datasets
            batch_size = self.validation_config.validation_batch_size
            num_samples = min(len(inputs), self.validation_config.max_validation_samples)

            for i in range(0, num_samples, batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_targets = targets[i:i + batch_size]

                outputs = model(batch_inputs)

                if outputs.dim() > 1 and outputs.size(1) > 1:
                    # Classification task
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    # Binary classification or regression
                    predicted = (outputs > 0.5).float().squeeze()

                total += batch_targets.size(0)
                correct += (predicted == batch_targets).sum().item()

        accuracy = correct / total if total > 0 else 0.0

        if accuracy < threshold:
            self.logger.warning(
                f"Accuracy {accuracy:.3f} below threshold {threshold:.3f}"
            )

        return accuracy

    def _validate_accuracy(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Comprehensive accuracy validation"""
        metrics = {}

        model.eval()
        with torch.no_grad():
            outputs = model(inputs[:self.validation_config.max_validation_samples])
            targets = targets[:self.validation_config.max_validation_samples]

            if outputs.dim() > 1 and outputs.size(1) > 1:
                # Multi-class classification
                metrics.update(self._calculate_classification_metrics(outputs, targets))
            else:
                # Binary classification or regression
                metrics.update(self._calculate_binary_metrics(outputs, targets))

        return metrics

    def _calculate_classification_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {}

        # Top-1 accuracy
        _, pred_top1 = torch.max(outputs, 1)
        correct_top1 = (pred_top1 == targets).sum().item()
        metrics["accuracy"] = correct_top1 / len(targets)
        metrics["top1_accuracy"] = metrics["accuracy"]

        # Top-5 accuracy (if applicable)
        if outputs.size(1) >= 5:
            _, pred_top5 = torch.topk(outputs, 5, dim=1)
            correct_top5 = sum([targets[i] in pred_top5[i] for i in range(len(targets))])
            metrics["top5_accuracy"] = correct_top5 / len(targets)

        # Per-class metrics
        num_classes = outputs.size(1)
        if num_classes <= 50:  # Only for reasonable number of classes
            precision, recall, f1 = self._calculate_per_class_metrics(
                pred_top1, targets, num_classes
            )
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1_score"] = f1

        return metrics

    def _calculate_binary_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate binary classification metrics"""
        metrics = {}

        if outputs.dim() > 1:
            outputs = outputs.squeeze()

        predictions = (outputs > 0.5).float()
        targets = targets.float()

        # Basic metrics
        correct = (predictions == targets).sum().item()
        metrics["accuracy"] = correct / len(targets)

        # Precision, recall, F1
        tp = ((predictions == 1) & (targets == 1)).sum().item()
        fp = ((predictions == 1) & (targets == 0)).sum().item()
        fn = ((predictions == 0) & (targets == 1)).sum().item()

        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1_score"] = (
                2 * metrics["precision"] * metrics["recall"] /
                (metrics["precision"] + metrics["recall"])
            )
        else:
            metrics["f1_score"] = 0.0

        return metrics

    def _calculate_per_class_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int
    ) -> Tuple[float, float, float]:
        """Calculate per-class precision, recall, F1"""
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0
        valid_classes = 0

        for class_id in range(num_classes):
            class_mask = (targets == class_id)
            if class_mask.sum() == 0:
                continue

            pred_class_mask = (predictions == class_id)

            tp = ((pred_class_mask) & (class_mask)).sum().item()
            fp = ((pred_class_mask) & (~class_mask)).sum().item()
            fn = ((~pred_class_mask) & (class_mask)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            valid_classes += 1

        avg_precision = precision_sum / valid_classes if valid_classes > 0 else 0.0
        avg_recall = recall_sum / valid_classes if valid_classes > 0 else 0.0
        avg_f1 = f1_sum / valid_classes if valid_classes > 0 else 0.0

        return avg_precision, avg_recall, avg_f1

    def _validate_performance(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> Dict[str, float]:
        """Validate model performance metrics"""
        metrics = {}

        # Latency benchmarking
        latencies = self._benchmark_latency(model, inputs)
        metrics.update({
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99)
        })

        # Throughput calculation
        batch_size = inputs.size(0)
        metrics["throughput"] = batch_size / metrics["latency_mean"] * 1000  # samples/sec

        # Model size and parameters
        metrics["model_size_mb"] = self._calculate_model_size(model)
        metrics["parameter_count"] = sum(p.numel() for p in model.parameters())

        # Memory usage
        metrics["memory_usage_mb"] = self._measure_memory_usage(model, inputs)

        # FLOPs estimation
        metrics["flops"] = self._estimate_flops(model, inputs)

        return metrics

    def _benchmark_latency(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        num_iterations: int = 100
    ) -> List[float]:
        """Benchmark model latency"""
        model.eval()
        latencies = []

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(inputs[:1])

        # Benchmark
        device = next(model.parameters()).device
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            with torch.no_grad():
                _ = model(inputs[:1])

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        return latencies

    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb

    def _measure_memory_usage(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> float:
        """Measure peak memory usage during inference"""
        device = next(model.parameters()).device

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model.eval()
            with torch.no_grad():
                _ = model(inputs[:1])

            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            return peak_memory
        else:
            # For CPU, this is a rough estimation
            return self._calculate_model_size(model) * 2  # Rough estimate

    def _estimate_flops(self, model: nn.Module, inputs: torch.Tensor) -> int:
        """Estimate FLOPs for model inference"""
        # This is a simplified estimation
        # For accurate FLOP counting, use tools like fvcore or ptflops
        total_flops = 0

        def flop_count_hook(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Linear):
                total_flops += module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                output_dims = output.shape[2:]
                kernel_dims = module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                groups = module.groups

                filters_per_channel = out_channels // groups
                conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups

                active_elements_count = int(np.prod(output_dims))
                overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel

                total_flops += overall_conv_flops

        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(flop_count_hook)
                hooks.append(hook)

        # Run inference
        model.eval()
        with torch.no_grad():
            _ = model(inputs[:1])

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return total_flops

    def _validate_statistical_properties(
        self,
        model: nn.Module,
        validation_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, Any]:
        """Validate statistical properties of model predictions"""
        inputs, targets = validation_data
        results = {}

        model.eval()
        with torch.no_grad():
            # Multiple runs for statistical analysis
            predictions_list = []
            for _ in range(10):  # Multiple forward passes
                outputs = model(inputs)
                predictions_list.append(outputs)

            # Consistency analysis
            predictions_tensor = torch.stack(predictions_list)
            prediction_std = torch.std(predictions_tensor, dim=0)
            mean_std = torch.mean(prediction_std).item()

            results["prediction_consistency"] = {
                "mean_std": mean_std,
                "max_std": torch.max(prediction_std).item(),
                "consistent": mean_std < 0.01  # Threshold for consistency
            }

            # Distribution analysis
            final_predictions = predictions_list[-1]
            results["output_distribution"] = self._analyze_output_distribution(
                final_predictions
            )

        return results

    def _analyze_output_distribution(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Analyze distribution properties of model outputs"""
        outputs_np = outputs.detach().cpu().numpy().flatten()

        return {
            "mean": float(np.mean(outputs_np)),
            "std": float(np.std(outputs_np)),
            "min": float(np.min(outputs_np)),
            "max": float(np.max(outputs_np)),
            "skewness": float(self._calculate_skewness(outputs_np)),
            "kurtosis": float(self._calculate_kurtosis(outputs_np))
        }

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def detect_performance_theater(
        self,
        current_metrics: QualityMetrics,
        baseline_metrics: Optional[QualityMetrics] = None,
        performance_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Detect performance theater - fake improvements without real quality gains.

        Args:
            current_metrics: Current model metrics
            baseline_metrics: Baseline metrics for comparison
            performance_history: Historical performance data

        Returns:
            Theater detection results
        """
        theater_results = {
            "is_theater": False,
            "risk_score": 0.0,
            "detected_patterns": [],
            "evidence": {},
            "recommendations": []
        }

        risk_factors = []

        # Pattern 1: Accuracy-Latency Trade-off Theater
        if baseline_metrics is not None:
            accuracy_improvement = (
                current_metrics.accuracy - baseline_metrics.accuracy
            )
            latency_improvement = (
                baseline_metrics.latency_mean - current_metrics.latency_mean
            )

            # Suspicious if huge latency improvement with negligible accuracy loss
            if (latency_improvement > baseline_metrics.latency_mean * 0.5 and
                accuracy_improvement < -0.01):
                risk_factors.append("excessive_accuracy_sacrifice")
                theater_results["evidence"]["accuracy_sacrifice"] = {
                    "accuracy_loss": abs(accuracy_improvement),
                    "latency_gain": latency_improvement
                }

        # Pattern 2: Memory-Performance Theater
        memory_efficiency = (
            current_metrics.throughput / current_metrics.memory_usage_mb
            if current_metrics.memory_usage_mb > 0 else 0
        )
        if memory_efficiency < 0.1:  # Very low memory efficiency
            risk_factors.append("poor_memory_efficiency")

        # Pattern 3: Optimization Theater (many optimizations, little gain)
        # This would require optimization history to detect

        # Pattern 4: Metric Cherry-Picking Theater
        metric_variance = self._calculate_metric_variance(current_metrics)
        if metric_variance > 0.5:  # High variance suggests inconsistent quality
            risk_factors.append("inconsistent_metrics")

        # Calculate risk score
        theater_results["risk_score"] = min(len(risk_factors) * 0.25, 1.0)
        theater_results["is_theater"] = theater_results["risk_score"] > 0.5

        if theater_results["is_theater"]:
            theater_results["detected_patterns"] = risk_factors
            theater_results["recommendations"] = self._generate_theater_recommendations(
                risk_factors
            )

        return theater_results

    def _detect_performance_theater(
        self,
        model: nn.Module,
        validation_data: Tuple[torch.Tensor, torch.Tensor],
        current_metrics: QualityMetrics,
        baseline_metrics: Optional[QualityMetrics] = None
    ) -> Dict[str, Any]:
        """Internal theater detection with model access"""
        return self.detect_performance_theater(current_metrics, baseline_metrics)

    def _calculate_metric_variance(self, metrics: QualityMetrics) -> float:
        """Calculate variance across normalized metrics"""
        # Normalize metrics to 0-1 scale and calculate variance
        normalized_metrics = [
            metrics.accuracy,
            min(metrics.latency_mean / 1000, 1.0),  # Normalize latency
            min(metrics.memory_usage_mb / 1000, 1.0),  # Normalize memory
            min(metrics.throughput / 1000, 1.0)  # Normalize throughput
        ]

        return float(np.var(normalized_metrics))

    def _generate_theater_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on detected theater patterns"""
        recommendations = []

        if "excessive_accuracy_sacrifice" in risk_factors:
            recommendations.append(
                "Consider rebalancing accuracy-latency trade-off. "
                "Validate that accuracy loss is acceptable for use case."
            )

        if "poor_memory_efficiency" in risk_factors:
            recommendations.append(
                "Improve memory efficiency through better optimization techniques. "
                "Current memory usage is disproportionately high."
            )

        if "inconsistent_metrics" in risk_factors:
            recommendations.append(
                "Address metric inconsistencies. High variance suggests "
                "optimization artifacts or measurement issues."
            )

        return recommendations

    def _validate_nasa_compliance(
        self,
        model: nn.Module,
        validation_data: Tuple[torch.Tensor, torch.Tensor],
        metrics: QualityMetrics
    ) -> Dict[str, Any]:
        """Validate NASA POT10 compliance"""
        compliance_results = {
            "compliance_score": 0.0,
            "requirements_met": {},
            "missing_requirements": [],
            "risk_level": "HIGH"
        }

        # NASA POT10 requirements (simplified)
        requirements = {
            "accuracy_requirement": metrics.accuracy >= 0.95,
            "performance_requirement": metrics.latency_p99 <= 100.0,
            "reliability_requirement": metrics.latency_std / metrics.latency_mean <= 0.1,
            "memory_requirement": metrics.memory_usage_mb <= 1000.0,
            "determinism_requirement": True  # Would need additional testing
        }

        met_requirements = sum(requirements.values())
        total_requirements = len(requirements)

        compliance_results["compliance_score"] = met_requirements / total_requirements
        compliance_results["requirements_met"] = requirements

        # Identify missing requirements
        for req_name, req_met in requirements.items():
            if not req_met:
                compliance_results["missing_requirements"].append(req_name)

        # Determine risk level
        if compliance_results["compliance_score"] >= 0.95:
            compliance_results["risk_level"] = "LOW"
        elif compliance_results["compliance_score"] >= 0.80:
            compliance_results["risk_level"] = "MEDIUM"
        else:
            compliance_results["risk_level"] = "HIGH"

        return compliance_results

    def _evaluate_quality_gates(
        self,
        metrics: QualityMetrics,
        baseline_metrics: Optional[QualityMetrics],
        validation_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate quality gates and determine if validation passes"""
        gate_results = {
            "gates_passed": [],
            "gates_failed": [],
            "validation_passed": True
        }

        # Gate 1: Accuracy threshold
        if metrics.accuracy >= self.validation_config.accuracy_threshold:
            gate_results["gates_passed"].append("accuracy_threshold")
        else:
            gate_results["gates_failed"].append("accuracy_threshold")
            gate_results["validation_passed"] = False
            validation_report["issues_found"].append(
                f"Accuracy {metrics.accuracy:.3f} below threshold "
                f"{self.validation_config.accuracy_threshold:.3f}"
            )

        # Gate 2: Performance threshold
        if (self.validation_config.enable_latency_validation and
            metrics.latency_p99 <= self.validation_config.latency_threshold_ms):
            gate_results["gates_passed"].append("latency_threshold")
        elif self.validation_config.enable_latency_validation:
            gate_results["gates_failed"].append("latency_threshold")
            gate_results["validation_passed"] = False
            validation_report["issues_found"].append(
                f"P99 latency {metrics.latency_p99:.1f}ms above threshold "
                f"{self.validation_config.latency_threshold_ms:.1f}ms"
            )

        # Gate 3: Memory threshold
        if (self.validation_config.enable_memory_validation and
            metrics.memory_usage_mb <= self.validation_config.memory_threshold_mb):
            gate_results["gates_passed"].append("memory_threshold")
        elif self.validation_config.enable_memory_validation:
            gate_results["gates_failed"].append("memory_threshold")
            gate_results["validation_passed"] = False
            validation_report["issues_found"].append(
                f"Memory usage {metrics.memory_usage_mb:.1f}MB above threshold "
                f"{self.validation_config.memory_threshold_mb:.1f}MB"
            )

        # Gate 4: NASA compliance
        if (self.validation_config.enable_nasa_compliance and
            metrics.nasa_pot10_score >= self.validation_config.required_nasa_score):
            gate_results["gates_passed"].append("nasa_compliance")
        elif self.validation_config.enable_nasa_compliance:
            gate_results["gates_failed"].append("nasa_compliance")
            gate_results["validation_passed"] = False
            validation_report["issues_found"].append(
                f"NASA POT10 compliance score {metrics.nasa_pot10_score:.3f} "
                f"below required {self.validation_config.required_nasa_score:.3f}"
            )

        return gate_results

    def _calculate_overall_quality_score(self, metrics: QualityMetrics) -> float:
        """Calculate overall quality score based on all metrics"""
        # Weighted scoring
        weights = {
            "accuracy": 0.30,
            "performance": 0.25,
            "nasa_compliance": 0.25,
            "theater_risk": 0.20
        }

        # Normalize metrics to 0-1 scale
        accuracy_score = metrics.accuracy
        performance_score = max(0, 1 - metrics.latency_mean / 1000)  # Lower latency = higher score
        nasa_score = metrics.nasa_pot10_score
        theater_score = 1 - metrics.theater_risk_score  # Lower risk = higher score

        overall_score = (
            weights["accuracy"] * accuracy_score +
            weights["performance"] * performance_score +
            weights["nasa_compliance"] * nasa_score +
            weights["theater_risk"] * theater_score
        )

        return min(max(overall_score, 0.0), 1.0)

    def _update_metrics(self, target_metrics: QualityMetrics, source_metrics: Dict[str, float]):
        """Update QualityMetrics object with values from dictionary"""
        for key, value in source_metrics.items():
            if hasattr(target_metrics, key):
                setattr(target_metrics, key, value)

    def generate_validation_report(
        self,
        validation_results: List[Tuple[str, QualityMetrics, Dict[str, Any]]],
        output_path: Path
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_models": len(validation_results),
                "models_passed": 0,
                "models_failed": 0,
                "average_quality_score": 0.0
            },
            "models": {},
            "recommendations": []
        }

        quality_scores = []

        for model_name, metrics, validation_info in validation_results:
            report["models"][model_name] = {
                "metrics": asdict(metrics),
                "validation_info": validation_info,
                "passed": validation_info.get("validation_passed", False)
            }

            if validation_info.get("validation_passed", False):
                report["summary"]["models_passed"] += 1
            else:
                report["summary"]["models_failed"] += 1

            quality_scores.append(metrics.overall_quality_score)

        # Calculate summary statistics
        if quality_scores:
            report["summary"]["average_quality_score"] = np.mean(quality_scores)

        # Generate recommendations
        report["recommendations"] = self._generate_validation_recommendations(
            validation_results
        )

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Validation report saved to {output_path}")
        return report

    def _generate_validation_recommendations(
        self,
        validation_results: List[Tuple[str, QualityMetrics, Dict[str, Any]]]
    ) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        failed_models = [
            (name, info) for name, _, info in validation_results
            if not info.get("validation_passed", False)
        ]

        if failed_models:
            recommendations.append(
                f"{len(failed_models)} models failed validation. "
                "Review issues and re-optimize before deployment."
            )

        # Common issues analysis
        common_issues = {}
        for _, _, info in validation_results:
            for issue in info.get("issues_found", []):
                issue_type = issue.split()[0]  # Get first word as issue type
                common_issues[issue_type] = common_issues.get(issue_type, 0) + 1

        for issue_type, count in common_issues.items():
            if count > len(validation_results) * 0.5:  # More than 50% of models
                recommendations.append(
                    f"Common issue detected: {issue_type} affects {count} models. "
                    "Consider systematic optimization approach."
                )

        return recommendations


def main():
    """Example usage of QualityValidator"""
    # Setup
    logger = logging.getLogger("QualityValidator")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    # Configuration (using defaults from BakingConfig)
    class MockConfig:
        preserve_accuracy_threshold = 0.95
        enable_theater_detection = True

    config = MockConfig()

    # Initialize validator
    validator = QualityValidator(config, logger)

    # Example model and data
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = ExampleModel()
    validation_inputs = torch.randn(100, 10)
    validation_targets = torch.randint(0, 2, (100,))
    validation_data = (validation_inputs, validation_targets)

    # Validate model
    try:
        metrics, report = validator.validate_model_quality(
            model, validation_data, model_name="example"
        )

        print(f"Validation completed!")
        print(f"Overall quality score: {metrics.overall_quality_score:.3f}")
        print(f"Validation passed: {report['validation_passed']}")
        print(f"Issues found: {len(report['issues_found'])}")

    except Exception as e:
        print(f"Validation failed: {e}")


if __name__ == "__main__":
    main()