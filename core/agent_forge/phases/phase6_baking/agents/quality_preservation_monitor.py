"""
Phase 6 Baking - Quality Preservation Monitor Agent
Monitors and preserves model quality during tool/persona baking optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import pickle
from collections import defaultdict
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    BLEU_SCORE = "bleu_score"
    ROUGE_SCORE = "rouge_score"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    GRADIENT_SIMILARITY = "gradient_similarity"
    ACTIVATION_SIMILARITY = "activation_similarity"
    WEIGHT_DRIFT = "weight_drift"


class QualityThreshold(Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"
    CUSTOM = "custom"


@dataclass
class QualityConfig:
    metrics: List[QualityMetric]
    threshold_mode: QualityThreshold
    custom_thresholds: Dict[str, float]
    monitoring_frequency: int
    early_stopping: bool
    quality_degradation_limit: float
    reference_model_path: Optional[str]
    evaluation_dataset_size: int
    enable_detailed_logging: bool
    preserve_gradients: bool


@dataclass
class QualitySnapshot:
    timestamp: datetime
    metrics: Dict[str, float]
    model_state_hash: str
    degradation_score: float
    quality_score: float
    warning_flags: List[str]
    recommendation: str


@dataclass
class QualityMetrics:
    snapshots: List[QualitySnapshot]
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    quality_trend: List[float]
    degradation_alerts: List[str]
    preservation_score: float
    monitoring_active: bool
    last_update: datetime


class QualityPreservationMonitor:
    """Advanced quality preservation monitoring with multiple metrics"""

    def __init__(self, config: QualityConfig):
        self.config = config
        self.metrics = QualityMetrics(
            snapshots=[],
            baseline_metrics={},
            current_metrics={},
            quality_trend=[],
            degradation_alerts=[],
            preservation_score=1.0,
            monitoring_active=False,
            last_update=datetime.now()
        )
        self.reference_model = None
        self.baseline_established = False
        self.quality_thresholds = self._initialize_thresholds()
        self.step_count = 0

    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize quality thresholds based on mode"""
        if self.config.threshold_mode == QualityThreshold.CUSTOM:
            return self.config.custom_thresholds

        base_thresholds = {
            QualityMetric.PERPLEXITY.value: 0.1,
            QualityMetric.ACCURACY.value: 0.05,
            QualityMetric.F1_SCORE.value: 0.05,
            QualityMetric.SEMANTIC_SIMILARITY.value: 0.1,
            QualityMetric.WEIGHT_DRIFT.value: 0.2,
            QualityMetric.GRADIENT_SIMILARITY.value: 0.15,
            QualityMetric.ACTIVATION_SIMILARITY.value: 0.1
        }

        if self.config.threshold_mode == QualityThreshold.STRICT:
            return {k: v * 0.5 for k, v in base_thresholds.items()}
        elif self.config.threshold_mode == QualityThreshold.LENIENT:
            return {k: v * 2.0 for k, v in base_thresholds.items()}
        else:  # MODERATE
            return base_thresholds

    def establish_baseline(self, model: nn.Module, evaluation_data: torch.Tensor = None) -> bool:
        """Establish baseline quality metrics"""
        try:
            model.eval()
            baseline_metrics = {}

            with torch.no_grad():
                if evaluation_data is not None:
                    # Compute metrics with evaluation data
                    baseline_metrics.update(self._compute_model_metrics(model, evaluation_data))

                # Compute model intrinsic metrics
                baseline_metrics.update(self._compute_intrinsic_metrics(model))

            self.metrics.baseline_metrics = baseline_metrics
            self.baseline_established = True

            # Create baseline snapshot
            snapshot = QualitySnapshot(
                timestamp=datetime.now(),
                metrics=baseline_metrics,
                model_state_hash=self._compute_model_hash(model),
                degradation_score=0.0,
                quality_score=1.0,
                warning_flags=[],
                recommendation="Baseline established"
            )
            self.metrics.snapshots.append(snapshot)

            logger.info(f"Quality baseline established with {len(baseline_metrics)} metrics")
            return True

        except Exception as e:
            logger.error(f"Failed to establish baseline: {e}")
            return False

    def monitor_quality(self, model: nn.Module, evaluation_data: torch.Tensor = None) -> QualitySnapshot:
        """Monitor current model quality"""
        if not self.baseline_established:
            logger.warning("Baseline not established, cannot monitor quality")
            return None

        try:
            model.eval()
            current_metrics = {}

            with torch.no_grad():
                if evaluation_data is not None:
                    current_metrics.update(self._compute_model_metrics(model, evaluation_data))

                current_metrics.update(self._compute_intrinsic_metrics(model))

            # Compute degradation and quality scores
            degradation_score = self._compute_degradation_score(current_metrics)
            quality_score = 1.0 - degradation_score

            # Check for quality issues
            warning_flags = self._check_quality_warnings(current_metrics, degradation_score)
            recommendation = self._generate_recommendation(warning_flags, degradation_score)

            # Create snapshot
            snapshot = QualitySnapshot(
                timestamp=datetime.now(),
                metrics=current_metrics,
                model_state_hash=self._compute_model_hash(model),
                degradation_score=degradation_score,
                quality_score=quality_score,
                warning_flags=warning_flags,
                recommendation=recommendation
            )

            # Update metrics
            self.metrics.snapshots.append(snapshot)
            self.metrics.current_metrics = current_metrics
            self.metrics.quality_trend.append(quality_score)
            self.metrics.preservation_score = quality_score
            self.metrics.last_update = datetime.now()

            # Check for degradation alerts
            if degradation_score > self.config.quality_degradation_limit:
                alert = f"Quality degradation detected: {degradation_score:.3f} > {self.config.quality_degradation_limit}"
                self.metrics.degradation_alerts.append(alert)
                logger.warning(alert)

            self.step_count += 1
            return snapshot

        except Exception as e:
            logger.error(f"Quality monitoring failed: {e}")
            return None

    def _compute_model_metrics(self, model: nn.Module, data: torch.Tensor) -> Dict[str, float]:
        """Compute model performance metrics"""
        metrics = {}

        try:
            if QualityMetric.PERPLEXITY in self.config.metrics:
                perplexity = self._compute_perplexity(model, data)
                metrics[QualityMetric.PERPLEXITY.value] = perplexity

            if QualityMetric.ACTIVATION_SIMILARITY in self.config.metrics:
                activation_sim = self._compute_activation_similarity(model, data)
                metrics[QualityMetric.ACTIVATION_SIMILARITY.value] = activation_sim

        except Exception as e:
            logger.warning(f"Failed to compute model metrics: {e}")

        return metrics

    def _compute_intrinsic_metrics(self, model: nn.Module) -> Dict[str, float]:
        """Compute model intrinsic quality metrics"""
        metrics = {}

        try:
            if QualityMetric.WEIGHT_DRIFT in self.config.metrics:
                weight_drift = self._compute_weight_drift(model)
                metrics[QualityMetric.WEIGHT_DRIFT.value] = weight_drift

            if QualityMetric.GRADIENT_SIMILARITY in self.config.metrics and self.config.preserve_gradients:
                grad_sim = self._compute_gradient_similarity(model)
                metrics[QualityMetric.GRADIENT_SIMILARITY.value] = grad_sim

        except Exception as e:
            logger.warning(f"Failed to compute intrinsic metrics: {e}")

        return metrics

    def _compute_perplexity(self, model: nn.Module, data: torch.Tensor) -> float:
        """Compute model perplexity"""
        try:
            with torch.no_grad():
                outputs = model(data)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                # Compute cross-entropy loss
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), data.view(-1), ignore_index=-100)
                perplexity = torch.exp(loss).item()

                return perplexity

        except Exception as e:
            logger.warning(f"Perplexity computation failed: {e}")
            return float('inf')

    def _compute_weight_drift(self, model: nn.Module) -> float:
        """Compute weight drift from baseline"""
        if not self.baseline_established or 'baseline_weights' not in self.metrics.baseline_metrics:
            # Store current weights as baseline if not available
            baseline_weights = {}
            for name, param in model.named_parameters():
                baseline_weights[name] = param.data.clone()
            self.metrics.baseline_metrics['baseline_weights'] = baseline_weights
            return 0.0

        try:
            baseline_weights = self.metrics.baseline_metrics['baseline_weights']
            total_drift = 0.0
            total_params = 0

            for name, param in model.named_parameters():
                if name in baseline_weights:
                    drift = torch.norm(param.data - baseline_weights[name]).item()
                    total_drift += drift
                    total_params += param.numel()

            return total_drift / total_params if total_params > 0 else 0.0

        except Exception as e:
            logger.warning(f"Weight drift computation failed: {e}")
            return 0.0

    def _compute_activation_similarity(self, model: nn.Module, data: torch.Tensor) -> float:
        """Compute activation similarity with baseline"""
        try:
            # This is a simplified implementation
            # In practice, you'd need to hook into intermediate activations
            with torch.no_grad():
                outputs = model(data)

                # Use output as proxy for activation similarity
                if hasattr(outputs, 'logits'):
                    current_activations = outputs.logits
                else:
                    current_activations = outputs

                # Compute similarity (simplified)
                activation_norm = torch.norm(current_activations).item()
                return min(1.0, 1.0 / (1.0 + activation_norm))

        except Exception as e:
            logger.warning(f"Activation similarity computation failed: {e}")
            return 0.0

    def _compute_gradient_similarity(self, model: nn.Module) -> float:
        """Compute gradient similarity with baseline"""
        try:
            # This would require stored baseline gradients
            # Simplified implementation
            grad_norm = 0.0
            param_count = 0

            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += torch.norm(param.grad).item()
                    param_count += 1

            return 1.0 / (1.0 + grad_norm / max(param_count, 1))

        except Exception as e:
            logger.warning(f"Gradient similarity computation failed: {e}")
            return 0.0

    def _compute_model_hash(self, model: nn.Module) -> str:
        """Compute hash of model state"""
        try:
            state_dict = model.state_dict()
            model_bytes = pickle.dumps(state_dict)
            return str(hash(model_bytes))
        except Exception:
            return str(hash(str(time.time())))

    def _compute_degradation_score(self, current_metrics: Dict[str, float]) -> float:
        """Compute overall quality degradation score"""
        degradation = 0.0
        metric_count = 0

        for metric_name, current_value in current_metrics.items():
            if metric_name in self.metrics.baseline_metrics:
                baseline_value = self.metrics.baseline_metrics[metric_name]
                threshold = self.quality_thresholds.get(metric_name, 0.1)

                # Compute relative change
                if baseline_value != 0:
                    relative_change = abs(current_value - baseline_value) / abs(baseline_value)
                else:
                    relative_change = abs(current_value)

                # Normalize by threshold
                metric_degradation = min(1.0, relative_change / threshold)
                degradation += metric_degradation
                metric_count += 1

        return degradation / max(metric_count, 1)

    def _check_quality_warnings(self, current_metrics: Dict[str, float], degradation_score: float) -> List[str]:
        """Check for quality warning conditions"""
        warnings = []

        # Overall degradation warning
        if degradation_score > self.config.quality_degradation_limit:
            warnings.append(f"High quality degradation: {degradation_score:.3f}")

        # Metric-specific warnings
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.metrics.baseline_metrics:
                baseline_value = self.metrics.baseline_metrics[metric_name]
                threshold = self.quality_thresholds.get(metric_name, 0.1)

                if baseline_value != 0:
                    change = abs(current_value - baseline_value) / abs(baseline_value)
                    if change > threshold:
                        warnings.append(f"{metric_name} degraded by {change:.3f}")

        return warnings

    def _generate_recommendation(self, warning_flags: List[str], degradation_score: float) -> str:
        """Generate quality preservation recommendation"""
        if not warning_flags:
            return "Quality preserved - continue optimization"

        if degradation_score > 0.5:
            return "Critical quality loss - consider reverting to previous checkpoint"
        elif degradation_score > 0.3:
            return "Significant quality loss - reduce optimization aggressiveness"
        elif degradation_score > 0.1:
            return "Minor quality loss - monitor closely"
        else:
            return "Quality stable - continue with caution"

    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report"""
        return {
            'baseline_established': self.baseline_established,
            'monitoring_active': self.metrics.monitoring_active,
            'current_quality_score': self.metrics.preservation_score,
            'degradation_alerts': self.metrics.degradation_alerts,
            'quality_trend': self.metrics.quality_trend[-10:],  # Last 10 measurements
            'latest_snapshot': asdict(self.metrics.snapshots[-1]) if self.metrics.snapshots else None,
            'total_snapshots': len(self.metrics.snapshots),
            'config': asdict(self.config)
        }

    async def monitor_quality_async(self, model: nn.Module, evaluation_data: torch.Tensor = None) -> Dict[str, Any]:
        """Asynchronously monitor model quality"""
        try:
            snapshot = self.monitor_quality(model, evaluation_data)

            if snapshot:
                return {
                    'quality_score': snapshot.quality_score,
                    'degradation_score': snapshot.degradation_score,
                    'warning_flags': snapshot.warning_flags,
                    'recommendation': snapshot.recommendation,
                    'metrics': snapshot.metrics
                }
            else:
                return {'error': 'Quality monitoring failed'}

        except Exception as e:
            logger.error(f"Async quality monitoring failed: {e}")
            return {'error': str(e)}


def create_default_quality_config() -> QualityConfig:
    """Create default quality configuration"""
    return QualityConfig(
        metrics=[
            QualityMetric.PERPLEXITY,
            QualityMetric.WEIGHT_DRIFT,
            QualityMetric.ACTIVATION_SIMILARITY
        ],
        threshold_mode=QualityThreshold.MODERATE,
        custom_thresholds={},
        monitoring_frequency=10,
        early_stopping=True,
        quality_degradation_limit=0.3,
        reference_model_path=None,
        evaluation_dataset_size=1000,
        enable_detailed_logging=True,
        preserve_gradients=False
    )


# Agent Integration Interface
class QualityPreservationMonitorAgent:
    """Agent wrapper for quality preservation monitor"""

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or create_default_quality_config()
        self.monitor = QualityPreservationMonitor(self.config)
        self.agent_id = "quality_preservation_monitor"
        self.status = "idle"

    async def run(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """Run quality monitoring agent"""
        self.status = "running"

        try:
            evaluation_data = kwargs.get('evaluation_data')

            # Establish baseline if not done
            if not self.monitor.baseline_established:
                self.monitor.establish_baseline(model, evaluation_data)

            # Monitor quality
            result = await self.monitor.monitor_quality_async(model, evaluation_data)

            self.status = "completed"
            return result

        except Exception as e:
            self.status = "failed"
            logger.error(f"Quality preservation monitor failed: {e}")
            return {'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'quality_report': self.monitor.get_quality_report()
        }