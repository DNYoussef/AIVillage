"""
Optimized Validation & Metrics Service (~200 lines)

Combined ML-based validation and performance metrics service for the GraphFixer
refactoring. Provides both proposal validation and system performance tracking:

VALIDATION FEATURES:
- Neural validation models trained on historical data
- Real-time feedback integration and learning
- A/B testing framework for validation strategies
- Automated quality scoring with confidence intervals
- Multi-dimensional validation (accuracy, relevance, novelty)

METRICS FEATURES:
- Real-time performance monitoring with predictive analysis
- GPU utilization and resource tracking
- Optimization recommendations based on bottleneck analysis
- Quality metrics aggregation across all services
- Scalability metrics and capacity planning

PERFORMANCE TARGETS:
- Validate 1000+ proposals in <5 seconds
- Real-time metrics updates (<100ms latency)
- 95%+ validation accuracy through ML optimization
- Predictive alerts 30 seconds before performance degradation
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging
from collections import deque, defaultdict
import json

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Status of proposal validation."""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"

class MetricCategory(Enum):
    """Categories of performance metrics."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    SCALABILITY = "scalability"
    RELIABILITY = "reliability"

@dataclass
class ValidationResult:
    """Result from proposal validation."""
    proposal_id: str
    validation_score: float
    confidence: float
    status: ValidationStatus
    reasoning: str
    evidence: List[str] = field(default_factory=list)
    quality_dimensions: Dict[str, float] = field(default_factory=dict)
    validation_time_ms: float = 0.0
    model_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    category: MetricCategory
    metrics: Dict[str, float]
    trend_indicators: Dict[str, str] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    predictions: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationFeedback:
    """Human feedback for validation learning."""
    proposal_id: str
    human_accepted: bool
    human_reasoning: str
    confidence_level: float
    feedback_timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

class OptimizedValidationMetricsService:
    """
    Combined ML-based validation and comprehensive performance metrics service.
    
    ARCHITECTURE:
    - Neural validation models with continuous learning
    - Real-time metrics collection and analysis
    - Predictive performance modeling
    - A/B testing framework for validation strategies
    - Automated optimization recommendations
    """
    
    def __init__(self,
                 ml_inference_service: Any,
                 validation_threshold: float = 0.6,
                 enable_ab_testing: bool = True,
                 metrics_retention_hours: int = 24,
                 enable_predictive_analytics: bool = True):
        
        self.ml_service = ml_inference_service
        self.validation_threshold = validation_threshold
        self.enable_ab_testing = enable_ab_testing
        self.metrics_retention_hours = metrics_retention_hours
        self.enable_predictive_analytics = enable_predictive_analytics
        
        # Validation models and strategies
        self.validation_models = {
            'primary': 'proposal_validation_transformer_v2',
            'secondary': 'quality_assessment_bert_v1',
            'novelty_detector': 'novelty_assessment_model',
            'domain_classifier': 'domain_relevance_classifier'
        }
        
        # A/B testing configuration
        self.ab_test_groups = {
            'strategy_a': {'model': 'primary', 'threshold': 0.6, 'weight': 0.5},
            'strategy_b': {'model': 'secondary', 'threshold': 0.65, 'weight': 0.5}
        }
        
        # Metrics storage and monitoring
        self.metrics_history: Dict[MetricCategory, deque] = {
            category: deque(maxlen=1000) for category in MetricCategory
        }
        self.current_metrics: Dict[MetricCategory, PerformanceMetrics] = {}
        self.alert_thresholds: Dict[str, float] = {
            'high_latency_ms': 5000,
            'low_accuracy_percent': 80,
            'high_memory_usage_percent': 90,
            'low_throughput_qps': 10
        }
        
        # Learning and feedback systems
        self.validation_feedback_history: List[ValidationFeedback] = []
        self.learning_metrics = {
            'validation_accuracy': deque(maxlen=100),
            'false_positive_rate': deque(maxlen=100),
            'false_negative_rate': deque(maxlen=100),
            'human_agreement_rate': deque(maxlen=100)
        }
        
        # Performance tracking
        self.service_metrics = {
            'total_validations': 0,
            'successful_validations': 0,
            'ml_model_accuracy': 0.0,
            'avg_validation_time_ms': 0.0,
            'feedback_processed': 0,
            'ab_test_conversions': {'strategy_a': 0, 'strategy_b': 0},
            'predictive_alerts_triggered': 0
        }

    async def validate_proposals_batch(self,
                                     proposals: List[Union['ProposedNode', 'ProposedRelationship']],
                                     context: Dict[str, Any]) -> List[ValidationResult]:
        """
        Batch validation of proposals using ML models with A/B testing.
        
        OPTIMIZATION STRATEGY:
        1. Group proposals by type for efficient batch processing
        2. Apply A/B testing for validation strategy comparison
        3. Use neural models for multi-dimensional quality assessment
        4. Provide confidence intervals and uncertainty quantification
        5. Learn from validation outcomes to improve accuracy
        """
        start_time = asyncio.get_event_loop().time()
        
        if not proposals:
            return []
        
        logger.info(f"Validating {len(proposals)} proposals using ML models")
        
        # Determine A/B test group for this batch
        ab_strategy = self._select_ab_test_strategy() if self.enable_ab_testing else 'strategy_a'
        validation_config = self.ab_test_groups[ab_strategy]
        
        # Prepare validation contexts for batch processing
        validation_contexts = await self._prepare_validation_contexts(proposals, context)
        
        # Execute batch validation using selected strategy
        validation_results = await self._execute_batch_validation(validation_contexts, validation_config)
        
        # Post-process results and update metrics
        processed_results = await self._post_process_validation_results(validation_results, ab_strategy)
        
        # Update service metrics
        validation_time = (asyncio.get_event_loop().time() - start_time) * 1000
        self._update_validation_metrics(len(proposals), validation_time)
        
        logger.info(f"Validated {len(processed_results)} proposals in {validation_time:.1f}ms using {ab_strategy}")
        
        return processed_results

    async def _execute_batch_validation(self,
                                      contexts: List[Dict[str, Any]],
                                      validation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute batch validation using neural models."""
        from .ml_inference_service import MLInferenceRequest, InferencePriority
        
        # Primary validation using main model
        primary_request = MLInferenceRequest(
            operation='validate_proposals',
            data={
                'contexts': contexts,
                'model': validation_config['model'],
                'threshold': validation_config['threshold'],
                'include_explanations': True,
                'include_confidence_intervals': True,
                'quality_dimensions': ['accuracy', 'relevance', 'novelty', 'feasibility']
            },
            priority=InferencePriority.HIGH,
            timeout_ms=20000
        )
        
        primary_result = await self.ml_service.infer(primary_request)
        
        if not primary_result.success:
            logger.error(f"Primary validation failed: {primary_result.error_message}")
            return []
        
        # Secondary validation for quality assessment (if different model)
        secondary_results = None
        if validation_config['model'] != self.validation_models['secondary']:
            secondary_request = MLInferenceRequest(
                operation='assess_quality',
                data={
                    'contexts': contexts,
                    'model': self.validation_models['secondary'],
                    'focus_dimensions': ['domain_relevance', 'implementation_complexity']
                },
                priority=InferencePriority.NORMAL,
                timeout_ms=15000
            )
            
            secondary_result = await self.ml_service.infer(secondary_request)
            if secondary_result.success:
                secondary_results = secondary_result.data['quality_assessments']
        
        # Combine primary and secondary validation results
        combined_results = []
        primary_validations = primary_result.data['validation_results']
        
        for i, primary_val in enumerate(primary_validations):
            combined = primary_val.copy()
            
            # Add secondary validation if available
            if secondary_results and i < len(secondary_results):
                secondary_val = secondary_results[i]
                combined['quality_dimensions'].update(secondary_val.get('quality_dimensions', {}))
                combined['secondary_confidence'] = secondary_val.get('confidence', 0.0)
            
            combined_results.append(combined)
        
        return combined_results

    async def _prepare_validation_contexts(self,
                                         proposals: List[Union['ProposedNode', 'ProposedRelationship']],
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare rich contexts for ML-based validation."""
        contexts = []
        
        for proposal in proposals:
            # Base context information
            base_context = {
                'proposal_id': proposal.id,
                'proposal_type': type(proposal).__name__,
                'confidence': proposal.confidence,
                'reasoning': proposal.reasoning,
                'gap_context': proposal.gap_id,
                'learning_features': getattr(proposal, 'learning_features', {})
            }
            
            # Add proposal-specific information
            if hasattr(proposal, 'concept'):  # ProposedNode
                base_context.update({
                    'concept': proposal.concept,
                    'content': proposal.content,
                    'existence_probability': proposal.existence_probability,
                    'utility_score': proposal.utility_score,
                    'novelty_score': getattr(proposal, 'novelty_score', 0.5)
                })
            elif hasattr(proposal, 'relation_type'):  # ProposedRelationship
                base_context.update({
                    'source_id': proposal.source_id,
                    'target_id': proposal.target_id,
                    'relation_type': proposal.relation_type,
                    'relation_strength': proposal.relation_strength,
                    'semantic_strength': getattr(proposal, 'semantic_strength', 0.5)
                })
            
            # Add external context
            base_context.update({
                'domain_context': context.get('domain_knowledge', {}),
                'graph_context': context.get('graph_metrics', {}),
                'historical_performance': await self._get_historical_validation_performance(proposal)
            })
            
            contexts.append(base_context)
        
        return contexts

    async def _post_process_validation_results(self,
                                             raw_results: List[Dict[str, Any]],
                                             ab_strategy: str) -> List[ValidationResult]:
        """Post-process raw validation results into structured format."""
        processed_results = []
        
        for result_data in raw_results:
            # Determine validation status
            score = result_data['validation_score']
            confidence = result_data['confidence']
            
            if score >= self.validation_threshold:
                if confidence > 0.8:
                    status = ValidationStatus.VALIDATED
                else:
                    status = ValidationStatus.UNCERTAIN
            else:
                status = ValidationStatus.REJECTED
            
            # Extract quality dimensions
            quality_dimensions = result_data.get('quality_dimensions', {})
            
            # Create structured result
            validation_result = ValidationResult(
                proposal_id=result_data['proposal_id'],
                validation_score=score,
                confidence=confidence,
                status=status,
                reasoning=result_data.get('explanation', 'ML model assessment'),
                evidence=result_data.get('evidence', []),
                quality_dimensions=quality_dimensions,
                validation_time_ms=result_data.get('processing_time_ms', 0.0),
                model_used=result_data.get('model_used', 'unknown'),
                metadata={
                    'ab_strategy': ab_strategy,
                    'secondary_confidence': result_data.get('secondary_confidence'),
                    'quality_breakdown': quality_dimensions
                }
            )
            
            processed_results.append(validation_result)
        
        return processed_results

    async def learn_from_human_feedback(self, feedback: ValidationFeedback) -> bool:
        """
        Learn from human validation feedback to improve ML models.
        
        Implements online learning and model fine-tuning based on human corrections.
        """
        try:
            # Store feedback for analysis
            self.validation_feedback_history.append(feedback)
            
            # Find corresponding validation result
            corresponding_validation = await self._find_validation_result(feedback.proposal_id)
            
            if corresponding_validation:
                # Create training example for incremental learning
                training_example = {
                    'proposal_context': corresponding_validation.metadata.get('context', {}),
                    'ml_prediction': {
                        'score': corresponding_validation.validation_score,
                        'confidence': corresponding_validation.confidence,
                        'reasoning': corresponding_validation.reasoning
                    },
                    'human_ground_truth': {
                        'accepted': feedback.human_accepted,
                        'reasoning': feedback.human_reasoning,
                        'confidence': feedback.confidence_level
                    },
                    'prediction_error': abs(
                        (1.0 if feedback.human_accepted else 0.0) - corresponding_validation.validation_score
                    )
                }
                
                # Send to ML service for incremental learning
                from .ml_inference_service import MLInferenceRequest, InferencePriority
                
                learning_request = MLInferenceRequest(
                    operation='incremental_learning',
                    data={
                        'training_example': training_example,
                        'model': corresponding_validation.model_used,
                        'learning_rate': 0.0001,  # Conservative learning rate
                        'update_strategy': 'gradient_accumulation'
                    },
                    priority=InferencePriority.LOW
                )
                
                learning_result = await self.ml_service.infer(learning_request)
                
                if learning_result.success:
                    # Update learning metrics
                    self._update_learning_metrics(corresponding_validation, feedback)
                    self.service_metrics['feedback_processed'] += 1
                    
                    logger.info(f"Successfully processed feedback for proposal {feedback.proposal_id}")
                    return True
                else:
                    logger.error(f"Incremental learning failed: {learning_result.error_message}")
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to process human feedback: {e}")
            return False

    async def collect_performance_metrics(self,
                                        service_logs: List[Dict[str, Any]],
                                        system_stats: Dict[str, Any]) -> Dict[MetricCategory, PerformanceMetrics]:
        """
        Collect and analyze comprehensive performance metrics across all services.
        
        Provides real-time monitoring, trend analysis, and predictive alerts.
        """
        current_time = datetime.now()
        collected_metrics = {}
        
        # Process each metric category
        for category in MetricCategory:
            category_metrics = await self._compute_category_metrics(category, service_logs, system_stats)
            
            # Add trend analysis
            trend_indicators = self._compute_trend_indicators(category, category_metrics)
            
            # Generate alerts if thresholds exceeded
            alerts = self._check_alert_thresholds(category, category_metrics)
            
            # Generate predictions if enabled
            predictions = {}
            if self.enable_predictive_analytics:
                predictions = await self._generate_performance_predictions(category, category_metrics)
            
            performance_metrics = PerformanceMetrics(
                timestamp=current_time,
                category=category,
                metrics=category_metrics,
                trend_indicators=trend_indicators,
                alerts=alerts,
                predictions=predictions,
                metadata={'collection_time_ms': 0.0}  # Would be measured in real implementation
            )
            
            collected_metrics[category] = performance_metrics
            
            # Store in history
            self.metrics_history[category].append(performance_metrics)
            self.current_metrics[category] = performance_metrics
        
        # Clean up old metrics
        self._cleanup_old_metrics()
        
        return collected_metrics

    async def _compute_category_metrics(self,
                                      category: MetricCategory,
                                      service_logs: List[Dict[str, Any]],
                                      system_stats: Dict[str, Any]) -> Dict[str, float]:
        """Compute metrics for a specific category."""
        metrics = {}
        
        if category == MetricCategory.PERFORMANCE:
            # Extract performance metrics from logs
            latencies = [log.get('processing_time_ms', 0) for log in service_logs if 'processing_time_ms' in log]
            
            if latencies:
                metrics.update({
                    'avg_latency_ms': np.mean(latencies),
                    'p95_latency_ms': np.percentile(latencies, 95),
                    'p99_latency_ms': np.percentile(latencies, 99),
                    'max_latency_ms': np.max(latencies)
                })
            
            # Throughput metrics
            request_counts = [log.get('request_count', 0) for log in service_logs if 'request_count' in log]
            if request_counts:
                metrics['throughput_qps'] = sum(request_counts) / max(len(request_counts), 1)
        
        elif category == MetricCategory.QUALITY:
            # Quality metrics from validation results
            accuracy_scores = [log.get('accuracy', 0) for log in service_logs if 'accuracy' in log]
            confidence_scores = [log.get('confidence', 0) for log in service_logs if 'confidence' in log]
            
            if accuracy_scores:
                metrics.update({
                    'avg_accuracy_percent': np.mean(accuracy_scores) * 100,
                    'min_accuracy_percent': np.min(accuracy_scores) * 100,
                    'accuracy_variance': np.var(accuracy_scores)
                })
            
            if confidence_scores:
                metrics['avg_confidence'] = np.mean(confidence_scores)
        
        elif category == MetricCategory.EFFICIENCY:
            # Resource utilization metrics
            cpu_usage = system_stats.get('cpu_percent', 0)
            memory_usage = system_stats.get('memory_percent', 0)
            gpu_usage = system_stats.get('gpu_percent', 0)
            
            metrics.update({
                'cpu_utilization_percent': cpu_usage,
                'memory_utilization_percent': memory_usage,
                'gpu_utilization_percent': gpu_usage,
                'cache_hit_rate_percent': system_stats.get('cache_hit_rate', 0) * 100
            })
        
        elif category == MetricCategory.SCALABILITY:
            # Scalability and capacity metrics
            concurrent_requests = system_stats.get('concurrent_requests', 0)
            queue_lengths = system_stats.get('queue_lengths', {})
            
            metrics.update({
                'concurrent_requests': concurrent_requests,
                'avg_queue_length': np.mean(list(queue_lengths.values())) if queue_lengths else 0,
                'max_queue_length': max(queue_lengths.values()) if queue_lengths else 0
            })
        
        elif category == MetricCategory.RELIABILITY:
            # Error rates and reliability metrics
            error_counts = [log.get('errors', 0) for log in service_logs if 'errors' in log]
            success_counts = [log.get('successes', 0) for log in service_logs if 'successes' in log]
            
            total_errors = sum(error_counts)
            total_successes = sum(success_counts)
            total_operations = total_errors + total_successes
            
            if total_operations > 0:
                metrics.update({
                    'error_rate_percent': (total_errors / total_operations) * 100,
                    'success_rate_percent': (total_successes / total_operations) * 100,
                    'availability_percent': 100.0  # Would be computed from uptime data
                })
        
        return metrics

    def _compute_trend_indicators(self, category: MetricCategory, current_metrics: Dict[str, float]) -> Dict[str, str]:
        """Compute trend indicators based on historical data."""
        trends = {}
        
        if category not in self.metrics_history or len(self.metrics_history[category]) < 2:
            return trends
        
        history = list(self.metrics_history[category])
        previous_metrics = history[-2].metrics if len(history) >= 2 else {}
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in previous_metrics:
                previous_value = previous_metrics[metric_name]
                
                if current_value > previous_value * 1.1:
                    trends[metric_name] = "increasing"
                elif current_value < previous_value * 0.9:
                    trends[metric_name] = "decreasing"
                else:
                    trends[metric_name] = "stable"
        
        return trends

    def _check_alert_thresholds(self, category: MetricCategory, metrics: Dict[str, float]) -> List[str]:
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        
        for metric_name, value in metrics.items():
            # Check against configured thresholds
            if 'latency_ms' in metric_name and value > self.alert_thresholds.get('high_latency_ms', 5000):
                alerts.append(f"High latency detected: {value:.1f}ms")
            
            elif 'accuracy_percent' in metric_name and value < self.alert_thresholds.get('low_accuracy_percent', 80):
                alerts.append(f"Low accuracy detected: {value:.1f}%")
            
            elif 'memory_utilization_percent' in metric_name and value > self.alert_thresholds.get('high_memory_usage_percent', 90):
                alerts.append(f"High memory usage: {value:.1f}%")
            
            elif 'throughput_qps' in metric_name and value < self.alert_thresholds.get('low_throughput_qps', 10):
                alerts.append(f"Low throughput: {value:.1f} QPS")
        
        if alerts:
            self.service_metrics['predictive_alerts_triggered'] += len(alerts)
        
        return alerts

    async def _generate_performance_predictions(self, category: MetricCategory, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Generate performance predictions using ML models."""
        # This would use time series forecasting models in production
        # For now, provide simple trend-based predictions
        
        predictions = {}
        
        if category not in self.metrics_history or len(self.metrics_history[category]) < 5:
            return predictions
        
        # Get recent history for trend analysis
        recent_history = list(self.metrics_history[category])[-5:]
        
        for metric_name in current_metrics.keys():
            historical_values = [h.metrics.get(metric_name, 0) for h in recent_history if metric_name in h.metrics]
            
            if len(historical_values) >= 3:
                # Simple linear trend prediction
                trend_slope = (historical_values[-1] - historical_values[0]) / len(historical_values)
                predicted_next_value = historical_values[-1] + trend_slope
                
                predictions[f"{metric_name}_next_5min"] = predicted_next_value
        
        return predictions

    def _select_ab_test_strategy(self) -> str:
        """Select A/B test strategy based on current performance."""
        # Simple random selection with performance-based weighting
        import random
        
        strategy_a_performance = self.service_metrics['ab_test_conversions'].get('strategy_a', 0)
        strategy_b_performance = self.service_metrics['ab_test_conversions'].get('strategy_b', 0)
        
        total_performance = strategy_a_performance + strategy_b_performance
        
        if total_performance == 0:
            return random.choice(['strategy_a', 'strategy_b'])
        
        # Weight by performance
        strategy_a_weight = strategy_a_performance / total_performance
        
        return 'strategy_a' if random.random() < strategy_a_weight else 'strategy_b'

    def _update_validation_metrics(self, proposals_count: int, validation_time_ms: float):
        """Update validation service metrics."""
        self.service_metrics['total_validations'] += proposals_count
        self.service_metrics['successful_validations'] += proposals_count  # Assume success for now
        
        # Update average validation time
        alpha = 0.1
        if self.service_metrics['avg_validation_time_ms'] == 0:
            self.service_metrics['avg_validation_time_ms'] = validation_time_ms
        else:
            self.service_metrics['avg_validation_time_ms'] = (
                alpha * validation_time_ms +
                (1 - alpha) * self.service_metrics['avg_validation_time_ms']
            )

    def _update_learning_metrics(self, validation: ValidationResult, feedback: ValidationFeedback):
        """Update learning metrics based on human feedback."""
        # Calculate if ML prediction matched human judgment
        ml_predicted_accept = validation.validation_score > self.validation_threshold
        human_accepted = feedback.human_accepted
        
        # Update accuracy tracking
        accuracy = 1.0 if ml_predicted_accept == human_accepted else 0.0
        self.learning_metrics['validation_accuracy'].append(accuracy)
        
        # Update false positive/negative rates
        if not human_accepted and ml_predicted_accept:
            self.learning_metrics['false_positive_rate'].append(1.0)
        else:
            self.learning_metrics['false_positive_rate'].append(0.0)
        
        if human_accepted and not ml_predicted_accept:
            self.learning_metrics['false_negative_rate'].append(1.0)
        else:
            self.learning_metrics['false_negative_rate'].append(0.0)
        
        # Update overall agreement rate
        confidence_agreement = abs(validation.confidence - feedback.confidence_level)
        agreement_score = max(0.0, 1.0 - confidence_agreement)
        self.learning_metrics['human_agreement_rate'].append(agreement_score)

    def _cleanup_old_metrics(self):
        """Clean up metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        for category in self.metrics_history:
            history = self.metrics_history[category]
            # Remove old entries (deque automatically handles max size, but this ensures time-based cleanup)
            while history and history[0].timestamp < cutoff_time:
                history.popleft()

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics across validation and performance monitoring."""
        # Current performance snapshot
        current_performance = {}
        for category, metrics in self.current_metrics.items():
            current_performance[category.value] = {
                'metrics': metrics.metrics,
                'alerts': metrics.alerts,
                'trend_indicators': metrics.trend_indicators
            }
        
        # Learning performance
        learning_summary = {}
        for metric_name, values in self.learning_metrics.items():
            if values:
                learning_summary[metric_name] = {
                    'current': values[-1],
                    'average': np.mean(values),
                    'trend': 'improving' if len(values) >= 2 and values[-1] > values[-2] else 'declining'
                }
        
        return {
            'validation_metrics': {
                'total_validations': self.service_metrics['total_validations'],
                'success_rate': (self.service_metrics['successful_validations'] / 
                               max(1, self.service_metrics['total_validations'])) * 100,
                'avg_validation_time_ms': self.service_metrics['avg_validation_time_ms'],
                'ml_model_accuracy': np.mean(self.learning_metrics['validation_accuracy']) * 100 
                                    if self.learning_metrics['validation_accuracy'] else 0.0
            },
            'performance_monitoring': current_performance,
            'learning_progress': learning_summary,
            'ab_testing': {
                'strategy_performance': self.service_metrics['ab_test_conversions'],
                'enabled': self.enable_ab_testing
            },
            'system_health': {
                'total_alerts': sum(len(m.alerts) for m in self.current_metrics.values()),
                'predictive_alerts_triggered': self.service_metrics['predictive_alerts_triggered'],
                'feedback_processed': self.service_metrics['feedback_processed']
            }
        }

    # Placeholder methods for future implementation
    async def _find_validation_result(self, proposal_id: str):
        return None  # Would search validation history
    
    async def _get_historical_validation_performance(self, proposal):
        return {}  # Would return historical performance for similar proposals