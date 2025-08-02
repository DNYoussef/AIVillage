"""Base class for evolvable agents - Sprint 6 Enhanced"""

import asyncio
import pickle
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy torch for type hints
    class torch:
        class Tensor:
            pass

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PerformanceRecord:
    """Individual performance measurement"""
    timestamp: float
    task_type: str
    success: bool
    execution_time_ms: float
    accuracy: Optional[float] = None
    confidence: Optional[float] = None
    resource_usage: Optional[Dict[str, float]] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class EvolutionMemory:
    """Agent memory with compression and wisdom extraction"""
    short_term: List[Dict] = field(default_factory=list)  # Last 7 days
    long_term_compressed: bytes = b''  # LZ4 compressed historical data
    distilled_wisdom: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_lineage: List[str] = field(default_factory=list)
    breakthrough_contributions: List[Dict] = field(default_factory=list)
    
    def consolidate(self):
        """Move short-term to long-term with compression"""
        if len(self.short_term) > 10000:  # Threshold for consolidation
            # Move old entries to long-term
            to_compress = self.short_term[:-1000]
            self.short_term = self.short_term[-1000:]
            
            if LZ4_AVAILABLE:
                # Decompress existing, add new, recompress
                if self.long_term_compressed:
                    try:
                        existing = pickle.loads(lz4.frame.decompress(self.long_term_compressed))
                    except:
                        existing = []
                else:
                    existing = []
                    
                existing.extend(to_compress)
                self.long_term_compressed = lz4.frame.compress(pickle.dumps(existing))
            else:
                # Fallback without compression
                logger.warning("LZ4 not available, using uncompressed storage")
                if self.long_term_compressed:
                    existing = pickle.loads(self.long_term_compressed)
                else:
                    existing = []
                existing.extend(to_compress)
                self.long_term_compressed = pickle.dumps(existing)


class EvolutionMetrics:
    """Tracks evolution performance and trends"""
    
    def __init__(self):
        self.kpi_history: List[Tuple[float, Dict[str, float]]] = []
        self.improvement_events: List[Dict[str, Any]] = []
        self.regression_events: List[Dict[str, Any]] = []
        
    def record_kpi(self, kpis: Dict[str, float]):
        """Record KPI measurement"""
        self.kpi_history.append((time.time(), kpis.copy()))
        
        # Detect improvements/regressions
        if len(self.kpi_history) >= 2:
            prev_kpis = self.kpi_history[-2][1]
            current_kpis = self.kpi_history[-1][1]
            
            for metric, value in current_kpis.items():
                if metric in prev_kpis:
                    change = value - prev_kpis[metric]
                    if abs(change) > 0.05:  # 5% threshold
                        event = {
                            'timestamp': time.time(),
                            'metric': metric,
                            'change': change,
                            'old_value': prev_kpis[metric],
                            'new_value': value
                        }
                        
                        if change > 0:
                            self.improvement_events.append(event)
                        else:
                            self.regression_events.append(event)
    
    def get_trend(self, metric: str, window_hours: int = 24) -> Optional[float]:
        """Get trend for specific metric over time window"""
        cutoff = time.time() - (window_hours * 3600)
        recent_records = [(ts, kpis) for ts, kpis in self.kpi_history if ts > cutoff]
        
        if len(recent_records) < 2:
            return None
            
        values = [kpis.get(metric, 0) for _, kpis in recent_records]
        if len(values) < 2:
            return None
            
        # Simple linear trend
        x = list(range(len(values)))
        y = values
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope


class EvolvableAgent:
    """Enhanced agent with evolution capabilities for Sprint 6"""
    
    def __init__(self, agent_config: Dict):
        self.agent_id = agent_config.get('agent_id', str(uuid.uuid4()))
        self.agent_type = agent_config.get('agent_type', 'unknown')
        self.config = agent_config
        
        # Evolution components
        self.performance_history: List[PerformanceRecord] = []
        self.uncertainty_metrics: Dict[str, float] = {}
        self.evolution_memory = EvolutionMemory()
        self.evolution_metrics = EvolutionMetrics()
        
        # Encrypted thoughts (only King can decrypt)
        self.encrypted_thoughts: List[bytes] = []
        
        # Current parameters and state
        self.parameters: Dict[str, Any] = agent_config.get('parameters', {})
        self.prompts: Dict[str, str] = agent_config.get('prompts', {})
        self.learned_patterns: Dict[str, Any] = {}
        
        # Performance thresholds
        self.retirement_threshold = 0.4
        self.evolution_threshold = 0.6
        
        # Specialization
        self.specialization_domain = agent_config.get('specialization', 'general')
        self.expertise_areas: List[str] = agent_config.get('expertise', [])
        
        logger.info(f"Created evolvable agent {self.agent_id} of type {self.agent_type}")
        
    def record_performance(self, task_type: str, success: bool,
                          execution_time_ms: float, accuracy: Optional[float] = None,
                          confidence: Optional[float] = None,
                          resource_usage: Optional[Dict[str, float]] = None,
                          context: Optional[Dict[str, Any]] = None):
        """Record a performance measurement"""
        
        record = PerformanceRecord(
            timestamp=time.time(),
            task_type=task_type,
            success=success,
            execution_time_ms=execution_time_ms,
            accuracy=accuracy,
            confidence=confidence,
            resource_usage=resource_usage,
            context=context
        )
        
        self.performance_history.append(record)
        
        # Update uncertainty metrics based on confidence
        if confidence is not None:
            self.uncertainty_metrics[task_type] = 1.0 - confidence
            
        # Limit history size
        if len(self.performance_history) > 10000:
            self.performance_history = self.performance_history[-5000:]
            
    def evaluate_kpi(self) -> Dict[str, float]:
        """Calculate comprehensive KPI metrics"""
        if not self.performance_history:
            return {
                "performance": 0.7,
                "confidence": 0.5,
                "efficiency": 0.5,
                "reliability": 0.5,
                "adaptability": 0.5
            }
            
        recent = self.performance_history[-1000:]  # Last 1000 records
        
        # Success rate
        success_rate = sum(1 for r in recent if r.success) / len(recent)
        
        # Average execution time (normalized)
        avg_time = sum(r.execution_time_ms for r in recent) / len(recent)
        time_efficiency = max(0, 1 - min(avg_time / 5000, 1))  # 5s baseline
        
        # Accuracy (if available)
        accuracy_records = [r.accuracy for r in recent if r.accuracy is not None]
        avg_accuracy = sum(accuracy_records) / len(accuracy_records) if accuracy_records else 0.7
        
        # Confidence/Uncertainty
        uncertainty = np.mean(list(self.uncertainty_metrics.values())) if self.uncertainty_metrics else 0.5
        confidence = 1 - uncertainty
        
        # Reliability (consistency over time)
        if len(recent) >= 10:
            # Calculate variance in success rate over time windows
            windows = [recent[i:i+10] for i in range(0, len(recent)-9, 10)]
            window_success_rates = [sum(1 for r in w if r.success) / len(w) for w in windows]
            reliability = 1 - (np.std(window_success_rates) if len(window_success_rates) > 1 else 0)
        else:
            reliability = success_rate
            
        # Adaptability (improvement over time)
        adaptability = 0.5  # Default
        if len(recent) >= 50:
            early_success = sum(1 for r in recent[:25] if r.success) / 25
            late_success = sum(1 for r in recent[-25:] if r.success) / 25
            adaptability = min(1.0, 0.5 + (late_success - early_success))
            
        # Composite performance score
        performance = (
            success_rate * 0.3 +
            avg_accuracy * 0.25 +
            time_efficiency * 0.2 +
            confidence * 0.15 +
            reliability * 0.1
        )
        
        kpis = {
            "performance": performance,
            "success_rate": success_rate,
            "accuracy": avg_accuracy,
            "confidence": confidence,
            "efficiency": time_efficiency,
            "reliability": reliability,
            "adaptability": adaptability,
            "avg_execution_time_ms": avg_time
        }
        
        # Record in evolution metrics
        self.evolution_metrics.record_kpi(kpis)
        
        return kpis
        
    async def reflect_on_performance(self) -> Dict[str, Any]:
        """Daily performance reflection with deep analysis"""
        daily_stats = self._calculate_daily_stats()
        insights = await self._extract_insights(daily_stats)
        
        reflection = {
            "date": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "generation": self.evolution_memory.generation,
            "daily_stats": daily_stats,
            "insights": insights,
            "areas_for_improvement": await self._identify_weak_spots(),
            "successful_patterns": await self._identify_strong_patterns(),
            "learning_opportunities": await self._identify_learning_opportunities(),
            "parameter_suggestions": await self._suggest_parameter_adjustments()
        }
        
        # Add to evolution memory
        self.evolution_memory.short_term.append(reflection)
        self.evolution_memory.consolidate()
        
        return reflection
        
    def _calculate_daily_stats(self) -> Dict[str, Any]:
        """Calculate statistics for the last 24 hours"""
        cutoff = time.time() - 86400  # 24 hours
        daily_records = [r for r in self.performance_history if r.timestamp > cutoff]
        
        if not daily_records:
            return {"record_count": 0}
            
        stats = {
            "record_count": len(daily_records),
            "success_count": sum(1 for r in daily_records if r.success),
            "failure_count": sum(1 for r in daily_records if not r.success),
            "avg_execution_time": sum(r.execution_time_ms for r in daily_records) / len(daily_records),
            "task_types": list(set(r.task_type for r in daily_records)),
            "peak_performance_hour": self._find_peak_performance_time(daily_records),
            "resource_usage": self._aggregate_resource_usage(daily_records)
        }
        
        return stats
        
    async def _extract_insights(self, daily_stats: Dict[str, Any]) -> List[str]:
        """Extract actionable insights from daily statistics"""
        insights = []
        
        if daily_stats.get("record_count", 0) > 0:
            success_rate = daily_stats["success_count"] / daily_stats["record_count"]
            
            if success_rate < 0.5:
                insights.append("Performance below threshold - need intervention")
            elif success_rate > 0.9:
                insights.append("Excellent performance - good candidate for knowledge transfer")
                
            if daily_stats.get("avg_execution_time", 0) > 3000:  # 3 seconds
                insights.append("Slow execution times - optimization needed")
                
            # Task-specific insights
            for task_type in daily_stats.get("task_types", []):
                task_records = [r for r in self.performance_history[-1000:]
                               if r.task_type == task_type]
                if len(task_records) >= 10:
                    task_success_rate = sum(1 for r in task_records if r.success) / len(task_records)
                    if task_success_rate < 0.6:
                        insights.append(f"Struggling with {task_type} tasks")
                    elif task_success_rate > 0.85:
                        insights.append(f"Expert level performance on {task_type} tasks")
                        
        return insights
        
    async def _identify_weak_spots(self) -> List[Dict[str, Any]]:
        """Identify areas needing improvement"""
        weak_spots = []
        
        # Task type analysis
        task_performance = {}
        for record in self.performance_history[-1000:]:
            if record.task_type not in task_performance:
                task_performance[record.task_type] = []
            task_performance[record.task_type].append(record.success)
            
        for task_type, successes in task_performance.items():
            if len(successes) >= 5:
                success_rate = sum(successes) / len(successes)
                if success_rate < 0.6:
                    weak_spots.append({
                        "area": task_type,
                        "success_rate": success_rate,
                        "sample_size": len(successes),
                        "priority": "high" if success_rate < 0.4 else "medium"
                    })
                    
        # Uncertainty analysis
        for domain, uncertainty in self.uncertainty_metrics.items():
            if uncertainty > 0.5:
                weak_spots.append({
                    "area": f"confidence_in_{domain}",
                    "uncertainty": uncertainty,
                    "priority": "medium"
                })
                
        return weak_spots
        
    async def _identify_strong_patterns(self) -> List[Dict[str, Any]]:
        """Identify successful patterns for replication"""
        strong_patterns = []
        
        # High-performance contexts
        successful_records = [r for r in self.performance_history[-1000:] if r.success]
        
        if len(successful_records) >= 10:
            # Analyze contexts of successful operations
            contexts = [r.context for r in successful_records if r.context]
            
            if contexts:
                # Find common patterns in successful contexts
                # This is simplified - in practice would use more sophisticated pattern detection
                common_keys = set()
                for context in contexts:
                    if isinstance(context, dict):
                        common_keys.update(context.keys())
                        
                for key in common_keys:
                    values = [ctx.get(key) for ctx in contexts if isinstance(ctx, dict) and key in ctx]
                    if len(values) >= 5:
                        strong_patterns.append({
                            "pattern_type": "context_feature",
                            "feature": key,
                            "success_correlation": len(values) / len(contexts),
                            "sample_values": list(set(values))[:5]
                        })
                        
        return strong_patterns
        
    async def _identify_learning_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for learning and improvement"""
        opportunities = []
        
        # Recent failures analysis
        recent_failures = [r for r in self.performance_history[-500:] if not r.success]
        
        if recent_failures:
            # Group failures by type
            failure_types = {}
            for failure in recent_failures:
                task_type = failure.task_type
                if task_type not in failure_types:
                    failure_types[task_type] = []
                failure_types[task_type].append(failure)
                
            for task_type, failures in failure_types.items():
                if len(failures) >= 3:
                    opportunities.append({
                        "type": "failure_pattern_analysis",
                        "task_type": task_type,
                        "failure_count": len(failures),
                        "avg_execution_time": sum(f.execution_time_ms for f in failures) / len(failures),
                        "priority": "high"
                    })
                    
        # Performance stagnation detection
        kpi_trend = self.evolution_metrics.get_trend("performance", window_hours=48)
        if kpi_trend is not None and abs(kpi_trend) < 0.01:
            opportunities.append({
                "type": "performance_stagnation",
                "trend": kpi_trend,
                "suggestion": "experiment_with_parameters",
                "priority": "medium"
            })
            
        return opportunities
        
    async def _suggest_parameter_adjustments(self) -> Dict[str, Any]:
        """Suggest parameter adjustments based on performance"""
        suggestions = {}
        
        current_kpis = self.evaluate_kpi()
        
        # Learning rate adjustments
        if current_kpis.get("adaptability", 0.5) < 0.4:
            suggestions["learning_rate"] = "increase"
        elif current_kpis.get("reliability", 0.5) < 0.6:
            suggestions["learning_rate"] = "decrease"
            
        # Confidence threshold adjustments
        if current_kpis.get("confidence", 0.5) < 0.4:
            suggestions["confidence_threshold"] = "lower"
        elif current_kpis.get("accuracy", 0.7) < 0.6:
            suggestions["confidence_threshold"] = "raise"
            
        return suggestions
        
    def curate_training_experiences(self) -> Dict[str, List[Dict]]:
        """Select best examples for successor training"""
        # High-performance examples
        successful_records = sorted(
            [r for r in self.performance_history if r.success],
            key=lambda r: (r.accuracy or 0.5) * (r.confidence or 0.5),
            reverse=True
        )[:1000]
        
        # Diverse failure cases for robustness
        failure_records = [r for r in self.performance_history if not r.success]
        
        # Edge cases (high uncertainty but successful)
        edge_cases = [
            r for r in successful_records
            if r.confidence is not None and r.confidence < 0.6
        ][:100]
        
        return {
            "high_performance": [self._serialize_record(r) for r in successful_records],
            "failure_analysis": [self._serialize_record(r) for r in failure_records[-200:]],
            "edge_cases": [self._serialize_record(r) for r in edge_cases],
            "learning_examples": [self._serialize_record(r) for r in self.performance_history[-500:]]
        }
        
    def _serialize_record(self, record: PerformanceRecord) -> Dict[str, Any]:
        """Convert performance record to serializable format"""
        return {
            "timestamp": record.timestamp,
            "task_type": record.task_type,
            "success": record.success,
            "execution_time_ms": record.execution_time_ms,
            "accuracy": record.accuracy,
            "confidence": record.confidence,
            "resource_usage": record.resource_usage,
            "context": record.context
        }
        
    def design_successor_prompts(self) -> Dict[str, str]:
        """Create optimized prompts for next generation"""
        current_prompts = self.prompts.copy()
        performance_by_context = self._analyze_prompt_effectiveness()
        
        improved_prompts = {}
        for prompt_name, prompt_text in current_prompts.items():
            effectiveness = performance_by_context.get(prompt_name, 0.5)
            
            if effectiveness < 0.6:
                # This prompt needs improvement
                improved_prompts[prompt_name] = self._improve_prompt(
                    prompt_text,
                    self.evolution_memory.distilled_wisdom
                )
            else:
                improved_prompts[prompt_name] = prompt_text
                
        return improved_prompts
        
    def _analyze_prompt_effectiveness(self) -> Dict[str, float]:
        """Analyze which prompts/contexts lead to better performance"""
        # Simplified analysis - in practice would be more sophisticated
        effectiveness = {}
        
        for prompt_name in self.prompts.keys():
            # Find records that used this prompt (from context)
            relevant_records = [
                r for r in self.performance_history[-1000:]
                if r.context and r.context.get("prompt_used") == prompt_name
            ]
            
            if relevant_records:
                success_rate = sum(1 for r in relevant_records if r.success) / len(relevant_records)
                effectiveness[prompt_name] = success_rate
            else:
                effectiveness[prompt_name] = 0.5  # Default
                
        return effectiveness
        
    def _improve_prompt(self, prompt_text: str, wisdom: Dict[str, Any]) -> str:
        """Improve prompt based on learned wisdom"""
        # This would use the wisdom to enhance prompts
        # For now, just return the original with a note
        improvements = wisdom.get("prompt_improvements", [])
        
        if improvements:
            # Apply the first applicable improvement
            for improvement in improvements:
                if improvement.get("applicable_to", "") in prompt_text:
                    return prompt_text + "\n\n" + improvement.get("enhancement", "")
                    
        return prompt_text
        
    def distill_wisdom(self) -> Dict[str, Any]:
        """Extract key learnings for transfer to successors"""
        wisdom = {
            "agent_id": self.agent_id,
            "generation": self.evolution_memory.generation,
            "specialization": self.specialization_domain,
            "extraction_timestamp": time.time()
        }
        
        # Extract successful strategies
        wisdom["successful_strategies"] = self._extract_successful_strategies()
        
        # Extract failure patterns to avoid
        wisdom["failure_patterns"] = self._extract_failure_patterns()
        
        # Extract optimal parameters
        wisdom["optimal_parameters"] = self._extract_optimal_parameters()
        
        # Extract domain knowledge
        wisdom["domain_knowledge"] = self._extract_domain_knowledge()
        
        # Extract learned patterns
        wisdom["learned_patterns"] = self.learned_patterns.copy()
        
        return wisdom
        
    def _extract_successful_strategies(self) -> List[Dict[str, Any]]:
        """Extract patterns from successful operations"""
        strategies = []
        
        # Analyze high-performance periods
        high_perf_records = [
            r for r in self.performance_history[-2000:]
            if r.success and (r.accuracy or 0.7) > 0.8
        ]
        
        if len(high_perf_records) >= 10:
            # Group by task type
            by_task = {}
            for record in high_perf_records:
                if record.task_type not in by_task:
                    by_task[record.task_type] = []
                by_task[record.task_type].append(record)
                
            for task_type, records in by_task.items():
                if len(records) >= 5:
                    strategy = {
                        "task_type": task_type,
                        "success_count": len(records),
                        "avg_execution_time": sum(r.execution_time_ms for r in records) / len(records),
                        "avg_accuracy": sum(r.accuracy or 0.7 for r in records) / len(records),
                        "common_contexts": self._find_common_context_features(records)
                    }
                    strategies.append(strategy)
                    
        return strategies
        
    def _extract_failure_patterns(self) -> List[Dict[str, Any]]:
        """Extract patterns to avoid from failures"""
        patterns = []
        
        recent_failures = [r for r in self.performance_history[-1000:] if not r.success]
        
        if len(recent_failures) >= 5:
            # Group by task type
            by_task = {}
            for record in recent_failures:
                if record.task_type not in by_task:
                    by_task[record.task_type] = []
                by_task[record.task_type].append(record)
                
            for task_type, records in by_task.items():
                if len(records) >= 3:
                    pattern = {
                        "task_type": task_type,
                        "failure_count": len(records),
                        "avg_execution_time": sum(r.execution_time_ms for r in records) / len(records),
                        "common_contexts": self._find_common_context_features(records),
                        "avoidance_strategy": f"Avoid {task_type} tasks with these contexts"
                    }
                    patterns.append(pattern)
                    
        return patterns
        
    def _extract_optimal_parameters(self) -> Dict[str, Any]:
        """Identify optimal parameter settings"""
        # This would analyze performance vs parameters over time
        # For now, return current parameters of best-performing periods
        best_kpis = self.evaluate_kpi()
        
        return {
            "parameters": self.parameters.copy(),
            "performance_at_extraction": best_kpis,
            "confidence": 0.7  # Would be calculated based on data stability
        }
        
    def _extract_domain_knowledge(self) -> Dict[str, Any]:
        """Extract domain-specific insights"""
        domain_knowledge = {
            "specialization": self.specialization_domain,
            "expertise_areas": self.expertise_areas.copy(),
            "task_type_preferences": {},
            "context_insights": {}
        }
        
        # Analyze task type performance
        task_performance = {}
        for record in self.performance_history[-2000:]:
            if record.task_type not in task_performance:
                task_performance[record.task_type] = []
            task_performance[record.task_type].append(record.success)
            
        for task_type, successes in task_performance.items():
            if len(successes) >= 10:
                success_rate = sum(successes) / len(successes)
                domain_knowledge["task_type_preferences"][task_type] = {
                    "success_rate": success_rate,
                    "confidence": min(1.0, len(successes) / 100),
                    "sample_size": len(successes)
                }
                
        return domain_knowledge
        
    def _find_common_context_features(self, records: List[PerformanceRecord]) -> Dict[str, Any]:
        """Find common features in record contexts"""
        common_features = {}
        
        contexts = [r.context for r in records if r.context]
        if not contexts:
            return common_features
            
        # Find keys that appear in most contexts
        all_keys = set()
        for context in contexts:
            if isinstance(context, dict):
                all_keys.update(context.keys())
                
        for key in all_keys:
            values = []
            for context in contexts:
                if isinstance(context, dict) and key in context:
                    values.append(context[key])
                    
            if len(values) >= len(contexts) * 0.7:  # Appears in 70% of contexts
                common_features[key] = {
                    "frequency": len(values) / len(contexts),
                    "sample_values": list(set(str(v) for v in values))[:5]
                }
                
        return common_features
        
    def _find_peak_performance_time(self, records: List[PerformanceRecord]) -> Optional[int]:
        """Find hour of day with best performance"""
        if not records:
            return None
            
        hourly_performance = {}
        for record in records:
            hour = datetime.fromtimestamp(record.timestamp).hour
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            hourly_performance[hour].append(record.success)
            
        if not hourly_performance:
            return None
            
        best_hour = max(
            hourly_performance.keys(),
            key=lambda h: sum(hourly_performance[h]) / len(hourly_performance[h])
        )
        
        return best_hour
        
    def _aggregate_resource_usage(self, records: List[PerformanceRecord]) -> Dict[str, float]:
        """Aggregate resource usage from records"""
        usage_records = [r.resource_usage for r in records if r.resource_usage]
        
        if not usage_records:
            return {}
            
        aggregated = {}
        for usage in usage_records:
            if isinstance(usage, dict):
                for resource, value in usage.items():
                    if resource not in aggregated:
                        aggregated[resource] = []
                    aggregated[resource].append(value)
                    
        # Calculate averages
        result = {}
        for resource, values in aggregated.items():
            result[f"avg_{resource}"] = sum(values) / len(values)
            result[f"max_{resource}"] = max(values)
            
        return result
        
    def should_retire(self) -> bool:
        """Determine if agent should be retired"""
        kpis = self.evaluate_kpi()
        
        # Multiple criteria for retirement
        performance_low = kpis.get("performance", 0.5) < self.retirement_threshold
        reliability_low = kpis.get("reliability", 0.5) < 0.3
        stagnation = self.evolution_metrics.get_trend("performance", 72) or 0 < -0.01
        
        return performance_low and (reliability_low or stagnation)
        
    def needs_evolution(self) -> bool:
        """Determine if agent needs evolutionary improvement"""
        kpis = self.evaluate_kpi()
        
        # Evolution needed if performance is below threshold
        # but agent is not ready for retirement
        performance_needs_help = kpis.get("performance", 0.5) < self.evolution_threshold
        not_hopeless = kpis.get("performance", 0.5) > self.retirement_threshold
        
        return performance_needs_help and not_hopeless
        
    def get_evolution_readiness_score(self) -> float:
        """Get score indicating readiness for evolution"""
        kpis = self.evaluate_kpi()
        
        # Factors that indicate readiness for evolution
        performance_gap = max(0, self.evolution_threshold - kpis.get("performance", 0.5))
        data_sufficiency = min(1.0, len(self.performance_history) / 1000)
        stability = kpis.get("reliability", 0.5)
        
        readiness = (performance_gap * 0.5 +
                    data_sufficiency * 0.3 +
                    (1 - stability) * 0.2)
                    
        return min(1.0, readiness)
        
    def export_state(self) -> Dict[str, Any]:
        """Export agent state for serialization"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "config": self.config,
            "parameters": self.parameters,
            "prompts": self.prompts,
            "evolution_memory": {
                "generation": self.evolution_memory.generation,
                "parent_lineage": self.evolution_memory.parent_lineage,
                "distilled_wisdom": self.evolution_memory.distilled_wisdom
            },
            "specialization_domain": self.specialization_domain,
            "expertise_areas": self.expertise_areas,
            "current_kpis": self.evaluate_kpi(),
            "export_timestamp": time.time()
        }
        
    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'EvolvableAgent':
        """Create agent from exported state"""
        agent = cls(state["config"])
        agent.agent_id = state["agent_id"]
        agent.parameters = state["parameters"]
        agent.prompts = state["prompts"]
        agent.specialization_domain = state["specialization_domain"]
        agent.expertise_areas = state["expertise_areas"]
        
        # Restore evolution memory
        memory_state = state.get("evolution_memory", {})
        agent.evolution_memory.generation = memory_state.get("generation", 0)
        agent.evolution_memory.parent_lineage = memory_state.get("parent_lineage", [])
        agent.evolution_memory.distilled_wisdom = memory_state.get("distilled_wisdom", {})
        
        return agent

