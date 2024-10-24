"""Analysis system for MAGI feedback."""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import logging
import json

from ..core.exceptions import ToolError
from ..utils.logging import setup_logger
from ..techniques.base import TechniqueResult

logger = setup_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for analysis."""
    success_rate: float
    average_execution_time: float
    error_rate: float
    confidence_score: float
    usage_frequency: float

@dataclass
class TechniqueMetrics:
    """Metrics for reasoning technique analysis."""
    technique_name: str
    success_rate: float
    average_confidence: float
    average_execution_time: float
    usage_count: int
    common_patterns: List[str]

@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    average_response_time: float
    peak_memory_usage: float
    active_techniques: List[str]
    active_tools: List[str]

class FeedbackAnalyzer:
    """
    Analyzes system feedback and performance.
    
    Responsibilities:
    - Performance analysis
    - Pattern recognition
    - Trend analysis
    - Root cause analysis
    - Success prediction
    """
    
    def __init__(self):
        """Initialize feedback analyzer."""
        self.technique_history: Dict[str, List[TechniqueResult]] = defaultdict(list)
        self.tool_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.system_metrics: List[SystemMetrics] = []
        self.error_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def analyze_technique_performance(
        self,
        technique_name: str,
        time_window: Optional[timedelta] = None
    ) -> TechniqueMetrics:
        """
        Analyze performance of a specific technique.
        
        Args:
            technique_name: Name of technique to analyze
            time_window: Time window for analysis (optional)
            
        Returns:
            Performance metrics for the technique
        """
        history = self.technique_history[technique_name]
        if not history:
            raise ToolError(f"No history found for technique '{technique_name}'")
        
        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - time_window
            history = [
                result for result in history
                if result.metadata.get('timestamp', datetime.now()) > cutoff
            ]
        
        # Calculate metrics
        success_count = sum(1 for result in history if not result.metadata.get('error'))
        success_rate = success_count / len(history) if history else 0
        
        confidences = [result.confidence for result in history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        execution_times = [
            result.metadata.get('execution_time', 0)
            for result in history
        ]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Identify common patterns
        patterns = self._identify_patterns(history)
        
        return TechniqueMetrics(
            technique_name=technique_name,
            success_rate=success_rate,
            average_confidence=avg_confidence,
            average_execution_time=avg_execution_time,
            usage_count=len(history),
            common_patterns=patterns
        )
    
    def analyze_tool_performance(
        self,
        tool_name: str,
        time_window: Optional[timedelta] = None
    ) -> PerformanceMetrics:
        """
        Analyze performance of a specific tool.
        
        Args:
            tool_name: Name of tool to analyze
            time_window: Time window for analysis (optional)
            
        Returns:
            Performance metrics for the tool
        """
        history = self.tool_history[tool_name]
        if not history:
            raise ToolError(f"No history found for tool '{tool_name}'")
        
        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - time_window
            history = [
                record for record in history
                if record.get('timestamp', datetime.now()) > cutoff
            ]
        
        # Calculate metrics
        success_count = sum(1 for record in history if record.get('success', False))
        success_rate = success_count / len(history) if history else 0
        
        error_count = sum(1 for record in history if record.get('error'))
        error_rate = error_count / len(history) if history else 0
        
        execution_times = [
            record.get('execution_time', 0)
            for record in history
        ]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Calculate confidence score based on multiple factors
        confidence_factors = [
            success_rate,
            1 - error_rate,
            min(1.0, 10.0 / avg_execution_time) if avg_execution_time > 0 else 0
        ]
        confidence_score = sum(confidence_factors) / len(confidence_factors)
        
        # Calculate usage frequency (executions per day)
        if time_window:
            days = time_window.days or 1
        else:
            days = (datetime.now() - min(
                record.get('timestamp', datetime.now())
                for record in history
            )).days or 1
        usage_frequency = len(history) / days
        
        return PerformanceMetrics(
            success_rate=success_rate,
            average_execution_time=avg_execution_time,
            error_rate=error_rate,
            confidence_score=confidence_score,
            usage_frequency=usage_frequency
        )
    
    def analyze_system_performance(
        self,
        time_window: Optional[timedelta] = None
    ) -> SystemMetrics:
        """
        Analyze overall system performance.
        
        Args:
            time_window: Time window for analysis (optional)
            
        Returns:
            System-wide performance metrics
        """
        metrics = self.system_metrics
        if not metrics:
            raise ToolError("No system metrics available")
        
        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - time_window
            metrics = [
                m for m in metrics
                if getattr(m, 'timestamp', datetime.now()) > cutoff
            ]
        
        if not metrics:
            raise ToolError("No metrics found in specified time window")
        
        # Calculate aggregated metrics
        total_tasks = sum(m.total_tasks for m in metrics)
        successful_tasks = sum(m.successful_tasks for m in metrics)
        failed_tasks = sum(m.failed_tasks for m in metrics)
        
        response_times = [m.average_response_time for m in metrics]
        avg_response_time = sum(response_times) / len(response_times)
        
        peak_memory = max(m.peak_memory_usage for m in metrics)
        
        # Get currently active components
        latest = metrics[-1]
        active_techniques = latest.active_techniques
        active_tools = latest.active_tools
        
        return SystemMetrics(
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            average_response_time=avg_response_time,
            peak_memory_usage=peak_memory,
            active_techniques=active_techniques,
            active_tools=active_tools
        )
    
    def analyze_error_patterns(
        self,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Analyze error patterns across the system.
        
        Args:
            time_window: Time window for analysis (optional)
            
        Returns:
            Analysis of error patterns
        """
        patterns = self.error_patterns
        if not patterns:
            return {"error": "No error patterns recorded"}
        
        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - time_window
            patterns = {
                category: [
                    error for error in errors
                    if error.get('timestamp', datetime.now()) > cutoff
                ]
                for category, errors in patterns.items()
            }
        
        # Analyze patterns
        analysis = {
            'total_errors': sum(len(errors) for errors in patterns.values()),
            'error_categories': {},
            'common_causes': [],
            'trends': {},
            'recommendations': []
        }
        
        # Analyze each error category
        for category, errors in patterns.items():
            if not errors:
                continue
                
            category_analysis = {
                'count': len(errors),
                'frequency': len(errors) / (time_window.total_seconds() if time_window else 1),
                'common_contexts': self._analyze_error_contexts(errors),
                'severity_distribution': self._analyze_error_severity(errors)
            }
            analysis['error_categories'][category] = category_analysis
        
        # Identify common causes
        all_errors = [
            error for errors in patterns.values()
            for error in errors
        ]
        analysis['common_causes'] = self._identify_common_causes(all_errors)
        
        # Analyze trends
        analysis['trends'] = self._analyze_error_trends(all_errors)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_error_recommendations(analysis)
        
        return analysis
    
    def _identify_patterns(self, history: List[TechniqueResult]) -> List[str]:
        """Identify common patterns in technique results."""
        patterns = []
        
        # Analyze thought processes
        thoughts = [result.thought for result in history]
        common_thoughts = self._find_common_subsequences(thoughts)
        if common_thoughts:
            patterns.extend([f"Common thought pattern: {p}" for p in common_thoughts])
        
        # Analyze confidence patterns
        confidences = [result.confidence for result in history]
        confidence_pattern = self._analyze_confidence_pattern(confidences)
        if confidence_pattern:
            patterns.append(f"Confidence pattern: {confidence_pattern}")
        
        # Analyze metadata patterns
        metadata_patterns = self._analyze_metadata_patterns(
            [result.metadata for result in history]
        )
        patterns.extend(metadata_patterns)
        
        return patterns
    
    def _find_common_subsequences(self, sequences: List[str]) -> List[str]:
        """Find common subsequences in a list of strings."""
        if not sequences:
            return []
        
        # Convert strings to words
        word_sequences = [seq.split() for seq in sequences]
        
        # Find common subsequences of at least 3 words
        common = []
        min_length = 3
        
        for i in range(len(word_sequences[0])):
            for length in range(min_length, len(word_sequences[0]) - i + 1):
                subsequence = word_sequences[0][i:i+length]
                
                # Check if subsequence appears in all sequences
                if all(self._contains_subsequence(seq, subsequence) for seq in word_sequences[1:]):
                    common.append(' '.join(subsequence))
        
        return common[:5]  # Return top 5 common subsequences
    
    def _contains_subsequence(self, sequence: List[str], subsequence: List[str]) -> bool:
        """Check if a sequence contains a subsequence."""
        n, m = len(sequence), len(subsequence)
        for i in range(n - m + 1):
            if sequence[i:i+m] == subsequence:
                return True
        return False
    
    def _analyze_confidence_pattern(self, confidences: List[float]) -> Optional[str]:
        """Analyze pattern in confidence scores."""
        if not confidences:
            return None
        
        # Calculate trend
        if len(confidences) > 1:
            slope = np.polyfit(range(len(confidences)), confidences, 1)[0]
            if abs(slope) < 0.01:
                return "Stable confidence"
            elif slope > 0:
                return "Increasing confidence trend"
            else:
                return "Decreasing confidence trend"
        
        return None
    
    def _analyze_metadata_patterns(
        self,
        metadata_list: List[Dict[str, Any]]
    ) -> List[str]:
        """Analyze patterns in result metadata."""
        patterns = []
        
        if not metadata_list:
            return patterns
        
        # Analyze execution times
        execution_times = [
            m.get('execution_time', 0)
            for m in metadata_list
        ]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            std_time = np.std(execution_times)
            if std_time / avg_time > 0.5:
                patterns.append("High execution time variability")
            else:
                patterns.append("Stable execution time")
        
        # Analyze error patterns
        errors = [
            m.get('error')
            for m in metadata_list
            if m.get('error')
        ]
        if errors:
            error_types = set(type(e).__name__ for e in errors)
            if len(error_types) == 1:
                patterns.append(f"Consistent error type: {error_types.pop()}")
            elif len(error_types) > 1:
                patterns.append(f"Multiple error types: {', '.join(error_types)}")
        
        return patterns
    
    def _analyze_error_contexts(
        self,
        errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze common contexts in which errors occur."""
        contexts = defaultdict(int)
        
        for error in errors:
            context = error.get('context', {})
            context_key = json.dumps(context, sort_keys=True)
            contexts[context_key] += 1
        
        # Return top 5 most common contexts
        return [
            {
                'context': json.loads(context_key),
                'frequency': count
            }
            for context_key, count in sorted(
                contexts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        ]
    
    def _analyze_error_severity(
        self,
        errors: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Analyze distribution of error severities."""
        severities = defaultdict(int)
        
        for error in errors:
            severity = error.get('severity', 'unknown')
            severities[severity] += 1
        
        return dict(severities)
    
    def _identify_common_causes(
        self,
        errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify common causes of errors."""
        causes = defaultdict(list)
        
        for error in errors:
            cause = error.get('root_cause', 'unknown')
            causes[cause].append(error)
        
        # Return top 5 most common causes with examples
        return [
            {
                'cause': cause,
                'frequency': len(errors),
                'examples': errors[:3]  # Include up to 3 examples
            }
            for cause, errors in sorted(
                causes.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:5]
        ]
    
    def _analyze_error_trends(
        self,
        errors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze trends in error occurrence."""
        if not errors:
            return {}
        
        # Sort errors by timestamp
        errors = sorted(
            errors,
            key=lambda x: x.get('timestamp', datetime.now())
        )
        
        # Analyze daily error counts
        daily_counts = defaultdict(int)
        for error in errors:
            date = error.get('timestamp', datetime.now()).date()
            daily_counts[date] += 1
        
        # Calculate trend
        dates = sorted(daily_counts.keys())
        counts = [daily_counts[date] for date in dates]
        
        if len(counts) > 1:
            slope = np.polyfit(range(len(counts)), counts, 1)[0]
            trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        else:
            trend = "insufficient data"
        
        return {
            'trend': trend,
            'daily_counts': dict(daily_counts),
            'total_days': len(dates),
            'average_daily_errors': sum(counts) / len(counts) if counts else 0
        }
    
    def _generate_error_recommendations(
        self,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []
        
        # Check error frequency
        total_errors = analysis['total_errors']
        if total_errors > 100:
            recommendations.append(
                "High error volume detected. Consider implementing automated error handling."
            )
        
        # Check error categories
        categories = analysis['error_categories']
        for category, data in categories.items():
            if data['frequency'] > 0.1:  # More than 1 error per 10 seconds
                recommendations.append(
                    f"Frequent {category} errors detected. "
                    f"Review error handling for this category."
                )
        
        # Check trends
        trends = analysis['trends']
        if trends.get('trend') == "increasing":
            recommendations.append(
                "Error frequency is increasing. "
                "Consider conducting a thorough system review."
            )
        
        # Add specific recommendations based on common causes
        for cause in analysis['common_causes']:
            recommendations.append(
                f"Address common error cause: {cause['cause']} "
                f"(frequency: {cause['frequency']})"
            )
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = FeedbackAnalyzer()
    
    # Analyze technique performance
    try:
        metrics = analyzer.analyze_technique_performance(
            "chain_of_thought",
            time_window=timedelta(days=7)
        )
        print(f"Technique metrics: {metrics}")
    except ToolError as e:
        print(f"Error: {e}")
    
    # Analyze system performance
    try:
        system_metrics = analyzer.analyze_system_performance(
            time_window=timedelta(days=7)
        )
        print(f"System metrics: {system_metrics}")
    except ToolError as e:
        print(f"Error: {e}")
    
    # Analyze error patterns
    error_analysis = analyzer.analyze_error_patterns(
        time_window=timedelta(days=7)
    )
    print(f"Error analysis: {json.dumps(error_analysis, indent=2)}")
