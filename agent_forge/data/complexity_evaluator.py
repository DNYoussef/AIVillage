"""Enhanced complexity evaluation system with adaptive thresholds."""

import logging
import json
import sqlite3
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict
from config.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

class ComplexityEvaluator:
    """
    Enhanced system for evaluating task complexity and managing thresholds
    for when to use frontier vs local models.
    """
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        """
        Initialize ComplexityEvaluator.
        
        Args:
            config: Optional unified configuration instance
        """
        # Initialize default thresholds and weights if no config provided
        self.thresholds = {
            "king": 0.6,
            "sage": 0.7,
            "magi": 0.65
        }
        
        self.token_weights = {
            "count": 0.3,
            "indicators": 0.4,
            "semantic": 0.3
        }
        
        self.performance_thresholds = {
            "king": 0.7,
            "sage": 0.75,
            "magi": 0.8
        }
        
        if config and hasattr(config, 'config') and 'complexity' in config.config:
            complexity_config = config.config['complexity']
            if 'thresholds' in complexity_config:
                self.thresholds.update(complexity_config['thresholds'])
            if 'token_weights' in complexity_config:
                self.token_weights.update(complexity_config['token_weights'])
            if 'performance_thresholds' in complexity_config:
                self.performance_thresholds.update(complexity_config['performance_thresholds'])
        
        # Initialize complexity indicators for each agent type
        self.complexity_indicators = {
            "king": [
                "analyze", "evaluate", "compare", "synthesize", "design",
                "optimize", "recommend", "strategic", "complex", "multi-step",
                "coordinate", "plan", "manage", "oversee", "decide",
                "prioritize", "allocate", "balance", "assess", "judge"
            ],
            "sage": [
                "research", "analyze", "synthesize", "evaluate", "implications",
                "relationship", "impact", "trends", "patterns", "framework",
                "investigate", "study", "examine", "correlate", "hypothesize",
                "theorize", "model", "predict", "infer", "deduce"
            ],
            "magi": [
                "optimize", "implement", "design", "architecture", "system",
                "algorithm", "performance", "scale", "concurrent", "async",
                "refactor", "debug", "test", "deploy", "integrate",
                "secure", "validate", "benchmark", "profile", "optimize"
            ]
        }
        
        # Initialize performance history
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Initialize adaptive thresholds
        self.adaptive_thresholds = {
            agent_type: {
                'base': threshold,
                'min': max(0.3, threshold - 0.2),
                'max': min(0.9, threshold + 0.2),
                'adjustment_rate': 0.05
            }
            for agent_type, threshold in self.thresholds.items()
        }
        
        logger.info("Initialized ComplexityEvaluator with adaptive thresholds")
    
    def evaluate_complexity(self, 
                          agent_type: str, 
                          task: str,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate task complexity with enhanced analysis.
        
        Args:
            agent_type: Type of agent ("king", "sage", or "magi")
            task: The task to evaluate
            context: Optional context information
            
        Returns:
            Dictionary containing complexity analysis
        """
        if agent_type not in self.thresholds:
            raise ValueError(f"Invalid agent type: {agent_type}")
        
        # Calculate component scores
        token_score = self._calculate_token_complexity(task)
        indicator_score = self._calculate_indicator_complexity(agent_type, task)
        semantic_score = self._calculate_semantic_complexity(task, context)
        structural_score = self._calculate_structural_complexity(task)
        
        # Calculate weighted complexity score
        weights = self.token_weights
        complexity_score = (
            token_score * weights['count'] +
            indicator_score * weights['indicators'] +
            semantic_score * weights['semantic'] +
            structural_score * (1 - sum(weights.values()))  # Use remaining weight
        )
        
        # Get current adaptive threshold
        threshold = self.adaptive_thresholds[agent_type]['base']
        
        # Determine if task is complex
        is_complex = complexity_score > threshold
        
        # Calculate confidence in the evaluation
        confidence = self._calculate_evaluation_confidence(
            token_score, indicator_score, semantic_score, structural_score
        )
        
        return {
            "complexity_score": complexity_score,
            "is_complex": is_complex,
            "confidence": confidence,
            "components": {
                "token_complexity": token_score,
                "indicator_complexity": indicator_score,
                "semantic_complexity": semantic_score,
                "structural_complexity": structural_score
            },
            "threshold_used": threshold,
            "analysis": self._generate_complexity_analysis(
                agent_type, task, complexity_score, threshold
            )
        }
    
    def _calculate_structural_complexity(self, task: str) -> float:
        """
        Calculate complexity score based on task structure.
        
        Args:
            task: The task to evaluate
            
        Returns:
            Float between 0 and 1 indicating structural complexity
        """
        # Count nested structures
        nesting_level = task.count('(') + task.count('{') + task.count('[')
        
        # Count bullet points or numbered lists
        lines = [line for line in task.split('\n') if line.strip()]  # Skip empty lines
        list_items = sum(1 for line in lines if (
            line.strip().startswith(('-', '*', '•')) or 
            (len(line.strip()) >= 2 and line.strip()[0].isdigit() and line.strip()[1] in '.)'))
        )
        
        # Count code-like patterns
        code_indicators = sum(1 for line in lines if 
                            line.strip().startswith(('def ', 'class ', 'function', 'import ', 'from ')))
        
        # Calculate structural score
        structural_score = (
            min(1.0, nesting_level / 5) * 0.4 +  # Max score at 5 nested levels
            min(1.0, list_items / 10) * 0.3 +    # Max score at 10 list items
            min(1.0, code_indicators / 3) * 0.3   # Max score at 3 code indicators
        )
        
        return structural_score
    
    def _calculate_token_complexity(self, task: str) -> float:
        """Calculate complexity score based on token count and structure."""
        words = task.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Calculate unique word ratio
        unique_words = len(set(words))
        unique_ratio = unique_words / word_count
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / word_count
        normalized_length = min(1.0, avg_word_length / 10)  # Normalize to 0-1
        
        # Combine metrics
        base_score = min(1.0, word_count / 200)  # Normalize to 0-1 for tasks up to 200 words
        return (base_score * 0.5 + unique_ratio * 0.3 + normalized_length * 0.2)
    
    def _calculate_indicator_complexity(self, agent_type: str, task: str) -> float:
        """Calculate complexity score based on agent-specific indicators."""
        task_lower = task.lower()
        indicators = self.complexity_indicators[agent_type]
        
        # Count indicator occurrences with weighting
        weighted_count = 0
        for i, indicator in enumerate(indicators):
            if indicator in task_lower:
                # Give higher weight to earlier indicators (assumed more important)
                weight = 1 - (i / len(indicators) * 0.5)  # Weight ranges from 1 to 0.5
                weighted_count += weight
        
        # Normalize score
        max_possible = sum(1 - (i / len(indicators) * 0.5) for i in range(len(indicators)))
        return min(1.0, weighted_count / (max_possible * 0.5))  # Divide by 0.5 to allow reaching 1.0 with fewer indicators
    
    def _calculate_semantic_complexity(self, task: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate complexity score based on semantic analysis."""
        # Split into sentences
        sentences = [s.strip() for s in task.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Analyze logical operators and conditions
        logical_operators = ['and', 'or', 'but', 'however', 'although', 'unless', 'if', 'then', 'else']
        condition_markers = ['if', 'when', 'while', 'unless', 'until', 'provided that', 'assuming that']
        temporal_markers = ['before', 'after', 'during', 'while', 'simultaneously', 'meanwhile']
        
        # Calculate operator density
        operator_count = sum(
            sum(1 for op in logical_operators if op in sentence.lower())
            for sentence in sentences
        )
        
        condition_count = sum(
            sum(1 for marker in condition_markers if marker in sentence.lower())
            for sentence in sentences
        )
        
        temporal_count = sum(
            sum(1 for marker in temporal_markers if marker in sentence.lower())
            for sentence in sentences
        )
        
        # Calculate base complexity
        total_markers = operator_count + condition_count + temporal_count
        density = total_markers / len(sentences)
        base_score = min(1.0, density / 3)  # Normalize to 0-1
        
        # Adjust for context if available
        if context:
            context_modifier = min(0.2, len(str(context)) / 1000)
            base_score = min(1.0, base_score + context_modifier)
        
        return base_score
    
    def _calculate_evaluation_confidence(self, *scores: float) -> float:
        """Calculate confidence in the complexity evaluation."""
        if not scores:
            return 0.0
            
        # Calculate mean and standard deviation of scores
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        
        # Higher confidence when scores are consistent (low std dev)
        consistency_factor = 1 - min(1.0, std_score * 2)
        
        # Higher confidence when scores are not clustered around 0.5
        decisiveness_factor = abs(mean_score - 0.5) * 2
        
        # Combine factors
        confidence = (consistency_factor * 0.7 + decisiveness_factor * 0.3)
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_complexity_analysis(self, 
                                   agent_type: str,
                                   task: str,
                                   complexity_score: float,
                                   threshold: float) -> Dict[str, Any]:
        """Generate detailed analysis of task complexity."""
        # Determine primary complexity factors
        indicators_found = [
            indicator for indicator in self.complexity_indicators[agent_type]
            if indicator in task.lower()
        ]
        
        # Generate explanation
        if complexity_score > threshold:
            explanation = (
                f"Task is considered complex for {agent_type} agent. "
                f"Found {len(indicators_found)} complexity indicators: {', '.join(indicators_found[:3])}... "
                f"Score {complexity_score:.2f} exceeds threshold {threshold:.2f}."
            )
        else:
            explanation = (
                f"Task is considered manageable for {agent_type} agent. "
                f"Complexity score {complexity_score:.2f} is below threshold {threshold:.2f}."
            )
        
        return {
            "explanation": explanation,
            "primary_factors": indicators_found[:5],
            "score_breakdown": {
                "raw_score": complexity_score,
                "threshold": threshold,
                "margin": abs(complexity_score - threshold)
            }
        }
    
    def record_performance(self,
                         agent_type: str,
                         task_complexity: Dict[str, Any],
                         performance_metrics: Dict[str, float]) -> None:
        """Record performance for threshold adjustment."""
        if agent_type not in self.adaptive_thresholds:
            raise ValueError(f"Invalid agent type: {agent_type}")
            
        record = {
            "timestamp": datetime.now().timestamp(),
            "complexity_score": task_complexity["complexity_score"],
            "was_complex": task_complexity["is_complex"],
            "performance": performance_metrics,
            "threshold_used": task_complexity["threshold_used"]
        }
        
        self.performance_history[agent_type].append(record)
        
        # Keep last 1000 records
        if len(self.performance_history[agent_type]) > 1000:
            self.performance_history[agent_type] = self.performance_history[agent_type][-1000:]
        
        # Adjust thresholds periodically
        if len(self.performance_history[agent_type]) % 100 == 0:
            self.adjust_thresholds(agent_type)
    
    def adjust_thresholds(self,
                         agent_type: str,
                         performance_metrics: Optional[Dict[str, float]] = None,
                         complexity_history: Optional[List[Dict[str, Any]]] = None) -> float:
        """
        Adjust complexity thresholds based on performance history.
        
        Args:
            agent_type: Type of agent
            performance_metrics: Optional current performance metrics
            complexity_history: Optional historical complexity data
            
        Returns:
            New threshold value
        """
        if agent_type not in self.adaptive_thresholds:
            raise ValueError(f"Invalid agent type: {agent_type}")
            
        if not self.performance_history[agent_type] and not complexity_history:
            return self.adaptive_thresholds[agent_type]['base']
        
        history = complexity_history if complexity_history else self.performance_history[agent_type][-100:]
        
        if not history:
            return self.adaptive_thresholds[agent_type]['base']
        
        # Analyze performance patterns
        complex_tasks = [r for r in history if r.get("was_complex", False)]
        simple_tasks = [r for r in history if not r.get("was_complex", False)]
        
        if not complex_tasks or not simple_tasks:
            return self.adaptive_thresholds[agent_type]['base']
        
        # Calculate average performance
        if performance_metrics:
            complex_performance = np.mean(list(performance_metrics.values()))
            simple_performance = complex_performance  # Use same metrics if no separate data
        else:
            complex_performance = np.mean([
                np.mean([v for v in r.get("performance", {}).values() if isinstance(v, (int, float))])
                for r in complex_tasks if r.get("performance")
            ])
            simple_performance = np.mean([
                np.mean([v for v in r.get("performance", {}).values() if isinstance(v, (int, float))])
                for r in simple_tasks if r.get("performance")
            ])
        
        # Get current threshold configuration
        threshold_config = self.adaptive_thresholds[agent_type]
        current_threshold = threshold_config['base']
        
        # Determine adjustment
        if complex_performance > self.performance_thresholds[agent_type]:
            # Complex tasks performing well, might want to increase threshold
            adjustment = threshold_config['adjustment_rate']
        elif simple_performance < self.performance_thresholds[agent_type] * 0.8:
            # Simple tasks performing poorly, might want to decrease threshold
            adjustment = -threshold_config['adjustment_rate']
        else:
            # Performance is balanced, small adjustment based on ratio
            performance_ratio = complex_performance / simple_performance if simple_performance > 0 else 1
            adjustment = threshold_config['adjustment_rate'] * (performance_ratio - 1)
        
        # Apply adjustment within bounds
        new_threshold = current_threshold + adjustment
        new_threshold = max(threshold_config['min'], 
                          min(threshold_config['max'], new_threshold))
        
        # Update threshold if changed
        if new_threshold != current_threshold:
            self.adaptive_thresholds[agent_type]['base'] = new_threshold
            logger.info(
                f"Adjusted {agent_type} complexity threshold: {current_threshold:.3f} -> {new_threshold:.3f}"
            )
        
        return new_threshold
    
    def get_threshold(self, agent_type: str) -> float:
        """Get current complexity threshold for an agent type."""
        if agent_type not in self.adaptive_thresholds:
            raise ValueError(f"Invalid agent type: {agent_type}")
        return self.adaptive_thresholds[agent_type]['base']
    
    def get_threshold_analysis(self, agent_type: str) -> Dict[str, Any]:
        """Get analysis of current threshold and performance."""
        if agent_type not in self.adaptive_thresholds:
            raise ValueError(f"Invalid agent type: {agent_type}")
        
        threshold_config = self.adaptive_thresholds[agent_type]
        history = self.performance_history[agent_type][-100:] if self.performance_history[agent_type] else []
        
        complex_tasks = [r for r in history if r.get("was_complex", False)]
        simple_tasks = [r for r in history if not r.get("was_complex", False)]
        
        complex_performance = 0.0
        simple_performance = 0.0
        
        if complex_tasks:
            complex_performance = np.mean([
                np.mean([v for v in r.get("performance", {}).values() if isinstance(v, (int, float))])
                for r in complex_tasks if r.get("performance")
            ])
            
        if simple_tasks:
            simple_performance = np.mean([
                np.mean([v for v in r.get("performance", {}).values() if isinstance(v, (int, float))])
                for r in simple_tasks if r.get("performance")
            ])
        
        return {
            "current_threshold": threshold_config['base'],
            "min_threshold": threshold_config['min'],
            "max_threshold": threshold_config['max'],
            "complex_task_ratio": len(complex_tasks) / len(history) if history else 0,
            "performance_by_complexity": {
                "complex": complex_performance,
                "simple": simple_performance
            }
        }
