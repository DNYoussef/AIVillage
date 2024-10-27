import logging
import json
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ComplexityEvaluator:
    """
    System for evaluating task complexity and managing thresholds
    for when to use frontier vs local models.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize ComplexityEvaluator.
        
        Args:
            data_dir: Directory for storing threshold data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize thresholds for each agent type
        self.thresholds = {
            "king": {
                "base_complexity_threshold": 0.6,
                "indicator_weights": {
                    "token_count": 0.3,
                    "complexity_indicators": 0.4,
                    "semantic_complexity": 0.3
                },
                "performance_threshold": 0.8
            },
            "sage": {
                "base_complexity_threshold": 0.7,
                "indicator_weights": {
                    "token_count": 0.25,
                    "complexity_indicators": 0.45,
                    "semantic_complexity": 0.3
                },
                "performance_threshold": 0.75
            },
            "magi": {
                "base_complexity_threshold": 0.65,
                "indicator_weights": {
                    "token_count": 0.2,
                    "complexity_indicators": 0.5,
                    "semantic_complexity": 0.3
                },
                "performance_threshold": 0.85
            }
        }
        
        # Load saved thresholds if they exist
        self._load_thresholds()
        
        logger.info("Initialized ComplexityEvaluator with thresholds:")
        for agent, config in self.thresholds.items():
            logger.info(f"  {agent}: base_threshold={config['base_complexity_threshold']:.2f}")
    
    def _load_thresholds(self):
        """Load saved threshold configurations."""
        threshold_file = self.data_dir / "complexity_thresholds.json"
        if threshold_file.exists():
            try:
                with open(threshold_file, 'r') as f:
                    saved_thresholds = json.load(f)
                self.thresholds.update(saved_thresholds)
                logger.info("Loaded saved complexity thresholds")
            except Exception as e:
                logger.error(f"Error loading thresholds: {str(e)}")
    
    def _save_thresholds(self):
        """Save current threshold configurations."""
        threshold_file = self.data_dir / "complexity_thresholds.json"
        try:
            with open(threshold_file, 'w') as f:
                json.dump(self.thresholds, f, indent=2)
            logger.info("Saved updated complexity thresholds")
        except Exception as e:
            logger.error(f"Error saving thresholds: {str(e)}")
    
    def evaluate_complexity(self, 
                          agent_type: str, 
                          task: str,
                          context: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate task complexity for a specific agent type.
        
        Args:
            agent_type: Type of agent ("king", "sage", or "magi")
            task: The task to evaluate
            context: Optional context information
            
        Returns:
            Dictionary containing complexity score and analysis
        """
        if agent_type not in self.thresholds:
            raise ValueError(f"Invalid agent type: {agent_type}")
        
        # Get agent-specific thresholds and weights
        config = self.thresholds[agent_type]
        weights = config["indicator_weights"]
        
        # Calculate component scores
        token_score = self._calculate_token_complexity(task)
        indicator_score = self._calculate_indicator_complexity(agent_type, task)
        semantic_score = self._calculate_semantic_complexity(task, context)
        
        # Calculate weighted complexity score
        complexity_score = (
            token_score * weights["token_count"] +
            indicator_score * weights["complexity_indicators"] +
            semantic_score * weights["semantic_complexity"]
        )
        
        # Compare with threshold
        is_complex = complexity_score > config["base_complexity_threshold"]
        
        return {
            "complexity_score": complexity_score,
            "is_complex": is_complex,
            "components": {
                "token_complexity": token_score,
                "indicator_complexity": indicator_score,
                "semantic_complexity": semantic_score
            },
            "threshold_used": config["base_complexity_threshold"]
        }
    
    def _calculate_token_complexity(self, task: str) -> float:
        """Calculate complexity score based on token count."""
        # Simple token-based scoring
        words = task.split()
        word_count = len(words)
        
        # Normalize score (0-1) based on typical task lengths
        if word_count <= 20:
            return 0.3
        elif word_count <= 50:
            return 0.5
        elif word_count <= 100:
            return 0.7
        else:
            return 0.9
    
    def _calculate_indicator_complexity(self, agent_type: str, task: str) -> float:
        """Calculate complexity score based on complexity indicators."""
        task_lower = task.lower()
        
        # Agent-specific complexity indicators
        indicators = {
            "king": [
                "analyze", "evaluate", "compare", "synthesize", "design",
                "optimize", "recommend", "strategic", "complex", "multi-step"
            ],
            "sage": [
                "research", "analyze", "synthesize", "evaluate", "implications",
                "relationship", "impact", "trends", "patterns", "framework"
            ],
            "magi": [
                "optimize", "implement", "design", "architecture", "system",
                "algorithm", "performance", "scale", "concurrent", "async"
            ]
        }
        
        # Count indicators present in task
        agent_indicators = indicators[agent_type]
        indicator_count = sum(1 for indicator in agent_indicators if indicator in task_lower)
        
        # Normalize score (0-1)
        return min(1.0, indicator_count / len(agent_indicators))
    
    def _calculate_semantic_complexity(self, task: str, context: Optional[str] = None) -> float:
        """Calculate complexity score based on semantic analysis."""
        # For now, use a simple heuristic based on sentence structure
        sentences = [s.strip() for s in task.split('.') if s.strip()]
        
        # Consider number of conjunctions and conditions
        complexity_terms = ['and', 'or', 'but', 'if', 'when', 'while', 'unless']
        term_count = sum(
            1 for sentence in sentences
            for term in complexity_terms
            if term in sentence.lower()
        )
        
        # Consider nested requirements
        nesting_level = task.count('(') + task.count('{') + task.count('[')
        
        # Calculate base score
        base_score = min(1.0, (len(sentences) + term_count + nesting_level * 2) / 10)
        
        # Adjust for context if provided
        if context:
            context_modifier = min(0.2, len(context.split()) / 1000)
            base_score = min(1.0, base_score + context_modifier)
        
        return base_score
    
    def adjust_thresholds(self, 
                         agent_type: str,
                         performance_metrics: Dict[str, float],
                         complexity_history: List[Dict[str, Any]]):
        """
        Adjust complexity thresholds based on performance metrics.
        
        Args:
            agent_type: Type of agent
            performance_metrics: Recent performance metrics
            complexity_history: History of complexity evaluations and outcomes
        """
        if agent_type not in self.thresholds:
            raise ValueError(f"Invalid agent type: {agent_type}")
            
        config = self.thresholds[agent_type]
        
        # Calculate local model performance
        local_performance = performance_metrics.get("local_model_performance", 0.0)
        
        # Analyze complexity history
        complex_tasks = [task for task in complexity_history if task["is_complex"]]
        simple_tasks = [task for task in complexity_history if not task["is_complex"]]
        
        if not complexity_history:
            return
        
        # Calculate success rates
        complex_success = np.mean([
            task.get("success", 0.0) for task in complex_tasks
        ]) if complex_tasks else 0.0
        
        simple_success = np.mean([
            task.get("success", 0.0) for task in simple_tasks
        ]) if simple_tasks else 0.0
        
        # Adjust threshold based on performance
        if local_performance > config["performance_threshold"]:
            # Local model performing well, gradually increase threshold
            adjustment = 0.05
        elif complex_success < 0.7 and simple_success > 0.8:
            # Too many tasks marked as complex, increase threshold
            adjustment = 0.03
        elif complex_success > 0.9 and simple_success < 0.7:
            # Too few tasks marked as complex, decrease threshold
            adjustment = -0.03
        else:
            # Performance in expected range, small adjustment
            adjustment = 0.01 if local_performance > 0.75 else -0.01
        
        # Apply adjustment with bounds
        new_threshold = config["base_complexity_threshold"] + adjustment
        config["base_complexity_threshold"] = max(0.3, min(0.9, new_threshold))
        
        # Save updated thresholds
        self._save_thresholds()
        
        logger.info(f"Adjusted {agent_type} complexity threshold to: {config['base_complexity_threshold']:.2f}")
    
    def get_threshold_history(self, agent_type: str) -> List[Dict[str, Any]]:
        """
        Get threshold adjustment history for an agent.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            List of threshold adjustments with timestamps
        """
        history_file = self.data_dir / f"{agent_type}_threshold_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading threshold history: {str(e)}")
                return []
        return []
