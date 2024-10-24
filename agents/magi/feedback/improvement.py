"""Improvement system for MAGI feedback."""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging
import json

from ..core.exceptions import ToolError
from ..utils.logging import setup_logger
from .analysis import (
    FeedbackAnalyzer,
    PerformanceMetrics,
    TechniqueMetrics,
    SystemMetrics
)

logger = setup_logger(__name__)

@dataclass
class ImprovementPlan:
    """Plan for implementing improvements."""
    target: str  # 'technique', 'tool', or 'system'
    target_name: Optional[str]  # Name of specific technique or tool
    improvements: List[Dict[str, Any]]
    priority: int  # 1 (highest) to 5 (lowest)
    estimated_impact: float  # 0.0 to 1.0
    dependencies: List[str]
    implementation_steps: List[str]

@dataclass
class ImprovementResult:
    """Result of implementing improvements."""
    plan: ImprovementPlan
    success: bool
    changes_made: List[str]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    errors: List[str]
    timestamp: datetime

class ImprovementManager:
    """
    Manages system improvements based on feedback analysis.
    
    Responsibilities:
    - Generate improvement plans
    - Prioritize improvements
    - Implement improvements
    - Validate results
    - Track improvement history
    """
    
    def __init__(self, analyzer: FeedbackAnalyzer):
        """
        Initialize improvement manager.
        
        Args:
            analyzer: Feedback analyzer to use
        """
        self.analyzer = analyzer
        self.improvement_history: List[ImprovementResult] = []
        self.active_improvements: Dict[str, ImprovementPlan] = {}
        self.improvement_queue: List[ImprovementPlan] = []
    
    async def generate_improvement_plans(
        self,
        analysis_window: Optional[timedelta] = None
    ) -> List[ImprovementPlan]:
        """
        Generate improvement plans based on analysis.
        
        Args:
            analysis_window: Time window for analysis (optional)
            
        Returns:
            List of improvement plans
        """
        plans = []
        
        # Analyze system performance
        try:
            system_metrics = self.analyzer.analyze_system_performance(analysis_window)
            system_plans = self._generate_system_improvements(system_metrics)
            plans.extend(system_plans)
        except ToolError as e:
            logger.warning(f"Error analyzing system performance: {e}")
        
        # Analyze techniques
        for technique in system_metrics.active_techniques:
            try:
                technique_metrics = self.analyzer.analyze_technique_performance(
                    technique,
                    analysis_window
                )
                technique_plans = self._generate_technique_improvements(technique_metrics)
                plans.extend(technique_plans)
            except ToolError as e:
                logger.warning(f"Error analyzing technique {technique}: {e}")
        
        # Analyze tools
        for tool in system_metrics.active_tools:
            try:
                tool_metrics = self.analyzer.analyze_tool_performance(
                    tool,
                    analysis_window
                )
                tool_plans = self._generate_tool_improvements(tool_metrics, tool)
                plans.extend(tool_plans)
            except ToolError as e:
                logger.warning(f"Error analyzing tool {tool}: {e}")
        
        # Analyze error patterns
        error_analysis = self.analyzer.analyze_error_patterns(analysis_window)
        error_plans = self._generate_error_improvements(error_analysis)
        plans.extend(error_plans)
        
        # Sort plans by priority and estimated impact
        plans.sort(key=lambda p: (p.priority, -p.estimated_impact))
        
        return plans
    
    async def implement_improvements(
        self,
        plans: Optional[List[ImprovementPlan]] = None
    ) -> List[ImprovementResult]:
        """
        Implement improvement plans.
        
        Args:
            plans: Plans to implement (if None, generate new plans)
            
        Returns:
            List of improvement results
        """
        if plans is None:
            plans = await self.generate_improvement_plans()
        
        results = []
        for plan in plans:
            # Skip if dependencies not met
            if not self._check_dependencies(plan):
                logger.warning(f"Skipping plan {plan.target}: dependencies not met")
                continue
            
            # Get current metrics
            metrics_before = await self._get_current_metrics(plan)
            
            # Implement improvements
            try:
                changes = await self._implement_plan(plan)
                success = True
                errors = []
            except Exception as e:
                logger.exception(f"Error implementing plan {plan.target}")
                changes = []
                success = False
                errors = [str(e)]
            
            # Get updated metrics
            metrics_after = await self._get_current_metrics(plan)
            
            # Record result
            result = ImprovementResult(
                plan=plan,
                success=success,
                changes_made=changes,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                errors=errors,
                timestamp=datetime.now()
            )
            
            self.improvement_history.append(result)
            results.append(result)
            
            # Update active improvements
            if success:
                self.active_improvements[plan.target] = plan
            
            # Sleep briefly to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        return results
    
    def _check_dependencies(self, plan: ImprovementPlan) -> bool:
        """Check if all dependencies for a plan are met."""
        return all(
            dep in self.active_improvements
            for dep in plan.dependencies
        )
    
    async def _get_current_metrics(self, plan: ImprovementPlan) -> Dict[str, float]:
        """Get current metrics for a plan's target."""
        try:
            if plan.target == "system":
                metrics = self.analyzer.analyze_system_performance()
                return {
                    'success_rate': metrics.successful_tasks / metrics.total_tasks if metrics.total_tasks > 0 else 1.0,
                    'response_time': metrics.average_response_time,
                    'memory_usage': metrics.peak_memory_usage
                }
            elif plan.target == "technique":
                metrics = self.analyzer.analyze_technique_performance(plan.target_name)
                return {
                    'success_rate': metrics.success_rate,
                    'confidence': metrics.average_confidence,
                    'execution_time': metrics.average_execution_time
                }
            elif plan.target == "tool":
                metrics = self.analyzer.analyze_tool_performance(plan.target_name)
                return {
                    'success_rate': metrics.success_rate,
                    'execution_time': metrics.average_execution_time,
                    'error_rate': metrics.error_rate
                }
            else:
                return {}
        except Exception as e:
            logger.warning(f"Error getting metrics for {plan.target}: {e}")
            return {}
    
    async def _implement_plan(self, plan: ImprovementPlan) -> List[str]:
        """Implement an improvement plan."""
        changes = []
        
        for step in plan.implementation_steps:
            try:
                # Implement the step
                # This is a placeholder - actual implementation would depend on the specific step
                logger.info(f"Implementing step: {step}")
                
                # Record the change
                changes.append(f"Completed: {step}")
                
                # Brief pause between steps
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.exception(f"Error implementing step: {step}")
                raise ToolError(f"Implementation failed at step '{step}': {str(e)}")
        
        return changes
    
    def get_improvement_history(
        self,
        target: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> List[ImprovementResult]:
        """
        Get improvement history.
        
        Args:
            target: Filter by target (optional)
            time_window: Time window to consider (optional)
            
        Returns:
            List of improvement results
        """
        history = self.improvement_history
        
        if target:
            history = [
                result for result in history
                if result.plan.target == target
            ]
        
        if time_window:
            cutoff = datetime.now() - time_window
            history = [
                result for result in history
                if result.timestamp > cutoff
            ]
        
        return history
    
    def get_active_improvements(self) -> Dict[str, ImprovementPlan]:
        """Get currently active improvements."""
        return self.active_improvements.copy()

# Example usage
if __name__ == "__main__":
    async def main():
        # Create analyzer and manager
        analyzer = FeedbackAnalyzer()
        manager = ImprovementManager(analyzer)
        
        # Generate improvement plans
        plans = await manager.generate_improvement_plans(
            analysis_window=timedelta(days=7)
        )
        print(f"Generated {len(plans)} improvement plans")
        
        # Implement improvements
        results = await manager.implement_improvements(plans)
        print(f"Implemented {len(results)} improvements")
        
        # Check improvement history
        history = manager.get_improvement_history(
            time_window=timedelta(days=7)
        )
        print(f"Improvement history: {len(history)} entries")
    
    asyncio.run(main())
