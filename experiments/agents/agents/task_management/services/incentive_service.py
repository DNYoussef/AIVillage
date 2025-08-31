"""
Incentive Service - Handles incentive calculation and agent performance tracking.
Extracted from UnifiedManagement god class.
"""
import logging
from typing import Any

from AIVillage.experimental.agents.agents.analytics.unified_analytics import UnifiedAnalytics
from core.error_handling import AIVillageException

from ..incentive_model import IncentiveModel
from ..task import Task

logger = logging.getLogger(__name__)


class IncentiveService:
    """Service responsible for incentive calculation and agent performance tracking."""
    
    def __init__(
        self,
        incentive_model: IncentiveModel,
        unified_analytics: UnifiedAnalytics,
    ) -> None:
        """Initialize with dependencies."""
        self._incentive_model = incentive_model
        self._unified_analytics = unified_analytics
        self._agent_performance: dict[str, float] = {}
        
    async def calculate_incentive(self, agent: str, task: Task) -> float:
        """Calculate incentive for an agent-task pair."""
        try:
            incentive_data = self._incentive_model.calculate_incentive(
                {"assigned_agent": agent, "task_id": task.id}, 
                self._agent_performance
            )
            
            incentive_value = incentive_data.get("incentive", 0.0)
            logger.debug("Calculated incentive %f for agent %s, task %s", 
                        incentive_value, agent, task.id)
            
            return incentive_value
        except Exception as e:
            logger.exception("Error calculating incentive: %s", e)
            msg = f"Error calculating incentive: {e!s}"
            raise AIVillageException(msg) from e

    async def update_agent_performance(self, agent: str, task_result: Any) -> None:
        """Update agent performance metrics based on task result."""
        try:
            # Update the incentive model with the task result
            self._incentive_model.update(
                {"assigned_agent": agent}, 
                task_result
            )
            
            # Update agent performance using the incentive model's method
            self._incentive_model.update_agent_performance(
                self._agent_performance,
                agent,
                task_result,
                self._unified_analytics,
            )
            
            logger.info("Updated performance for agent %s", agent)
            
        except Exception as e:
            logger.exception("Error updating agent performance: %s", e)
            msg = f"Error updating agent performance: {e!s}"
            raise AIVillageException(msg) from e

    def get_agent_performance(self, agent: str) -> float:
        """Get performance score for a specific agent."""
        return self._agent_performance.get(agent, 0.0)

    def get_all_agent_performance(self) -> dict[str, float]:
        """Get performance scores for all agents."""
        return self._agent_performance.copy()

    def get_top_performing_agents(self, limit: int = 5) -> list[tuple[str, float]]:
        """Get top performing agents sorted by performance score."""
        try:
            sorted_agents = sorted(
                self._agent_performance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_agents[:limit]
        except Exception as e:
            logger.exception("Error getting top performing agents: %s", e)
            return []

    def reset_agent_performance(self, agent: str) -> None:
        """Reset performance score for a specific agent."""
        try:
            if agent in self._agent_performance:
                self._agent_performance[agent] = 0.0
                logger.info("Reset performance for agent %s", agent)
            else:
                logger.warning("Agent %s not found in performance tracking", agent)
        except Exception as e:
            logger.exception("Error resetting agent performance: %s", e)
            msg = f"Error resetting agent performance: {e!s}"
            raise AIVillageException(msg) from e

    def get_performance_statistics(self) -> dict[str, Any]:
        """Get overall performance statistics."""
        try:
            if not self._agent_performance:
                return {
                    "total_agents": 0,
                    "average_performance": 0.0,
                    "max_performance": 0.0,
                    "min_performance": 0.0,
                }
                
            performances = list(self._agent_performance.values())
            
            return {
                "total_agents": len(performances),
                "average_performance": sum(performances) / len(performances),
                "max_performance": max(performances),
                "min_performance": min(performances),
                "top_agents": self.get_top_performing_agents(3),
            }
        except Exception as e:
            logger.exception("Error getting performance statistics: %s", e)
            return {"error": str(e)}