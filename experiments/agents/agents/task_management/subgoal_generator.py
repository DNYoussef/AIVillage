# Import constants for future extensibility
from infrastructure.constants import TaskConstants, get_config_manager


class SubGoalGenerator:
    """Generates sub-goals for complex tasks.
    
    This is a simple implementation that returns the original description.
    Future enhancements could use AI/ML techniques for better sub-goal generation.
    """
    
    def __init__(self):
        self._config_manager = get_config_manager()
    
    async def generate_subgoals(self, description: str, context: dict) -> list[str]:
        """Generate sub-goals from a task description and context.
        
        Args:
            description: The main task description
            context: Additional context for sub-goal generation
            
        Returns:
            List of sub-goal descriptions
        """
        # For now, return the original description as a single sub-goal
        # Future implementations could split complex tasks into smaller components
        return [description]
