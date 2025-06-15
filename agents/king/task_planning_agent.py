from rag_system.agents.task_planning_agent import TaskPlanningAgent as _BaseTaskPlanningAgent

class TaskPlanningAgent(_BaseTaskPlanningAgent):
    def generate_task_plan(self, *args, **kwargs):
        return super().plan_tasks(*args, **kwargs)

__all__ = ["TaskPlanningAgent"]
