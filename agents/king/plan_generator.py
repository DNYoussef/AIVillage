from typing import Dict, Any, List
from ..utils.ai_provider import AIProvider
from ..utils.exceptions import AIVillageException

class PlanGenerator:
    def __init__(self, ai_provider: AIProvider):
        self.ai_provider = ai_provider

    async def generate_plan(self, goal: str, problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        try:
            current_resources = await self._get_current_resources()
            plan_tree = await self._create_plan_tree(goal, current_resources)
            gaps = await self._identify_capability_gaps(plan_tree, current_resources)
            tasks = await self._break_down_tasks(plan_tree)
            checkpoints = await self._plan_checkpoints(tasks)
            premortem = await self._conduct_premortem(plan_tree, tasks)
            swot_analysis = await self._perform_swot_analysis(plan_tree, problem_analysis)

            return {
                'goal': goal,
                'plan_tree': plan_tree,
                'capability_gaps': gaps,
                'tasks': tasks,
                'checkpoints': checkpoints,
                'premortem': premortem,
                'swot_analysis': swot_analysis
            }
        except Exception as e:
            raise AIVillageException(f"Error in plan generation: {str(e)}")

    async def _get_current_resources(self) -> Dict[str, Any]:
        prompt = "List the current resources, position, and tools available for the AI Village project. Output as a JSON dictionary with keys 'resources', 'position', and 'tools'."
        return await self.ai_provider.generate_structured_response(prompt)

    async def _create_plan_tree(self, goal: str, current_resources: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Create a reverse Markov tree for achieving the goal: '{goal}'.
        Start from the goal and work backwards, asking "What must necessarily be true as a prerequisite for this to be accomplished?"
        Continue this process until you reach nodes that correspond to our current resources: {current_resources}
        Output the tree as a nested JSON structure where each node has 'name' and 'prerequisites' keys.
        The 'prerequisites' should be a list of prerequisite nodes.
        """
        return await self.ai_provider.generate_structured_response(prompt)

    async def _identify_capability_gaps(self, plan_tree: Dict[str, Any], current_resources: Dict[str, Any]) -> List[str]:
        prompt = f"""
        Analyze the plan tree: {plan_tree}
        Compare it with our current resources: {current_resources}
        Identify any capability gaps that may require new agent forging or resource acquisition.
        Output the result as a JSON list of strings, each describing a capability gap.
        """
        return await self.ai_provider.generate_structured_response(prompt)

    async def _break_down_tasks(self, plan_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = f"""
        Based on the plan tree: {plan_tree}
        Break down the plan into a series of concrete tasks that need to be assigned and completed.
        Output the result as a JSON list of dictionaries, where each dictionary represents a task and has keys:
        'name', 'description', 'prerequisites' (list of task names that must be completed before this task),
        and 'estimated_effort' (in arbitrary units).
        """
        return await self.ai_provider.generate_structured_response(prompt)

    async def _plan_checkpoints(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt = f"""
        Given the following list of tasks: {tasks}
        Create a series of checkpoints to monitor progress.
        Each checkpoint should be a SMART goal (Specific, Measurable, Achievable, Relevant, Time-bound).
        Output the result as a JSON list of dictionaries, where each dictionary represents a checkpoint and has keys:
        'name', 'description', 'related_tasks' (list of task names this checkpoint covers), 'metric' (how to measure progress),
        and 'target_date'.
        """
        return await self.ai_provider.generate_structured_response(prompt)

    async def _conduct_premortem(self, plan_tree: Dict[str, Any], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""
        Conduct a premortem analysis for the following plan:
        Plan Tree: {plan_tree}
        Tasks: {tasks}

        Imagine that the plan has failed. Describe:
        1. What went wrong
        2. Why it went wrong
        3. What could have been done to prevent each issue

        Output the result as a JSON dictionary with keys 'failures' (list of what went wrong),
        'reasons' (list of why things went wrong), and 'preventive_measures' (list of preventive actions).
        """
        return await self.ai_provider.generate_structured_response(prompt)

    async def _perform_swot_analysis(self, plan_tree: Dict[str, Any], problem_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        prompt = f"""
        Perform a SWOT analysis for the following plan and problem:
        Plan Tree: {plan_tree}
        Problem Analysis: {problem_analysis}

        Consider:
        - Strengths: Internal attributes and resources that support a successful outcome.
        - Weaknesses: Internal attributes and resources that work against a successful outcome.
        - Opportunities: External factors that the project could capitalize on or use to its advantage.
        - Threats: External factors that could jeopardize the project's success.

        Output the result as a JSON dictionary with keys 'strengths', 'weaknesses', 'opportunities', and 'threats',
        each containing a list of relevant points.
        """
        return await self.ai_provider.generate_structured_response(prompt)