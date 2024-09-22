from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from ..utils.ai_provider import AIProvider
from ..utils.exceptions import AIVillageException

@dataclass
class Node:
    name: str
    prerequisites: List['Node'] = field(default_factory=list)
    probability: float = 1.0

class PlanGenerator:
    def __init__(self, ai_provider: AIProvider):
        self.ai_provider = ai_provider

    async def generate_plan(self, goal: str, problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        try:
            current_resources = await self._get_current_resources()
            plan_tree = await self._create_plan_tree(goal, current_resources)
            gaps = self._identify_capability_gaps(plan_tree, current_resources)
            tasks = self._break_down_tasks(plan_tree)
            checkpoints = await self._plan_checkpoints(tasks)
            
            success_likelihood = 0
            iteration = 0
            while success_likelihood < 0.95 and iteration < 10:
                premortem_results = await self._conduct_premortem(plan_tree, tasks)
                plan_tree, tasks = self._update_plan(plan_tree, tasks, premortem_results)
                success_likelihood = self._calculate_success_likelihood(plan_tree)
                iteration += 1
            
            swot_analysis = await self._perform_swot_analysis(plan_tree, problem_analysis)

            return {
                'goal': goal,
                'plan_tree': self._tree_to_dict(plan_tree),
                'capability_gaps': gaps,
                'tasks': tasks,
                'checkpoints': checkpoints,
                'premortem_results': premortem_results,
                'swot_analysis': swot_analysis,
                'success_likelihood': success_likelihood,
                'iterations': iteration
            }
        except Exception as e:
            raise AIVillageException(f"Error in plan generation: {str(e)}")

    async def _get_current_resources(self) -> Dict[str, Any]:
        prompt = "List the current resources, position, and tools available for the AI Village project. Output as a JSON dictionary with keys 'resources', 'position', and 'tools'."
        return await self.ai_provider.generate_structured_response(prompt)

    async def _create_plan_tree(self, goal: str, current_resources: Dict[str, Any]) -> Node:
        root = Node(name=goal)
        await self._expand_node(root, current_resources)
        return root

    async def _expand_node(self, node: Node, current_resources: Dict[str, Any]):
        if self._is_resource(node.name, current_resources):
            return

        prompt = f"What are the immediate prerequisites for achieving '{node.name}'? Output as a JSON list of strings."
        prerequisites = await self.ai_provider.generate_structured_response(prompt)

        for prereq in prerequisites:
            child = Node(name=prereq)
            node.prerequisites.append(child)
            await self._expand_node(child, current_resources)

    def _is_resource(self, name: str, resources: Dict[str, Any]) -> bool:
        return any(name in resource_list for resource_list in resources.values())

    def _identify_capability_gaps(self, plan_tree: Node, current_resources: Dict[str, Any]) -> List[str]:
        gaps = []
        self._find_gaps(plan_tree, current_resources, gaps)
        return gaps

    def _find_gaps(self, node: Node, resources: Dict[str, Any], gaps: List[str]):
        if not node.prerequisites and not self._is_resource(node.name, resources):
            gaps.append(node.name)
        for prereq in node.prerequisites:
            self._find_gaps(prereq, resources, gaps)

    def _break_down_tasks(self, plan_tree: Node) -> List[Dict[str, Any]]:
        tasks = []
        self._extract_tasks(plan_tree, tasks)
        return tasks

    def _extract_tasks(self, node: Node, tasks: List[Dict[str, Any]]):
        tasks.append({
            'name': node.name,
            'description': f"Achieve {node.name}",
            'prerequisites': [prereq.name for prereq in node.prerequisites],
            'estimated_effort': 1  # This could be refined with AI assistance if needed
        })
        for prereq in node.prerequisites:
            self._extract_tasks(prereq, tasks)

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

    async def _conduct_premortem(self, plan_tree: Node, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = f"""
        Conduct a premortem analysis for the following plan:
        Plan Tree: {self._tree_to_dict(plan_tree)}
        Tasks: {tasks}

        Imagine that the plan has failed. Describe:
        1. What went wrong
        2. Why it went wrong
        3. What could have been done to prevent each issue

        Output the result as a JSON dictionary with keys 'failures' (list of what went wrong),
        'reasons' (list of why things went wrong), and 'preventive_measures' (list of preventive actions).
        """
        return await self.ai_provider.generate_structured_response(prompt)

    def _update_plan(self, plan_tree: Node, tasks: List[Dict[str, Any]], premortem_results: Dict[str, Any]) -> Tuple[Node, List[Dict[str, Any]]]:
        self._update_node_probabilities(plan_tree, premortem_results)
        updated_tasks = self._update_tasks(tasks, premortem_results)
        return plan_tree, updated_tasks

    def _update_node_probabilities(self, node: Node, premortem_results: Dict[str, Any]):
        failure_count = sum(1 for failure in premortem_results['failures'] if failure.lower() in node.name.lower())
        node.probability *= (1 - 0.1 * failure_count)  # Decrease probability by 10% for each related failure
        for prereq in node.prerequisites:
            self._update_node_probabilities(prereq, premortem_results)

    def _update_tasks(self, tasks: List[Dict[str, Any]], premortem_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        updated_tasks = tasks.copy()
        for measure in premortem_results['preventive_measures']:
            updated_tasks.append({
                'name': f"Implement: {measure}",
                'description': measure,
                'prerequisites': [],
                'estimated_effort': 1
            })
        return updated_tasks

    def _calculate_success_likelihood(self, plan_tree: Node) -> float:
        return self._calculate_node_probability(plan_tree)

    def _calculate_node_probability(self, node: Node) -> float:
        if not node.prerequisites:
            return node.probability
        return node.probability * min(self._calculate_node_probability(prereq) for prereq in node.prerequisites)

    async def _perform_swot_analysis(self, plan_tree: Node, problem_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        prompt = f"""
        Perform a SWOT analysis for the following plan and problem:
        Plan Tree: {self._tree_to_dict(plan_tree)}
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

    def _tree_to_dict(self, node: Node) -> Dict[str, Any]:
        return {
            'name': node.name,
            'probability': node.probability,
            'prerequisites': [self._tree_to_dict(prereq) for prereq in node.prerequisites]
        }