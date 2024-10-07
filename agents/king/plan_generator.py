import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt

# Assume these are implemented in separate files
from king_agent import KingAgent, KingAgentConfig
from exceptions import AIVillageException, PlanningException, ModelInteractionException

@dataclass
class Node:
    name: str
    description: str
    prerequisites: List['Node'] = field(default_factory=list)
    probability: float = 0.5  # 0 to 1
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    failure_modes: List[Dict[str, Any]] = field(default_factory=list)
    antifragility_score: float = 0.0  # -10 to 10
    xanatos_factor: float = 0.0  # -10 to 10
    xanatos_gambits: List[Dict[str, Any]] = field(default_factory=list)
    expected_utility: float = 0.0  # -10 to 10

@dataclass
class PlanConfig:
    success_likelihood_threshold: float = 0.95
    max_iterations: int = 10
    parallelization: bool = True

class PlanGenerator:
    def __init__(self, king_agent: KingAgent = None, config: PlanConfig = PlanConfig()):
        self.king_agent = king_agent or KingAgent()
        self.config = config

    async def generate_plan(self, goal: str, problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        try:
            current_resources = await self._get_current_resources()
            plan_tree = await self._create_plan_tree(goal, current_resources)
            await self._extract_tasks(plan_tree)
            await self._optimize_tasks(plan_tree)
            
            success_likelihood = 0
            iteration = 0
            while success_likelihood < self.config.success_likelihood_threshold and iteration < self.config.max_iterations:
                await self._conduct_premortem(plan_tree)
                await self._assess_antifragility(plan_tree)
                await self._develop_xanatos_gambits(plan_tree)
                plan_tree = await self._update_plan(plan_tree)
                success_likelihood = self._calculate_success_likelihood(plan_tree)
                iteration += 1
            
            gaps = self._identify_capability_gaps(plan_tree, current_resources)
            checkpoints = await self._plan_checkpoints(self._extract_all_tasks(plan_tree))
            swot_analysis = await self._perform_swot_analysis(plan_tree, problem_analysis)
            metrics = self._calculate_plan_metrics(plan_tree)

            plan_data = {
                'goal': goal,
                'plan_tree': self._tree_to_dict(plan_tree),
                'capability_gaps': gaps,
                'checkpoints': checkpoints,
                'swot_analysis': swot_analysis,
                'success_likelihood': success_likelihood,
                'iterations': iteration,
                'metrics': metrics
            }

            # Save the plan to a file
            save_plan_to_file(plan_data, f"plan_{goal.replace(' ', '_')}.json")

            # Generate a visualization of the plan tree
            visualization = self._create_plan_visualization(plan_tree)

            return {**plan_data, 'visualization': visualization}
        except Exception as e:
            raise PlanningException(f"Error in plan generation: {str(e)}")

    async def _get_current_resources(self) -> Dict[str, Any]:
        prompt = "List the current resources, position, and tools available for the AI Village project. Output as a JSON dictionary with keys 'resources', 'position', and 'tools'."
        try:
            return await self.king_agent.generate_structured_response(prompt)
        except Exception as e:
            raise ModelInteractionException(f"Error getting current resources: {str(e)}")

    async def _create_plan_tree(self, goal: str, current_resources: Dict[str, Any]) -> Node:
        root = Node(name=goal, description=f"Achieve: {goal}")
        await self._expand_node(root, current_resources)
        return root

    async def _expand_node(self, node: Node, current_resources: Dict[str, Any]):
        if self._is_resource(node.name, current_resources):
            return

        prompt = f"What are the immediate prerequisites for achieving '{node.name}'? Output as a JSON list of dictionaries, each with 'name' and 'description' keys."
        prerequisites = await self.ai_provider.generate_structured_response(prompt)

        for prereq in prerequisites:
            child = Node(name=prereq['name'], description=prereq['description'])
            node.prerequisites.append(child)
            await self._expand_node(child, current_resources)

    def _is_resource(self, name: str, resources: Dict[str, Any]) -> bool:
        return any(name in resource_list for resource_list in resources.values())

    async def _extract_tasks(self, node: Node):
        if self.config.parallelization:
            await asyncio.gather(
                self._extract_node_tasks(node),
                *[self._extract_tasks(prereq) for prereq in node.prerequisites]
            )
        else:
            await self._extract_node_tasks(node)
            for prereq in node.prerequisites:
                await self._extract_tasks(prereq)

    async def _extract_node_tasks(self, node: Node):
        prompt = f"""
        For the goal: "{node.name}"
        Description: "{node.description}"
        Given the prerequisites: {[prereq.name for prereq in node.prerequisites]}
        Generate a list of clear, actionable tasks that are mutually exclusive and collectively exhaustive.
        Ensure each task is solvable given the prerequisite resources.
        Output as a JSON list of dictionaries, where each dictionary has keys: 'name', 'description', 'acceptance_criteria'
        """
        node.tasks = await self.ai_provider.generate_structured_response(prompt)

    async def _optimize_tasks(self, root_node: Node):
        all_tasks = self._extract_all_tasks(root_node)
        
        optimization_prompt = f"""
        Analyze the following list of tasks from all nodes in the plan:
        {all_tasks}
        
        1. Identify and remove any redundant tasks.
        2. Identify tasks that serve as prerequisites for multiple parts and consolidate them.
        3. Ensure that the tasks for each node follow the MECE (Mutually Exclusive, Collectively Exhaustive) framework.
        
        Output a JSON dictionary where:
        - Keys are node names
        - Values are lists of optimized tasks (dictionaries with 'name', 'description', 'acceptance_criteria')
        """
        
        optimized_tasks = await self.ai_provider.generate_structured_response(optimization_prompt)
        
        self._update_node_tasks(root_node, optimized_tasks)

    def _extract_all_tasks(self, node: Node) -> List[Dict[str, Any]]:
        all_tasks = [{'node': node.name, 'task': task} for task in node.tasks]
        for prereq in node.prerequisites:
            all_tasks.extend(self._extract_all_tasks(prereq))
        return all_tasks

    def _update_node_tasks(self, node: Node, optimized_tasks: Dict[str, List[Dict[str, Any]]]):
        if node.name in optimized_tasks:
            node.tasks = optimized_tasks[node.name]
        for prereq in node.prerequisites:
            self._update_node_tasks(prereq, optimized_tasks)

    async def _conduct_premortem(self, node: Node):
        prompt = f"""
        Conduct a premortem analysis for the following plan step:
        Step: {node.name}
        Description: {node.description}
        Tasks: {node.tasks}

        Imagine that this step has failed. Describe:
        1. What went wrong
        2. Why it went wrong
        3. The probability of this failure occurring (0.0 to 1.0)
        4. The impact of this failure (-10 to 10, where -10 is catastrophic and 10 is unexpectedly beneficial)

        Output the result as a JSON list of dictionaries, each with keys 'description', 'reason', 'probability', and 'impact'.
        """
        node.failure_modes = await self.ai_provider.generate_structured_response(prompt)
        
        if self.config.parallelization:
            await asyncio.gather(*[self._conduct_premortem(prereq) for prereq in node.prerequisites])
        else:
            for prereq in node.prerequisites:
                await self._conduct_premortem(prereq)

    async def _assess_antifragility(self, node: Node):
        if self.config.parallelization:
            await asyncio.gather(
                self._assess_node_antifragility(node),
                *[self._assess_antifragility(prereq) for prereq in node.prerequisites]
            )
        else:
            await self._assess_node_antifragility(node)
            for prereq in node.prerequisites:
                await self._assess_antifragility(prereq)

    async def _assess_node_antifragility(self, node: Node):
        prompt = f"""
        Assess the antifragility of the following plan step:
        Step: {node.name}
        Description: {node.description}
        Tasks: {node.tasks}
        Failure Modes: {node.failure_modes}

        Consider:
        1. How could this step benefit or suffer from stress or failure?
        2. What learning opportunities are present in potential failures?
        3. How adaptable is this step to unexpected changes?

        Rate the antifragility on a scale from -10 to 10, where:
        - -10 is extremely fragile (breaks easily under stress)
        - 0 is robust (withstands stress but doesn't improve)
        - 10 is highly antifragile (gets stronger under stress)

        Output a JSON with keys:
        - 'antifragility_score': a float from -10.0 to 10.0
        - 'reasoning': a string explaining the score
        - 'improvement_suggestions': a list of suggestions to increase antifragility
        """
        result = await self.ai_provider.generate_structured_response(prompt)
        node.antifragility_score = result['antifragility_score']
        return result

    async def _develop_xanatos_gambits(self, node: Node):
        prompt = f"""
        Develop Xanatos Gambits for the following plan step:
        Step: {node.name}
        Description: {node.description}
        Tasks: {node.tasks}
        Failure Modes: {node.failure_modes}

        For each failure mode, develop an alternative path that turns the apparent failure into a success.
        Consider how these alternative paths might synergize with each other or with the original plan.

        Rate the Xanatos factor on a scale from -10 to 10, where:
        - -10: The plan backfires completely
        - 0: Neutral outcome regardless of success or failure
        - 10: The plan succeeds spectacularly regardless of apparent success or failure

        Output a JSON with keys:
        - 'xanatos_factor': a float from -10.0 to 10.0
        - 'gambits': a list of dictionaries, each with keys 'failure_mode', 'alternative_path', and 'synergy_description'
        - 'reasoning': a string explaining the Xanatos factor score
        """
        result = await self.ai_provider.generate_structured_response(prompt)
        node.xanatos_factor = result['xanatos_factor']
        node.xanatos_gambits = result['gambits']
        
        if self.config.parallelization:
            await asyncio.gather(*[self._develop_xanatos_gambits(prereq) for prereq in node.prerequisites])
        else:
            for prereq in node.prerequisites:
                await self._develop_xanatos_gambits(prereq)

    async def _update_plan(self, node: Node) -> Node:
        node.expected_utility = self._calculate_expected_utility(node)
        
        update_prompt = f"""
        Given the following information about a plan step:
        Step: {node.name}
        Description: {node.description}
        Tasks: {node.tasks}
        Failure Modes: {node.failure_modes}
        Antifragility Score: {node.antifragility_score}
        Xanatos Factor: {node.xanatos_factor}
        Xanatos Gambits: {node.xanatos_gambits}
        Expected Utility: {node.expected_utility}

        Suggest improvements to make this step more robust, antifragile, and aligned with Xanatos Gambit principles.
        Output a JSON with keys 'name', 'description', and 'tasks', representing the updated step.
        """
        updated_node_data = await self.ai_provider.generate_structured_response(update_prompt)
        
        node.name = updated_node_data['name']
        node.description = updated_node_data['description']
        node.tasks = updated_node_data['tasks']
        
        if self.config.parallelization:
            updated_prerequisites = await asyncio.gather(*[self._update_plan(prereq) for prereq in node.prerequisites])
            node.prerequisites = updated_prerequisites
        else:
            for i, prereq in enumerate(node.prerequisites):
                node.prerequisites[i] = await self._update_plan(prereq)
        
        return node

    def _calculate_expected_utility(self, node: Node) -> float:
        success_utility = node.probability * 10  # Max success utility is 10

        failure_utilities = []
        for f in node.failure_modes:
            impact = f['impact']  # Already -10 to 10
            antifragility = node.antifragility_score  # -10 to 10
            xanatos = node.xanatos_factor  # -10 to 10
            
            # Calculate a weighted sum, giving more weight to positive factors
            weighted_sum = (
                max(0, impact) * 1.5 + 
                min(0, impact) * 0.5 +
                max(0, antifragility) * 1.5 + 
                min(0, antifragility) * 0.5 +
                max(0, xanatos) * 1.5 + 
                min(0, xanatos) * 0.5
            ) / 4.5  # Normalize by sum of weights

            # Adjust for the probability of this failure mode not occurring
            failure_utility = (1 - f['probability']) * weighted_sum
            failure_utilities.append(failure_utility)

        # Take the average of failure utilities
        avg_failure_utility = sum(failure_utilities) / len(failure_utilities) if failure_utilities else 0

        # Combine success and failure utilities
        total_utility = (success_utility + avg_failure_utility) / 2

        # Ensure the result is within the -10 to 10 range
        return max(min(total_utility, 10), -10)

    def _calculate_success_likelihood(self, node: Node) -> float:
        if not node.prerequisites:
            return node.probability
        return node.probability * min(self._calculate_success_likelihood(prereq) for prereq in node.prerequisites)

    def _identify_capability_gaps(self, plan_tree: Node, current_resources: Dict[str, Any]) -> List[str]:
        gaps = []
        self._find_gaps(plan_tree, current_resources, gaps)
        return gaps

    def _find_gaps(self, node: Node, resources: Dict[str, Any], gaps: List[str]):
        if not node.prerequisites and not self._is_resource(node.name, resources):
            gaps.append(node.name)
        for prereq in node.prerequisites:
            self._find_gaps(prereq, resources, gaps)

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

    def _calculate_plan_metrics(self, plan_tree: Node) -> Dict[str, float]:
        metrics = {
            'total_nodes': 0,
            'total_tasks': 0,
            'average_antifragility': 0,
            'average_xanatos_factor': 0,
            'overall_expected_utility': 0,
            'fragile_nodes': 0,
            'robust_nodes': 0,
            'antifragile_nodes': 0,
            'negative_xanatos_nodes': 0,
            'neutral_xanatos_nodes': 0,
            'positive_xanatos_nodes': 0,
        }

        def traverse(node: Node):
            metrics['total_nodes'] += 1
            metrics['total_tasks'] += len(node.tasks)
            metrics['average_antifragility'] += node.antifragility_score
            metrics['average_xanatos_factor'] += node.xanatos_factor
            metrics['overall_expected_utility'] += node.expected_utility

            if node.antifragility_score < -3:
                metrics['fragile_nodes'] += 1
            elif node.antifragility_score > 3:
                metrics['antifragile_nodes'] += 1
            else:
                metrics['robust_nodes'] += 1

            if node.xanatos_factor < -3:
                metrics['negative_xanatos_nodes'] += 1
            elif node.xanatos_factor > 3:
                metrics['positive_xanatos_nodes'] += 1
            else:
                metrics['neutral_xanatos_nodes'] += 1

            for prereq in node.prerequisites:
                traverse(prereq)

        traverse(plan_tree)

        if metrics['total_nodes'] > 0:
            metrics['average_antifragility'] /= metrics['total_nodes']
            metrics['average_xanatos_factor'] /= metrics['total_nodes']
            metrics['overall_expected_utility'] /= metrics['total_nodes']

        return metrics

    def _tree_to_dict(self, node: Node) -> Dict[str, Any]:
        return {
            'name': node.name,
            'description': node.description,
            'probability': node.probability,
            'tasks': node.tasks,
            'failure_modes': node.failure_modes,
            'antifragility_score': node.antifragility_score,
            'xanatos_factor': node.xanatos_factor,
            'xanatos_gambits': node.xanatos_gambits,
            'expected_utility': node.expected_utility,
            'prerequisites': [self._tree_to_dict(prereq) for prereq in node.prerequisites]
        }

    def _create_plan_visualization(self, plan_tree: Node) -> str:
        G = nx.DiGraph()
        
        def add_nodes(node: Node, parent=None):
            # Determine node color based on antifragility
            if node.antifragility_score < -3:
                anti_color = 'red'
            elif node.antifragility_score > 3:
                anti_color = 'green'
            else:
                anti_color = 'yellow'
            
            # Determine node shape based on Xanatos factor
            if node.xanatos_factor < -3:
                shape = 's'  # square
            elif node.xanatos_factor > 3:
                shape = '^'  # triangle up
            else:
                shape = 'o'  # circle
            
            G.add_node(node.name, 
                       description=node.description, 
                       antifragility=node.antifragility_score,
                       xanatos_factor=node.xanatos_factor,
                       expected_utility=node.expected_utility,
                       color=anti_color,
                       shape=shape)
            if parent:
                G.add_edge(parent.name, node.name)
            for prereq in node.prerequisites:
                add_nodes(prereq, node)
        
        add_nodes(plan_tree)
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=(20, 20))
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        node_shapes = [G.nodes[node]['shape'] for node in G.nodes()]
        
        for shape in set(node_shapes):
            node_list = [node for node in G.nodes() if G.nodes[node]['shape'] == shape]
            nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=[G.nodes[node]['color'] for node in node_list], 
                                   node_shape=shape, node_size=3000)
        
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, {node: node for node in G.nodes()}, font_size=8, font_weight='bold')
        
        node_labels = nx.get_node_attributes(G, 'description')
        pos_attrs = {}
        for node, coords in pos.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.08)
        nx.draw_networkx_labels(G, pos_attrs, labels=node_labels, font_size=6)
        
        plt.title("Plan Tree Visualization\nColors: Red (Fragile), Yellow (Robust), Green (Antifragile)\nShapes: Square (Negative Xanatos), Circle (Neutral Xanatos), Triangle (Positive Xanatos)")
        plt.axis('off')
        
        filename = 'plan_visualization.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename

    async def incorporate_feedback(self, plan_file: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        plan_data = load_plan_from_file(plan_file)
        plan_tree = self._dict_to_tree(plan_data['plan_tree'])

        feedback_prompt = f"""
        Given the following plan and real-world feedback:
        Plan: {json.dumps(plan_data, indent=2)}
        Feedback: {json.dumps(feedback, indent=2)}

        Suggest updates to the plan to incorporate this feedback. Consider:
        1. How does the feedback affect our understanding of the problem?
        2. What parts of the plan need to be adjusted based on this new information?
        3. How can we leverage this feedback to improve the plan's antifragility and Xanatos factors?

        Output a JSON with the following structure:
        {{
            "updated_nodes": [
                {{
                    "node_name": "Name of the node to update",
                    "updates": {{
                        "description": "Updated description",
                        "tasks": [{{
                            "name": "Updated task name",
                            "description": "Updated task description",
                            "acceptance_criteria": "Updated acceptance criteria"
                        }}],
                        "antifragility_score": -10.0 to 10.0,
                        "xanatos_factor": -10.0 to 10.0
                    }}
                }}
            ],
            "new_nodes": [
                {{
                    "name": "New node name",
                    "description": "New node description",
                    "parent_node": "Name of the parent node",
                    "tasks": [...],
                    "antifragility_score": -10.0 to 10.0,
                    "xanatos_factor": -10.0 to 10.0
                }}
            ]
        }}
        """

        updates = await self.ai_provider.generate_structured_response(feedback_prompt)

        # Apply updates to existing nodes
        for update in updates['updated_nodes']:
            node = self._find_node(plan_tree, update['node_name'])
            if node:
                self._update_node_with_feedback(node, update['updates'])

        # Add new nodes
        for new_node_data in updates['new_nodes']:
            parent_node = self._find_node(plan_tree, new_node_data['parent_node'])
            if parent_node:
                new_node = Node(
                    name=new_node_data['name'],
                    description=new_node_data['description'],
                    tasks=new_node_data['tasks'],
                    antifragility_score=new_node_data['antifragility_score'],
                    xanatos_factor=new_node_data['xanatos_factor']
                )
                parent_node.prerequisites.append(new_node)

        # Recalculate metrics and update plan data
        plan_data['plan_tree'] = self._tree_to_dict(plan_tree)
        plan_data['metrics'] = self._calculate_plan_metrics(plan_tree)

        # Save the updated plan
        save_plan_to_file(plan_data, plan_file)

        return plan_data

    def _find_node(self, node: Node, name: str) -> Optional[Node]:
        if node.name == name:
            return node
        for prereq in node.prerequisites:
            found = self._find_node(prereq, name)
            if found:
                return found
        return None

    def _update_node_with_feedback(self, node: Node, updates: Dict[str, Any]):
        node.description = updates.get('description', node.description)
        node.tasks = updates.get('tasks', node.tasks)
        node.antifragility_score = updates.get('antifragility_score', node.antifragility_score)
        node.xanatos_factor = updates.get('xanatos_factor', node.xanatos_factor)

    def _dict_to_tree(self, node_dict: Dict[str, Any]) -> Node:
        node = Node(
            name=node_dict['name'],
            description=node_dict['description'],
            probability=node_dict['probability'],
            tasks=node_dict['tasks'],
            failure_modes=node_dict['failure_modes'],
            antifragility_score=node_dict['antifragility_score'],
            xanatos_factor=node_dict['xanatos_factor'],
            xanatos_gambits=node_dict['xanatos_gambits'],
            expected_utility=node_dict['expected_utility']
        )
        node.prerequisites = [self._dict_to_tree(prereq) for prereq in node_dict['prerequisites']]
        return node

# Utility functions

def save_plan_to_file(plan_data: Dict[str, Any], filename: str):
    with open(filename, 'w') as f:
        json.dump(plan_data, f, indent=2)

def load_plan_from_file(filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as f:
        return json.load(f)