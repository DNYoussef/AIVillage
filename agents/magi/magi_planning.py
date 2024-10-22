import math
import random
from typing import List, Dict, Any, Tuple
import asyncio
from collections import defaultdict
from matplotlib import pyplot as plt
import networkx as nx
import json
import logging
import io
import sys
import time
from functools import lru_cache

from agents.language_models.openai_gpt import OpenAIGPTConfig
from agents.quality_assurance_layer import QualityAssuranceLayer
from agents.utils.exceptions import AIVillageException
from communications.protocol import StandardCommunicationProtocol

logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

class GraphManager:
    def __init__(self):
        self.G = nx.DiGraph()

    def add_agent_node(self, agent_id: str, attributes: Dict[str, Any]):
        self.G.add_node(agent_id, **attributes, type='agent')
        logger.info(f"Added agent node: {agent_id} with attributes: {attributes}")

    def add_task_node(self, task_id: str, attributes: Dict[str, Any]):
        self.G.add_node(task_id, **attributes, type='task')
        logger.info(f"Added task node: {task_id} with attributes: {attributes}")

    def update_agent_experience(self, agent_id: str, task_id: str, performance: float):
        if self.G.has_edge(agent_id, task_id):
            self.G[agent_id][task_id]['weight'] *= (1 + performance)
            logger.info(f"Updated edge weight between {agent_id} and {task_id} to {self.G[agent_id][task_id]['weight']}")
        else:
            self.G.add_edge(agent_id, task_id, weight=performance)
            logger.info(f"Added edge from {agent_id} to {task_id} with weight {performance}")

    def merge_task_graph(self, task_graph: nx.DiGraph):
        self.G = nx.compose(self.G, task_graph)
        logger.info("Merged new task graph into existing agent graph")

    def get_graph(self) -> nx.DiGraph:
        return self.G

    def visualize_graph(self, filename: str = "agent_task_graph.png"):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.G)
        node_colors = ['lightblue' if data['type'] == 'agent' else 'lightgreen' for _, data in self.G.nodes(data=True)]
        nx.draw(self.G, pos, with_labels=True, node_color=node_colors, node_size=1500, font_size=10, arrows=True)
        plt.title("Agent and Task Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logger.info(f"Graph visualization saved as {filename}")

class MagiPlanning:
    """
    MagiPlanning is a sophisticated planning and problem-solving class that integrates
    various advanced reasoning techniques to generate, optimize, and execute plans.
    
    It uses techniques such as Tree-of-Thoughts, Chain-of-Thought, Program-of-Thoughts,
    Least-to-Most Prompting, and Contrastive Chain-of-Thought to approach complex problems.
    The class also incorporates antifragility assessments, premortem analyses, and
    Xanatos Gambits to create robust and adaptable plans.
    """

    def __init__(self, communication_protocol: StandardCommunicationProtocol, quality_assurance_layer: QualityAssuranceLayer, graph_manager: GraphManager):
        """
        Initialize the MagiPlanning instance.

        Args:
            communication_protocol (StandardCommunicationProtocol): Protocol for communication with other components.
            quality_assurance_layer (QualityAssuranceLayer): Layer for ensuring quality and safety of operations.
            graph_manager (GraphManager): Manager for handling graph-based representations of plans and tasks.
        """
        self.communication_protocol = communication_protocol
        self.qa_layer = quality_assurance_layer
        self.graph_manager = graph_manager
        self.llm = OpenAIGPTConfig(chat_model="gpt-4").create()
        self.technique_performance = defaultdict(list)

    async def tree_of_thoughts(self, goal: str, depth: int = 2, branches: int = 3) -> Dict[str, Any]:
        async def expand_node(node: str, current_depth: int) -> Dict[str, Any]:
            if current_depth == 0:
                return {"thought": node, "children": []}
            
            prompt = f"Given the goal: '{goal}' and the current thought: '{node}', generate {branches} possible next steps or sub-goals."
            response = await self.llm.complete(prompt)
            next_thoughts = response.text.split('\n')[:branches]
            
            children = []
            for thought in next_thoughts:
                child = await expand_node(thought, current_depth - 1)
                children.append(child)
            
            return {"thought": node, "children": children}

        root_prompt = f"Given the goal: '{goal}', provide an initial thought or approach to achieve this goal."
        root_response = await self.llm.complete(root_prompt)
        root_thought = root_response.text.strip()

        tree = await expand_node(root_thought, depth)

        evaluation_prompt = f"Evaluate the following tree of thoughts for the goal: '{goal}'. Provide a score from 0 to 10 for each leaf node, where 10 is the most promising approach. Return the scores as a JSON object."
        evaluation_response = await self.llm.complete(evaluation_prompt + "\n" + json.dumps(tree, indent=2))
        
        try:
            scores = json.loads(evaluation_response.text)
        except json.JSONDecodeError:
            logger.error("Failed to parse evaluation scores")
            scores = {}

        def find_best_path(node: Dict[str, Any], current_path: List[str]) -> Tuple[List[str], float]:
            if not node["children"]:
                return current_path + [node["thought"]], scores.get(node["thought"], 0)
            
            best_child_path, best_child_score = max(
                (find_best_path(child, current_path + [node["thought"]]) for child in node["children"]),
                key=lambda x: x[1]
            )
            return best_child_path, best_child_score

        best_path, best_score = find_best_path(tree, [])

        return {
            "tree": tree,
            "best_path": best_path,
            "best_score": best_score
        }

    async def chain_of_thought(self, problem: str) -> Dict[str, Any]:
        prompt = f"""
        Solve the following problem using a step-by-step approach:
        
        Problem: {problem}
        
        Please provide your reasoning in the following format:
        1. [First step of reasoning]
        2. [Second step of reasoning]
        3. [Third step of reasoning]
        ...
        
        Final Answer: [Your final answer]
        
        Explain your thought process clearly at each step.
        """
        
        response = await self.llm.complete(prompt)
        
        # Parse the response to extract steps and final answer
        lines = response.text.split('\n')
        steps = []
        final_answer = ""
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                steps.append(line.strip())
            elif line.strip().startswith("Final Answer:"):
                final_answer = line.strip()[len("Final Answer:"):].strip()
        
        return {
            "steps": steps,
            "final_answer": final_answer
        }

    async def program_of_thoughts(self, problem: str) -> Dict[str, Any]:
        prompt = f"""
        Solve the following problem by writing a Python program:
        
        Problem: {problem}
        
        Please provide your solution in the following format:
        1. Explain your approach
        2. Write the Python code
        3. Explain how the code solves the problem
        
        Make sure to include print statements in your code to show the intermediate steps and the final result.
        """
        
        response = await self.llm.complete(prompt)
        
        # Parse the response to extract explanation, code, and solution explanation
        sections = response.text.split('\n\n')
        approach = sections[0] if len(sections) > 0 else ""
        code = sections[1] if len(sections) > 1 else ""
        explanation = sections[2] if len(sections) > 2 else ""
        
        # Execute the code in a safe environment
        result = self._execute_code_safely(code)
        
        return {
            "approach": approach,
            "code": code,
            "explanation": explanation,
            "execution_result": result
        }

    def _execute_code_safely(self, code: str) -> str:
        # Create a string buffer to capture printed output
        buffer = io.StringIO()
        sys.stdout = buffer

        try:
            # Execute the code
            exec(code, {'__builtins__': __builtins__}, {})
            output = buffer.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}"
        finally:
            # Restore the standard output
            sys.stdout = sys.__stdout__

        return output

    async def least_to_most_prompting(self, problem: str) -> Dict[str, Any]:
        # Step 1: Break down the problem into sub-problems
        decomposition_prompt = f"""
        Break down the following problem into a list of simpler sub-problems, ordered from least to most complex:
        
        Problem: {problem}
        
        Output the sub-problems as a JSON list of strings.
        """
        decomposition_response = await self.llm.complete(decomposition_prompt)
        sub_problems = json.loads(decomposition_response.text)
        
        # Step 2: Solve each sub-problem sequentially
        context = []
        solutions = []
        
        for i, sub_problem in enumerate(sub_problems):
            solve_prompt = f"""
            Solve the following sub-problem:
            
            Sub-problem: {sub_problem}
            
            Previous context and solutions:
            {json.dumps(context, indent=2)}
            
            Provide a step-by-step solution to this sub-problem, using the previous context and solutions if relevant.
            """
            
            solution_response = await self.llm.complete(solve_prompt)
            solution = solution_response.text
            
            context.append({"sub_problem": sub_problem, "solution": solution})
            solutions.append(solution)
        
        # Step 3: Combine solutions to address the original problem
        combination_prompt = f"""
        Using the solutions to the sub-problems, provide a comprehensive solution to the original problem:
        
        Original problem: {problem}
        
        Sub-problems and their solutions:
        {json.dumps(context, indent=2)}
        
        Combine these solutions to address the original problem. Provide a step-by-step explanation of the final solution.
        """
        
        final_solution_response = await self.llm.complete(combination_prompt)
        final_solution = final_solution_response.text
        
        return {
            "original_problem": problem,
            "sub_problems": sub_problems,
            "sub_solutions": solutions,
            "final_solution": final_solution
        }

    async def contrastive_chain_of_thought(self, problem: str) -> Dict[str, Any]:
        prompt = f"""
        Solve the following problem using a contrastive chain-of-thought approach:
        
        Problem: {problem}
        
        Please provide your reasoning in the following format:
        1. Correct approach:
           [Step-by-step correct reasoning]
        
        2. Incorrect approach:
           [Step-by-step incorrect reasoning]
        
        3. Contrast and analysis:
           [Compare and contrast the correct and incorrect approaches, explaining why the correct approach is better]
        
        4. Final Answer:
           [Your final answer based on the correct approach]
        
        Explain your thought process clearly at each step.
        """
        
        response = await self.llm.complete(prompt)
        
        # Parse the response to extract the different sections
        sections = response.text.split('\n\n')
        correct_approach = sections[0] if len(sections) > 0 else ""
        incorrect_approach = sections[1] if len(sections) > 1 else ""
        contrast_analysis = sections[2] if len(sections) > 2 else ""
        final_answer = sections[3] if len(sections) > 3 else ""
        
        return {
            "correct_approach": correct_approach,
            "incorrect_approach": incorrect_approach,
            "contrast_analysis": contrast_analysis,
            "final_answer": final_answer
        }

    async def choose_technique(self, task: str) -> str:
        """
        Dynamically choose the most appropriate technique based on the task.
        """
        prompt = f"""
        Given the following task, choose the most appropriate technique from the list below:
        
        Task: {task}
        
        Techniques:
        1. Tree-of-Thoughts
        2. Chain-of-Thought
        3. Program-of-Thoughts
        4. Least-to-Most Prompting
        5. Contrastive Chain-of-Thought
        
        Provide your answer as a single string (e.g., "Tree-of-Thoughts").
        Explain your reasoning for choosing this technique.
        """
        
        response = await self.llm.complete(prompt)
        lines = response.text.split('\n')
        chosen_technique = lines[0].strip()
        explanation = '\n'.join(lines[1:]).strip()
        
        logger.info(f"Chosen technique for task '{task}': {chosen_technique}")
        logger.info(f"Reasoning: {explanation}")
        
        return chosen_technique

    @lru_cache(maxsize=100)
    async def execute_task(self, task: str) -> Dict[str, Any]:
        """
        Execute a task using the most appropriate technique.
        Results are cached for similar tasks.
        """
        technique = await self.choose_technique(task)
        
        start_time = time.time()
        try:
            if technique == "Tree-of-Thoughts":
                result = await self.tree_of_thoughts(task)
            elif technique == "Chain-of-Thought":
                result = await self.chain_of_thought(task)
            elif technique == "Program-of-Thoughts":
                result = await self.program_of_thoughts(task)
            elif technique == "Least-to-Most Prompting":
                result = await self.least_to_most_prompting(task)
            elif technique == "Contrastive Chain-of-Thought":
                result = await self.contrastive_chain_of_thought(task)
            else:
                raise ValueError(f"Unknown technique: {technique}")
            
            execution_time = time.time() - start_time
            logger.info(f"Task executed successfully using {technique} in {execution_time:.2f} seconds")
            
            self.technique_performance[technique].append(execution_time)
            
            return {
                "task": task,
                "technique": technique,
                "result": result,
                "execution_time": execution_time
            }
        except Exception as e:
            logger.exception(f"Error executing task '{task}' with technique '{technique}': {str(e)}")
            raise AIVillageException(f"Error executing task: {str(e)}") from e

    async def choose_technique(self, task: str) -> str:
        """
        Dynamically choose the most appropriate technique based on the task and past performance.
        """
        prompt = f"""
        Given the following task and performance history, choose the most appropriate technique:
        
        Task: {task}
        
        Performance History:
        {json.dumps(dict(self.technique_performance), indent=2)}
        
        Techniques:
        1. Tree-of-Thoughts
        2. Chain-of-Thought
        3. Program-of-Thoughts
        4. Least-to-Most Prompting
        5. Contrastive Chain-of-Thought
        
        Provide your answer as a single string (e.g., "Tree-of-Thoughts").
        Explain your reasoning for choosing this technique, considering both the task characteristics and past performance.
        """
        
        response = await self.llm.complete(prompt)
        lines = response.text.split('\n')
        chosen_technique = lines[0].strip()
        explanation = '\n'.join(lines[1:]).strip()
        
        logger.info(f"Chosen technique for task '{task}': {chosen_technique}")
        logger.info(f"Reasoning: {explanation}")
        
        return chosen_technique

    async def generate_plan(self, goal: str, problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive plan for achieving the given goal.

        This method orchestrates the entire planning process, including:
        - Creating an initial plan tree
        - Executing each step of the plan using the most appropriate technique
        - Optimizing tasks
        - Conducting premortem analysis
        - Assessing antifragility
        - Developing Xanatos Gambits
        - Calculating success likelihood
        - Identifying capability gaps
        - Planning checkpoints
        - Performing SWOT analysis

        Args:
            goal (str): The main goal to be achieved.
            problem_analysis (Dict[str, Any]): Initial analysis of the problem.

        Returns:
            Dict[str, Any]: A comprehensive plan data structure including the plan tree,
                            various analyses, metrics, and visualizations.
        """
        try:
            start_time = time.time()
            
            current_resources = await self._get_current_resources()
            plan_tree = await self._create_plan_tree(goal, current_resources)
            
            for step in self._extract_all_tasks(plan_tree):
                step_result = await self.execute_task(step['description'])
                step['result'] = step_result
            
            await self._optimize_tasks(plan_tree)
            
            success_likelihood = 0
            iteration = 0
            max_iterations = 10
            success_threshold = 0.95
            
            while success_likelihood < success_threshold and iteration < max_iterations:
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

            visualization = self._create_plan_visualization(plan_tree)

            execution_time = time.time() - start_time
            logger.info(f"Plan generated in {execution_time:.2f} seconds")

            plan_data = {
                'goal': goal,
                'plan_tree': self._tree_to_dict(plan_tree),
                'capability_gaps': gaps,
                'checkpoints': checkpoints,
                'swot_analysis': swot_analysis,
                'success_likelihood': success_likelihood,
                'iterations': iteration,
                'metrics': metrics,
                'visualization': visualization,
                'execution_time': execution_time,
                'technique_performance': dict(self.technique_performance),
                'explanations': await self._generate_explanations(plan_tree)
            }

            return plan_data
        except Exception as e:
            logger.exception(f"Error in generate_plan: {str(e)}")
            raise AIVillageException(f"Error in generate_plan: {str(e)}") from e

    @lru_cache(maxsize=100)
    async def execute_task(self, task: str) -> Dict[str, Any]:
        """
        Execute a task using the most appropriate technique.
        Results are cached for similar tasks to improve efficiency.

        Args:
            task (str): The task to be executed.

        Returns:
            Dict[str, Any]: A dictionary containing the task execution results,
                            including the chosen technique and execution time.
        """
        technique = await self.choose_technique(task)
        
        start_time = time.time()
        try:
            if technique == "Tree-of-Thoughts":
                result = await self.tree_of_thoughts(task)
            elif technique == "Chain-of-Thought":
                result = await self.chain_of_thought(task)
            elif technique == "Program-of-Thoughts":
                result = await self.program_of_thoughts(task)
            elif technique == "Least-to-Most Prompting":
                result = await self.least_to_most_prompting(task)
            elif technique == "Contrastive Chain-of-Thought":
                result = await self.contrastive_chain_of_thought(task)
            else:
                raise ValueError(f"Unknown technique: {technique}")
            
            execution_time = time.time() - start_time
            logger.info(f"Task executed successfully using {technique} in {execution_time:.2f} seconds")
            
            self.technique_performance[technique].append(execution_time)
            
            return {
                "task": task,
                "technique": technique,
                "result": result,
                "execution_time": execution_time
            }
        except Exception as e:
            logger.exception(f"Error executing task '{task}' with technique '{technique}': {str(e)}")
            raise AIVillageException(f"Error executing task: {str(e)}") from e

    async def choose_technique(self, task: str) -> str:
        """
        Dynamically choose the most appropriate technique based on the task and past performance.

        Args:
            task (str): The task for which to choose a technique.

        Returns:
            str: The name of the chosen technique.
        """
        prompt = f"""
        Given the following task and performance history, choose the most appropriate technique:
        
        Task: {task}
        
        Performance History:
        {json.dumps(dict(self.technique_performance), indent=2)}
        
        Techniques:
        1. Tree-of-Thoughts
        2. Chain-of-Thought
        3. Program-of-Thoughts
        4. Least-to-Most Prompting
        5. Contrastive Chain-of-Thought
        
        Provide your answer as a single string (e.g., "Tree-of-Thoughts").
        Explain your reasoning for choosing this technique, considering both the task characteristics and past performance.
        """
        
        response = await self.llm.complete(prompt)
        lines = response.text.split('\n')
        chosen_technique = lines[0].strip()
        explanation = '\n'.join(lines[1:]).strip()
        
        logger.info(f"Chosen technique for task '{task}': {chosen_technique}")
        logger.info(f"Reasoning: {explanation}")
        
        return chosen_technique

    async def _generate_explanations(self, plan_tree: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate explanations for the decisions made during the planning process.
        """
        prompt = f"""
        Given the following plan tree, generate explanations for the key decisions made during the planning process:
        
        Plan Tree: {json.dumps(plan_tree, indent=2)}
        
        Provide explanations for:
        1. Choice of main approach
        2. Key sub-goals and their importance
        3. Risk mitigation strategies
        4. Resource allocation decisions
        5. Antifragility considerations
        6. Xanatos Gambits employed
        
        Output the explanations as a JSON dictionary with the above points as keys.
        """
        
        explanations = await self._generate_structured_response(prompt)
        return explanations

    async def _get_current_resources(self) -> Dict[str, Any]:
        prompt = "List the current resources, position, and tools available for the AI Village project. Output as a JSON dictionary with keys 'resources', 'position', and 'tools'."
        try:
            return await self._generate_structured_response(prompt)
        except Exception as e:
            logger.exception(f"Error getting current resources: {str(e)}")
            raise AIVillageException(f"Error getting current resources: {str(e)}") from e

    async def _create_plan_tree(self, goal: str, current_resources: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = f"""
            Create a detailed plan tree for the goal: "{goal}"
            Consider the following current resources: {json.dumps(current_resources)}
            
            The plan tree should include:
            1. Main goal
            2. Sub-goals (multiple levels if necessary)
            3. Tasks for each sub-goal
            4. Dependencies between tasks and sub-goals
            5. Estimated time and resources required for each task
            6. Potential risks and mitigation strategies
            
            Output the plan tree as a nested JSON structure.
            """
            plan_tree = await self._generate_structured_response(prompt)
            return plan_tree
        except Exception as e:
            logger.exception(f"Error creating plan tree: {str(e)}")
            raise AIVillageException(f"Error creating plan tree: {str(e)}") from e

    async def _extract_tasks(self, plan_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        tasks = []
        
        def extract_tasks_recursive(node):
            if 'tasks' in node:
                tasks.extend(node['tasks'])
            for sub_goal in node.get('sub_goals', []):
                extract_tasks_recursive(sub_goal)
        
        extract_tasks_recursive(plan_tree)
        return tasks

    async def _optimize_tasks(self, plan_tree: Dict[str, Any]) -> Dict[str, Any]:
        try:
            tasks = await self._extract_tasks(plan_tree)
            prompt = f"""
            Optimize the following set of tasks:
            {json.dumps(tasks, indent=2)}
            
            Consider the following aspects for optimization:
            1. Task dependencies and potential parallelization
            2. Resource allocation and efficiency
            3. Risk mitigation and contingency planning
            4. Time estimation accuracy
            
            Provide an optimized version of the tasks, maintaining the same structure.
            """
            optimized_tasks = await self._generate_structured_response(prompt)
            
            # Update the plan tree with optimized tasks
            def update_tasks_recursive(node):
                if 'tasks' in node:
                    node['tasks'] = [task for task in optimized_tasks if task['id'] in [t['id'] for t in node['tasks']]]
                for sub_goal in node.get('sub_goals', []):
                    update_tasks_recursive(sub_goal)
            
            update_tasks_recursive(plan_tree)
            return plan_tree
        except Exception as e:
            logger.exception(f"Error optimizing tasks: {str(e)}")
            raise AIVillageException(f"Error optimizing tasks: {str(e)}") from e

    async def _conduct_premortem(self, plan_tree: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = f"""
            Conduct a premortem analysis for the following plan:
            {json.dumps(plan_tree, indent=2)}
            
            Imagine that the plan has failed spectacularly. Identify:
            1. Potential points of failure
            2. Unforeseen risks or challenges
            3. Assumptions that might prove false
            4. External factors that could negatively impact the plan
            
            For each identified issue, suggest preventive measures or plan adjustments.
            Output the results as a JSON structure with 'issues' and 'preventive_measures' keys.
            """
            premortem_results = await self._generate_structured_response(prompt)
            
            # Update the plan tree with premortem insights
            plan_tree['premortem_analysis'] = premortem_results
            return plan_tree
        except Exception as e:
            logger.exception(f"Error conducting premortem: {str(e)}")
            raise AIVillageException(f"Error conducting premortem: {str(e)}") from e

    async def _assess_antifragility(self, plan_tree: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = f"""
            Assess the antifragility of the following plan:
            {json.dumps(plan_tree, indent=2)}
            
            For each component of the plan, evaluate:
            1. How it responds to volatility, randomness, and chaos
            2. Its potential to gain from disorder
            3. Its resilience and adaptability
            
            Provide an antifragility score (-10 to 10) for each component, where:
            - Negative scores indicate fragility
            - Scores around 0 indicate robustness
            - Positive scores indicate antifragility
            
            Suggest improvements to increase the overall antifragility of the plan.
            Output the results as a JSON structure with 'antifragility_scores' and 'improvement_suggestions' keys.
            """
            antifragility_assessment = await self._generate_structured_response(prompt)
            
            # Update the plan tree with antifragility assessment
            plan_tree['antifragility_assessment'] = antifragility_assessment
            return plan_tree
        except Exception as e:
            logger.exception(f"Error assessing antifragility: {str(e)}")
            raise AIVillageException(f"Error assessing antifragility: {str(e)}") from e

    async def _develop_xanatos_gambits(self, plan_tree: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = f"""
            Develop Xanatos Gambits for the following plan:
            {json.dumps(plan_tree, indent=2)}
            
            A Xanatos Gambit is a plan that benefits the planner regardless of the outcome.
            For each major component or goal in the plan:
            1. Identify potential failure scenarios
            2. Devise strategies that turn these failures into alternative successes
            3. Explain how each gambit benefits the overall goal, even if the original plan fails
            
            Output the results as a JSON structure with 'gambits' key, where each gambit includes:
            - 'component': The part of the plan it applies to
            - 'failure_scenario': The potential failure
            - 'alternative_success': How the failure is turned into a success
            - 'benefit_explanation': How this gambit benefits the overall goal
            """
            xanatos_gambits = await self._generate_structured_response(prompt)
            
            # Update the plan tree with Xanatos Gambits
            plan_tree['xanatos_gambits'] = xanatos_gambits
            return plan_tree
        except Exception as e:
            logger.exception(f"Error developing Xanatos Gambits: {str(e)}")
            raise AIVillageException(f"Error developing Xanatos Gambits: {str(e)}") from e

    async def _update_plan(self, plan_tree: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = f"""
            Update the following plan based on the premortem analysis, antifragility assessment, and Xanatos Gambits:
            {json.dumps(plan_tree, indent=2)}
            
            Integrate the insights and strategies from these analyses to create an improved plan that is:
            1. More robust against potential failures
            2. More antifragile and able to benefit from volatility
            3. Structured to succeed in multiple scenarios
            
            Maintain the original structure of the plan tree, but update and add elements as necessary.
            Provide explanations for significant changes made to the plan.
            """
            updated_plan = await self._generate_structured_response(prompt)
            return updated_plan
        except Exception as e:
            logger.exception(f"Error updating plan: {str(e)}")
            raise AIVillageException(f"Error updating plan: {str(e)}") from e

    def _calculate_success_likelihood(self, plan_tree: Dict[str, Any]) -> float:
        try:
            antifragility_scores = plan_tree.get('antifragility_assessment', {}).get('antifragility_scores', {})
            if not antifragility_scores:
                return 0.5  # Default to 50% if no scores are available
            
            avg_antifragility = sum(antifragility_scores.values()) / len(antifragility_scores)
            success_likelihood = (avg_antifragility + 10) / 20  # Convert -10 to 10 scale to 0 to 1 scale
            return max(0, min(1, success_likelihood))  # Ensure the result is between 0 and 1
        except Exception as e:
            logger.exception(f"Error calculating success likelihood: {str(e)}")
            return 0.5  # Return a default value in case of error

    def _identify_capability_gaps(self, plan_tree: Dict[str, Any], current_resources: Dict[str, Any]) -> List[str]:
        try:
            required_resources = set()
            for task in self._extract_all_tasks(plan_tree):
                required_resources.update(task.get('required_resources', []))
            
            available_resources = set(current_resources.get('resources', []) + 
                                      current_resources.get('tools', []))
            
            return list(required_resources - available_resources)
        except Exception as e:
            logger.exception(f"Error identifying capability gaps: {str(e)}")
            return []

    async def _plan_checkpoints(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            prompt = f"""
            Plan checkpoints for the following set of tasks:
            {json.dumps(tasks, indent=2)}
            
            Create a list of checkpoints that:
            1. Cover critical milestones in the plan
            2. Allow for progress evaluation and course correction
            3. Are spaced appropriately throughout the timeline
            
            For each checkpoint, provide:
            - Description: What should be achieved by this point
            - Evaluation criteria: How to measure progress
            - Potential adjustments: Actions to take if progress is off-track
            
            Output the checkpoints as a JSON list of dictionaries.
            """
            checkpoints = await self._generate_structured_response(prompt)
            return checkpoints
        except Exception as e:
            logger.exception(f"Error planning checkpoints: {str(e)}")
            raise AIVillageException(f"Error planning checkpoints: {str(e)}") from e

    async def _perform_swot_analysis(self, plan_tree: Dict[str, Any], problem_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        try:
            prompt = f"""
            Perform a SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis for the following plan and problem analysis:
            
            Plan: {json.dumps(plan_tree, indent=2)}
            Problem Analysis: {json.dumps(problem_analysis, indent=2)}
            
            Provide a comprehensive SWOT analysis that considers:
            1. Internal factors (Strengths and Weaknesses)
            2. External factors (Opportunities and Threats)
            3. How the plan addresses the problem
            4. Potential future developments
            
            Output the analysis as a JSON dictionary with 'strengths', 'weaknesses', 'opportunities', and 'threats' keys,
            each containing a list of relevant points.
            """
            swot_analysis = await self._generate_structured_response(prompt)
            return swot_analysis
        except Exception as e:
            logger.exception(f"Error performing SWOT analysis: {str(e)}")
            raise AIVillageException(f"Error performing SWOT analysis: {str(e)}") from e

    def _calculate_plan_metrics(self, plan_tree: Dict[str, Any]) -> Dict[str, Any]:
        try:
            tasks = self._extract_all_tasks(plan_tree)
            total_time = sum(task.get('estimated_time', 0) for task in tasks)
            total_resources = sum(len(task.get('required_resources', [])) for task in tasks)
            risk_levels = [task.get('risk_level', 0) for task in tasks if 'risk_level' in task]
            avg_risk = sum(risk_levels) / len(risk_levels) if risk_levels else 0
            
            return {
                'total_tasks': len(tasks),
                'total_estimated_time': total_time,
                'total_required_resources': total_resources,
                'average_risk_level': avg_risk
            }
        except Exception as e:
            logger.exception(f"Error calculating plan metrics: {str(e)}")
            return {}

    def _tree_to_dict(self, node: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            'name': node.get('name', ''),
            'description': node.get('description', ''),
            'tasks': node.get('tasks', []),
            'antifragility_score': node.get('antifragility_score', 0),
            'xanatos_factor': node.get('xanatos_factor', 0)
        }
        if 'sub_goals' in node:
            result['children'] = [self._tree_to_dict(child) for child in node['sub_goals']]
        return result

    def _create_plan_visualization(self, plan_tree: Dict[str, Any]) -> str:
        try:
            import matplotlib.pyplot as plt
            G = nx.DiGraph()
            
            def add_nodes(node, parent=None):
                node_id = node.get('name', f"task_{random.randint(1000,9999)}")
                G.add_node(node_id, 
                           description=node.get('description', ''),
                           antifragility=node.get('antifragility_score', 0),
                           xanatos_factor=node.get('xanatos_factor', 0),
                           type=node.get('type', 'task'))
                if parent:
                    G.add_edge(parent, node_id)
                for child in node.get('sub_goals', []):
                    add_nodes(child, node_id)
            
            add_nodes(plan_tree)
            
            pos = nx.spring_layout(G)
            plt.figure(figsize=(20, 20))
            
            node_colors = ['red' if G.nodes[node]['antifragility'] < -3 else 
                           'green' if G.nodes[node]['antifragility'] > 3 else 
                           'yellow' for node in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000)
            nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
            nx.draw_networkx_labels(G, pos, {node: node for node in G.nodes()}, font_size=8, font_weight='bold')
            
            plt.title("Agent and Task Plan Visualization")
            plt.axis('off')
            plt.tight_layout()
            visualization_path = "plan_visualization.png"
            plt.savefig(visualization_path)
            plt.close()
            logger.info(f"Plan visualization saved as {visualization_path}")
            return visualization_path
        except Exception as e:
            logger.exception(f"Error creating plan visualization: {str(e)}")
            return ""

    def _extract_all_tasks(self, plan_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        tasks = []
        
        def extract_tasks_recursive(node):
            if 'tasks' in node:
                tasks.extend(node['tasks'])
            for sub_goal in node.get('sub_goals', []):
                extract_tasks_recursive(sub_goal)
        
        extract_tasks_recursive(plan_tree)
        return tasks

    async def _analyze_execution_result(self, execution_result: Dict[str, Any], optimized_plan: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = f"""
            Analyze the following execution results and the optimized plan:
            
            Execution Results: {json.dumps(execution_result, indent=2)}
            Optimized Plan: {json.dumps(optimized_plan, indent=2)}
            
            Provide insights on:
            1. Success factors
            2. Areas needing improvement
            3. Recommendations for future tasks
            
            Output the analysis as a JSON dictionary with 'success_factors', 'areas_for_improvement', and 'recommendations' keys.
            """
            analysis = await self.llm.complete(p_prompt=prompt)
            return json.loads(analysis.text)
        except Exception as e:
            logger.exception(f"Error analyzing execution result: {str(e)}")
            raise AIVillageException(f"Error analyzing execution result: {str(e)}") from e

    async def _update_models(self, task: Dict[str, Any], execution_result: Dict[str, Any], analysis: Dict[str, Any]):
        try:
            # Update GraphManager with new task and execution results
            task_id = task.get('id', f"task_{random.randint(1000,9999)}")
            agent_id = execution_result.get('assigned_agent', 'default_agent')
            performance = execution_result.get('performance', 0.5)
            self.graph_manager.update_agent_experience(agent_id, task_id, performance)
            
            # Optionally, update task history or other models here
            await self.qa_layer.update_task_history(task, performance, execution_result.get('uncertainty', 0.5))
            
            # Save updated models if necessary
            logger.info("Models updated successfully after task execution")
        except Exception as e:
            logger.exception(f"Error updating models: {str(e)}")
            raise AIVillageException(f"Error updating models: {str(e)}") from e

    async def _update_models_from_workflow(self, tasks: List[Dict[str, Any]], results: Dict[str, Any], analysis: Dict[str, Any]):
        try:
            for task in tasks:
                task_id = task.get('id', f"task_{random.randint(1000,9999)}")
                result = results.get(task_id, {})
                agent_id = result.get('assigned_agent', 'default_agent')
                performance = result.get('performance', 0.5)
                self.graph_manager.update_agent_experience(agent_id, task_id, performance)
                
                await self.qa_layer.update_task_history(task, performance, result.get('uncertainty', 0.5))
            
            logger.info("Models updated successfully after workflow execution")
        except Exception as e:
            logger.exception(f"Error updating models from workflow: {str(e)}")
            raise AIVillageException(f"Error updating models from workflow: {str(e)}") from e

    async def _generate_structured_response(self, prompt: str) -> Any:
        # Placeholder method for generating structured JSON responses
        response = await self.llm.complete(prompt)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from LLM response")
            return {}

async def demo_magi_planning():
    """
    Demonstrate the main features of the MagiPlanning class.
    """
    # Initialize required components
    communication_protocol = StandardCommunicationProtocol()
    quality_assurance_layer = QualityAssuranceLayer()
    graph_manager = GraphManager()

    # Create MagiPlanning instance
    magi_planner = MagiPlanning(communication_protocol, quality_assurance_layer, graph_manager)

    # Define a goal and problem analysis
    goal = "Develop a new feature for our AI-powered chatbot"
    problem_analysis = {
        "problem_statement": "Our chatbot needs to handle multi-turn conversations more effectively.",
        "constraints": ["Must be implemented within 2 weeks", "Should not increase latency by more than 100ms"],
        "resources": ["2 senior developers", "1 ML engineer", "Access to GPT-4 API"]
    }

    # Generate a plan
    plan = await magi_planner.generate_plan(goal, problem_analysis)

    # Print plan summary
    print(f"Plan for goal: {plan['goal']}")
    print(f"Success likelihood: {plan['success_likelihood']:.2f}")
    print(f"Number of tasks: {plan['metrics']['total_tasks']}")
    print(f"Estimated time: {plan['metrics']['total_estimated_time']} hours")
    print(f"Capability gaps: {', '.join(plan['capability_gaps'])}")
    print(f"Plan visualization saved as: {plan['visualization']}")

    # Execute a sample task
    sample_task = "Implement a context management system for the chatbot"
    task_result = await magi_planner.execute_task(sample_task)
    print(f"\nTask executed using {task_result['technique']} technique")
    print(f"Execution time: {task_result['execution_time']:.2f} seconds")

    # Analyze execution results
    analysis = await magi_planner._analyze_execution_result(task_result, plan['plan_tree'])
    print("\nExecution Analysis:")
    print("Success factors:", analysis['success_factors'])
    print("Areas for improvement:", analysis['areas_for_improvement'])

    # Update models based on execution results
    await magi_planner._update_models(task_result, task_result, analysis)
    print("\nModels updated based on execution results")

if __name__ == "__main__":
    asyncio.run(demo_magi_planning())
