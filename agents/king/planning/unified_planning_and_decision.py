import logging
import math
import random
import os
import json
import asyncio
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import io
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from rag_system.core.pipeline import EnhancedRAGPipeline
from .quality_assurance_layer import QualityAssuranceLayer
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from agents.utils.exceptions import AIVillageException
from .reasoning_engine import ReasoningEngine
from .task_handling import TaskHandler
from .optimization import Optimizer
from .routing import Router

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

class UnifiedPlanningAndDecision:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, rag_system: EnhancedRAGPipeline, agent, quality_assurance_layer: QualityAssuranceLayer, exploration_weight=1.0, max_depth=10):
        self.communication_protocol = communication_protocol
        self.rag_system = rag_system
        self.agent = agent
        self.quality_assurance_layer = quality_assurance_layer
        self.llm = OpenAIGPTConfig(chat_model="gpt-4").create()
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.stats = defaultdict(lambda: {'visits': 0, 'value': 0})
        self.available_agents = []
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.reasoning_engine = ReasoningEngine()
        self.task_handler = TaskHandler()
        self.optimizer = Optimizer()
        self.router = Router()
        self.graph_manager = GraphManager()

    async def make_decision(self, content: str, eudaimonia_score: float) -> Dict[str, Any]:
        try:
            rag_info = await self.rag_system.process_query(content)
            task_vector = self.quality_assurance_layer.eudaimonia_triangulator.get_embedding(content)
            rule_compliance = self.quality_assurance_layer.evaluate_rule_compliance(task_vector)

            # Update agent nodes with new experiences
            # Assuming 'agent_id' is part of rag_info or another source
            agent_id = rag_info.get('assigned_agent', 'default_agent')
            self.graph_manager.update_agent_experience(agent_id, content, eudaimonia_score)

            # Convert King's breakdown into graph nodes
            plan = await self.generate_plan(content, {"content": content, "rag_info": rag_info})
            self.graph_manager.add_task_node(content, {"description": plan.get("description", "")})

            # Merge new task nodes with existing agent graph
            # Assuming 'plan_graph' is part of the plan
            plan_graph = plan.get("plan_tree", {})
            if plan_graph:
                nx_plan_graph = self._convert_plan_to_graph(plan_graph)
                self.graph_manager.merge_task_graph(nx_plan_graph)

            visualization_filename = "agent_task_graph.png"
            self.graph_manager.visualize_graph(visualization_filename)

            decision = plan["decision"]
            best_alternative = plan["best_alternative"]
            implementation_plan = await self._create_implementation_plan(plan)

            return {
                "decision": decision,
                "eudaimonia_score": eudaimonia_score,
                "rule_compliance": rule_compliance,
                "rag_info": rag_info,
                "best_alternative": best_alternative,
                "implementation_plan": implementation_plan,
                "full_plan": plan,
                "visualization": visualization_filename
            }
        except Exception as e:
            logger.exception(f"Error making decision: {str(e)}")
            raise AIVillageException(f"Error making decision: {str(e)}") from e

    def _convert_plan_to_graph(self, plan_tree: Dict[str, Any]) -> nx.DiGraph:
        G = nx.DiGraph()

        def add_nodes(node, parent=None):
            node_id = node.get('name', f"task_{random.randint(1000,9999)}")
            G.add_node(node_id, description=node.get('description', ''), type='task')
            if parent:
                G.add_edge(parent, node_id)
            for sub_goal in node.get('sub_goals', []):
                add_nodes(sub_goal, node_id)

        add_nodes(plan_tree)
        logger.info("Converted plan tree to networkx graph")
        return G

    async def generate_plan(self, goal: str, problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        try:
            current_resources = await self._get_current_resources()
            plan_tree = await self._create_plan_tree(goal, current_resources)
            await self._extract_tasks(plan_tree)
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

            visualization = self._create_plan_visualization(plan_tree)

            return {**plan_data, 'visualization': visualization}
        except Exception as e:
            logger.exception(f"Error in plan generation: {str(e)}")
            raise AIVillageException(f"Error in plan generation: {str(e)}") from e

    async def manage_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            decision = await self.make_decision(task['description'], 0.5)  # Assuming default eudaimonia_score
            plan = await self.reasoning_engine.analyze_and_reason(decision)
            optimized_plan = await self.optimizer.optimize_plan(plan)
            routed_task = await self.router.route_task(optimized_plan)
            execution_result = await self.task_handler.execute_task(routed_task)
            
            # Perform post-execution analysis
            analysis = await self._analyze_execution_result(execution_result, optimized_plan)
            
            # Update models based on execution results
            await self._update_models(task, execution_result, analysis)
            
            return {**execution_result, "analysis": analysis}
        except Exception as e:
            logger.exception(f"Error in manage_task: {str(e)}")
            raise AIVillageException(f"Error in manage_task: {str(e)}") from e

    async def create_and_execute_workflow(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            workflow = await self.task_handler.create_workflow(tasks)
            optimized_workflow = await self.optimizer.optimize_workflow(workflow)
            execution_plan = await self._create_execution_plan(optimized_workflow)
            results = await self._execute_workflow_in_parallel(execution_plan)
            
            # Perform post-execution analysis
            analysis = await self._analyze_workflow_execution(results, optimized_workflow)
            
            # Update models based on workflow execution results
            await self._update_models_from_workflow(tasks, results, analysis)
            
            return {"results": results, "analysis": analysis}
        except Exception as e:
            logger.exception(f"Error in create_and_execute_workflow: {str(e)}")
            raise AIVillageException(f"Error in create_and_execute_workflow: {str(e)}") from e

    async def mcts_search(self, task, problem_analyzer, plan_generator, iterations=1000):
        root = MCTSNode(task)

        for _ in range(iterations):
            node = self.select(root)
            if node.visits < 1 or len(node.children) == 0:
                child = await self.expand(node, problem_analyzer)
            else:
                child = self.best_uct_child(node)
            result = await self.simulate(child, plan_generator)
            self.backpropagate(child, result)

        return self.best_child(root).state

    def select(self, node):
        path = []
        while True:
            path.append(node)
            if not node.children or len(path) > self.max_depth:
                return node
            unexplored = [child for child in node.children if child.visits == 0]
            if unexplored:
                return random.choice(unexplored)
            node = self.best_uct_child(node)

    async def expand(self, node, problem_analyzer):
        if problem_analyzer:
            new_states = await problem_analyzer.generate_possible_states(node.state)
        else:
            new_states = [node.state]  # Placeholder for when problem_analyzer is not provided
        for state in new_states:
            if state not in [child.state for child in node.children]:
                new_node = MCTSNode(state, parent=node)
                node.children.append(new_node)
                return new_node
        return random.choice(node.children)

    async def simulate(self, node, plan_generator):
        return await plan_generator.evaluate(node.state)

    def backpropagate(self, node, result):
        while node:
            self.stats[node.state]['visits'] += 1
            self.stats[node.state]['value'] += result
            node.visits += 1
            node.value += result
            node = node.parent

    def best_uct_child(self, node):
        log_n_visits = math.log(self.stats[node.state]['visits'])
        return max(
            node.children,
            key=lambda c: (self.stats[c.state]['value'] / self.stats[c.state]['visits']) +
                self.exploration_weight * math.sqrt(log_n_visits / self.stats[c.state]['visits'])
        )

    def best_child(self, node):
        return max(node.children, key=lambda c: self.stats[c.state]['visits'])

    async def update_model(self, task: Dict[str, Any], result: Any):
        try:
            logger.info(f"Updating unified planning and decision model with task result: {result}")
            await self.mcts_update(task, result)
            await self.quality_assurance_layer.update_task_history(task, result.get('performance', 0.5), result.get('uncertainty', 0.5))
            await self.reasoning_engine.update_model(task, result)
            await self.optimizer.update_model(task, result)
            await self.router.update_model(task, result)
            await self.task_handler.update_model(task, result)
        except Exception as e:
            logger.exception(f"Error updating unified planning and decision model: {str(e)}")
            raise AIVillageException(f"Error updating unified planning and decision model: {str(e)}") from e

    async def mcts_update(self, task, result):
        self.stats[task]['visits'] += 1
        self.stats[task]['value'] += result

    async def mcts_prune(self, node, threshold):
        node.children = [child for child in node.children if self.stats[child.state]['visits'] > threshold]
        for child in node.children:
            await self.mcts_prune(child, threshold)

    async def parallel_mcts_search(self, task, problem_analyzer, plan_generator, iterations=1000, num_workers=4):
        root = MCTSNode(task)
        semaphore = asyncio.Semaphore(num_workers)

        async def worker():
            async with semaphore:
                node = self.select(root)
                if node.visits < 1 or len(node.children) == 0:
                    child = await self.expand(node, problem_analyzer)
                else:
                    child = self.best_uct_child(node)
                result = await self.simulate(child, plan_generator)
                self.backpropagate(child, result)

        await asyncio.gather(*[worker() for _ in range(iterations)])
        return self.best_child(root).state

    def update_agent_list(self, agent_list: List[str]):
        self.available_agents = agent_list
        self.router.update_agent_list(agent_list)
        self.graph_manager.available_agents = agent_list
        logger.info(f"Updated available agents: {self.available_agents}")

    async def save_models(self, path: str):
        try:
            os.makedirs(path, exist_ok=True)
            
            with open(os.path.join(path, "mcts_stats.json"), "w") as f:
                json.dump(dict(self.stats), f)
            
            await self.quality_assurance_layer.save(os.path.join(path, "quality_assurance_layer.json"))
            await self.optimizer.save_models(os.path.join(path, "optimizer"))
            await self.reasoning_engine.save_models(os.path.join(path, "reasoning_engine"))
            await self.router.save_models(os.path.join(path, "router"))
            await self.task_handler.save_models(os.path.join(path, "task_handler"))
            
            data = {
                "exploration_weight": self.exploration_weight,
                "max_depth": self.max_depth,
                "available_agents": self.available_agents
            }
            with open(os.path.join(path, "unified_planning_and_decision_data.json"), "w") as f:
                json.dump(data, f)
            
            logger.info(f"Models saved successfully to {path}")
        except Exception as e:
            logger.exception(f"Error saving models: {str(e)}")
            raise AIVillageException(f"Error saving models: {str(e)}") from e

    async def load_models(self, path: str):
        try:
            with open(os.path.join(path, "mcts_stats.json"), "r") as f:
                self.stats = defaultdict(lambda: {'visits': 0, 'value': 0}, json.load(f))
            
            self.quality_assurance_layer = QualityAssuranceLayer.load(os.path.join(path, "quality_assurance_layer.json"))
            await self.optimizer.load_models(os.path.join(path, "optimizer"))
            await self.reasoning_engine.load_models(os.path.join(path, "reasoning_engine"))
            await self.router.load_models(os.path.join(path, "router"))
            await self.task_handler.load_models(os.path.join(path, "task_handler"))
            
            with open(os.path.join(path, "unified_planning_and_decision_data.json"), "r") as f:
                data = json.load(f)
                self.exploration_weight = data["exploration_weight"]
                self.max_depth = data["max_depth"]
                self.available_agents = data["available_agents"]
                self.graph_manager.available_agents = self.available_agents
            
            logger.info(f"Models loaded successfully from {path}")
        except Exception as e:
            logger.exception(f"Error loading models: {str(e)}")
            raise AIVillageException(f"Error loading models: {str(e)}") from e

    async def introspect(self) -> Dict[str, Any]:
        return {
            "type": "UnifiedPlanningAndDecision",
            "description": "Manages decision-making, planning, and task execution",
            "available_agents": self.available_agents,
            "quality_assurance_info": self.quality_assurance_layer.get_info(),
            "mcts_stats": dict(self.stats),
            "reasoning_engine_info": await self.reasoning_engine.introspect(),
            "task_handler_info": await self.task_handler.introspect(),
            "optimizer_info": await self.optimizer.introspect(),
            "router_info": await self.router.introspect(),
            "graph_manager_info": self.graph_manager.get_graph().number_of_nodes()
        }

    async def _get_current_resources(self) -> Dict[str, Any]:
        prompt = "List the current resources, position, and tools available for the AI Village project. Output as a JSON dictionary with keys 'resources', 'position', and 'tools'."
        try:
            return await self.agent.generate_structured_response(prompt)
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
            plan_tree = await self.agent.generate_structured_response(prompt)
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
            optimized_tasks = await self.agent.generate_structured_response(prompt)
            
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
            premortem_results = await self.agent.generate_structured_response(prompt)
            
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
            antifragility_assessment = await self.agent.generate_structured_response(prompt)
            
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
            xanatos_gambits = await self.agent.generate_structured_response(prompt)
            
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
            updated_plan = await self.agent.generate_structured_response(prompt)
            return updated_plan
        except Exception as e:
            logger.exception(f"Error updating plan: {str(e)}")
            raise AIVillageException(f"Error updating plan: {str(e)}") from e

    def _calculate_success_likelihood(self, plan_tree: Dict[str, Any]) -> float:
        try:
            # This is a simplified calculation and should be expanded based on your specific requirements
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
            checkpoints = await self.agent.generate_structured_response(prompt)
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
            swot_analysis = await self.agent.generate_structured_response(prompt)
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
            analysis = await self.agent.generate_structured_response(prompt)
            return analysis
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
            await self.quality_assurance_layer.update_task_history(task, performance, execution_result.get('uncertainty', 0.5))
            
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
                
                await self.quality_assurance_layer.update_task_history(task, performance, result.get('uncertainty', 0.5))
            
            logger.info("Models updated successfully after workflow execution")
        except Exception as e:
            logger.exception(f"Error updating models from workflow: {str(e)}")
            raise AIVillageException(f"Error updating models from workflow: {str(e)}") from e

    # Additional helper methods would be here...
