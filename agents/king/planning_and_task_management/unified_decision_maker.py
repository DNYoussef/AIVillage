import logging
import math
import random
from typing import List, Dict, Any, Tuple
import asyncio
from collections import defaultdict
import itertools
import json
import os
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from rag_system.core.pipeline import EnhancedRAGPipeline
from .quality_assurance_layer import QualityAssuranceLayer
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from agents.utils.exceptions import AIVillageException
import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)

@dataclass
class Node:
    name: str
    description: str
    prerequisites: List['Node'] = field(default_factory=list)
    probability: float = 0.5
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    failure_modes: List[Dict[str, Any]] = field(default_factory=list)
    antifragility_score: float = 0.0
    xanatos_factor: float = 0.0
    xanatos_gambits: List[Dict[str, Any]] = field(default_factory=list)
    expected_utility: float = 0.0

@dataclass
class PlanConfig:
    success_likelihood_threshold: float = 0.95
    max_iterations: int = 10
    parallelization: bool = True

class UnifiedDecisionMaker:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, rag_system: EnhancedRAGPipeline, agent, quality_assurance_layer: QualityAssuranceLayer):
        self.communication_protocol = communication_protocol
        self.rag_system = rag_system
        self.agent = agent
        self.quality_assurance_layer = quality_assurance_layer
        self.llm = OpenAIGPTConfig(chat_model="gpt-4").create()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.plan_config = PlanConfig()
        self.available_agents = []

    async def make_decision(self, content: str, eudaimonia_score: float) -> Dict[str, Any]:
        try:
            rag_info = await self.rag_system.process_query(content)
            task_vector = self.quality_assurance_layer.eudaimonia_triangulator.get_embedding(content)
            rule_compliance = self.quality_assurance_layer.evaluate_rule_compliance(task_vector)

            plan = await self.generate_plan(content, {"content": content, "rag_info": rag_info})
            
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
                "full_plan": plan
            }
        except Exception as e:
            logger.exception(f"Error making decision: {str(e)}")
            raise AIVillageException(f"Error making decision: {str(e)}") from e

    async def generate_plan(self, goal: str, problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        try:
            current_resources = await self._get_current_resources()
            plan_tree = await self._create_plan_tree(goal, current_resources)
            await self._extract_tasks(plan_tree)
            await self._optimize_tasks(plan_tree)
            
            success_likelihood = 0
            iteration = 0
            while success_likelihood < self.plan_config.success_likelihood_threshold and iteration < self.plan_config.max_iterations:
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
            raise AIVillageException(f"Error in plan generation: {str(e)}")

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

class UnifiedDecisionMaker:
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

    async def make_decision(self, content: str, eudaimonia_score: float) -> Dict[str, Any]:
        """
        Make a decision based on the given content and eudaimonia score.

        Args:
            content (str): The content to base the decision on.
            eudaimonia_score (float): The eudaimonia score to consider in the decision making process.

        Returns:
            Dict[str, Any]: A dictionary containing the decision and related information.

        Raises:
            AIVillageException: If there is an error during the decision-making process.
        """
        try:
            rag_info = await self.rag_system.process_query(content)
            task_vector = self.quality_assurance_layer.eudaimonia_triangulator.get_embedding(content)
            rule_compliance = self.quality_assurance_layer.evaluate_rule_compliance(task_vector)

            alternatives = await self._generate_alternatives({"content": content})
            ranked_criteria = await self._rank_criteria(await self._determine_success_criteria({"content": content}))
            evaluated_alternatives = await self._evaluate_alternatives(alternatives, ranked_criteria)
            best_alternative = max(evaluated_alternatives, key=lambda x: x['score'])['alternative']

            outcomes = await self._simplify_outcomes(ranked_criteria)
            prob_trees = await self._determine_probabilities(alternatives, outcomes)
            utility_chart = await self._create_utility_chart(outcomes)
            expected_utilities = await self._calculate_expected_utility(alternatives, prob_trees, utility_chart)
            chosen_alternative = max(expected_utilities, key=lambda x: x['score'])['alternative']

            plan = {
                "decision": await self.llm.complete(f"Make a decision about {content}"),
                "best_alternative": best_alternative,
                "chosen_alternative": chosen_alternative
            }
            implementation_plan = await self._create_implementation_plan(plan)

            return {
                "decision": plan["decision"].text,
                "eudaimonia_score": eudaimonia_score,
                "rule_compliance": rule_compliance,
                "rag_info": rag_info,
                "best_alternative": plan["best_alternative"],
                "chosen_alternative": plan["chosen_alternative"],
                "implementation_plan": implementation_plan
            }
        except Exception as e:
            logger.exception(f"Error making decision: {str(e)}")
            raise AIVillageException(f"Error making decision: {str(e)}") from e

    async def update_model(self, task: Dict[str, Any], result: Any):
        """
        Update the decision maker's model based on the task and result.

        Args:
            task (Dict[str, Any]): The task that was performed.
            result (Any): The result of the task.

        Raises:
            AIVillageException: If there is an error updating the model.
        """
        try:
            logger.info(f"Updating unified decision maker model with task result: {result}")
            await self.mcts_update(task, result)
            await self.quality_assurance_layer.update_task_history(task, result.get('performance', 0.5), result.get('uncertainty', 0.5))
        except Exception as e:
            logger.exception(f"Error updating unified decision maker model: {str(e)}")
            raise AIVillageException(f"Error updating unified decision maker model: {str(e)}") from e

    async def mcts_search(self, task, problem_analyzer, plan_generator, iterations=1000):
        """
        Perform a Monte Carlo Tree Search (MCTS) for the given task.

        Args:
            task: The task to perform the search for.
            problem_analyzer: The problem analyzer to use for generating possible states.
            plan_generator: The plan generator to use for evaluating states.
            iterations (int, optional): The number of iterations to perform. Defaults to 1000.

        Returns:
            The best state found by the search.
        """
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
        """
        Select the best child node to explore based on the UCT (Upper Confidence Bound applied to Trees) algorithm.

        Args:
            node (MCTSNode): The node to select the best child from.

        Returns:
            MCTSNode: The selected child node.
        """
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
        """
        Expand the given node by generating possible states using the problem analyzer.

        Args:
            node (MCTSNode): The node to expand.
            problem_analyzer: The problem analyzer to use for generating possible states.

        Returns:
            MCTSNode: The expanded child node.
        """
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
        """
        Simulate the execution of the plan for the given node using the plan generator.

        Args:
            node (MCTSNode): The node to simulate the plan for.
            plan_generator: The plan generator to use for evaluating the state.

        Returns:
            The result of the simulation.
        """
        return await plan_generator.evaluate(node.state)

    def backpropagate(self, node, result):
        """
        Backpropagate the result of the simulation up the tree.

        Args:
            node (MCTSNode): The node to start the backpropagation from.
            result: The result of the simulation.
        """
        while node:
            self.stats[node.state]['visits'] += 1
            self.stats[node.state]['value'] += result
            node.visits += 1
            node.value += result
            node = node.parent

    def best_uct_child(self, node):
        """
        Select the best child node based on the UCT (Upper Confidence Bound applied to Trees) algorithm.

        Args:
            node (MCTSNode): The node to select the best child from.

        Returns:
            MCTSNode: The best child node according to UCT.
        """
        log_n_visits = math.log(self.stats[node.state]['visits'])
        return max(
            node.children,
            key=lambda c: (self.stats[c.state]['value'] / self.stats[c.state]['visits']) +
                self.exploration_weight * math.sqrt(log_n_visits / self.stats[c.state]['visits'])
        )

    def best_child(self, node):
        """
        Select the child node with the most visits.

        Args:
            node (MCTSNode): The node to select the best child from.

        Returns:
            MCTSNode: The child node with the most visits.
        """
        return max(node.children, key=lambda c: self.stats[c.state]['visits'])

    async def mcts_update(self, task, result):
        """
        Update the MCTS statistics based on the task execution results.

        Args:
            task: The task that was executed.
            result: The result of the task execution.
        """
        self.stats[task]['visits'] += 1
        self.stats[task]['value'] += result

    async def mcts_prune(self, node, threshold):
        """
        Prune the MCTS tree by removing nodes with visits below the given threshold.

        Args:
            node (MCTSNode): The node to start the pruning from.
            threshold (int): The minimum number of visits required to keep a node.
        """
        node.children = [child for child in node.children if self.stats[child.state]['visits'] > threshold]
        for child in node.children:
            await self.mcts_prune(child, threshold)

    async def parallel_mcts_search(self, task, problem_analyzer, plan_generator, iterations=1000, num_workers=4):
        """
        Perform a parallel Monte Carlo Tree Search (MCTS) for the given task.

        Args:
            task: The task to perform the search for.
            problem_analyzer: The problem analyzer to use for generating possible states.
            plan_generator: The plan generator to use for evaluating states.
            iterations (int, optional): The number of iterations to perform. Defaults to 1000.
            num_workers (int, optional): The number of parallel workers to use. Defaults to 4.

        Returns:
            The best state found by the search.
        """
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
        """
        Update the list of available agents.

        Args:
            agent_list (List[str]): The updated list of available agents.
        """
        self.available_agents = agent_list
        logger.info(f"Updated available agents: {self.available_agents}")

    async def save_models(self, path: str):
        """
        Save the decision maker's models to the specified path.

        Args:
            path (str): The path to save the models to.

        Raises:
            AIVillageException: If there is an error saving the models.
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save MCTS stats
            with open(os.path.join(path, "mcts_stats.json"), "w") as f:
                json.dump(dict(self.stats), f)
            
            # Save quality assurance layer
            await self.quality_assurance_layer.save(os.path.join(path, "quality_assurance_layer.json"))
            
            # Save other relevant data
            data = {
                "exploration_weight": self.exploration_weight,
                "max_depth": self.max_depth,
                "available_agents": self.available_agents
            }
            with open(os.path.join(path, "unified_decision_maker_data.json"), "w") as f:
                json.dump(data, f)
            
            logger.info(f"Models saved successfully to {path}")
        except Exception as e:
            logger.exception(f"Error saving models: {str(e)}")
            raise AIVillageException(f"Error saving models: {str(e)}") from e

    async def load_models(self, path: str):
        """
        Load the decision maker's models from the specified path.

        Args:
            path (str): The path to load the models from.

        Raises:
            AIVillageException: If there is an error loading the models.
        """
        try:
            # Load MCTS stats
            with open(os.path.join(path, "mcts_stats.json"), "r") as f:
                self.stats = defaultdict(lambda: {'visits': 0, 'value': 0}, json.load(f))
            
            # Load quality assurance layer
            await self.quality_assurance_layer.load(os.path.join(path, "quality_assurance_layer.json"))
            
            # Load other relevant data
            with open(os.path.join(path, "unified_decision_maker_data.json"), "r") as f:
                data = json.load(f)
                self.exploration_weight = data["exploration_weight"]
                self.max_depth = data["max_depth"]
                self.available_agents = data["available_agents"]
            
            logger.info(f"Models loaded successfully from {path}")
        except Exception as e:
            logger.exception(f"Error loading models: {str(e)}")
            raise AIVillageException(f"Error loading models: {str(e)}") from e

    async def introspect(self) -> Dict[str, Any]:
        """
        Introspect the decision maker and return information about its current state.

        Returns:
            Dict[str, Any]: A dictionary containing information about the decision maker's state.
        """
        return {
            "type": "UnifiedDecisionMaker",
            "description": "Makes decisions based on task content, RAG information, eudaimonia score, and rule compliance",
            "available_agents": self.available_agents,
            "quality_assurance_info": self.quality_assurance_layer.get_info(),
            "mcts_stats": dict(self.stats)
        }

    async def _generate_alternatives(self, problem_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate alternative solutions for the given problem analysis.

        Args:
            problem_analysis (Dict[str, Any]): The problem analysis to generate alternatives for.

        Returns:
            List[str]: A list of alternative solutions.
        """
        king_alternatives = await self.agent.generate_structured_response(
            f"Given the problem analysis: {problem_analysis}, generate 3 potential solutions. Output as a JSON list of strings."
        )
        
        all_alternatives = king_alternatives.copy()
        
        for agent in self.available_agents:
            agent_alternatives_request = Message(
                type=MessageType.QUERY,
                sender="King",
                receiver=agent,
                content={"action": "generate_alternatives", "problem_analysis": problem_analysis}
            )
            response = await self.communication_protocol.send_and_wait(agent_alternatives_request)
            all_alternatives.extend(response.content["alternatives"])
        
        return list(dict.fromkeys(all_alternatives))

    async def _evaluate_alternatives(self, alternatives: List[str], ranked_criteria: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate the given alternatives based on the ranked criteria.

        Args:
            alternatives (List[str]): The alternatives to evaluate.
            ranked_criteria (List[Dict[str, Any]]): The ranked criteria to evaluate the alternatives against.

        Returns:
            List[Dict[str, Any]]: A list of evaluated alternatives with their scores.
        """
        evaluated_alternatives = []
        for alt in alternatives:
            alt_vector = self.quality_assurance_layer.eudaimonia_triangulator.get_embedding(alt)
            eudaimonia_score = self.quality_assurance_layer.eudaimonia_triangulator.triangulate(alt_vector)
            rule_compliance = self.quality_assurance_layer.evaluate_rule_compliance(alt_vector)
            
            total_score = sum(
                criterion['weight'] * (eudaimonia_score if criterion['criterion'] == 'eudaimonia' else rule_compliance)
                for criterion in ranked_criteria
            )
            
            evaluated_alternatives.append({'alternative': alt, 'score': total_score})
        
        return sorted(evaluated_alternatives, key=lambda x: x['score'], reverse=True)
    
    async def _determine_probabilities(self, alternatives: List[str], outcomes: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Determine the probabilities of outcomes for each alternative.

        Args:
            alternatives (List[str]): The alternatives to determine probabilities for.
            outcomes (Dict[str, List[str]]): The possible outcomes for each criterion.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary mapping alternatives to their outcome probabilities.
        """
        prob_trees = {}
        for alt in alternatives:
            prompt = f"For the alternative '{alt}', estimate the probability of each outcome in {outcomes}. Output a JSON dictionary where keys are criteria and values are dictionaries mapping outcomes to probabilities (0-1)."
            prob_trees[alt] = await self.agent.generate_structured_response(prompt)
        return prob_trees

    async def _calculate_expected_utility(self, alternatives: List[str], prob_trees: Dict[str, Dict[str, float]], utility_chart: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Calculate the expected utility for each alternative.

        Args:
            alternatives (List[str]): List of alternative solutions.
            prob_trees (Dict[str, Dict[str, float]]): Probability trees for each alternative.
            utility_chart (Dict[str, float]): Utility values for outcome combinations.

        Returns:
            List[Dict[str, Any]]: List of alternatives with their expected utility scores.
        """
        evaluated_alternatives = []
        for alt in alternatives:
            expected_utility = 0
            for outcome_combination in self._generate_combinations(prob_trees[alt]):
                probability = 1
                for criterion, outcome in zip(prob_trees[alt].keys(), outcome_combination):
                    probability *= prob_trees[alt][criterion][outcome]
                utility = utility_chart.get(outcome_combination, 0)
                expected_utility += probability * utility
            
            evaluated_alternatives.append({'alternative': alt, 'score': expected_utility})

        return sorted(evaluated_alternatives, key=lambda x: x['score'], reverse=True)

    async def _create_implementation_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an implementation plan based on the given plan.

        Args:
            plan (Dict[str, Any]): The plan to create an implementation strategy for.

        Returns:
            Dict[str, Any]: An implementation plan with monitoring, feedback analysis, troubleshooting, and adaptive measures.

        Raises:
            AIVillageException: If there is an error creating the implementation plan.
        """
        try:
            logger.info("Creating implementation plan")
            prompt = f"""
            Given the following plan: {plan}, create an implementation strategy that includes:
            1. Monitoring steps to track progress and alignment with eudaimonia
            2. Feedback analysis to continuously improve the plan
            3. Troubleshooting steps to address potential issues
            4. Adaptive measures to adjust the plan based on new information or changing circumstances

            Output the result as a JSON dictionary with keys 'monitoring', 'feedback_analysis', 'troubleshooting', and 'adaptive_measures', each containing a list of steps.
            """
            implementation_plan = await self.agent.generate_structured_response(prompt)
            logger.debug(f"Implementation plan created: {implementation_plan}")
            return implementation_plan
        except Exception as e:
            logger.error(f"Error creating implementation plan: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error creating implementation plan: {str(e)}")

    async def _determine_success_criteria(self, problem_analysis: Dict[str, Any]) -> List[str]:
        """
        Determine the key success criteria for the given problem analysis.

        Args:
            problem_analysis (Dict[str, Any]): The problem analysis to determine success criteria for.

        Returns:
            List[str]: A list of key success criteria.
        """
        prompt = f"Based on the problem analysis: {problem_analysis}, determine the key success criteria for this task. Output as a JSON list of strings."
        return await self.agent.generate_structured_response(prompt)

    async def _rank_criteria(self, criteria: List[str]) -> List[Dict[str, Any]]:
        """
        Rank the given success criteria in order of importance.

        Args:
            criteria (List[str]): The list of success criteria to rank.

        Returns:
            List[Dict[str, Any]]: A list of ranked criteria with weights and explanations.
        """
        prompt = f"Rank the following success criteria in order of importance: {criteria}. For each criterion, provide a weight (0-1) and a brief explanation. Output as a JSON list of dictionaries, each containing 'criterion', 'weight', and 'explanation' keys."
        return await self.agent.generate_structured_response(prompt)

    async def _simplify_outcomes(self, ranked_criteria: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Simplify the possible outcomes for each ranked criterion.

        Args:
            ranked_criteria (List[Dict[str, Any]]): The list of ranked criteria.

        Returns:
            Dict[str, List[str]]: A dictionary mapping criteria to lists of possible outcomes.
        """
        prompt = f"For each of the following ranked criteria: {ranked_criteria}, provide a list of possible outcomes (success, partial success, failure). Output as a JSON dictionary where keys are criteria and values are lists of outcomes."
        return await self.agent.generate_structured_response(prompt)

    async def _create_utility_chart(self, outcomes: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Create a utility chart for the given outcomes.

        Args:
            outcomes (Dict[str, List[str]]): The possible outcomes for each criterion.

        Returns:
            Dict[str, float]: A utility chart mapping outcome combinations to utility values.
        """
        prompt = f"Create a utility chart for the following outcomes: {outcomes}. Assign a utility value (-10 to 10) for each combination of outcomes. Output as a JSON dictionary where keys are tuples of outcomes and values are utility scores."
        return await self.agent.generate_structured_response(prompt)

    def _generate_combinations(self, probabilities: Dict[str, Dict[str, float]]) -> List[tuple]:
        """
        Generate all possible combinations of outcomes based on the given probabilities.

        Args:
            probabilities (Dict[str, Dict[str, float]]): The probabilities of outcomes for each criterion.

        Returns:
            List[tuple]: A list of all possible outcome combinations.
        """
        criteria = list(probabilities.keys())
        outcomes = [list(probabilities[criterion].keys()) for criterion in criteria]
        return [combo for combo in itertools.product(*outcomes)]
    
    async def _get_current_resources(self) -> Dict[str, Any]:
        prompt = "List the current resources, position, and tools available for the AI Village project. Output as a JSON dictionary with keys 'resources', 'position', and 'tools'."
        try:
            return await self.agent.generate_structured_response(prompt)
        except Exception as e:
            raise AIVillageException(f"Error getting current resources: {str(e)}")

    async def _create_plan_tree(self, goal: str, current_resources: Dict[str, Any]) -> Node:
        root = Node(name=goal, description=f"Achieve: {goal}")
        await self._expand_node(root, current_resources)
        return root

    async def _expand_node(self, node: Node, current_resources: Dict[str, Any]):
        if self._is_resource(node.name, current_resources):
            return

        prompt = f"What are the immediate prerequisites for achieving '{node.name}'? Output as a JSON list of dictionaries, each with 'name' and 'description' keys."
        prerequisites = await self.agent.generate_structured_response(prompt)

        for prereq in prerequisites:
            child = Node(name=prereq['name'], description=prereq['description'])
            node.prerequisites.append(child)
            await self._expand_node(child, current_resources)

    def _is_resource(self, name: str, resources: Dict[str, Any]) -> bool:
        return any(name in resource_list for resource_list in resources.values())

    def _create_plan_visualization(self, plan_tree: Node) -> str:
        G = nx.DiGraph()
        
        def add_nodes(node: Node, parent=None):
            if node.antifragility_score < -3:
                anti_color = 'red'
            elif node.antifragility_score > 3:
                anti_color = 'green'
            else:
                anti_color = 'yellow'
            
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
    
class SEALEnhancedPlanGenerator:
    def __init__(self, model_name='gpt2'):
        self.reverse_tree_planner = PlanGenerator()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    async def generate_enhanced_plan(self, task: str, rag_info: Dict[str, Any]) -> Dict[str, Any]:
        # First, generate the reverse tree plan
        initial_plan = await self.reverse_tree_planner.generate_plan(task, rag_info)
        
        # Now, enhance each sub-goal in the plan
        enhanced_plan = await self._enhance_plan(initial_plan, rag_info)
        
        return enhanced_plan

    async def _enhance_plan(self, plan: Dict[str, Any], rag_info: Dict[str, Any]) -> Dict[str, Any]:
        enhanced_plan = {}
        for key, value in plan.items():
            if isinstance(value, dict):
                # This is a sub-goal, enhance it
                enhanced_sub_goal = await self._enhance_sub_goal(key, value, rag_info)
                enhanced_plan[key] = await self._enhance_plan(enhanced_sub_goal, rag_info)
            else:
                # This is a leaf node (action), keep it as is
                enhanced_plan[key] = value
        return enhanced_plan

    async def _enhance_sub_goal(self, sub_goal: str, sub_plan: Dict[str, Any], rag_info: Dict[str, Any]) -> Dict[str, Any]:
        context = f"Task: {sub_goal}\nExisting Plan: {sub_plan}\nAdditional Info: {rag_info}\nEnhance and expand this sub-goal:"
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        
        output = self.model.generate(
            input_ids,
            max_length=200,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        enhanced_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Parse the enhanced text into a dictionary structure
        # This is a simplified parsing, you might need a more sophisticated parser
        enhanced_lines = enhanced_text.split('\n')
        enhanced_sub_plan = {}
        for line in enhanced_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                enhanced_sub_plan[key.strip()] = value.strip()
        
        # Merge the enhanced sub-plan with the original sub-plan
        merged_sub_plan = {**sub_plan, **enhanced_sub_plan}
        
        return merged_sub_plan

    async def update(self, task: Dict, result: Dict):
        # Update both the reverse tree planner and the language model
        await self.reverse_tree_planner.update(task, result)
        # Fine-tune the language model based on task execution results
        # This is a placeholder; actual implementation would involve more complex fine-tuning
        pass

    def save(self, path: str):
        self.model.save_pretrained(f"{path}/seal_model")
        self.tokenizer.save_pretrained(f"{path}/seal_tokenizer")
        # Add logic to save reverse_tree_planner if needed

    def load(self, path: str):
        self.model = GPT2LMHeadModel.from_pretrained(f"{path}/seal_model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(f"{path}/seal_tokenizer")
        self.model.to(self.device)
        # Add logic to load reverse_tree_planner if needed

     @error_handler.handle_error
    async def generate_task_plan(self, interpreted_intent: Dict[str, Any], key_concepts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a task plan based on the interpreted intent and key concepts.

        Args:
            interpreted_intent (Dict[str, Any]): The interpreted user intent.
            key_concepts (Dict[str, Any]): The extracted key concepts.

        Returns:
            Dict[str, Any]: A dictionary containing the generated task plan.
        """
        prompt = self._create_task_planning_prompt(interpreted_intent, key_concepts)
        response = await self.llm.complete(prompt)
        task_plan = self._parse_task_plan_response(response.text)
        task_graph = self._build_task_graph(task_plan)
        return {
            "plan": task_plan,
            "graph": task_graph
        }

    def _create_task_planning_prompt(self, interpreted_intent: Dict[str, Any], key_concepts: Dict[str, Any]) -> str:
        return f"""
        Based on the following interpreted user intent and key concepts, generate a detailed task plan:

        Interpreted Intent: {interpreted_intent}
        Key Concepts: {key_concepts}

        Please provide a structured task plan that includes:
        1. Main Goal: The overall objective of the plan.
        2. Sub-tasks: Break down the main goal into smaller, actionable tasks.
        3. Priority: Assign a priority level (High, Medium, Low) to each task.
        4. Time Estimate: Provide an estimated time to complete each task.
        5. Required Resources: List any resources or skills needed for each task.
        6. Dependencies: Identify any tasks that depend on the completion of others.
        7. Milestones: Define key milestones or checkpoints in the plan.

        Provide your task plan in a structured JSON format, where each task is an object with properties for priority, time estimate, resources, and dependencies.
        """

    def _parse_task_plan_response(self, response: str) -> Dict[str, Any]:
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse task plan response: {response}")
            raise AIVillageException("Failed to parse task plan response")

    def _build_task_graph(self, task_plan: Dict[str, Any]) -> nx.DiGraph:
        G = nx.DiGraph()
        for task_id, task_data in task_plan['sub_tasks'].items():
            G.add_node(task_id, **task_data)
            for dependency in task_data.get('dependencies', []):
                G.add_edge(dependency, task_id)
        return G

    @error_handler.handle_error
    async def optimize_task_plan(self, task_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize the generated task plan.

        Args:
            task_plan (Dict[str, Any]): The initial task plan.

        Returns:
            Dict[str, Any]: An optimized version of the task plan.
        """
        prompt = self._create_optimization_prompt(task_plan)
        response = await self.llm.complete(prompt)
        optimized_plan = self._parse_task_plan_response(response.text)
        return optimized_plan

    def _create_optimization_prompt(self, task_plan: Dict[str, Any]) -> str:
        return f"""
        Analyze and optimize the following task plan:

        Task Plan: {task_plan}

        Please optimize the task plan considering the following factors:
        1. Efficiency: Identify any redundant or unnecessary tasks.
        2. Parallelization: Determine which tasks can be performed concurrently.
        3. Resource Allocation: Suggest better ways to allocate resources across tasks.
        4. Time Management: Propose ways to reduce the overall time required.
        5. Risk Mitigation: Identify potential risks and suggest mitigation strategies.

        Provide your optimized task plan in the same JSON format as the original plan, with additional comments explaining your optimizations.
        """

    @safe_execute
    async def process_input(self, interpreted_intent: Dict[str, Any], key_concepts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the interpreted intent and key concepts to generate and optimize a task plan.

        Args:
            interpreted_intent (Dict[str, Any]): The interpreted user intent.
            key_concepts (Dict[str, Any]): The extracted key concepts.

        Returns:
            Dict[str, Any]: A dictionary containing the original and optimized task plans.
        """
        initial_plan = await self.generate_task_plan(interpreted_intent, key_concepts)
        optimized_plan = await self.optimize_task_plan(initial_plan['plan'])
        
        return {
            "initial_plan": initial_plan['plan'],
            "optimized_plan": optimized_plan,
            "task_graph": initial_plan['graph']
        }

    def visualize_task_graph(self, graph: nx.DiGraph) -> bytes:
        """
        Visualize the task graph and return the image as bytes.

        Args:
            graph (nx.DiGraph): The NetworkX graph of tasks.

        Returns:
            bytes: The PNG image of the graph visualization as bytes.
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color='lightblue')
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")
        
        # Add priority and time estimate as node labels
        node_labels = {node: f"{data['priority']}\n{data['time_estimate']}" for node, data in graph.nodes(data=True)}
        pos_attrs = {node: (coord[0], coord[1] - 0.1) for node, coord in pos.items()}
        nx.draw_networkx_labels(graph, pos_attrs, labels=node_labels, font_size=6)
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()

# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        planner = TaskPlanningAgent(llm_config)
        
        interpreted_intent = {
            "primary_intent": "Improve team performance",
            "secondary_intents": ["Enhance productivity", "Develop communication skills"],
            "key_entities": ["team", "productivity", "communication skills"],
            "sentiment": "Determined",
            "urgency": "Medium",
            "context": "Workplace improvement"
        }
        
        key_concepts = {
            "team_performance": {
                "attributes": {"current_level": "moderate", "desired_level": "high"},
                "relationships": [
                    {"concept": "productivity", "type": "contributes_to"},
                    {"concept": "communication_skills", "type": "contributes_to"}
                ]
            },
            "productivity": {
                "attributes": {"importance": "high", "current_level": "moderate"},
                "relationships": [
                    {"concept": "time_management", "type": "improves"},
                    {"concept": "task_prioritization", "type": "improves"}
                ]
            },
            "communication_skills": {
                "attributes": {"importance": "high", "current_level": "low"},
                "relationships": [
                    {"concept": "team_collaboration", "type": "improves"},
                    {"concept": "conflict_resolution", "type": "improves"}
                ]
            }
        }
        
        result = await planner.process_input(interpreted_intent, key_concepts)
        
        print("Initial Task Plan:")
        print(result["initial_plan"])
        print("\nOptimized Task Plan:")
        print(result["optimized_plan"])
        
        # Visualize the task graph
        graph_image = planner.visualize_task_graph(result["task_graph"])
        with open("task_graph.png", "wb") as f:
            f.write(graph_image)
        print("\nTask graph visualization saved as 'task_graph.png'")

    asyncio.run(main())

    import logging
from typing import List, Dict, Any
from langroid.language_models.openai_gpt import OpenAIGPTConfig

logger = logging.getLogger(__name__)

class SubGoalGenerator:
    def __init__(self, llm_config: OpenAIGPTConfig):
        self.llm = llm_config.create()

    async def generate_subgoals(self, task: str, context: Dict[str, Any]) -> List[str]:
        """
        Generate a list of subgoals for a given task.

        Args:
            task (str): The main task description.
            context (Dict[str, Any]): Additional context or constraints for the task.

        Returns:
            List[str]: A list of generated subgoals.
        """
        prompt = self._create_prompt(task, context)
        response = await self.llm.complete(prompt)
        subgoals = self._parse_response(response.text)
        return subgoals

    def _create_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """
        Create a prompt for the language model to generate subgoals.

        Args:
            task (str): The main task description.
            context (Dict[str, Any]): Additional context or constraints for the task.

        Returns:
            str: The generated prompt.
        """
        prompt = f"""
        Task: {task}

        Context:
        {self._format_context(context)}

        Given the above task and context, generate a list of specific, actionable subgoals that will help accomplish the main task. Each subgoal should be a clear, concise step towards completing the overall task.

        Please format the subgoals as a numbered list, with each subgoal on a new line.

        Subgoals:
        """
        return prompt

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format the context dictionary into a string.

        Args:
            context (Dict[str, Any]): The context dictionary.

        Returns:
            str: A formatted string representation of the context.
        """
        return "\n".join([f"- {key}: {value}" for key, value in context.items()])

    def _parse_response(self, response: str) -> List[str]:
        """
        Parse the response from the language model into a list of subgoals.

        Args:
            response (str): The raw response from the language model.

        Returns:
            List[str]: A list of parsed subgoals.
        """
        lines = response.strip().split("\n")
        subgoals = [line.split(". ", 1)[-1].strip() for line in lines if line.strip()]
        return subgoals

    async def refine_subgoals(self, subgoals: List[str], feedback: str) -> List[str]:
        """
        Refine the generated subgoals based on feedback.

        Args:
            subgoals (List[str]): The initial list of subgoals.
            feedback (str): Feedback on the subgoals.

        Returns:
            List[str]: A refined list of subgoals.
        """
        prompt = f"""
        Original Subgoals:
        {self._format_subgoals(subgoals)}

        Feedback: {feedback}

        Based on the above feedback, please refine and improve the subgoals. Ensure that the refined subgoals address the feedback while maintaining clarity and actionability.

        Refined Subgoals:
        """
        response = await self.llm.complete(prompt)
        refined_subgoals = self._parse_response(response.text)
        return refined_subgoals

    def _format_subgoals(self, subgoals: List[str]) -> str:
        """
        Format a list of subgoals into a numbered string.

        Args:
            subgoals (List[str]): The list of subgoals.

        Returns:
            str: A formatted string of numbered subgoals.
        """
        return "\n".join([f"{i+1}. {subgoal}" for i, subgoal in enumerate(subgoals)])
import logging
from typing import Dict, Any, List
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from rag_system.error_handling.error_handler import error_handler, safe_execute, AIVillageException
import networkx as nx
import matplotlib.pyplot as plt
import io

logger = logging.getLogger(__name__)

class TaskPlanningAgent:
    def __init__(self, llm_config: OpenAIGPTConfig):
        self.llm = llm_config.create()

    @error_handler.handle_error
    async def generate_task_plan(self, interpreted_intent: Dict[str, Any], key_concepts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a task plan based on the interpreted intent and key concepts.

        Args:
            interpreted_intent (Dict[str, Any]): The interpreted user intent.
            key_concepts (Dict[str, Any]): The extracted key concepts.

        Returns:
            Dict[str, Any]: A dictionary containing the generated task plan.
        """
        prompt = self._create_task_planning_prompt(interpreted_intent, key_concepts)
        response = await self.llm.complete(prompt)
        task_plan = self._parse_task_plan_response(response.text)
        task_graph = self._build_task_graph(task_plan)
        return {
            "plan": task_plan,
            "graph": task_graph
        }

    def _create_task_planning_prompt(self, interpreted_intent: Dict[str, Any], key_concepts: Dict[str, Any]) -> str:
        return f"""
        Based on the following interpreted user intent and key concepts, generate a detailed task plan:

        Interpreted Intent: {interpreted_intent}
        Key Concepts: {key_concepts}

        Please provide a structured task plan that includes:
        1. Main Goal: The overall objective of the plan.
        2. Sub-tasks: Break down the main goal into smaller, actionable tasks.
        3. Priority: Assign a priority level (High, Medium, Low) to each task.
        4. Time Estimate: Provide an estimated time to complete each task.
        5. Required Resources: List any resources or skills needed for each task.
        6. Dependencies: Identify any tasks that depend on the completion of others.
        7. Milestones: Define key milestones or checkpoints in the plan.

        Provide your task plan in a structured JSON format, where each task is an object with properties for priority, time estimate, resources, and dependencies.
        """

    def _parse_task_plan_response(self, response: str) -> Dict[str, Any]:
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse task plan response: {response}")
            raise AIVillageException("Failed to parse task plan response")

    def _build_task_graph(self, task_plan: Dict[str, Any]) -> nx.DiGraph:
        G = nx.DiGraph()
        for task_id, task_data in task_plan['sub_tasks'].items():
            G.add_node(task_id, **task_data)
            for dependency in task_data.get('dependencies', []):
                G.add_edge(dependency, task_id)
        return G

    @error_handler.handle_error
    async def optimize_task_plan(self, task_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize the generated task plan.

        Args:
            task_plan (Dict[str, Any]): The initial task plan.

        Returns:
            Dict[str, Any]: An optimized version of the task plan.
        """
        prompt = self._create_optimization_prompt(task_plan)
        response = await self.llm.complete(prompt)
        optimized_plan = self._parse_task_plan_response(response.text)
        return optimized_plan

    def _create_optimization_prompt(self, task_plan: Dict[str, Any]) -> str:
        return f"""
        Analyze and optimize the following task plan:

        Task Plan: {task_plan}

        Please optimize the task plan considering the following factors:
        1. Efficiency: Identify any redundant or unnecessary tasks.
        2. Parallelization: Determine which tasks can be performed concurrently.
        3. Resource Allocation: Suggest better ways to allocate resources across tasks.
        4. Time Management: Propose ways to reduce the overall time required.
        5. Risk Mitigation: Identify potential risks and suggest mitigation strategies.

        Provide your optimized task plan in the same JSON format as the original plan, with additional comments explaining your optimizations.
        """

    @safe_execute
    async def process_input(self, interpreted_intent: Dict[str, Any], key_concepts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the interpreted intent and key concepts to generate and optimize a task plan.

        Args:
            interpreted_intent (Dict[str, Any]): The interpreted user intent.
            key_concepts (Dict[str, Any]): The extracted key concepts.

        Returns:
            Dict[str, Any]: A dictionary containing the original and optimized task plans.
        """
        initial_plan = await self.generate_task_plan(interpreted_intent, key_concepts)
        optimized_plan = await self.optimize_task_plan(initial_plan['plan'])
        
        return {
            "initial_plan": initial_plan['plan'],
            "optimized_plan": optimized_plan,
            "task_graph": initial_plan['graph']
        }

    def visualize_task_graph(self, graph: nx.DiGraph) -> bytes:
        """
        Visualize the task graph and return the image as bytes.

        Args:
            graph (nx.DiGraph): The NetworkX graph of tasks.

        Returns:
            bytes: The PNG image of the graph visualization as bytes.
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color='lightblue')
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")
        
        # Add priority and time estimate as node labels
        node_labels = {node: f"{data['priority']}\n{data['time_estimate']}" for node, data in graph.nodes(data=True)}
        pos_attrs = {node: (coord[0], coord[1] - 0.1) for node, coord in pos.items()}
        nx.draw_networkx_labels(graph, pos_attrs, labels=node_labels, font_size=6)
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()

# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        planner = TaskPlanningAgent(llm_config)
        
        interpreted_intent = {
            "primary_intent": "Improve team performance",
            "secondary_intents": ["Enhance productivity", "Develop communication skills"],
            "key_entities": ["team", "productivity", "communication skills"],
            "sentiment": "Determined",
            "urgency": "Medium",
            "context": "Workplace improvement"
        }
        
        key_concepts = {
            "team_performance": {
                "attributes": {"current_level": "moderate", "desired_level": "high"},
                "relationships": [
                    {"concept": "productivity", "type": "contributes_to"},
                    {"concept": "communication_skills", "type": "contributes_to"}
                ]
            },
            "productivity": {
                "attributes": {"importance": "high", "current_level": "moderate"},
                "relationships": [
                    {"concept": "time_management", "type": "improves"},
                    {"concept": "task_prioritization", "type": "improves"}
                ]
            },
            "communication_skills": {
                "attributes": {"importance": "high", "current_level": "low"},
                "relationships": [
                    {"concept": "team_collaboration", "type": "improves"},
                    {"concept": "conflict_resolution", "type": "improves"}
                ]
            }
        }
        
        result = await planner.process_input(interpreted_intent, key_concepts)
        
        print("Initial Task Plan:")
        print(result["initial_plan"])
        print("\nOptimized Task Plan:")
        print(result["optimized_plan"])
        
        # Visualize the task graph
        graph_image = planner.visualize_task_graph(result["task_graph"])
        with open("task_graph.png", "wb") as f:
            f.write(graph_image)
        print("\nTask graph visualization saved as 'task_graph.png'")

    asyncio.run(main())
