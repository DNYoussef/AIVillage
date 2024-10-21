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

logger = logging.getLogger(__name__)

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
