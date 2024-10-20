import logging
import math
import random
from typing import List, Dict, Any, Tuple
import asyncio
from collections import defaultdict
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from rag_system.core.pipeline import EnhancedRAGPipeline
from .quality_assurance_layer import QualityAssuranceLayer
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from agents.utils.exceptions import AIVillageException
from .decision_maker import DecisionMaker

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
        self.decision_maker = DecisionMaker(communication_protocol, rag_system, agent, quality_assurance_layer)

    async def make_decision(self, content: str, eudaimonia_score: float) -> Dict[str, Any]:
        return await self.decision_maker.make_decision(content, eudaimonia_score)

    async def update_model(self, task: Dict[str, Any], result: Any):
        try:
            logger.info(f"Updating unified decision maker model with task result: {result}")
            await self.mcts_update(task, result)
            await self.quality_assurance_layer.update_task_history(task, result.get('performance', 0.5), result.get('uncertainty', 0.5))
        except Exception as e:
            logger.error(f"Error updating unified decision maker model: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error updating unified decision maker model: {str(e)}")

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

    async def mcts_update(self, task, result):
        # Update MCTS statistics based on task execution results
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
        logger.info(f"Updated available agents: {self.available_agents}")

    async def save_models(self, path: str):
        # Implement save logic for unified decision maker models
        pass

    async def load_models(self, path: str):
        # Implement load logic for unified decision maker models
        pass

    async def introspect(self) -> Dict[str, Any]:
        return {
            "type": "UnifiedDecisionMaker",
            "description": "Makes decisions based on task content, RAG information, eudaimonia score, and rule compliance",
            "available_agents": self.available_agents,
            "quality_assurance_info": self.quality_assurance_layer.get_info(),
            "mcts_stats": dict(self.stats)
        }
