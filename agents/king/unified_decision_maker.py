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
        rag_info = await self.rag_system.process_query(content)
        
        task_vector = self.quality_assurance_layer.eudaimonia_triangulator.get_embedding(content)
        rule_compliance = self.quality_assurance_layer.evaluate_rule_compliance(task_vector)
        
        decision_prompt = f"""
        Task: {content}
        RAG Information: {rag_info}
        Eudaimonia Score: {eudaimonia_score}
        Rule Compliance: {rule_compliance}

        Given the task, the provided information, eudaimonia score, and rule compliance, make a decision that:
        1. Aligns with the goal of moving all living things towards eudaimonia
        2. Embraces and encourages curiosity
        3. Protects the AI village and its inhabitants
        4. Maintains self-preservation unless it interferes with the above points

        Provide your decision and a brief explanation of how it aligns with these principles.
        """

        response = await self.llm.complete(decision_prompt)
        
        decision = response.text
        alternatives = await self._generate_alternatives({"content": content, "rag_info": rag_info})
        
        ranked_criteria = [
            {"criterion": "eudaimonia", "weight": 0.4},
            {"criterion": "curiosity", "weight": 0.2},
            {"criterion": "protection", "weight": 0.3},
            {"criterion": "self_preservation", "weight": 0.1}
        ]
        
        evaluated_alternatives = await self._evaluate_alternatives(alternatives, ranked_criteria)
        
        best_alternative = evaluated_alternatives[0]['alternative']
        
        implementation_plan = await self._create_implementation_plan({"decision": decision, "best_alternative": best_alternative})
        
        return {
            "decision": decision,
            "eudaimonia_score": eudaimonia_score,
            "rule_compliance": rule_compliance,
            "rag_info": rag_info,
            "best_alternative": best_alternative,
            "implementation_plan": implementation_plan
        }

    async def _generate_alternatives(self, problem_analysis: Dict[str, Any]) -> List[str]:
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

    async def _create_implementation_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
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

    async def update_model(self, task: Dict[str, Any], result: Any):
        try:
            logger.info(f"Updating decision maker model with task result: {result}")
            await self.mcts_update(task, result)
            await self.quality_assurance_layer.update_task_history(task, result.get('performance', 0.5), result.get('uncertainty', 0.5))
        except Exception as e:
            logger.error(f"Error updating decision maker model: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error updating decision maker model: {str(e)}")

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
        # Implement save logic for decision maker models
        pass

    async def load_models(self, path: str):
        # Implement load logic for decision maker models
        pass

    async def introspect(self) -> Dict[str, Any]:
        return {
            "type": "UnifiedDecisionMaker",
            "description": "Makes decisions based on task content, RAG information, eudaimonia score, and rule compliance",
            "available_agents": self.available_agents,
            "quality_assurance_info": self.quality_assurance_layer.get_info(),
            "mcts_stats": dict(self.stats)
        }
