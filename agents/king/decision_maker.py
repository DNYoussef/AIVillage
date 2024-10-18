import logging
import itertools
import os
import json
from typing import Dict, List, Any
from communications.protocol import StandardCommunicationProtocol, Message, MessageType
from .problem_analyzer import ProblemAnalyzer
from .plan_generator import PlanGenerator
from ..utils.exceptions import AIVillageException
from .mcts import MCTS
from rag_system.core.pipeline import EnhancedRAGPipeline
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from .quality_assurance_layer import QualityAssuranceLayer

logger = logging.getLogger(__name__)

class DecisionMaker:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, rag_system: EnhancedRAGPipeline, agent, quality_assurance_layer: QualityAssuranceLayer):
        self.communication_protocol = communication_protocol
        self.rag_system = rag_system
        self.agent = agent
        self.llm = OpenAIGPTConfig(chat_model="gpt-4").create()
        self.quality_assurance_layer = quality_assurance_layer
        self.mcts = MCTS()
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
        evaluated_alternatives = await self._evaluate_alternatives(alternatives, [
            {"criterion": "eudaimonia", "weight": 0.4},
            {"criterion": "curiosity", "weight": 0.2},
            {"criterion": "protection", "weight": 0.3},
            {"criterion": "self_preservation", "weight": 0.1}
        ])
        
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

    async def update_model(self, task: Dict[str, Any], result: Any):
        try:
            logger.info(f"Updating decision maker model with task result: {result}")
            await self.mcts.update(task, result)
            await self.quality_assurance_layer.update_task_history(task, result.get('performance', 0.5), result.get('uncertainty', 0.5))
        except Exception as e:
            logger.error(f"Error updating decision maker model: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error updating decision maker model: {str(e)}")

    async def save_models(self, path: str):
        try:
            logger.info(f"Saving decision maker models to {path}")
            os.makedirs(path, exist_ok=True)
            self.mcts.save(os.path.join(path, "mcts_model.pt"))
            self.quality_assurance_layer.save(os.path.join(path, "quality_assurance_layer.json"))
            
            # Save other necessary data
            data = {
                "available_agents": self.available_agents
            }
            with open(os.path.join(path, "decision_maker_data.json"), 'w') as f:
                json.dump(data, f)
            
            logger.info("Decision maker models saved successfully")
        except Exception as e:
            logger.error(f"Error saving decision maker models: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error saving decision maker models: {str(e)}")

    async def load_models(self, path: str):
        try:
            logger.info(f"Loading decision maker models from {path}")
            self.mcts.load(os.path.join(path, "mcts_model.pt"))
            self.quality_assurance_layer = QualityAssuranceLayer.load(os.path.join(path, "quality_assurance_layer.json"))
            
            # Load other necessary data
            with open(os.path.join(path, "decision_maker_data.json"), 'r') as f:
                data = json.load(f)
            self.available_agents = data["available_agents"]
            
            logger.info("Decision maker models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading decision maker models: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error loading decision maker models: {str(e)}")

    def update_agent_list(self, agent_list: List[str]):
        self.available_agents = agent_list
        logger.info(f"Updated available agents: {self.available_agents}")

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
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(all_alternatives))

    async def _evaluate_alternatives(self, alternatives: List[str], ranked_criteria: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outcomes = await self._simplify_outcomes(ranked_criteria)
        utility_chart = await self._create_utility_chart(outcomes)
        prob_trees = await self._determine_probabilities(alternatives, outcomes)
        return await self._calculate_expected_utility(alternatives, prob_trees, utility_chart)

    async def _determine_probabilities(self, alternatives: List[str], outcomes: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        prob_trees = {}
        for alt in alternatives:
            prompt = f"For the alternative '{alt}', estimate the probability of each outcome in {outcomes}. Output a JSON dictionary where keys are criteria and values are dictionaries mapping outcomes to probabilities (0-1)."
            prob_trees[alt] = await self.king_agent.generate_structured_response(prompt)
        return prob_trees

    async def _calculate_expected_utility(self, alternatives: List[str], prob_trees: Dict[str, Dict[str, float]], utility_chart: Dict[str, float]) -> List[Dict[str, Any]]:
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

    async def _determine_success_criteria(self, problem_analysis: Dict[str, Any]) -> List[str]:
        prompt = f"Based on the problem analysis: {problem_analysis}, determine the key success criteria for this task. Output as a JSON list of strings."
        return await self.king_agent.generate_structured_response(prompt)

    async def _rank_criteria(self, criteria: List[str]) -> List[Dict[str, Any]]:
        prompt = f"Rank the following success criteria in order of importance: {criteria}. For each criterion, provide a weight (0-1) and a brief explanation. Output as a JSON list of dictionaries, each containing 'criterion', 'weight', and 'explanation' keys."
        return await self.king_agent.generate_structured_response(prompt)

    async def _simplify_outcomes(self, ranked_criteria: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        prompt = f"For each of the following ranked criteria: {ranked_criteria}, provide a list of possible outcomes (success, partial success, failure). Output as a JSON dictionary where keys are criteria and values are lists of outcomes."
        return await self.king_agent.generate_structured_response(prompt)

    async def _create_utility_chart(self, outcomes: Dict[str, List[str]]) -> Dict[str, float]:
        prompt = f"Create a utility chart for the following outcomes: {outcomes}. Assign a utility value (-10 to 10) for each combination of outcomes. Output as a JSON dictionary where keys are tuples of outcomes and values are utility scores."
        return await self.king_agent.generate_structured_response(prompt)

    def _generate_combinations(self, probabilities: Dict[str, Dict[str, float]]) -> List[tuple]:
        criteria = list(probabilities.keys())
        outcomes = [list(probabilities[criterion].keys()) for criterion in criteria]
        return [combo for combo in itertools.product(*outcomes)]

    async def introspect(self) -> Dict[str, Any]:
        return {
            "type": "DecisionMaker",
            "description": "Makes decisions based on task content, RAG information, eudaimonia score, and rule compliance",
            "available_agents": self.available_agents,
            "quality_assurance_info": self.quality_assurance_layer.get_info()
        }

    def save_models(self, path: str):
        # Implement logic to save decision-making models
        pass

    def load_models(self, path: str):
        # Implement logic to load decision-making models
        pass
