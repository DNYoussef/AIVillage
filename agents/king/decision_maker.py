import logging
from typing import List, Dict, Any
from .problem_analyzer import ProblemAnalyzer
from .plan_generator import PlanGenerator
from ..communication.protocol import StandardCommunicationProtocol, Message, MessageType
from ..utils.exceptions import AIVillageException
from ..utils.ai_provider import AIProvider
from .mcts import MCTS

logger = logging.getLogger(__name__)

class DecisionMaker:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, rag_system, ai_provider: AIProvider):
        self.communication_protocol = communication_protocol
        self.rag_system = rag_system
        self.problem_analyzer = ProblemAnalyzer(communication_protocol)
        self.plan_generator = PlanGenerator(ai_provider)
        self.ai_provider = ai_provider
        self.mcts = MCTS()
        self.available_agents = []

    async def make_decision(self, task: str) -> Dict[str, Any]:
        try:
            logger.info(f"Starting decision-making process for task: {task}")
            
            rag_info = await self.rag_system.query(task)
            logger.debug(f"RAG info retrieved: {rag_info}")
            
            problem_analysis = await self.problem_analyzer.analyze(task, rag_info)
            logger.debug(f"Problem analysis completed: {problem_analysis}")
            
            criteria = await self._determine_success_criteria(problem_analysis)
            ranked_criteria = await self._rank_criteria(criteria)
            logger.debug(f"Ranked criteria: {ranked_criteria}")
            
            alternatives = await self._generate_alternatives(problem_analysis)
            evaluated_alternatives = await self._evaluate_alternatives(alternatives, ranked_criteria)
            logger.debug(f"Evaluated alternatives: {evaluated_alternatives}")
            
            best_alternative = max(evaluated_alternatives, key=lambda x: x['score'])
            logger.info(f"Best alternative selected: {best_alternative['alternative']}")
            
            optimized_workflow = await self.mcts.parallel_search(best_alternative['alternative'], self.problem_analyzer, self.plan_generator)
            logger.debug(f"Optimized workflow generated: {optimized_workflow}")
            
            plan = await self.plan_generator.generate_plan(optimized_workflow, problem_analysis)
            logger.info(f"Final plan generated: {plan}")
            
            best_agent = await self.suggest_best_agent(task, problem_analysis, best_alternative['alternative'])
            logger.info(f"Best agent suggested: {best_agent}")
            
            return {
                'chosen_alternative': best_alternative['alternative'],
                'optimized_workflow': optimized_workflow,
                'plan': plan,
                'problem_analysis': problem_analysis,
                'criteria': ranked_criteria,
                'alternatives': evaluated_alternatives,
                'suggested_agent': best_agent
            }
        except Exception as e:
            logger.error(f"Error in decision making process: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error in decision making process: {str(e)}")

    async def suggest_best_agent(self, task: str, problem_analysis: Dict[str, Any], chosen_alternative: str) -> str:
        try:
            prompt = f"Given the task: '{task}', the problem analysis: {problem_analysis}, and the chosen alternative: '{chosen_alternative}', suggest the best agent from the following list to handle this task: {self.available_agents}. Output only the name of the suggested agent."
            suggested_agent = await self.ai_provider.generate_text(prompt)
            return suggested_agent.strip()
        except Exception as e:
            logger.error(f"Error suggesting best agent: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error suggesting best agent: {str(e)}")

    async def update_model(self, task: Dict[str, Any], result: Any):
        try:
            logger.info(f"Updating decision maker model with task result: {result}")
            # Implement model update logic here
            pass
        except Exception as e:
            logger.error(f"Error updating decision maker model: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error updating decision maker model: {str(e)}")

    async def update_mcts(self, task: Dict[str, Any], result: Any):
        try:
            logger.info(f"Updating MCTS with task result: {result}")
            await self.mcts.update(task, result)
        except Exception as e:
            logger.error(f"Error updating MCTS: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error updating MCTS: {str(e)}")

    def save_models(self, path: str):
        try:
            logger.info(f"Saving decision maker models to {path}")
            self.mcts.save(f"{path}/mcts_model.pt")
            # Save other models if necessary
        except Exception as e:
            logger.error(f"Error saving decision maker models: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error saving decision maker models: {str(e)}")

    def load_models(self, path: str):
        try:
            logger.info(f"Loading decision maker models from {path}")
            self.mcts.load(f"{path}/mcts_model.pt")
            # Load other models if necessary
        except Exception as e:
            logger.error(f"Error loading decision maker models: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error loading decision maker models: {str(e)}")

    def update_agent_list(self, agent_list: List[str]):
        self.available_agents = agent_list
        logger.info(f"Updated available agents: {self.available_agents}")

    async def _generate_alternatives(self, problem_analysis: Dict[str, Any]) -> List[str]:
        king_alternatives = await self.ai_provider.generate_structured_response(
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
            prob_trees[alt] = await self.ai_provider.generate_structured_response(prompt)
        return prob_trees

    async def _calculate_expected_utility(self, alternatives: List[str], prob_trees: Dict[str, Dict[str, float]], utility_chart: Dict[str, float]) -> List[Dict[str, Any]]:
        evaluated_alternatives = []
        for alt in alternatives:
            probabilities = prob_trees[alt]
            combinations = self._generate_combinations(probabilities)
            expected_utility = 0
            for combo in combinations:
                prob = 1
                for i, outcome in enumerate(combo):
                    criterion = list(probabilities.keys())[i]
                    prob *= probabilities[criterion][outcome]
                expected_utility += prob * utility_chart[tuple(combo)]
            evaluated_alternatives.append({'alternative': alt, 'score': expected_utility})
        return sorted(evaluated_alternatives, key=lambda x: x['score'], reverse=True)

    async def _create_implementation_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Creating implementation plan")
            prompt = f"Given the following plan: {plan}, create an implementation strategy that includes monitoring, feedback analysis, and troubleshooting steps. Output the result as a JSON dictionary with keys 'monitoring', 'feedback_analysis', and 'troubleshooting', each containing a list of steps."
            implementation_plan = await self.ai_provider.generate_structured_response(prompt)
            logger.debug(f"Implementation plan created: {implementation_plan}")
            return implementation_plan
        except Exception as e:
            logger.error(f"Error creating implementation plan: {str(e)}", exc_info=True)
            raise AIVillageException(f"Error creating implementation plan: {str(e)}")


