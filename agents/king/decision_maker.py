from typing import List, Dict, Any
from .problem_analyzer import ProblemAnalyzer
from .plan_generator import PlanGenerator
from ..communication.protocol import StandardCommunicationProtocol
from ..utils.exceptions import AIVillageException

class DecisionMaker:
    def __init__(self, communication_protocol: StandardCommunicationProtocol, rag_system):
        self.communication_protocol = communication_protocol
        self.rag_system = rag_system
        self.problem_analyzer = ProblemAnalyzer(communication_protocol)
        self.plan_generator = PlanGenerator()

    async def make_decision(self, task: str) -> Dict[str, Any]:
        try:
            # Check RAG system for similar past tasks
            rag_info = await self.rag_system.query(task)
            
            # Problem exploration and analysis
            problem_analysis = await self.problem_analyzer.analyze(task, rag_info)
            
            # Determine and rank success criteria
            criteria = self._determine_success_criteria(problem_analysis)
            ranked_criteria = self._rank_criteria(criteria)
            
            # Generate and evaluate alternatives
            alternatives = self._generate_alternatives(problem_analysis)
            evaluated_alternatives = self._evaluate_alternatives(alternatives, ranked_criteria)
            
            # Choose best alternative
            best_alternative = max(evaluated_alternatives, key=lambda x: x['score'])
            
            # Generate plan
            plan = await self.plan_generator.generate_plan(best_alternative['alternative'], problem_analysis)
            
            return {
                'chosen_alternative': best_alternative['alternative'],
                'plan': plan,
                'problem_analysis': problem_analysis,
                'criteria': ranked_criteria,
                'alternatives': evaluated_alternatives
            }
        except Exception as e:
            raise AIVillageException(f"Error in decision making process: {str(e)}")

    async def _determine_success_criteria(self, problem_analysis: Dict[str, Any]) -> List[str]:
        prompt = f"Given the following problem analysis: {problem_analysis}, determine the key success criteria for this task. Output the criteria as a comma-separated list."
        response = await self.ai_provider.generate_text(prompt)
        return [criterion.strip() for criterion in response.split(',')]

    async def _rank_criteria(self, criteria: List[str]) -> List[Dict[str, Any]]:
        prompt = f"Rank the following criteria in order of importance: {', '.join(criteria)}. Output the ranking as a JSON list of dictionaries, each containing 'criterion' and 'rank' keys."
        response = await self.ai_provider.generate_structured_response(prompt)
        return response

    async def _simplify_outcomes(self, ranked_criteria: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        prompt = f"For each of the following ranked criteria: {ranked_criteria}, provide two possible outcomes: a positive and a negative. Output the results as a JSON dictionary where each key is a criterion and the value is a list of two outcomes."
        return await self.ai_provider.generate_structured_response(prompt)

    async def _create_utility_chart(self, outcomes: Dict[str, List[str]]) -> Dict[str, float]:
        combinations = self._generate_combinations(outcomes)
        utility_chart = {}
        for combo in combinations:
            prompt = f"On a scale of 0 to 100, rate the desirability of the following outcome combination: {combo}. Output only the numerical score."
            score = float(await self.ai_provider.generate_text(prompt))
            utility_chart[tuple(combo)] = score
        return utility_chart

    def _generate_combinations(self, outcomes: Dict[str, List[str]]) -> List[List[str]]:
        import itertools
        return list(itertools.product(*outcomes.values()))

    async def _generate_alternatives(self, problem_analysis: Dict[str, Any]) -> List[str]:
        prompt = f"Given the following problem analysis: {problem_analysis}, generate a list of potential solutions or alternatives. Output the alternatives as a JSON list of strings."
        return await self.ai_provider.generate_structured_response(prompt)

    async def _rank_alternatives(self, alternatives: List[str], utility_chart: Dict[str, float]) -> List[Dict[str, Any]]:
        ranked_alternatives = []
        for alt in alternatives:
            prompt = f"Given the alternative '{alt}' and the utility chart {utility_chart}, estimate how well this alternative satisfies each outcome combination. Output a JSON dictionary where keys are outcome combinations and values are probabilities (0-1) of achieving that combination."
            probabilities = await self.ai_provider.generate_structured_response(prompt)
            score = sum(prob * utility_chart[combo] for combo, prob in probabilities.items())
            ranked_alternatives.append({'alternative': alt, 'score': score})
        return sorted(ranked_alternatives, key=lambda x: x['score'], reverse=True)

    async def _determine_probabilities(self, alternatives: List[Dict[str, Any]], outcomes: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        prob_trees = {}
        for alt in alternatives:
            prompt = f"For the alternative '{alt['alternative']}', estimate the probability of each outcome in {outcomes}. Output a JSON dictionary where keys are criteria and values are dictionaries mapping outcomes to probabilities (0-1)."
            prob_trees[alt['alternative']] = await self.ai_provider.generate_structured_response(prompt)
        return prob_trees

    async def _calculate_expected_utility(self, alternatives: List[Dict[str, Any]], prob_trees: Dict[str, Dict[str, float]], utility_chart: Dict[str, float]) -> List[Dict[str, Any]]:
        for alt in alternatives:
            probabilities = prob_trees[alt['alternative']]
            combinations = self._generate_combinations(probabilities)
            expected_utility = 0
            for combo in combinations:
                prob = 1
                for i, outcome in enumerate(combo):
                    criterion = list(probabilities.keys())[i]
                    prob *= probabilities[criterion][outcome]
                expected_utility += prob * utility_chart[tuple(combo)]
            alt['score'] = expected_utility
        return sorted(alternatives, key=lambda x: x['score'], reverse=True)

    async def _create_implementation_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Given the following plan: {plan}, create an implementation strategy that includes monitoring, feedback analysis, and troubleshooting steps. Output the result as a JSON dictionary with keys 'monitoring', 'feedback_analysis', and 'troubleshooting', each containing a list of steps."
        return await self.ai_provider.generate_structured_response(prompt)