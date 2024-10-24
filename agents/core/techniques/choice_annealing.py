"""Choice Annealing reasoning technique implementation."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import random
import math
from langroid.agent.chat_agent import ChatAgent

from .base import AgentTechnique, TechniqueResult

@dataclass
class Solution:
    """A candidate solution in the annealing process."""
    content: str
    energy: float  # Lower is better
    temperature: float
    iteration: int

@dataclass
class AnnealingStep:
    """A step in the annealing process."""
    iteration: int
    temperature: float
    current_solution: str
    alternatives: List[str]
    selected_solution: str
    energy_delta: float
    acceptance_probability: float

class ChoiceAnnealingTechnique(AgentTechnique):
    """
    Implementation of Choice Annealing reasoning technique.
    
    This technique uses simulated annealing to gradually refine solutions,
    allowing for:
    - Broad exploration at high temperatures
    - Fine-tuning at low temperatures
    - Escape from local optima
    - Controlled convergence to high-quality solutions
    """
    
    def __init__(
        self,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        iterations_per_temp: int = 3
    ):
        super().__init__(
            thought="Choice Annealing gradually narrows down a large set of initial ideas "
                   "to a final, refined solution through temperature-controlled exploration.",
            name="Choice Annealing",
            code=self.__class__.__module__
        )
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.iterations_per_temp = iterations_per_temp
        
        self.steps: List[AnnealingStep] = []
        self.best_solution: Optional[Solution] = None
        self.current_solution: Optional[Solution] = None
        self.temperature = initial_temperature
    
    def get_initial_prompt(self, task: str) -> str:
        """Create prompt for generating initial solutions."""
        return f"""
        Generate {self.iterations_per_temp} diverse initial solutions for this task:

        Task: {task}

        Be creative and explore different approaches. The solutions should be
        significantly different from each other.

        Format each solution as:
        Solution X:
        [your solution]
        Reasoning: [explain your approach]
        """

    def get_alternative_prompt(
        self,
        task: str,
        current_solution: str,
        temperature: float
    ) -> str:
        """Create prompt for generating alternative solutions."""
        creativity_level = "very creative and diverse" if temperature > 0.7 else \
                         "somewhat creative while improving existing ideas" if temperature > 0.3 else \
                         "focused on refining and optimizing"
        
        return f"""
        Current solution:
        {current_solution}

        Temperature: {temperature:.2f}
        (Higher temperature means more creative freedom, lower means more refinement)

        Generate {self.iterations_per_temp} alternative solutions that are {creativity_level}.
        The alternatives should be {'significantly different' if temperature > 0.5 else 'incrementally better'} 
        than the current solution.

        Original task: {task}

        Format each alternative as:
        Alternative X:
        [your solution]
        Improvements: [how this improves upon the current solution]
        """

    def get_evaluation_prompt(self, solutions: List[str], task: str) -> str:
        """Create prompt for evaluating solutions."""
        solutions_text = []
        for i, solution in enumerate(solutions, 1):
            solutions_text.extend([
                f"Solution {i}:",
                solution,
                "---"
            ])

        return f"""
        Evaluate these solutions for the original task:

        Task: {task}

        Solutions:
        {'\n'.join(solutions_text)}

        For each solution, provide:
        1. A score from 0-100 (lower is better)
        2. A brief explanation of the score

        Format each evaluation as:
        Solution X:
        Score: [0-100]
        Explanation: [why this score]
        """

    async def apply(self, agent: ChatAgent, task: str) -> TechniqueResult:
        """
        Apply Choice Annealing reasoning to a task.
        
        Args:
            agent: ChatAgent instance to use for reasoning
            task: Task description or problem to solve
            
        Returns:
            TechniqueResult containing the reasoning process and result
        """
        # Clear previous state
        self.steps = []
        self.best_solution = None
        self.current_solution = None
        self.temperature = self.initial_temperature

        # Generate initial solutions
        initial_prompt = self.get_initial_prompt(task)
        initial_response = await agent.llm_response(initial_prompt)
        initial_solutions = self._parse_solutions(initial_response.content)

        # Evaluate initial solutions
        evaluation_prompt = self.get_evaluation_prompt(initial_solutions, task)
        evaluation_response = await agent.llm_response(evaluation_prompt)
        evaluations = self._parse_evaluations(evaluation_response.content)

        # Select initial solution
        initial_solution = min(
            [
                Solution(
                    content=solution,
                    energy=evaluations[i],
                    temperature=self.temperature,
                    iteration=0
                )
                for i, solution in enumerate(initial_solutions)
            ],
            key=lambda x: x.energy
        )

        self.current_solution = initial_solution
        self.best_solution = initial_solution

        # Annealing process
        iteration = 1
        while self.temperature > self.min_temperature:
            for _ in range(self.iterations_per_temp):
                # Generate alternatives
                alternative_prompt = self.get_alternative_prompt(
                    task,
                    self.current_solution.content,
                    self.temperature
                )
                alternative_response = await agent.llm_response(alternative_prompt)
                alternatives = self._parse_solutions(alternative_response.content)

                # Evaluate alternatives
                evaluation_prompt = self.get_evaluation_prompt(alternatives, task)
                evaluation_response = await agent.llm_response(evaluation_prompt)
                evaluations = self._parse_evaluations(evaluation_response.content)

                # Select next solution
                candidate = min(
                    [
                        Solution(
                            content=solution,
                            energy=evaluations[i],
                            temperature=self.temperature,
                            iteration=iteration
                        )
                        for i, solution in enumerate(alternatives)
                    ],
                    key=lambda x: x.energy
                )

                # Calculate acceptance probability
                energy_delta = candidate.energy - self.current_solution.energy
                acceptance_prob = self._acceptance_probability(
                    energy_delta,
                    self.temperature
                )

                # Record step
                self.steps.append(
                    AnnealingStep(
                        iteration=iteration,
                        temperature=self.temperature,
                        current_solution=self.current_solution.content,
                        alternatives=alternatives,
                        selected_solution=candidate.content,
                        energy_delta=energy_delta,
                        acceptance_probability=acceptance_prob
                    )
                )

                # Accept or reject candidate
                if random.random() < acceptance_prob:
                    self.current_solution = candidate
                    if candidate.energy < self.best_solution.energy:
                        self.best_solution = candidate

                iteration += 1

            # Cool down
            self.temperature *= self.cooling_rate

        return self._create_result()

    def _parse_solutions(self, response: str) -> List[str]:
        """Parse solutions from response."""
        solutions = []
        current_solution = []
        in_solution = False
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Solution ') or line.startswith('Alternative '):
                if current_solution:
                    solutions.append('\n'.join(current_solution))
                    current_solution = []
                in_solution = True
            elif line.startswith('Reasoning:') or line.startswith('Improvements:'):
                in_solution = False
            elif in_solution:
                current_solution.append(line)

        if current_solution:
            solutions.append('\n'.join(current_solution))

        return solutions

    def _parse_evaluations(self, response: str) -> List[float]:
        """Parse evaluation scores from response."""
        scores = []
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Score:'):
                try:
                    score = float(line[len('Score:'):].strip())
                    scores.append(score)
                except ValueError:
                    scores.append(100.0)  # Worst possible score as fallback

        return scores

    def _acceptance_probability(
        self,
        energy_delta: float,
        temperature: float
    ) -> float:
        """Calculate probability of accepting a worse solution."""
        if energy_delta < 0:  # Better solution
            return 1.0
        return math.exp(-energy_delta / temperature)

    def _create_result(self) -> TechniqueResult:
        """Create the final technique result."""
        thought_process = []
        for step in self.steps:
            thought_process.extend([
                f"Iteration {step.iteration} (T={step.temperature:.3f}):",
                f"Current: {step.current_solution}",
                f"Selected: {step.selected_solution}",
                f"Energy Δ: {step.energy_delta:.2f}",
                f"Accept Prob: {step.acceptance_probability:.2f}",
                "---"
            ])

        return TechniqueResult(
            thought='\n'.join(thought_process),
            result=self.best_solution.content,
            confidence=1.0 - (self.best_solution.energy / 100.0),  # Convert energy to confidence
            metadata={
                'steps': [
                    {
                        'iteration': step.iteration,
                        'temperature': step.temperature,
                        'current_solution': step.current_solution,
                        'selected_solution': step.selected_solution,
                        'energy_delta': step.energy_delta,
                        'acceptance_probability': step.acceptance_probability
                    }
                    for step in self.steps
                ],
                'final_temperature': self.temperature,
                'best_energy': self.best_solution.energy
            }
        )

# Register the technique
from .registry import TechniqueRegistry
TechniqueRegistry.register(ChoiceAnnealingTechnique())
