"""Evolutionary Tournament reasoning technique implementation."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import random
from langroid.agent.chat_agent import ChatAgent

from .base import AgentTechnique, TechniqueResult

@dataclass
class Individual:
    """A solution individual in the evolutionary process."""
    content: str
    reasoning: str
    fitness: float
    generation: int
    parent_ids: List[str]  # IDs of parent solutions
    mutation_rate: float

@dataclass
class Generation:
    """A generation in the evolutionary process."""
    number: int
    individuals: List[Individual]
    best_fitness: float
    average_fitness: float
    diversity: float

class EvolutionaryTournamentTechnique(AgentTechnique):
    """
    Implementation of Evolutionary Tournament reasoning technique.
    
    This technique evolves solutions through:
    - Tournament selection
    - Crossover between solutions
    - Controlled mutation
    - Fitness-based survival
    """
    
    def __init__(
        self,
        population_size: int = 10,
        num_generations: int = 5,
        tournament_size: int = 3,
        mutation_rate: float = 0.1,
        elite_size: int = 2
    ):
        super().__init__(
            thought="Evolutionary Tournament evolves and refines ideas through cycles of "
                   "mutation, selection, and recombination.",
            name="Evolutionary Tournament",
            code=self.__class__.__module__
        )
        self.population_size = population_size
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        self.generations: List[Generation] = []
        self.best_individual: Optional[Individual] = None
        self.overall_confidence: float = 0.0
    
    def get_initial_prompt(self, task: str) -> str:
        """Create prompt for generating initial population."""
        return f"""
        Generate a diverse initial solution for this task:

        Task: {task}

        Be creative and explore different approaches.
        Each solution should be significantly different from others.

        Format your response as:
        Solution: [your solution]
        Reasoning: [explain your approach]
        """

    def get_fitness_prompt(
        self,
        solution: str,
        task: str
    ) -> str:
        """Create prompt for evaluating solution fitness."""
        return f"""
        Evaluate this solution's fitness for the given task:

        Task: {task}

        Solution:
        {solution}

        Consider these criteria:
        1. Correctness
        2. Completeness
        3. Efficiency
        4. Creativity
        5. Practicality

        Rate each criterion from 0-10 and provide an overall fitness score.

        Format your response as:
        Correctness: [0-10]
        Completeness: [0-10]
        Efficiency: [0-10]
        Creativity: [0-10]
        Practicality: [0-10]
        Overall Fitness: [0-1]
        Explanation: [justify your ratings]
        """

    def get_crossover_prompt(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> str:
        """Create prompt for solution crossover."""
        return f"""
        Create a new solution by combining the best aspects of these parent solutions:

        Parent 1:
        {parent1.content}
        Reasoning: {parent1.reasoning}

        Parent 2:
        {parent2.content}
        Reasoning: {parent2.reasoning}

        Create a solution that:
        1. Combines strengths from both parents
        2. Avoids their weaknesses
        3. Introduces some novel elements

        Format your response as:
        Solution: [combined solution]
        Reasoning: [explain how you combined the parents]
        Novel Elements: [what new ideas you introduced]
        """

    def get_mutation_prompt(
        self,
        solution: Individual,
        mutation_rate: float
    ) -> str:
        """Create prompt for solution mutation."""
        creativity_level = "major changes" if mutation_rate > 0.5 else \
                         "moderate improvements" if mutation_rate > 0.2 else \
                         "minor refinements"

        return f"""
        Modify this solution with {creativity_level}:

        Original Solution:
        {solution.content}
        Original Reasoning: {solution.reasoning}

        Mutation Rate: {mutation_rate}
        (Higher rate means more significant changes)

        Create a modified version that:
        1. Preserves core strengths
        2. Addresses weaknesses
        3. Introduces novel elements
        4. Maintains solution validity

        Format your response as:
        Solution: [modified solution]
        Reasoning: [explain your modifications]
        Changes Made: [list specific changes]
        """

    async def apply(self, agent: ChatAgent, task: str) -> TechniqueResult:
        """
        Apply Evolutionary Tournament reasoning to a task.
        
        Args:
            agent: ChatAgent instance to use for reasoning
            task: Task description or problem to solve
            
        Returns:
            TechniqueResult containing the reasoning process and result
        """
        # Clear previous state
        self.generations = []
        self.best_individual = None
        self.overall_confidence = 0.0

        # Generate initial population
        population = await self._initialize_population(agent, task)
        generation = await self._evaluate_generation(agent, task, population, 0)
        self.generations.append(generation)
        self.best_individual = max(population, key=lambda x: x.fitness)

        # Evolution loop
        for gen_num in range(1, self.num_generations):
            # Select elite individuals
            population.sort(key=lambda x: x.fitness, reverse=True)
            elite = population[:self.elite_size]

            # Generate rest of population through tournament selection and breeding
            new_population = list(elite)  # Start with elite
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = await self._tournament_select(agent, task, population)
                parent2 = await self._tournament_select(agent, task, population)

                # Crossover
                child = await self._crossover(agent, parent1, parent2)

                # Mutation
                if random.random() < self.mutation_rate:
                    child = await self._mutate(agent, child)

                new_population.append(child)

            # Evaluate new generation
            population = new_population
            generation = await self._evaluate_generation(agent, task, population, gen_num)
            self.generations.append(generation)

            # Update best individual
            gen_best = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > self.best_individual.fitness:
                self.best_individual = gen_best

        # Calculate overall confidence based on evolution progress
        self.overall_confidence = self._calculate_overall_confidence()

        return self._create_result()

    async def _initialize_population(
        self,
        agent: ChatAgent,
        task: str
    ) -> List[Individual]:
        """Generate initial population."""
        population = []
        for _ in range(self.population_size):
            prompt = self.get_initial_prompt(task)
            response = await agent.llm_response(prompt)
            solution = self._parse_solution(response.content)
            
            individual = Individual(
                content=solution['content'],
                reasoning=solution['reasoning'],
                fitness=0.0,  # Will be evaluated later
                generation=0,
                parent_ids=[],
                mutation_rate=self.mutation_rate
            )
            population.append(individual)

        return population

    def _parse_solution(self, response: str) -> Dict[str, Any]:
        """Parse solution from response."""
        result = {
            'content': '',
            'reasoning': ''
        }

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Solution:'):
                result['content'] = line[len('Solution:'):].strip()
            elif line.startswith('Reasoning:'):
                result['reasoning'] = line[len('Reasoning:'):].strip()

        return result

    async def _evaluate_generation(
        self,
        agent: ChatAgent,
        task: str,
        population: List[Individual],
        generation: int
    ) -> Generation:
        """Evaluate fitness of all individuals in a generation."""
        # Evaluate each individual
        for individual in population:
            fitness_prompt = self.get_fitness_prompt(individual.content, task)
            fitness_response = await agent.llm_response(fitness_prompt)
            fitness_result = self._parse_fitness(fitness_response.content)
            individual.fitness = fitness_result['overall_fitness']

        # Calculate generation statistics
        best_fitness = max(ind.fitness for ind in population)
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        diversity = self._calculate_diversity(population)

        return Generation(
            number=generation,
            individuals=population,
            best_fitness=best_fitness,
            average_fitness=avg_fitness,
            diversity=diversity
        )

    def _parse_fitness(self, response: str) -> Dict[str, Any]:
        """Parse fitness evaluation response."""
        result = {
            'correctness': 0,
            'completeness': 0,
            'efficiency': 0,
            'creativity': 0,
            'practicality': 0,
            'overall_fitness': 0.0,
            'explanation': ''
        }

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Correctness:'):
                result['correctness'] = int(line[len('Correctness:'):].strip())
            elif line.startswith('Completeness:'):
                result['completeness'] = int(line[len('Completeness:'):].strip())
            elif line.startswith('Efficiency:'):
                result['efficiency'] = int(line[len('Efficiency:'):].strip())
            elif line.startswith('Creativity:'):
                result['creativity'] = int(line[len('Creativity:'):].strip())
            elif line.startswith('Practicality:'):
                result['practicality'] = int(line[len('Practicality:'):].strip())
            elif line.startswith('Overall Fitness:'):
                try:
                    result['overall_fitness'] = float(line[len('Overall Fitness:'):].strip())
                except ValueError:
                    result['overall_fitness'] = 0.0
            elif line.startswith('Explanation:'):
                result['explanation'] = line[len('Explanation:'):].strip()

        return result

    async def _tournament_select(
        self,
        agent: ChatAgent,
        task: str,
        population: List[Individual]
    ) -> Individual:
        """Select individual through tournament selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    async def _crossover(
        self,
        agent: ChatAgent,
        parent1: Individual,
        parent2: Individual
    ) -> Individual:
        """Create new individual through crossover."""
        crossover_prompt = self.get_crossover_prompt(parent1, parent2)
        crossover_response = await agent.llm_response(crossover_prompt)
        child_solution = self._parse_solution(crossover_response.content)

        return Individual(
            content=child_solution['content'],
            reasoning=child_solution['reasoning'],
            fitness=0.0,  # Will be evaluated later
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[id(parent1), id(parent2)],
            mutation_rate=self.mutation_rate
        )

    async def _mutate(
        self,
        agent: ChatAgent,
        individual: Individual
    ) -> Individual:
        """Mutate an individual."""
        mutation_prompt = self.get_mutation_prompt(
            individual,
            individual.mutation_rate
        )
        mutation_response = await agent.llm_response(mutation_prompt)
        mutated_solution = self._parse_solution(mutation_response.content)

        return Individual(
            content=mutated_solution['content'],
            reasoning=mutated_solution['reasoning'],
            fitness=0.0,  # Will be evaluated later
            generation=individual.generation,
            parent_ids=individual.parent_ids,
            mutation_rate=individual.mutation_rate
        )

    def _calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0

        # Calculate average pairwise difference between solutions
        total_diff = 0
        count = 0
        for i, ind1 in enumerate(population):
            for ind2 in population[i+1:]:
                # Simple string difference metric
                total_diff += len(set(ind1.content.split()) ^ set(ind2.content.split()))
                count += 1

        return total_diff / count if count > 0 else 0.0

    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence from evolution progress."""
        if not self.generations:
            return 0.0

        # Consider:
        # 1. Best fitness achieved
        # 2. Fitness improvement over generations
        # 3. Population diversity
        best_fitness = self.best_individual.fitness
        
        fitness_improvements = [
            gen.best_fitness - prev_gen.best_fitness
            for gen, prev_gen in zip(self.generations[1:], self.generations[:-1])
        ]
        avg_improvement = sum(fitness_improvements) / len(fitness_improvements) if fitness_improvements else 0

        final_diversity = self.generations[-1].diversity

        # Combine factors (weights could be adjusted)
        confidence = (
            0.5 * best_fitness +
            0.3 * (1.0 if avg_improvement > 0 else 0.5) +
            0.2 * (final_diversity / (self.population_size / 2))
        )

        return min(1.0, max(0.0, confidence))

    def _create_result(self) -> TechniqueResult:
        """Create the final technique result."""
        thought_process = []
        
        # Document evolution process
        for gen in self.generations:
            thought_process.extend([
                f"Generation {gen.number}:",
                f"Best Fitness: {gen.best_fitness:.3f}",
                f"Average Fitness: {gen.average_fitness:.3f}",
                f"Diversity: {gen.diversity:.3f}",
                "---"
            ])

        return TechniqueResult(
            thought='\n'.join(thought_process),
            result=self.best_individual.content,
            confidence=self.overall_confidence,
            metadata={
                'generations': [
                    {
                        'number': gen.number,
                        'best_fitness': gen.best_fitness,
                        'average_fitness': gen.average_fitness,
                        'diversity': gen.diversity,
                        'population_size': len(gen.individuals)
                    }
                    for gen in self.generations
                ],
                'best_individual': {
                    'content': self.best_individual.content,
                    'reasoning': self.best_individual.reasoning,
                    'fitness': self.best_individual.fitness,
                    'generation': self.best_individual.generation
                }
            }
        )

# Register the technique
from .registry import TechniqueRegistry
TechniqueRegistry.register(EvolutionaryTournamentTechnique())
