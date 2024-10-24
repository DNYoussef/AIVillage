"""Least-to-Most reasoning technique implementation."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from langroid.agent.chat_agent import ChatAgent

from .base import AgentTechnique, TechniqueResult

@dataclass
class SubProblem:
    """A sub-problem in the least-to-most decomposition."""
    number: int
    description: str
    solution: str
    complexity: float  # 0-1 scale
    dependencies: List[int]  # Numbers of sub-problems this depends on
    confidence: float

class LeastToMostTechnique(AgentTechnique):
    """
    Implementation of Least-to-Most reasoning technique.
    
    This technique breaks down complex problems into simpler sub-problems,
    solving them in order from least to most complex. Each solution builds
    upon previous solutions, creating a natural progression toward the final answer.
    """
    
    def __init__(self):
        super().__init__(
            thought="Least-to-Most Prompting breaks complex problems into simpler sub-problems, "
                   "solving them sequentially to build up to the final solution.",
            name="Least-to-Most",
            code=self.__class__.__module__
        )
        self.sub_problems: List[SubProblem] = []
        self.final_solution: Optional[str] = None
        self.overall_confidence: float = 0.0
    
    def get_decomposition_prompt(self, task: str) -> str:
        """Create prompt for problem decomposition."""
        return f"""
        Break down this complex task into simpler sub-problems:

        Task: {task}

        Decompose the task into 3-5 sub-problems, ordered from least to most complex.
        Each sub-problem should build toward solving the main task.

        Format your response as:
        Sub-problem 1:
        Description: [simplest sub-problem]
        Complexity: [0-1]
        Dependencies: [list of sub-problem numbers this depends on, or "none"]

        Sub-problem 2:
        ...

        Make sure each sub-problem:
        1. Is simpler than the original task
        2. Builds logically toward the solution
        3. Depends only on earlier sub-problems
        """

    def get_solution_prompt(self, sub_problem: str, context: str) -> str:
        """Create prompt for solving a sub-problem."""
        return f"""
        Solve this sub-problem using the given context:

        Context from previous solutions:
        {context}

        Sub-problem: {sub_problem}

        Format your response as:
        Solution: [your solution]
        Reasoning: [how you arrived at this solution]
        Confidence: [0-1]
        """

    async def apply(self, agent: ChatAgent, task: str) -> TechniqueResult:
        """
        Apply Least-to-Most reasoning to a task.
        
        Args:
            agent: ChatAgent instance to use for reasoning
            task: Task description or problem to solve
            
        Returns:
            TechniqueResult containing the reasoning process and result
        """
        # Clear previous state
        self.sub_problems = []
        self.final_solution = None
        self.overall_confidence = 0.0

        # Get problem decomposition
        decomp_prompt = self.get_decomposition_prompt(task)
        decomp_response = await agent.llm_response(decomp_prompt)
        sub_problems = self._parse_decomposition(decomp_response.content)

        # Sort sub-problems by dependencies
        sorted_problems = self._topological_sort(sub_problems)
        context = task

        # Solve each sub-problem in order
        for sp in sorted_problems:
            # Get solution
            solution_prompt = self.get_solution_prompt(sp.description, context)
            solution_response = await agent.llm_response(solution_prompt)
            solution_data = self._parse_solution(solution_response.content)

            # Update sub-problem with solution
            sp.solution = solution_data['solution']
            sp.confidence = solution_data['confidence']
            self.sub_problems.append(sp)

            # Update context with new solution
            context = self._build_context()

        # Generate final synthesis
        synthesis = await self._synthesize_solution(agent, task)
        self.final_solution = synthesis.result
        self.overall_confidence = synthesis.confidence

        return self._create_result()

    def _parse_decomposition(self, response: str) -> List[SubProblem]:
        """Parse the problem decomposition response."""
        sub_problems = []
        current_problem = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Sub-problem '):
                if current_problem:
                    sub_problems.append(current_problem)
                current_problem = SubProblem(
                    number=len(sub_problems) + 1,
                    description='',
                    solution='',
                    complexity=0.0,
                    dependencies=[],
                    confidence=0.0
                )
            elif line.startswith('Description:'):
                if current_problem:
                    current_problem.description = line[len('Description:'):].strip()
            elif line.startswith('Complexity:'):
                if current_problem:
                    try:
                        current_problem.complexity = float(line[len('Complexity:'):].strip())
                    except ValueError:
                        current_problem.complexity = 0.5
            elif line.startswith('Dependencies:'):
                if current_problem:
                    deps = line[len('Dependencies:'):].strip()
                    if deps.lower() != 'none':
                        current_problem.dependencies = [
                            int(d.strip()) for d in deps.split(',')
                            if d.strip().isdigit()
                        ]

        if current_problem:
            sub_problems.append(current_problem)

        return sub_problems

    def _parse_solution(self, response: str) -> Dict[str, Any]:
        """Parse a solution response."""
        result = {
            'solution': '',
            'reasoning': '',
            'confidence': 0.0
        }

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Solution:'):
                result['solution'] = line[len('Solution:'):].strip()
            elif line.startswith('Reasoning:'):
                result['reasoning'] = line[len('Reasoning:'):].strip()
            elif line.startswith('Confidence:'):
                try:
                    result['confidence'] = float(line[len('Confidence:'):].strip())
                except ValueError:
                    result['confidence'] = 0.5

        return result

    def _topological_sort(self, problems: List[SubProblem]) -> List[SubProblem]:
        """Sort sub-problems based on dependencies."""
        # Create adjacency list
        graph = {p.number: p.dependencies for p in problems}
        
        # Find nodes with no dependencies
        no_deps = [p for p in problems if not p.dependencies]
        
        sorted_problems = []
        while no_deps:
            # Add a node with no dependencies
            current = no_deps.pop(0)
            sorted_problems.append(current)
            
            # Remove current node from dependencies
            for p in problems:
                if current.number in p.dependencies:
                    p.dependencies.remove(current.number)
                    if not p.dependencies:
                        no_deps.append(p)
        
        return sorted_problems

    def _build_context(self) -> str:
        """Build context string from solved sub-problems."""
        context = []
        for sp in self.sub_problems:
            context.extend([
                f"Sub-problem {sp.number}: {sp.description}",
                f"Solution: {sp.solution}",
                "---"
            ])
        return '\n'.join(context)

    async def _synthesize_solution(
        self,
        agent: ChatAgent,
        task: str
    ) -> TechniqueResult:
        """Synthesize final solution from sub-problem solutions."""
        synthesis_prompt = f"""
        Based on the solutions to all sub-problems:

        {self._build_context()}

        Provide a final solution to the original task:
        {task}

        Format your response as:
        Solution: [your final solution]
        Reasoning: [how the sub-solutions led to this solution]
        Confidence: [0-1]
        """

        response = await agent.llm_response(synthesis_prompt)
        synthesis = self._parse_solution(response.content)

        return TechniqueResult(
            thought=f"Synthesized solution from {len(self.sub_problems)} sub-problems",
            result=synthesis['solution'],
            confidence=synthesis['confidence'],
            metadata={
                'sub_problems': [
                    {
                        'number': sp.number,
                        'description': sp.description,
                        'solution': sp.solution,
                        'complexity': sp.complexity,
                        'dependencies': sp.dependencies,
                        'confidence': sp.confidence
                    }
                    for sp in self.sub_problems
                ]
            }
        )

    def _create_result(self) -> TechniqueResult:
        """Create the final technique result."""
        thought_process = []
        for sp in self.sub_problems:
            thought_process.extend([
                f"Sub-problem {sp.number}: {sp.description}",
                f"Complexity: {sp.complexity}",
                f"Dependencies: {sp.dependencies}",
                f"Solution: {sp.solution}",
                "---"
            ])

        return TechniqueResult(
            thought='\n'.join(thought_process),
            result=self.final_solution,
            confidence=self.overall_confidence,
            metadata={
                'sub_problems': [
                    {
                        'number': sp.number,
                        'description': sp.description,
                        'solution': sp.solution,
                        'complexity': sp.complexity,
                        'dependencies': sp.dependencies,
                        'confidence': sp.confidence
                    }
                    for sp in self.sub_problems
                ]
            }
        )

# Register the technique
from .registry import TechniqueRegistry
TechniqueRegistry.register(LeastToMostTechnique())
