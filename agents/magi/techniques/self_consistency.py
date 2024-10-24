"""Self Consistency reasoning technique implementation."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
from collections import defaultdict
import numpy as np
from langroid.agent.chat_agent import ChatAgent

from .base import AgentTechnique, TechniqueResult

@dataclass
class Solution:
    """A candidate solution in the self-consistency process."""
    content: str
    reasoning: str
    confidence: float
    similarity_scores: Dict[str, float]  # Similarity to other solutions

@dataclass
class ConsistencyGroup:
    """A group of consistent solutions."""
    solutions: List[Solution]
    average_similarity: float
    group_confidence: float

class SelfConsistencyTechnique(AgentTechnique):
    """
    Implementation of Self Consistency reasoning technique.
    
    This technique generates multiple solutions independently and then:
    - Measures consistency between solutions
    - Groups similar solutions
    - Selects the most reliable consensus
    - Provides confidence based on agreement
    """
    
    def __init__(
        self,
        num_solutions: int = 5,
        similarity_threshold: float = 0.7,
        min_group_size: int = 2
    ):
        super().__init__(
            thought="Self-Consistency generates multiple solutions and selects the majority answer, "
                   "increasing answer reliability through an ensemble method.",
            name="Self-Consistency",
            code=self.__class__.__module__
        )
        self.num_solutions = num_solutions
        self.similarity_threshold = similarity_threshold
        self.min_group_size = min_group_size
        
        self.solutions: List[Solution] = []
        self.consistency_groups: List[ConsistencyGroup] = []
        self.final_solution: Optional[str] = None
        self.overall_confidence: float = 0.0
    
    def get_solution_prompt(self, task: str) -> str:
        """Create prompt for generating a solution."""
        return f"""
        Solve this task independently:

        Task: {task}

        Provide a complete solution with your reasoning.
        Don't reference other potential solutions.

        Format your response as:
        Solution: [your solution]
        Reasoning: [explain how you arrived at this solution]
        Confidence: [0-1]
        """

    def get_similarity_prompt(
        self,
        solution1: str,
        solution2: str
    ) -> str:
        """Create prompt for comparing two solutions."""
        return f"""
        Compare these two solutions and rate their similarity:

        Solution 1:
        {solution1}

        Solution 2:
        {solution2}

        Consider:
        1. Core approach/methodology
        2. Key steps/components
        3. Final outcome
        4. Underlying reasoning

        Rate the overall similarity from 0 (completely different) to 1 (essentially identical).
        Explain your rating.

        Format your response as:
        Similarity: [0-1]
        Explanation: [why you gave this rating]
        """

    async def apply(self, agent: ChatAgent, task: str) -> TechniqueResult:
        """
        Apply Self Consistency reasoning to a task.
        
        Args:
            agent: ChatAgent instance to use for reasoning
            task: Task description or problem to solve
            
        Returns:
            TechniqueResult containing the reasoning process and result
        """
        # Clear previous state
        self.solutions = []
        self.consistency_groups = []
        self.final_solution = None
        self.overall_confidence = 0.0

        # Generate multiple solutions
        for _ in range(self.num_solutions):
            solution_prompt = self.get_solution_prompt(task)
            solution_response = await agent.llm_response(solution_prompt)
            solution = self._parse_solution(solution_response.content)
            self.solutions.append(solution)

        # Calculate similarity between all pairs
        for i, sol1 in enumerate(self.solutions):
            sol1.similarity_scores = {}
            for j, sol2 in enumerate(self.solutions):
                if i != j:
                    similarity = await self._calculate_similarity(agent, sol1, sol2)
                    sol1.similarity_scores[j] = similarity

        # Group similar solutions
        self.consistency_groups = self._group_solutions()

        # Select best group and solution
        best_group = max(
            self.consistency_groups,
            key=lambda g: g.group_confidence * g.average_similarity * len(g.solutions)
        )

        best_solution = max(
            best_group.solutions,
            key=lambda s: s.confidence
        )

        self.final_solution = best_solution.content
        self.overall_confidence = best_group.group_confidence

        return self._create_result()

    def _parse_solution(self, response: str) -> Solution:
        """Parse solution from response."""
        content = ''
        reasoning = ''
        confidence = 0.0

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Solution:'):
                content = line[len('Solution:'):].strip()
            elif line.startswith('Reasoning:'):
                reasoning = line[len('Reasoning:'):].strip()
            elif line.startswith('Confidence:'):
                try:
                    confidence = float(line[len('Confidence:'):].strip())
                except ValueError:
                    confidence = 0.5

        return Solution(
            content=content,
            reasoning=reasoning,
            confidence=confidence,
            similarity_scores={}
        )

    async def _calculate_similarity(
        self,
        agent: ChatAgent,
        solution1: Solution,
        solution2: Solution
    ) -> float:
        """Calculate similarity between two solutions."""
        similarity_prompt = self.get_similarity_prompt(
            solution1.content,
            solution2.content
        )
        similarity_response = await agent.llm_response(similarity_prompt)
        similarity_result = self._parse_similarity(similarity_response.content)
        return similarity_result['similarity']

    def _parse_similarity(self, response: str) -> Dict[str, Any]:
        """Parse similarity comparison result."""
        result = {
            'similarity': 0.0,
            'explanation': ''
        }

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Similarity:'):
                try:
                    result['similarity'] = float(line[len('Similarity:'):].strip())
                except ValueError:
                    result['similarity'] = 0.0
            elif line.startswith('Explanation:'):
                result['explanation'] = line[len('Explanation:'):].strip()

        return result

    def _group_solutions(self) -> List[ConsistencyGroup]:
        """Group similar solutions together."""
        # Start with each solution in its own group
        groups: List[List[Solution]] = [[s] for s in self.solutions]

        # Merge groups until no more merges are possible
        while True:
            merged = False
            for i in range(len(groups)):
                if merged:
                    break
                for j in range(i + 1, len(groups)):
                    if self._should_merge_groups(groups[i], groups[j]):
                        groups[i].extend(groups[j])
                        groups.pop(j)
                        merged = True
                        break
            if not merged:
                break

        # Convert to ConsistencyGroup objects
        return [
            ConsistencyGroup(
                solutions=group,
                average_similarity=self._calculate_group_similarity(group),
                group_confidence=self._calculate_group_confidence(group)
            )
            for group in groups
            if len(group) >= self.min_group_size
        ]

    def _should_merge_groups(
        self,
        group1: List[Solution],
        group2: List[Solution]
    ) -> bool:
        """Determine if two groups should be merged."""
        # Calculate average similarity between all pairs across groups
        similarities = []
        for sol1 in group1:
            for sol2 in group2:
                if str(id(sol2)) in sol1.similarity_scores:
                    similarities.append(sol1.similarity_scores[str(id(sol2))])

        if not similarities:
            return False

        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity >= self.similarity_threshold

    def _calculate_group_similarity(self, group: List[Solution]) -> float:
        """Calculate average similarity within a group."""
        if len(group) < 2:
            return 1.0

        similarities = []
        for i, sol1 in enumerate(group):
            for j, sol2 in enumerate(group[i+1:], i+1):
                if str(id(sol2)) in sol1.similarity_scores:
                    similarities.append(sol1.similarity_scores[str(id(sol2))])

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_group_confidence(self, group: List[Solution]) -> float:
        """Calculate overall confidence for a group."""
        if not group:
            return 0.0

        # Consider both individual confidences and group size
        avg_confidence = sum(s.confidence for s in group) / len(group)
        size_factor = min(len(group) / self.num_solutions, 1.0)

        return avg_confidence * size_factor

    def _create_result(self) -> TechniqueResult:
        """Create the final technique result."""
        thought_process = []
        
        # Document all solutions
        thought_process.extend([
            "Generated Solutions:",
            "---"
        ])
        for i, solution in enumerate(self.solutions):
            thought_process.extend([
                f"Solution {i+1}:",
                f"Content: {solution.content}",
                f"Reasoning: {solution.reasoning}",
                f"Confidence: {solution.confidence}",
                "---"
            ])

        # Document consistency groups
        thought_process.extend([
            "Consistency Groups:",
            "---"
        ])
        for i, group in enumerate(self.consistency_groups):
            thought_process.extend([
                f"Group {i+1}:",
                f"Size: {len(group.solutions)}",
                f"Average Similarity: {group.average_similarity:.2f}",
                f"Group Confidence: {group.group_confidence:.2f}",
                "---"
            ])

        return TechniqueResult(
            thought='\n'.join(thought_process),
            result=self.final_solution,
            confidence=self.overall_confidence,
            metadata={
                'solutions': [
                    {
                        'content': s.content,
                        'reasoning': s.reasoning,
                        'confidence': s.confidence,
                        'similarity_scores': s.similarity_scores
                    }
                    for s in self.solutions
                ],
                'groups': [
                    {
                        'size': len(g.solutions),
                        'average_similarity': g.average_similarity,
                        'group_confidence': g.group_confidence
                    }
                    for g in self.consistency_groups
                ]
            }
        )

# Register the technique
from .registry import TechniqueRegistry
TechniqueRegistry.register(SelfConsistencyTechnique())
