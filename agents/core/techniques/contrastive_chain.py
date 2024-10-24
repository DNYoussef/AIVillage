"""Contrastive Chain of Thought reasoning technique implementation."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
from langroid.agent.chat_agent import ChatAgent

from .base import AgentTechnique, TechniqueResult

@dataclass
class ContrastiveStep:
    """A step in the contrastive reasoning process."""
    step_number: int
    correct_reasoning: str
    incorrect_reasoning: str
    explanation: str  # Why correct is better than incorrect
    confidence: float

class ContrastiveChainTechnique(AgentTechnique):
    """
    Implementation of Contrastive Chain of Thought reasoning technique.
    
    This technique enhances reasoning by explicitly considering both correct
    and incorrect approaches, helping to:
    - Identify potential pitfalls
    - Understand why certain approaches work better
    - Strengthen confidence in correct solutions
    - Learn from common mistakes
    """
    
    def __init__(self):
        super().__init__(
            thought="Contrastive Chain-of-Thought includes both correct and incorrect explanations "
                   "to enhance reasoning by showing what not to do, leveraging contrast learning.",
            name="Contrastive Chain",
            code=self.__class__.__module__
        )
        self.steps: List[ContrastiveStep] = []
        self.final_answer: Optional[str] = None
        self.overall_confidence: float = 0.0
    
    def get_prompt(self, task: str) -> str:
        """Create the Contrastive Chain of Thought prompt."""
        return f"""
        Consider this task:

        {task}

        For each step in solving this task:
        1. Provide a correct approach
        2. Provide a plausible but incorrect approach
        3. Explain why the correct approach is better

        Format each step as:
        Step X:
        Correct Approach:
        [Explain the right way to think about this step]

        Incorrect Approach:
        [Explain a common or plausible mistake]

        Explanation:
        [Explain why the correct approach is better and what makes the incorrect approach problematic]

        Confidence: [0-1]

        Provide 2-3 steps that build toward the solution.
        """

    def get_synthesis_prompt(self, task: str) -> str:
        """Create prompt for final synthesis."""
        steps_text = []
        for step in self.steps:
            steps_text.extend([
                f"Step {step.step_number}:",
                f"Correct: {step.correct_reasoning}",
                f"Incorrect: {step.incorrect_reasoning}",
                f"Why Correct is Better: {step.explanation}",
                "---"
            ])

        return f"""
        Based on the contrastive analysis:

        {'\n'.join(steps_text)}

        Provide a final answer to the original task:
        {task}

        Format your response as:
        Answer: [your final answer]
        Reasoning: [explain how the contrastive analysis led to this answer]
        Confidence: [0-1]
        """

    async def apply(self, agent: ChatAgent, task: str) -> TechniqueResult:
        """
        Apply Contrastive Chain of Thought reasoning to a task.
        
        Args:
            agent: ChatAgent instance to use for reasoning
            task: Task description or problem to solve
            
        Returns:
            TechniqueResult containing the reasoning process and result
        """
        # Clear previous state
        self.steps = []
        self.final_answer = None
        self.overall_confidence = 0.0

        # Get contrastive steps
        prompt = self.get_prompt(task)
        response = await agent.llm_response(prompt)
        steps = self._parse_steps(response.content)
        self.steps = steps

        # Generate final synthesis
        synthesis_prompt = self.get_synthesis_prompt(task)
        synthesis_response = await agent.llm_response(synthesis_prompt)
        synthesis = self._parse_synthesis(synthesis_response.content)

        self.final_answer = synthesis['answer']
        self.overall_confidence = synthesis['confidence']

        return self._create_result()

    def _parse_steps(self, response: str) -> List[ContrastiveStep]:
        """Parse contrastive reasoning steps from response."""
        steps = []
        current_step = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Step '):
                if current_step:
                    steps.append(current_step)
                current_step = ContrastiveStep(
                    step_number=len(steps) + 1,
                    correct_reasoning='',
                    incorrect_reasoning='',
                    explanation='',
                    confidence=0.0
                )
            elif line.startswith('Correct Approach:'):
                if current_step:
                    current_step.correct_reasoning = self._collect_multiline_text(response, line)
            elif line.startswith('Incorrect Approach:'):
                if current_step:
                    current_step.incorrect_reasoning = self._collect_multiline_text(response, line)
            elif line.startswith('Explanation:'):
                if current_step:
                    current_step.explanation = self._collect_multiline_text(response, line)
            elif line.startswith('Confidence:'):
                if current_step:
                    try:
                        current_step.confidence = float(line[len('Confidence:'):].strip())
                    except ValueError:
                        current_step.confidence = 0.5

        if current_step:
            steps.append(current_step)

        return steps

    def _collect_multiline_text(self, response: str, start_line: str) -> str:
        """Collect multi-line text from response."""
        lines = []
        started = False
        
        for line in response.split('\n'):
            if line.strip() == start_line.strip():
                started = True
                continue
            elif started:
                if line.strip() and not any(line.startswith(x) for x in ['Step ', 'Correct Approach:', 'Incorrect Approach:', 'Explanation:', 'Confidence:']):
                    lines.append(line.strip())
                else:
                    break
        
        return ' '.join(lines)

    def _parse_synthesis(self, response: str) -> Dict[str, Any]:
        """Parse synthesis response."""
        result = {
            'answer': '',
            'reasoning': '',
            'confidence': 0.0
        }

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Answer:'):
                result['answer'] = line[len('Answer:'):].strip()
            elif line.startswith('Reasoning:'):
                result['reasoning'] = line[len('Reasoning:'):].strip()
            elif line.startswith('Confidence:'):
                try:
                    result['confidence'] = float(line[len('Confidence:'):].strip())
                except ValueError:
                    result['confidence'] = 0.5

        return result

    def _create_result(self) -> TechniqueResult:
        """Create the final technique result."""
        thought_process = []
        for step in self.steps:
            thought_process.extend([
                f"Step {step.step_number}:",
                f"Correct Approach: {step.correct_reasoning}",
                f"Incorrect Approach: {step.incorrect_reasoning}",
                f"Why Correct is Better: {step.explanation}",
                f"Confidence: {step.confidence}",
                "---"
            ])

        return TechniqueResult(
            thought='\n'.join(thought_process),
            result=self.final_answer,
            confidence=self.overall_confidence,
            metadata={
                'steps': [
                    {
                        'step_number': step.step_number,
                        'correct_reasoning': step.correct_reasoning,
                        'incorrect_reasoning': step.incorrect_reasoning,
                        'explanation': step.explanation,
                        'confidence': step.confidence
                    }
                    for step in self.steps
                ]
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the technique state to a dictionary."""
        return {
            'name': self.name,
            'thought': self.thought,
            'steps': [
                {
                    'step_number': step.step_number,
                    'correct_reasoning': step.correct_reasoning,
                    'incorrect_reasoning': step.incorrect_reasoning,
                    'explanation': step.explanation,
                    'confidence': step.confidence
                }
                for step in self.steps
            ],
            'final_answer': self.final_answer,
            'overall_confidence': self.overall_confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContrastiveChainTechnique':
        """Create a technique instance from a dictionary."""
        instance = cls()
        instance.name = data['name']
        instance.thought = data['thought']
        instance.steps = [
            ContrastiveStep(
                step_number=step['step_number'],
                correct_reasoning=step['correct_reasoning'],
                incorrect_reasoning=step['incorrect_reasoning'],
                explanation=step['explanation'],
                confidence=step['confidence']
            )
            for step in data['steps']
        ]
        instance.final_answer = data['final_answer']
        instance.overall_confidence = data['overall_confidence']
        return instance

# Register the technique
from .registry import TechniqueRegistry
TechniqueRegistry.register(ContrastiveChainTechnique())
