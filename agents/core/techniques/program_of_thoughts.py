"""Program of Thoughts reasoning technique implementation."""

from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass
import asyncio
from langroid.agent.chat_agent import ChatAgent

from .base import AgentTechnique, TechniqueResult

@dataclass
class ProgramStep:
    """A step in the program-based reasoning process."""
    step_number: int
    description: str
    code: str
    output: Optional[str]
    explanation: str
    confidence: float

class ProgramOfThoughtsTechnique(AgentTechnique):
    """
    Implementation of Program of Thoughts reasoning technique.
    
    This technique uses code generation and execution as part of the reasoning process,
    breaking down problems into programmable components and using their outputs to
    guide the solution process.
    """
    
    def __init__(self):
        super().__init__(
            thought="Program-of-Thoughts generates programming code as reasoning steps, "
                   "excelling in mathematical and programming tasks by leveraging code-based reasoning.",
            name="Program-of-Thoughts",
            code=self.__class__.__module__
        )
        self.steps: List[ProgramStep] = []
        self.final_result: Optional[str] = None
        self.overall_confidence: float = 0.0
    
    def get_prompt(self, task: str) -> str:
        """Create the Program of Thoughts prompt."""
        return f"""
        Solve the following task by breaking it down into programmable steps:

        Task: {task}

        For each step:
        1. Describe what the code will do
        2. Write Python code to implement it
        3. Explain how the code helps solve the problem
        4. Rate your confidence in this step (0-1)

        Format each step as:
        Step X:
        Description: [what this step does]
        Code:
        ```python
        [your code here]
        ```
        Explanation: [how this helps solve the problem]
        Confidence: [0-1]

        Make sure each code snippet is self-contained and includes print statements
        to show intermediate results. The final step should produce the answer to
        the original task.
        """

    async def apply(self, agent: ChatAgent, task: str) -> TechniqueResult:
        """
        Apply Program of Thoughts reasoning to a task.
        
        Args:
            agent: ChatAgent instance to use for reasoning
            task: Task description or problem to solve
            
        Returns:
            TechniqueResult containing the reasoning process and result
        """
        # Clear previous state
        self.steps = []
        self.final_result = None
        self.overall_confidence = 0.0

        # Get initial program steps
        prompt = self.get_prompt(task)
        response = await agent.llm_response(prompt)
        steps = self._parse_steps(response.content)

        # Process each step
        for step in steps:
            # Execute code
            execution_prompt = f"""
            Execute this code and provide its output:
            
            ```python
            {step.code}
            ```
            
            Format your response as:
            Output: [code output or error message]
            """
            execution_response = await agent.llm_response(execution_prompt)
            step.output = self._parse_output(execution_response.content)

            self.steps.append(step)

        # Generate final answer
        final_answer = await self._generate_final_answer(agent, task)
        
        return final_answer

    def _parse_steps(self, response: str) -> List[ProgramStep]:
        """Parse program steps from model response."""
        steps = []
        current_step = None
        code_block = []
        in_code_block = False
        
        for line in response.split('\n'):
            line = line.strip()
            
            if line.startswith('Step '):
                if current_step and code_block:
                    current_step.code = '\n'.join(code_block)
                    steps.append(current_step)
                current_step = ProgramStep(
                    step_number=len(steps) + 1,
                    description='',
                    code='',
                    output=None,
                    explanation='',
                    confidence=0.0
                )
                code_block = []
                in_code_block = False
            elif line.startswith('Description: '):
                if current_step:
                    current_step.description = line[len('Description: '):].strip()
            elif line.startswith('```python'):
                in_code_block = True
            elif line.startswith('```') and in_code_block:
                in_code_block = False
            elif in_code_block:
                code_block.append(line)
            elif line.startswith('Explanation: '):
                if current_step:
                    current_step.explanation = line[len('Explanation: '):].strip()
            elif line.startswith('Confidence: '):
                if current_step:
                    try:
                        current_step.confidence = float(line[len('Confidence: '):].strip())
                    except ValueError:
                        current_step.confidence = 0.5
        
        if current_step and code_block:
            current_step.code = '\n'.join(code_block)
            steps.append(current_step)
        
        return steps

    def _parse_output(self, response: str) -> str:
        """Parse code execution output from model response."""
        for line in response.split('\n'):
            if line.startswith('Output: '):
                return line[len('Output: '):].strip()
        return ''

    async def _generate_final_answer(
        self,
        agent: ChatAgent,
        task: str
    ) -> TechniqueResult:
        """Generate final answer based on program execution results."""
        steps_summary = []
        for step in self.steps:
            steps_summary.append(
                f"Step {step.step_number}:\n"
                f"Description: {step.description}\n"
                f"Output: {step.output}\n"
                f"Explanation: {step.explanation}"
            )

        synthesis_prompt = f"""
        Based on these program execution steps:

        {'\n\n'.join(steps_summary)}

        Provide a final answer to the original task:
        {task}

        Format your response as:
        Reasoning: [explain how the program steps led to the answer]
        Answer: [final answer]
        Confidence: [0-1]
        """

        response = await agent.llm_response(synthesis_prompt)
        final_result = self._parse_final_result(response.content)

        # Calculate overall confidence as weighted average of step confidences
        total_confidence = 0.0
        total_weight = 0.0
        for i, step in enumerate(self.steps, 1):
            weight = i  # Later steps weighted more heavily
            total_confidence += step.confidence * weight
            total_weight += weight
        
        self.overall_confidence = (
            total_confidence / total_weight if total_weight > 0 else 0.0
        )

        return TechniqueResult(
            thought=final_result['reasoning'],
            result=final_result['answer'],
            confidence=final_result['confidence'],
            metadata={
                'steps': [
                    {
                        'step_number': step.step_number,
                        'description': step.description,
                        'code': step.code,
                        'output': step.output,
                        'explanation': step.explanation,
                        'confidence': step.confidence
                    }
                    for step in self.steps
                ]
            }
        )

    def _parse_final_result(self, response: str) -> Dict[str, Any]:
        """Parse final result from model response."""
        result = {
            'reasoning': '',
            'answer': '',
            'confidence': 0.0
        }
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Reasoning: '):
                result['reasoning'] = line[len('Reasoning: '):].strip()
            elif line.startswith('Answer: '):
                result['answer'] = line[len('Answer: '):].strip()
            elif line.startswith('Confidence: '):
                try:
                    result['confidence'] = float(line[len('Confidence: '):].strip())
                except ValueError:
                    result['confidence'] = 0.5
        
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert the technique state to a dictionary."""
        return {
            'name': self.name,
            'thought': self.thought,
            'steps': [
                {
                    'step_number': step.step_number,
                    'description': step.description,
                    'code': step.code,
                    'output': step.output,
                    'explanation': step.explanation,
                    'confidence': step.confidence
                }
                for step in self.steps
            ],
            'final_result': self.final_result,
            'overall_confidence': self.overall_confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgramOfThoughtsTechnique':
        """Create a technique instance from a dictionary."""
        instance = cls()
        instance.name = data['name']
        instance.thought = data['thought']
        instance.steps = [
            ProgramStep(
                step_number=step['step_number'],
                description=step['description'],
                code=step['code'],
                output=step['output'],
                explanation=step['explanation'],
                confidence=step['confidence']
            )
            for step in data['steps']
        ]
        instance.final_result = data['final_result']
        instance.overall_confidence = data['overall_confidence']
        return instance

# Register the technique
from .registry import TechniqueRegistry
TechniqueRegistry.register(ProgramOfThoughtsTechnique())
