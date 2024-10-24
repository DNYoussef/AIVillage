"""Chain of Thought reasoning technique implementation."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime

from .base import (
    BaseTechnique,
    TechniqueResult,
    TechniqueMetrics,
    ProbabilisticTechnique
)
from ..core.exceptions import ExecutionError
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ChainOfThoughtInput:
    """Input for Chain of Thought technique."""
    question: str
    context: Optional[str] = None
    max_steps: int = 5
    step_timeout: float = 30.0
    temperature: float = 0.7

@dataclass
class ChainOfThoughtStep:
    """Single step in the chain of thought."""
    step_number: int
    thought: str
    reasoning: str
    confidence: float
    intermediate_result: Optional[str] = None

@dataclass
class ChainOfThoughtOutput:
    """Output from Chain of Thought technique."""
    steps: List[ChainOfThoughtStep]
    final_answer: str
    confidence: float
    execution_time: float

class ChainOfThoughtTechnique(ProbabilisticTechnique[ChainOfThoughtInput, ChainOfThoughtOutput]):
    """Implementation of Chain of Thought reasoning."""
    
    def __init__(
        self,
        name: str = "chain_of_thought",
        description: str = "Chain of Thought reasoning technique",
        confidence_threshold: float = 0.7
    ):
        super().__init__(name, description, confidence_threshold)
        self.prompt_template = """
        Think through this step-by-step:
        
        Question: {question}
        {context_str}
        
        Let's approach this step by step:
        1) First, let's understand what we're being asked
        2) Then, break down the problem into smaller parts
        3) Solve each part systematically
        4) Finally, combine the parts to reach our conclusion
        
        Current step ({step_number}/{max_steps}):
        Previous steps: {previous_steps}
        
        What should we think about next?
        
        Provide your response in the following format:
        Thought: [Your current thought]
        Reasoning: [Your reasoning for this thought]
        Intermediate Result: [Any intermediate result, if applicable]
        Confidence: [Your confidence in this step, 0-1]
        """
    
    async def initialize(self) -> None:
        """Initialize the technique."""
        logger.info(f"Initializing {self.name} technique")
    
    async def validate_input(self, input_data: ChainOfThoughtInput) -> bool:
        """Validate input data."""
        if not input_data.question.strip():
            return False
        if input_data.max_steps < 1:
            return False
        if input_data.step_timeout <= 0:
            return False
        if not 0 <= input_data.temperature <= 1:
            return False
        return True
    
    async def validate_output(self, output_data: ChainOfThoughtOutput) -> bool:
        """Validate output data."""
        if not output_data.steps:
            return False
        if not output_data.final_answer.strip():
            return False
        if not 0 <= output_data.confidence <= 1:
            return False
        return True
    
    async def execute(self, input_data: ChainOfThoughtInput) -> TechniqueResult[ChainOfThoughtOutput]:
        """Execute the Chain of Thought technique."""
        start_time = datetime.now()
        steps: List[ChainOfThoughtStep] = []
        
        try:
            for step_num in range(1, input_data.max_steps + 1):
                # Format previous steps for context
                previous_steps = "\n".join([
                    f"Step {s.step_number}: {s.thought} ({s.confidence:.2f} confidence)"
                    for s in steps
                ])
                
                # Format context if provided
                context_str = f"\nContext: {input_data.context}" if input_data.context else ""
                
                # Generate prompt
                prompt = self.prompt_template.format(
                    question=input_data.question,
                    context_str=context_str,
                    step_number=step_num,
                    max_steps=input_data.max_steps,
                    previous_steps=previous_steps or "None"
                )
                
                # Get response with timeout
                try:
                    async with asyncio.timeout(input_data.step_timeout):
                        response = await self.llm.complete(
                            prompt,
                            temperature=input_data.temperature
                        )
                except asyncio.TimeoutError:
                    raise ExecutionError(f"Step {step_num} timed out")
                
                # Parse response
                step = self._parse_step_response(response.text, step_num)
                steps.append(step)
                
                # Check if we've reached a conclusion
                if step.confidence > self.confidence_threshold and step.intermediate_result:
                    break
            
            # Generate final answer
            final_answer = await self._generate_final_answer(
                input_data.question,
                steps,
                input_data.context
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(steps)
            
            output = ChainOfThoughtOutput(
                steps=steps,
                final_answer=final_answer,
                confidence=confidence,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            metrics = TechniqueMetrics(
                execution_time=output.execution_time,
                success=True,
                confidence=output.confidence,
                uncertainty=1 - output.confidence,
                timestamp=datetime.now()
            )
            
            return TechniqueResult(
                output=output,
                metrics=metrics,
                intermediate_steps=[step.__dict__ for step in steps],
                reasoning_trace=[step.reasoning for step in steps]
            )
            
        except Exception as e:
            logger.error(f"Error in Chain of Thought execution: {str(e)}")
            raise ExecutionError(f"Chain of Thought failed: {str(e)}")
    
    def _parse_step_response(self, response: str, step_num: int) -> ChainOfThoughtStep:
        """Parse the response for a single step."""
        lines = response.strip().split('\n')
        thought = ""
        reasoning = ""
        confidence = 0.0
        intermediate_result = None
        
        for line in lines:
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Reasoning:"):
                reasoning = line[10:].strip()
            elif line.startswith("Intermediate Result:"):
                intermediate_result = line[19:].strip()
            elif line.startswith("Confidence:"):
                try:
                    confidence = float(line[11:].strip())
                except ValueError:
                    confidence = 0.0
        
        return ChainOfThoughtStep(
            step_number=step_num,
            thought=thought,
            reasoning=reasoning,
            confidence=confidence,
            intermediate_result=intermediate_result
        )
    
    async def _generate_final_answer(
        self,
        question: str,
        steps: List[ChainOfThoughtStep],
        context: Optional[str]
    ) -> str:
        """Generate the final answer based on all steps."""
        prompt = f"""
        Based on the following chain of thought, provide a final answer to the question:
        
        Question: {question}
        {f'Context: {context}' if context else ''}
        
        Reasoning steps:
        {self._format_steps(steps)}
        
        Final Answer:
        """
        
        response = await self.llm.complete(prompt)
        return response.text.strip()
    
    def _format_steps(self, steps: List[ChainOfThoughtStep]) -> str:
        """Format steps for display."""
        return "\n".join([
            f"Step {step.step_number}:"
            f"\nThought: {step.thought}"
            f"\nReasoning: {step.reasoning}"
            f"\nIntermediate Result: {step.intermediate_result or 'None'}"
            f"\nConfidence: {step.confidence:.2f}\n"
            for step in steps
        ])
    
    def _calculate_confidence(self, steps: List[ChainOfThoughtStep]) -> float:
        """Calculate overall confidence based on individual step confidences."""
        if not steps:
            return 0.0
        
        # Weight later steps more heavily
        weights = [i/len(steps) for i in range(1, len(steps) + 1)]
        weighted_confidences = [step.confidence * weight for step, weight in zip(steps, weights)]
        
        return sum(weighted_confidences) / sum(weights)
    
    async def estimate_confidence(self, result: ChainOfThoughtOutput) -> float:
        """Estimate confidence in the result."""
        return result.confidence
    
    async def estimate_uncertainty(self, result: ChainOfThoughtOutput) -> float:
        """Estimate uncertainty in the result."""
        return 1 - result.confidence
