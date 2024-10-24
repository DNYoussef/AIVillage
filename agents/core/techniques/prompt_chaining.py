"""Prompt Chaining reasoning technique implementation."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json
from langroid.agent.chat_agent import ChatAgent

from .base import AgentTechnique, TechniqueResult

@dataclass
class ChainLink:
    """A link in the prompt chain."""
    name: str
    prompt: str
    response: str
    output: Any
    confidence: float
    depends_on: List[str]  # Names of links this depends on

class PromptChainingTechnique(AgentTechnique):
    """
    Implementation of Prompt Chaining reasoning technique.
    
    This technique breaks down complex tasks into a chain of prompts,
    where each prompt:
    - Builds on previous responses
    - Handles a specific sub-task
    - Produces structured output
    - Can be validated independently
    """
    
    def __init__(self):
        super().__init__(
            thought="Prompt Chaining uses multiple prompts in succession to handle complex "
                   "multi-step tasks, allowing for a more structured approach to problem-solving.",
            name="Prompt Chaining",
            code=self.__class__.__module__
        )
        self.chain: List[ChainLink] = []
        self.final_output: Optional[Any] = None
        self.overall_confidence: float = 0.0
    
    def get_chain_definition_prompt(self, task: str) -> str:
        """Create prompt for defining the chain structure."""
        return f"""
        Design a chain of prompts to solve this task:

        Task: {task}

        Break down the task into 3-5 sequential steps, where each step:
        1. Has a clear, focused objective
        2. Produces output needed by later steps
        3. Can be validated independently

        Format each step as:
        Step Name: [short identifier]
        Objective: [what this step should accomplish]
        Dependencies: [list of step names this depends on, or "none"]
        Output Format: [expected structure of the output]
        Validation Criteria: [how to check if the output is valid]
        """

    def get_step_prompt(
        self,
        step_name: str,
        objective: str,
        context: Dict[str, Any]
    ) -> str:
        """Create prompt for a specific chain step."""
        context_text = []
        for name, value in context.items():
            context_text.extend([
                f"From {name}:",
                str(value),
                "---"
            ])

        return f"""
        Complete this step in the prompt chain:

        Step: {step_name}
        Objective: {objective}

        Previous Context:
        {'\n'.join(context_text)}

        Provide your response in the specified output format.
        Explain your reasoning and rate your confidence (0-1).

        Format your response as:
        Output: [your structured output]
        Reasoning: [explain how you arrived at this output]
        Confidence: [0-1]
        """

    async def apply(self, agent: ChatAgent, task: str) -> TechniqueResult:
        """
        Apply Prompt Chaining reasoning to a task.
        
        Args:
            agent: ChatAgent instance to use for reasoning
            task: Task description or problem to solve
            
        Returns:
            TechniqueResult containing the reasoning process and result
        """
        # Clear previous state
        self.chain = []
        self.final_output = None
        self.overall_confidence = 0.0

        # Get chain definition
        definition_prompt = self.get_chain_definition_prompt(task)
        definition_response = await agent.llm_response(definition_prompt)
        chain_definition = self._parse_chain_definition(definition_response.content)

        # Sort steps by dependencies
        sorted_steps = self._topological_sort(chain_definition)

        # Execute chain
        context = {}
        for step in sorted_steps:
            # Get required context
            step_context = {
                name: context[name]
                for name in step['dependencies']
                if name in context
            }

            # Execute step
            step_prompt = self.get_step_prompt(
                step['name'],
                step['objective'],
                step_context
            )
            step_response = await agent.llm_response(step_prompt)
            step_result = self._parse_step_result(step_response.content)

            # Validate output
            validation_result = await self._validate_step(
                agent,
                step,
                step_result,
                step_context
            )

            # Store result in context
            context[step['name']] = step_result['output']

            # Create chain link
            link = ChainLink(
                name=step['name'],
                prompt=step_prompt,
                response=step_response.content,
                output=step_result['output'],
                confidence=validation_result['confidence'],
                depends_on=step['dependencies']
            )
            self.chain.append(link)

        # Get final output from last step
        self.final_output = self.chain[-1].output
        self.overall_confidence = self._calculate_overall_confidence()

        return self._create_result()

    def _parse_chain_definition(self, response: str) -> List[Dict[str, Any]]:
        """Parse chain definition from response."""
        steps = []
        current_step = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Step Name:'):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    'name': line[len('Step Name:'):].strip(),
                    'objective': '',
                    'dependencies': [],
                    'output_format': '',
                    'validation_criteria': ''
                }
            elif current_step:
                if line.startswith('Objective:'):
                    current_step['objective'] = line[len('Objective:'):].strip()
                elif line.startswith('Dependencies:'):
                    deps = line[len('Dependencies:'):].strip()
                    if deps.lower() != 'none':
                        current_step['dependencies'] = [
                            d.strip() for d in deps.split(',')
                        ]
                elif line.startswith('Output Format:'):
                    current_step['output_format'] = line[len('Output Format:'):].strip()
                elif line.startswith('Validation Criteria:'):
                    current_step['validation_criteria'] = line[len('Validation Criteria:'):].strip()

        if current_step:
            steps.append(current_step)

        return steps

    def _parse_step_result(self, response: str) -> Dict[str, Any]:
        """Parse step execution result."""
        result = {
            'output': None,
            'reasoning': '',
            'confidence': 0.0
        }

        current_section = None
        current_content = []

        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('Output:'):
                if current_section:
                    result[current_section] = '\n'.join(current_content)
                current_section = 'output'
                current_content = [line[len('Output:'):].strip()]
            elif line.startswith('Reasoning:'):
                if current_section:
                    result[current_section] = '\n'.join(current_content)
                current_section = 'reasoning'
                current_content = [line[len('Reasoning:'):].strip()]
            elif line.startswith('Confidence:'):
                if current_section:
                    result[current_section] = '\n'.join(current_content)
                try:
                    result['confidence'] = float(line[len('Confidence:'):].strip())
                except ValueError:
                    result['confidence'] = 0.5
            elif current_section:
                current_content.append(line)

        if current_section:
            result[current_section] = '\n'.join(current_content)

        return result

    async def _validate_step(
        self,
        agent: ChatAgent,
        step: Dict[str, Any],
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate step output against criteria."""
        validation_prompt = f"""
        Validate this step output against its criteria:

        Step: {step['name']}
        Objective: {step['objective']}
        Output Format: {step['output_format']}
        Validation Criteria: {step['validation_criteria']}

        Output to validate:
        {result['output']}

        Context used:
        {json.dumps(context, indent=2)}

        Assess whether the output:
        1. Matches the required format
        2. Fulfills the objective
        3. Meets the validation criteria

        Format your response as:
        Valid: [true/false]
        Issues: [list any problems found, or "none"]
        Confidence: [0-1]
        """

        validation_response = await agent.llm_response(validation_prompt)
        return self._parse_validation_result(validation_response.content)

    def _parse_validation_result(self, response: str) -> Dict[str, Any]:
        """Parse validation result."""
        result = {
            'valid': False,
            'issues': [],
            'confidence': 0.0
        }

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Valid:'):
                result['valid'] = line[len('Valid:'):].strip().lower() == 'true'
            elif line.startswith('Issues:'):
                issues = line[len('Issues:'):].strip()
                if issues.lower() != 'none':
                    result['issues'] = [i.strip() for i in issues.split(',')]
            elif line.startswith('Confidence:'):
                try:
                    result['confidence'] = float(line[len('Confidence:'):].strip())
                except ValueError:
                    result['confidence'] = 0.5

        return result

    def _topological_sort(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort steps based on dependencies."""
        # Create adjacency list
        graph = {step['name']: step['dependencies'] for step in steps}
        
        # Find nodes with no dependencies
        no_deps = [
            step for step in steps
            if not step['dependencies']
        ]
        
        sorted_steps = []
        while no_deps:
            # Add a node with no dependencies
            current = no_deps.pop(0)
            sorted_steps.append(current)
            
            # Remove current node from dependencies
            current_name = current['name']
            for step in steps:
                if current_name in step['dependencies']:
                    step['dependencies'].remove(current_name)
                    if not step['dependencies']:
                        no_deps.append(step)
        
        return sorted_steps

    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence from chain links."""
        if not self.chain:
            return 0.0

        # Use geometric mean to ensure that low confidence in any step
        # significantly impacts overall confidence
        product = 1.0
        for link in self.chain:
            product *= link.confidence

        return product ** (1.0 / len(self.chain))

    def _create_result(self) -> TechniqueResult:
        """Create the final technique result."""
        thought_process = []
        for link in self.chain:
            thought_process.extend([
                f"Step: {link.name}",
                f"Dependencies: {', '.join(link.depends_on) if link.depends_on else 'none'}",
                f"Output: {link.output}",
                f"Confidence: {link.confidence}",
                "---"
            ])

        return TechniqueResult(
            thought='\n'.join(thought_process),
            result=self.final_output,
            confidence=self.overall_confidence,
            metadata={
                'chain': [
                    {
                        'name': link.name,
                        'output': link.output,
                        'confidence': link.confidence,
                        'depends_on': link.depends_on
                    }
                    for link in self.chain
                ]
            }
        )

# Register the technique
from .registry import TechniqueRegistry
TechniqueRegistry.register(PromptChainingTechnique())
