"""Memory of Thought reasoning technique implementation."""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import json
import numpy as np
from langroid.agent.chat_agent import ChatAgent

from .base import AgentTechnique, TechniqueResult

@dataclass
class MemoryEntry:
    """An entry in the memory bank."""
    id: str
    content: str
    embedding: np.ndarray
    tags: Set[str]
    usage_count: int = 0
    success_rate: float = 0.0

@dataclass
class ReasoningStep:
    """A step in the memory-based reasoning process."""
    step_number: int
    reasoning: str
    relevant_memories: List[str]  # IDs of relevant memories
    conclusion: str
    confidence: float

class MemoryOfThoughtTechnique(AgentTechnique):
    """
    Implementation of Memory of Thought reasoning technique.
    
    This technique maintains a memory bank of past experiences and solutions,
    using them to inform current reasoning. It:
    - Retrieves relevant memories for new problems
    - Learns from successful and unsuccessful attempts
    - Adapts reasoning based on past experiences
    - Builds a growing knowledge base over time
    """
    
    def __init__(self):
        super().__init__(
            thought="Memory-of-Thought uses unlabeled data to build Few-Shot Chain-of-Thought prompts, "
                   "improving performance on various reasoning tasks through dynamic example retrieval.",
            name="Memory-of-Thought",
            code=self.__class__.__module__
        )
        self.memory_bank: Dict[str, MemoryEntry] = {}
        self.steps: List[ReasoningStep] = []
        self.final_answer: Optional[str] = None
        self.overall_confidence: float = 0.0
    
    def get_prompt(self, task: str, relevant_memories: List[MemoryEntry]) -> str:
        """Create the Memory of Thought prompt."""
        memory_text = []
        for i, memory in enumerate(relevant_memories, 1):
            memory_text.extend([
                f"Memory {i}:",
                memory.content,
                "---"
            ])

        return f"""
        Consider this task and relevant past experiences:

        Task: {task}

        Relevant Past Experiences:
        {'\n'.join(memory_text)}

        Using these past experiences as guidance, solve the task step by step.
        For each step:
        1. Explain your reasoning
        2. Reference specific memories that inform this step
        3. State your conclusion
        4. Rate your confidence (0-1)

        Format each step as:
        Step X:
        Reasoning: [your reasoning]
        Relevant Memories: [list memory numbers that informed this step]
        Conclusion: [intermediate conclusion]
        Confidence: [0-1]
        """

    async def apply(self, agent: ChatAgent, task: str) -> TechniqueResult:
        """
        Apply Memory of Thought reasoning to a task.
        
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

        # Get task embedding
        task_embedding = await self._get_embedding(agent, task)

        # Retrieve relevant memories
        relevant_memories = self._retrieve_relevant_memories(task_embedding)

        # Generate solution using memories
        prompt = self.get_prompt(task, relevant_memories)
        response = await agent.llm_response(prompt)
        steps = self._parse_steps(response.content)
        self.steps = steps

        # Generate final synthesis
        synthesis = await self._generate_synthesis(agent, task)
        self.final_answer = synthesis['answer']
        self.overall_confidence = synthesis['confidence']

        # Update memory bank with new experience
        await self._store_experience(agent, task, self.final_answer, self.overall_confidence)

        return self._create_result()

    async def _get_embedding(self, agent: ChatAgent, text: str) -> np.ndarray:
        """Get embedding for text using the agent's language model."""
        # This is a placeholder - in practice, you'd use a proper embedding model
        # For now, we'll create a random embedding
        return np.random.randn(768)  # Common embedding size

    def _retrieve_relevant_memories(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[MemoryEntry]:
        """Retrieve k most relevant memories using embedding similarity."""
        if not self.memory_bank:
            return []

        # Calculate similarities
        similarities = []
        for memory in self.memory_bank.values():
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            similarities.append((similarity, memory))

        # Sort by similarity and return top k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [memory for _, memory in similarities[:k]]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _parse_steps(self, response: str) -> List[ReasoningStep]:
        """Parse reasoning steps from response."""
        steps = []
        current_step = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Step '):
                if current_step:
                    steps.append(current_step)
                current_step = ReasoningStep(
                    step_number=len(steps) + 1,
                    reasoning='',
                    relevant_memories=[],
                    conclusion='',
                    confidence=0.0
                )
            elif line.startswith('Reasoning:'):
                if current_step:
                    current_step.reasoning = line[len('Reasoning:'):].strip()
            elif line.startswith('Relevant Memories:'):
                if current_step:
                    memory_refs = line[len('Relevant Memories:'):].strip()
                    current_step.relevant_memories = [
                        ref.strip() for ref in memory_refs.split(',')
                        if ref.strip()
                    ]
            elif line.startswith('Conclusion:'):
                if current_step:
                    current_step.conclusion = line[len('Conclusion:'):].strip()
            elif line.startswith('Confidence:'):
                if current_step:
                    try:
                        current_step.confidence = float(line[len('Confidence:'):].strip())
                    except ValueError:
                        current_step.confidence = 0.5

        if current_step:
            steps.append(current_step)

        return steps

    async def _generate_synthesis(
        self,
        agent: ChatAgent,
        task: str
    ) -> Dict[str, Any]:
        """Generate final synthesis from reasoning steps."""
        steps_text = []
        for step in self.steps:
            steps_text.extend([
                f"Step {step.step_number}:",
                f"Reasoning: {step.reasoning}",
                f"Conclusion: {step.conclusion}",
                "---"
            ])

        synthesis_prompt = f"""
        Based on these memory-guided reasoning steps:

        {'\n'.join(steps_text)}

        Provide a final answer to the original task:
        {task}

        Format your response as:
        Answer: [your final answer]
        Reasoning: [explain how the memories and reasoning led to this answer]
        Confidence: [0-1]
        """

        response = await agent.llm_response(synthesis_prompt)
        return self._parse_synthesis(response.content)

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

    async def _store_experience(
        self,
        agent: ChatAgent,
        task: str,
        solution: str,
        confidence: float
    ) -> None:
        """Store a new experience in the memory bank."""
        import uuid

        # Create memory entry
        memory_id = str(uuid.uuid4())
        memory_content = f"Task: {task}\nSolution: {solution}\nConfidence: {confidence}"
        memory_embedding = await self._get_embedding(agent, memory_content)

        # Extract tags (this could be more sophisticated)
        tags = set(word.lower() for word in task.split() if len(word) > 3)

        memory = MemoryEntry(
            id=memory_id,
            content=memory_content,
            embedding=memory_embedding,
            tags=tags,
            usage_count=0,
            success_rate=confidence
        )

        self.memory_bank[memory_id] = memory

    def _create_result(self) -> TechniqueResult:
        """Create the final technique result."""
        thought_process = []
        for step in self.steps:
            thought_process.extend([
                f"Step {step.step_number}:",
                f"Reasoning: {step.reasoning}",
                f"Relevant Memories: {', '.join(step.relevant_memories)}",
                f"Conclusion: {step.conclusion}",
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
                        'reasoning': step.reasoning,
                        'relevant_memories': step.relevant_memories,
                        'conclusion': step.conclusion,
                        'confidence': step.confidence
                    }
                    for step in self.steps
                ],
                'memory_bank_size': len(self.memory_bank)
            }
        )

# Register the technique
from .registry import TechniqueRegistry
TechniqueRegistry.register(MemoryOfThoughtTechnique())
