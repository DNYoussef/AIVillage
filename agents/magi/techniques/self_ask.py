"""Self-Ask reasoning technique implementation."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from langroid.agent.chat_agent import ChatAgent

from .base import AgentTechnique, TechniqueResult

@dataclass
class Question:
    """A question in the self-ask process."""
    number: int
    question: str
    answer: str
    reasoning: str
    confidence: float
    leads_to: Optional[int] = None  # Next question number, if any

class SelfAskTechnique(AgentTechnique):
    """
    Implementation of Self-Ask reasoning technique.
    
    This technique prompts the model to ask and answer its own follow-up questions,
    breaking down complex problems through self-questioning. It helps in:
    - Problem decomposition
    - Information gathering
    - Step-by-step reasoning
    - Identifying and filling knowledge gaps
    """
    
    def __init__(self):
        super().__init__(
            thought="Self-Ask prompts the model to ask and answer follow-up questions, "
                   "improving problem decomposition and solving through self-questioning.",
            name="Self-Ask",
            code=self.__class__.__module__
        )
        self.questions: List[Question] = []
        self.max_questions = 10  # Prevent infinite loops
        self.final_answer: Optional[str] = None
        self.overall_confidence: float = 0.0
    
    def get_prompt(self, task: str, context: Optional[str] = None) -> str:
        """Create the Self-Ask prompt."""
        if context:
            return f"""
            Consider this task and the current context:

            Task: {task}
            Context so far: {context}

            Ask yourself the most relevant follow-up question needed to solve this task.
            If you have enough information to provide a final answer, respond with "FINAL ANSWER" instead.

            Format your response as either:

            Question:
            [Your follow-up question]
            Reasoning: [Why you're asking this question]
            Confidence: [0-1]

            OR

            FINAL ANSWER:
            Answer: [Your final answer]
            Reasoning: [How you arrived at this answer]
            Confidence: [0-1]
            """
        else:
            return f"""
            Consider this task:

            Task: {task}

            Break down the problem by asking yourself the first question needed to solve it.

            Format your response as:
            Question:
            [Your first question]
            Reasoning: [Why you're asking this question]
            Confidence: [0-1]
            """

    def get_answer_prompt(self, question: str, context: str) -> str:
        """Create prompt for answering a self-asked question."""
        return f"""
        Answer this question based on the given context:

        Context: {context}
        Question: {question}

        Format your response as:
        Answer: [Your answer]
        Reasoning: [How you arrived at this answer]
        Confidence: [0-1]
        """

    async def apply(self, agent: ChatAgent, task: str) -> TechniqueResult:
        """
        Apply Self-Ask reasoning to a task.
        
        Args:
            agent: ChatAgent instance to use for reasoning
            task: Task description or problem to solve
            
        Returns:
            TechniqueResult containing the reasoning process and result
        """
        # Clear previous state
        self.questions = []
        self.final_answer = None
        self.overall_confidence = 0.0

        context = task
        question_count = 0

        while question_count < self.max_questions:
            # Get next question or final answer
            prompt = self.get_prompt(task, context)
            response = await agent.llm_response(prompt)
            parsed = self._parse_response(response.content)

            if parsed['type'] == 'final':
                self.final_answer = parsed['answer']
                self.overall_confidence = parsed['confidence']
                break

            # Store the question
            question = Question(
                number=question_count + 1,
                question=parsed['question'],
                answer='',
                reasoning=parsed['reasoning'],
                confidence=parsed['confidence']
            )

            # Get answer to the question
            answer_prompt = self.get_answer_prompt(question.question, context)
            answer_response = await agent.llm_response(answer_prompt)
            answer_parsed = self._parse_answer(answer_response.content)

            # Update question with answer
            question.answer = answer_parsed['answer']
            question.confidence = min(question.confidence, answer_parsed['confidence'])
            self.questions.append(question)

            # Update context with new information
            context = self._build_context(task)
            question_count += 1

        if not self.final_answer:
            # If we hit max questions without a final answer, synthesize one
            synthesis_prompt = f"""
            Based on all the questions and answers so far:

            {self._build_context(task)}

            Provide a final answer to the original task:
            {task}

            Format your response as:
            Answer: [Your final answer]
            Reasoning: [How you arrived at this answer]
            Confidence: [0-1]
            """
            synthesis_response = await agent.llm_response(synthesis_prompt)
            synthesis = self._parse_answer(synthesis_response.content)
            self.final_answer = synthesis['answer']
            self.overall_confidence = synthesis['confidence']

        return self._create_result()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse response to determine if it's a question or final answer."""
        if 'FINAL ANSWER:' in response:
            return self._parse_answer(response[response.index('FINAL ANSWER:'):])

        result = {
            'type': 'question',
            'question': '',
            'reasoning': '',
            'confidence': 0.0
        }

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Question:'):
                result['question'] = line[len('Question:'):].strip()
            elif line.startswith('Reasoning:'):
                result['reasoning'] = line[len('Reasoning:'):].strip()
            elif line.startswith('Confidence:'):
                try:
                    result['confidence'] = float(line[len('Confidence:'):].strip())
                except ValueError:
                    result['confidence'] = 0.5

        return result

    def _parse_answer(self, response: str) -> Dict[str, Any]:
        """Parse an answer response."""
        result = {
            'type': 'final',
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

    def _build_context(self, task: str) -> str:
        """Build context string from all questions and answers so far."""
        context = [f"Original task: {task}"]
        
        for q in self.questions:
            context.extend([
                f"Q{q.number}: {q.question}",
                f"Reasoning: {q.reasoning}",
                f"A{q.number}: {q.answer}"
            ])
        
        return '\n'.join(context)

    def _create_result(self) -> TechniqueResult:
        """Create the final technique result."""
        # Calculate overall confidence as weighted average
        if self.questions:
            total_confidence = sum(q.confidence for q in self.questions)
            avg_question_confidence = total_confidence / len(self.questions)
            self.overall_confidence = (avg_question_confidence + self.overall_confidence) / 2

        thought_process = []
        for q in self.questions:
            thought_process.extend([
                f"Question {q.number}: {q.question}",
                f"Reasoning: {q.reasoning}",
                f"Answer: {q.answer}",
                "---"
            ])

        return TechniqueResult(
            thought='\n'.join(thought_process),
            result=self.final_answer,
            confidence=self.overall_confidence,
            metadata={
                'questions': [
                    {
                        'number': q.number,
                        'question': q.question,
                        'answer': q.answer,
                        'reasoning': q.reasoning,
                        'confidence': q.confidence
                    }
                    for q in self.questions
                ]
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the technique state to a dictionary."""
        return {
            'name': self.name,
            'thought': self.thought,
            'questions': [
                {
                    'number': q.number,
                    'question': q.question,
                    'answer': q.answer,
                    'reasoning': q.reasoning,
                    'confidence': q.confidence,
                    'leads_to': q.leads_to
                }
                for q in self.questions
            ],
            'final_answer': self.final_answer,
            'overall_confidence': self.overall_confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelfAskTechnique':
        """Create a technique instance from a dictionary."""
        instance = cls()
        instance.name = data['name']
        instance.thought = data['thought']
        instance.questions = [
            Question(
                number=q['number'],
                question=q['question'],
                answer=q['answer'],
                reasoning=q['reasoning'],
                confidence=q['confidence'],
                leads_to=q.get('leads_to')
            )
            for q in data['questions']
        ]
        instance.final_answer = data['final_answer']
        instance.overall_confidence = data['overall_confidence']
        return instance

# Register the technique
from .registry import TechniqueRegistry
TechniqueRegistry.register(SelfAskTechnique())
