"""
Teacher prompt system for Quiet-STaR distillation.
Generates training pairs (reflection, answer) for supervised learning.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import QuietSTaRConfig


@dataclass
class ReflectionPrompt:
    """A reflection prompt template for generating internal thoughts."""

    template: str
    context_type: str = "general"  # general, reasoning, coding, creative
    reflection_style: str = "step_by_step"  # step_by_step, critical, exploratory, analytical
    max_reflection_tokens: int = 128
    metadata: dict[str, Any] = field(default_factory=dict)

    def format(self, question: str, context: str = "") -> str:
        """Format the template with the question and context."""
        return self.template.format(question=question, context=context, start_token="<SoT>", end_token="</SoT>")


@dataclass
class TrainingPair:
    """A training pair containing reflection and answer."""

    question: str
    reflection: str  # Hidden from end users
    answer: str  # Visible to end users
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_training_text(self) -> str:
        """Convert to training format with thought tokens."""
        return f"{self.question} <SoT>{self.reflection}</SoT> {self.answer}"

    def to_inference_text(self) -> str:
        """Convert to inference format without thought tokens."""
        return f"{self.question} {self.answer}"


class TeacherPromptGenerator:
    """
    Generates teacher prompts for Quiet-STaR distillation.
    Creates training pairs where reflections are hidden from end users.
    """

    def __init__(
        self,
        config: QuietSTaRConfig,
        model_name: str = "microsoft/DialoGPT-small",
        reflection_prompts_path: Path | None = None,
    ):
        self.config = config
        self.model_name = model_name
        self.reflection_prompts = self._load_reflection_prompts(reflection_prompts_path)

        # Initialize model and tokenizer for generation
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if not self._model_loaded:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Add special tokens if not present
            special_tokens = [
                self.config.start_of_thought_token,
                self.config.end_of_thought_token,
                self.config.no_thought_token,
            ]
            added_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
            if added_tokens > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))

            self._model_loaded = True

    def _load_reflection_prompts(self, prompts_path: Path | None) -> list[ReflectionPrompt]:
        """Load reflection prompt templates."""
        default_prompts = [
            ReflectionPrompt(
                template="{question} {start_token}Let me think step by step about this problem. First, I need to understand what's being asked. Then I'll break down the solution approach and work through each step carefully.{end_token}",
                context_type="reasoning",
                reflection_style="step_by_step",
            ),
            ReflectionPrompt(
                template="{question} {start_token}I should critically evaluate this question. What assumptions might be hidden? Are there edge cases I need to consider? Let me examine this from multiple angles.{end_token}",
                context_type="general",
                reflection_style="critical",
            ),
            ReflectionPrompt(
                template="{question} {start_token}This requires careful analysis. Let me identify the key components, consider the relationships between them, and systematically work toward a solution.{end_token}",
                context_type="analytical",
                reflection_style="analytical",
            ),
            ReflectionPrompt(
                template="{question} {start_token}Let me explore different approaches to this problem. What are the possible solutions? Which approach would be most effective? I'll consider the trade-offs.{end_token}",
                context_type="creative",
                reflection_style="exploratory",
            ),
            ReflectionPrompt(
                template="{question} {start_token}For this coding problem, I need to think about the algorithm, data structures, edge cases, and implementation details. Let me plan this out systematically.{end_token}",
                context_type="coding",
                reflection_style="step_by_step",
            ),
        ]

        if prompts_path and prompts_path.exists():
            try:
                with open(prompts_path, encoding="utf-8") as f:
                    prompts_data = json.load(f)
                return [ReflectionPrompt(**prompt) for prompt in prompts_data]
            except Exception as e:
                print(f"Warning: Could not load prompts from {prompts_path}: {e}")
                return default_prompts

        return default_prompts

    def generate_reflection_for_question(
        self,
        question: str,
        context: str = "",
        style: str = "step_by_step",
        max_tokens: int = 128,
    ) -> str:
        """Generate a reflection for a given question."""
        # Select appropriate prompt template
        suitable_prompts = [p for p in self.reflection_prompts if p.reflection_style == style]

        if not suitable_prompts:
            suitable_prompts = self.reflection_prompts

        random.choice(suitable_prompts)

        # Generate reflection content
        reflection_starters = [
            "Let me think about this carefully.",
            "I need to analyze this step by step.",
            "This requires some thought.",
            "Let me break this down.",
            "I should consider multiple approaches.",
            "Let me work through this systematically.",
            "This needs careful consideration.",
            "I'll think through this methodically.",
        ]

        # Create a structured reflection
        reflection_parts = [
            random.choice(reflection_starters),
            self._generate_problem_analysis(question),
            self._generate_approach_consideration(question),
            self._generate_solution_steps(question),
        ]

        reflection = " ".join(part for part in reflection_parts if part)

        # Truncate if too long
        if len(reflection.split()) > max_tokens:
            words = reflection.split()[:max_tokens]
            reflection = " ".join(words) + "..."

        return reflection

    def _generate_problem_analysis(self, question: str) -> str:
        """Generate problem analysis component of reflection."""
        analysis_templates = [
            "The question is asking about {topic}.",
            "I need to understand {topic} to answer this.",
            "This involves {topic} considerations.",
            "The key aspect here is {topic}.",
        ]

        # Simple topic extraction (in practice, could use more sophisticated NLP)
        topic = "this problem"
        if "how" in question.lower():
            topic = "the process or method"
        elif "why" in question.lower():
            topic = "the reasoning or cause"
        elif "what" in question.lower():
            topic = "the definition or identification"
        elif any(word in question.lower() for word in ["code", "program", "function"]):
            topic = "programming implementation"
        elif any(word in question.lower() for word in ["math", "calculate", "solve"]):
            topic = "mathematical computation"

        return random.choice(analysis_templates).format(topic=topic)

    def _generate_approach_consideration(self, question: str) -> str:
        """Generate approach consideration component of reflection."""
        approaches = [
            "I should consider different approaches to ensure I find the best solution.",
            "Let me think about various ways to tackle this problem.",
            "There might be multiple valid approaches here.",
            "I need to evaluate the most effective method.",
            "Let me consider the trade-offs of different approaches.",
        ]
        return random.choice(approaches)

    def _generate_solution_steps(self, question: str) -> str:
        """Generate solution steps component of reflection."""
        step_templates = [
            "First, I'll identify the core requirements. Then, I'll develop a systematic approach.",
            "Step 1: Understand the problem. Step 2: Plan the solution. Step 3: Execute carefully.",
            "I'll start by gathering information, then analyze it, and finally provide a comprehensive answer.",
            "My approach will be to break this down into manageable parts and address each one.",
            "I'll work through this systematically to ensure accuracy and completeness.",
        ]
        return random.choice(step_templates)

    def generate_answer_from_reflection(self, question: str, reflection: str, max_tokens: int = 256) -> str:
        """Generate an answer based on the question and reflection."""
        self._load_model()

        # Create prompt for answer generation
        prompt = f"Question: {question}\nThinking: {reflection}\nAnswer:"

        # Tokenize and generate
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and extract answer
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            # Fallback: use generated text after prompt
            answer = self.tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True).strip()

        return answer

    def create_training_pair(
        self,
        question: str,
        context: str = "",
        style: str = "step_by_step",
        max_reflection_tokens: int = 128,
        max_answer_tokens: int = 256,
    ) -> TrainingPair:
        """Create a complete training pair with reflection and answer."""

        # Generate reflection
        reflection = self.generate_reflection_for_question(
            question=question,
            context=context,
            style=style,
            max_tokens=max_reflection_tokens,
        )

        # Generate answer based on reflection
        answer = self.generate_answer_from_reflection(
            question=question, reflection=reflection, max_tokens=max_answer_tokens
        )

        return TrainingPair(
            question=question,
            reflection=reflection,
            answer=answer,
            metadata={
                "style": style,
                "context": context,
                "reflection_tokens": len(reflection.split()),
                "answer_tokens": len(answer.split()),
            },
        )

    def generate_training_dataset(
        self,
        questions: list[str],
        output_path: Path,
        styles: list[str] | None = None,
        pairs_per_question: int = 1,
    ) -> list[TrainingPair]:
        """Generate a complete training dataset."""

        if styles is None:
            styles = ["step_by_step", "critical", "exploratory", "analytical"]

        training_pairs = []

        for question in questions:
            for _ in range(pairs_per_question):
                style = random.choice(styles)

                try:
                    pair = self.create_training_pair(
                        question=question,
                        style=style,
                        max_reflection_tokens=self.config.max_thought_tokens,
                        max_answer_tokens=256,
                    )
                    training_pairs.append(pair)
                except Exception as e:
                    print(f"Warning: Failed to generate pair for question '{question[:50]}...': {e}")
                    continue

        # Save dataset
        dataset = [
            {
                "question": pair.question,
                "reflection": pair.reflection,
                "answer": pair.answer,
                "training_text": pair.to_training_text(),
                "inference_text": pair.to_inference_text(),
                "metadata": pair.metadata,
            }
            for pair in training_pairs
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"Generated {len(training_pairs)} training pairs saved to {output_path}")
        return training_pairs

    def validate_training_pairs(self, training_pairs: list[TrainingPair]) -> dict[str, Any]:
        """Validate the quality of generated training pairs."""

        validation_results = {
            "total_pairs": len(training_pairs),
            "avg_reflection_length": 0,
            "avg_answer_length": 0,
            "reflection_token_distribution": {},
            "style_distribution": {},
            "quality_issues": [],
        }

        if not training_pairs:
            return validation_results

        reflection_lengths = [len(pair.reflection.split()) for pair in training_pairs]
        answer_lengths = [len(pair.answer.split()) for pair in training_pairs]

        validation_results["avg_reflection_length"] = sum(reflection_lengths) / len(reflection_lengths)
        validation_results["avg_answer_length"] = sum(answer_lengths) / len(answer_lengths)

        # Token distribution
        for length in reflection_lengths:
            bucket = f"{(length // 20) * 20}-{(length // 20) * 20 + 19}"
            validation_results["reflection_token_distribution"][bucket] = (
                validation_results["reflection_token_distribution"].get(bucket, 0) + 1
            )

        # Style distribution
        for pair in training_pairs:
            style = pair.metadata.get("style", "unknown")
            validation_results["style_distribution"][style] = validation_results["style_distribution"].get(style, 0) + 1

        # Quality checks
        for i, pair in enumerate(training_pairs):
            if len(pair.reflection.strip()) < 10:
                validation_results["quality_issues"].append(f"Pair {i}: Reflection too short")

            if len(pair.answer.strip()) < 5:
                validation_results["quality_issues"].append(f"Pair {i}: Answer too short")

            if pair.question.lower() in pair.reflection.lower():
                validation_results["quality_issues"].append(f"Pair {i}: Reflection repeats question")

        return validation_results


def load_sample_questions() -> list[str]:
    """Load sample questions for testing the teacher system."""
    return [
        "What is the difference between machine learning and deep learning?",
        "How do you implement a binary search algorithm?",
        "Why is climate change considered a global challenge?",
        "What are the main principles of object-oriented programming?",
        "How does photosynthesis work in plants?",
        "What factors should I consider when choosing a programming language?",
        "Explain the concept of recursion with an example.",
        "What are the advantages and disadvantages of renewable energy?",
        "How do neural networks learn from data?",
        "What is the importance of database normalization?",
    ]


if __name__ == "__main__":
    # Demo usage
    from .config import get_training_config

    config = get_training_config()
    teacher = TeacherPromptGenerator(config)

    # Generate a few sample training pairs
    questions = load_sample_questions()[:3]  # Test with first 3 questions

    print("=== Teacher Prompt Generator Demo ===\n")

    for i, question in enumerate(questions, 1):
        print(f"Example {i}:")
        print(f"Question: {question}")

        pair = teacher.create_training_pair(question, style="step_by_step")

        print(f"Reflection (hidden): {pair.reflection}")
        print(f"Answer (visible): {pair.answer}")
        print(f"Training format: {pair.to_training_text()}")
        print(f"Inference format: {pair.to_inference_text()}")
        print("-" * 80)
        print()
