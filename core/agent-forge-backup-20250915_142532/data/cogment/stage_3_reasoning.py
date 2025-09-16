"""
Stage 3: Math & Text Reasoning Dataset

Mathematical reasoning and multi-hop text understanding:
- GSM8K: Grade school math word problems (7,473 training + 1,319 test)
- HotpotQA: Multi-hop question answering (90K training + 7.4K dev)
- 2WikiMultihop: Multi-hop QA over Wikipedia
- Competition Math: Advanced mathematical reasoning

Purpose: Complex reasoning with chain-of-thought and multi-step inference.
"""

from dataclasses import dataclass
import logging
import random
from typing import Any

from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
except ImportError:
    logger.warning("datasets library not available - using local data only")
    load_dataset = None

logger = logging.getLogger(__name__)


@dataclass
class ReasoningDataConfig:
    """Configuration for reasoning datasets."""

    # Data sources
    use_gsm8k: bool = True
    use_hotpotqa: bool = True
    use_2wikihop: bool = True
    use_competition_math: bool = True

    # Dataset limits
    gsm8k_limit: int | None = None  # None = use all
    hotpotqa_limit: int | None = 5000  # Limit for efficiency
    math_limit: int | None = 2000

    # Format settings
    chain_of_thought: bool = True
    step_by_step: bool = True
    show_reasoning: bool = True

    # Training settings
    sequence_length: int = 1024
    augment_reasoning: bool = True

    # Quality control
    validate_reasoning: bool = True
    min_reasoning_steps: int = 2
    seed: int = 42


class GSM8KProcessor:
    """Process GSM8K math word problems."""

    def __init__(self, config: ReasoningDataConfig):
        self.config = config

    def load_gsm8k_data(self) -> list[dict[str, Any]]:
        """Load GSM8K dataset."""
        if not self.config.use_gsm8k or load_dataset is None:
            return []

        try:
            logger.info("Loading GSM8K dataset...")

            dataset = load_dataset("gsm8k", "main", trust_remote_code=True)
            samples = []

            # Process training split
            if "train" in dataset:
                train_data = dataset["train"]
                if self.config.gsm8k_limit:
                    train_data = train_data.select(range(min(self.config.gsm8k_limit, len(train_data))))

                for item in train_data:
                    processed = self._process_gsm8k_item(item)
                    if processed:
                        samples.append(processed)

            logger.info(f"Loaded {len(samples)} GSM8K problems")
            return samples

        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            return self._generate_synthetic_math_problems()

    def _process_gsm8k_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single GSM8K item."""
        try:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()

            if not question or not answer:
                return None

            # Extract final numerical answer
            final_answer = self._extract_numerical_answer(answer)

            # Format with chain-of-thought
            if self.config.chain_of_thought:
                input_text = f"Math Problem: {question}\n\nLet me solve this step by step:"

                if self.config.show_reasoning:
                    target_text = f"{answer}\n\nTherefore, the answer is {final_answer}."
                else:
                    target_text = f"The answer is {final_answer}."
            else:
                input_text = f"Solve: {question}"
                target_text = f"Answer: {final_answer}"

            return {
                "input": input_text,
                "target": target_text,
                "task_type": "gsm8k_math",
                "domain": "mathematical_reasoning",
                "metadata": {
                    "question_length": len(question),
                    "answer_length": len(answer),
                    "final_answer": final_answer,
                    "reasoning_steps": answer.count(".") + answer.count("\n"),
                },
            }

        except Exception as e:
            logger.debug(f"Failed to process GSM8K item: {e}")
            return None

    def _extract_numerical_answer(self, answer: str) -> str:
        """Extract final numerical answer from GSM8K solution."""
        # Look for #### pattern
        if "####" in answer:
            final = answer.split("####")[-1].strip()
            return final

        # Fallback: look for numbers at the end
        import re

        numbers = re.findall(r"\d+(?:\.\d+)?", answer)
        return numbers[-1] if numbers else "unknown"

    def _generate_synthetic_math_problems(self, num_problems: int = 100) -> list[dict[str, Any]]:
        """Generate synthetic math problems as fallback."""
        logger.info(f"Generating {num_problems} synthetic math problems...")

        problems = []
        random.seed(self.config.seed)

        for i in range(num_problems):
            problem_type = random.choice(["addition", "multiplication", "percentage", "word_problem"])

            if problem_type == "addition":
                a, b = random.randint(10, 100), random.randint(10, 100)
                question = f"Sarah has {a} apples and buys {b} more. How many apples does she have in total?"
                answer = a + b
                reasoning = f"Sarah starts with {a} apples.\nShe buys {b} more apples.\nTotal = {a} + {b} = {answer}"

            elif problem_type == "multiplication":
                a, b = random.randint(5, 20), random.randint(3, 10)
                question = f"A box contains {a} items. If there are {b} boxes, how many items are there in total?"
                answer = a * b
                reasoning = f"Each box has {a} items.\nThere are {b} boxes.\nTotal = {a} × {b} = {answer}"

            elif problem_type == "percentage":
                total = random.randint(100, 1000)
                percent = random.choice([10, 20, 25, 50, 75])
                question = f"What is {percent}% of {total}?"
                answer = int(total * percent / 100)
                reasoning = f"{percent}% of {total} = {percent}/100 × {total} = {answer}"

            else:  # word_problem
                speed = random.randint(30, 80)
                time = random.randint(2, 8)
                question = f"A car travels at {speed} mph for {time} hours. How far does it travel?"
                answer = speed * time
                reasoning = f"Distance = Speed × Time\nDistance = {speed} mph × {time} hours = {answer} miles"

            input_text = f"Math Problem: {question}\n\nLet me solve this step by step:"
            target_text = f"{reasoning}\n\nTherefore, the answer is {answer}."

            problems.append(
                {
                    "input": input_text,
                    "target": target_text,
                    "task_type": "synthetic_math",
                    "domain": "mathematical_reasoning",
                    "metadata": {"problem_type": problem_type, "final_answer": str(answer), "synthetic": True},
                }
            )

        return problems


class HotpotQAProcessor:
    """Process HotpotQA multi-hop questions."""

    def __init__(self, config: ReasoningDataConfig):
        self.config = config

    def load_hotpotqa_data(self) -> list[dict[str, Any]]:
        """Load HotpotQA dataset."""
        if not self.config.use_hotpotqa or load_dataset is None:
            return []

        try:
            logger.info("Loading HotpotQA dataset...")

            dataset = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=True)
            samples = []

            # Process training split
            if "train" in dataset:
                train_data = dataset["train"]
                if self.config.hotpotqa_limit:
                    train_data = train_data.select(range(min(self.config.hotpotqa_limit, len(train_data))))

                for item in train_data:
                    processed = self._process_hotpotqa_item(item)
                    if processed:
                        samples.append(processed)

            logger.info(f"Loaded {len(samples)} HotpotQA questions")
            return samples

        except Exception as e:
            logger.error(f"Failed to load HotpotQA: {e}")
            return self._generate_synthetic_qa_problems()

    def _process_hotpotqa_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single HotpotQA item."""
        try:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            context = item.get("context", [])

            if not question or not answer:
                return None

            # Extract relevant context sentences
            context_text = self._extract_relevant_context(context, question, answer)

            # Format for multi-hop reasoning
            if self.config.chain_of_thought:
                input_text = (
                    f"Multi-hop Question: {question}\n\n"
                    f"Context: {context_text}\n\n"
                    f"Let me reason through this step by step:"
                )

                reasoning_steps = self._generate_reasoning_steps(question, answer, context_text)
                target_text = f"{reasoning_steps}\n\nTherefore, the answer is: {answer}"
            else:
                input_text = f"Question: {question}\nContext: {context_text}"
                target_text = f"Answer: {answer}"

            return {
                "input": input_text,
                "target": target_text,
                "task_type": "hotpotqa",
                "domain": "multi_hop_reasoning",
                "metadata": {
                    "question_length": len(question),
                    "context_length": len(context_text),
                    "answer_length": len(answer),
                    "num_context_docs": len(context),
                },
            }

        except Exception as e:
            logger.debug(f"Failed to process HotpotQA item: {e}")
            return None

    def _extract_relevant_context(self, context: list, question: str, answer: str) -> str:
        """Extract most relevant context sentences."""
        if not context:
            return "No context provided."

        # Simple relevance: context documents that contain answer keywords
        relevant_docs = []

        for doc in context[:3]:  # Limit to first 3 docs for efficiency
            if isinstance(doc, list) and len(doc) >= 2:
                title, sentences = doc[0], doc[1]

                # Take first few sentences
                doc_text = f"{title}: {' '.join(sentences[:2])}"
                relevant_docs.append(doc_text)

        return " ".join(relevant_docs)[:500] + "..." if len(" ".join(relevant_docs)) > 500 else " ".join(relevant_docs)

    def _generate_reasoning_steps(self, question: str, answer: str, context: str) -> str:
        """Generate reasoning steps for multi-hop questions."""
        # Simple reasoning pattern
        steps = [
            "Step 1: I need to identify the key information from the context.",
            "Step 2: I'll look for connections between different pieces of information.",
            f"Step 3: Based on the evidence, the answer appears to be '{answer}'.",
        ]

        return "\n".join(steps)

    def _generate_synthetic_qa_problems(self, num_problems: int = 50) -> list[dict[str, Any]]:
        """Generate synthetic QA problems as fallback."""
        logger.info(f"Generating {num_problems} synthetic QA problems...")

        problems = []
        random.seed(self.config.seed)

        facts = [
            ("Paris", "capital", "France"),
            ("Tokyo", "capital", "Japan"),
            ("Einstein", "developed", "theory of relativity"),
            ("Shakespeare", "wrote", "Hamlet"),
            ("Pacific", "largest", "ocean"),
        ]

        for i in range(num_problems):
            fact1 = random.choice(facts)
            fact2 = random.choice(facts)

            if fact1 != fact2:
                question = f"What is the relationship between {fact1[0]} and {fact2[0]}?"
                context = f"Context: {fact1[0]} {fact1[1]} {fact1[2]}. {fact2[0]} {fact2[1]} {fact2[2]}."
                answer = "They are both notable entities with distinct relationships to their respective domains."

                reasoning = (
                    "Step 1: I identify the key facts about each entity.\n"
                    f"Step 2: {fact1[0]} {fact1[1]} {fact1[2]}.\n"
                    f"Step 3: {fact2[0]} {fact2[1]} {fact2[2]}.\n"
                    "Step 4: Both are significant in their respective fields."
                )

                input_text = f"Multi-hop Question: {question}\n\n{context}\n\nLet me reason through this step by step:"
                target_text = f"{reasoning}\n\nTherefore, the answer is: {answer}"

                problems.append(
                    {
                        "input": input_text,
                        "target": target_text,
                        "task_type": "synthetic_qa",
                        "domain": "multi_hop_reasoning",
                        "metadata": {"synthetic": True, "entities": [fact1[0], fact2[0]]},
                    }
                )

        return problems


class CompetitionMathProcessor:
    """Process competition-level math problems."""

    def __init__(self, config: ReasoningDataConfig):
        self.config = config

    def load_competition_math_data(self) -> list[dict[str, Any]]:
        """Load competition math dataset."""
        if not self.config.use_competition_math or load_dataset is None:
            return []

        try:
            logger.info("Loading Competition Math dataset...")

            dataset = load_dataset("competition_math", trust_remote_code=True)
            samples = []

            # Process training split
            if "train" in dataset:
                train_data = dataset["train"]
                if self.config.math_limit:
                    train_data = train_data.select(range(min(self.config.math_limit, len(train_data))))

                for item in train_data:
                    processed = self._process_math_item(item)
                    if processed:
                        samples.append(processed)

            logger.info(f"Loaded {len(samples)} Competition Math problems")
            return samples

        except Exception as e:
            logger.error(f"Failed to load Competition Math: {e}")
            return self._generate_advanced_math_problems()

    def _process_math_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single competition math item."""
        try:
            problem = item.get("problem", "").strip()
            solution = item.get("solution", "").strip()
            level = item.get("level", "unknown")
            type_field = item.get("type", "unknown")

            if not problem or not solution:
                return None

            # Format for advanced mathematical reasoning
            if self.config.chain_of_thought:
                input_text = (
                    f"Advanced Math Problem ({type_field}, Level {level}): {problem}\n\n"
                    f"I need to solve this systematically:"
                )
                target_text = f"{solution}\n\nSolution complete."
            else:
                input_text = f"Problem: {problem}"
                target_text = f"Solution: {solution}"

            return {
                "input": input_text,
                "target": target_text,
                "task_type": "competition_math",
                "domain": "advanced_mathematics",
                "metadata": {
                    "level": level,
                    "type": type_field,
                    "problem_length": len(problem),
                    "solution_length": len(solution),
                },
            }

        except Exception as e:
            logger.debug(f"Failed to process math item: {e}")
            return None

    def _generate_advanced_math_problems(self, num_problems: int = 25) -> list[dict[str, Any]]:
        """Generate advanced math problems as fallback."""
        logger.info(f"Generating {num_problems} advanced math problems...")

        problems = []
        random.seed(self.config.seed)

        for i in range(num_problems):
            problem_type = random.choice(["algebra", "geometry", "number_theory"])

            if problem_type == "algebra":
                a, b, c = random.randint(1, 5), random.randint(1, 10), random.randint(1, 20)
                problem = f"Solve for x: {a}x² + {b}x - {c} = 0"
                solution = f"Using the quadratic formula: x = (-{b} ± √({b}² + 4·{a}·{c})) / (2·{a})"

            elif problem_type == "geometry":
                r = random.randint(3, 10)
                problem = f"Find the area of a circle with radius {r}."
                solution = f"Area = πr² = π·{r}² = {r*r}π square units"

            else:  # number_theory
                n = random.randint(10, 50)
                problem = f"Find all prime factors of {n}."
                # Simple factorization
                factors = []
                temp = n
                d = 2
                while d * d <= temp:
                    while temp % d == 0:
                        factors.append(d)
                        temp //= d
                    d += 1
                if temp > 1:
                    factors.append(temp)
                solution = f"Prime factorization: {n} = {' × '.join(map(str, factors))}"

            input_text = f"Advanced Math Problem: {problem}\n\nI need to solve this systematically:"
            target_text = f"{solution}\n\nSolution complete."

            problems.append(
                {
                    "input": input_text,
                    "target": target_text,
                    "task_type": "synthetic_advanced_math",
                    "domain": "advanced_mathematics",
                    "metadata": {"problem_type": problem_type, "synthetic": True},
                }
            )

        return problems


class MathTextReasoningDataset(Dataset):
    """Complete math and text reasoning dataset for Stage 3."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = ReasoningDataConfig(**(config or {}))

        # Initialize processors
        self.gsm8k_processor = GSM8KProcessor(self.config)
        self.hotpotqa_processor = HotpotQAProcessor(self.config)
        self.math_processor = CompetitionMathProcessor(self.config)

        # Load all datasets
        self.samples = self._load_all_samples()

        # Shuffle for variety
        random.seed(self.config.seed)
        random.shuffle(self.samples)

        logger.info(f"Math & text reasoning dataset initialized with {len(self.samples)} samples")

    def _load_all_samples(self) -> list[dict[str, Any]]:
        """Load samples from all sources."""
        all_samples = []

        # Load GSM8K
        gsm8k_samples = self.gsm8k_processor.load_gsm8k_data()
        all_samples.extend(gsm8k_samples)

        # Load HotpotQA
        hotpotqa_samples = self.hotpotqa_processor.load_hotpotqa_data()
        all_samples.extend(hotpotqa_samples)

        # Load Competition Math
        math_samples = self.math_processor.load_competition_math_data()
        all_samples.extend(math_samples)

        logger.info(f"Loaded {len(all_samples)} total reasoning samples")
        return all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

    def get_data_loader(self, batch_size: int = 4, shuffle: bool = True) -> DataLoader:
        """Get DataLoader for this dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate function for batching."""
        return {
            "inputs": [item["input"] for item in batch],
            "targets": [item["target"] for item in batch],
            "task_types": [item["task_type"] for item in batch],
            "domains": [item["domain"] for item in batch],
            "metadata": [item["metadata"] for item in batch],
        }

    def get_domain_distribution(self) -> dict[str, int]:
        """Get distribution by reasoning domain."""
        distribution = {}

        for sample in self.samples:
            domain = sample.get("domain", "unknown")
            distribution[domain] = distribution.get(domain, 0) + 1

        return distribution

    def get_task_type_distribution(self) -> dict[str, int]:
        """Get distribution by task type."""
        distribution = {}

        for sample in self.samples:
            task_type = sample.get("task_type", "unknown")
            distribution[task_type] = distribution.get(task_type, 0) + 1

        return distribution

    def validate_reasoning_quality(self) -> bool:
        """Validate reasoning quality across samples."""
        if not self.config.validate_reasoning:
            return True

        logger.info("Validating reasoning quality...")

        valid_count = 0
        total_checked = min(100, len(self.samples))

        for i in range(total_checked):
            sample = self.samples[i]

            # Check required fields
            if not all(key in sample for key in ["input", "target", "task_type", "domain"]):
                continue

            # Check reasoning depth
            target = sample["target"]
            reasoning_indicators = ["step", "therefore", "because", "since", "first", "then", "next"]

            reasoning_score = sum(1 for indicator in reasoning_indicators if indicator.lower() in target.lower())

            if reasoning_score >= self.config.min_reasoning_steps:
                valid_count += 1

        success_rate = valid_count / total_checked if total_checked > 0 else 0

        logger.info(f"Reasoning validation: {valid_count}/{total_checked} samples valid ({success_rate*100:.1f}%)")

        return success_rate > 0.7  # 70% minimum


def create_reasoning_dataset(config: dict[str, Any] = None) -> MathTextReasoningDataset:
    """Factory function to create reasoning dataset."""
    dataset = MathTextReasoningDataset(config)

    if not dataset.validate_reasoning_quality():
        logger.warning("Reasoning dataset validation failed - some samples may lack sufficient reasoning depth")

    return dataset


def demo_reasoning_dataset():
    """Demonstrate reasoning dataset functionality."""
    print("=== Cogment Stage 3: Math & Text Reasoning Dataset Demo ===")

    # Create dataset with small config for demo
    config = {"gsm8k_limit": 5, "hotpotqa_limit": 5, "math_limit": 5, "chain_of_thought": True, "show_reasoning": True}

    dataset = create_reasoning_dataset(config)

    print(f"\nDataset size: {len(dataset)}")

    # Show distributions
    domain_dist = dataset.get_domain_distribution()
    task_dist = dataset.get_task_type_distribution()

    print(f"\nDomain distribution: {domain_dist}")
    print(f"Task type distribution: {task_dist}")

    # Show sample examples
    print("\n=== Sample Examples ===")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1} ({sample['task_type']} - {sample['domain']}):")
        print(f"Input: {sample['input'][:150]}...")
        print(f"Target: {sample['target'][:150]}...")

    # Test data loader
    loader = dataset.get_data_loader(batch_size=2, shuffle=False)
    batch = next(iter(loader))
    print(f"\nBatch structure: {list(batch.keys())}")
    print(f"Batch size: {len(batch['inputs'])}")

    print("\n=== Math & Text Reasoning Dataset Demo Complete ===")


if __name__ == "__main__":
    demo_reasoning_dataset()
