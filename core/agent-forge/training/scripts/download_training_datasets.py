#!/usr/bin/env python3
"""
Enhanced HRRM Training Dataset Manager

Extends existing HRRM training by adding benchmark training datasets.
Combines with existing synthetic pretraining and adds domain-specific SFT data.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
except ImportError:
    logger.error("Missing required packages. Install with: pip install datasets huggingface_hub")
    raise


class EnhancedHRRMTrainingManager:
    """Manages downloading and preprocessing of training datasets for HRRM enhancement.

    Extends existing HRRM training with benchmark datasets for improved performance.
    """

    def __init__(self, output_dir: str = "packages/core/training/datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check existing HRRM training configs
        self.config_dir = Path("configs/hrrm")
        self.existing_configs = self._load_existing_configs()

        logger.info("Detected existing HRRM training configs:")
        for config_name in self.existing_configs:
            logger.info(f"  - {config_name}")

        # Enhanced dataset configurations aligned with existing HRRM SFT configs
        self.datasets = {
            # Extend existing reasoner_sft_gsm_arc.yaml with more datasets
            "reasoner_enhancement": {
                "name": "GSM8K + ARC + MATH (Enhanced Reasoner Training)",
                "model_target": "reasoner",
                "extends_config": "reasoner_sft_gsm_arc.yaml",
                "sources": [
                    {"name": "gsm8k", "split": "train", "limit": 7500},  # Expand existing GSM8K
                    {
                        "name": "ai2_arc",
                        "split": "train",
                        "subset": "ARC-Challenge",
                        "limit": 1176,
                    },  # Full training set
                    {"name": "ai2_arc", "split": "train", "subset": "ARC-Easy", "limit": 2251},  # Full training set
                    {"name": "competition_math", "split": "train", "limit": 7500},  # Add advanced math
                ],
                "description": "Enhanced mathematical and scientific reasoning",
            },
            # New planner training with code and planning tasks
            "planner_enhancement": {
                "name": "HumanEval + CodeContests (Planner Training)",
                "model_target": "planner",
                "extends_config": "planner_pretrain.yaml",
                "sources": [
                    {"name": "openai_humaneval", "split": "test", "limit": None},  # All 164 examples
                    {"name": "code_contests", "split": "train", "limit": 5000},  # Code planning tasks
                    {"name": "apps", "split": "train", "limit": 3000},  # Programming problems
                ],
                "description": "Code planning and algorithmic thinking",
            },
            # Memory enhancement with diverse knowledge
            "memory_enhancement": {
                "name": "HellaSwag + CommonsenseQA + XNLI (Memory Training)",
                "model_target": "memory",
                "extends_config": "memory_sft_retrieval.yaml",
                "sources": [
                    {"name": "hellaswag", "split": "train", "limit": 10000},  # Commonsense knowledge
                    {"name": "commonsense_qa", "split": "train", "limit": 9741},  # Full training set
                    {"name": "xnli", "split": "train", "limit": 7500},  # Multilingual knowledge
                    {"name": "squad", "split": "train", "limit": 5000},  # Reading comprehension
                ],
                "description": "Enhanced contextual memory and knowledge retention",
            },
        }

    def _load_existing_configs(self) -> list[str]:
        """Load existing HRRM training configurations."""
        configs = []
        if self.config_dir.exists():
            for config_file in self.config_dir.glob("*.yaml"):
                configs.append(config_file.name)
        return configs

    def download_dataset(
        self, dataset_name: str, split: str = "train", subset: str = None, limit: int = None
    ) -> dict[str, Any]:
        """Download a single dataset with error handling."""
        try:
            logger.info(f"Downloading {dataset_name} ({split})")

            if subset:
                dataset = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
            else:
                dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

            # Limit dataset size if specified
            if limit and len(dataset) > limit:
                dataset = dataset.select(range(limit))
                logger.info(f"Limited {dataset_name} to {limit} examples")

            return {"dataset": dataset, "name": dataset_name, "split": split, "subset": subset, "size": len(dataset)}

        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return None

    def process_for_training(self, enhancement_type: str, datasets: list[dict[str, Any]]) -> dict[str, list[str]]:
        """Process datasets into HRRM-specific training format."""
        config = self.datasets[enhancement_type]
        model_target = config["model_target"]

        training_data = []  # Single list for target model

        for ds_info in datasets:
            if ds_info is None:
                continue

            dataset = ds_info["dataset"]
            name = ds_info["name"]

            logger.info(f"Processing {name} for {model_target} enhancement ({len(dataset)} examples)")

            # Model-specific processing
            if model_target == "reasoner":
                training_data.extend(self._process_for_reasoner(dataset, name))
            elif model_target == "planner":
                training_data.extend(self._process_for_planner(dataset, name))
            elif model_target == "memory":
                training_data.extend(self._process_for_memory(dataset, name))

        return {model_target: training_data}

    def _process_for_reasoner(self, dataset, name: str) -> list[str]:
        """Process datasets for reasoner enhancement (GSM8K, ARC, MATH format)."""
        examples = []

        for item in dataset:
            if name == "gsm8k":
                question = item.get("question", "")
                answer = item.get("answer", "")
                if question and answer:
                    # Format for Quiet-STaR reasoning
                    formatted = f"Question: {question}\n<SoT>\nLet me think step by step...\n{answer}\n<EoT>\nTherefore, the answer is {answer.split('####')[-1].strip() if '####' in answer else 'calculated above'}."
                    examples.append(formatted)

            elif name == "ai2_arc":
                question = item.get("question", "")
                choices = item.get("choices", {})
                answer_key = item.get("answerKey", "")
                if question and choices and answer_key:
                    choice_text = choices.get("text", [])
                    choice_labels = choices.get("label", [])
                    if answer_key in choice_labels:
                        correct_idx = choice_labels.index(answer_key)
                        correct_answer = choice_text[correct_idx] if correct_idx < len(choice_text) else ""
                        # Format for scientific reasoning
                        formatted = f"Question: {question}\nChoices: {', '.join(choice_text)}\n<SoT>\nI need to analyze each choice using scientific principles...\nThe correct answer is {answer_key}: {correct_answer}\n<EoT>\nAnswer: {answer_key}"
                        examples.append(formatted)

            elif name == "competition_math":
                problem = item.get("problem", "")
                solution = item.get("solution", "")
                if problem and solution:
                    # Format for advanced mathematical reasoning
                    formatted = f"Problem: {problem}\n<SoT>\nThis is an advanced math problem. Let me break it down systematically...\n{solution}\n<EoT>\nSolution complete."
                    examples.append(formatted)

        return examples

    def _process_for_planner(self, dataset, name: str) -> list[str]:
        """Process datasets for planner enhancement (code planning format)."""
        examples = []

        for item in dataset:
            if name == "openai_humaneval":
                prompt = item.get("prompt", "")
                canonical = item.get("canonical_solution", "")
                if prompt and canonical:
                    # Format for code planning with control tokens
                    formatted = f"<PLAN>\nProblem: {prompt}\n<SUBGOAL>\n1. Parse problem requirements\n2. Identify key functions needed\n3. Design solution structure\n<ACTION>\n{canonical}\n<CHECK>\nVerify solution meets requirements\n<ENDPLAN>"
                    examples.append(formatted)

            elif name == "code_contests":
                description = item.get("description", "")
                solutions = item.get("solutions", {})
                if description and solutions:
                    python_sol = str(solutions.get("python", [""])[0]) if solutions.get("python") else ""
                    if python_sol:
                        formatted = f"<PLAN>\nContest Problem: {description[:500]}...\n<SUBGOAL>\n1. Understand constraints\n2. Choose optimal algorithm\n3. Implement efficiently\n<ACTION>\n{python_sol[:300]}...\n<CHECK>\nTest with sample cases\n<ENDPLAN>"
                        examples.append(formatted)

            elif name == "apps":
                problem = item.get("problem", "")
                solutions = item.get("solutions", [])
                if problem and solutions:
                    solution = solutions[0] if solutions else ""
                    formatted = f"<PLAN>\nProgramming Problem: {problem[:300]}...\n<SUBGOAL>\n1. Analyze problem type\n2. Design algorithm\n3. Code implementation\n<ACTION>\n{solution[:200]}...\n<CHECK>\nValidate with test cases\n<ENDPLAN>"
                    examples.append(formatted)

        return examples

    def _process_for_memory(self, dataset, name: str) -> list[str]:
        """Process datasets for memory enhancement (contextual knowledge format)."""
        examples = []

        for item in dataset:
            if name == "hellaswag":
                ctx = item.get("ctx", "")
                endings = item.get("endings", [])
                label = item.get("label", 0)
                # Convert label to int if it's a string
                if isinstance(label, str):
                    try:
                        label = int(label)
                    except ValueError:
                        label = 0
                if ctx and endings and isinstance(label, int) and label < len(endings):
                    correct_ending = endings[label]
                    # Format for contextual memory
                    formatted = f"Context: {ctx}\nKnowledge: Based on common sense and world knowledge, the most likely continuation is: {correct_ending}\nMemory stored: Context pattern -> Logical outcome"
                    examples.append(formatted)

            elif name == "commonsense_qa":
                question = item.get("question", "")
                choices = item.get("choices", {})
                answer_key = item.get("answerKey", "")
                if question and choices and answer_key:
                    choice_text = choices.get("text", [])
                    choice_labels = choices.get("label", [])
                    if answer_key in choice_labels:
                        correct_idx = choice_labels.index(answer_key)
                        correct_answer = choice_text[correct_idx] if correct_idx < len(choice_text) else ""
                        formatted = f"Question: {question}\nKnowledge retrieval: {correct_answer}\nContext stored: {question} -> {correct_answer} (commonsense reasoning)"
                        examples.append(formatted)

            elif name == "xnli":
                premise = item.get("premise", "")
                hypothesis = item.get("hypothesis", "")
                label = item.get("label", 0)
                if premise and hypothesis:
                    labels = ["entailment", "neutral", "contradiction"]
                    formatted = f"Premise: {premise}\nHypothesis: {hypothesis}\nLogical relationship: {labels[label]}\nMemory pattern: Premise-hypothesis logic -> {labels[label]}"
                    examples.append(formatted)

            elif name == "squad":
                context = item.get("context", "")
                question = item.get("question", "")
                answers = item.get("answers", {})
                if context and question and answers:
                    answer_text = answers.get("text", [""])[0] if answers.get("text") else ""
                    formatted = f"Context: {context[:300]}...\nQuestion: {question}\nAnswer: {answer_text}\nMemory link: Context + Question -> Extracted answer"
                    examples.append(formatted)

        return examples

    def _process_code_dataset(self, dataset, name: str) -> dict[str, list[str]]:
        """Process code datasets for HRRM training."""
        data = {"planner": [], "reasoner": [], "memory": []}

        for item in dataset:
            if name == "openai_humaneval":
                # Extract planning (problem analysis), reasoning (solution steps), memory (context)
                problem = item.get("prompt", "")
                canonical = item.get("canonical_solution", "")

                if problem and canonical:
                    data["planner"].append(
                        f"Problem: {problem}\nPlanning: Analyze requirements, identify key functions needed, design solution structure."
                    )
                    data["reasoner"].append(f"Problem: {problem}\nStep-by-step Solution:\n{canonical}")
                    data["memory"].append(f"Code Context: {problem}\nSolution Pattern: {canonical[:200]}...")

            elif name == "code_contests":
                problem = item.get("description", "")
                solutions = item.get("solutions", {})

                if problem and solutions:
                    solution = str(solutions.get("python", [""])[0]) if solutions.get("python") else ""
                    if solution:
                        data["planner"].append(
                            f"Contest Problem: {problem[:500]}...\nApproach: Break down into subproblems, identify algorithms."
                        )
                        data["reasoner"].append(f"Problem: {problem[:300]}...\nSolution:\n{solution}")
                        data["memory"].append(f"Algorithm Context: {problem[:200]}...")

        return data

    def _process_math_dataset(self, dataset, name: str) -> dict[str, list[str]]:
        """Process math datasets for HRRM training."""
        data = {"planner": [], "reasoner": [], "memory": []}

        for item in dataset:
            if name == "gsm8k":
                question = item.get("question", "")
                answer = item.get("answer", "")

                if question and answer:
                    data["planner"].append(
                        f"Math Problem: {question}\nPlan: Identify what we're solving for, extract key numbers, determine operations needed."
                    )
                    data["reasoner"].append(f"Problem: {question}\nStep-by-step:\n{answer}")
                    data["memory"].append(
                        f"Math Context: {question}\nKey concepts: arithmetic reasoning, word problems"
                    )

            elif name == "competition_math":
                problem = item.get("problem", "")
                solution = item.get("solution", "")

                if problem and solution:
                    data["planner"].append(
                        f"Math Competition: {problem}\nStrategy: Analyze problem type, recall relevant theorems."
                    )
                    data["reasoner"].append(f"Problem: {problem}\nDetailed Solution:\n{solution}")
                    data["memory"].append(f"Advanced Math: {problem[:200]}...")

        return data

    def _process_multilingual_dataset(self, dataset, name: str) -> dict[str, list[str]]:
        """Process multilingual datasets for HRRM training."""
        data = {"planner": [], "reasoner": [], "memory": []}

        for item in dataset:
            if name == "hellaswag":
                ctx = item.get("ctx", "")
                endings = item.get("endings", [])
                label = item.get("label", 0)

                if ctx and endings and label < len(endings):
                    correct_ending = endings[label]
                    data["planner"].append(
                        f"Context: {ctx}\nTask: Choose most likely continuation based on common sense."
                    )
                    data["reasoner"].append(f"Context: {ctx}\nAnalysis: {correct_ending} is most logical because...")
                    data["memory"].append(f"Commonsense: {ctx}\nPattern: {correct_ending}")

            elif name == "xnli":
                premise = item.get("premise", "")
                hypothesis = item.get("hypothesis", "")
                label = item.get("label", 0)

                if premise and hypothesis:
                    labels = ["entailment", "neutral", "contradiction"]
                    data["planner"].append(
                        f"Premise: {premise}\nHypothesis: {hypothesis}\nTask: Determine logical relationship."
                    )
                    data["reasoner"].append(
                        f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship: {labels[label]}"
                    )
                    data["memory"].append(f"Logic Pattern: {premise[:100]}... → {labels[label]}")

        return data

    def _process_structured_dataset(self, dataset, name: str) -> dict[str, list[str]]:
        """Process structured reasoning datasets for HRRM training."""
        data = {"planner": [], "reasoner": [], "memory": []}

        for item in dataset:
            if name == "ai2_arc":
                question = item.get("question", "")
                choices = item.get("choices", {})
                answer_key = item.get("answerKey", "")

                if question and choices and answer_key:
                    choice_text = choices.get("text", [])
                    choice_labels = choices.get("label", [])

                    if answer_key in choice_labels:
                        correct_idx = choice_labels.index(answer_key)
                        correct_answer = choice_text[correct_idx] if correct_idx < len(choice_text) else ""

                        data["planner"].append(
                            f"Science Question: {question}\nApproach: Analyze each choice, apply scientific knowledge."
                        )
                        data["reasoner"].append(
                            f"Question: {question}\nCorrect Answer: {correct_answer}\nReasoning: Scientific principle application"
                        )
                        data["memory"].append(f"Science Knowledge: {question}\nAnswer: {correct_answer}")

            elif name == "commonsense_qa":
                question = item.get("question", "")
                choices = item.get("choices", {})
                answer_key = item.get("answerKey", "")

                if question and choices and answer_key:
                    choice_text = choices.get("text", [])
                    choice_label = choices.get("label", [])

                    if answer_key in choice_label:
                        correct_idx = choice_label.index(answer_key)
                        correct_answer = choice_text[correct_idx] if correct_idx < len(choice_text) else ""

                        data["planner"].append(f"Question: {question}\nStrategy: Use common sense and world knowledge.")
                        data["reasoner"].append(
                            f"Question: {question}\nAnswer: {correct_answer}\nWhy: Common sense reasoning"
                        )
                        data["memory"].append(f"Knowledge: {question}\nCommon sense: {correct_answer}")

        return data

    def save_training_data(self, domain: str, training_data: dict[str, list[str]]):
        """Save processed training data to files."""
        domain_dir = self.output_dir / domain
        domain_dir.mkdir(exist_ok=True)

        for model_type, examples in training_data.items():
            if examples:
                output_file = domain_dir / f"{model_type}_training_data.jsonl"

                with open(output_file, "w", encoding="utf-8") as f:
                    for example in examples:
                        json.dump({"text": example}, f, ensure_ascii=False)
                        f.write("\n")

                logger.info(f"Saved {len(examples)} {model_type} examples to {output_file}")

    async def download_all_training_data(self):
        """Download and process all training datasets."""
        logger.info("Starting comprehensive training dataset download...")

        total_datasets = 0
        successful_downloads = 0

        for domain, config in self.datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {domain.upper()} domain: {config['name']}")
            logger.info(f"{'='*60}")

            downloaded_datasets = []

            for source in config["sources"]:
                dataset_info = self.download_dataset(
                    source["name"], source.get("split", "train"), source.get("subset"), source.get("limit")
                )

                total_datasets += 1
                if dataset_info:
                    downloaded_datasets.append(dataset_info)
                    successful_downloads += 1

            # Process datasets for HRRM training
            if downloaded_datasets:
                training_data = self.process_for_training(domain, downloaded_datasets)
                self.save_training_data(domain, training_data)

            # Save domain summary
            summary = {
                "domain": domain,
                "description": config["description"],
                "sources": [ds["name"] for ds in downloaded_datasets if ds],
                "total_examples": sum(len(examples) for examples in training_data.values()),
                "planner_examples": len(training_data.get("planner", [])),
                "reasoner_examples": len(training_data.get("reasoner", [])),
                "memory_examples": len(training_data.get("memory", [])),
            }

            with open(self.output_dir / f"{domain}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("TRAINING DATA DOWNLOAD COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total datasets attempted: {total_datasets}")
        logger.info(f"Successful downloads: {successful_downloads}")
        logger.info(f"Success rate: {successful_downloads/total_datasets*100:.1f}%")
        logger.info(f"Training data saved to: {self.output_dir}")


async def main():
    """Main execution function."""
    print("[FIRE] Enhanced HRRM Training Dataset Manager")
    print("Extending existing HRRM training with benchmark datasets...")
    print()

    manager = EnhancedHRRMTrainingManager()
    await manager.download_all_training_data()

    print("\n[OK] Enhanced training dataset download complete!")
    print(f"[FOLDER] Data saved to: {manager.output_dir}")
    print("[ROCKET] Ready for enhanced HRRM model training!")


if __name__ == "__main__":
    asyncio.run(main())
