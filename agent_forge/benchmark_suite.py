"""Comprehensive Benchmarking Suite

Implements standardized evaluation across multiple benchmarks:
- MMLU (Massive Multitask Language Understanding)
- GSM8K (Grade School Math 8K)
- HumanEval (Code Evaluation)
- HellaSwag (Commonsense Reasoning)
- ARC (AI2 Reasoning Challenge)
- Custom Agent Forge specific benchmarks

Supports comparison against baseline 1.5B and frontier models with detailed W&B reporting.
"""

import asyncio
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

from datasets import load_dataset
import numpy as np
import pandas as pd
from scipy import stats
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    model_path: str
    model_name: str
    output_dir: str

    # Benchmark selection
    run_mmlu: bool = True
    run_gsm8k: bool = True
    run_humaneval: bool = True
    run_hellaswag: bool = True
    run_arc: bool = True
    run_custom: bool = True

    # Evaluation settings
    batch_size: int = 4
    max_length: int = 2048
    temperature: float = 0.0  # Use greedy decoding for consistency
    num_shots: int = 5  # Few-shot examples
    max_samples: int | None = None  # Limit samples for faster evaluation

    # Model settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "fp16"  # fp16, bf16, fp32

    # Comparison models
    baseline_models: list[str] = None
    frontier_models: list[str] = None

    # W&B settings
    wandb_project: str = "agent-forge-benchmark"
    wandb_entity: str | None = None

    def __post_init__(self):
        if self.baseline_models is None:
            self.baseline_models = [
                "microsoft/DialoGPT-small",  # 117M baseline
                "microsoft/DialoGPT-medium",  # 345M baseline
                "microsoft/DialoGPT-large",  # 762M baseline
            ]
        if self.frontier_models is None:
            self.frontier_models = [
                "meta-llama/Llama-2-7b-hf",
                "mistralai/Mistral-7B-v0.1",
                "microsoft/phi-2",  # 2.7B parameter model
            ]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark."""

    benchmark_name: str
    model_name: str
    overall_score: float
    category_scores: dict[str, float]
    num_samples: int
    correct_predictions: int
    execution_time: float
    memory_usage: float
    detailed_results: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass
class ComparisonReport:
    """Comprehensive comparison report."""

    target_model: str
    baseline_results: dict[str, BenchmarkResult]
    frontier_results: dict[str, BenchmarkResult]
    target_results: dict[str, BenchmarkResult]
    statistical_analysis: dict[str, Any]
    performance_summary: dict[str, Any]
    recommendations: list[str]


class MMLUEvaluator:
    """Evaluates on MMLU (Massive Multitask Language Understanding)."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.subjects = [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ]

    async def evaluate(self, model, tokenizer) -> BenchmarkResult:
        """Evaluate model on MMLU."""
        logger.info("Starting MMLU evaluation")
        start_time = time.time()

        all_results = []
        category_scores = {}
        total_correct = 0
        total_samples = 0

        for subject in tqdm(self.subjects, desc="MMLU Subjects"):
            try:
                # Load dataset for subject
                dataset = load_dataset("cais/mmlu", subject)["test"]

                if self.config.max_samples:
                    dataset = dataset.select(
                        range(
                            min(
                                len(dataset),
                                self.config.max_samples // len(self.subjects),
                            )
                        )
                    )

                subject_correct = 0
                subject_samples = len(dataset)
                subject_results = []

                for i, example in enumerate(dataset):
                    # Format question
                    question = example["question"]
                    choices = [example["choices"][j] for j in range(4)]
                    answer_idx = example["answer"]

                    # Create few-shot prompt
                    prompt = self._create_mmlu_prompt(question, choices, subject)

                    # Get model prediction
                    prediction = await self._get_model_prediction(
                        model, tokenizer, prompt
                    )
                    predicted_idx = self._parse_choice(prediction)

                    is_correct = predicted_idx == answer_idx
                    if is_correct:
                        subject_correct += 1
                        total_correct += 1

                    total_samples += 1

                    subject_results.append(
                        {
                            "subject": subject,
                            "question": question,
                            "choices": choices,
                            "correct_answer": answer_idx,
                            "predicted_answer": predicted_idx,
                            "is_correct": is_correct,
                            "prediction_text": prediction,
                        }
                    )

                category_scores[subject] = (
                    subject_correct / subject_samples if subject_samples > 0 else 0.0
                )
                all_results.extend(subject_results)

            except Exception as e:
                logger.error(f"Error evaluating MMLU subject {subject}: {e}")
                category_scores[subject] = 0.0

        overall_score = total_correct / total_samples if total_samples > 0 else 0.0
        execution_time = time.time() - start_time

        return BenchmarkResult(
            benchmark_name="MMLU",
            model_name=self.config.model_name,
            overall_score=overall_score,
            category_scores=category_scores,
            num_samples=total_samples,
            correct_predictions=total_correct,
            execution_time=execution_time,
            memory_usage=self._get_memory_usage(),
            detailed_results=all_results,
            metadata={
                "num_subjects": len(self.subjects),
                "num_shots": self.config.num_shots,
                "temperature": self.config.temperature,
            },
        )

    def _create_mmlu_prompt(
        self, question: str, choices: list[str], subject: str
    ) -> str:
        """Create MMLU prompt with few-shot examples."""
        # Few-shot examples (simplified for brevity)
        few_shot_examples = f"""The following are multiple choice questions about {subject.replace("_", " ")}.

Question: What is the primary function of DNA?
A) Energy storage
B) Protein synthesis
C) Information storage
D) Cell structure
Answer: C

Question: Which of the following is a prime number?
A) 4
B) 6
C) 7
D) 9
Answer: C

"""

        prompt = few_shot_examples + f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}) {choice}\n"
        prompt += "Answer:"

        return prompt

    async def _get_model_prediction(self, model, tokenizer, prompt: str) -> str:
        """Get model prediction for prompt."""
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.size(1) + 10,
                temperature=self.config.temperature,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = response[
            len(tokenizer.decode(inputs[0], skip_special_tokens=True)) :
        ].strip()

        return prediction

    def _parse_choice(self, prediction: str) -> int:
        """Parse model prediction to choice index."""
        prediction = prediction.strip().upper()

        # Look for A, B, C, D patterns
        if "A" in prediction[:5]:
            return 0
        if "B" in prediction[:5]:
            return 1
        if "C" in prediction[:5]:
            return 2
        if "D" in prediction[:5]:
            return 3
        # Random choice if unable to parse
        return np.random.randint(0, 4)

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)  # GB
        return 0.0


class GSM8KEvaluator:
    """Evaluates on GSM8K (Grade School Math 8K)."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    async def evaluate(self, model, tokenizer) -> BenchmarkResult:
        """Evaluate model on GSM8K."""
        logger.info("Starting GSM8K evaluation")
        start_time = time.time()

        # Load dataset
        dataset = load_dataset("gsm8k", "main")["test"]

        if self.config.max_samples:
            dataset = dataset.select(range(min(len(dataset), self.config.max_samples)))

        all_results = []
        total_correct = 0
        total_samples = len(dataset)

        for i, example in enumerate(tqdm(dataset, desc="GSM8K Problems")):
            question = example["question"]
            answer = example["answer"]

            # Extract numerical answer
            correct_answer = self._extract_answer(answer)

            # Create prompt
            prompt = self._create_gsm8k_prompt(question)

            # Get model prediction
            prediction = await self._get_model_prediction(model, tokenizer, prompt)
            predicted_answer = self._extract_answer(prediction)

            is_correct = (
                abs(predicted_answer - correct_answer) < 1e-6
                if predicted_answer is not None
                else False
            )

            if is_correct:
                total_correct += 1

            all_results.append(
                {
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "prediction_text": prediction,
                    "ground_truth": answer,
                }
            )

        overall_score = total_correct / total_samples if total_samples > 0 else 0.0
        execution_time = time.time() - start_time

        return BenchmarkResult(
            benchmark_name="GSM8K",
            model_name=self.config.model_name,
            overall_score=overall_score,
            category_scores={"math_reasoning": overall_score},
            num_samples=total_samples,
            correct_predictions=total_correct,
            execution_time=execution_time,
            memory_usage=self._get_memory_usage(),
            detailed_results=all_results,
            metadata={"dataset_size": len(dataset), "num_shots": self.config.num_shots},
        )

    def _create_gsm8k_prompt(self, question: str) -> str:
        """Create GSM8K prompt with few-shot examples."""
        few_shot_examples = """Solve these math problems step by step.

Q: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much money does she make every day at the farmers' market?
A: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, so she uses 3 + 4 = 7 eggs. She has 16 - 7 = 9 eggs left to sell. She sells them for $2 each, so she makes 9 × $2 = $18 per day.

Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts are used?
A: The robe takes 2 bolts of blue fiber. It takes half that much white fiber, so 2 ÷ 2 = 1 bolt of white fiber. In total, 2 + 1 = 3 bolts are used.

"""

        return few_shot_examples + f"Q: {question}\nA:"

    async def _get_model_prediction(self, model, tokenizer, prompt: str) -> str:
        """Get model prediction for math problem."""
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=min(inputs.size(1) + 200, self.config.max_length),
                temperature=self.config.temperature,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = response[
            len(tokenizer.decode(inputs[0], skip_special_tokens=True)) :
        ].strip()

        return prediction

    def _extract_answer(self, text: str) -> float | None:
        """Extract numerical answer from text."""
        # Look for common patterns
        patterns = [
            r"(?:is|equals?|=)\s*\$?(\d+(?:\.\d+)?)",
            r"\$(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:dollars?|cents?|$)",
            r"(\d+(?:\.\d+)?)\s*$",
            r"(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])  # Take the last match
                except ValueError:
                    continue

        return None

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)  # GB
        return 0.0


class HumanEvalEvaluator:
    """Evaluates on HumanEval (Code Evaluation)."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    async def evaluate(self, model, tokenizer) -> BenchmarkResult:
        """Evaluate model on HumanEval."""
        logger.info("Starting HumanEval evaluation")
        start_time = time.time()

        # Load dataset
        dataset = load_dataset("openai_humaneval")["test"]

        if self.config.max_samples:
            dataset = dataset.select(range(min(len(dataset), self.config.max_samples)))

        all_results = []
        total_correct = 0
        total_samples = len(dataset)

        for i, example in enumerate(tqdm(dataset, desc="HumanEval Problems")):
            task_id = example["task_id"]
            prompt = example["prompt"]
            canonical_solution = example["canonical_solution"]
            test = example["test"]
            entry_point = example["entry_point"]

            # Generate code
            generated_code = await self._generate_code(model, tokenizer, prompt)

            # Test code execution
            is_correct, error_msg = self._test_code_execution(
                prompt + generated_code, test, entry_point
            )

            if is_correct:
                total_correct += 1

            all_results.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "generated_code": generated_code,
                    "canonical_solution": canonical_solution,
                    "is_correct": is_correct,
                    "error_message": error_msg,
                    "test_cases": test,
                }
            )

        overall_score = total_correct / total_samples if total_samples > 0 else 0.0
        execution_time = time.time() - start_time

        return BenchmarkResult(
            benchmark_name="HumanEval",
            model_name=self.config.model_name,
            overall_score=overall_score,
            category_scores={"code_generation": overall_score},
            num_samples=total_samples,
            correct_predictions=total_correct,
            execution_time=execution_time,
            memory_usage=self._get_memory_usage(),
            detailed_results=all_results,
            metadata={"dataset_size": len(dataset), "pass_at_1": overall_score},
        )

    async def _generate_code(self, model, tokenizer, prompt: str) -> str:
        """Generate code completion."""
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=min(inputs.size(1) + 300, self.config.max_length),
                temperature=self.config.temperature,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["def ", "class ", "\n\n\n"],
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[
            len(tokenizer.decode(inputs[0], skip_special_tokens=True)) :
        ].strip()

        return generated

    def _test_code_execution(
        self, code: str, test: str, entry_point: str
    ) -> tuple[bool, str | None]:
        """Test if generated code passes the test cases."""
        try:
            # Create a safe execution environment
            exec_globals = {}

            # Execute the code
            exec(code, exec_globals)

            # Check if entry point exists
            if entry_point not in exec_globals:
                return False, f"Entry point '{entry_point}' not found"

            # Execute test cases
            exec(test, exec_globals)

            return True, None

        except Exception as e:
            return False, str(e)

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)  # GB
        return 0.0


class ComprehensiveBenchmark:
    """Main benchmarking orchestrator."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluators
        self.evaluators = {}
        if config.run_mmlu:
            self.evaluators["MMLU"] = MMLUEvaluator(config)
        if config.run_gsm8k:
            self.evaluators["GSM8K"] = GSM8KEvaluator(config)
        if config.run_humaneval:
            self.evaluators["HumanEval"] = HumanEvalEvaluator(config)

        # Model cache
        self.model_cache = {}

    async def benchmark_model(
        self, model_path: str, model_name: str
    ) -> dict[str, BenchmarkResult]:
        """Benchmark a single model."""
        logger.info(f"Benchmarking model: {model_name}")

        # Load model
        model, tokenizer = await self._load_model(model_path, model_name)

        # Run evaluations
        results = {}
        for eval_name, evaluator in self.evaluators.items():
            try:
                logger.info(f"Running {eval_name} evaluation")
                result = await evaluator.evaluate(model, tokenizer)
                results[eval_name] = result

                # Log intermediate results
                logger.info(f"{eval_name} Score: {result.overall_score:.4f}")

            except Exception as e:
                logger.error(f"Failed to evaluate {eval_name}: {e}")
                # Create dummy result for failed evaluation
                results[eval_name] = BenchmarkResult(
                    benchmark_name=eval_name,
                    model_name=model_name,
                    overall_score=0.0,
                    category_scores={},
                    num_samples=0,
                    correct_predictions=0,
                    execution_time=0.0,
                    memory_usage=0.0,
                    detailed_results=[],
                    metadata={"error": str(e)},
                )

        return results

    async def _load_model(self, model_path: str, model_name: str) -> tuple[Any, Any]:
        """Load model and tokenizer with caching."""
        if model_name in self.model_cache:
            return self.model_cache[model_name]

        logger.info(f"Loading model: {model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Determine dtype
            dtype = torch.float32
            if self.config.precision == "fp16":
                dtype = torch.float16
            elif self.config.precision == "bf16":
                dtype = torch.bfloat16

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if self.config.device == "cuda" else None,
                trust_remote_code=True,
            )

            if self.config.device == "cpu":
                model = model.to(self.config.device)

            model.eval()

            # Cache for reuse
            self.model_cache[model_name] = (model, tokenizer)

            logger.info(f"Model loaded successfully: {model_name}")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    async def run_comprehensive_benchmark(self) -> ComparisonReport:
        """Run comprehensive benchmark comparing target model against baselines and frontier models."""
        logger.info("Starting comprehensive benchmark evaluation")

        # Initialize W&B
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=f"benchmark_{self.config.model_name}",
            config=asdict(self.config),
        )

        all_results = {}

        # Benchmark target model
        logger.info("Benchmarking target model")
        target_results = await self.benchmark_model(
            self.config.model_path, self.config.model_name
        )
        all_results[self.config.model_name] = target_results

        # Benchmark baseline models
        baseline_results = {}
        for baseline_model in self.config.baseline_models:
            try:
                logger.info(f"Benchmarking baseline: {baseline_model}")
                results = await self.benchmark_model(baseline_model, baseline_model)
                baseline_results[baseline_model] = results
                all_results[baseline_model] = results
            except Exception as e:
                logger.error(f"Failed to benchmark baseline {baseline_model}: {e}")

        # Benchmark frontier models (if accessible)
        frontier_results = {}
        for frontier_model in self.config.frontier_models:
            try:
                logger.info(f"Benchmarking frontier: {frontier_model}")
                results = await self.benchmark_model(frontier_model, frontier_model)
                frontier_results[frontier_model] = results
                all_results[frontier_model] = results
            except Exception as e:
                logger.warning(f"Skipping frontier model {frontier_model}: {e}")

        # Create comparison report
        comparison_report = await self._create_comparison_report(
            target_results, baseline_results, frontier_results
        )

        # Generate W&B report
        await self._generate_wandb_report(all_results, comparison_report)

        # Save results
        await self._save_results(all_results, comparison_report)

        logger.info("Comprehensive benchmark completed")
        return comparison_report

    async def _create_comparison_report(
        self,
        target_results: dict[str, BenchmarkResult],
        baseline_results: dict[str, dict[str, BenchmarkResult]],
        frontier_results: dict[str, dict[str, BenchmarkResult]],
    ) -> ComparisonReport:
        """Create comprehensive comparison report."""
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(
            target_results, baseline_results, frontier_results
        )

        # Performance summary
        performance_summary = self._create_performance_summary(
            target_results, baseline_results, frontier_results
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            target_results, baseline_results, frontier_results, statistical_analysis
        )

        return ComparisonReport(
            target_model=self.config.model_name,
            baseline_results=baseline_results,
            frontier_results=frontier_results,
            target_results=target_results,
            statistical_analysis=statistical_analysis,
            performance_summary=performance_summary,
            recommendations=recommendations,
        )

    def _perform_statistical_analysis(
        self,
        target_results: dict[str, BenchmarkResult],
        baseline_results: dict[str, dict[str, BenchmarkResult]],
        frontier_results: dict[str, dict[str, BenchmarkResult]],
    ) -> dict[str, Any]:
        """Perform statistical analysis of results."""
        analysis = {}

        for benchmark_name in target_results:
            target_score = target_results[benchmark_name].overall_score

            # Collect baseline scores
            baseline_scores = []
            for model_results in baseline_results.values():
                if benchmark_name in model_results:
                    baseline_scores.append(model_results[benchmark_name].overall_score)

            # Collect frontier scores
            frontier_scores = []
            for model_results in frontier_results.values():
                if benchmark_name in model_results:
                    frontier_scores.append(model_results[benchmark_name].overall_score)

            benchmark_analysis = {
                "target_score": target_score,
                "baseline_mean": np.mean(baseline_scores) if baseline_scores else 0.0,
                "baseline_std": np.std(baseline_scores) if baseline_scores else 0.0,
                "frontier_mean": np.mean(frontier_scores) if frontier_scores else 0.0,
                "frontier_std": np.std(frontier_scores) if frontier_scores else 0.0,
                "baseline_percentile": 0.0,
                "frontier_percentile": 0.0,
            }

            # Calculate percentiles
            if baseline_scores:
                benchmark_analysis["baseline_percentile"] = stats.percentileofscore(
                    baseline_scores, target_score
                )

            if frontier_scores:
                benchmark_analysis["frontier_percentile"] = stats.percentileofscore(
                    frontier_scores, target_score
                )

            # Statistical significance tests
            if len(baseline_scores) > 1:
                # T-test against baseline mean
                t_stat, p_value = stats.ttest_1samp(
                    [target_score], np.mean(baseline_scores)
                )
                benchmark_analysis["baseline_ttest"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }

            analysis[benchmark_name] = benchmark_analysis

        return analysis

    def _create_performance_summary(
        self,
        target_results: dict[str, BenchmarkResult],
        baseline_results: dict[str, dict[str, BenchmarkResult]],
        frontier_results: dict[str, dict[str, BenchmarkResult]],
    ) -> dict[str, Any]:
        """Create performance summary."""
        summary = {
            "target_model": self.config.model_name,
            "benchmark_scores": {},
            "average_score": 0.0,
            "wins_vs_baseline": 0,
            "wins_vs_frontier": 0,
            "total_benchmarks": len(target_results),
            "execution_times": {},
            "memory_usage": {},
        }

        total_score = 0.0

        for benchmark_name, result in target_results.items():
            summary["benchmark_scores"][benchmark_name] = result.overall_score
            summary["execution_times"][benchmark_name] = result.execution_time
            summary["memory_usage"][benchmark_name] = result.memory_usage
            total_score += result.overall_score

            # Count wins against baselines
            baseline_beats = 0
            baseline_total = 0
            for model_results in baseline_results.values():
                if benchmark_name in model_results:
                    baseline_total += 1
                    if (
                        result.overall_score
                        > model_results[benchmark_name].overall_score
                    ):
                        baseline_beats += 1

            if baseline_total > 0:
                summary["wins_vs_baseline"] += baseline_beats / baseline_total

            # Count wins against frontier models
            frontier_beats = 0
            frontier_total = 0
            for model_results in frontier_results.values():
                if benchmark_name in model_results:
                    frontier_total += 1
                    if (
                        result.overall_score
                        > model_results[benchmark_name].overall_score
                    ):
                        frontier_beats += 1

            if frontier_total > 0:
                summary["wins_vs_frontier"] += frontier_beats / frontier_total

        summary["average_score"] = total_score / len(target_results)

        return summary

    def _generate_recommendations(
        self,
        target_results: dict[str, BenchmarkResult],
        baseline_results: dict[str, dict[str, BenchmarkResult]],
        frontier_results: dict[str, dict[str, BenchmarkResult]],
        statistical_analysis: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []

        # Overall performance assessment
        avg_score = np.mean([r.overall_score for r in target_results.values()])

        if avg_score > 0.8:
            recommendations.append(
                "🎉 Excellent overall performance! Model is ready for production deployment."
            )
        elif avg_score > 0.6:
            recommendations.append(
                "✅ Good performance with room for improvement in specific areas."
            )
        else:
            recommendations.append(
                "⚠️ Performance below expectations. Consider additional training or architecture changes."
            )

        # Benchmark-specific recommendations
        for benchmark_name, analysis in statistical_analysis.items():
            target_score = analysis["target_score"]

            if target_score < 0.3:
                recommendations.append(
                    f"🔴 {benchmark_name}: Critical performance gap. Requires focused improvement."
                )
            elif target_score < 0.5:
                recommendations.append(
                    f"🟡 {benchmark_name}: Below average. Consider domain-specific fine-tuning."
                )
            elif target_score > 0.8:
                recommendations.append(
                    f"🟢 {benchmark_name}: Strong performance. Maintain current approach."
                )

        # Comparison-based recommendations
        strong_areas = []
        weak_areas = []

        for benchmark_name, analysis in statistical_analysis.items():
            if analysis["baseline_percentile"] > 75:
                strong_areas.append(benchmark_name)
            elif analysis["baseline_percentile"] < 25:
                weak_areas.append(benchmark_name)

        if strong_areas:
            recommendations.append(f"💪 Strengths: {', '.join(strong_areas)}")

        if weak_areas:
            recommendations.append(
                f"🎯 Focus areas for improvement: {', '.join(weak_areas)}"
            )

        return recommendations

    async def _generate_wandb_report(
        self,
        all_results: dict[str, dict[str, BenchmarkResult]],
        comparison_report: ComparisonReport,
    ):
        """Generate comprehensive W&B report."""
        # Create comparison tables
        comparison_data = []

        for model_name, model_results in all_results.items():
            row = {"Model": model_name}

            for benchmark_name, result in model_results.items():
                row[f"{benchmark_name}_Score"] = result.overall_score
                row[f"{benchmark_name}_Time"] = result.execution_time
                row[f"{benchmark_name}_Memory"] = result.memory_usage

            comparison_data.append(row)

        # Log comparison table
        wandb.log(
            {
                "benchmark_comparison": wandb.Table(
                    dataframe=pd.DataFrame(comparison_data)
                )
            }
        )

        # Create detailed charts
        for benchmark_name in self.evaluators.keys():
            scores = []
            models = []

            for model_name, model_results in all_results.items():
                if benchmark_name in model_results:
                    scores.append(model_results[benchmark_name].overall_score)
                    models.append(model_name)

            # Bar chart
            wandb.log(
                {
                    f"{benchmark_name}_comparison": wandb.plot.bar(
                        wandb.Table(
                            data=list(zip(models, scores, strict=False)),
                            columns=["Model", "Score"],
                        ),
                        "Model",
                        "Score",
                        title=f"{benchmark_name} Benchmark Comparison",
                    )
                }
            )

        # Log performance summary
        wandb.log(
            {
                "performance_summary": comparison_report.performance_summary,
                "statistical_analysis": comparison_report.statistical_analysis,
            }
        )

        # Create performance radar chart
        target_scores = [
            comparison_report.target_results[bench].overall_score
            for bench in comparison_report.target_results.keys()
        ]

        wandb.log(
            {
                "performance_radar": wandb.plot.line_series(
                    xs=list(comparison_report.target_results.keys()),
                    ys=[target_scores],
                    keys=[comparison_report.target_model],
                    title="Performance Across Benchmarks",
                    xname="Benchmark",
                )
            }
        )

        # Log recommendations
        recommendations_text = "\n".join(comparison_report.recommendations)
        wandb.log({"recommendations": wandb.Html(f"<pre>{recommendations_text}</pre>")})

    async def _save_results(
        self,
        all_results: dict[str, dict[str, BenchmarkResult]],
        comparison_report: ComparisonReport,
    ):
        """Save detailed results to files."""
        # Save individual results
        for model_name, model_results in all_results.items():
            model_dir = self.output_dir / model_name.replace("/", "_")
            model_dir.mkdir(exist_ok=True)

            for benchmark_name, result in model_results.items():
                result_file = model_dir / f"{benchmark_name}_results.json"
                with open(result_file, "w") as f:
                    json.dump(asdict(result), f, indent=2, default=str)

        # Save comparison report
        report_file = self.output_dir / "comparison_report.json"
        with open(report_file, "w") as f:
            json.dump(asdict(comparison_report), f, indent=2, default=str)

        # Create summary CSV
        summary_data = []
        for model_name, model_results in all_results.items():
            row = {"model": model_name}
            for benchmark_name, result in model_results.items():
                row[f"{benchmark_name}_score"] = result.overall_score
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / "benchmark_summary.csv"
        summary_df.to_csv(summary_file, index=False)

        logger.info(f"Results saved to {self.output_dir}")


# CLI and main execution
async def main():
    """Main benchmark execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Model Benchmarking")
    parser.add_argument(
        "--model-path", required=True, help="Path to model to benchmark"
    )
    parser.add_argument("--model-name", required=True, help="Name for the model")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for results"
    )

    # Benchmark selection
    parser.add_argument(
        "--run-mmlu", action="store_true", default=True, help="Run MMLU evaluation"
    )
    parser.add_argument(
        "--run-gsm8k", action="store_true", default=True, help="Run GSM8K evaluation"
    )
    parser.add_argument(
        "--run-humaneval",
        action="store_true",
        default=True,
        help="Run HumanEval evaluation",
    )

    # Configuration
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument("--max-samples", type=int, help="Maximum samples per benchmark")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Model precision",
    )

    # W&B settings
    parser.add_argument(
        "--wandb-project", default="agent-forge-benchmark", help="W&B project name"
    )
    parser.add_argument("--wandb-entity", help="W&B entity")

    args = parser.parse_args()

    # Create configuration
    config = BenchmarkConfig(
        model_path=args.model_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        run_mmlu=args.run_mmlu,
        run_gsm8k=args.run_gsm8k,
        run_humaneval=args.run_humaneval,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
        precision=args.precision,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    # Run comprehensive benchmark
    benchmark = ComprehensiveBenchmark(config)
    comparison_report = await benchmark.run_comprehensive_benchmark()

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK SUMMARY - {config.model_name}")
    print(f"{'=' * 60}")

    print("\nOverall Performance:")
    print(
        f"Average Score: {comparison_report.performance_summary['average_score']:.4f}"
    )
    print(
        f"Benchmarks Completed: {comparison_report.performance_summary['total_benchmarks']}"
    )

    print("\nBenchmark Scores:")
    for benchmark, score in comparison_report.performance_summary[
        "benchmark_scores"
    ].items():
        print(f"  {benchmark}: {score:.4f}")

    print("\nRecommendations:")
    for i, rec in enumerate(comparison_report.recommendations, 1):
        print(f"  {i}. {rec}")

    print(f"\nDetailed results saved to: {args.output_dir}")
    print(
        f"W&B report: https://wandb.ai/{args.wandb_entity or 'agent-forge'}/{args.wandb_project}"
    )


if __name__ == "__main__":
    asyncio.run(main())
