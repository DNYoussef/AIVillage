"""
ADAS Runner - Main orchestrator for the specialization process.
Runs the complete ADAS×Transformer² pipeline with evaluation and ranking.
"""

import logging
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

from ..t2.features import FeatureExtractor
from ..t2.mixer import T2Mixer
from .archive import ADASArchive, ExperimentResult
from .proposer import ADASProposer

logger = logging.getLogger(__name__)


class TaskSuite:
    """Simple task suite for evaluating expert configurations."""

    def __init__(self, suite_name: str = "coding_small"):
        self.suite_name = suite_name
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> list[dict[str, Any]]:
        """Load task suite (simplified for MVP)."""
        if self.suite_name == "coding_small":
            return [
                {
                    "prompt": "def fibonacci(n): # Complete this function",
                    "expected_keywords": ["if", "return", "recursion or loop"],
                    "complexity": 0.3,
                },
                {
                    "prompt": "def quicksort(arr): # Implement quicksort",
                    "expected_keywords": ["partition", "recursive", "pivot"],
                    "complexity": 0.7,
                },
                {
                    "prompt": "class BinaryTree: # Implement binary tree with insert",
                    "expected_keywords": ["__init__", "insert", "left", "right"],
                    "complexity": 0.5,
                },
            ]
        else:
            # Default simple tasks
            return [
                {
                    "prompt": f"Generate solution for task {i}",
                    "expected_keywords": ["solution", "function"],
                    "complexity": np.random.random(),
                }
                for i in range(5)
            ]

    def evaluate(self, model_outputs: list[str]) -> float:
        """Simple evaluation based on keyword matching."""
        if len(model_outputs) != len(self.tasks):
            return 0.0

        scores = []
        for output, task in zip(model_outputs, self.tasks, strict=False):
            output_lower = output.lower()
            keyword_score = sum(
                1
                for kw in task["expected_keywords"]
                if any(k.lower() in output_lower for k in kw.split())
            )
            keyword_score /= len(task["expected_keywords"])

            # Length-based quality heuristic
            length_score = min(1.0, len(output) / 100)  # Prefer longer responses

            task_score = 0.7 * keyword_score + 0.3 * length_score
            scores.append(task_score)

        return np.mean(scores)


class ADASRunner:
    """Main ADAS runner that orchestrates the specialization search."""

    def __init__(self, base_model, tokenizer, archive_path: Path, device: str = "auto"):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize components
        self.archive = ADASArchive(archive_path)
        self.proposer = ADASProposer(self.archive)
        self.feature_extractor = FeatureExtractor()

        # Performance tracking
        self.trial_start_time = None
        self.total_trials = 0

    def run_specialization(
        self,
        n_trials: int = 24,
        time_budget_minutes: int = 30,
        task_suite: str = "coding_small",
        max_concurrent: int = 2,
    ) -> dict[str, Any]:
        """
        Run complete ADAS specialization search.

        Args:
            n_trials: Number of configurations to evaluate
            time_budget_minutes: Maximum time budget
            task_suite: Name of task suite to use
            max_concurrent: Maximum concurrent evaluations

        Returns:
            Summary results including leaderboard
        """
        self.trial_start_time = time.time()
        time_budget_seconds = time_budget_minutes * 60

        logger.info(
            f"Starting ADAS specialization with {n_trials} trials, "
            f"{time_budget_minutes}min budget"
        )

        # Initialize task suite
        tasks = TaskSuite(task_suite)

        # Track results
        all_results = []

        # Run trials in batches to respect concurrency limits
        batch_size = min(max_concurrent, n_trials)

        for batch_start in range(0, n_trials, batch_size):
            batch_end = min(batch_start + batch_size, n_trials)
            current_batch_size = batch_end - batch_start

            # Check time budget
            elapsed = time.time() - self.trial_start_time
            if elapsed >= time_budget_seconds:
                logger.warning(f"Time budget exceeded, stopping at trial {batch_start}")
                break

            logger.info(f"Running trial batch {batch_start + 1}-{batch_end}")

            # Generate proposals for this batch
            previous_results = all_results if all_results else None
            proposals = self.proposer.propose(
                n_proposals=current_batch_size,
                target_latency_ms=100,  # Default budget
                previous_results=previous_results,
            )

            # Evaluate batch
            batch_results = self._evaluate_batch(proposals, tasks)
            all_results.extend(batch_results)

            # Add to archive
            for result in batch_results:
                self.archive.add_result(result)

        self.total_trials = len(all_results)

        # Generate summary
        summary = self._generate_summary(all_results)

        logger.info(
            f"Specialization complete: {self.total_trials} trials in "
            f"{time.time() - self.trial_start_time:.1f}s"
        )

        return summary

    def _evaluate_batch(
        self, proposals: list[dict[str, Any]], tasks: TaskSuite
    ) -> list[ExperimentResult]:
        """Evaluate a batch of proposals."""
        results = []

        for i, proposal in enumerate(proposals):
            trial_id = f"trial_{self.total_trials + i + 1:03d}"
            logger.info(f"Evaluating {trial_id}: {proposal['motivation']}")

            try:
                result = self._evaluate_single_proposal(proposal, tasks, trial_id)
                results.append(result)

                logger.info(
                    f"{trial_id} complete: score={result.score:.4f}, "
                    f"latency={result.latency_ms:.1f}ms"
                )

            except Exception as e:
                logger.error(f"{trial_id} failed: {e}")

                # Create failure result
                result = ExperimentResult(
                    expert_spec=proposal["expert"],
                    dispatch_spec=proposal["dispatch"],
                    trial_id=trial_id,
                    success=False,
                    error_msg=str(e),
                    task_suite=tasks.suite_name,
                )
                results.append(result)

        return results

    def _evaluate_single_proposal(
        self, proposal: dict[str, Any], tasks: TaskSuite, trial_id: str
    ) -> ExperimentResult:
        """Evaluate a single expert configuration proposal."""
        start_time = time.time()

        # Create T2Mixer from proposal
        mixer = T2Mixer(
            dispatch_spec=proposal["dispatch"],
            expert_lib={trial_id: proposal["expert"]},  # Single expert for now
        )

        # Initialize VRAM tracking
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_vram = self._get_vram_usage()

        # Run inference on tasks
        outputs = []
        inference_times = []

        for task in tasks.tasks:
            task_start = time.time()

            # Tokenize input
            inputs = self.tokenizer(
                task["prompt"], return_tensors="pt", max_length=256, truncation=True
            ).to(self.device)

            # Extract features for dispatch
            prompt_stats = self.feature_extractor.extract_prompt_stats(task["prompt"])

            # Generate with expert mixing
            with torch.no_grad():
                # Compute dispatch weights
                weights = mixer.dispatch(
                    prompt_stats, {}
                )  # Empty activation sketch for now

                # Apply expert patches and generate
                with mixer.patch(self.base_model, weights):
                    output_ids = self.base_model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # Decode output
                output_text = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                outputs.append(output_text)

            inference_times.append((time.time() - task_start) * 1000)  # Convert to ms

        # Compute metrics
        total_time = time.time() - start_time
        score = tasks.evaluate(outputs)
        avg_latency = np.mean(inference_times)
        peak_vram = self._get_vram_usage()
        vram_used = max(0, peak_vram - initial_vram)

        # Create result
        result = ExperimentResult(
            expert_spec=proposal["expert"],
            dispatch_spec=proposal["dispatch"],
            score=score,
            latency_ms=avg_latency,
            vram_gb=vram_used,
            trial_id=trial_id,
            task_suite=tasks.suite_name,
            num_samples=len(tasks.tasks),
            success=True,
            metrics={
                "total_time_s": total_time,
                "min_latency_ms": min(inference_times),
                "max_latency_ms": max(inference_times),
                "std_latency_ms": np.std(inference_times),
            },
        )

        return result

    def _get_vram_usage(self) -> float:
        """Get current VRAM usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def _generate_summary(self, results: list[ExperimentResult]) -> dict[str, Any]:
        """Generate comprehensive summary of specialization run."""
        successful = [r for r in results if r.success]

        summary = {
            "total_trials": len(results),
            "successful_trials": len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_time_s": time.time() - self.trial_start_time
            if self.trial_start_time
            else 0,
        }

        if successful:
            scores = [r.score for r in successful]
            latencies = [r.latency_ms for r in successful]
            vram_usage = [r.vram_gb for r in successful]

            summary.update(
                {
                    "best_score": max(scores),
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "best_latency_ms": min(latencies),
                    "mean_latency_ms": np.mean(latencies),
                    "worst_latency_ms": max(latencies),
                    "mean_vram_gb": np.mean(vram_usage),
                    "max_vram_gb": max(vram_usage),
                }
            )

            # Get leaderboard
            leaderboard = self.archive.get_leaderboard(top_k=10)
            summary["leaderboard"] = [
                {
                    "rank": i + 1,
                    "score": r.score,
                    "latency_ms": r.latency_ms,
                    "vram_gb": r.vram_gb,
                    "trial_id": r.trial_id,
                    "motivation": self._get_proposal_motivation(r),
                }
                for i, r in enumerate(leaderboard)
            ]

            # Pareto frontier analysis
            pareto = self.archive.get_pareto_frontier(["score", "latency_ms"])
            summary["pareto_frontier_size"] = len(pareto)

        return summary

    def _get_proposal_motivation(self, result: ExperimentResult) -> str:
        """Extract motivation from result (would be stored with proposal)."""
        return f"Expert on {result.expert_spec.get('layers', 'unknown')} with rank {result.expert_spec.get('rank', '?')}"


# CLI Integration
@click.group()
def specialize():
    """ADAS×Transformer² specialization commands."""
    pass


@specialize.command()
@click.option("--trials", default=24, type=int, help="Number of trials to run")
@click.option("--time-budget-min", default=30, type=int, help="Time budget in minutes")
@click.option("--tasks", default="coding_small", help="Task suite to use")
@click.option("--model-name", default="gpt2", help="Base model name")
@click.option("--archive-path", default="./adas_archive.jsonl", help="Archive path")
@click.option("--output-dir", default="./adas_output", help="Output directory")
@click.option("--device", default="auto", help="Device to use")
def run(
    trials: int,
    time_budget_min: int,
    tasks: str,
    model_name: str,
    archive_path: str,
    output_dir: str,
    device: str,
):
    """Run ADAS specialization search."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {model_name}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize runner
    runner = ADASRunner(
        base_model=model,
        tokenizer=tokenizer,
        archive_path=Path(archive_path),
        device=device,
    )

    # Run specialization
    results = runner.run_specialization(
        n_trials=trials, time_budget_minutes=time_budget_min, task_suite=tasks
    )

    # Print results
    print("\n" + "=" * 80)
    print("ADAS×Transformer² Specialization Results")
    print("=" * 80)
    print(f"Total trials: {results['total_trials']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Total time: {results['total_time_s']:.1f}s")

    if "leaderboard" in results:
        print("\nTop configurations:")
        print(
            f"{'Rank':<4} {'Score':<8} {'Latency':<10} {'VRAM':<8} {'Trial ID':<12} {'Description'}"
        )
        print("-" * 70)

        for entry in results["leaderboard"][:5]:
            print(
                f"{entry['rank']:<4} {entry['score']:<8.4f} "
                f"{entry['latency_ms']:<10.1f} {entry['vram_gb']:<8.3f} "
                f"{entry['trial_id']:<12} {entry['motivation'][:30]}..."
            )

    # Save detailed results
    import json

    results_path = output_path / "specialization_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Export top configs
    config_exports = runner.archive.export_yaml_configs(
        output_dir=output_path / "top_configs", top_k=5
    )

    print(f"\nResults saved to: {output_path}")
    print(f"Archive: {archive_path}")
    print(f"Config exports: {len(config_exports)} files")


@specialize.command()
@click.option("--archive-path", default="./adas_archive.jsonl", help="Archive path")
@click.option("--top-k", default=10, type=int, help="Number of top results to show")
def analyze(archive_path: str, top_k: int):
    """Analyze ADAS archive results."""
    archive = ADASArchive(Path(archive_path))

    stats = archive.get_statistics()
    print("Archive Statistics:")
    print(f"  Total results: {stats['total_results']}")
    print(f"  Successful: {stats['successful_results']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")

    if stats["total_results"] > 0:
        leaderboard = archive.get_leaderboard(top_k=top_k)

        print(f"\nTop {len(leaderboard)} Configurations:")
        print(
            f"{'Rank':<4} {'Score':<8} {'Latency':<10} {'VRAM':<8} {'Layers':<15} {'Rank':<4}"
        )
        print("-" * 70)

        for i, result in enumerate(leaderboard):
            layers_str = str(result.expert_spec.get("layers", []))[:12]
            rank = result.expert_spec.get("rank", 0)

            print(
                f"{i + 1:<4} {result.score:<8.4f} {result.latency_ms:<10.1f} "
                f"{result.vram_gb:<8.3f} {layers_str:<15} {rank:<4}"
            )


if __name__ == "__main__":
    specialize()
