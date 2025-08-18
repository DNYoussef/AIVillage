"""Benchmark orchestrator for EvoMerge with modular suite support and W&B integration."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import wandb
import yaml
from pydantic import BaseModel


class BenchmarkSuite(BaseModel):
    """Benchmark suite configuration."""

    objectives: list[str]
    task_groups: list[dict[str, Any]]


class BenchmarkOrchestrator:
    """Orchestrates benchmarking with configurable suites and W&B logging."""

    def __init__(self, suite_name: str = "general", wandb_mode: str = "offline"):
        """Initialize benchmark orchestrator.

        Args:
            suite_name: Name of benchmark suite (writing, coding, math, general)
            wandb_mode: W&B mode (offline, online, disabled)
        """
        self.suite_name = suite_name
        self.wandb_mode = wandb_mode
        self.suite_config = self._load_suite_config(suite_name)

        # Setup W&B environment
        self._setup_wandb_env()

    def _setup_wandb_env(self) -> None:
        """Setup W&B environment variables."""
        if "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = self.wandb_mode

        if "WANDB_DIR" not in os.environ:
            wandb_dir = os.environ.get("AIV_ROOT", "D:\\AIVillage") + "\\wandb"
            os.environ["WANDB_DIR"] = wandb_dir
            Path(wandb_dir).mkdir(parents=True, exist_ok=True)

    def _load_suite_config(self, suite_name: str) -> BenchmarkSuite:
        """Load benchmark suite configuration from YAML."""
        aiv_root = os.environ.get("AIV_ROOT", "D:\\AIVillage")
        suite_path = Path(aiv_root) / "benchmarks" / "suites" / f"{suite_name}.yaml"

        if not suite_path.exists():
            raise FileNotFoundError(f"Benchmark suite not found: {suite_path}")

        with open(suite_path) as f:
            config_data = yaml.safe_load(f)

        return BenchmarkSuite(**config_data)

    def _get_task_list(self) -> list[str]:
        """Get flattened list of all tasks from task groups."""
        tasks = []
        for group in self.suite_config.task_groups:
            tasks.extend(group.get("tasks", []))
        return list(set(tasks))  # Remove duplicates

    def benchmark_model(self, model_path: str, generation: int, phase: str, child_id: str) -> dict[str, float]:
        """Benchmark a single model and return aggregated metrics.

        Args:
            model_path: Path to model directory
            generation: Generation number
            phase: Phase name (pre/post)
            child_id: Child identifier (e.g., "child_01")

        Returns:
            Dict mapping objective names to scores
        """
        model_name = Path(model_path).name

        # Initialize W&B run
        wandb.init(
            project="AIVillage-EvoMerge",
            group=model_name,
            name=f"{model_name}-G{generation:04d}-{phase}",
            tags=[
                f"generation:{generation}",
                f"phase:{phase}",
                f"suite:{self.suite_name}",
                child_id,
            ],
            config={
                "model_path": model_path,
                "suite": self.suite_name,
                "generation": generation,
                "phase": phase,
            },
        )

        try:
            # Run lm-evaluation-harness
            results = self._run_lm_eval(model_path, generation, phase, child_id)

            # Extract objective scores
            objective_scores = self._extract_objective_scores(results)

            # Log to W&B
            wandb.log(objective_scores)
            wandb.log({"aggregated_fitness": sum(objective_scores.values()) / len(objective_scores)})

            # Save detailed results
            self._save_detailed_results(results, model_path, generation, phase, child_id)

            return objective_scores

        finally:
            wandb.finish()

    def _run_lm_eval(self, model_path: str, generation: int, phase: str, child_id: str) -> dict[str, Any]:
        """Run lm-evaluation-harness on the model."""
        tasks = self._get_task_list()
        task_str = ",".join(tasks)

        # Setup output directory
        aiv_benchmarks = os.environ.get("AIV_BENCHMARKS_DIR", "D:\\AIVillage\\benchmarks\\results")
        output_dir = Path(aiv_benchmarks) / f"G{generation:04d}" / phase / child_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary results file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_results_file = f.name

        try:
            # Run lm_eval command
            cmd = [
                "lm_eval",
                "--model",
                "hf",
                "--model_args",
                f"pretrained={model_path}",
                "--tasks",
                task_str,
                "--batch_size",
                "1",
                "--output_path",
                str(output_dir),
                "--log_samples",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"lm_eval failed: {result.stderr}")

            # Load results from output directory
            results_file = output_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"Results file not found: {results_file}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("lm_eval timed out after 1 hour")
        finally:
            # Cleanup temp file
            if os.path.exists(temp_results_file):
                os.unlink(temp_results_file)

    def _extract_objective_scores(self, results: dict[str, Any]) -> dict[str, float]:
        """Extract objective scores from lm_eval results."""
        objective_scores = {}

        for objective in self.suite_config.objectives:
            # Map objective names to lm_eval result keys
            score_key = self._map_objective_to_result_key(objective, results)

            if score_key and score_key in results.get("results", {}):
                task_result = results["results"][score_key]
                # Get the primary metric (usually accuracy or exact_match)
                if "acc" in task_result:
                    objective_scores[objective] = task_result["acc"]
                elif "exact_match" in task_result:
                    objective_scores[objective] = task_result["exact_match"]
                elif "acc_norm" in task_result:
                    objective_scores[objective] = task_result["acc_norm"]
                else:
                    # Take first numeric value
                    for value in task_result.values():
                        if isinstance(value, int | float):
                            objective_scores[objective] = float(value)
                            break
                    else:
                        objective_scores[objective] = 0.0
            else:
                objective_scores[objective] = 0.0

        return objective_scores

    def _map_objective_to_result_key(self, objective: str, results: dict[str, Any]) -> str | None:
        """Map objective name to lm_eval result key."""
        # Handle common mappings
        objective_mappings = {
            "mmlu_score": "mmlu",
            "gsm8k_score": "gsm8k",
            "hellaswag_score": "hellaswag",
            "humaneval_score": "humaneval",
            "mbpp_score": "mbpp",
            "boolq_score": "boolq",
            "truthfulqa_mc_score": "truthfulqa_mc",
            "winogrande_score": "winogrande",
            "mmlu_stem_score": "mmlu_stem",
        }

        mapped_key = objective_mappings.get(objective, objective.replace("_score", ""))

        # Check if the mapped key exists in results
        if mapped_key in results.get("results", {}):
            return mapped_key

        # Try partial matching
        for key in results.get("results", {}).keys():
            if mapped_key in key or key in mapped_key:
                return key

        return None

    def _save_detailed_results(
        self,
        results: dict[str, Any],
        model_path: str,
        generation: int,
        phase: str,
        child_id: str,
    ) -> None:
        """Save detailed benchmark results."""
        aiv_benchmarks = os.environ.get("AIV_BENCHMARKS_DIR", "D:\\AIVillage\\benchmarks\\results")
        output_dir = Path(aiv_benchmarks) / f"G{generation:04d}" / phase / child_id

        # Save full results
        full_results_file = output_dir / "full_results.json"
        with open(full_results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save aggregate summary
        objective_scores = self._extract_objective_scores(results)
        aggregate_summary = {
            "model_path": model_path,
            "generation": generation,
            "phase": phase,
            "child_id": child_id,
            "suite": self.suite_name,
            "objective_scores": objective_scores,
            "aggregated_fitness": sum(objective_scores.values()) / len(objective_scores) if objective_scores else 0.0,
        }

        aggregate_file = output_dir / "aggregate.json"
        with open(aggregate_file, "w") as f:
            json.dump(aggregate_summary, f, indent=2)


def determine_model_suite(model_path: str, override_suite: str | None = None) -> str:
    """Determine benchmark suite for a model based on metadata or override.

    Args:
        model_path: Path to model directory
        override_suite: Optional suite override from CLI

    Returns:
        Suite name (writing, coding, math, general)
    """
    if override_suite:
        return override_suite

    # Try to load model metadata
    aiv_root = os.environ.get("AIV_ROOT", "D:\\AIVillage")
    models_yaml = Path(aiv_root) / "models" / "models.yaml"

    if models_yaml.exists():
        with open(models_yaml) as f:
            models_config = yaml.safe_load(f)

        model_name = Path(model_path).name
        for model in models_config.get("models", []):
            if model_name in model.get("local", "") or model_name in model.get("id", ""):
                return model.get("type", "general")

    # Default to general suite
    return "general"


if __name__ == "__main__":
    # Test benchmark orchestrator
    orchestrator = BenchmarkOrchestrator("general")
    print(f"Loaded suite: {orchestrator.suite_name}")
    print(f"Objectives: {orchestrator.suite_config.objectives}")
    print(f"Tasks: {orchestrator._get_task_list()}")
