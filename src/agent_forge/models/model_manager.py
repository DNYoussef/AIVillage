"""Model Management System for Agent Forge.

Handles downloading, caching, and lifecycle management of models with
intelligent storage on D: drive and automatic cleanup to maintain
maximum 8 models at any time.
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any

from huggingface_hub import snapshot_download
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import wandb

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a managed model."""

    model_id: str
    local_path: str
    size_gb: float
    downloaded_at: datetime
    last_used: datetime
    generation: int  # 0 for seed models, 1+ for evolved models
    parent_models: list[str]  # For tracking evolution lineage
    performance_score: float | None
    compression_ratio: float | None
    metadata: dict[str, Any]


@dataclass
class BenchmarkResult:
    """Benchmark result for a model."""

    model_id: str
    benchmark_name: str
    score: float
    timestamp: datetime
    details: dict[str, Any]


class ModelManager:
    """Manages model lifecycle with D: drive storage and automatic cleanup."""

    def __init__(
        self, base_dir: str = "D:/AIVillage/models", max_models: int = 8, wandb_project: str = "agent-forge-models"
    ):
        self.base_dir = Path(base_dir)
        self.max_models = max_models
        self.wandb_project = wandb_project

        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_dir / "model_registry.json"

        # Model registry
        self.models: dict[str, ModelInfo] = self._load_registry()

        # Benchmark tracking
        self.benchmarks: list[BenchmarkResult] = []
        self.benchmark_path = self.base_dir / "benchmarks.json"
        self._load_benchmarks()

        logger.info(f"Model Manager initialized with base dir: {self.base_dir}")
        logger.info(f"Current models: {len(self.models)}")

    async def download_seed_models(self, model_specs: list[dict[str, str]]) -> list[str]:
        """Download the seed models for Agent Forge pipeline."""

        logger.info("Starting seed model download process...")
        downloaded_models = []

        for spec in model_specs:
            model_id = spec["model_id"]
            purpose = spec.get("purpose", "general")

            try:
                logger.info(f"Downloading {model_id} ({purpose})")
                local_path = await self._download_model(model_id, generation=0)
                downloaded_models.append(model_id)

                # Run initial benchmarks
                await self._benchmark_model(model_id, local_path)

            except Exception as e:
                logger.error(f"Failed to download {model_id}: {e}")
                continue

        # Cleanup if needed
        await self._cleanup_models()

        logger.info(f"Downloaded {len(downloaded_models)} seed models")
        return downloaded_models

    async def _download_model(self, model_id: str, generation: int = 0, parent_models: list[str] = None) -> str:
        """Download a model to D: drive."""

        # Check if already downloaded
        if model_id in self.models:
            self.models[model_id].last_used = datetime.now()
            self._save_registry()
            return self.models[model_id].local_path

        # Create safe directory name
        safe_name = model_id.replace("/", "_").replace(":", "_")
        local_path = self.base_dir / safe_name

        logger.info(f"Downloading {model_id} to {local_path}")

        try:
            # Download using huggingface_hub
            snapshot_download(
                repo_id=model_id, cache_dir=str(local_path), local_dir=str(local_path / "model"), local_files_only=False
            )

            # Get model size
            size_gb = self._calculate_directory_size(local_path) / (1024**3)

            # Create model info
            model_info = ModelInfo(
                model_id=model_id,
                local_path=str(local_path),
                size_gb=size_gb,
                downloaded_at=datetime.now(),
                last_used=datetime.now(),
                generation=generation,
                parent_models=parent_models or [],
                performance_score=None,
                compression_ratio=None,
                metadata={"download_method": "huggingface_hub", "downloaded_from": "huggingface.co"},
            )

            self.models[model_id] = model_info
            self._save_registry()

            logger.info(f"Successfully downloaded {model_id} ({size_gb:.2f} GB)")
            return str(local_path)

        except Exception as e:
            logger.error(f"Download failed for {model_id}: {e}")
            # Cleanup partial download
            if local_path.exists():
                shutil.rmtree(local_path, ignore_errors=True)
            raise

    async def _benchmark_model(self, model_id: str, local_path: str):
        """Run benchmarks on a downloaded model."""

        logger.info(f"Running benchmarks for {model_id}")

        try:
            # Load model for benchmarking
            model_path = Path(local_path) / "model"

            # Basic model loading test
            start_time = datetime.now()

            try:
                config = AutoConfig.from_pretrained(str(model_path))
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path), torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
                )

                load_time = (datetime.now() - start_time).total_seconds()

                # Basic inference test
                test_prompt = "def fibonacci(n):"
                inputs = tokenizer(test_prompt, return_tensors="pt")

                inference_start = datetime.now()
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                inference_time = (datetime.now() - inference_start).total_seconds()
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Calculate basic performance metrics
                model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

                benchmark_result = BenchmarkResult(
                    model_id=model_id,
                    benchmark_name="basic_performance",
                    score=1.0 / max(inference_time, 0.001),  # Inverse of inference time
                    timestamp=datetime.now(),
                    details={
                        "load_time_seconds": load_time,
                        "inference_time_seconds": inference_time,
                        "model_size_mb": model_size_mb,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "test_response": response,
                        "config": {
                            "vocab_size": config.vocab_size,
                            "hidden_size": getattr(config, "hidden_size", None),
                            "num_layers": getattr(config, "num_hidden_layers", None),
                            "num_attention_heads": getattr(config, "num_attention_heads", None),
                        },
                    },
                )

                self.benchmarks.append(benchmark_result)
                self._save_benchmarks()

                # Update model performance score
                if model_id in self.models:
                    self.models[model_id].performance_score = benchmark_result.score
                    self._save_registry()

                # Log to W&B if available
                await self._log_to_wandb(benchmark_result)

                logger.info(f"Benchmark completed for {model_id}: score={benchmark_result.score:.3f}")

                # Cleanup GPU memory
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Model loading/inference failed for {model_id}: {e}")

                # Create a failure benchmark
                benchmark_result = BenchmarkResult(
                    model_id=model_id,
                    benchmark_name="basic_performance",
                    score=0.0,
                    timestamp=datetime.now(),
                    details={"error": str(e), "load_time_seconds": load_time, "status": "failed"},
                )

                self.benchmarks.append(benchmark_result)
                self._save_benchmarks()

        except Exception as e:
            logger.error(f"Benchmarking failed for {model_id}: {e}")

    async def _log_to_wandb(self, benchmark: BenchmarkResult):
        """Log benchmark results to Weights & Biases."""

        try:
            # Initialize W&B run
            run = wandb.init(
                project=self.wandb_project,
                name=f"benchmark_{benchmark.model_id}_{benchmark.timestamp.strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model_id": benchmark.model_id,
                    "benchmark_name": benchmark.benchmark_name,
                    **benchmark.details,
                },
            )

            # Log metrics
            wandb.log(
                {
                    "benchmark_score": benchmark.score,
                    "load_time": benchmark.details.get("load_time_seconds", 0),
                    "inference_time": benchmark.details.get("inference_time_seconds", 0),
                    "model_size_mb": benchmark.details.get("model_size_mb", 0),
                    "parameters": benchmark.details.get("parameters", 0),
                }
            )

            # Log model config as table
            if "config" in benchmark.details:
                config_table = wandb.Table(
                    columns=["Parameter", "Value"],
                    data=[[k, v] for k, v in benchmark.details["config"].items() if v is not None],
                )
                wandb.log({"model_config": config_table})

            wandb.finish()
            logger.info(f"Logged benchmark to W&B for {benchmark.model_id}")

        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    async def _cleanup_models(self):
        """Cleanup old models to maintain max_models limit."""

        if len(self.models) <= self.max_models:
            return

        logger.info(f"Cleaning up models: {len(self.models)} > {self.max_models}")

        # Sort by last used time (oldest first)
        sorted_models = sorted(self.models.items(), key=lambda x: x[1].last_used)

        # Keep the most recent max_models
        to_remove = sorted_models[: -self.max_models]

        for model_id, model_info in to_remove:
            try:
                logger.info(f"Removing old model: {model_id}")

                # Remove from disk
                model_path = Path(model_info.local_path)
                if model_path.exists():
                    shutil.rmtree(model_path, ignore_errors=True)

                # Remove from registry
                del self.models[model_id]

            except Exception as e:
                logger.error(f"Failed to remove model {model_id}: {e}")

        self._save_registry()
        logger.info(f"Cleanup complete. Models remaining: {len(self.models)}")

    def get_model_path(self, model_id: str) -> str | None:
        """Get the local path for a model."""
        if model_id in self.models:
            self.models[model_id].last_used = datetime.now()
            self._save_registry()
            return self.models[model_id].local_path
        return None

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get information about a model."""
        return self.models.get(model_id)

    def list_models(self) -> list[ModelInfo]:
        """List all managed models."""
        return list(self.models.values())

    def get_benchmark_results(self, model_id: str | None = None) -> list[BenchmarkResult]:
        """Get benchmark results."""
        if model_id:
            return [b for b in self.benchmarks if b.model_id == model_id]
        return self.benchmarks.copy()

    def _calculate_directory_size(self, path: Path) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        try:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating size for {path}: {e}")
        return total_size

    def _load_registry(self) -> dict[str, ModelInfo]:
        """Load model registry from disk."""
        if not self.registry_path.exists():
            return {}

        try:
            with open(self.registry_path) as f:
                data = json.load(f)

            models = {}
            for model_id, model_data in data.items():
                # Convert datetime strings back to datetime objects
                model_data["downloaded_at"] = datetime.fromisoformat(model_data["downloaded_at"])
                model_data["last_used"] = datetime.fromisoformat(model_data["last_used"])
                models[model_id] = ModelInfo(**model_data)

            return models

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {}

    def _save_registry(self):
        """Save model registry to disk."""
        try:
            data = {}
            for model_id, model_info in self.models.items():
                model_data = asdict(model_info)
                # Convert datetime objects to strings
                model_data["downloaded_at"] = model_info.downloaded_at.isoformat()
                model_data["last_used"] = model_info.last_used.isoformat()
                data[model_id] = model_data

            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def _load_benchmarks(self):
        """Load benchmarks from disk."""
        if not self.benchmark_path.exists():
            return

        try:
            with open(self.benchmark_path) as f:
                data = json.load(f)

            self.benchmarks = []
            for benchmark_data in data:
                benchmark_data["timestamp"] = datetime.fromisoformat(benchmark_data["timestamp"])
                self.benchmarks.append(BenchmarkResult(**benchmark_data))

        except Exception as e:
            logger.error(f"Failed to load benchmarks: {e}")

    def _save_benchmarks(self):
        """Save benchmarks to disk."""
        try:
            data = []
            for benchmark in self.benchmarks:
                benchmark_data = asdict(benchmark)
                benchmark_data["timestamp"] = benchmark.timestamp.isoformat()
                data.append(benchmark_data)

            with open(self.benchmark_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save benchmarks: {e}")


class EvoMergeManager:
    """Manages the evolutionary merging process."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.generation_history: dict[int, list[str]] = {}

    async def start_evomerge_process(self, seed_models: list[str], max_generations: int = 5) -> list[str]:
        """Start the EvoMerge evolutionary process."""

        logger.info(f"Starting EvoMerge process with {len(seed_models)} seed models")
        logger.info(f"Target generations: {max_generations}")

        current_generation = seed_models.copy()
        self.generation_history[0] = current_generation

        for generation in range(1, max_generations + 1):
            logger.info(f"Creating generation {generation}")

            # Select best performing models from previous generation
            parent_models = await self._select_parents(current_generation)

            # Create offspring through merging
            offspring = await self._create_offspring(parent_models, generation)

            # Evaluate offspring
            evaluated_offspring = []
            for model_id in offspring:
                try:
                    model_path = self.model_manager.get_model_path(model_id)
                    await self.model_manager._benchmark_model(model_id, model_path)
                    evaluated_offspring.append(model_id)
                except Exception as e:
                    logger.error(f"Failed to evaluate offspring {model_id}: {e}")

            current_generation = evaluated_offspring
            self.generation_history[generation] = current_generation

            # Cleanup previous generation (keep only best)
            if generation > 1:
                await self._cleanup_generation(generation - 2)

            logger.info(f"Generation {generation} complete: {len(current_generation)} models")

        # Return best models from final generation
        best_models = await self._select_best_models(current_generation, top_k=3)
        logger.info(f"EvoMerge complete. Best models: {best_models}")

        return best_models

    async def _select_parents(self, models: list[str]) -> list[str]:
        """Select parent models based on performance."""

        model_scores = []
        for model_id in models:
            model_info = self.model_manager.get_model_info(model_id)
            if model_info and model_info.performance_score:
                model_scores.append((model_id, model_info.performance_score))

        # Sort by performance (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top 50% as parents
        num_parents = max(2, len(model_scores) // 2)
        parents = [model_id for model_id, _ in model_scores[:num_parents]]

        logger.info(f"Selected {len(parents)} parent models for breeding")
        return parents

    async def _create_offspring(self, parents: list[str], generation: int) -> list[str]:
        """Create offspring models through evolutionary merging."""

        logger.info(f"Creating offspring from {len(parents)} parents")
        offspring = []

        # Create multiple offspring combinations
        for i in range(min(4, len(parents))):  # Limit offspring per generation
            try:
                # Simple merging strategy: combine two best parents
                parent1 = parents[i % len(parents)]
                parent2 = parents[(i + 1) % len(parents)]

                offspring_id = f"evomerge_gen{generation}_offspring{i + 1}"

                # For now, create a symbolic offspring (actual merging would require model architecture knowledge)
                offspring_path = await self._create_merged_model(parent1, parent2, offspring_id, generation)

                offspring.append(offspring_id)

            except Exception as e:
                logger.error(f"Failed to create offspring {i}: {e}")

        logger.info(f"Created {len(offspring)} offspring models")
        return offspring

    async def _create_merged_model(self, parent1: str, parent2: str, offspring_id: str, generation: int) -> str:
        """Create a merged model (simplified implementation)."""

        # For now, create a copy of the better performing parent
        parent1_info = self.model_manager.get_model_info(parent1)
        parent2_info = self.model_manager.get_model_info(parent2)

        if not parent1_info or not parent2_info:
            raise ValueError("Parent model info not found")

        # Select better performing parent as base
        if (parent1_info.performance_score or 0) >= (parent2_info.performance_score or 0):
            base_parent = parent1
            base_path = parent1_info.local_path
        else:
            base_parent = parent2
            base_path = parent2_info.local_path

        # Create offspring directory
        safe_name = offspring_id.replace("/", "_").replace(":", "_")
        offspring_path = self.model_manager.base_dir / safe_name

        # Copy base parent model
        shutil.copytree(base_path, offspring_path)

        # Register as new model
        offspring_info = ModelInfo(
            model_id=offspring_id,
            local_path=str(offspring_path),
            size_gb=parent1_info.size_gb,  # Approximate
            downloaded_at=datetime.now(),
            last_used=datetime.now(),
            generation=generation,
            parent_models=[parent1, parent2],
            performance_score=None,
            compression_ratio=None,
            metadata={"merge_method": "copy_best_parent", "base_parent": base_parent, "created_via": "evomerge"},
        )

        self.model_manager.models[offspring_id] = offspring_info
        self.model_manager._save_registry()

        logger.info(f"Created merged model {offspring_id} from {parent1} + {parent2}")
        return str(offspring_path)

    async def _select_best_models(self, models: list[str], top_k: int = 3) -> list[str]:
        """Select the best performing models."""

        model_scores = []
        for model_id in models:
            model_info = self.model_manager.get_model_info(model_id)
            if model_info and model_info.performance_score:
                model_scores.append((model_id, model_info.performance_score))

        # Sort by performance (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k models
        best_models = [model_id for model_id, _ in model_scores[:top_k]]
        return best_models

    async def _cleanup_generation(self, generation: int):
        """Cleanup models from a specific generation to save space."""

        if generation not in self.generation_history:
            return

        models_to_remove = self.generation_history[generation]

        for model_id in models_to_remove:
            if model_id in self.model_manager.models:
                model_info = self.model_manager.models[model_id]

                # Don't remove if it's a top performer
                if model_info.performance_score and model_info.performance_score > 1.0:
                    continue

                try:
                    # Remove from disk
                    model_path = Path(model_info.local_path)
                    if model_path.exists():
                        shutil.rmtree(model_path, ignore_errors=True)

                    # Remove from registry
                    del self.model_manager.models[model_id]

                    logger.info(f"Cleaned up generation {generation} model: {model_id}")

                except Exception as e:
                    logger.error(f"Failed to cleanup model {model_id}: {e}")

        self.model_manager._save_registry()


# Default seed models configuration
SEED_MODELS = [
    {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "purpose": "coding_specialist",
        "description": "Instruction-tuned coding model with 32K context",
    },
    {
        "model_id": "Qwen/Qwen2-1.5B",
        "purpose": "general_coding",
        "description": "General LLM with strong coding capabilities",
    },
    {
        "model_id": "microsoft/phi-1_5",
        "purpose": "efficient_coding",
        "description": "Compact model with strong Python capabilities",
    },
]


async def main():
    """Main function to test the model management system."""

    # Initialize W&B
    os.environ["WANDB_PROJECT"] = "agent-forge-models"

    # Create model manager
    model_manager = ModelManager()

    # Download seed models
    logger.info("Starting Agent Forge model pipeline...")
    downloaded_models = await model_manager.download_seed_models(SEED_MODELS)

    if not downloaded_models:
        logger.error("No models downloaded successfully")
        return

    # Start EvoMerge process
    evomerge_manager = EvoMergeManager(model_manager)
    best_models = await evomerge_manager.start_evomerge_process(seed_models=downloaded_models, max_generations=3)

    # Display results
    logger.info("=== Agent Forge Pipeline Complete ===")
    logger.info(f"Best evolved models: {best_models}")

    for model_id in best_models:
        model_info = model_manager.get_model_info(model_id)
        if model_info:
            logger.info(f"Model: {model_id}")
            logger.info(f"  Performance: {model_info.performance_score:.3f}")
            logger.info(f"  Generation: {model_info.generation}")
            logger.info(f"  Parents: {model_info.parent_models}")
            logger.info(f"  Size: {model_info.size_gb:.2f} GB")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
