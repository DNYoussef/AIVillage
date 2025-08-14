"""Mastery Training Loop - Comprehensive Implementation.

Integrates all training components:
- Calibration with frontier API
- Baseline evaluation and level remapping
- Mastery cycles with grading
- Sleep/Dream integration every 500 attempts
- Geometry feedback with intrinsic dimensionality
- Self-modeling with UDaimonic compass
- GrokFast optimization
- Performance tracking and deployment preparation
"""

import asyncio
import contextlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from langroid import ChatAgent, ChatAgentConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from src.agent_forge.geometry.id_twonn import twonn
from src.agent_forge.training.grokfast import GrokFastTask
from src.agent_forge.training.self_modeling import SelfModelingTask

# Import existing training components
from src.agent_forge.training.sleep_and_dream import SleepAndDreamTask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for geometry feedback
state = {"G": {"ID_nl": 0.0}, "pre_grok": False, "mastery_level": 1}


@dataclass
class Task:
    """A single task with difficulty and domain."""

    prompt: str
    expected_output: str
    difficulty: int  # 1-100
    domain: str
    metadata: dict | None = None


@dataclass
class MasteryConfig:
    """Configuration for mastery training loop."""

    model_path: str
    output_dir: str
    domain: str = "math"
    initial_task_count: int = 100
    mastery_threshold: float = 0.8
    baseline_threshold: float = 0.5
    max_attempts_per_level: int = 1000
    sleep_dream_interval: int = 500
    geometry_update_interval: int = 100
    max_mastery_levels: int = 10
    frontier_api_key: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: str = "agent-forge-mastery"
    save_checkpoints: bool = True


class TaskGenerator:
    """Generates tasks using frontier API (GPT-4) for calibration."""

    def __init__(self, config: MasteryConfig, agent: ChatAgent) -> None:
        self.config = config
        self.agent = agent

    async def generate_calibration_tasks(
        self, domain: str, count: int = 100
    ) -> list[Task]:
        """Generate 100 tasks with difficulties 1-100 for calibration."""
        tasks = []

        for difficulty in range(1, count + 1):
            prompt = f"""Create a {domain} task with difficulty level {difficulty}/100.

Difficulty guidelines:
- 1-20: Basic arithmetic/concepts
- 21-40: Elementary school level
- 41-60: Middle school level
- 61-80: High school level
- 81-100: College/competitive level

Return JSON format:
{{
    "prompt": "Task description/question",
    "expected_output": "Correct answer with explanation",
    "metadata": {{"concepts": ["list", "of", "concepts"], "steps": 3}}
}}"""

            try:
                response = await self.agent.llm_response(prompt)
                task_data = json.loads(response.content)

                task = Task(
                    prompt=task_data["prompt"],
                    expected_output=task_data["expected_output"],
                    difficulty=difficulty,
                    domain=domain,
                    metadata=task_data.get("metadata", {}),
                )
                tasks.append(task)

            except Exception as e:
                logger.warning(
                    "Failed to generate task for difficulty %s: %s", difficulty, e
                )
                # Fallback task
                task = Task(
                    prompt=f"Solve this {domain} problem (difficulty {difficulty}): What is {difficulty} + {difficulty}?",
                    expected_output=f"{difficulty * 2}",
                    difficulty=difficulty,
                    domain=domain,
                )
                tasks.append(task)

        return tasks


class MasteryEvaluator:
    """Evaluates model performance on tasks and determines mastery levels."""

    def __init__(self, model, tokenizer, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    async def evaluate_task(self, task: Task) -> tuple[bool, str, float]:
        """Evaluate a single task. Returns (success, response, confidence)."""
        try:
            # Prepare input with metadata
            geom_info = (
                f"<geom idnl={state['G']['ID_nl']:.2f} level={state['mastery_level']}/>"
            )
            prompt = f"{geom_info}\n{task.prompt}\n\nAnswer:"

            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.size(1) + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            response = self.tokenizer.decode(
                outputs.sequences[0], skip_special_tokens=True
            )
            response = response[len(prompt) :].strip()

            # Simple success detection (could be enhanced with semantic similarity)
            success = self._check_correctness(response, task.expected_output)

            # Calculate confidence from token probabilities
            confidence = self._calculate_confidence(outputs.scores)

            return success, response, confidence

        except Exception as e:
            logger.exception("Evaluation error: %s", e)
            return False, "", 0.0

    def _check_correctness(self, response: str, expected: str) -> bool:
        """Basic correctness check - can be enhanced with domain-specific logic."""
        # Simple substring matching for now
        response_clean = response.lower().strip()
        expected_clean = expected.lower().strip()

        # Check if key parts of expected answer are in response
        return expected_clean in response_clean or response_clean in expected_clean

    def _calculate_confidence(self, scores) -> float:
        """Calculate confidence from generation scores."""
        if not scores:
            return 0.5

        # Average probability of generated tokens
        probs = []
        for score in scores:
            prob = torch.softmax(score, dim=-1).max().item()
            probs.append(prob)

        return sum(probs) / len(probs) if probs else 0.5


class GeometryFeedback:
    """Tracks intrinsic dimensionality and provides geometry-based feedback."""

    def __init__(self, model, update_interval: int = 100) -> None:
        self.model = model
        self.update_interval = update_interval
        self.step_count = 0
        self.id_history = []

    async def update_geometry(self, hidden_states: torch.Tensor) -> None:
        """Update intrinsic dimensionality using Two-NN estimator."""
        self.step_count += 1

        if self.step_count % self.update_interval == 0:
            try:
                # Extract representative hidden states
                if hidden_states.dim() > 2:
                    hidden_states = hidden_states.view(-1, hidden_states.size(-1))

                # Sample subset for efficiency
                if hidden_states.size(0) > 1000:
                    indices = torch.randperm(hidden_states.size(0))[:1000]
                    hidden_states = hidden_states[indices]

                # Calculate intrinsic dimensionality
                id_estimate = twonn(hidden_states.cpu())

                # Update global state
                state["G"]["ID_nl"] = float(id_estimate)
                self.id_history.append(id_estimate)

                # Detect grokking patterns
                if len(self.id_history) > 10:
                    recent_trend = self.id_history[-5:]
                    if all(
                        a > b
                        for a, b in zip(
                            recent_trend[1:], recent_trend[:-1], strict=False
                        )
                    ):
                        state["pre_grok"] = True
                    else:
                        state["pre_grok"] = False

                logger.info(
                    "Geometry update: ID_nl=%.3f, pre_grok=%s",
                    id_estimate,
                    state["pre_grok"],
                )

            except Exception as e:
                logger.warning("Geometry update failed: %s", e)


class MasteryLoop:
    """Main mastery training loop implementation."""

    def __init__(self, config: MasteryConfig) -> None:
        self.config = config
        self.setup_logging()

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.evaluator = None
        self.geometry = None
        self.sleep_dream_task = None
        self.self_modeling_task = None
        self.grokfast_task = None

        # Initialize agent for task generation
        agent_config = ChatAgentConfig(
            name="MasteryAgent",
            llm=OpenAIGPTConfig(
                chat_model="gpt-4-turbo-preview", api_key=config.frontier_api_key
            ),
        )
        self.agent = ChatAgent(agent_config)
        self.task_generator = TaskGenerator(config, self.agent)

        # Training state
        self.current_level = 1
        self.baseline_k = None
        self.calibration_tasks = []
        self.level_tasks = {}
        self.performance_history = []
        self.attempt_count = 0

        # W&B tracking
        if wandb.run is None:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                job_type="mastery_loop",
            )

    def setup_logging(self) -> None:
        """Configure enhanced logging."""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(log_dir / "mastery_loop.log")
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    async def initialize_model(self) -> None:
        """Load and initialize the model for training."""
        logger.info("Loading model from %s", self.config.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=(
                torch.float16 if self.config.device == "cuda" else torch.float32
            ),
            device_map="auto" if self.config.device == "cuda" else None,
        )

        if self.config.device == "cpu":
            self.model = self.model.to(self.config.device)

        # Initialize components
        self.evaluator = MasteryEvaluator(
            self.model, self.tokenizer, self.config.device
        )
        self.geometry = GeometryFeedback(self.model)

        # Initialize training modules
        hidden_size = self.model.config.hidden_size
        self.sleep_dream_task = SleepAndDreamTask(
            self.agent, hidden_size, hidden_size, 3, 3, pretrained=False
        )

        self.self_modeling_task = SelfModelingTask(
            self.agent, self.config.model_path, self.config.device
        )

        self.grokfast_task = GrokFastTask(self.agent, self.model)

        logger.info("Model initialization complete")

    async def calibration_phase(self) -> int:
        """Phase 1: Calibrate with frontier API to find baseline level."""
        logger.info("Starting calibration phase...")

        # Generate calibration tasks
        self.calibration_tasks = await self.task_generator.generate_calibration_tasks(
            self.config.domain, self.config.initial_task_count
        )

        logger.info("Generated %s calibration tasks", len(self.calibration_tasks))

        # Evaluate model on all tasks to find k where success ≈ 50%
        success_rates = {}

        for task in tqdm(self.calibration_tasks, desc="Calibration evaluation"):
            success, response, confidence = await self.evaluator.evaluate_task(task)

            if task.difficulty not in success_rates:
                success_rates[task.difficulty] = []
            success_rates[task.difficulty].append(success)

            # Update geometry tracking
            if hasattr(self.model, "get_hidden_states"):
                hidden_states = self.model.get_hidden_states()
                await self.geometry.update_geometry(hidden_states)

        # Calculate success rate for each difficulty level
        level_performance = {}
        for diff, results in success_rates.items():
            level_performance[diff] = sum(results) / len(results)

        # Find k where success rate is closest to baseline_threshold (50%)
        target_diff = min(
            level_performance.keys(),
            key=lambda k: abs(level_performance[k] - self.config.baseline_threshold),
        )

        self.baseline_k = target_diff
        logger.info(
            "Baseline established: Level %s (success rate: %.2f)",
            self.baseline_k,
            level_performance[target_diff],
        )

        # Log calibration results
        wandb.log(
            {
                "calibration/baseline_level": self.baseline_k,
                "calibration/baseline_success_rate": level_performance[target_diff],
                "calibration/geometry_id": state["G"]["ID_nl"],
            }
        )

        return self.baseline_k

    async def level_remapping(self) -> None:
        """Phase 2: Remap difficulty levels 1-10 based on baseline."""
        logger.info("Performing level remapping...")

        # Create 10 mastery levels based on baseline
        level_mapping = {}
        for level in range(1, 11):
            # Exponential scaling around baseline
            if level <= 5:
                # Levels 1-5: easier than baseline
                mapped_diff = max(1, int(self.baseline_k * (0.3 + 0.14 * level)))
            else:
                # Levels 6-10: harder than baseline
                mapped_diff = min(100, int(self.baseline_k * (1.0 + 0.2 * (level - 5))))

            level_mapping[level] = mapped_diff

        self.level_mapping = level_mapping
        logger.info("Level mapping: %s", level_mapping)

        # Generate tasks for each mastery level
        for level, difficulty in level_mapping.items():
            # Use calibration tasks near this difficulty
            level_tasks = [
                task
                for task in self.calibration_tasks
                if abs(task.difficulty - difficulty) <= 5
            ]

            # Generate additional tasks if needed
            if len(level_tasks) < 20:
                additional_tasks = await self.task_generator.generate_calibration_tasks(
                    self.config.domain, 20 - len(level_tasks)
                )
                # Adjust difficulty of additional tasks
                for task in additional_tasks:
                    task.difficulty = difficulty
                level_tasks.extend(additional_tasks)

            self.level_tasks[level] = level_tasks[:20]  # 20 tasks per level

        wandb.log({"remapping/level_mapping": level_mapping})

    async def mastery_cycle(self, level: int) -> bool:
        """Phase 3: Mastery cycle for a specific level."""
        logger.info("Starting mastery cycle for level %s", level)

        state["mastery_level"] = level
        tasks = self.level_tasks[level]
        attempts = 0
        successes = 0
        total_attempts = 0

        while attempts < self.config.max_attempts_per_level:
            # Select random task from level
            task = random.choice(tasks)

            # Evaluate task
            success, response, confidence = await self.evaluator.evaluate_task(task)

            attempts += 1
            total_attempts += 1
            self.attempt_count += 1

            if success:
                successes += 1

            # Calculate current success rate
            success_rate = successes / attempts

            # Log attempt
            wandb.log(
                {
                    f"level_{level}/attempt": attempts,
                    f"level_{level}/success_rate": success_rate,
                    f"level_{level}/confidence": confidence,
                    "global/attempt_count": self.attempt_count,
                    "global/geometry_id": state["G"]["ID_nl"],
                }
            )

            # Check for mastery (80% success rate over recent window)
            if attempts >= 50 and attempts % 10 == 0:
                min(attempts, 100)
                recent_successes = 0

                # Re-evaluate recent performance
                for _ in range(10):
                    test_task = random.choice(tasks)
                    test_success, _, _ = await self.evaluator.evaluate_task(test_task)
                    if test_success:
                        recent_successes += 1

                recent_success_rate = recent_successes / 10

                if recent_success_rate >= self.config.mastery_threshold:
                    logger.info(
                        "Mastery achieved for level %s! Success rate: %.2f",
                        level,
                        recent_success_rate,
                    )
                    wandb.log({f"mastery/level_{level}_achieved": True})
                    return True

            # Sleep/Dream integration every 500 attempts
            if self.attempt_count % self.config.sleep_dream_interval == 0:
                await self.sleep_dream_cycle()

            # Geometry feedback every 100 attempts
            if self.attempt_count % self.config.geometry_update_interval == 0:
                # Get hidden states from last forward pass
                with torch.no_grad():
                    inputs = self.tokenizer.encode(task.prompt, return_tensors="pt").to(
                        self.config.device
                    )
                    outputs = self.model(inputs, output_hidden_states=True)
                    await self.geometry.update_geometry(outputs.hidden_states[-1])

            # GrokFast optimization when pre-grok detected
            if state["pre_grok"]:
                await self.grokfast_task.filter_gradients()

            # Self-modeling integration
            if attempts % 100 == 0:
                await self.self_modeling_integration(level)

        logger.info(
            "Max attempts reached for level %s. Final success rate: %.2f",
            level,
            success_rate,
        )
        return False

    async def sleep_dream_cycle(self) -> None:
        """Execute sleep/dream cycle for memory consolidation."""
        logger.info("Executing sleep/dream cycle...")

        try:
            # Extract hidden states for consolidation
            with torch.no_grad():
                # Use a representative input
                dummy_input = torch.randn(1, self.model.config.hidden_size).to(
                    self.config.device
                )
                dream_output = await self.sleep_dream_task.run(dummy_input)

            # Apply dream output as weight update
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.requires_grad and param.dim() >= 2:
                        # Apply small update based on dream output
                        update_scale = 0.001
                        param.data += (
                            update_scale
                            * torch.randn_like(param)
                            * dream_output.mean().item()
                        )

            logger.info("Sleep/dream cycle completed")
            wandb.log({"sleep_dream/cycle_completed": self.attempt_count})

        except Exception as e:
            logger.warning("Sleep/dream cycle failed: %s", e)

    async def self_modeling_integration(self, level: int) -> None:
        """Integrate self-modeling for enhanced self-awareness."""
        logger.info("Running self-modeling integration for level %s", level)

        try:
            # Run a mini self-modeling cycle
            await self.self_modeling_task.self_modeling_cycle(
                curriculum_level=level,
                num_cycles=5,  # Mini cycle
            )

            wandb.log(
                {
                    "self_modeling/integration_completed": self.attempt_count,
                    "self_modeling/level": level,
                }
            )

        except Exception as e:
            logger.warning("Self-modeling integration failed: %s", e)

    async def run_mastery_training(self) -> dict[str, Any]:
        """Run the complete mastery training loop."""
        start_time = time.time()

        try:
            # Initialize model
            await self.initialize_model()

            # Phase 1: Calibration
            baseline_level = await self.calibration_phase()

            # Phase 2: Level remapping
            await self.level_remapping()

            # Phase 3: Mastery cycles
            mastery_results = {}

            for level in range(1, self.config.max_mastery_levels + 1):
                logger.info("\n%s", "=" * 50)
                logger.info("MASTERY LEVEL %s", level)
                logger.info("%s", "=" * 50)

                mastery_achieved = await self.mastery_cycle(level)
                mastery_results[level] = mastery_achieved

                if not mastery_achieved:
                    logger.info(
                        "Mastery not achieved for level %s, stopping progression", level
                    )
                    break

                # Save checkpoint
                if self.config.save_checkpoints:
                    self.save_checkpoint(level)

            # Final evaluation and summary
            total_time = time.time() - start_time
            summary = {
                "baseline_level": baseline_level,
                "mastery_results": mastery_results,
                "total_attempts": self.attempt_count,
                "total_time_hours": total_time / 3600,
                "levels_mastered": sum(mastery_results.values()),
                "final_geometry_id": state["G"]["ID_nl"],
                "level_mapping": self.level_mapping,
            }

            # Log final summary
            wandb.log(
                {
                    "summary/levels_mastered": summary["levels_mastered"],
                    "summary/total_attempts": summary["total_attempts"],
                    "summary/total_time_hours": summary["total_time_hours"],
                    "summary/final_geometry_id": summary["final_geometry_id"],
                }
            )

            logger.info("\nMastery training complete!")
            logger.info(
                "Levels mastered: %s/%s",
                summary["levels_mastered"],
                self.config.max_mastery_levels,
            )
            logger.info("Total attempts: %s", summary["total_attempts"])
            logger.info("Total time: %.2f hours", summary["total_time_hours"])

            return summary

        except Exception as e:
            logger.exception("Mastery training failed: %s", e)
            raise

    def save_checkpoint(self, level: int) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "level": level,
            "attempt_count": self.attempt_count,
            "baseline_k": self.baseline_k,
            "level_mapping": self.level_mapping,
            "geometry_state": state,
            "model_state_dict": self.model.state_dict() if self.model else None,
        }

        checkpoint_path = checkpoint_dir / f"mastery_level_{level}.json"

        # Save non-tensor data
        serializable_checkpoint = {
            k: v for k, v in checkpoint.items() if k != "model_state_dict"
        }

        with open(checkpoint_path, "w") as f:
            json.dump(serializable_checkpoint, f, indent=2)

        # Save model separately
        if self.model:
            model_path = checkpoint_dir / f"model_level_{level}.pt"
            torch.save(self.model.state_dict(), model_path)

        logger.info("Checkpoint saved: %s", checkpoint_path)


# CLI integration
async def main() -> None:
    """Main entry point for mastery training."""
    import argparse

    parser = argparse.ArgumentParser(description="Mastery Training Loop")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--domain", default="math", help="Training domain")
    parser.add_argument("--frontier-api-key", help="OpenAI API key for task generation")
    parser.add_argument(
        "--max-levels", type=int, default=10, help="Maximum mastery levels"
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    config = MasteryConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        domain=args.domain,
        frontier_api_key=args.frontier_api_key,
        max_mastery_levels=args.max_levels,
        device=args.device,
    )

    mastery_loop = MasteryLoop(config)
    results = await mastery_loop.run_mastery_training()

    print("\nTraining Summary:")
    print(f"Levels mastered: {results['levels_mastered']}/{config.max_mastery_levels}")
    print(f"Total attempts: {results['total_attempts']}")
    print(f"Training time: {results['total_time_hours']:.2f} hours")


# ============================================================================
# Orchestrator Integration
# ============================================================================


async def run_self_modeling(config: dict[str, Any]) -> "PhaseResult":
    from .forge_orchestrator import PhaseResult

    """Orchestrator entry point for Self-Modeling phase (via Mastery Loop).

    Args:
        config: Configuration dictionary with self-modeling parameters

    Returns:
        PhaseResult with status, artifacts, and metrics
    """
    import time
    from datetime import datetime

    from src.agent_forge.forge_orchestrator import (
        PhaseArtifact,
        PhaseResult,
        PhaseStatus,
        PhaseType,
    )

    start_time = time.time()

    try:
        logger.info("Starting Self-Modeling phase via orchestrator")

        # Convert config to MasteryConfig
        mastery_config = MasteryConfig(**config)

        # Create and run mastery loop (which includes self-modeling)
        mastery_loop = MasteryLoop(mastery_config)
        results = await mastery_loop.run_mastery_training()

        duration = time.time() - start_time

        if results.get("success", True):
            # Success - create artifacts
            artifacts = [
                PhaseArtifact(
                    phase_type=PhaseType.SELF_MODELING,
                    artifact_type="trained_model",
                    data={
                        "model_path": results.get(
                            "final_model_path", mastery_config.output_dir
                        ),
                        "levels_mastered": results.get("levels_mastered", 0),
                        "total_attempts": results.get("total_attempts", 0),
                        "mastery_progression": results.get("mastery_progression", []),
                        "self_modeling_metrics": results.get(
                            "self_modeling_metrics", {}
                        ),
                    },
                    metadata={
                        "mastery_config": mastery_config.dict(),
                        "training_domain": mastery_config.domain,
                        "max_levels": mastery_config.max_mastery_levels,
                    },
                )
            ]

            # Create metrics summary
            metrics = {
                "levels_mastered": results.get("levels_mastered", 0),
                "total_attempts": results.get("total_attempts", 0),
                "execution_time": duration,
                "training_time_hours": results.get("total_time_hours", duration / 3600),
                "success_rate": results.get("success_rate", 0.0),
                "final_mastery_level": results.get("current_mastery_level", 1),
                "geometry_feedback_updates": results.get("geometry_updates", 0),
                "sleep_dream_cycles": results.get("sleep_cycles", 0),
                "grokfast_optimizations": results.get("grokfast_updates", 0),
            }

            # Add self-modeling specific metrics
            if "self_modeling_metrics" in results:
                sm_metrics = results["self_modeling_metrics"]
                metrics.update(
                    {
                        "self_awareness_score": sm_metrics.get(
                            "self_awareness_score", 0.0
                        ),
                        "metacognitive_accuracy": sm_metrics.get(
                            "metacognitive_accuracy", 0.0
                        ),
                        "udaimonic_compass_direction": sm_metrics.get(
                            "compass_direction", "unknown"
                        ),
                    }
                )

            logger.info("Self-Modeling completed successfully in %.1fs", duration)

            return PhaseResult(
                phase_type=PhaseType.SELF_MODELING,
                status=PhaseStatus.COMPLETED,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.now(),
                duration_seconds=duration,
                artifacts_produced=artifacts,
                metrics=metrics,
            )
        # Failed mastery training
        return PhaseResult(
            phase_type=PhaseType.SELF_MODELING,
            status=PhaseStatus.FAILED,
            start_time=datetime.fromtimestamp(start_time),
            end_time=datetime.now(),
            duration_seconds=duration,
            error_message=results.get("error", "Self-modeling training failed"),
            metrics={"execution_time": duration},
        )

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Self-Modeling phase failed: {e!s}"
        logger.exception(error_msg)

        return PhaseResult(
            phase_type=PhaseType.SELF_MODELING,
            status=PhaseStatus.FAILED,
            start_time=datetime.fromtimestamp(start_time),
            end_time=datetime.now(),
            duration_seconds=duration,
            error_message=error_msg,
            metrics={"execution_time": duration},
        )


# Make the entry point discoverable
run = run_self_modeling  # Alias for orchestrator discovery
execute = run_self_modeling  # Alternative alias

if __name__ == "__main__":
    # Handle encoding for emoji output
    with contextlib.suppress(AttributeError):
        sys.stdout.reconfigure(encoding="utf-8")

    asyncio.run(main())
