import logging
import math
from typing import Any

import torch
from torch import nn
from transformers import AutoTokenizer

from AIVillage.experimental.training.geometry.snapshot import snapshot
from AIVillage.experimental.training.meta.geo2z_policy import Geo2Z, Replay
from AIVillage.experimental.training.optim.augmented_adam import AugmentedAdam
from AIVillage.experimental.training.training.curriculum import (
    CurriculumGenerator,
    CurriculumLevel,
)
from AIVillage.experimental.training.training.pid_edgechaos import EdgePID
from AIVillage.experimental.training.training.quiet_star import QuietSTaRModel
from AIVillage.experimental.training.training.svf_ops import apply_svf

logger = logging.getLogger(__name__)


class AgentForgeTrainingLoop:
    """Enhanced training loop with Quiet-STaR integration and curriculum learning."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        enable_quiet_star: bool = False,
        curriculum_domain: str = "general",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.enable_quiet_star = enable_quiet_star

        # Initialize Quiet-STaR if enabled
        if enable_quiet_star:
            self.quiet_star_model = QuietSTaRModel(model)
            self.model = self.quiet_star_model
            logger.info("Quiet-STaR integration enabled")

        # Initialize curriculum generator
        self.curriculum = CurriculumGenerator(
            frontier_model="microsoft/DialoGPT-small",  # Fallback model
            domain=curriculum_domain,
        )

        # Initialize training components
        self.optimizer = AugmentedAdam(self.model.parameters(), lr=2e-5)
        self.pid = EdgePID()
        self.geo2z = Geo2Z()
        self.replay = Replay(50000)
        self.state = {"G": None, "pre_grok": False, "G_prev": None}

        # Training statistics
        self.level_accuracy = {}
        self.training_metrics = {
            "steps": 0,
            "total_reward": 0.0,
            "grok_events": 0,
            "quiet_star_activations": 0,
        }

    def generate_curriculum_level(
        self, level: int, num_tasks: int = 100
    ) -> CurriculumLevel:
        """Generate curriculum for a specific level."""
        logger.info(f"Generating curriculum level {level} with {num_tasks} tasks")

        # Generate assessment questions for this level
        questions = self.curriculum.create_assessment_questions(num_tasks)

        # Filter questions by difficulty level
        level_questions = [q for q in questions if q.difficulty == level]

        # Create curriculum level
        curriculum_level = CurriculumLevel(
            level=level,
            difficulty=level,
            organic_data=[q.text for q in level_questions[: num_tasks // 4]],
            synthetic_data=[
                q.text for q in level_questions[num_tasks // 4 : num_tasks // 2]
            ],
            rag_data=[
                q.text for q in level_questions[num_tasks // 2 : 3 * num_tasks // 4]
            ],
            interaction_data=[q.text for q in level_questions[3 * num_tasks // 4 :]],
        )

        return curriculum_level

    def process_quiet_star_thoughts(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Process input through Quiet-STaR for thought generation."""
        if not self.enable_quiet_star:
            return self.model(input_ids, attention_mask=attention_mask), None

        # Generate thoughts using Quiet-STaR
        logits, thought_logits = self.quiet_star_model(
            input_ids, attention_mask=attention_mask, generate_thoughts=True
        )

        self.training_metrics["quiet_star_activations"] += 1

        return logits, thought_logits

    def calculate_reward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        step: int,
        thought_logits: torch.Tensor | None = None,
    ) -> float:
        """Calculate training reward with optional thought penalty."""
        # Base task reward
        task_accuracy = (logits.argmax(-1) == target).float().mean().item()

        # Gradient flow reward
        gslow = (
            self.optimizer._grad_window.abs().mean().item()
            if self.optimizer._grad_window is not None
            else 0.0
        )

        # Geometry reward (intrinsic dimensionality improvement)
        geom_reward = 0.0
        if step > 0 and self.state["G_prev"] is not None:
            geom_reward = max(
                0, self.state["G_prev"]["ID_nl"] - self.state["G"]["ID_nl"]
            )

        # Thought coherence reward (if Quiet-STaR is enabled)
        thought_reward = 0.0
        if thought_logits is not None:
            # Reward coherent thought generation
            thought_entropy = (
                torch.distributions.Categorical(logits=thought_logits).entropy().mean()
            )
            thought_reward = 0.1 * (1.0 - torch.clamp(thought_entropy, 0, 1))

        # Combined reward
        reward = (
            0.3 * task_accuracy
            + 0.4 * math.tanh(gslow)
            + 0.2 * geom_reward
            + 0.1 * thought_reward
        )

        return reward

    def run_level(
        self, curriculum_level: CurriculumLevel, max_steps: int = 1000
    ) -> dict[str, Any]:
        """Run training for a specific curriculum level."""
        logger.info(f"Starting training for level {curriculum_level.level}")

        # Prepare dataset from curriculum level
        all_data = (
            curriculum_level.organic_data
            + curriculum_level.synthetic_data
            + curriculum_level.rag_data
            + curriculum_level.interaction_data
        )

        level_metrics = {
            "level": curriculum_level.level,
            "steps": 0,
            "accuracy": 0.0,
            "reward": 0.0,
            "grok_detected": False,
            "quiet_star_thoughts": 0,
        }

        correct_predictions = 0
        total_predictions = 0

        for step, text in enumerate(all_data[:max_steps]):
            # Tokenize input
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

            # Create target (shift labels for language modeling)
            target = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

            if input_ids.size(1) == 0:
                continue

            # Forward pass with optional Quiet-STaR
            logits, thought_logits = self.process_quiet_star_thoughts(
                input_ids, attention_mask
            )

            # Calculate loss
            loss_task = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1)
            )
            loss_task.backward()

            # Geometry snapshot
            hidden_states = logits.detach()  # Use logits as proxy for hidden states
            self.state["G"] = snapshot(hidden_states.view(-1, hidden_states.size(-1)))

            # Calculate reward
            reward = self.calculate_reward(logits, target, step, thought_logits)

            # SVF action from meta-policy
            geom_vec = torch.tensor(
                [self.state["G"][k] for k in ["ID_nl", "ID_lin", "ratio", "entropy"]],
                device=logits.device,
            )
            z = self.geo2z(geom_vec)

            # Apply SVF to linear layers
            for m in self.model.modules():
                if isinstance(m, torch.nn.Linear):
                    apply_svf(m, z)

            # Grokfast filter & optimizer
            gslow = (
                self.optimizer._grad_window.abs().mean().item()
                if self.optimizer._grad_window is not None
                else 0.0
            )

            self.state["pre_grok"] = gslow > 0.03 and self.state["G"]["ratio"] < 0.1
            lr_gain = self.pid.update(self.state["G"]["ratio"])

            for group in self.optimizer.param_groups:
                group["lr"] *= 1 + lr_gain

            self.optimizer.step(amplify=self.state["pre_grok"])
            self.optimizer.zero_grad()

            # Replay buffer
            if reward > 0.2:
                self.replay.add(geom_vec.detach(), z.detach(), reward)

            # Grok detection
            if self.state["pre_grok"] and abs(self.state["G"]["ratio"] - 0.05) < 0.01:
                logger.info(f"Grok detected at step {step}!")
                level_metrics["grok_detected"] = True
                self.training_metrics["grok_events"] += 1
                break

            # Update metrics
            predictions = logits.argmax(-1)
            correct = (predictions == target).float().sum().item()
            total = target.numel()

            correct_predictions += correct
            total_predictions += total

            level_metrics["steps"] += 1
            level_metrics["reward"] += reward

            if thought_logits is not None:
                level_metrics["quiet_star_thoughts"] += 1

            # Store previous geometry state
            self.state["G_prev"] = self.state["G"]

            # Update global metrics
            self.training_metrics["steps"] += 1
            self.training_metrics["total_reward"] += reward

        # Calculate final level accuracy
        level_metrics["accuracy"] = (
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        )
        level_metrics["reward"] /= (
            level_metrics["steps"] if level_metrics["steps"] > 0 else 1
        )

        # Store level accuracy
        self.level_accuracy[curriculum_level.level] = level_metrics["accuracy"]

        logger.info(
            f"Level {curriculum_level.level} completed: "
            f"accuracy={level_metrics['accuracy']:.3f}, "
            f"reward={level_metrics['reward']:.3f}, "
            f"steps={level_metrics['steps']}"
        )

        return level_metrics

    def run_curriculum(
        self, max_levels: int = 10, tasks_per_level: int = 100
    ) -> dict[str, Any]:
        """Run complete curriculum training."""
        logger.info(f"Starting curriculum training with {max_levels} levels")

        curriculum_results = {
            "levels_completed": 0,
            "level_metrics": [],
            "overall_accuracy": 0.0,
            "total_steps": 0,
            "quiet_star_enabled": self.enable_quiet_star,
        }

        for level in range(1, max_levels + 1):
            # Generate curriculum level
            curriculum_level = self.generate_curriculum_level(level, tasks_per_level)

            # Train on this level
            level_metrics = self.run_level(curriculum_level)

            # Store results
            curriculum_results["level_metrics"].append(level_metrics)
            curriculum_results["levels_completed"] += 1
            curriculum_results["total_steps"] += level_metrics["steps"]

            # Early stopping if accuracy is very low
            if level_metrics["accuracy"] < 0.1 and level > 3:
                logger.warning(f"Low accuracy at level {level}, stopping curriculum")
                break

        # Calculate overall accuracy
        if curriculum_results["level_metrics"]:
            curriculum_results["overall_accuracy"] = sum(
                m["accuracy"] for m in curriculum_results["level_metrics"]
            ) / len(curriculum_results["level_metrics"])

        logger.info(
            f"Curriculum training completed: "
            f"levels={curriculum_results['levels_completed']}, "
            f"accuracy={curriculum_results['overall_accuracy']:.3f}, "
            f"steps={curriculum_results['total_steps']}"
        )

        return curriculum_results


# Legacy compatibility
def run_level(dataset) -> None:
    """Legacy function for backward compatibility."""
    logger.warning(
        "Using legacy run_level function. Consider migrating to AgentForgeTrainingLoop."
    )

    # Initialize basic components for legacy support
    global state, optimizer, pid, geo2z, replay, model

    for step, (prompt, target, _tag) in enumerate(dataset):
        # forward
        logits, H = model(prompt, return_h=True)
        loss_task = torch.nn.functional.cross_entropy(logits, target)
        loss_task.backward()

        # geometry
        state["G"] = snapshot(H.view(-1, H.size(-1)))
        gslow = (
            optimizer._grad_window.abs().mean().item()
            if optimizer._grad_window is not None
            else 0.0
        )

        # RL reward shaping
        reward = (
            0.3 * (logits.argmax(-1) == target).float().mean().item()
            + 0.5 * math.tanh(gslow)
            + 0.2 * max(0, state["G_prev"]["ID_nl"] - state["G"]["ID_nl"])
            if step
            else 0
        )

        # SVF action from meta-policy
        geom_vec = torch.tensor(
            [state["G"][k] for k in ["ID_nl", "ID_lin", "ratio", "entropy"]],
            device=H.device,
        )
        z = geo2z(geom_vec)
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                apply_svf(m, z)

        # Grokfast filter & optimiser
        state["pre_grok"] = gslow > 0.03 and state["G"]["ratio"] < 0.1
        lr_gain = pid.update(state["G"]["ratio"])
        for group in optimizer.param_groups:
            group["lr"] *= 1 + lr_gain
        optimizer.step(amplify=state["pre_grok"])
        optimizer.zero_grad()

        # replay
        if reward > 0.2:
            replay.add(geom_vec.detach(), z.detach(), reward)

        # grok detection
        if state["pre_grok"] and abs(state["G"]["ratio"] - 0.05) < 0.01:
            break

        state["G_prev"] = state["G"]


# Initialize global objects for legacy compatibility
try:
    optimizer = AugmentedAdam(model.parameters(), lr=2e-5)
    pid = EdgePID()
    geo2z = Geo2Z()
    replay = Replay(50000)
    state = {"G": None, "pre_grok": False, "G_prev": None}
except NameError:
    # Model not defined in global scope
    pass


def run_level(dataset) -> None:
    global state
    for step, (prompt, target, _tag) in enumerate(dataset):
        # forward
        logits, H = model(prompt, return_h=True)
        loss_task = torch.nn.functional.cross_entropy(logits, target)
        loss_task.backward()

        # geometry
        state["G"] = snapshot(H.view(-1, H.size(-1)))
        gslow = (
            optimizer._grad_window.abs().mean().item()
            if optimizer._grad_window is not None
            else 0.0
        )

        # RL reward shaping
        reward = (
            0.3 * (logits.argmax(-1) == target).float().mean().item()
            + 0.5 * math.tanh(gslow)
            + 0.2 * max(0, state["G_prev"]["ID_nl"] - state["G"]["ID_nl"])
            if step
            else 0
        )

        # SVF action from meta-policy
        geom_vec = torch.tensor(
            [state["G"][k] for k in ["ID_nl", "ID_lin", "ratio", "entropy"]],
            device=H.device,
        )
        z = geo2z(geom_vec)
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                apply_svf(m, z)

        # Grokfast filter & optimiser
        state["pre_grok"] = gslow > 0.03 and state["G"]["ratio"] < 0.1
        lr_gain = pid.update(state["G"]["ratio"])
        for group in optimizer.param_groups:
            group["lr"] *= 1 + lr_gain
        optimizer.step(amplify=state["pre_grok"])
        optimizer.zero_grad()

        # replay
        if reward > 0.2:
            replay.add(geom_vec.detach(), z.detach(), reward)

        # grok detection
        if state["pre_grok"] and abs(state["G"]["ratio"] - 0.05) < 0.01:
            break

        state["G_prev"] = state["G"]
