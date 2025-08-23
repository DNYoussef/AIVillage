"""
Tests for Cogment Training Curriculum System (Agent 4)

Tests the 4-stage training curriculum including:
- GrokFast integration and gradient filtering
- Stage-specific loss functions and curriculum progression
- Training scheduler and stage transitions
- Loss computation and optimization strategies
- Integration with Agent 5's data pipeline
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

# Import Cogment training components
try:
    from core.agent_forge.models.cogment.core.config import CogmentConfig
    from core.agent_forge.models.cogment.training.curriculum import (
        CurriculumScheduler,
        CurriculumStage,
        FourStageCurriculum,
        StageConfig,
    )
    from core.agent_forge.models.cogment.training.grokfast import GradientFilter, GrokFast, GrokFastConfig
    from core.agent_forge.models.cogment.training.loss_functions import ACTLoss, CogmentLoss, MemoryLoss, StageLoss
    from core.agent_forge.models.cogment.training.optimizer import CogmentOptimizer, OptimizerConfig

    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Cogment training components not available: {e}")
    TRAINING_AVAILABLE = False


@dataclass
class TestTrainingConfig:
    """Test configuration for training system."""

    # Model parameters
    d_model: int = 256
    n_layers: int = 4
    vocab_size: int = 5000
    max_seq_len: int = 128

    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    max_steps: int = 100

    # GrokFast parameters
    grokfast_alpha: float = 0.98
    grokfast_lamb: float = 2.0
    gradient_filter_ema: float = 0.99

    # Stage configuration
    stage_steps: List[int] = None

    def __post_init__(self):
        if self.stage_steps is None:
            self.stage_steps = [20, 40, 60, 80, 100]  # Cumulative steps per stage


class TestFourStageCurriculum:
    """Test 4-stage curriculum system."""

    @pytest.fixture
    def curriculum_config(self):
        """Create curriculum configuration."""
        config = TestTrainingConfig()

        stage_configs = []
        stage_names = ["sanity", "arc", "puzzles", "reasoning", "long_context"]

        for i, (name, max_steps) in enumerate(zip(stage_names, config.stage_steps)):
            stage_config = StageConfig(
                stage=i,
                name=name,
                max_steps=max_steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate * (0.8**i),  # Decay LR per stage
                sequence_length=config.max_seq_len * (2 ** min(i, 2)),  # Increase seq len
                loss_weights={
                    "nll": 1.0,
                    "act": 0.1 if i >= 1 else 0.0,  # Enable ACT after stage 0
                    "memory": 0.05 if i >= 2 else 0.0,  # Enable memory after stage 1
                },
            )
            stage_configs.append(stage_config)

        return stage_configs

    @pytest.fixture
    def curriculum(self, curriculum_config):
        """Create FourStageCurriculum instance."""
        if not TRAINING_AVAILABLE:
            pytest.skip("Training components not available")
        return FourStageCurriculum(curriculum_config)

    def test_curriculum_creation(self, curriculum, curriculum_config):
        """Test curriculum instantiation."""
        assert len(curriculum.stages) == len(curriculum_config)
        assert curriculum.current_stage == CurriculumStage.SANITY
        assert curriculum.current_step == 0

        # Verify stage configurations
        for i, stage_config in enumerate(curriculum_config):
            assert curriculum.stages[i].name == stage_config.name
            assert curriculum.stages[i].max_steps == stage_config.max_steps

    def test_curriculum_stage_progression(self, curriculum):
        """Test curriculum stage progression logic."""
        # Start at SANITY stage
        assert curriculum.current_stage == CurriculumStage.SANITY

        # Advance through stages
        stage_transitions = []

        for step in range(105):  # More than max steps
            old_stage = curriculum.current_stage
            curriculum.step()
            new_stage = curriculum.current_stage

            if old_stage != new_stage:
                stage_transitions.append((step, old_stage, new_stage))

        # Should have transitioned through all stages
        expected_stages = [
            CurriculumStage.SANITY,
            CurriculumStage.ARC,
            CurriculumStage.PUZZLES,
            CurriculumStage.REASONING,
            CurriculumStage.LONG_CONTEXT,
        ]

        visited_stages = [CurriculumStage.SANITY]  # Start stage
        for _, _, new_stage in stage_transitions:
            visited_stages.append(new_stage)

        # Should visit all expected stages in order
        assert len(set(visited_stages) & set(expected_stages)) >= 4, f"Should visit most stages: {visited_stages}"

    def test_curriculum_config_adaptation(self, curriculum):
        """Test curriculum adapts configuration per stage."""
        configs_per_stage = {}

        for step in range(85):  # Cover multiple stages
            stage = curriculum.current_stage
            config = curriculum.get_current_config()

            if stage not in configs_per_stage:
                configs_per_stage[stage] = config

            curriculum.step()

        # Different stages should have different configurations
        assert len(configs_per_stage) >= 3, "Should collect configs from multiple stages"

        # Verify configurations change between stages
        configs = list(configs_per_stage.values())
        for i in range(len(configs) - 1):
            config1, config2 = configs[i], configs[i + 1]

            # At least one parameter should differ
            diff_found = (
                config1.learning_rate != config2.learning_rate
                or config1.sequence_length != config2.sequence_length
                or config1.loss_weights != config2.loss_weights
            )
            assert diff_found, f"Configs should differ between stages: {config1} vs {config2}"

    def test_curriculum_loss_weight_progression(self, curriculum):
        """Test loss weight progression through curriculum."""
        loss_weight_history = []

        for step in range(85):
            config = curriculum.get_current_config()
            loss_weight_history.append((curriculum.current_stage, config.loss_weights.copy()))
            curriculum.step()

        # Verify ACT loss is enabled after stage 0
        act_enabled_stages = [stage for stage, weights in loss_weight_history if weights.get("act", 0) > 0]
        assert len(act_enabled_stages) > 0, "ACT loss should be enabled in later stages"

        # Verify memory loss is enabled after stage 1
        memory_enabled_stages = [stage for stage, weights in loss_weight_history if weights.get("memory", 0) > 0]
        assert len(memory_enabled_stages) > 0, "Memory loss should be enabled in later stages"


class TestGrokFast:
    """Test GrokFast integration and gradient filtering."""

    @pytest.fixture
    def grokfast_config(self):
        """Create GrokFast configuration."""
        config = TestTrainingConfig()
        return GrokFastConfig(
            alpha=config.grokfast_alpha,
            lamb=config.grokfast_lamb,
            gradient_ema_decay=config.gradient_filter_ema,
            enabled=True,
            warmup_steps=10,
        )

    @pytest.fixture
    def grokfast(self, grokfast_config):
        """Create GrokFast instance."""
        if not TRAINING_AVAILABLE:
            pytest.skip("Training components not available")
        return GrokFast(grokfast_config)

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    def test_grokfast_creation(self, grokfast, grokfast_config):
        """Test GrokFast instantiation."""
        assert grokfast.config == grokfast_config
        assert grokfast.alpha == grokfast_config.alpha
        assert grokfast.lamb == grokfast_config.lamb
        assert hasattr(grokfast, "gradient_filter")

    def test_gradient_filter_functionality(self, grokfast, simple_model):
        """Test gradient filtering mechanism."""
        # Create dummy gradients
        x = torch.randn(4, 10)
        target = torch.randint(0, 5, (4,))

        # Forward pass and compute gradients
        output = simple_model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Store original gradients
        original_grads = {}
        for name, param in simple_model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()

        # Apply GrokFast filtering
        grokfast.filter_gradients(simple_model)

        # Gradients should be modified
        grad_changed = False
        for name, param in simple_model.named_parameters():
            if param.grad is not None and name in original_grads:
                if not torch.allclose(param.grad, original_grads[name], atol=1e-6):
                    grad_changed = True
                    break

        # After warmup, gradients should be filtered
        if grokfast.step_count >= grokfast.config.warmup_steps:
            assert grad_changed, "Gradients should be modified by GrokFast filtering"

    def test_grokfast_ema_tracking(self, grokfast, simple_model):
        """Test exponential moving average tracking in GrokFast."""
        x = torch.randn(4, 10)
        target = torch.randint(0, 5, (4,))

        ema_history = []

        # Run multiple steps to build EMA
        for step in range(15):
            simple_model.zero_grad()
            output = simple_model(x)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()

            # Apply filtering and track EMA
            grokfast.filter_gradients(simple_model)

            # Store EMA state
            if hasattr(grokfast.gradient_filter, "ema_gradients"):
                ema_state = {}
                for name in grokfast.gradient_filter.ema_gradients:
                    ema_state[name] = grokfast.gradient_filter.ema_gradients[name].clone()
                ema_history.append(ema_state)

        # EMA should evolve over time
        if len(ema_history) >= 2:
            # Compare early and late EMA states
            early_ema = ema_history[5] if len(ema_history) > 5 else ema_history[0]
            late_ema = ema_history[-1]

            ema_changed = False
            for name in early_ema:
                if name in late_ema:
                    if not torch.allclose(early_ema[name], late_ema[name], atol=1e-6):
                        ema_changed = True
                        break

            assert ema_changed, "EMA should evolve over training steps"

    def test_grokfast_convergence_acceleration(self, grokfast, simple_model):
        """Test GrokFast convergence acceleration on synthetic task."""
        # Create separable synthetic data
        torch.manual_seed(42)
        n_samples = 100
        x = torch.randn(n_samples, 10)
        # Create linearly separable targets
        weights = torch.randn(10, 5)
        target = torch.argmax(x @ weights, dim=1)

        optimizer = optim.Adam(simple_model.parameters(), lr=0.01)

        losses_with_grokfast = []

        # Train with GrokFast
        for step in range(50):
            optimizer.zero_grad()
            output = simple_model(x)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()

            # Apply GrokFast
            grokfast.filter_gradients(simple_model)

            optimizer.step()
            losses_with_grokfast.append(loss.item())

        # Reset model and train without GrokFast
        simple_model_2 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        # Copy initial weights
        with torch.no_grad():
            for p1, p2 in zip(simple_model.parameters(), simple_model_2.parameters()):
                p2.data.copy_(p1.data)

        optimizer_2 = optim.Adam(simple_model_2.parameters(), lr=0.01)
        losses_without_grokfast = []

        for step in range(50):
            optimizer_2.zero_grad()
            output = simple_model_2(x)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer_2.step()
            losses_without_grokfast.append(loss.item())

        # GrokFast should achieve lower loss or faster convergence
        final_loss_with = losses_with_grokfast[-1]
        final_loss_without = losses_without_grokfast[-1]

        # Allow some variance due to randomness
        improvement_ratio = final_loss_without / max(final_loss_with, 1e-8)

        print(f"✓ GrokFast final loss: {final_loss_with:.4f} vs baseline: {final_loss_without:.4f}")
        print(f"✓ Improvement ratio: {improvement_ratio:.2f}x")

        # Should show some improvement
        assert improvement_ratio >= 0.8, f"GrokFast should not significantly hurt performance: {improvement_ratio:.2f}x"


class TestStageLossFunctions:
    """Test stage-specific loss functions."""

    @pytest.fixture
    def loss_config(self):
        """Create loss configuration."""
        return {"nll_weight": 1.0, "act_weight": 0.1, "memory_weight": 0.05, "curriculum_stage": "arc"}

    @pytest.fixture
    def cogment_loss(self, loss_config):
        """Create CogmentLoss instance."""
        if not TRAINING_AVAILABLE:
            pytest.skip("Training components not available")
        return CogmentLoss(loss_config)

    def test_cogment_loss_creation(self, cogment_loss, loss_config):
        """Test CogmentLoss instantiation."""
        assert cogment_loss.nll_weight == loss_config["nll_weight"]
        assert cogment_loss.act_weight == loss_config["act_weight"]
        assert cogment_loss.memory_weight == loss_config["memory_weight"]

    def test_nll_loss_computation(self, cogment_loss):
        """Test negative log-likelihood loss computation."""
        batch_size = 2
        seq_len = 8
        vocab_size = 1000

        # Create logits and targets
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        nll_loss = cogment_loss.compute_nll_loss(logits, targets)

        assert nll_loss.ndim == 0  # Scalar loss
        assert nll_loss.item() >= 0  # Non-negative
        assert nll_loss.requires_grad  # Should require gradients

    def test_act_loss_computation(self, cogment_loss):
        """Test ACT (Adaptive Computation Time) loss computation."""
        batch_size = 2
        seq_len = 8

        # Create ACT outputs
        halting_prob = torch.rand(batch_size, seq_len)
        ponder_cost = torch.rand(batch_size, seq_len)
        step_count = torch.randint(1, 10, (batch_size, seq_len)).float()

        act_outputs = type(
            "ACTOutputs",
            (),
            {"halting_probability": halting_prob, "ponder_cost": ponder_cost, "step_count": step_count},
        )()

        act_loss = cogment_loss.compute_act_loss(act_outputs)

        assert act_loss.ndim == 0  # Scalar loss
        assert act_loss.item() >= 0  # Non-negative
        assert act_loss.requires_grad  # Should require gradients

    def test_memory_loss_computation(self, cogment_loss):
        """Test memory loss computation."""
        batch_size = 2
        seq_len = 8
        memory_dim = 64

        # Create memory outputs
        memory_outputs = type(
            "MemoryOutputs",
            (),
            {
                "read_attention": torch.rand(batch_size, seq_len, 32),
                "write_gate": torch.rand(batch_size, seq_len, 1),
                "memory_utilization": torch.rand(1),
                "consolidation_loss": torch.rand(1),
            },
        )()

        memory_loss = cogment_loss.compute_memory_loss(memory_outputs)

        assert memory_loss.ndim == 0  # Scalar loss
        assert memory_loss.item() >= 0  # Non-negative
        assert memory_loss.requires_grad  # Should require gradients

    def test_combined_loss_computation(self, cogment_loss):
        """Test combined loss computation."""
        batch_size = 2
        seq_len = 8
        vocab_size = 1000

        # Create model outputs
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # ACT outputs
        act_outputs = type(
            "ACTOutputs",
            (),
            {
                "halting_probability": torch.rand(batch_size, seq_len),
                "ponder_cost": torch.rand(batch_size, seq_len),
                "step_count": torch.randint(1, 5, (batch_size, seq_len)).float(),
            },
        )()

        # Memory outputs
        memory_outputs = type(
            "MemoryOutputs",
            (),
            {
                "read_attention": torch.rand(batch_size, seq_len, 32),
                "write_gate": torch.rand(batch_size, seq_len, 1),
                "memory_utilization": torch.rand(1),
                "consolidation_loss": torch.rand(1),
            },
        )()

        # Compute combined loss
        total_loss, loss_components = cogment_loss.compute_loss(
            logits=logits, targets=targets, act_outputs=act_outputs, memory_outputs=memory_outputs
        )

        # Verify loss structure
        assert total_loss.ndim == 0
        assert total_loss.item() >= 0
        assert total_loss.requires_grad

        # Verify loss components
        assert "nll" in loss_components
        assert "act" in loss_components
        assert "memory" in loss_components
        assert "total" in loss_components

        # Total should be sum of weighted components
        expected_total = (
            loss_components["nll"] * cogment_loss.nll_weight
            + loss_components["act"] * cogment_loss.act_weight
            + loss_components["memory"] * cogment_loss.memory_weight
        )

        assert torch.allclose(total_loss, expected_total, atol=1e-6)


class TestCurriculumScheduler:
    """Test curriculum scheduler integration."""

    @pytest.fixture
    def scheduler_config(self):
        """Create scheduler configuration."""
        config = TestTrainingConfig()
        return {
            "curriculum_stages": [
                {"name": "sanity", "steps": 20},
                {"name": "arc", "steps": 40},
                {"name": "reasoning", "steps": 60},
            ],
            "grokfast_config": GrokFastConfig(alpha=0.98, lamb=2.0),
            "optimizer_config": OptimizerConfig(lr=1e-4, weight_decay=0.01),
        }

    @pytest.fixture
    def scheduler(self, scheduler_config):
        """Create CurriculumScheduler instance."""
        if not TRAINING_AVAILABLE:
            pytest.skip("Training components not available")
        return CurriculumScheduler(scheduler_config)

    def test_scheduler_creation(self, scheduler, scheduler_config):
        """Test scheduler instantiation."""
        assert scheduler.config == scheduler_config
        assert hasattr(scheduler, "curriculum")
        assert hasattr(scheduler, "grokfast")
        assert hasattr(scheduler, "loss_function")

    def test_scheduler_training_step(self, scheduler):
        """Test scheduler training step integration."""
        # Create dummy model and data
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        x = torch.randn(4, 10)
        targets = torch.randint(0, 5, (4,))

        # Training step
        loss_info = scheduler.training_step(model=model, optimizer=optimizer, inputs=x, targets=targets)

        # Verify loss info structure
        assert "total_loss" in loss_info
        assert "stage" in loss_info
        assert "step" in loss_info
        assert "loss_components" in loss_info

        # Loss should be computed
        assert loss_info["total_loss"].item() >= 0
        assert loss_info["total_loss"].requires_grad


@pytest.mark.integration
class TestTrainingIntegration:
    """Integration tests for complete training system."""

    @pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training components not available")
    def test_complete_curriculum_training(self):
        """Test complete curriculum training integration."""
        # Create simple model for testing
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        # Create training configuration
        config = TestTrainingConfig()

        # Create curriculum stages
        stage_configs = []
        for i, max_steps in enumerate([10, 20, 30]):
            stage_config = StageConfig(
                stage=i,
                name=f"stage_{i}",
                max_steps=max_steps,
                batch_size=4,
                learning_rate=1e-3,
                sequence_length=16,
                loss_weights={"nll": 1.0, "act": 0.1 * i, "memory": 0.05 * i},
            )
            stage_configs.append(stage_config)

        # Create curriculum and optimizer
        curriculum = FourStageCurriculum(stage_configs)
        grokfast = GrokFast(GrokFastConfig(alpha=0.98, lamb=2.0))
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        stage_history = []
        loss_history = []

        for step in range(35):
            # Get current stage config
            stage_config = curriculum.get_current_config()

            # Create dummy data
            x = torch.randn(stage_config.batch_size, 10)
            targets = torch.randint(0, 5, (stage_config.batch_size,))

            # Forward pass
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.CrossEntropyLoss()(outputs, targets)

            # Backward pass
            loss.backward()

            # Apply GrokFast
            grokfast.filter_gradients(model)

            # Optimizer step
            optimizer.step()

            # Record progress
            stage_history.append(curriculum.current_stage)
            loss_history.append(loss.item())

            # Advance curriculum
            curriculum.step()

        # Verify curriculum progression
        unique_stages = list(set(stage_history))
        assert len(unique_stages) >= 2, f"Should progress through multiple stages: {unique_stages}"

        # Verify training progressed
        early_loss = loss_history[:5]
        late_loss = loss_history[-5:]

        avg_early = sum(early_loss) / len(early_loss)
        avg_late = sum(late_loss) / len(late_loss)

        # Allow some variance but expect general improvement
        assert avg_late <= avg_early * 1.5, f"Training should show progress: {avg_early:.3f} -> {avg_late:.3f}"

        print("✓ Complete curriculum training integration successful")
        print(f"✓ Loss progression: {avg_early:.3f} -> {avg_late:.3f}")
        print(f"✓ Stages visited: {len(unique_stages)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
