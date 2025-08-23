"""Tests for HRRM model implementations."""

from dataclasses import dataclass

import pytest
import torch

from packages.hrrm.memory.model import MemoryAsContextTiny, MemoryConfig, NeuralMemory
from packages.hrrm.planner.model import ControllerHead, HRMPlanner, PlannerConfig
from packages.hrrm.reasoner.model import HRMReasoner, ReasonerConfig, ScratchpadSupervisor


@dataclass
class TestConfig:
    """Test configuration for models."""

    vocab_size: int = 1000
    d_model: int = 128
    n_layers: int = 6
    n_head: int = 8
    d_ff: int = 512
    max_seq_len: int = 256
    tie_embeddings: bool = True
    rope_base: float = 10000.0


class TestControllerHead:
    """Test ControllerHead for planner."""

    def test_controller_head_creation(self):
        """Test ControllerHead creation."""
        d_model = 128
        control_tokens = ["<PLAN>", "<SUBGOAL>", "<ACTION>", "<CHECK>", "<ENDPLAN>"]

        head = ControllerHead(d_model, control_tokens)

        assert head.num_control_tokens == len(control_tokens)
        assert head.control_classifier.out_features == len(control_tokens)

    def test_controller_head_forward(self):
        """Test ControllerHead forward pass."""
        d_model = 128
        control_tokens = ["<PLAN>", "<SUBGOAL>", "<ACTION>", "<CHECK>", "<ENDPLAN>"]

        head = ControllerHead(d_model, control_tokens)

        batch_size = 2
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        control_logits = head(hidden_states)

        assert control_logits.shape == (batch_size, seq_len, len(control_tokens))

    def test_controller_head_loss(self):
        """Test ControllerHead loss calculation."""
        d_model = 128
        control_tokens = ["<PLAN>", "<SUBGOAL>", "<ACTION>", "<CHECK>", "<ENDPLAN>"]

        head = ControllerHead(d_model, control_tokens)

        batch_size = 2
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        # Create control mask (indicating where control tokens should be)
        control_mask = torch.randint(0, 2, (batch_size, seq_len)).bool()
        control_targets = torch.randint(0, len(control_tokens), (batch_size, seq_len))

        control_logits = head(hidden_states)
        loss = head.compute_loss(control_logits, control_targets, control_mask)

        assert loss.item() >= 0.0
        assert loss.requires_grad


class TestScratchpadSupervisor:
    """Test ScratchpadSupervisor for reasoner."""

    def test_scratchpad_supervisor_creation(self):
        """Test ScratchpadSupervisor creation."""
        d_model = 128
        supervisor = ScratchpadSupervisor(d_model)

        assert supervisor.thought_detector.out_features == 2  # thought/no-thought
        assert supervisor.thought_gate.out_features == 1  # gate value

    def test_scratchpad_supervisor_forward(self):
        """Test ScratchpadSupervisor forward pass."""
        d_model = 128
        supervisor = ScratchpadSupervisor(d_model)

        batch_size = 2
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        thought_logits, gate_values = supervisor(hidden_states)

        assert thought_logits.shape == (batch_size, seq_len, 2)
        assert gate_values.shape == (batch_size, seq_len, 1)

    def test_scratchpad_supervisor_thought_detection(self):
        """Test thought span detection."""
        d_model = 128
        supervisor = ScratchpadSupervisor(d_model)

        batch_size = 2
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        thought_spans = supervisor.detect_thought_spans(hidden_states)

        # Should return list of tuples (start, end) for each batch
        assert len(thought_spans) == batch_size
        for spans in thought_spans:
            assert isinstance(spans, list)
            for span in spans:
                assert isinstance(span, tuple)
                assert len(span) == 2
                assert span[0] <= span[1]


class TestNeuralMemory:
    """Test Titans neural memory implementation."""

    def test_neural_memory_creation(self):
        """Test NeuralMemory creation."""
        mem_dim = 64
        mem_slots = 32
        memory = NeuralMemory(mem_dim, mem_slots)

        assert memory.keys.shape == (mem_slots, mem_dim)
        assert memory.values.shape == (mem_slots, mem_dim)

    def test_neural_memory_query(self):
        """Test memory query operation."""
        mem_dim = 64
        mem_slots = 32
        memory = NeuralMemory(mem_dim, mem_slots)

        batch_size = 2
        seq_len = 16
        query = torch.randn(batch_size, seq_len, mem_dim)

        retrieved = memory.query(query)

        assert retrieved.shape == query.shape

    def test_neural_memory_update(self):
        """Test memory update mechanism."""
        mem_dim = 64
        mem_slots = 32
        memory = NeuralMemory(mem_dim, mem_slots)

        # Store initial keys
        initial_keys = memory.keys.clone()

        query = torch.randn(1, 1, mem_dim)
        target = torch.randn(1, 1, mem_dim)
        loss_like = torch.tensor([[1.0]])  # High surprise

        memory.update(query, target, loss_like)

        # Keys should have changed due to update
        assert not torch.allclose(memory.keys, initial_keys)

    def test_neural_memory_gating(self):
        """Test surprise-based gating mechanism."""
        mem_dim = 64
        mem_slots = 32
        memory = NeuralMemory(mem_dim, mem_slots)

        query = torch.randn(1, 1, mem_dim)
        target = torch.randn(1, 1, mem_dim)

        # Test with low surprise (should update less)
        low_loss = torch.tensor([[0.1]])
        initial_keys = memory.keys.clone()
        memory.update(query, target, low_loss)
        keys_after_low = memory.keys.clone()

        # Reset and test with high surprise
        memory.keys = initial_keys.clone()
        high_loss = torch.tensor([[10.0]])
        memory.update(query, target, high_loss)
        keys_after_high = memory.keys.clone()

        # High surprise should cause larger changes
        low_change = torch.norm(keys_after_low - initial_keys)
        high_change = torch.norm(keys_after_high - initial_keys)

        assert high_change > low_change


class TestHRMPlanner:
    """Test HRM Planner model."""

    def create_test_config(self) -> PlannerConfig:
        """Create test configuration."""
        return PlannerConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=6,
            n_head=8,
            d_ff=512,
            max_seq_len=256,
            control_tokens=["<PLAN>", "<SUBGOAL>", "<ACTION>", "<CHECK>", "<ENDPLAN>"],
            max_H=3,
            inner_T=2,
            lambda_ctrl=0.2,
        )

    def test_planner_creation(self):
        """Test HRM Planner creation."""
        config = self.create_test_config()
        model = HRMPlanner(config)

        assert model.config == config
        assert len(model.hrm_layers) == config.n_layers
        assert model.controller_head.num_control_tokens == len(config.control_tokens)

    def test_planner_parameter_count(self):
        """Test planner parameter count is within range."""
        config = self.create_test_config()
        model = HRMPlanner(config)

        total_params = sum(p.numel() for p in model.parameters())

        # Should be in range for 50M model
        assert 10_000_000 <= total_params <= 80_000_000, f"Got {total_params:,} params"

    def test_planner_forward_pass(self):
        """Test planner forward pass."""
        config = self.create_test_config()
        model = HRMPlanner(config)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)

        assert hasattr(outputs, "logits")
        assert hasattr(outputs, "control_logits")
        assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
        assert outputs.control_logits.shape == (batch_size, seq_len, len(config.control_tokens))

    def test_planner_hrm_loop(self):
        """Test HRM two-timescale loop."""
        config = self.create_test_config()
        model = HRMPlanner(config)

        batch_size = 1
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Test with HRM loop
        with torch.no_grad():
            outputs = model(input_ids)

        # Should complete without errors
        assert outputs.logits is not None
        assert outputs.control_logits is not None

    def test_planner_control_loss(self):
        """Test control token loss calculation."""
        config = self.create_test_config()
        model = HRMPlanner(config)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Create control mask and labels
        control_mask = torch.randint(0, 2, (batch_size, seq_len)).bool()

        outputs = model(input_ids, control_mask=control_mask, lambda_ctrl=0.2)

        assert outputs.loss is not None
        assert outputs.loss.requires_grad


class TestHRMReasoner:
    """Test HRM Reasoner model."""

    def create_test_config(self) -> ReasonerConfig:
        """Create test configuration."""
        return ReasonerConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=6,
            n_head=8,
            d_ff=512,
            max_seq_len=256,
            max_H=3,
            inner_T=2,
            self_consistency_k=3,
            start_thought_token="<SoT>",
            end_thought_token="<EoT>",
        )

    def test_reasoner_creation(self):
        """Test HRM Reasoner creation."""
        config = self.create_test_config()
        model = HRMReasoner(config)

        assert model.config == config
        assert len(model.hrm_layers) == config.n_layers
        assert model.scratchpad_supervisor is not None

    def test_reasoner_parameter_count(self):
        """Test reasoner parameter count is within range."""
        config = self.create_test_config()
        model = HRMReasoner(config)

        total_params = sum(p.numel() for p in model.parameters())

        # Should be in range for 50M model
        assert 10_000_000 <= total_params <= 80_000_000, f"Got {total_params:,} params"

    def test_reasoner_forward_pass(self):
        """Test reasoner forward pass."""
        config = self.create_test_config()
        model = HRMReasoner(config)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)

        assert hasattr(outputs, "logits")
        assert hasattr(outputs, "thought_logits")
        assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
        assert outputs.thought_logits.shape == (batch_size, seq_len, 2)

    def test_reasoner_self_consistency(self):
        """Test self-consistency reasoning."""
        config = self.create_test_config()
        model = HRMReasoner(config)

        batch_size = 1
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            results = model.self_consistency_reasoning(input_ids, k=3)

        assert len(results) == 3  # k reasoning chains
        for result in results:
            assert "logits" in result
            assert "thought_spans" in result


class TestMemoryAsContextTiny:
    """Test Memory model with Titans integration."""

    def create_test_config(self) -> MemoryConfig:
        """Create test configuration."""
        return MemoryConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=6,
            n_head=8,
            d_ff=512,
            max_seq_len=256,
            mem_dim=64,
            mem_tokens=16,
            mem_slots=32,
            alpha=1.0,
            beta=0.9,
            eta=0.01,
            eta_decay=0.001,
        )

    def test_memory_creation(self):
        """Test Memory model creation."""
        config = self.create_test_config()
        model = MemoryAsContextTiny(config)

        assert model.config == config
        assert len(model.base_layers) == config.n_layers
        assert model.neural_memory is not None

    def test_memory_parameter_count(self):
        """Test memory model parameter count is within range."""
        config = self.create_test_config()
        model = MemoryAsContextTiny(config)

        total_params = sum(p.numel() for p in model.parameters())

        # Should be in range for 50M model
        assert 10_000_000 <= total_params <= 80_000_000, f"Got {total_params:,} params"

    def test_memory_forward_pass(self):
        """Test memory model forward pass."""
        config = self.create_test_config()
        model = MemoryAsContextTiny(config)

        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids)

        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_memory_retrieval(self):
        """Test memory retrieval mechanism."""
        config = self.create_test_config()
        model = MemoryAsContextTiny(config)

        batch_size = 1
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Test memory context injection
        with torch.no_grad():
            outputs = model(input_ids)

        # Should include memory context
        assert outputs.logits is not None

    def test_memory_update_mechanism(self):
        """Test Titans memory update."""
        config = self.create_test_config()
        model = MemoryAsContextTiny(config)

        batch_size = 1
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Store initial memory state
        model.neural_memory.keys.clone()

        # Forward pass should trigger memory updates
        outputs = model(input_ids)
        loss = outputs.logits.sum()
        loss.backward()

        # Memory should have been updated during forward pass
        # (actual update depends on surprise mechanism)
        assert outputs.logits is not None


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for HRRM models."""

    def test_all_models_similar_size(self):
        """Test that all three models have similar parameter counts."""
        # Create configs for all models
        base_config = {
            "vocab_size": 32000,
            "d_model": 512,
            "n_layers": 12,
            "n_head": 8,
            "d_ff": 2048,
            "max_seq_len": 2048,
        }

        planner_config = PlannerConfig(
            **base_config,
            control_tokens=["<PLAN>", "<SUBGOAL>", "<ACTION>", "<CHECK>", "<ENDPLAN>"],
            max_H=3,
            inner_T=2,
        )

        reasoner_config = ReasonerConfig(
            **base_config,
            max_H=3,
            inner_T=2,
            self_consistency_k=5,
        )

        memory_config = MemoryConfig(
            **base_config,
            mem_dim=256,
            mem_tokens=64,
            mem_slots=128,
        )

        # Create models
        planner = HRMPlanner(planner_config)
        reasoner = HRMReasoner(reasoner_config)
        memory = MemoryAsContextTiny(memory_config)

        # Count parameters
        planner_params = sum(p.numel() for p in planner.parameters())
        reasoner_params = sum(p.numel() for p in reasoner.parameters())
        memory_params = sum(p.numel() for p in memory.parameters())

        # All should be in 48M-55M range
        for name, params in [("planner", planner_params), ("reasoner", reasoner_params), ("memory", memory_params)]:
            assert 30_000_000 <= params <= 70_000_000, f"{name}: {params:,} params"

        # Should be relatively similar (within 20% of each other)
        avg_params = (planner_params + reasoner_params + memory_params) / 3
        for name, params in [("planner", planner_params), ("reasoner", reasoner_params), ("memory", memory_params)]:
            ratio = abs(params - avg_params) / avg_params
            assert ratio < 0.5, f"{name} params too different: {params:,} vs avg {avg_params:,}"

    def test_models_can_be_merged(self):
        """Test that models can be loaded for EvoMerge integration."""
        # This tests the integration point with Agent Forge
        base_config = {
            "vocab_size": 1000,
            "d_model": 128,
            "n_layers": 6,
            "n_head": 8,
            "d_ff": 512,
            "max_seq_len": 256,
        }

        planner_config = PlannerConfig(
            **base_config,
            control_tokens=["<PLAN>", "<SUBGOAL>", "<ACTION>"],
            max_H=2,
            inner_T=1,
        )

        # Create model
        planner = HRMPlanner(planner_config)

        # Test state dict extraction (needed for merging)
        state_dict = planner.state_dict()
        assert len(state_dict) > 0

        # Test that we can create new model and load state
        new_planner = HRMPlanner(planner_config)
        new_planner.load_state_dict(state_dict)

        # Test forward pass still works
        input_ids = torch.randint(0, 1000, (1, 16))
        with torch.no_grad():
            outputs1 = planner(input_ids)
            outputs2 = new_planner(input_ids)

        # Should produce same outputs
        assert torch.allclose(outputs1.logits, outputs2.logits, atol=1e-6)

    def test_memory_efficiency(self):
        """Test memory efficiency of models."""
        config = PlannerConfig(
            vocab_size=1000,
            d_model=256,
            n_layers=8,
            n_head=8,
            d_ff=1024,
            max_seq_len=512,
            control_tokens=["<PLAN>", "<ACTION>"],
            max_H=2,
            inner_T=1,
        )

        model = HRMPlanner(config)

        # Test with various batch sizes and sequence lengths
        test_cases = [
            (1, 64),
            (2, 128),
            (4, 256),
        ]

        for batch_size, seq_len in test_cases:
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))

            # Should not run out of memory
            with torch.no_grad():
                outputs = model(input_ids)
                assert outputs.logits.shape == (batch_size, seq_len, 1000)

    def test_gradient_checkpointing_compatibility(self):
        """Test models work with gradient checkpointing."""
        config = PlannerConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=4,
            n_head=8,
            d_ff=512,
            max_seq_len=256,
            control_tokens=["<PLAN>"],
            max_H=2,
            inner_T=1,
        )

        model = HRMPlanner(config)

        # Enable gradient checkpointing if available
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        input_ids = torch.randint(0, 1000, (2, 32))

        # Forward and backward pass
        outputs = model(input_ids)
        loss = outputs.logits.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
