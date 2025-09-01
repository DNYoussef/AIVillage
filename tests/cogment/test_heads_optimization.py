"""
Tests for Cogment Heads Optimization System (Agent 3)

Tests the output head components including:
- Image head for visual reasoning tasks
- Text head with vocabulary optimization and tied embeddings
- Task adapters for specialized domain processing
- Vocabulary optimization and factorization strategies
- Parameter efficiency through weight tying
"""

import pytest
import torch
import torch.nn.functional as F

# Import Cogment heads components
try:
    from core.agent_forge.models.cogment.heads.image_head import ImageHead, ImageHeadConfig
    from core.agent_forge.models.cogment.heads.task_adapters import TaskAdapter, TaskAdapterConfig
    from core.agent_forge.models.cogment.heads.text_head import TextHead, TextHeadConfig
    from core.agent_forge.models.cogment.heads.vocabulary_optimization import (
        TiedEmbedding,
        VocabOptimConfig,
        VocabularyOptimizer,
    )

    HEADS_AVAILABLE = True
except ImportError as e:
    print(f"Cogment heads components not available: {e}")
    HEADS_AVAILABLE = False


class TestImageHead:
    """Test Image Head for visual reasoning tasks."""

    @pytest.fixture
    def image_config(self):
        """Create image head configuration."""
        return ImageHeadConfig(
            d_model=512,
            image_channels=3,
            patch_size=16,
            image_size=224,
            projection_dim=256,
            use_cls_token=True,
            positional_encoding=True,
        )

    @pytest.fixture
    def image_head(self, image_config):
        """Create ImageHead instance."""
        if not HEADS_AVAILABLE:
            pytest.skip("Heads components not available")
        return ImageHead(image_config)

    def test_image_head_creation(self, image_head, image_config):
        """Test ImageHead instantiation."""
        assert image_head.config == image_config
        assert hasattr(image_head, "patch_embedding")
        assert hasattr(image_head, "position_encoding")
        assert hasattr(image_head, "projection_layer")

        if image_config.use_cls_token:
            assert hasattr(image_head, "cls_token")

    def test_image_head_patch_embedding(self, image_head, image_config):
        """Test image patch embedding functionality."""
        batch_size = 2

        # Create image tensor
        images = torch.randn(batch_size, image_config.image_channels, image_config.image_size, image_config.image_size)

        with torch.no_grad():
            patches = image_head.extract_patches(images)

        # Calculate expected number of patches
        expected_patches = (image_config.image_size // image_config.patch_size) ** 2
        if image_config.use_cls_token:
            expected_patches += 1  # Add CLS token

        assert patches.shape == (batch_size, expected_patches, image_config.d_model)

    def test_image_head_forward_pass(self, image_head, image_config):
        """Test complete image head forward pass."""
        batch_size = 2

        images = torch.randn(batch_size, image_config.image_channels, image_config.image_size, image_config.image_size)

        with torch.no_grad():
            outputs = image_head(images)

        # Verify output structure
        assert hasattr(outputs, "visual_features")
        assert hasattr(outputs, "patch_embeddings")
        assert hasattr(outputs, "attention_mask")

        # Verify tensor shapes
        expected_seq_len = (image_config.image_size // image_config.patch_size) ** 2
        if image_config.use_cls_token:
            expected_seq_len += 1

        assert outputs.visual_features.shape == (batch_size, expected_seq_len, image_config.projection_dim)
        assert outputs.patch_embeddings.shape == (batch_size, expected_seq_len, image_config.d_model)

    def test_image_head_positional_encoding(self, image_head, image_config):
        """Test positional encoding for image patches."""
        if not image_config.positional_encoding:
            pytest.skip("Positional encoding disabled")

        batch_size = 1
        images = torch.randn(batch_size, image_config.image_channels, image_config.image_size, image_config.image_size)

        # Test with and without positional encoding
        with torch.no_grad():
            outputs_with_pos = image_head(images)

            # Temporarily disable positional encoding
            image_head.config.positional_encoding = False
            outputs_without_pos = image_head(images)
            image_head.config.positional_encoding = True

        # Outputs should be different when positional encoding is used
        features_diff = torch.norm(outputs_with_pos.visual_features - outputs_without_pos.visual_features)
        assert features_diff > 0.1, "Positional encoding should affect output features"

    def test_image_head_arc_compatibility(self, image_head, image_config):
        """Test compatibility with ARC visual reasoning tasks."""
        batch_size = 2

        # ARC tasks often use smaller grids, test with different sizes
        arc_sizes = [64, 96, 128]  # Smaller than standard 224

        for size in arc_sizes:
            # Create ARC-like images (simpler patterns)
            arc_images = torch.zeros(batch_size, 3, size, size)

            # Add some pattern (simulate ARC grid)
            arc_images[:, :, ::8, ::8] = 1.0  # Grid pattern

            with torch.no_grad():
                # Resize for compatibility if needed
                if size != image_config.image_size:
                    arc_images = F.interpolate(
                        arc_images, size=(image_config.image_size, image_config.image_size), mode="nearest"
                    )

                outputs = image_head(arc_images)

            # Should produce valid outputs for ARC-style inputs
            assert outputs.visual_features is not None
            assert not torch.isnan(outputs.visual_features).any()


class TestTextHead:
    """Test Text Head with vocabulary optimization."""

    @pytest.fixture
    def text_config(self):
        """Create text head configuration."""
        return TextHeadConfig(
            d_model=512,
            vocab_size=13000,
            tie_embeddings=True,
            use_bias=False,
            dropout=0.1,
            factorize_large_vocab=False,  # Option A: no factorization
            factorization_rank=None,
        )

    @pytest.fixture
    def text_head(self, text_config):
        """Create TextHead instance."""
        if not HEADS_AVAILABLE:
            pytest.skip("Heads components not available")
        return TextHead(text_config)

    def test_text_head_creation(self, text_head, text_config):
        """Test TextHead instantiation."""
        assert text_head.config == text_config
        assert hasattr(text_head, "output_projection")
        assert hasattr(text_head, "embedding_layer")

        if text_config.tie_embeddings:
            # Verify weight tying
            assert text_head.output_projection.weight is text_head.embedding_layer.weight

    def test_text_head_forward_pass(self, text_head, text_config):
        """Test text head forward pass."""
        batch_size = 2
        seq_len = 32
        d_model = text_config.d_model

        hidden_states = torch.randn(batch_size, seq_len, d_model)

        with torch.no_grad():
            logits = text_head(hidden_states)

        assert logits.shape == (batch_size, seq_len, text_config.vocab_size)

    def test_text_head_embedding_forward(self, text_head, text_config):
        """Test embedding layer forward pass."""
        batch_size = 2
        seq_len = 32

        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            embeddings = text_head.embed(input_ids)

        assert embeddings.shape == (batch_size, seq_len, text_config.d_model)

    def test_text_head_parameter_efficiency(self, text_head, text_config):
        """Test parameter efficiency through tied embeddings."""
        total_params = sum(p.numel() for p in text_head.parameters())

        if text_config.tie_embeddings:
            # With tied embeddings: only one copy of vocab x d_model
            expected_params = text_config.vocab_size * text_config.d_model
            if text_config.use_bias:
                expected_params += text_config.vocab_size  # Output bias

            # Allow some tolerance for other parameters
            assert (
                total_params <= expected_params * 1.1
            ), f"Tied embeddings should reduce parameters: {total_params:,} vs expected ~{expected_params:,}"

        print(f"✓ Text head parameters: {total_params:,}")

    def test_text_head_vocabulary_coverage(self, text_head, text_config):
        """Test vocabulary coverage and special tokens."""
        vocab_size = text_config.vocab_size

        # Test all vocabulary indices
        test_indices = torch.arange(0, vocab_size).unsqueeze(0)  # [1, vocab_size]

        with torch.no_grad():
            embeddings = text_head.embed(test_indices)
            logits = text_head(embeddings)

        assert embeddings.shape == (1, vocab_size, text_config.d_model)
        assert logits.shape == (1, vocab_size, vocab_size)

        # Verify no NaN or infinite values
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


class TestTaskAdapters:
    """Test Task Adapters for specialized domain processing."""

    @pytest.fixture
    def adapter_config(self):
        """Create task adapter configuration."""
        return TaskAdapterConfig(
            d_model=512,
            adapter_dim=128,
            task_types=["arc", "math", "text", "long_context"],
            use_gating=True,
            dropout=0.1,
            residual_connection=True,
        )

    @pytest.fixture
    def task_adapter(self, adapter_config):
        """Create TaskAdapter instance."""
        if not HEADS_AVAILABLE:
            pytest.skip("Heads components not available")
        return TaskAdapter(adapter_config)

    def test_task_adapter_creation(self, task_adapter, adapter_config):
        """Test TaskAdapter instantiation."""
        assert task_adapter.config == adapter_config
        assert hasattr(task_adapter, "task_adapters")
        assert len(task_adapter.task_adapters) == len(adapter_config.task_types)

        for task_type in adapter_config.task_types:
            assert task_type in task_adapter.task_adapters

    def test_task_adapter_forward(self, task_adapter, adapter_config):
        """Test task adapter forward pass."""
        batch_size = 2
        seq_len = 16
        d_model = adapter_config.d_model

        hidden_states = torch.randn(batch_size, seq_len, d_model)
        task_type = "arc"  # Test with ARC task

        with torch.no_grad():
            adapted_states = task_adapter(hidden_states, task_type)

        assert adapted_states.shape == hidden_states.shape

        # With residual connection, output should be different from input
        if adapter_config.residual_connection:
            diff = torch.norm(adapted_states - hidden_states)
            assert diff > 0.01, "Adapter should modify hidden states"

    def test_task_adapter_specialization(self, task_adapter, adapter_config):
        """Test that different task types produce different adaptations."""
        batch_size = 1
        seq_len = 8
        d_model = adapter_config.d_model

        hidden_states = torch.randn(batch_size, seq_len, d_model)

        # Test different task types
        task_outputs = {}
        for task_type in adapter_config.task_types:
            with torch.no_grad():
                adapted = task_adapter(hidden_states, task_type)
                task_outputs[task_type] = adapted

        # Different task types should produce different outputs
        task_types = list(task_outputs.keys())
        for i in range(len(task_types)):
            for j in range(i + 1, len(task_types)):
                diff = torch.norm(task_outputs[task_types[i]] - task_outputs[task_types[j]])
                assert diff > 0.01, f"Tasks {task_types[i]} and {task_types[j]} should produce different adaptations"

    def test_task_adapter_parameter_efficiency(self, task_adapter, adapter_config):
        """Test parameter efficiency of task adapters."""
        total_params = sum(p.numel() for p in task_adapter.parameters())

        # Task adapters should be much smaller than full model layers
        max_expected = len(adapter_config.task_types) * adapter_config.adapter_dim * adapter_config.d_model * 3

        assert total_params <= max_expected, f"Task adapters should be parameter efficient: {total_params:,} params"

        print(f"✓ Task adapter parameters: {total_params:,}")

    def test_task_adapter_gating(self, task_adapter, adapter_config):
        """Test gating mechanism in task adapters."""
        if not adapter_config.use_gating:
            pytest.skip("Gating disabled")

        batch_size = 2
        seq_len = 16
        d_model = adapter_config.d_model

        hidden_states = torch.randn(batch_size, seq_len, d_model)

        # Test with different task types to see gating effects
        for task_type in adapter_config.task_types:
            with torch.no_grad():
                adapted = task_adapter(hidden_states, task_type)

            # Gating should modulate the adaptation
            assert adapted.shape == hidden_states.shape

            # The adaptation should not be too extreme (gating should control it)
            adaptation_magnitude = torch.norm(adapted - hidden_states, dim=-1).mean()
            assert adaptation_magnitude < 10.0, "Gating should prevent extreme adaptations"


class TestVocabularyOptimization:
    """Test vocabulary optimization strategies."""

    @pytest.fixture
    def vocab_config(self):
        """Create vocabulary optimization configuration."""
        return VocabOptimConfig(
            vocab_size=13000,
            d_model=512,
            tie_embeddings=True,
            factorize=False,  # Option A: no factorization
            factorization_rank=None,
            shared_embedding_fraction=0.8,
            specialized_tokens=["<ARC>", "<MATH>", "<LONG>"],
        )

    @pytest.fixture
    def vocab_optimizer(self, vocab_config):
        """Create VocabularyOptimizer instance."""
        if not HEADS_AVAILABLE:
            pytest.skip("Heads components not available")
        return VocabularyOptimizer(vocab_config)

    def test_vocabulary_optimizer_creation(self, vocab_optimizer, vocab_config):
        """Test VocabularyOptimizer instantiation."""
        assert vocab_optimizer.config == vocab_config
        assert hasattr(vocab_optimizer, "embedding_layer")
        assert hasattr(vocab_optimizer, "output_layer")

        if vocab_config.tie_embeddings:
            assert vocab_optimizer.embedding_layer.weight is vocab_optimizer.output_layer.weight

    def test_tied_embedding_functionality(self, vocab_optimizer, vocab_config):
        """Test tied embedding functionality."""
        if not vocab_config.tie_embeddings:
            pytest.skip("Tied embeddings disabled")

        # Create TiedEmbedding directly
        tied_emb = TiedEmbedding(vocab_config.vocab_size, vocab_config.d_model)

        batch_size = 2
        seq_len = 16

        # Test embedding forward
        input_ids = torch.randint(0, vocab_config.vocab_size, (batch_size, seq_len))
        embeddings = tied_emb.embed(input_ids)

        assert embeddings.shape == (batch_size, seq_len, vocab_config.d_model)

        # Test output projection
        hidden_states = torch.randn(batch_size, seq_len, vocab_config.d_model)
        logits = tied_emb.project(hidden_states)

        assert logits.shape == (batch_size, seq_len, vocab_config.vocab_size)

        # Verify weight sharing
        assert tied_emb.embedding.weight is tied_emb.projection.weight

    def test_vocabulary_parameter_count(self, vocab_optimizer, vocab_config):
        """Test vocabulary parameter count optimization."""
        total_params = sum(p.numel() for p in vocab_optimizer.parameters())

        if vocab_config.tie_embeddings:
            # Tied embeddings: only one copy of vocabulary parameters
            expected_base = vocab_config.vocab_size * vocab_config.d_model
        else:
            # Separate embeddings: two copies
            expected_base = 2 * vocab_config.vocab_size * vocab_config.d_model

        # Allow some tolerance for specialized components
        max_expected = expected_base * 1.2

        assert (
            total_params <= max_expected
        ), f"Vocabulary parameters should be optimized: {total_params:,} vs max expected {max_expected:,}"

        print(f"✓ Vocabulary parameters: {total_params:,}")

    def test_specialized_token_handling(self, vocab_optimizer, vocab_config):
        """Test handling of specialized tokens."""
        if not vocab_config.specialized_tokens:
            pytest.skip("No specialized tokens configured")

        batch_size = 1
        seq_len = len(vocab_config.specialized_tokens)

        # Create input with specialized tokens (assume they're at end of vocab)
        specialized_start = vocab_config.vocab_size - len(vocab_config.specialized_tokens)
        specialized_ids = torch.arange(specialized_start, vocab_config.vocab_size).unsqueeze(0)

        with torch.no_grad():
            embeddings = vocab_optimizer.embed(specialized_ids)
            logits = vocab_optimizer.project(embeddings)

        assert embeddings.shape == (batch_size, seq_len, vocab_config.d_model)
        assert logits.shape == (batch_size, seq_len, vocab_config.vocab_size)

        # Specialized tokens should have distinct embeddings
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                emb_diff = torch.norm(embeddings[0, i] - embeddings[0, j])
                assert emb_diff > 0.1, "Specialized tokens should have distinct embeddings"


@pytest.mark.integration
class TestHeadsIntegration:
    """Integration tests for all head components."""

    @pytest.mark.skipif(not HEADS_AVAILABLE, reason="Heads components not available")
    def test_all_heads_integration(self):
        """Test integration of all head components."""
        batch_size = 2
        seq_len = 16
        d_model = 512
        vocab_size = 13000

        # Create configurations
        image_config = ImageHeadConfig(d_model=d_model, image_size=224)
        text_config = TextHeadConfig(d_model=d_model, vocab_size=vocab_size, tie_embeddings=True)
        adapter_config = TaskAdapterConfig(d_model=d_model, task_types=["arc", "text"])

        # Create heads
        image_head = ImageHead(image_config)
        text_head = TextHead(text_config)
        task_adapter = TaskAdapter(adapter_config)

        # Test text processing pathway
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            # Text embedding
            text_embeddings = text_head.embed(input_ids)

            # Task adaptation
            adapted_embeddings = task_adapter(text_embeddings, "text")

            # Output projection
            text_logits = text_head(adapted_embeddings)

        assert text_logits.shape == (batch_size, seq_len, vocab_size)

        # Test image processing pathway
        images = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            # Image encoding
            image_outputs = image_head(images)

            # Task adaptation for visual reasoning
            task_adapter(image_outputs.visual_features, "arc")

            # Can project visual features to text space if needed
            # (this would require additional projection layer in practice)

        print("✓ All heads integration successful")

    @pytest.mark.skipif(not HEADS_AVAILABLE, reason="Heads components not available")
    def test_heads_parameter_budget(self):
        """Test that all heads fit within parameter budget."""
        d_model = 512
        vocab_size = 13000

        # Create all heads
        image_head = ImageHead(ImageHeadConfig(d_model=d_model))
        text_head = TextHead(TextHeadConfig(d_model=d_model, vocab_size=vocab_size, tie_embeddings=True))
        task_adapter = TaskAdapter(TaskAdapterConfig(d_model=d_model))
        vocab_optimizer = VocabularyOptimizer(
            VocabOptimConfig(vocab_size=vocab_size, d_model=d_model, tie_embeddings=True)
        )

        # Count parameters
        image_params = sum(p.numel() for p in image_head.parameters())
        text_params = sum(p.numel() for p in text_head.parameters())
        adapter_params = sum(p.numel() for p in task_adapter.parameters())
        vocab_params = sum(p.numel() for p in vocab_optimizer.parameters())

        total_head_params = image_params + text_params + adapter_params + vocab_params

        # Should be reasonable fraction of 23.7M total budget
        max_head_budget = 10_000_000  # 10M parameters for all heads

        assert (
            total_head_params <= max_head_budget
        ), f"Head parameters exceed budget: {total_head_params:,} vs {max_head_budget:,}"

        print(f"✓ Total head parameters: {total_head_params:,}")
        print(f"  - Image head: {image_params:,}")
        print(f"  - Text head: {text_params:,}")
        print(f"  - Task adapters: {adapter_params:,}")
        print(f"  - Vocabulary: {vocab_params:,}")

    @pytest.mark.skipif(not HEADS_AVAILABLE, reason="Heads components not available")
    def test_heads_memory_efficiency(self):
        """Test memory efficiency of head components."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create heads
        heads = []

        # Multiple instances to test scaling
        for _ in range(3):
            image_head = ImageHead(ImageHeadConfig(d_model=256))  # Smaller for test
            text_head = TextHead(TextHeadConfig(d_model=256, vocab_size=5000, tie_embeddings=True))
            heads.extend([image_head, text_head])

        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_head = (current_memory - initial_memory) / len(heads)

        # Each head should use reasonable memory
        max_memory_per_head = 50  # MB

        assert memory_per_head <= max_memory_per_head, f"Head memory usage too high: {memory_per_head:.1f}MB per head"

        print(f"✓ Head memory efficiency: {memory_per_head:.1f}MB per head")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
