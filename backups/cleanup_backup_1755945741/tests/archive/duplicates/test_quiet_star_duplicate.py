"""
Comprehensive tests for Quiet-STaR system.
Verifies no leakage, loss terms wired, and tokens present.
"""

from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from src.agent_forge.quiet_star.config import get_default_config, get_inference_config, get_training_config
from src.agent_forge.quiet_star.losses import QuietSTaRLoss
from src.agent_forge.quiet_star.model import QuietSTaRModelWrapper
from src.agent_forge.quiet_star.sampler import ThoughtLeakDetector, ThoughtSampler


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.vocab = {
            "<SoT>": 50001,
            "</SoT>": 50002,
            "<NoT>": 50003,
            "hello": 1,
            "world": 2,
            "test": 3,
            "<pad>": 0,
        }
        self.pad_token_id = 0
        self.eos_token_id = 4

    def encode(
        self,
        text,
        add_special_tokens=False,
        return_tensors=None,
        truncation=False,
        max_length=None,
    ):
        tokens = [self.vocab.get(token, 1) for token in text.split()]
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [inv_vocab.get(token_id, "unk") for token_id in token_ids]
        return " ".join(tokens)

    def add_tokens(self, tokens, special_tokens=False):
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, -1)


class MockModel(nn.Module):
    """Mock language model for testing."""

    def __init__(self, vocab_size=50005, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = Mock()
        self.config.hidden_size = hidden_size
        self.config.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        embeddings = self.embedding(input_ids)
        hidden_states = self.transformer(embeddings)
        logits = self.lm_head(hidden_states)
        output = {"logits": logits, "last_hidden_state": hidden_states}
        if output_hidden_states:
            output["hidden_states"] = [embeddings, hidden_states]
        return output


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def default_config():
    return get_default_config()


class TestQuietSTaRConfig:
    """Test Quiet-STaR configuration system."""

    def test_default_config_creation(self):
        config = get_default_config()
        assert config.enable_quiet_star is True
        assert config.start_of_thought_token == "<SoT>"
        assert config.end_of_thought_token == "</SoT>"
        assert config.no_thought_token == "<NoT>"
        assert 0 <= config.thought_ratio <= 1
        assert config.max_thought_tokens > 0
        assert config.w_leak > 0  # Critical for preventing leaks


class TestThoughtLeakDetector:
    """Test thought leak detection system."""

    def test_leak_detector_initialization(self, default_config):
        detector = ThoughtLeakDetector(default_config)
        assert detector.config == default_config
        assert len(detector.leak_patterns) == 4
        assert len(detector.semantic_leak_patterns) == 3

    def test_token_leak_detection(self, default_config):
        detector = ThoughtLeakDetector(default_config)
        test_cases = [
            ("Normal response", False),
            ("<SoT> leaked thought </SoT>", True),
            ("Dangling <SoT> without end", True),
            ("No-thought token <NoT> present", True),
        ]

        for text, should_leak in test_cases:
            results = detector.detect_leaks(text, check_semantic=False)
            assert results["has_leaks"] == should_leak

    def test_safety_assessment(self, default_config):
        detector = ThoughtLeakDetector(default_config)
        safe_texts = ["Here is the solution to your problem.", "The answer is 42."]
        unsafe_texts = [
            "<SoT> Let me think </SoT> The answer is 42.",
            "Response with <NoT> token.",
        ]

        for text in safe_texts:
            assert detector.is_safe_output(text), f"Text should be safe: {text}"

        for text in unsafe_texts:
            results = detector.detect_leaks(text)
            if results["token_leaks"]:
                assert not detector.is_safe_output(text), f"Text should be unsafe: {text}"


class TestQuietSTaRLoss:
    """Test the combined loss function."""

    def test_loss_initialization(self, default_config):
        loss_fn = QuietSTaRLoss(default_config)
        assert loss_fn.config == default_config
        assert hasattr(loss_fn, "task_loss_fn")
        assert hasattr(loss_fn, "reflection_loss_fn")
        assert len(loss_fn.thought_leak_patterns) == 4

    def test_task_loss_computation(self, default_config):
        loss_fn = QuietSTaRLoss(default_config)
        batch_size, seq_len, vocab_size = 2, 5, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        thought_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        thought_mask[:, 1:3] = True

        task_loss = loss_fn.compute_task_loss(logits, labels, thought_mask)
        assert task_loss.requires_grad
        assert task_loss.item() >= 0
        assert not torch.isnan(task_loss)


class TestIntegration:
    """Integration tests for the complete Quiet-STaR system."""

    def test_critical_no_leakage_requirement(self, mock_model, mock_tokenizer):
        """CRITICAL: Verify no thought leakage in production mode."""
        inference_config = get_inference_config()
        special_token_ids = {
            inference_config.start_of_thought_token: 50001,
            inference_config.end_of_thought_token: 50002,
            inference_config.no_thought_token: 50003,
        }

        wrapper = QuietSTaRModelWrapper(
            base_model=mock_model,
            config=inference_config,
            special_token_ids=special_token_ids,
        )
        sampler = ThoughtSampler(inference_config, mock_tokenizer)
        detector = ThoughtLeakDetector(inference_config)

        wrapper.eval()  # Critical: inference mode
        input_ids = torch.tensor([[1, 2, 3]])

        for _ in range(3):  # Test multiple runs
            with torch.no_grad():
                result = sampler.sample_with_thoughts(
                    model=wrapper,
                    input_ids=input_ids,
                    max_new_tokens=10,
                    force_thoughts=False,
                )

            output_ids = result.stripped_ids if result.stripped_ids is not None else result.generated_ids
            output_text = mock_tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # CRITICAL: No leaks allowed
            leak_results = detector.detect_leaks(output_text)
            assert not leak_results["has_leaks"], f"LEAK DETECTED: {output_text}"
            assert detector.is_safe_output(output_text), f"OUTPUT UNSAFE: {output_text}"

    def test_loss_terms_properly_wired(self, mock_model, mock_tokenizer):
        """Verify all loss terms are properly connected and functional."""
        training_config = get_training_config()
        special_token_ids = {
            training_config.start_of_thought_token: 50001,
            training_config.end_of_thought_token: 50002,
            training_config.no_thought_token: 50003,
        }

        wrapper = QuietSTaRModelWrapper(
            base_model=mock_model,
            config=training_config,
            special_token_ids=special_token_ids,
        )
        loss_fn = QuietSTaRLoss(training_config)

        input_ids = torch.tensor([[1, 50001, 2, 50002, 3, 4]])  # With thoughts
        labels = input_ids.clone()

        wrapper.train()
        outputs = wrapper(input_ids=input_ids)
        thought_mask = wrapper.thought_head.create_thought_mask(input_ids, special_token_ids)

        loss_components = loss_fn(
            logits=outputs["logits"],
            labels=labels,
            thought_mask=thought_mask,
            special_token_ids=special_token_ids,
            generated_texts=["test with <SoT> leak </SoT>"],
        )

        # Verify all loss components are wired
        assert loss_components.task_loss.requires_grad
        assert loss_components.reflection_loss.requires_grad
        assert loss_components.leak_loss.requires_grad
        assert loss_components.total_loss.requires_grad

        # Verify loss values are reasonable
        assert loss_components.task_loss.item() > 0
        assert loss_components.reflection_loss.item() >= 0
        assert loss_components.leak_loss.item() > 0  # Should detect the intentional leak

    def test_special_tokens_present_and_functional(self, mock_tokenizer):
        """Verify special tokens are present and properly handled."""
        training_config = get_training_config()
        sot_id = mock_tokenizer.convert_tokens_to_ids("<SoT>")
        eot_id = mock_tokenizer.convert_tokens_to_ids("</SoT>")
        not_id = mock_tokenizer.convert_tokens_to_ids("<NoT>")

        assert sot_id != -1, "Start-of-thought token missing from vocabulary"
        assert eot_id != -1, "End-of-thought token missing from vocabulary"
        assert not_id != -1, "No-thought token missing from vocabulary"

        # Test token encoding/decoding
        test_text = "<SoT> thinking </SoT> response"
        encoded = mock_tokenizer.encode(test_text)
        decoded = mock_tokenizer.decode(encoded)
        assert "<SoT>" in decoded
        assert "</SoT>" in decoded

        # Test sampler recognizes tokens
        sampler = ThoughtSampler(training_config, mock_tokenizer)
        special_ids = sampler.special_token_ids
        assert special_ids[training_config.start_of_thought_token] == sot_id
        assert special_ids[training_config.end_of_thought_token] == eot_id
        assert special_ids[training_config.no_thought_token] == not_id


@patch("src.agent_forge.quiet_star.cli.AutoModelForCausalLM")
@patch("src.agent_forge.quiet_star.cli.AutoTokenizer")
def test_smoke_test_integration(mock_tokenizer_cls, mock_model_cls):
    """Test the smoke test functionality."""
    mock_model_cls.from_pretrained.return_value = MockModel()
    mock_tokenizer_cls.from_pretrained.return_value = MockTokenizer()

    from src.agent_forge.quiet_star.cli import run_smoke_test

    config = get_default_config()

    with tempfile.TemporaryDirectory() as temp_dir:
        results = run_smoke_test(
            model_name="test-model",
            config=config,
            num_samples=3,
            output_dir=Path(temp_dir),
        )

        # Verify critical test results
        assert results["tests_passed"]["no_leakage"], "Smoke test failed: leakage detected"
        assert results["tests_passed"]["loss_terms_wired"], "Smoke test failed: loss terms not wired"
        assert results["tests_passed"]["tokens_present"], "Smoke test failed: tokens not present"
        assert results["tests_passed"]["generation_successful"], "Smoke test failed: generation failed"

        # Check leak statistics
        assert results["leak_count"] == 0, f"Smoke test detected {results['leak_count']} leaks"
        assert results["leak_rate"] == 0.0, f"Smoke test leak rate: {results['leak_rate']}"
        assert results["overall_pass"], "Smoke test overall failure"


if __name__ == "__main__":
    pytest.main(
        [
            __file__ + "::TestIntegration::test_critical_no_leakage_requirement",
            __file__ + "::TestIntegration::test_loss_terms_properly_wired",
            __file__ + "::TestIntegration::test_special_tokens_present_and_functional",
            __file__ + "::test_smoke_test_integration",
            "-v",
        ]
    )
