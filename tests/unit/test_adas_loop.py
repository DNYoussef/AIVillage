"""
Comprehensive tests for ADAS (Automatic Discovery of Agentic Space) loop.
Tests archive, proposer, runner, and complete search orchestration.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from src.agent_forge.adas.archive import ADASArchive, ExperimentResult
from src.agent_forge.adas.proposer import ADASProposer
from src.agent_forge.adas.runner import ADASRunner, TaskSuite


@pytest.fixture
def temp_archive_path():
    """Create temporary archive file."""
    temp_dir = tempfile.mkdtemp()
    archive_path = Path(temp_dir) / "test_archive.jsonl"
    yield archive_path
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_expert_spec():
    """Sample expert specification."""
    return {
        "layers": ["attn_qkv", "mlp"],
        "rank": 4,
        "svd_scope": "per-matrix",
        "init": "random",
        "activation_rule": "gated",
        "budget": {"max_active": 2, "max_latency_ms": 100},
    }


@pytest.fixture
def sample_dispatch_spec():
    """Sample dispatch specification."""
    return {
        "features": ["prompt_stats", "activation_sketch"],
        "mix_fn": "softmax",
        "granularity": "sequence",
    }


@pytest.fixture
def sample_experiment_result(sample_expert_spec, sample_dispatch_spec):
    """Sample successful experiment result."""
    return ExperimentResult(
        expert_spec=sample_expert_spec,
        dispatch_spec=sample_dispatch_spec,
        score=0.75,
        latency_ms=85.5,
        vram_gb=0.25,
        trial_id="test_001",
        task_suite="coding_small",
        num_samples=3,
        success=True,
        metrics={
            "total_time_s": 12.3,
            "min_latency_ms": 80.0,
            "max_latency_ms": 90.0,
            "std_latency_ms": 4.2,
        },
    )


class TestADASArchive:
    """Test the ADAS archive component."""

    def test_archive_creation(self, temp_archive_path):
        """Test archive creation and initialization."""
        archive = ADASArchive(temp_archive_path)

        assert archive.archive_path == temp_archive_path
        assert archive.archive_path.exists()

        stats = archive.get_statistics()
        assert stats["total_results"] == 0
        assert stats["successful_results"] == 0

    def test_add_and_retrieve_results(self, temp_archive_path, sample_experiment_result):
        """Test adding and retrieving experiment results."""
        archive = ADASArchive(temp_archive_path)

        # Add result
        archive.add_result(sample_experiment_result)

        # Check statistics
        stats = archive.get_statistics()
        assert stats["total_results"] == 1
        assert stats["successful_results"] == 1
        assert stats["success_rate"] == 1.0

        # Retrieve results
        results = archive.get_all_results()
        assert len(results) == 1
        assert results[0].trial_id == "test_001"
        assert results[0].score == 0.75

    def test_leaderboard(self, temp_archive_path):
        """Test leaderboard generation."""
        archive = ADASArchive(temp_archive_path)

        # Add multiple results with different scores
        results = []
        for i, score in enumerate([0.8, 0.6, 0.9, 0.7, 0.5]):
            result = ExperimentResult(
                expert_spec={"layers": ["attn_qkv"], "rank": 2},
                dispatch_spec={"features": ["prompt_stats"]},
                score=score,
                latency_ms=100 - i * 5,  # Vary latency too
                vram_gb=0.1 + i * 0.05,
                trial_id=f"trial_{i:03d}",
                task_suite="coding_small",
                success=True,
            )
            results.append(result)
            archive.add_result(result)

        # Test leaderboard
        leaderboard = archive.get_leaderboard(top_k=3)
        assert len(leaderboard) == 3

        # Should be sorted by score (descending)
        scores = [r.score for r in leaderboard]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 0.9  # Best score should be first

    def test_pareto_frontier(self, temp_archive_path):
        """Test Pareto frontier computation."""
        archive = ADASArchive(temp_archive_path)

        # Add results with tradeoffs between score and latency
        results_data = [
            (0.9, 150),  # High score, high latency
            (0.7, 80),  # Medium score, low latency
            (0.6, 200),  # Low score, high latency (dominated)
            (0.8, 90),  # Good score, low latency (Pareto optimal)
            (0.85, 120),  # High score, medium latency
        ]

        for i, (score, latency) in enumerate(results_data):
            result = ExperimentResult(
                expert_spec={"layers": ["mlp"], "rank": 4},
                dispatch_spec={"features": ["logits_entropy"]},
                score=score,
                latency_ms=latency,
                vram_gb=0.2,
                trial_id=f"pareto_{i:03d}",
                task_suite="coding_small",
                success=True,
            )
            archive.add_result(result)

        # Get Pareto frontier
        pareto = archive.get_pareto_frontier(["score", "latency_ms"])

        # Should have multiple points, excluding dominated ones
        assert len(pareto) >= 2

        # Verify Pareto optimality (no point should dominate another)
        for i, result_i in enumerate(pareto):
            for j, result_j in enumerate(pareto):
                if i != j:
                    # result_i should not dominate result_j
                    dominates = (
                        result_i.score >= result_j.score
                        and result_i.latency_ms <= result_j.latency_ms
                        and (result_i.score > result_j.score or result_i.latency_ms < result_j.latency_ms)
                    )
                    assert not dominates, f"Point {i} dominates point {j} in Pareto set"

    def test_yaml_export(self, temp_archive_path, sample_experiment_result):
        """Test YAML configuration export."""
        archive = ADASArchive(temp_archive_path)
        archive.add_result(sample_experiment_result)

        # Create temp output directory
        temp_dir = Path(tempfile.mkdtemp())
        try:
            exported_configs = archive.export_yaml_configs(output_dir=temp_dir, top_k=1)

            assert len(exported_configs) == 1
            config_path = exported_configs[0]
            assert config_path.exists()
            assert config_path.suffix == ".yaml"

            # Check that file contains expected content
            content = config_path.read_text()
            assert "expert:" in content
            assert "dispatch:" in content
            assert "attn_qkv" in content

        finally:
            shutil.rmtree(temp_dir)

    def test_archive_persistence(self, temp_archive_path, sample_experiment_result):
        """Test that archive persists across instances."""
        # Add result with first instance
        archive1 = ADASArchive(temp_archive_path)
        archive1.add_result(sample_experiment_result)

        # Create new instance and verify data persists
        archive2 = ADASArchive(temp_archive_path)
        results = archive2.get_all_results()

        assert len(results) == 1
        assert results[0].trial_id == sample_experiment_result.trial_id
        assert results[0].score == sample_experiment_result.score


class TestADASProposer:
    """Test the ADAS proposer component."""

    def test_proposer_initialization(self):
        """Test proposer initialization."""
        proposer = ADASProposer()

        assert len(proposer.layer_options) > 0
        assert len(proposer.rank_options) > 0
        assert len(proposer.feature_combinations) > 0

    def test_random_proposals(self):
        """Test random proposal generation."""
        proposer = ADASProposer()

        proposals = proposer._generate_random_proposal(target_latency_ms=100)

        assert "expert" in proposals
        assert "dispatch" in proposals
        assert "motivation" in proposals

        expert = proposals["expert"]
        assert "layers" in expert
        assert "rank" in expert
        assert "svd_scope" in expert
        assert "init" in expert
        assert "activation_rule" in expert
        assert "budget" in expert

        dispatch = proposals["dispatch"]
        assert "features" in dispatch
        assert "mix_fn" in dispatch
        assert "granularity" in dispatch

        # Check budget constraint
        assert expert["budget"]["max_latency_ms"] == 100

    def test_template_proposals(self):
        """Test template-based proposal generation."""
        proposer = ADASProposer()

        proposals = proposer._generate_template_proposal(target_latency_ms=50)

        assert "expert" in proposals
        assert "dispatch" in proposals
        assert "motivation" in proposals

        # Should respect latency constraint in budget
        assert proposals["expert"]["budget"]["max_latency_ms"] <= 50

    def test_archive_guided_mutations(self, temp_archive_path):
        """Test archive-guided proposal mutations."""
        # Create archive with some results
        archive = ADASArchive(temp_archive_path)

        # Add successful results
        successful_results = []
        for i in range(3):
            result = ExperimentResult(
                expert_spec={
                    "layers": ["attn_qkv"],
                    "rank": 2 + i,
                    "svd_scope": "per-matrix",
                    "init": "random",
                    "activation_rule": "gated",
                    "budget": {"max_active": 2, "max_latency_ms": 100},
                },
                dispatch_spec={
                    "features": ["prompt_stats"],
                    "mix_fn": "softmax",
                    "granularity": "sequence",
                },
                score=0.7 + i * 0.1,  # Increasing scores
                latency_ms=90 - i * 10,  # Decreasing latency
                vram_gb=0.2,
                trial_id=f"archive_test_{i:03d}",
                task_suite="coding_small",
                success=True,
            )
            successful_results.append(result)
            archive.add_result(result)

        proposer = ADASProposer(archive)

        # Generate mutation-based proposal
        proposal = proposer._mutate_from_archive(successful_results, target_latency_ms=80)

        assert "expert" in proposal
        assert "dispatch" in proposal
        assert "motivation" in proposal
        assert "Mutation from" in proposal["motivation"]

    def test_diversity_maximization(self):
        """Test diversity-maximizing proposal generation."""
        proposer = ADASProposer()

        # Create some existing proposals
        existing_proposals = [
            {
                "expert": {"layers": ["attn_qkv"], "rank": 2, "init": "random"},
                "dispatch": {"features": ["prompt_stats"], "granularity": "sequence"},
            },
            {
                "expert": {"layers": ["mlp"], "rank": 4, "init": "fisher"},
                "dispatch": {"features": ["logits_entropy"], "granularity": "token"},
            },
        ]

        diverse_proposal = proposer._generate_diverse_proposal(
            target_latency_ms=100, existing_proposals=existing_proposals
        )

        assert "expert" in diverse_proposal
        assert "dispatch" in diverse_proposal
        assert diverse_proposal["motivation"] == "Diversity-maximizing configuration"

        # Should be different from existing proposals
        diverse_features = (
            tuple(sorted(diverse_proposal["expert"]["layers"])),
            diverse_proposal["expert"]["rank"],
            diverse_proposal["expert"]["init"],
        )

        for existing in existing_proposals:
            existing_features = (
                tuple(sorted(existing["expert"]["layers"])),
                existing["expert"]["rank"],
                existing["expert"]["init"],
            )
            assert diverse_features != existing_features  # Should be different

    def test_complete_proposal_generation(self):
        """Test complete proposal generation with multiple strategies."""
        proposer = ADASProposer()

        proposals = proposer.propose(n_proposals=8, target_latency_ms=100)

        assert len(proposals) == 8

        # Check that all proposals are valid
        for proposal in proposals:
            assert "expert" in proposal
            assert "dispatch" in proposal
            assert "motivation" in proposal

            # Validate expert spec
            expert = proposal["expert"]
            assert isinstance(expert["layers"], list)
            assert isinstance(expert["rank"], int)
            assert expert["rank"] > 0
            assert expert["svd_scope"] in ["per-matrix", "per-block"]
            assert expert["init"] in ["random", "pca_activations", "fisher"]

            # Validate dispatch spec
            dispatch = proposal["dispatch"]
            assert isinstance(dispatch["features"], list)
            assert len(dispatch["features"]) > 0

    def test_search_statistics(self):
        """Test search space statistics."""
        proposer = ADASProposer()
        stats = proposer.get_search_statistics()

        assert "layer_combinations" in stats
        assert "rank_options" in stats
        assert "init_methods" in stats
        assert "feature_combinations" in stats
        assert "total_expert_configs" in stats
        assert "total_dispatch_configs" in stats

        # Verify calculations
        expected_expert_configs = (
            len(proposer.layer_options)
            * len(proposer.rank_options)
            * len(proposer.svd_scope_options)
            * len(proposer.init_options)
            * len(proposer.activation_rule_options)
        )
        assert stats["total_expert_configs"] == expected_expert_configs


class TestTaskSuite:
    """Test the task suite component."""

    def test_coding_small_suite(self):
        """Test coding_small task suite."""
        suite = TaskSuite("coding_small")

        assert suite.suite_name == "coding_small"
        assert len(suite.tasks) == 3  # Should have 3 coding tasks

        for task in suite.tasks:
            assert "prompt" in task
            assert "expected_keywords" in task
            assert "complexity" in task
            assert 0 <= task["complexity"] <= 1

    def test_task_evaluation(self):
        """Test task evaluation logic."""
        suite = TaskSuite("coding_small")

        # Mock outputs that match expected keywords
        mock_outputs = [
            "def fibonacci(n): if n <= 1: return n; return fibonacci(n-1) + fibonacci(n-2)",
            "def quicksort(arr): pivot = arr[0]; return quicksort(left) + [pivot] + quicksort(right)",
            "class BinaryTree: def __init__(self): self.left = None; self.right = None",
        ]

        score = suite.evaluate(mock_outputs)

        assert 0 <= score <= 1
        assert score > 0  # Should get some points for keyword matching

    def test_evaluation_edge_cases(self):
        """Test evaluation with edge cases."""
        suite = TaskSuite("coding_small")

        # Test with wrong number of outputs
        score = suite.evaluate(["single output"])
        assert score == 0.0

        # Test with empty outputs
        empty_outputs = ["", "", ""]
        score = suite.evaluate(empty_outputs)
        assert score >= 0  # Should handle gracefully


class TestADASRunner:
    """Test the main ADAS runner orchestrator."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        model = Mock(spec=nn.Module)
        model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))

        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        tokenizer.decode = Mock(return_value="Generated output text")
        tokenizer.pad_token_id = 0

        return model, tokenizer

    def test_runner_initialization(self, mock_model_and_tokenizer, temp_archive_path):
        """Test ADAS runner initialization."""
        model, tokenizer = mock_model_and_tokenizer

        runner = ADASRunner(base_model=model, tokenizer=tokenizer, archive_path=temp_archive_path)

        assert runner.base_model is model
        assert runner.tokenizer is tokenizer
        assert isinstance(runner.archive, ADASArchive)
        assert isinstance(runner.proposer, ADASProposer)

    @patch("src.agent_forge.adas.runner.T2Mixer")
    @patch("torch.cuda.is_available", return_value=False)
    def test_single_proposal_evaluation(self, mock_cuda, mock_t2mixer, mock_model_and_tokenizer, temp_archive_path):
        """Test evaluation of a single proposal."""
        model, tokenizer = mock_model_and_tokenizer

        # Mock T2Mixer
        mock_mixer_instance = Mock()
        mock_mixer_instance.dispatch.return_value = {"expert_1": 0.8}
        mock_mixer_instance.patch = Mock()
        mock_mixer_instance.patch.__enter__ = Mock(return_value=model)
        mock_mixer_instance.patch.__exit__ = Mock()
        mock_t2mixer.return_value = mock_mixer_instance

        runner = ADASRunner(base_model=model, tokenizer=tokenizer, archive_path=temp_archive_path)

        # Mock proposal
        proposal = {
            "expert": {
                "layers": ["attn_qkv"],
                "rank": 2,
                "svd_scope": "per-matrix",
                "init": "random",
                "activation_rule": "gated",
                "budget": {"max_active": 2, "max_latency_ms": 100},
            },
            "dispatch": {
                "features": ["prompt_stats"],
                "mix_fn": "softmax",
                "granularity": "sequence",
            },
            "motivation": "Test proposal",
        }

        tasks = TaskSuite("coding_small")
        result = runner._evaluate_single_proposal(proposal, tasks, "test_001")

        assert isinstance(result, ExperimentResult)
        assert result.trial_id == "test_001"
        assert result.success is True
        assert result.score >= 0
        assert result.latency_ms >= 0
        assert result.vram_gb >= 0

    @patch("src.agent_forge.adas.runner.T2Mixer")
    @patch("torch.cuda.is_available", return_value=False)
    def test_batch_evaluation(self, mock_cuda, mock_t2mixer, mock_model_and_tokenizer, temp_archive_path):
        """Test batch evaluation of proposals."""
        model, tokenizer = mock_model_and_tokenizer

        # Mock T2Mixer
        mock_mixer_instance = Mock()
        mock_mixer_instance.dispatch.return_value = {"expert_1": 0.5}
        mock_mixer_instance.patch = Mock()
        mock_mixer_instance.patch.__enter__ = Mock(return_value=model)
        mock_mixer_instance.patch.__exit__ = Mock()
        mock_t2mixer.return_value = mock_mixer_instance

        runner = ADASRunner(base_model=model, tokenizer=tokenizer, archive_path=temp_archive_path)

        # Create multiple proposals
        proposals = []
        for i in range(3):
            proposals.append(
                {
                    "expert": {
                        "layers": ["attn_qkv"],
                        "rank": i + 1,
                        "svd_scope": "per-matrix",
                        "init": "random",
                        "activation_rule": "gated",
                        "budget": {"max_active": 2, "max_latency_ms": 100},
                    },
                    "dispatch": {
                        "features": ["prompt_stats"],
                        "mix_fn": "softmax",
                        "granularity": "sequence",
                    },
                    "motivation": f"Test proposal {i}",
                }
            )

        tasks = TaskSuite("coding_small")
        results = runner._evaluate_batch(proposals, tasks)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ExperimentResult)
            assert result.success is True

    @patch("src.agent_forge.adas.runner.T2Mixer")
    @patch("torch.cuda.is_available", return_value=False)
    def test_full_specialization_run(self, mock_cuda, mock_t2mixer, mock_model_and_tokenizer, temp_archive_path):
        """Test complete specialization run."""
        model, tokenizer = mock_model_and_tokenizer

        # Mock T2Mixer
        mock_mixer_instance = Mock()
        mock_mixer_instance.dispatch.return_value = {"expert_1": 0.6}
        mock_mixer_instance.patch = Mock()
        mock_mixer_instance.patch.__enter__ = Mock(return_value=model)
        mock_mixer_instance.patch.__exit__ = Mock()
        mock_t2mixer.return_value = mock_mixer_instance

        runner = ADASRunner(base_model=model, tokenizer=tokenizer, archive_path=temp_archive_path)

        # Run short specialization
        summary = runner.run_specialization(
            n_trials=4,
            time_budget_minutes=1,  # Short budget
            task_suite="coding_small",
            max_concurrent=2,
        )

        assert isinstance(summary, dict)
        assert "total_trials" in summary
        assert "successful_trials" in summary
        assert "success_rate" in summary
        assert "total_time_s" in summary

        assert summary["total_trials"] <= 4  # Should respect trial limit
        assert 0 <= summary["success_rate"] <= 1

        if summary["successful_trials"] > 0:
            assert "best_score" in summary
            assert "leaderboard" in summary
            assert len(summary["leaderboard"]) <= 10  # Should limit leaderboard size

    def test_vram_tracking(self, mock_model_and_tokenizer, temp_archive_path):
        """Test VRAM usage tracking."""
        model, tokenizer = mock_model_and_tokenizer

        runner = ADASRunner(base_model=model, tokenizer=tokenizer, archive_path=temp_archive_path)

        # Should work even without CUDA
        vram = runner._get_vram_usage()
        assert isinstance(vram, float)
        assert vram >= 0

    def test_summary_generation(self, temp_archive_path, mock_model_and_tokenizer):
        """Test summary generation from results."""
        model, tokenizer = mock_model_and_tokenizer

        runner = ADASRunner(base_model=model, tokenizer=tokenizer, archive_path=temp_archive_path)

        # Create mock results
        results = []
        for i in range(5):
            result = ExperimentResult(
                expert_spec={"layers": ["attn_qkv"], "rank": i + 1},
                dispatch_spec={"features": ["prompt_stats"]},
                score=0.5 + i * 0.1,
                latency_ms=100 - i * 5,
                vram_gb=0.1 + i * 0.02,
                trial_id=f"summary_test_{i:03d}",
                task_suite="coding_small",
                success=True,
            )
            results.append(result)
            runner.archive.add_result(result)

        summary = runner._generate_summary(results)

        assert summary["total_trials"] == 5
        assert summary["successful_trials"] == 5
        assert summary["success_rate"] == 1.0
        assert summary["best_score"] == 0.9  # Max score from mock data
        assert "leaderboard" in summary
        assert len(summary["leaderboard"]) <= 5


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_archive_corruption_handling(self, temp_archive_path):
        """Test handling of corrupted archive files."""
        # Create corrupted archive file
        with open(temp_archive_path, "w") as f:
            f.write("invalid json data\n{broken: json}")

        # Should handle gracefully
        archive = ADASArchive(temp_archive_path)
        results = archive.get_all_results()
        assert isinstance(results, list)  # Should return empty list or valid results

    def test_invalid_expert_specs(self):
        """Test handling of invalid expert specifications."""
        proposer = ADASProposer()

        # Test with extreme values
        proposal = proposer._generate_random_proposal(target_latency_ms=1)  # Very low budget

        # Should still generate valid proposal
        assert "expert" in proposal
        assert proposal["expert"]["rank"] >= 1

    def test_empty_archive_operations(self, temp_archive_path):
        """Test operations on empty archive."""
        archive = ADASArchive(temp_archive_path)

        # All operations should work on empty archive
        leaderboard = archive.get_leaderboard(top_k=5)
        assert len(leaderboard) == 0

        pareto = archive.get_pareto_frontier(["score", "latency_ms"])
        assert len(pareto) == 0

        stats = archive.get_statistics()
        assert stats["success_rate"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
