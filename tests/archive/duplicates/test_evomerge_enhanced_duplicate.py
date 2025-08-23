"""
Unit tests for enhanced EvoMerge functionality including:
- Dual-phase generation loop
- Early stopping logic
- Benchmark suite integration
- Multi-objective optimization
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.production.evolution.evomerge.bench_orchestrator import BenchmarkOrchestrator, determine_model_suite
from src.production.evolution.evomerge.evolutionary_tournament import EvolutionaryTournament


class TestBenchmarkOrchestrator(unittest.TestCase):
    """Test benchmark orchestrator functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.aiv_root = self.temp_dir
        os.environ["AIV_ROOT"] = self.aiv_root

        # Create benchmark suites directory
        suites_dir = Path(self.aiv_root) / "benchmarks" / "suites"
        suites_dir.mkdir(parents=True, exist_ok=True)

        # Create test suite file
        test_suite = {
            "objectives": ["mmlu_score", "gsm8k_score"],
            "task_groups": [{"name": "core", "tasks": ["mmlu", "gsm8k"]}],
        }

        with open(suites_dir / "test.yaml", "w") as f:
            import yaml

            yaml.dump(test_suite, f)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_suite_config(self):
        """Test loading benchmark suite configuration."""
        orchestrator = BenchmarkOrchestrator("test", wandb_mode="disabled")

        self.assertEqual(orchestrator.suite_name, "test")
        self.assertEqual(orchestrator.suite_config.objectives, ["mmlu_score", "gsm8k_score"])
        self.assertEqual(len(orchestrator.suite_config.task_groups), 1)

    def test_get_task_list(self):
        """Test extracting task list from suite configuration."""
        orchestrator = BenchmarkOrchestrator("test", wandb_mode="disabled")
        tasks = orchestrator._get_task_list()

        expected_tasks = ["mmlu", "gsm8k"]
        self.assertEqual(sorted(tasks), sorted(expected_tasks))

    def test_objective_mapping(self):
        """Test mapping objectives to result keys."""
        orchestrator = BenchmarkOrchestrator("test", wandb_mode="disabled")

        mock_results = {"results": {"mmlu": {"acc": 0.85}, "gsm8k": {"exact_match": 0.72}}}

        mmlu_key = orchestrator._map_objective_to_result_key("mmlu_score", mock_results)
        gsm8k_key = orchestrator._map_objective_to_result_key("gsm8k_score", mock_results)

        self.assertEqual(mmlu_key, "mmlu")
        self.assertEqual(gsm8k_key, "gsm8k")

    def test_extract_objective_scores(self):
        """Test extracting objective scores from lm_eval results."""
        orchestrator = BenchmarkOrchestrator("test", wandb_mode="disabled")

        mock_results = {
            "results": {
                "mmlu": {"acc": 0.85, "acc_norm": 0.87},
                "gsm8k": {"exact_match": 0.72, "acc": 0.70},
            }
        }

        scores = orchestrator._extract_objective_scores(mock_results)

        self.assertEqual(scores["mmlu_score"], 0.85)  # Should prefer 'acc'
        self.assertEqual(scores["gsm8k_score"], 0.70)  # Should prefer 'acc' over 'exact_match'


class TestModelSuiteDetection(unittest.TestCase):
    """Test automatic model suite detection."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.aiv_root = self.temp_dir
        os.environ["AIV_ROOT"] = self.aiv_root

        # Create models.yaml
        models_dir = Path(self.aiv_root) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        models_config = {
            "models": [
                {
                    "id": "test/coder-model",
                    "local": "D:\\AIVillage\\models\\test_coder",
                    "type": "coding",
                },
                {
                    "id": "test/math-model",
                    "local": "D:\\AIVillage\\models\\test_math",
                    "type": "math",
                },
            ]
        }

        with open(models_dir / "models.yaml", "w") as f:
            import yaml

            yaml.dump(models_config, f)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_determine_model_suite_from_metadata(self):
        """Test determining suite from model metadata."""
        # Test coding model
        suite = determine_model_suite("D:\\AIVillage\\models\\test_coder")
        self.assertEqual(suite, "coding")

        # Test math model
        suite = determine_model_suite("D:\\AIVillage\\models\\test_math")
        self.assertEqual(suite, "math")

        # Test unknown model (should default to general)
        suite = determine_model_suite("D:\\AIVillage\\models\\unknown_model")
        self.assertEqual(suite, "general")

    def test_determine_model_suite_with_override(self):
        """Test suite override functionality."""
        suite = determine_model_suite("any/path", override_suite="writing")
        self.assertEqual(suite, "writing")


class TestEvolutionaryTournament(unittest.TestCase):
    """Test evolutionary tournament functionality."""

    def setUp(self):
        """Set up test environment."""
        from src.production.evolution.evomerge.config import Configuration, EvolutionSettings, MergeSettings

        self.config = Configuration(
            models=[],
            merge_settings=MergeSettings(),
            evolution_settings=EvolutionSettings(),
        )

    def test_early_stopping_logic(self):
        """Test early stopping criteria."""
        tournament = EvolutionaryTournament(self.config)

        # Test insufficient history
        fitness_history = [0.8, 0.82]
        should_stop = tournament.should_early_stop(fitness_history, 0.5, 3)
        self.assertFalse(should_stop)

        # Test significant improvement (should continue)
        fitness_history = [0.8, 0.82, 0.84, 0.86, 0.90]
        should_stop = tournament.should_early_stop(fitness_history, 0.5, 3)
        self.assertFalse(should_stop)

        # Test minimal improvement (should stop)
        fitness_history = [0.8, 0.82, 0.84, 0.841, 0.842]
        should_stop = tournament.should_early_stop(fitness_history, 0.5, 3)
        self.assertTrue(should_stop)

    @patch("src.production.evolution.evomerge.evolutionary_tournament.AdvancedModelMerger")
    @patch("src.production.evolution.evomerge.evolutionary_tournament.BenchmarkOrchestrator")
    def test_create_generation_children(self, mock_orchestrator, mock_merger):
        """Test generation children creation."""
        tournament = EvolutionaryTournament(self.config)

        # Mock merger behavior
        mock_merger_instance = Mock()
        mock_merger_instance.merge.return_value = "/tmp/merged_model"
        mock_merger.return_value = mock_merger_instance

        # Test children creation
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["AIV_ROOT"] = temp_dir

            parents = ["/path/parent1", "/path/parent2", "/path/parent3"]

            with patch("shutil.move"), patch("shutil.copytree"):
                children = tournament.create_generation_children(parents, generation=0, phase="pre", target_count=3)

            self.assertEqual(len(children), 3)
            self.assertTrue(all("child_" in child for child in children))

    def test_fitness_calculation(self):
        """Test aggregated fitness calculation in evolve method."""
        EvolutionaryTournament(self.config)

        # Mock scores for testing
        mock_scores = [
            {"mmlu_score": 0.8, "gsm8k_score": 0.7},
            {"mmlu_score": 0.75, "gsm8k_score": 0.72},
            {"mmlu_score": 0.78, "gsm8k_score": 0.68},
        ]

        # Calculate expected best aggregated fitness
        expected_best = max(sum(score.values()) / len(score) for score in mock_scores)

        # Test score from first entry: (0.8 + 0.7) / 2 = 0.75
        # Test score from second entry: (0.75 + 0.72) / 2 = 0.735
        # Test score from third entry: (0.78 + 0.68) / 2 = 0.73
        # Max should be 0.75

        self.assertAlmostEqual(expected_best, 0.75, places=3)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete EvoMerge scenarios."""

    @patch("src.production.evolution.evomerge.evolutionary_tournament.BenchmarkOrchestrator")
    @patch("src.production.evolution.evomerge.evolutionary_tournament.AdvancedModelMerger")
    def test_dual_phase_benchmarking(self, mock_merger, mock_orchestrator):
        """Test complete dual-phase benchmarking workflow."""
        from src.production.evolution.evomerge.config import Configuration, EvolutionSettings, MergeSettings

        config = Configuration(
            models=[],
            merge_settings=MergeSettings(),
            evolution_settings=EvolutionSettings(),
        )

        tournament = EvolutionaryTournament(config)

        # Mock benchmark orchestrator
        mock_orch_instance = Mock()
        mock_orch_instance.benchmark_model.return_value = {
            "mmlu_score": 0.8,
            "gsm8k_score": 0.7,
            "hellaswag_score": 0.75,
        }
        mock_orchestrator.return_value = mock_orch_instance

        # Mock merger
        mock_merger_instance = Mock()
        mock_merger_instance.merge.return_value = "/tmp/merged"
        mock_merger.return_value = mock_merger_instance

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["AIV_ROOT"] = temp_dir

            # Test that both phases are called with correct parameters
            with patch.object(tournament, "create_generation_children") as mock_create:
                mock_create.side_effect = [
                    [f"/tmp/pre_child_{i}" for i in range(8)],  # pre phase
                    [f"/tmp/post_child_{i}" for i in range(8)],  # post phase
                ]

                with patch("src.production.evolution.evomerge.evolutionary_tournament.nsga2_select") as mock_nsga:
                    mock_nsga.side_effect = [
                        (
                            [f"/tmp/pre_child_{i}" for i in range(3)],
                            [{"score": 0.8}] * 3,
                        ),  # pre selection
                        (
                            [f"/tmp/post_child_{i}" for i in range(3)],
                            [{"score": 0.8}] * 3,
                        ),  # post selection
                    ]

                    with patch.object(tournament, "create_initial_population") as mock_init:
                        mock_init.return_value = [
                            "/tmp/parent1",
                            "/tmp/parent2",
                            "/tmp/parent3",
                        ]

                        with patch.object(tournament, "calculate_diversity") as mock_div:
                            mock_div.return_value = 0.5

                            # Run single generation
                            tournament.evolve(max_gens=1, suite="general", phase_both=True)

                            # Verify both phases were executed
                            self.assertEqual(mock_create.call_count, 2)

                            # Verify benchmark was called for both phases
                            expected_calls = 8 + 8 + 3  # pre + post + final
                            self.assertEqual(
                                mock_orch_instance.benchmark_model.call_count,
                                expected_calls,
                            )


if __name__ == "__main__":
    # Set up test environment
    os.environ.setdefault("AIV_ROOT", "D:\\AIVillage")
    os.environ.setdefault("WANDB_MODE", "disabled")

    unittest.main()
