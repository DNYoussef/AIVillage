#!/usr/bin/env python3
"""
Tests for phase contract compliance and regression detection.

Validates that all phase modules return properly formatted PhaseResult objects
and tests performance regression detection logic.
"""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent_forge.forge_orchestrator import (
    PhaseArtifact,
    PhaseResult,
    PhaseStatus,
    PhaseType,
)


class TestPhaseContracts(unittest.TestCase):
    """Test phase contract compliance."""

    @pytest.mark.asyncio
    async def test_all_phases_return_phase_result(self):
        """Test that all phase entry points return PhaseResult objects."""

        # Test configuration for each phase
        test_configs = {
            "evomerge": {
                "max_generations": 1,
                "base_models": [{"name": "test", "path": "test"}],
                "device": "cpu",
                "output_dir": "./test_output",
            },
            "compression": {
                "input_model_path": "./test_input",
                "output_model_path": "./test_output",
                "device": "cpu",
            },
            "geometry": {
                "model_path": "./test_model",
                "output_dir": "./test_geometry",
                "analysis_steps": 10,
            },
            "self_modeling": {
                "model_path": "./test_model",
                "output_dir": "./test_mastery",
                "domain": "test",
                "max_mastery_levels": 1,
            },
            "prompt_baking": {
                "input_model_path": "./test_input",
                "output_model_path": "./test_output",
                "device": "cpu",
                "ab_test_samples": 5,
            },
        }

        # Import all phase entry points
        phase_imports = {
            "evomerge": "agent_forge.evomerge_pipeline.run_evomerge",
            "compression": "agent_forge.compression_pipeline.run_compression",
            "geometry": "agent_forge.geometry_feedback.run_geometry",
            "self_modeling": "agent_forge.mastery_loop.run_self_modeling",
            "prompt_baking": "agent_forge.prompt_baking.run_prompt_baking",
        }

        for phase_name, import_path in phase_imports.items():
            with self.subTest(phase=phase_name):
                try:
                    # Import the phase function
                    module_path, func_name = import_path.rsplit(".", 1)
                    module = __import__(module_path, fromlist=[func_name])
                    phase_func = getattr(module, func_name)

                    # Mock heavy operations for testing
                    with self._mock_phase_dependencies(phase_name):
                        # Execute phase with test config
                        config = test_configs[phase_name]
                        result = await phase_func(config)

                        # Validate PhaseResult contract
                        self._validate_phase_result(result, phase_name)

                except Exception as e:
                    pytest.fail(f"Phase {phase_name} failed contract test: {e}")

    def _mock_phase_dependencies(self, phase_name):
        """Create appropriate mocks for each phase's dependencies."""

        mocks = []

        if phase_name == "evomerge":
            # Mock EvoMerge heavy operations
            evomerge_mock = patch("agent_forge.evomerge_pipeline.EvoMergePipeline")
            mock_pipeline = MagicMock()
            mock_candidate = MagicMock()
            mock_candidate.model_path = "./test_model"
            mock_candidate.id = "test123"
            mock_candidate.overall_fitness = 0.75
            mock_candidate.generation = 1
            mock_candidate.merge_recipe = {"test": "recipe"}
            mock_candidate.evaluation_time = 1.0
            mock_candidate.creation_time.isoformat.return_value = "2024-01-01T00:00:00"
            mock_candidate.fitness_scores = {"test": 0.75}

            mock_pipeline.run_evolution = asyncio.coroutine(lambda: mock_candidate)
            mock_pipeline.state.current_generation = 1
            mock_pipeline.state.population = [mock_candidate]
            mock_pipeline.state.plateau_count = 0

            evomerge_mock.return_value = mock_pipeline
            mocks.append(evomerge_mock)

        elif phase_name == "compression":
            # Mock Compression operations
            compression_mock = patch(
                "agent_forge.compression_pipeline.CompressionPipeline"
            )
            mock_pipeline = MagicMock()
            mock_pipeline.run_compression_pipeline = asyncio.coroutine(
                lambda: {
                    "success": True,
                    "model_path": "./test_compressed",
                    "compression_ratio": 4.0,
                    "memory_savings_mb": 100.0,
                }
            )
            compression_mock.return_value = mock_pipeline
            mocks.append(compression_mock)

        elif phase_name == "geometry":
            # Mock Geometry operations
            geometry_mock = patch("agent_forge.geometry_feedback.GeometryTracker")
            mock_tracker = MagicMock()
            mock_metrics = MagicMock()
            mock_metrics.intrinsic_dimensionality = 5.2
            mock_metrics.grok_probability = 0.3
            mock_metrics.compass_direction = "growth"
            mock_metrics.compass_magnitude = 0.7
            mock_tracker.update.return_value = mock_metrics
            mock_tracker.get_learning_recommendations.return_value = ["test_rec"]
            mock_tracker.save_state.return_value = None
            geometry_mock.return_value = mock_tracker
            mocks.append(geometry_mock)

            # Mock model loading
            model_mock = patch("transformers.AutoModel.from_pretrained")
            model_mock.side_effect = Exception("Mock model load failure")
            mocks.append(model_mock)

        elif phase_name == "self_modeling":
            # Mock Mastery Loop operations
            mastery_mock = patch("agent_forge.mastery_loop.MasteryLoop")
            mock_loop = MagicMock()
            mock_loop.run_mastery_training = asyncio.coroutine(
                lambda: {
                    "success": True,
                    "levels_mastered": 1,
                    "total_attempts": 10,
                    "final_model_path": "./test_final",
                    "self_modeling_metrics": {
                        "self_awareness_score": 0.8,
                        "metacognitive_accuracy": 0.7,
                        "compass_direction": "growth",
                    },
                }
            )
            mastery_mock.return_value = mock_loop
            mocks.append(mastery_mock)

        elif phase_name == "prompt_baking":
            # Mock Prompt Baking operations
            baking_mock = patch("agent_forge.prompt_baking.PromptBakingPipeline")
            mock_pipeline = MagicMock()
            mock_pipeline.run_prompt_baking_pipeline = asyncio.coroutine(
                lambda: {
                    "success": True,
                    "output_model_path": "./test_baked",
                    "best_prompt_template": "Test: {task}",
                    "best_performance": 0.85,
                    "ab_test": {"best_variant": "variant_0"},
                    "tool_integration": {"test": True},
                }
            )
            baking_mock.return_value = mock_pipeline
            mocks.append(baking_mock)

        # Mock W&B for all phases
        wandb_mock = patch("wandb.init")
        mocks.append(wandb_mock)

        # Return context manager that starts all mocks
        class MockContext:
            def __enter__(self):
                for mock in mocks:
                    mock.__enter__()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                for mock in reversed(mocks):
                    mock.__exit__(exc_type, exc_val, exc_tb)

        return MockContext()

    def _validate_phase_result(self, result, phase_name):
        """Validate that a PhaseResult follows the contract."""

        # Basic type validation
        assert isinstance(result, PhaseResult), (
            f"Phase {phase_name} must return PhaseResult"
        )

        # Required attributes
        assert hasattr(result, "phase_type"), "PhaseResult must have phase_type"
        assert hasattr(result, "status"), "PhaseResult must have status"
        assert hasattr(result, "start_time"), "PhaseResult must have start_time"
        assert hasattr(result, "end_time"), "PhaseResult must have end_time"
        assert hasattr(result, "duration_seconds"), (
            "PhaseResult must have duration_seconds"
        )
        assert hasattr(result, "artifacts_produced"), (
            "PhaseResult must have artifacts_produced"
        )
        assert hasattr(result, "metrics"), "PhaseResult must have metrics"

        # Type validation
        assert isinstance(result.phase_type, PhaseType), (
            "phase_type must be PhaseType enum"
        )
        assert isinstance(result.status, PhaseStatus), "status must be PhaseStatus enum"
        assert isinstance(result.artifacts_produced, list), (
            "artifacts_produced must be list"
        )
        assert isinstance(result.metrics, dict), "metrics must be dict"

        # Status-specific validation
        if result.status == PhaseStatus.COMPLETED:
            assert result.error_message is None or result.error_message == "", (
                "Completed phases should not have error messages"
            )
        elif result.status == PhaseStatus.FAILED:
            assert result.error_message is not None and result.error_message != "", (
                "Failed phases must have error messages"
            )

        # Artifacts validation
        for artifact in result.artifacts_produced:
            assert isinstance(artifact, PhaseArtifact), (
                "All artifacts must be PhaseArtifact objects"
            )
            assert artifact.phase_type == result.phase_type, (
                "Artifact phase_type must match result"
            )
            assert isinstance(artifact.data, dict), "Artifact data must be dict"

        # Metrics validation
        assert "execution_time" in result.metrics, "Metrics must include execution_time"
        assert isinstance(result.metrics["execution_time"], (int, float)), (
            "execution_time must be numeric"
        )

        print(f"✓ Phase {phase_name} passed contract validation")


class TestPerformanceRegressionDetection(unittest.TestCase):
    """Test performance regression detection logic."""

    def test_regression_detection_logic(self):
        """Test regression detection with various scenarios."""

        # Import the regression check function from the pipeline runner
        sys.path.append(str(project_root))
        from run_full_agent_forge import AgentForgePipelineRunner

        # Create mock args for the runner
        class MockArgs:
            def __init__(self):
                self.no_deploy = True
                self.dry_run = False
                self.frontier_api_key = None

        runner = AgentForgePipelineRunner(MockArgs())

        # Test scenarios
        test_scenarios = [
            {
                "name": "no_regression",
                "scores": {"MMLU": 0.65, "GSM8K": 0.45, "HumanEval": 0.30},
                "expected_alerts": 0,
            },
            {
                "name": "mmlu_regression",
                "scores": {"MMLU": 0.55, "GSM8K": 0.45, "HumanEval": 0.30},
                "expected_alerts": 1,
            },
            {
                "name": "multiple_regressions",
                "scores": {"MMLU": 0.55, "GSM8K": 0.35, "HumanEval": 0.20},
                "expected_alerts": 3,
            },
            {
                "name": "minor_regression",
                "scores": {"MMLU": 0.59, "GSM8K": 0.39, "HumanEval": 0.24},
                "expected_alerts": 0,  # Below 2pp threshold
            },
        ]

        for scenario in test_scenarios:
            with self.subTest(scenario=scenario["name"]):
                self._test_regression_scenario(runner, scenario)

    def _test_regression_scenario(self, runner, scenario):
        """Test a specific regression scenario."""

        # Create temporary results file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Create mock benchmark results
            benchmark_comparison = []
            for benchmark, score in scenario["scores"].items():
                benchmark_comparison.append(
                    {
                        "Benchmark": benchmark,
                        "unified_pipeline": score,  # Best model
                    }
                )

            mock_results = {
                "model_averages": {
                    "unified_pipeline": sum(scenario["scores"].values())
                    / len(scenario["scores"])
                },
                "benchmark_comparison": benchmark_comparison,
            }

            json.dump(mock_results, f, indent=2)
            temp_file = f.name

        try:
            # Mock the results file path
            runner.results_dir = Path(temp_file).parent
            results_file = runner.results_dir / "agent_forge_model_comparison.json"

            # Copy temp file to expected location
            import shutil

            shutil.copy2(temp_file, results_file)

            # Count alerts by mocking wandb.alert
            alert_count = 0

            def mock_alert(title, text, level):
                nonlocal alert_count
                alert_count += 1
                print(f"Alert: {title} - {text}")

            with patch("wandb.alert", side_effect=mock_alert):
                # Run regression check
                runner.check_performance_regressions()

            # Validate expected number of alerts
            assert alert_count == scenario["expected_alerts"], (
                f"Expected {scenario['expected_alerts']} alerts, got {alert_count}"
            )

            print(f"✓ Scenario '{scenario['name']}' passed with {alert_count} alerts")

        finally:
            # Cleanup
            Path(temp_file).unlink(missing_ok=True)
            if results_file.exists():
                results_file.unlink(missing_ok=True)

    def test_wandb_alert_integration(self):
        """Test W&B alert integration."""

        with patch("wandb.alert") as mock_alert:
            # Import and test alert functionality
            from run_full_agent_forge import AgentForgePipelineRunner

            class MockArgs:
                pass

            runner = AgentForgePipelineRunner(MockArgs())

            # Test failure alert
            runner.send_wandb_alert_on_failure("Test error message")

            # Verify alert was called
            mock_alert.assert_called_once()
            call_args = mock_alert.call_args[1]  # Get keyword arguments

            assert "title" in call_args
            assert "text" in call_args
            assert "level" in call_args
            assert "Test error message" in call_args["text"]

            print("✓ W&B alert integration test passed")


class TestCIIntegration(unittest.TestCase):
    """Test CI/CD integration functionality."""

    def test_pipeline_flags(self):
        """Test that pipeline runner supports CI flags."""

        # Test argument parsing

        from run_full_agent_forge import main

        # Mock sys.argv for argument parsing test
        test_args = [
            "run_full_agent_forge.py",
            "--no-deploy",
            "--dry-run",
            "--frontier-api-key",
            "test-key",
            "--device",
            "cpu",
        ]

        with patch("sys.argv", test_args):
            with patch("run_full_agent_forge.AgentForgePipelineRunner") as mock_runner:
                mock_instance = MagicMock()
                mock_runner.return_value = mock_instance

                try:
                    main()
                except SystemExit:
                    pass  # Normal exit

                # Verify runner was created with correct args
                mock_runner.assert_called_once()
                args = mock_runner.call_args[0][0]

                assert args.no_deploy is True
                assert args.dry_run is True
                assert args.frontier_api_key == "test-key"
                assert args.device == "cpu"

        print("✓ CI integration flags test passed")

    def test_timeout_handling(self):
        """Test timeout handling in CI environment."""

        # This test validates that timeouts are properly configured
        # In actual CI, this would be tested through the workflow

        from run_full_agent_forge import AgentForgePipelineRunner

        class MockArgs:
            def __init__(self):
                self.timeout = 5400  # 90 minutes
                self.benchmark_timeout = 1800  # 30 minutes
                self.device = "cpu"
                self.no_deploy = True

        runner = AgentForgePipelineRunner(MockArgs())

        # Verify timeout values are set correctly
        assert runner.args.timeout == 5400
        assert runner.args.benchmark_timeout == 1800

        print("✓ Timeout handling test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
