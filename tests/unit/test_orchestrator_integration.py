#!/usr/bin/env python3
"""
Integration tests for Agent Forge orchestrator with real phase modules.

Tests that the orchestrator can discover and execute all phase modules
with their new entry points, and that they return valid PhaseResult objects.
"""

import asyncio
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

from agent_forge.forge_orchestrator import ForgeOrchestrator, OrchestratorConfig, PhaseResult, PhaseStatus, PhaseType

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestOrchestratorIntegration:
    """Test orchestrator integration with real phase modules."""

    @pytest.fixture
    def orchestrator_config(self):
        """Create test orchestrator configuration."""
        return OrchestratorConfig(
            base_models=["test-model"],
            max_generations=2,
            device="cpu",
            quick_mode=True,
            output_dir=Path("./test_output"),
            checkpoint_dir=Path("./test_checkpoints"),
        )

    @pytest.fixture
    def orchestrator(self, orchestrator_config):
        """Create test orchestrator instance."""
        return ForgeOrchestrator(orchestrator_config)

    def test_orchestrator_initialization(self, orchestrator):
        """Test that orchestrator initializes correctly."""
        assert orchestrator.config is not None
        assert orchestrator.run_id is not None
        assert len(orchestrator.run_id) > 0

    def test_phase_discovery(self, orchestrator):
        """Test that orchestrator discovers phase modules."""
        orchestrator.discover_phase_modules()

        # Check that phases were discovered
        assert len(orchestrator.discovered_phases) > 0

        # Check for expected phases
        expected_phases = {
            PhaseType.EVOMERGE,
            PhaseType.COMPRESSION,
            PhaseType.GEOMETRY,
            PhaseType.SELF_MODELING,
            PhaseType.PROMPT_BAKING,
        }

        discovered_types = {phase.phase_type for phase in orchestrator.discovered_phases.values()}

        # At least some expected phases should be discovered
        assert len(expected_phases.intersection(discovered_types)) > 0

    def test_evomerge_entry_point_exists(self):
        """Test that EvoMerge phase has discoverable entry point."""
        try:
            from agent_forge.evomerge_pipeline import execute, run, run_evomerge

            assert callable(run_evomerge)
            assert callable(run)
            assert callable(execute)
        except ImportError:
            pytest.fail("EvoMerge entry points not found")

    def test_compression_entry_point_exists(self):
        """Test that Compression phase has discoverable entry point."""
        try:
            from agent_forge.compression_pipeline import execute, run, run_compression

            assert callable(run_compression)
            assert callable(run)
            assert callable(execute)
        except ImportError:
            pytest.fail("Compression entry points not found")

    def test_geometry_entry_point_exists(self):
        """Test that Geometry phase has discoverable entry point."""
        try:
            from agent_forge.geometry_feedback import execute, run, run_geometry

            assert callable(run_geometry)
            assert callable(run)
            assert callable(execute)
        except ImportError:
            pytest.fail("Geometry entry points not found")

    def test_self_modeling_entry_point_exists(self):
        """Test that Self-Modeling phase has discoverable entry point."""
        try:
            from agent_forge.mastery_loop import execute, run, run_self_modeling

            assert callable(run_self_modeling)
            assert callable(run)
            assert callable(execute)
        except ImportError:
            pytest.fail("Self-Modeling entry points not found")

    def test_prompt_baking_entry_point_exists(self):
        """Test that Prompt Baking phase has discoverable entry point."""
        try:
            from agent_forge.prompt_baking import execute, run, run_prompt_baking

            assert callable(run_prompt_baking)
            assert callable(run)
            assert callable(execute)
        except ImportError:
            pytest.fail("Prompt Baking entry points not found")

    @pytest.mark.asyncio
    async def test_evomerge_phase_contract(self):
        """Test that EvoMerge phase returns valid PhaseResult."""
        from agent_forge.evomerge_pipeline import run_evomerge

        # Mock config for quick test
        config = {
            "max_generations": 1,
            "base_models": [
                {"name": "test1", "path": "microsoft/DialoGPT-medium"},
            ],
            "device": "cpu",
            "output_dir": "./test_evomerge",
        }

        # Mock the heavy computation parts
        with patch("agent_forge.evomerge_pipeline.EvoMergePipeline") as mock_pipeline:
            mock_instance = MagicMock()
            mock_pipeline.return_value = mock_instance

            # Mock successful evolution
            mock_candidate = MagicMock()
            mock_candidate.model_path = "./test_model"
            mock_candidate.id = "test123"
            mock_candidate.overall_fitness = 0.75
            mock_candidate.generation = 1
            mock_candidate.merge_recipe = {"test": "recipe"}
            mock_candidate.evaluation_time = 10.0
            mock_candidate.creation_time.isoformat.return_value = "2024-01-01T00:00:00"
            mock_candidate.fitness_scores = {"test": 0.75}

            mock_instance.run_evolution = asyncio.coroutine(lambda: mock_candidate)
            mock_instance.state.current_generation = 1
            mock_instance.state.population = [mock_candidate]
            mock_instance.state.plateau_count = 0

            # Run the phase
            result = await run_evomerge(config)

            # Validate PhaseResult contract
            assert isinstance(result, PhaseResult)
            assert result.phase_type == PhaseType.EVOMERGE
            assert result.status in [PhaseStatus.COMPLETED, PhaseStatus.FAILED]
            assert result.start_time is not None
            assert result.end_time is not None
            assert result.duration_seconds is not None
            assert isinstance(result.metrics, dict)

            if result.status == PhaseStatus.COMPLETED:
                assert len(result.artifacts_produced) > 0
                assert result.error_message is None
            else:
                assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_compression_phase_contract(self):
        """Test that Compression phase returns valid PhaseResult."""
        from agent_forge.compression_pipeline import run_compression

        config = {
            "input_model_path": "./test_input",
            "output_model_path": "./test_output",
            "device": "cpu",
            "ab_test_samples": 10,
        }

        # Mock the compression pipeline
        with patch("agent_forge.compression_pipeline.CompressionPipeline") as mock_pipeline:
            mock_instance = MagicMock()
            mock_pipeline.return_value = mock_instance

            # Mock successful compression
            mock_instance.run_compression_pipeline = asyncio.coroutine(
                lambda: {
                    "success": True,
                    "model_path": "./test_compressed",
                    "compression_ratio": 4.0,
                    "memory_savings_mb": 100.0,
                    "evaluation_metrics": {"test": 0.8},
                }
            )

            result = await run_compression(config)

            # Validate PhaseResult contract
            assert isinstance(result, PhaseResult)
            assert result.phase_type == PhaseType.COMPRESSION
            assert result.status in [PhaseStatus.COMPLETED, PhaseStatus.FAILED]
            assert isinstance(result.metrics, dict)

    @pytest.mark.asyncio
    async def test_prompt_baking_phase_contract(self):
        """Test that Prompt Baking phase returns valid PhaseResult."""
        from agent_forge.prompt_baking import run_prompt_baking

        config = {
            "input_model_path": "./test_input",
            "output_model_path": "./test_output",
            "device": "cpu",
            "ab_test_samples": 5,
            "baking_epochs": 1,
        }

        # Mock the prompt baking pipeline
        with patch("agent_forge.prompt_baking.PromptBakingPipeline") as mock_pipeline:
            mock_instance = MagicMock()
            mock_pipeline.return_value = mock_instance

            # Mock successful baking
            mock_instance.run_prompt_baking_pipeline = asyncio.coroutine(
                lambda: {
                    "success": True,
                    "output_model_path": "./test_baked",
                    "best_prompt_template": "Test template: {task}",
                    "best_performance": 0.85,
                    "ab_test": {"best_variant": "variant_0", "best_performance": 0.85},
                    "tool_integration": {"calculator": {"success": True}},
                }
            )

            result = await run_prompt_baking(config)

            # Validate PhaseResult contract
            assert isinstance(result, PhaseResult)
            assert result.phase_type == PhaseType.PROMPT_BAKING
            assert result.status in [PhaseStatus.COMPLETED, PhaseStatus.FAILED]
            assert isinstance(result.metrics, dict)

    @pytest.mark.asyncio
    async def test_orchestrator_execution_flow(self, orchestrator):
        """Test full orchestrator execution flow with mocked phases."""

        # Mock W&B to avoid initialization issues
        with patch("wandb.init"), patch("wandb.log"), patch("wandb.finish"):
            # Mock phase execution to avoid heavy computation
            original_execute = orchestrator.execute_phase

            async def mock_execute_phase(phase_module, config):
                # Return a successful mock result
                return PhaseResult(
                    phase_type=PhaseType.EVOMERGE,  # Mock type
                    status=PhaseStatus.COMPLETED,
                    start_time=orchestrator._get_current_time(),
                    end_time=orchestrator._get_current_time(),
                    duration_seconds=1.0,
                    artifacts_produced=[],
                    metrics={"test": True},
                )

            orchestrator.execute_phase = mock_execute_phase

            # Run orchestrator
            results = await orchestrator.run_pipeline()

            # Validate results
            assert "pipeline_duration_seconds" in results
            assert "phases_completed" in results
            assert "success_rate" in results

            # Restore original method
            orchestrator.execute_phase = original_execute


class TestPhaseContracts:
    """Test that all phases follow the expected contract."""

    def test_phase_result_attributes(self):
        """Test PhaseResult has all required attributes."""
        from datetime import datetime

        result = PhaseResult(
            phase_type=PhaseType.EVOMERGE,
            status=PhaseStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=10.0,
            artifacts_produced=[],
            metrics={},
        )

        # Test required attributes exist
        assert hasattr(result, "phase_type")
        assert hasattr(result, "status")
        assert hasattr(result, "start_time")
        assert hasattr(result, "end_time")
        assert hasattr(result, "duration_seconds")
        assert hasattr(result, "artifacts_produced")
        assert hasattr(result, "metrics")
        assert hasattr(result, "error_message")
        assert hasattr(result, "warnings")
        assert hasattr(result, "todos")

        # Test success property
        assert result.success is True

        # Test failed result
        failed_result = PhaseResult(
            phase_type=PhaseType.EVOMERGE,
            status=PhaseStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=5.0,
            error_message="Test error",
        )

        assert failed_result.success is False

    def test_phase_type_enum(self):
        """Test that all expected phase types are defined."""
        expected_types = [
            "evomerge",
            "geometry",
            "self_modeling",
            "prompt_baking",
            "adas",
            "compression",
        ]

        for phase_type in expected_types:
            assert hasattr(PhaseType, phase_type.upper())

    def test_phase_status_enum(self):
        """Test that all expected phase statuses are defined."""
        expected_statuses = [
            "pending",
            "running",
            "completed",
            "failed",
            "skipped",
            "stub_detected",
        ]

        for status in expected_statuses:
            assert hasattr(PhaseStatus, status.upper())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
