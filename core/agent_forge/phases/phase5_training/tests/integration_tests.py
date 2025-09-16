"""
Comprehensive Integration Tests for Phase 5 Training Components
Tests all integration components to ensure seamless Agent Forge compatibility.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from training.phase5.integration.phase4_connector import Phase4Connector, BitNetCompatibility
from training.phase5.integration.phase6_preparer import Phase6Preparer, BakingReadiness
from training.phase5.integration.pipeline_validator import PipelineValidator, ValidationLevel, ValidationStatus
from training.phase5.integration.state_manager import CrossPhaseStateManager, StateType, StateStatus
from training.phase5.integration.mlops_coordinator import MLOpsCoordinator, PipelineConfig, PipelineStatus
from training.phase5.integration.quality_coordinator import QualityCoordinator, QualityGate, QualityGateType, Severity, GateStatus


class TestPhase4Connector:
    """Test Phase 4 BitNet integration connector."""

    @pytest.fixture
    async def connector(self):
        """Create connector for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "phase4_config.json"
            connector = Phase4Connector(config_path)
            await connector.initialize()
            yield connector

    @pytest.mark.asyncio
    async def test_initialize_success(self, connector):
        """Test successful initialization."""
        assert connector.phase4_config is not None
        assert connector.compatibility_cache == {}

    @pytest.mark.asyncio
    async def test_load_bitnet_model_compatible(self, connector):
        """Test loading compatible BitNet model."""
        success, model = await connector.load_bitnet_model("bitnet_v1.0")

        assert success is True
        assert model is not None
        assert model.get("phase5_enhancements") is not None

    @pytest.mark.asyncio
    async def test_load_bitnet_model_needs_conversion(self, connector):
        """Test loading model that needs conversion."""
        success, model = await connector.load_bitnet_model("legacy_model")

        assert success is True
        assert model is not None
        assert model.get("converted") is True

    @pytest.mark.asyncio
    async def test_load_bitnet_model_incompatible(self, connector):
        """Test loading incompatible model."""
        success, model = await connector.load_bitnet_model("incompatible_model")

        assert success is False
        assert model is None

    @pytest.mark.asyncio
    async def test_configure_quantization_training(self, connector):
        """Test quantization training configuration."""
        training_config = {
            "learning_rate": 1e-4,
            "batch_size": 32
        }

        enhanced_config = await connector.configure_quantization_training(training_config)

        assert "quantization" in enhanced_config
        assert "optimization" in enhanced_config
        assert enhanced_config["quantization"]["phase5_enhancements"]["dynamic_precision"] is True

    @pytest.mark.asyncio
    async def test_sync_performance_metrics(self, connector):
        """Test performance metrics synchronization."""
        phase5_metrics = {
            "accuracy": 0.93,
            "inference_speed": 95.0,
            "memory_usage": 0.7
        }

        sync_result = await connector.sync_performance_metrics(phase5_metrics)

        assert "phase4_targets" in sync_result
        assert "phase5_actual" in sync_result
        assert "comparisons" in sync_result

    @pytest.mark.asyncio
    async def test_prepare_model_export(self, connector):
        """Test model export preparation."""
        mock_model = {"model_id": "test_model", "trained": True}
        export_config = {"enhancements": {"quantized": True}}

        export_metadata = await connector.prepare_model_export(mock_model, export_config)

        assert "phase4_config" in export_metadata
        assert "compatibility_info" in export_metadata
        assert export_metadata["compatibility_info"]["phase6_ready"] is True


class TestPhase6Preparer:
    """Test Phase 6 baking preparation module."""

    @pytest.fixture
    async def preparer(self):
        """Create preparer for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preparer = Phase6Preparer(Path(temp_dir))
            await preparer.initialize()
            yield preparer

    @pytest.mark.asyncio
    async def test_initialize_success(self, preparer):
        """Test successful initialization."""
        assert preparer.export_dir.exists()
        assert (preparer.export_dir / "models").exists()
        assert (preparer.export_dir / "metadata").exists()

    @pytest.mark.asyncio
    async def test_assess_baking_readiness_ready(self, preparer):
        """Test baking readiness assessment - ready."""
        mock_model = {"model_id": "test_model", "trained": True}
        training_results = {
            "status": "completed",
            "final_metrics": {"accuracy": 0.92, "precision": 0.89},
            "training_history": [],
            "model_architecture": "transformer",
            "performance_metrics": {"inference_time": 0.08, "memory_usage": 0.6, "throughput": 120}
        }

        readiness, issues = await preparer.assess_baking_readiness(mock_model, training_results)

        assert readiness in [BakingReadiness.READY, BakingReadiness.NEEDS_VALIDATION]
        assert isinstance(issues, list)

    @pytest.mark.asyncio
    async def test_assess_baking_readiness_not_ready(self, preparer):
        """Test baking readiness assessment - not ready."""
        mock_model = {"model_id": "test_model"}
        training_results = {"status": "failed"}

        readiness, issues = await preparer.assess_baking_readiness(mock_model, training_results)

        assert readiness == BakingReadiness.NOT_READY
        assert len(issues) > 0

    @pytest.mark.asyncio
    async def test_prepare_export_package_success(self, preparer):
        """Test successful export package preparation."""
        mock_model = {"model_id": "test_model", "trained": True}
        training_results = {
            "status": "completed",
            "final_metrics": {"accuracy": 0.92, "precision": 0.89},
            "training_history": [],
            "model_architecture": "transformer",
            "performance_metrics": {"inference_time": 0.08, "memory_usage": 0.6, "throughput": 120}
        }

        package = await preparer.prepare_export_package(mock_model, training_results, "test_model_1")

        assert package is not None
        assert package.metadata.model_id == "test_model_1"
        assert package.checksum is not None
        assert len(package.model_data) > 0

    @pytest.mark.asyncio
    async def test_create_baking_manifest(self, preparer):
        """Test baking manifest creation."""
        # Create mock packages
        mock_model = {"model_id": "test_model"}
        training_results = {
            "status": "completed",
            "final_metrics": {"accuracy": 0.92},
            "training_history": [],
            "model_architecture": "transformer",
            "performance_metrics": {"inference_time": 0.08}
        }

        packages = []
        for i in range(2):
            package = await preparer.prepare_export_package(
                mock_model, training_results, f"test_model_{i}"
            )
            if package:
                packages.append(package)

        if packages:
            manifest = await preparer.create_baking_manifest(packages)

            assert "version" in manifest
            assert "packages" in manifest
            assert "summary" in manifest
            assert manifest["summary"]["total_packages"] == len(packages)

    @pytest.mark.asyncio
    async def test_validate_phase6_handoff(self, preparer):
        """Test Phase 6 handoff validation."""
        manifest = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "packages": [
                {
                    "model_id": "test_model_1",
                    "readiness_status": "ready",
                    "quality_score": 0.92
                },
                {
                    "model_id": "test_model_2",
                    "readiness_status": "ready",
                    "quality_score": 0.89
                }
            ],
            "summary": {
                "total_packages": 2,
                "ready_packages": 2,
                "total_size_mb": 100.0
            }
        }

        validation_result = await preparer.validate_phase6_handoff(manifest)

        assert "valid" in validation_result
        assert "statistics" in validation_result
        assert validation_result["statistics"]["ready_percentage"] == 100.0


class TestPipelineValidator:
    """Test end-to-end pipeline validator."""

    @pytest.fixture
    def validator(self):
        """Create validator for testing."""
        return PipelineValidator(ValidationLevel.COMPREHENSIVE)

    @pytest.mark.asyncio
    async def test_validate_pipeline_comprehensive(self, validator):
        """Test comprehensive pipeline validation."""
        config = {
            "model": {"architecture": "transformer", "num_layers": 12, "hidden_size": 768},
            "training": {"epochs": 10, "batch_size": 32, "learning_rate": 1e-4, "optimizer": "adam"},
            "data": {"train_path": "/mock/train", "validation_path": "/mock/val"},
            "monitoring": {"metrics": ["accuracy", "loss"], "logging_interval": 100},
            "quality_gates": {"accuracy_threshold": 0.85, "performance_threshold": 0.1},
            "integration": {"phase4_connector": True, "phase6_preparer": True}
        }

        report = await validator.validate_pipeline(config)

        assert report.validation_id is not None
        assert report.level == ValidationLevel.COMPREHENSIVE
        assert len(report.results) > 0
        assert report.summary is not None

    @pytest.mark.asyncio
    async def test_validate_phase_transition(self, validator):
        """Test phase transition validation."""
        transition_data = {
            "bitnet_model": {"model_id": "test_model"},
            "quantization_config": {"bits": 8},
            "phase4_metrics": {"accuracy": 0.90}
        }

        result = await validator.validate_phase_transition("phase4", "phase5", transition_data)

        assert result.name.startswith("phase_transition")
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.WARNING, ValidationStatus.FAILED]
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_validate_model_pipeline(self, validator):
        """Test model pipeline validation."""
        mock_model = {
            "forward": Mock(),
            "parameters": Mock(return_value=[]),
            "get": lambda x: {"phase5_enhancements": True} if x == "phase5_enhancements" else None
        }

        training_config = {
            "learning_rate": 1e-4,
            "batch_size": 32
        }

        validations = await validator.validate_model_pipeline(mock_model, training_config)

        assert "model_structure" in validations
        assert "training_compatibility" in validations
        assert "inference_capability" in validations


class TestCrossPhaseStateManager:
    """Test cross-phase state manager."""

    @pytest.fixture
    async def state_manager(self):
        """Create state manager for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CrossPhaseStateManager(Path(temp_dir))
            await manager.initialize()
            yield manager

    @pytest.mark.asyncio
    async def test_initialize_success(self, state_manager):
        """Test successful initialization."""
        assert state_manager.state_dir.exists()
        for phase_dir in state_manager.phase_dirs.values():
            assert phase_dir.exists()

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, state_manager):
        """Test state save and load operations."""
        test_data = {
            "model_config": {"architecture": "transformer"},
            "training_progress": {"epoch": 5, "loss": 0.1}
        }

        # Save state
        success = await state_manager.save_state("phase5", StateType.TRAINING_STATE, "test_state", test_data)
        assert success is True

        # Load state
        loaded_data = await state_manager.load_state("phase5", "test_state")
        assert loaded_data is not None
        assert loaded_data["model_config"]["architecture"] == "transformer"

    @pytest.mark.asyncio
    async def test_migrate_state(self, state_manager):
        """Test state migration between phases."""
        # Create initial state
        test_data = {"model": "phase4_model", "metrics": {"accuracy": 0.85}}
        await state_manager.save_state("phase4", StateType.MODEL_STATE, "test_migration", test_data)

        # Migrate state
        success = await state_manager.migrate_state("phase4", "phase5", "test_migration")
        assert success is True

        # Verify migrated state
        migrated_data = await state_manager.load_state("phase5", "test_migration")
        assert migrated_data is not None
        assert "phase5_enhancements" in migrated_data

    @pytest.mark.asyncio
    async def test_create_and_restore_checkpoint(self, state_manager):
        """Test checkpoint creation and restoration."""
        # Create test states
        test_states = ["state1", "state2", "state3"]
        for state_id in test_states:
            await state_manager.save_state(
                "phase5", StateType.MODEL_STATE, state_id,
                {"data": f"test_data_{state_id}"}
            )

        # Create checkpoint
        success = await state_manager.create_checkpoint("phase5", "test_checkpoint", test_states)
        assert success is True

        # Restore checkpoint
        success = await state_manager.restore_checkpoint("phase5", "test_checkpoint")
        assert success is True

    @pytest.mark.asyncio
    async def test_get_state_summary(self, state_manager):
        """Test state summary generation."""
        # Create test states
        await state_manager.save_state("phase5", StateType.MODEL_STATE, "test1", {"data": "test"})
        await state_manager.save_state("phase5", StateType.TRAINING_STATE, "test2", {"data": "test"})

        summary = await state_manager.get_state_summary("phase5")

        assert "total_states" in summary
        assert "by_phase" in summary
        assert "by_type" in summary
        assert summary["by_phase"]["phase5"] >= 2


class TestMLOpsCoordinator:
    """Test MLOps coordinator."""

    @pytest.fixture
    async def coordinator(self):
        """Create MLOps coordinator for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = MLOpsCoordinator(Path(temp_dir))
            await coordinator.initialize()
            yield coordinator

    @pytest.mark.asyncio
    async def test_initialize_success(self, coordinator):
        """Test successful initialization."""
        assert coordinator.workspace_dir.exists()
        assert coordinator.experiment_tracker is not None
        assert coordinator.model_registry is not None

    @pytest.mark.asyncio
    async def test_create_training_pipeline(self, coordinator):
        """Test training pipeline creation."""
        config = PipelineConfig(
            pipeline_id="test_pipeline",
            experiment_name="test_experiment",
            model_config={"architecture": "transformer"},
            training_config={"epochs": 10, "lr": 1e-4},
            validation_config={"split": 0.2},
            deployment_config={"target": "production"},
            monitoring_config={"metrics": ["accuracy"]},
            resource_config={"gpu": 1, "memory": "8GB"}
        )

        pipeline_id = await coordinator.create_training_pipeline(config)

        assert pipeline_id == "test_pipeline"
        assert pipeline_id in coordinator.active_pipelines

    @pytest.mark.asyncio
    async def test_execute_training_pipeline(self, coordinator):
        """Test training pipeline execution."""
        # Create pipeline first
        config = PipelineConfig(
            pipeline_id="test_execution",
            experiment_name="test_experiment",
            model_config={"architecture": "transformer"},
            training_config={"epochs": 5, "lr": 1e-4},
            validation_config={"split": 0.2},
            deployment_config={"target": "production"},
            monitoring_config={"metrics": ["accuracy"]},
            resource_config={"gpu": 1}
        )

        pipeline_id = await coordinator.create_training_pipeline(config)

        # Execute pipeline
        training_data = {"dataset": "mock_data", "size": 10000}
        results = await coordinator.execute_training_pipeline(pipeline_id, training_data)

        assert "model_training" in results
        assert results["model_training"]["success"] is True
        assert "model" in results["model_training"]

    @pytest.mark.asyncio
    async def test_monitor_training_progress(self, coordinator):
        """Test training progress monitoring."""
        # Create and start pipeline
        config = PipelineConfig(
            pipeline_id="test_monitoring",
            experiment_name="test_experiment",
            model_config={},
            training_config={},
            validation_config={},
            deployment_config={},
            monitoring_config={},
            resource_config={}
        )

        pipeline_id = await coordinator.create_training_pipeline(config)

        # Monitor progress
        progress = await coordinator.monitor_training_progress(pipeline_id)

        assert "pipeline_id" in progress
        assert "status" in progress
        assert "current_metrics" in progress

    @pytest.mark.asyncio
    async def test_get_pipeline_recommendations(self, coordinator):
        """Test pipeline recommendations."""
        # Create pipeline with metrics
        config = PipelineConfig(
            pipeline_id="test_recommendations",
            experiment_name="test_experiment",
            model_config={},
            training_config={},
            validation_config={},
            deployment_config={},
            monitoring_config={},
            resource_config={}
        )

        pipeline_id = await coordinator.create_training_pipeline(config)

        # Add some metrics to the experiment run
        pipeline_info = coordinator.active_pipelines[pipeline_id]
        experiment_run = pipeline_info["experiment_run"]
        experiment_run.metrics = {
            "training_accuracy": 0.75,  # Below threshold
            "gpu_utilization": 0.3,     # Low utilization
            "epoch_time_seconds": 400   # Long epoch time
        }

        recommendations = await coordinator.get_pipeline_recommendations(pipeline_id)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestQualityCoordinator:
    """Test quality gate coordinator."""

    @pytest.fixture
    async def coordinator(self):
        """Create quality coordinator for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = QualityCoordinator(Path(temp_dir))
            await coordinator.initialize()
            yield coordinator

    @pytest.mark.asyncio
    async def test_initialize_success(self, coordinator):
        """Test successful initialization."""
        assert coordinator.config_dir.exists()
        assert len(coordinator.quality_gates) > 0  # Default gates should be created

    @pytest.mark.asyncio
    async def test_register_quality_gate(self, coordinator):
        """Test quality gate registration."""
        gate = QualityGate(
            gate_id="test_gate",
            gate_type=QualityGateType.ACCURACY_THRESHOLD,
            name="Test Accuracy Gate",
            description="Test gate for accuracy validation",
            threshold_config={"min_accuracy": 0.9},
            severity=Severity.HIGH,
            enabled=True,
            phase="phase5"
        )

        success = await coordinator.register_quality_gate(gate)
        assert success is True
        assert "test_gate" in coordinator.quality_gates

    @pytest.mark.asyncio
    async def test_run_quality_checks(self, coordinator):
        """Test quality checks execution."""
        model_data = {"model_id": "test_model", "architecture": "transformer"}
        training_metrics = {
            "accuracy": 0.92,
            "precision": 0.88,
            "recall": 0.87,
            "final_accuracy": 0.92,
            "inference_time": 0.08,
            "throughput": 120,
            "memory_usage_gb": 4.2,
            "gpu_hours": 5.5
        }

        report = await coordinator.run_quality_checks("phase5", model_data, training_metrics)

        assert report.report_id is not None
        assert report.phase == "phase5"
        assert len(report.gate_results) > 0
        assert report.overall_status in [GateStatus.PASSED, GateStatus.WARNING, GateStatus.FAILED]

    @pytest.mark.asyncio
    async def test_validate_phase_transition(self, coordinator):
        """Test phase transition validation."""
        transition_data = {
            "export_package": {
                "validation_results": {"overall_valid": True},
                "metadata": {
                    "quality_scores": {
                        "accuracy": 0.90,
                        "precision": 0.85,
                        "recall": 0.83
                    }
                }
            }
        }

        passed, issues = await coordinator.validate_phase_transition("phase5", "phase6", transition_data)

        assert isinstance(passed, bool)
        assert isinstance(issues, list)

    @pytest.mark.asyncio
    async def test_get_quality_trends(self, coordinator):
        """Test quality trends analysis."""
        # This would normally require historical data
        # For testing, we'll check the structure
        trends = await coordinator.get_quality_trends("phase5", days=7)

        # Should return error for no historical data, or proper structure if data exists
        assert "error" in trends or "phase" in trends


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.fixture
    async def integration_setup(self):
        """Setup all integration components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Initialize all components
            phase4_connector = Phase4Connector()
            await phase4_connector.initialize()

            phase6_preparer = Phase6Preparer(temp_path / "exports")
            await phase6_preparer.initialize()

            pipeline_validator = PipelineValidator(ValidationLevel.COMPREHENSIVE)

            state_manager = CrossPhaseStateManager(temp_path / "state")
            await state_manager.initialize()

            mlops_coordinator = MLOpsCoordinator(temp_path / "mlops")
            await mlops_coordinator.initialize()

            quality_coordinator = QualityCoordinator(temp_path / "quality")
            await quality_coordinator.initialize()

            yield {
                "phase4_connector": phase4_connector,
                "phase6_preparer": phase6_preparer,
                "pipeline_validator": pipeline_validator,
                "state_manager": state_manager,
                "mlops_coordinator": mlops_coordinator,
                "quality_coordinator": quality_coordinator
            }

    @pytest.mark.asyncio
    async def test_complete_phase5_integration_workflow(self, integration_setup):
        """Test complete Phase 5 integration workflow."""
        components = integration_setup

        # 1. Phase 4 to Phase 5 transition
        phase4_connector = components["phase4_connector"]
        success, model = await phase4_connector.load_bitnet_model("bitnet_test_model")
        assert success is True
        assert model is not None

        # 2. Configure training with quantization
        training_config = {"learning_rate": 1e-4, "batch_size": 32, "epochs": 10}
        enhanced_config = await phase4_connector.configure_quantization_training(training_config)
        assert "quantization" in enhanced_config

        # 3. Validate pipeline configuration
        pipeline_validator = components["pipeline_validator"]
        full_config = {
            **enhanced_config,
            "model": {"architecture": "transformer"},
            "data": {"train_path": "/mock/train"},
            "monitoring": {"metrics": ["accuracy"]},
            "quality_gates": {"accuracy_threshold": 0.85},
            "integration": {"phase4_connector": True, "phase6_preparer": True}
        }

        validation_report = await pipeline_validator.validate_pipeline(full_config)
        assert validation_report.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING]

        # 4. Setup MLOps pipeline
        mlops_coordinator = components["mlops_coordinator"]
        mlops_config = PipelineConfig(
            pipeline_id="integration_test_pipeline",
            experiment_name="integration_test",
            model_config={"architecture": "transformer"},
            training_config=training_config,
            validation_config={"split": 0.2},
            deployment_config={"target": "test"},
            monitoring_config={"metrics": ["accuracy"]},
            resource_config={"gpu": 1}
        )

        pipeline_id = await mlops_coordinator.create_training_pipeline(mlops_config)
        assert pipeline_id is not None

        # 5. Execute training (mocked)
        training_data = {"dataset": "test_data", "size": 1000}
        training_results = await mlops_coordinator.execute_training_pipeline(pipeline_id, training_data)
        assert training_results["model_training"]["success"] is True

        # 6. Save training state
        state_manager = components["state_manager"]
        trained_model = training_results["model_training"]["model"]
        await state_manager.save_state("phase5", StateType.MODEL_STATE, "integration_test_model", {
            "model": trained_model,
            "training_config": enhanced_config,
            "training_results": training_results
        })

        # 7. Run quality checks
        quality_coordinator = components["quality_coordinator"]
        model_data = {"model": trained_model}
        training_metrics = {
            "accuracy": 0.91,
            "precision": 0.87,
            "recall": 0.86,
            "final_accuracy": 0.91,
            "inference_time": 0.07,
            "throughput": 130
        }

        quality_report = await quality_coordinator.run_quality_checks("phase5", model_data, training_metrics)
        assert quality_report.overall_status in [GateStatus.PASSED, GateStatus.WARNING]

        # 8. Prepare for Phase 6
        phase6_preparer = components["phase6_preparer"]
        mock_training_results = {
            "status": "completed",
            "final_metrics": training_metrics,
            "training_history": [],
            "model_architecture": "transformer",
            "performance_metrics": {"inference_time": 0.07, "memory_usage": 0.6, "throughput": 130}
        }

        readiness, issues = await phase6_preparer.assess_baking_readiness(trained_model, mock_training_results)
        assert readiness in [BakingReadiness.READY, BakingReadiness.NEEDS_VALIDATION]

        # 9. Create export package if ready
        if readiness != BakingReadiness.NOT_READY:
            export_package = await phase6_preparer.prepare_export_package(
                trained_model, mock_training_results, "integration_test_final"
            )
            assert export_package is not None
            assert export_package.metadata.model_id == "integration_test_final"

        # 10. Validate phase transition
        transition_data = {
            "trained_model": trained_model,
            "export_package": export_package.__dict__ if 'export_package' in locals() else None,
            "validation_results": quality_report.__dict__
        }

        transition_result = await pipeline_validator.validate_phase_transition(
            "phase5", "phase6", transition_data
        )
        assert transition_result.status in [ValidationStatus.PASSED, ValidationStatus.WARNING]

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, integration_setup):
        """Test error handling and recovery mechanisms."""
        components = integration_setup

        # Test Phase 4 connector with invalid model
        phase4_connector = components["phase4_connector"]
        success, model = await phase4_connector.load_bitnet_model("invalid_model")
        assert success is False
        assert model is None

        # Test state manager with corrupted data
        state_manager = components["state_manager"]
        # Try to load non-existent state
        loaded_data = await state_manager.load_state("phase5", "non_existent_state")
        assert loaded_data is None

        # Test quality coordinator with missing metrics
        quality_coordinator = components["quality_coordinator"]
        model_data = {}
        incomplete_metrics = {"accuracy": 0.5}  # Only one metric

        quality_report = await quality_coordinator.run_quality_checks("phase5", model_data, incomplete_metrics)
        # Should still generate a report, possibly with failures
        assert quality_report is not None
        assert isinstance(quality_report.gate_results, list)

    @pytest.mark.asyncio
    async def test_performance_and_scalability(self, integration_setup):
        """Test performance and scalability aspects."""
        components = integration_setup

        # Test multiple concurrent state operations
        state_manager = components["state_manager"]

        # Save multiple states concurrently
        tasks = []
        for i in range(10):
            task = state_manager.save_state(
                "phase5", StateType.METRICS, f"perf_test_{i}",
                {"metric": f"value_{i}", "timestamp": datetime.now().isoformat()}
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        assert all(results), "All state save operations should succeed"

        # Test batch quality checks
        quality_coordinator = components["quality_coordinator"]

        # Run multiple quality checks
        quality_tasks = []
        for i in range(5):
            model_data = {"model_id": f"test_model_{i}"}
            metrics = {"accuracy": 0.8 + (i * 0.02), "precision": 0.85}
            task = quality_coordinator.run_quality_checks("phase5", model_data, metrics)
            quality_tasks.append(task)

        quality_reports = await asyncio.gather(*quality_tasks, return_exceptions=True)
        successful_reports = [r for r in quality_reports if not isinstance(r, Exception)]
        assert len(successful_reports) > 0, "At least some quality checks should succeed"


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    import sys

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ], capture_output=True, text=True)

    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")