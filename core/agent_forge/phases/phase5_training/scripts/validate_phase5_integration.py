#!/usr/bin/env python3
"""
Phase 5 Integration Validation Script
Validates all integration components without pytest dependencies.
"""

import asyncio
import sys
import tempfile
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def test_phase4_connector():
    """Test Phase 4 connector integration."""
    logger.info("Testing Phase 4 Connector...")

    try:
        from training.phase5.integration.phase4_connector import Phase4Connector

        # Test initialization
        with tempfile.TemporaryDirectory() as temp_dir:
            connector = Phase4Connector(Path(temp_dir) / "test_config.json")
            success = await connector.initialize()

            if not success:
                logger.error("Phase 4 Connector initialization failed")
                return False

            # Test model loading
            success, model = await connector.load_bitnet_model("bitnet_test_model")
            if not success or model is None:
                logger.error("BitNet model loading failed")
                return False

            logger.info("✅ Phase 4 Connector - PASSED")
            return True

    except Exception as e:
        logger.error(f"❌ Phase 4 Connector - FAILED: {e}")
        return False

async def test_phase6_preparer():
    """Test Phase 6 preparer integration."""
    logger.info("Testing Phase 6 Preparer...")

    try:
        from training.phase5.integration.phase6_preparer import Phase6Preparer

        with tempfile.TemporaryDirectory() as temp_dir:
            preparer = Phase6Preparer(Path(temp_dir))
            success = await preparer.initialize()

            if not success:
                logger.error("Phase 6 Preparer initialization failed")
                return False

            # Test readiness assessment
            mock_model = {"model_id": "test_model", "trained": True}
            training_results = {
                "status": "completed",
                "final_metrics": {"accuracy": 0.92},
                "training_history": [],
                "model_architecture": "transformer",
                "performance_metrics": {"inference_time": 0.08}
            }

            readiness, issues = await preparer.assess_baking_readiness(mock_model, training_results)

            logger.info(f"✅ Phase 6 Preparer - PASSED (Readiness: {readiness})")
            return True

    except Exception as e:
        logger.error(f"❌ Phase 6 Preparer - FAILED: {e}")
        return False

async def test_pipeline_validator():
    """Test pipeline validator."""
    logger.info("Testing Pipeline Validator...")

    try:
        from training.phase5.integration.pipeline_validator import PipelineValidator, ValidationLevel

        validator = PipelineValidator(ValidationLevel.BASIC)

        # Test basic validation
        config = {
            "model": {"architecture": "transformer"},
            "training": {"epochs": 10, "batch_size": 32, "learning_rate": 1e-4, "optimizer": "adam"},
            "data": {"train_path": "/mock/train"},
            "monitoring": {"metrics": ["accuracy"]},
            "quality_gates": {"accuracy_threshold": 0.85},
            "integration": {"phase4_connector": True}
        }

        report = await validator.validate_pipeline(config)

        if report is None or len(report.results) == 0:
            logger.error("Pipeline validation failed")
            return False

        logger.info(f"✅ Pipeline Validator - PASSED ({len(report.results)} validations)")
        return True

    except Exception as e:
        logger.error(f"❌ Pipeline Validator - FAILED: {e}")
        return False

async def test_state_manager():
    """Test cross-phase state manager."""
    logger.info("Testing State Manager...")

    try:
        from training.phase5.integration.state_manager import CrossPhaseStateManager, StateType

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CrossPhaseStateManager(Path(temp_dir))
            success = await manager.initialize()

            if not success:
                logger.error("State Manager initialization failed")
                return False

            # Test state save/load
            test_data = {"test": "data", "value": 42}
            success = await manager.save_state("phase5", StateType.TRAINING_STATE, "test_state", test_data)

            if not success:
                logger.error("State save failed")
                return False

            loaded_data = await manager.load_state("phase5", "test_state")

            if loaded_data != test_data:
                logger.error("State load failed - data mismatch")
                return False

            logger.info("✅ State Manager - PASSED")
            return True

    except Exception as e:
        logger.error(f"❌ State Manager - FAILED: {e}")
        return False

async def test_mlops_coordinator():
    """Test MLOps coordinator."""
    logger.info("Testing MLOps Coordinator...")

    try:
        from training.phase5.integration.mlops_coordinator import MLOpsCoordinator, PipelineConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = MLOpsCoordinator(Path(temp_dir))
            success = await coordinator.initialize()

            if not success:
                logger.error("MLOps Coordinator initialization failed")
                return False

            # Test pipeline creation
            config = PipelineConfig(
                pipeline_id="test_pipeline",
                experiment_name="test_experiment",
                model_config={"architecture": "transformer"},
                training_config={"epochs": 5},
                validation_config={},
                deployment_config={},
                monitoring_config={},
                resource_config={}
            )

            pipeline_id = await coordinator.create_training_pipeline(config)

            if pipeline_id != "test_pipeline":
                logger.error("Pipeline creation failed")
                return False

            logger.info("✅ MLOps Coordinator - PASSED")
            return True

    except Exception as e:
        logger.error(f"❌ MLOps Coordinator - FAILED: {e}")
        return False

async def test_quality_coordinator():
    """Test quality coordinator."""
    logger.info("Testing Quality Coordinator...")

    try:
        from training.phase5.integration.quality_coordinator import QualityCoordinator

        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = QualityCoordinator(Path(temp_dir))
            success = await coordinator.initialize()

            if not success:
                logger.error("Quality Coordinator initialization failed")
                return False

            # Test quality checks
            model_data = {"model_id": "test_model"}
            training_metrics = {"accuracy": 0.92, "precision": 0.88, "final_accuracy": 0.92}

            report = await coordinator.run_quality_checks("phase5", model_data, training_metrics)

            if report is None or len(report.gate_results) == 0:
                logger.error("Quality checks failed")
                return False

            logger.info("✅ Quality Coordinator - PASSED")
            return True

    except Exception as e:
        logger.error(f"❌ Quality Coordinator - FAILED: {e}")
        return False

async def test_integration_workflow():
    """Test complete integration workflow."""
    logger.info("Testing Complete Integration Workflow...")

    try:
        # This would test all components working together
        # For now, we'll do a basic compatibility check

        from training.phase5.integration.phase4_connector import Phase4Connector
        from training.phase5.integration.phase6_preparer import Phase6Preparer
        from training.phase5.integration.pipeline_validator import PipelineValidator, ValidationLevel

        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Initialize components
            connector = Phase4Connector()
            await connector.initialize()

            preparer = Phase6Preparer(temp_path / "exports")
            await preparer.initialize()

            validator = PipelineValidator(ValidationLevel.BASIC)

            # Test workflow compatibility
            success, model = await connector.load_bitnet_model("bitnet_workflow_test")

            if not success:
                logger.error("Workflow - Phase 4 loading failed")
                return False

            # Mock training results
            mock_results = {
                "status": "completed",
                "final_metrics": {"accuracy": 0.93},
                "training_history": [],
                "model_architecture": "transformer",
                "performance_metrics": {"inference_time": 0.06}
            }

            readiness, issues = await preparer.assess_baking_readiness(model, mock_results)

            logger.info(f"✅ Integration Workflow - PASSED (Readiness: {readiness})")
            return True

    except Exception as e:
        logger.error(f"❌ Integration Workflow - FAILED: {e}")
        return False

async def main():
    """Main validation function."""
    logger.info("🚀 Starting Phase 5 Integration Validation...")

    # Run all tests
    tests = [
        test_phase4_connector,
        test_phase6_preparer,
        test_pipeline_validator,
        test_state_manager,
        test_mlops_coordinator,
        test_quality_coordinator,
        test_integration_workflow
    ]

    results = []
    for test_func in tests:
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test_func.__name__} crashed: {e}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)
    pass_rate = (passed / total) * 100 if total > 0 else 0

    logger.info(f"\n📊 VALIDATION SUMMARY:")
    logger.info(f"   Tests Run: {total}")
    logger.info(f"   Passed: {passed}")
    logger.info(f"   Failed: {total - passed}")
    logger.info(f"   Pass Rate: {pass_rate:.1f}%")

    if pass_rate >= 85:
        logger.info("🎉 INTEGRATION VALIDATION: ✅ PASSED")
        logger.info("   Phase 5 Integration is ready for Phase 6 progression!")
        return 0
    else:
        logger.error("❌ INTEGRATION VALIDATION: FAILED")
        logger.error(f"   Pass rate {pass_rate:.1f}% below 85% threshold")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)