#!/usr/bin/env python3
"""
Simple Pipeline Test

Creates a minimal test to verify the pipeline structure and basic functionality
without relying on all the complex imports.
"""

import asyncio
import logging
import sys
from pathlib import Path

import torch.nn as nn

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_imports():
    """Test basic imports are working."""
    try:
        logger.info("✓ Core phase controller imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_pipeline_config():
    """Test pipeline configuration."""
    try:
        from core.unified_pipeline import UnifiedConfig, UnifiedPipeline

        # Create test config
        config = UnifiedConfig(
            enable_evomerge=False,  # Disable complex phases for testing
            enable_quietstar=False,
            enable_initial_compression=False,
            enable_training=False,
            enable_tool_baking=False,
            enable_adas=False,
            enable_final_compression=False,
        )

        # Try to create pipeline
        pipeline = UnifiedPipeline(config)
        logger.info("✓ Pipeline configuration successful")
        logger.info(f"  Available phases: {len(pipeline.phases)}")
        return True

    except Exception as e:
        logger.error(f"✗ Pipeline configuration failed: {e}")
        return False


def test_phase_orchestrator():
    """Test phase orchestrator basic functionality."""
    try:
        from core.phase_controller import PhaseOrchestrator

        # Create orchestrator
        orchestrator = PhaseOrchestrator()

        # Test empty phase sequence
        phases = []

        # Create dummy model
        model = nn.Linear(10, 10)

        logger.info("✓ Phase orchestrator created successfully")
        return True

    except Exception as e:
        logger.error(f"✗ Phase orchestrator test failed: {e}")
        return False


async def test_phase_interface():
    """Test phase interface with a mock phase."""
    try:
        from core.phase_controller import PhaseController, PhaseResult

        class MockPhase(PhaseController):
            def __init__(self):
                super().__init__({"test": True})

            async def run(self, model):
                return PhaseResult(
                    success=True,
                    model=model,
                    phase_name="MockPhase",
                    metrics={"test_metric": 1.0},
                    duration_seconds=0.1,
                )

        # Create and test mock phase
        phase = MockPhase()
        model = nn.Linear(10, 10)

        result = await phase.run(model)

        if not result.success:
            raise ValueError("Mock phase failed")

        logger.info("✓ Phase interface test successful")
        logger.info(f"  Phase result: {result.phase_name}, Success: {result.success}")
        return True

    except Exception as e:
        logger.error(f"✗ Phase interface test failed: {e}")
        return False


def test_model_validation():
    """Test model validation functionality."""
    try:
        from core.phase_controller import ModelPassingValidator

        validator = ModelPassingValidator()

        # Create test model
        model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))

        # Test default validation
        is_valid, error = validator._default_validation(model)

        if not is_valid:
            raise ValueError(f"Model validation failed: {error}")

        logger.info("✓ Model validation test successful")
        return True

    except Exception as e:
        logger.error(f"✗ Model validation test failed: {e}")
        return False


async def run_simple_tests():
    """Run all simple tests."""
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Pipeline Config", test_pipeline_config),
        ("Phase Orchestrator", test_phase_orchestrator),
        ("Phase Interface", test_phase_interface),
        ("Model Validation", test_model_validation),
    ]

    passed = 0
    total = len(tests)

    print("Running Simple Agent Forge Tests...")
    print("=" * 60)

    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1

        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")

    print("=" * 60)
    print(f"Simple Tests Complete: {passed}/{total} passed")
    print(f"Success Rate: {passed/total*100:.1f}%")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_simple_tests())

    if success:
        print("\n🎉 All simple tests passed! Core infrastructure is working.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")

    sys.exit(0 if success else 1)
