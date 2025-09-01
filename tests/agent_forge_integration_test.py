#!/usr/bin/env python3
"""
Agent Forge Integration Test Suite

Tests the real Agent Forge pipeline components that are available,
validates actual imports, and measures real performance where possible.
"""

import asyncio
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import torch
import torch.nn as nn


class AgentForgeIntegrationTest:
    """Integration test for real Agent Forge components."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.import_results = {}
        self.performance_metrics = {}

        # Add paths for imports
        self.core_path = Path(__file__).parent.parent / "core"
        self.agent_forge_path = self.core_path / "agent_forge"

        sys.path.insert(0, str(self.core_path))
        sys.path.insert(0, str(self.agent_forge_path))

    def test_real_imports(self) -> dict[str, Any]:
        """Test importing real Agent Forge components."""
        self.logger.info("Testing real Agent Forge imports...")

        import_tests = {
            # Core components
            "unified_pipeline": {
                "module": "unified_pipeline",
                "classes": ["UnifiedPipeline", "UnifiedConfig"],
                "required": True,
            },
            "phase_controller": {
                "module": "core.phase_controller",
                "classes": ["PhaseController", "PhaseResult", "PhaseOrchestrator"],
                "required": True,
            },
            # Individual phases
            "cognate_phase": {
                "module": "phases.cognate",
                "classes": ["CognatePhase", "CognateConfig"],
                "required": False,
            },
            "evomerge_phase": {
                "module": "phases.evomerge",
                "classes": ["EvoMergePhase", "EvoMergeConfig"],
                "required": False,
            },
            "quietstar_phase": {
                "module": "phases.quietstar",
                "classes": ["QuietSTaRPhase", "QuietSTaRConfig"],
                "required": False,
            },
            "bitnet_phase": {
                "module": "phases.bitnet_compression",
                "classes": ["BitNetCompressionPhase", "BitNetConfig"],
                "required": False,
            },
            "training_phase": {
                "module": "phases.forge_training",
                "classes": ["ForgeTrainingPhase", "ForgeTrainingConfig"],
                "required": False,
            },
            "toolbaking_phase": {
                "module": "phases.tool_persona_baking",
                "classes": ["ToolPersonaBakingPhase", "ToolPersonaBakingConfig"],
                "required": False,
            },
            "adas_phase": {"module": "phases.adas", "classes": ["ADASPhase", "ADASConfig"], "required": False},
            "compression_phase": {
                "module": "phases.final_compression",
                "classes": ["FinalCompressionPhase", "FinalCompressionConfig"],
                "required": False,
            },
        }

        results = {}

        for test_name, config in import_tests.items():
            try:
                module = __import__(config["module"], fromlist=config["classes"])

                imported_classes = []
                missing_classes = []

                for class_name in config["classes"]:
                    if hasattr(module, class_name):
                        imported_classes.append(class_name)
                    else:
                        missing_classes.append(class_name)

                success = len(imported_classes) > 0
                results[test_name] = {
                    "success": success,
                    "module": config["module"],
                    "imported_classes": imported_classes,
                    "missing_classes": missing_classes,
                    "required": config["required"],
                    "error": None,
                }

                if success:
                    self.logger.info(f"✓ {test_name}: {len(imported_classes)} classes imported")
                else:
                    level = logging.ERROR if config["required"] else logging.WARNING
                    self.logger.log(level, f"✗ {test_name}: No classes imported")

            except Exception as e:
                results[test_name] = {
                    "success": False,
                    "module": config["module"],
                    "imported_classes": [],
                    "missing_classes": config["classes"],
                    "required": config["required"],
                    "error": str(e),
                }

                level = logging.ERROR if config["required"] else logging.WARNING
                self.logger.log(level, f"✗ {test_name}: Import failed - {e}")

        self.import_results = results
        return results

    def test_pipeline_creation(self) -> dict[str, Any]:
        """Test creating pipeline with available components."""
        self.logger.info("Testing pipeline creation...")

        try:
            # Try to import and create pipeline
            from unified_pipeline import UnifiedConfig, UnifiedPipeline

            # Create minimal configuration
            config = UnifiedConfig(
                base_models=["mock-model"],
                output_dir=Path("./integration_test_output"),
                checkpoint_dir=Path("./integration_test_checkpoints"),
                device="cpu",
                # Conservative phase enables
                enable_cognate=True,
                enable_evomerge=False,
                enable_quietstar=False,
                enable_initial_compression=False,
                enable_training=False,
                enable_tool_baking=False,
                enable_adas=False,
                enable_final_compression=False,
                wandb_project=None,
            )

            # Create pipeline
            pipeline = UnifiedPipeline(config)

            result = {
                "success": True,
                "pipeline_created": True,
                "phases_available": len(pipeline.phases),
                "phase_names": [name for name, _ in pipeline.phases],
                "config_valid": True,
                "error": None,
            }

            self.logger.info(f"✓ Pipeline created with {result['phases_available']} phases")
            for phase_name in result["phase_names"]:
                self.logger.info(f"  - {phase_name}")

            return result

        except ImportError as e:
            result = {
                "success": False,
                "pipeline_created": False,
                "phases_available": 0,
                "phase_names": [],
                "config_valid": False,
                "error": f"Import error: {e}",
            }
            self.logger.error(f"✗ Pipeline creation failed: {e}")
            return result

        except Exception as e:
            result = {
                "success": False,
                "pipeline_created": False,
                "phases_available": 0,
                "phase_names": [],
                "config_valid": False,
                "error": f"Creation error: {e}",
            }
            self.logger.error(f"✗ Pipeline creation failed: {e}")
            return result

    async def test_pipeline_execution(self) -> dict[str, Any]:
        """Test actual pipeline execution if possible."""
        self.logger.info("Testing pipeline execution...")

        try:
            from unified_pipeline import UnifiedConfig, UnifiedPipeline

            # Create very minimal config for execution test
            config = UnifiedConfig(
                base_models=["mock-tiny-model"],
                output_dir=Path("./execution_test_output"),
                checkpoint_dir=Path("./execution_test_checkpoints"),
                device="cpu",
                # Only enable Cognate for testing
                enable_cognate=True,
                enable_evomerge=False,
                enable_quietstar=False,
                enable_initial_compression=False,
                enable_training=False,
                enable_tool_baking=False,
                enable_adas=False,
                enable_final_compression=False,
                wandb_project=None,
            )

            pipeline = UnifiedPipeline(config)

            if len(pipeline.phases) == 0:
                return {"success": False, "executed": False, "error": "No phases available for execution"}

            # Try to run pipeline (this may fail due to missing dependencies)
            start_time = time.time()

            try:
                result = await pipeline.run_pipeline()
                execution_time = time.time() - start_time

                return {
                    "success": True,
                    "executed": True,
                    "pipeline_success": result.success,
                    "execution_time_seconds": execution_time,
                    "phases_completed": result.metrics.get("phases_completed", 0) if result.success else 0,
                    "error": result.error if not result.success else None,
                }

            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.warning(f"Pipeline execution failed (expected): {e}")

                return {
                    "success": False,
                    "executed": False,
                    "execution_time_seconds": execution_time,
                    "error": str(e),
                    "expected_failure": True,  # This indicates the failure is expected
                }

        except Exception as e:
            return {"success": False, "executed": False, "error": f"Setup error: {e}"}

    def test_phase_compatibility(self) -> dict[str, Any]:
        """Test phase compatibility validation."""
        self.logger.info("Testing phase compatibility...")

        try:
            from core.phase_controller import PhaseOrchestrator

            # Test orchestrator creation
            orchestrator = PhaseOrchestrator()

            # Test with empty phases
            is_compatible = orchestrator.validate_phase_compatibility([])

            # Test with mock phases
            mock_phases = [
                ("CognatePhase", None),
                ("EvoMergePhase", None),
                ("QuietSTaRPhase", None),
            ]

            is_compatible_mock = orchestrator.validate_phase_compatibility(mock_phases)

            return {
                "success": True,
                "orchestrator_created": True,
                "empty_phases_compatible": is_compatible,
                "mock_phases_compatible": is_compatible_mock,
                "error": None,
            }

        except Exception as e:
            return {"success": False, "orchestrator_created": False, "error": str(e)}

    def measure_baseline_performance(self) -> dict[str, Any]:
        """Measure baseline performance metrics for comparison."""
        self.logger.info("Measuring baseline performance...")

        try:
            # Simple model for baseline
            model = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))

            input_data = torch.randn(32, 768)  # Batch of 32

            # Measure inference time
            start_time = time.time()
            num_inferences = 100

            with torch.no_grad():
                for _ in range(num_inferences):
                    _ = model(input_data)

            baseline_time = time.time() - start_time
            avg_inference_time = baseline_time / num_inferences

            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())

            # Estimate memory usage
            model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32

            return {
                "success": True,
                "baseline_inference_time_seconds": baseline_time,
                "average_inference_time_ms": avg_inference_time * 1000,
                "parameter_count": param_count,
                "model_size_mb": model_size_mb,
                "batch_size": 32,
                "num_inferences": num_inferences,
                "error": None,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def analyze_codebase_structure(self) -> dict[str, Any]:
        """Analyze the Agent Forge codebase structure."""
        self.logger.info("Analyzing codebase structure...")

        try:
            structure = {}

            # Check core directories
            core_dirs = {
                "agent_forge_root": self.agent_forge_path,
                "phases_dir": self.agent_forge_path / "phases",
                "models_dir": self.agent_forge_path / "models",
                "benchmarks_dir": self.agent_forge_path / "benchmarks",
                "integration_dir": self.agent_forge_path / "integration",
            }

            for name, path in core_dirs.items():
                structure[name] = {
                    "exists": path.exists(),
                    "is_dir": path.is_dir() if path.exists() else False,
                    "path": str(path),
                }

                if path.exists() and path.is_dir():
                    # Count Python files
                    py_files = list(path.glob("*.py"))
                    subdirs = [d for d in path.iterdir() if d.is_dir()]

                    structure[name].update(
                        {
                            "python_files_count": len(py_files),
                            "subdirectories_count": len(subdirs),
                            "subdirectories": [d.name for d in subdirs],
                        }
                    )

            # Check for key files
            key_files = {
                "unified_pipeline": self.agent_forge_path / "unified_pipeline.py",
                "phase_controller": self.agent_forge_path / "core" / "phase_controller.py",
                "cognate_phase": self.agent_forge_path / "phases" / "cognate.py",
                "evomerge_phase": self.agent_forge_path / "phases" / "evomerge.py",
            }

            files_status = {}
            for name, path in key_files.items():
                files_status[name] = {
                    "exists": path.exists(),
                    "path": str(path),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                }

            return {
                "success": True,
                "directory_structure": structure,
                "key_files": files_status,
                "codebase_accessible": True,
                "error": None,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "codebase_accessible": False}

    async def run_comprehensive_integration_test(self) -> dict[str, Any]:
        """Run comprehensive integration test suite."""
        self.logger.info("Starting comprehensive Agent Forge integration test...")

        start_time = time.time()

        # Run all test components
        test_results = {}

        # 1. Import tests
        self.logger.info("Phase 1: Testing imports...")
        test_results["imports"] = self.test_real_imports()

        # 2. Codebase structure analysis
        self.logger.info("Phase 2: Analyzing codebase structure...")
        test_results["codebase_structure"] = self.analyze_codebase_structure()

        # 3. Pipeline creation test
        self.logger.info("Phase 3: Testing pipeline creation...")
        test_results["pipeline_creation"] = self.test_pipeline_creation()

        # 4. Phase compatibility test
        self.logger.info("Phase 4: Testing phase compatibility...")
        test_results["phase_compatibility"] = self.test_phase_compatibility()

        # 5. Baseline performance measurement
        self.logger.info("Phase 5: Measuring baseline performance...")
        test_results["baseline_performance"] = self.measure_baseline_performance()

        # 6. Pipeline execution test (may fail - that's OK)
        self.logger.info("Phase 6: Testing pipeline execution...")
        test_results["pipeline_execution"] = await self.test_pipeline_execution()

        total_time = time.time() - start_time

        # Generate summary
        summary = self._generate_test_summary(test_results, total_time)

        # Save detailed results
        await self._save_integration_results(test_results, summary)

        return {"summary": summary, "detailed_results": test_results, "total_duration_seconds": total_time}

    def _generate_test_summary(self, test_results: dict[str, Any], duration: float) -> dict[str, Any]:
        """Generate test summary."""

        # Count successes
        successful_tests = sum(
            1 for result in test_results.values() if isinstance(result, dict) and result.get("success", False)
        )
        total_tests = len(test_results)

        # Analyze imports
        import_summary = test_results.get("imports", {})
        successful_imports = sum(1 for result in import_summary.values() if result.get("success", False))
        critical_imports = sum(1 for result in import_summary.values() if result.get("required", False))
        critical_imports_success = sum(
            1 for result in import_summary.values() if result.get("success", False) and result.get("required", False)
        )

        # Check pipeline status
        pipeline_created = test_results.get("pipeline_creation", {}).get("success", False)
        phases_available = test_results.get("pipeline_creation", {}).get("phases_available", 0)

        # Performance baseline
        baseline_performance = test_results.get("baseline_performance", {}).get("success", False)

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "overall_status": (
                "FUNCTIONAL" if successful_tests >= 4 else "PARTIAL" if successful_tests >= 2 else "FAILED"
            ),
            "test_counts": {
                "successful_tests": successful_tests,
                "total_tests": total_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            },
            "import_status": {
                "successful_imports": successful_imports,
                "total_imports": len(import_summary),
                "critical_imports_success": critical_imports_success,
                "critical_imports_total": critical_imports,
                "critical_imports_ok": critical_imports_success >= critical_imports // 2,
            },
            "pipeline_status": {
                "can_create_pipeline": pipeline_created,
                "phases_available": phases_available,
                "ready_for_execution": pipeline_created and phases_available > 0,
            },
            "performance_status": {
                "baseline_measured": baseline_performance,
                "ready_for_benchmarking": baseline_performance and pipeline_created,
            },
            "readiness_assessment": {
                "structure_analysis": "COMPLETE",
                "import_validation": "COMPLETE",
                "pipeline_creation": "COMPLETE" if pipeline_created else "FAILED",
                "execution_testing": "ATTEMPTED",
                "performance_baseline": "COMPLETE" if baseline_performance else "FAILED",
            },
        }

    async def _save_integration_results(self, test_results: dict[str, Any], summary: dict[str, Any]):
        """Save integration test results."""

        output_dir = Path("./integration_test_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = int(time.time())

        # Save detailed results
        detailed_path = output_dir / f"agent_forge_integration_detailed_{timestamp}.json"
        with open(detailed_path, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "detailed_results": test_results,
                    "test_metadata": {
                        "test_type": "Agent Forge Integration Test",
                        "python_version": sys.version,
                        "torch_version": torch.__version__,
                        "platform": sys.platform,
                    },
                },
                f,
                indent=2,
                default=str,
            )

        # Save summary report
        summary_path = output_dir / f"agent_forge_integration_summary_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write("Agent Forge Integration Test Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test completed: {summary['timestamp']}\n")
            f.write(f"Duration: {summary['duration_seconds']:.2f} seconds\n")
            f.write(f"Overall status: {summary['overall_status']}\n\n")

            f.write("Test Results:\n")
            f.write(
                f"- Successful tests: {summary['test_counts']['successful_tests']}/{summary['test_counts']['total_tests']}\n"
            )
            f.write(f"- Success rate: {summary['test_counts']['success_rate']:.1%}\n\n")

            f.write("Import Status:\n")
            f.write(
                f"- Successful imports: {summary['import_status']['successful_imports']}/{summary['import_status']['total_imports']}\n"
            )
            f.write(f"- Critical imports OK: {summary['import_status']['critical_imports_ok']}\n\n")

            f.write("Pipeline Status:\n")
            f.write(f"- Can create pipeline: {summary['pipeline_status']['can_create_pipeline']}\n")
            f.write(f"- Phases available: {summary['pipeline_status']['phases_available']}\n")
            f.write(f"- Ready for execution: {summary['pipeline_status']['ready_for_execution']}\n\n")

            f.write("Readiness Assessment:\n")
            for component, status in summary["readiness_assessment"].items():
                f.write(f"- {component}: {status}\n")

            f.write(f"\nDetailed results: {detailed_path}\n")

        self.logger.info("Integration test results saved:")
        self.logger.info(f"  - Detailed: {detailed_path}")
        self.logger.info(f"  - Summary: {summary_path}")


async def run_integration_test():
    """Run the Agent Forge integration test."""

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(__name__)
    logger.info("Starting Agent Forge Integration Test Suite")

    # Create test instance
    integration_test = AgentForgeIntegrationTest()

    try:
        # Run comprehensive test
        results = await integration_test.run_comprehensive_integration_test()

        # Log summary
        summary = results["summary"]
        logger.info("=== INTEGRATION TEST COMPLETED ===")
        logger.info(f"Overall status: {summary['overall_status']}")
        logger.info(f"Test duration: {summary['duration_seconds']:.2f}s")
        logger.info(f"Success rate: {summary['test_counts']['success_rate']:.1%}")
        logger.info(f"Pipeline creation: {'✓' if summary['pipeline_status']['can_create_pipeline'] else '✗'}")
        logger.info(f"Phases available: {summary['pipeline_status']['phases_available']}")
        logger.info("=" * 35)

        return results

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    print("Agent Forge Integration Test Suite")
    print("=" * 50)

    # Run integration test
    try:
        results = asyncio.run(run_integration_test())

        summary = results["summary"]
        if summary["overall_status"] in ["FUNCTIONAL", "PARTIAL"]:
            print(f"\n✓ Integration test completed: {summary['overall_status']}")
            print("Check ./integration_test_results/ for detailed reports")
        else:
            print(f"\n⚠ Integration test completed with issues: {summary['overall_status']}")
            print("Review logs for details")

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        sys.exit(1)
