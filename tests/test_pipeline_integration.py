#!/usr/bin/env python3
"""Integration test script for the complete Agent Forge pipeline

This script tests each stage of the compression and training pipeline
to ensure they work correctly and verify proper handoff between components.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_forge.compression.stage1 import run_stage1_compression
from agent_forge.compression.stage1_config import Stage1Config
from agent_forge.compression.stage2 import Stage2Compressor
from agent_forge.deployment.manifest_generator import DeploymentManifestGenerator
from agent_forge.training.enhanced_self_modeling import (
    EnhancedSelfModeling,
    SelfModelingConfig,
)
from agent_forge.training.training_loop import AgentForgeTrainingLoop

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineIntegrationTest:
    """Test the complete Agent Forge pipeline integration"""

    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Test model parameters
        self.model_name = "microsoft/DialoGPT-small"
        self.test_results = {}

        logger.info("Initialized pipeline test in %s", self.temp_dir)

    def create_test_model(self) -> str:
        """Create a test model for pipeline testing"""
        logger.info("Creating test model...")

        try:
            # Load a small model for testing
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Save model
            model_path = self.temp_dir / "original_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model.config.to_dict(),
                    "model_info": {
                        "model_path": self.model_name,
                        "tokenizer_config": tokenizer.init_kwargs
                        if hasattr(tokenizer, "init_kwargs")
                        else {},
                    },
                },
                model_path,
            )

            logger.info("Test model created: %s", model_path)
            logger.info(
                "Model size: %.2f MB", model_path.stat().st_size / (1024 * 1024)
            )

            return str(model_path)

        except Exception as e:
            logger.error("Failed to create test model: %s", e)
            raise

    def test_stage1_compression(self, input_path: str) -> str:
        """Test Stage 1 compression pipeline"""
        logger.info("Testing Stage 1 compression...")

        try:
            # Configure Stage 1 for testing
            config = Stage1Config(
                bitnet_enabled=False,  # Disable to avoid training complexity
                seedlm_enabled=True,
                seedlm_block_size=4,
                seedlm_latent_dim=2,
                seedlm_num_seeds=64,
                eval_max_samples=5,
                target_compression_ratio=5.0,  # Lower for testing
                max_accuracy_drop=0.1,  # More lenient for testing
            )

            # Output path
            output_path = self.temp_dir / "stage1_compressed.pt"

            # Run Stage 1 compression
            result = run_stage1_compression(input_path, str(output_path), config)

            # Verify result
            assert result is not None, "Stage 1 compression returned None"
            assert "output_path" in result, "Stage 1 result missing output_path"
            assert os.path.exists(output_path), "Stage 1 output file not created"

            # Load and verify output
            stage1_data = torch.load(output_path)
            assert "compressed_state" in stage1_data, (
                "Stage 1 output missing compressed_state"
            )
            assert "config" in stage1_data, "Stage 1 output missing config"

            # Check compression occurred
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)

            logger.info("Stage 1 compression completed:")
            logger.info("  Original size: %.2f MB", original_size / (1024 * 1024))
            logger.info("  Compressed size: %.2f MB", compressed_size / (1024 * 1024))
            logger.info("  File size ratio: %.2fx", original_size / compressed_size)

            # Store results
            self.test_results["stage1"] = {
                "success": True,
                "input_path": input_path,
                "output_path": str(output_path),
                "original_size_mb": original_size / (1024 * 1024),
                "compressed_size_mb": compressed_size / (1024 * 1024),
                "file_size_ratio": original_size / compressed_size,
                "result": result,
            }

            return str(output_path)

        except Exception as e:
            logger.error("Stage 1 compression failed: %s", e)
            self.test_results["stage1"] = {"success": False, "error": str(e)}
            raise

    def test_stage2_compression(self, stage1_path: str) -> str:
        """Test Stage 2 compression pipeline"""
        logger.info("Testing Stage 2 compression...")

        try:
            # Create Stage 2 compressor
            compressor = Stage2Compressor(
                vptq_bits=2.0,
                vptq_vector_length=8,
                use_hyperfn=True,
                hyperfn_clusters=4,
            )

            # Output path
            output_path = self.temp_dir / "stage2_compressed.pt"

            # Run Stage 2 compression
            result = compressor.run_pipeline(stage1_path, str(output_path))

            # Verify result
            assert result is not None, "Stage 2 compression returned None"
            assert result.get("success", False), (
                f"Stage 2 failed: {result.get('error', 'Unknown error')}"
            )
            assert os.path.exists(output_path), "Stage 2 output file not created"

            # Load and verify output
            stage2_data = torch.load(output_path)
            assert "stage2_compressed_data" in stage2_data, (
                "Stage 2 output missing stage2_compressed_data"
            )
            assert "stage1_metadata" in stage2_data, (
                "Stage 2 output missing stage1_metadata"
            )

            # Check compression
            stage1_size = os.path.getsize(stage1_path)
            stage2_size = os.path.getsize(output_path)

            logger.info("Stage 2 compression completed:")
            logger.info("  Stage 1 size: %.2f MB", stage1_size / (1024 * 1024))
            logger.info("  Stage 2 size: %.2f MB", stage2_size / (1024 * 1024))
            logger.info(
                "  Overall compression ratio: %.2fx",
                result.get("overall_compression_ratio", 0.0),
            )

            # Store results
            self.test_results["stage2"] = {
                "success": True,
                "input_path": stage1_path,
                "output_path": str(output_path),
                "stage1_size_mb": stage1_size / (1024 * 1024),
                "stage2_size_mb": stage2_size / (1024 * 1024),
                "result": result,
            }

            return str(output_path)

        except Exception as e:
            logger.error("Stage 2 compression failed: %s", e)
            self.test_results["stage2"] = {"success": False, "error": str(e)}
            raise

    def test_training_pipeline(self, original_model_path: str) -> dict:
        """Test the training pipeline with Quiet-STaR"""
        logger.info("Testing training pipeline...")

        try:
            # Load original model for training
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Add padding token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create training loop with Quiet-STaR
            training_loop = AgentForgeTrainingLoop(
                model=model,
                tokenizer=tokenizer,
                enable_quiet_star=True,
                curriculum_domain="general",
            )

            # Run minimal curriculum (1 level, few tasks)
            logger.info("Running minimal curriculum training...")
            curriculum_results = training_loop.run_curriculum(
                max_levels=1,
                tasks_per_level=3,  # Very small for testing
            )

            # Verify results
            assert curriculum_results is not None, "Training pipeline returned None"
            assert "levels_completed" in curriculum_results, (
                "Training result missing levels_completed"
            )
            assert "level_metrics" in curriculum_results, (
                "Training result missing level_metrics"
            )
            assert curriculum_results["levels_completed"] >= 1, (
                "No training levels completed"
            )

            logger.info("Training pipeline completed:")
            logger.info(
                "  Levels completed: %d", curriculum_results["levels_completed"]
            )
            logger.info(
                "  Overall accuracy: %.3f", curriculum_results["overall_accuracy"]
            )
            logger.info(
                "  Quiet-STaR enabled: %s", curriculum_results["quiet_star_enabled"]
            )

            # Store results
            self.test_results["training"] = {
                "success": True,
                "curriculum_results": curriculum_results,
                "quiet_star_enabled": curriculum_results["quiet_star_enabled"],
                "levels_completed": curriculum_results["levels_completed"],
                "overall_accuracy": curriculum_results["overall_accuracy"],
            }

            return curriculum_results

        except Exception as e:
            logger.error("Training pipeline failed: %s", e)
            self.test_results["training"] = {"success": False, "error": str(e)}
            raise

    def test_self_modeling(self, original_model_path: str) -> dict:
        """Test self-modeling with temperature sweeps"""
        logger.info("Testing self-modeling...")

        try:
            # Load model for self-modeling
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create minimal config for testing
            config = SelfModelingConfig(
                num_temperature_samples=20,  # Very small for testing
                reflection_depth=1,
                save_checkpoints=False,
                max_sequence_length=128,
            )

            # Create self-modeling system
            self_modeling = EnhancedSelfModeling(model, tokenizer, config)

            # Run minimal self-modeling cycle
            logger.info("Running minimal self-modeling cycle...")
            prompts = ["What is AI?", "Hello world"]

            results = self_modeling.run_self_modeling_cycle(prompts, num_cycles=1)

            # Verify results
            assert results is not None, "Self-modeling returned None"
            assert "cycles_completed" in results, (
                "Self-modeling result missing cycles_completed"
            )
            assert "temperature_insights" in results, (
                "Self-modeling result missing temperature_insights"
            )
            assert results["cycles_completed"] >= 1, "No self-modeling cycles completed"

            # Get insights
            insights = self_modeling.get_insights_summary()

            logger.info("Self-modeling completed:")
            logger.info("  Cycles completed: %d", results["cycles_completed"])
            logger.info(
                "  Temperature insights: %d", len(results["temperature_insights"])
            )
            logger.info("  Recommendations: %d", len(insights["recommendations"]))

            # Store results
            self.test_results["self_modeling"] = {
                "success": True,
                "results": results,
                "insights": insights,
                "cycles_completed": results["cycles_completed"],
            }

            return results

        except Exception as e:
            logger.error("Self-modeling failed: %s", e)
            self.test_results["self_modeling"] = {"success": False, "error": str(e)}
            raise

    def test_deployment_manifest(self, final_model_path: str) -> str:
        """Test deployment manifest generation"""
        logger.info("Testing deployment manifest generation...")

        try:
            # Create manifest generator
            output_dir = self.temp_dir / "releases"
            generator = DeploymentManifestGenerator(final_model_path, str(output_dir))

            # Generate manifest
            manifest = generator.generate_manifest("v1.0.0-test")

            # Verify manifest structure
            assert manifest is not None, "Manifest generation returned None"
            assert "model_info" in manifest, "Manifest missing model_info"
            assert "evaluation_metrics" in manifest, (
                "Manifest missing evaluation_metrics"
            )
            assert "deployment_requirements" in manifest, (
                "Manifest missing deployment_requirements"
            )
            assert "security" in manifest, "Manifest missing security"

            # Create release bundle
            version = manifest["model_info"]["version"]
            bundle_path = generator.create_release_bundle(manifest, version)

            # Verify bundle
            bundle_dir = Path(bundle_path)
            assert bundle_dir.exists(), "Release bundle directory not created"
            assert (bundle_dir / "model.pt").exists(), "Model file not in bundle"
            assert (bundle_dir / "manifest.json").exists(), (
                "Manifest file not in bundle"
            )
            assert (bundle_dir / "README.md").exists(), "README file not in bundle"

            logger.info("Deployment manifest generated:")
            logger.info("  Version: %s", version)
            logger.info("  Model size: %.2f MB", manifest["model_info"]["file_size_mb"])
            logger.info(
                "  Deployment tier: %s",
                manifest["deployment_requirements"]["deployment_tier"],
            )
            logger.info("  Bundle path: %s", bundle_path)

            # Store results
            self.test_results["deployment"] = {
                "success": True,
                "manifest": manifest,
                "bundle_path": bundle_path,
                "version": version,
                "model_size_mb": manifest["model_info"]["file_size_mb"],
                "deployment_tier": manifest["deployment_requirements"][
                    "deployment_tier"
                ],
            }

            return bundle_path

        except Exception as e:
            logger.error("Deployment manifest generation failed: %s", e)
            self.test_results["deployment"] = {"success": False, "error": str(e)}
            raise

    def test_model_handoff_integrity(self) -> dict:
        """Test that model data is properly handed off between stages"""
        logger.info("Testing model handoff integrity...")

        try:
            integrity_results = {}

            # Check Stage 1 -> Stage 2 handoff
            if self.test_results.get("stage1", {}).get(
                "success"
            ) and self.test_results.get("stage2", {}).get("success"):
                stage1_path = self.test_results["stage1"]["output_path"]
                stage2_path = self.test_results["stage2"]["output_path"]

                # Load both files
                stage1_data = torch.load(stage1_path)
                stage2_data = torch.load(stage2_path)

                # Verify Stage 1 metadata is preserved in Stage 2
                assert "stage1_metadata" in stage2_data, (
                    "Stage 1 metadata not preserved in Stage 2"
                )
                assert "config" in stage2_data["stage1_metadata"], (
                    "Stage 1 config not preserved"
                )

                # Verify compression pipeline information
                assert "compression_pipeline" in stage2_data, (
                    "Compression pipeline info missing"
                )

                integrity_results["stage1_to_stage2"] = {
                    "success": True,
                    "metadata_preserved": True,
                    "pipeline_info_present": True,
                }

                logger.info("Stage 1 -> Stage 2 handoff: ✓ PASSED")
            else:
                integrity_results["stage1_to_stage2"] = {
                    "success": False,
                    "reason": "Previous stages failed",
                }
                logger.warning(
                    "Stage 1 -> Stage 2 handoff: ✗ SKIPPED (previous stages failed)"
                )

            # Check model format consistency
            if self.test_results.get("stage2", {}).get("success"):
                stage2_path = self.test_results["stage2"]["output_path"]
                stage2_data = torch.load(stage2_path)

                # Verify expected data structure
                expected_keys = [
                    "stage2_compressed_data",
                    "stage1_metadata",
                    "compression_pipeline",
                ]
                for key in expected_keys:
                    assert key in stage2_data, f"Missing key: {key}"

                integrity_results["format_consistency"] = {
                    "success": True,
                    "all_keys_present": True,
                }

                logger.info("Model format consistency: ✓ PASSED")
            else:
                integrity_results["format_consistency"] = {
                    "success": False,
                    "reason": "Stage 2 failed",
                }
                logger.warning("Model format consistency: ✗ SKIPPED (Stage 2 failed)")

            # Store results
            self.test_results["integrity"] = integrity_results

            return integrity_results

        except Exception as e:
            logger.error("Model handoff integrity test failed: %s", e)
            self.test_results["integrity"] = {"success": False, "error": str(e)}
            raise

    def generate_test_report(self) -> str:
        """Generate a comprehensive test report"""
        logger.info("Generating test report...")

        report_path = self.temp_dir / "test_report.json"

        # Create summary
        summary = {
            "test_timestamp": str(torch.datetime.now()),
            "test_directory": str(self.temp_dir),
            "model_used": self.model_name,
            "overall_success": all(
                result.get("success", False) for result in self.test_results.values()
            ),
        }

        # Combine all results
        full_report = {"summary": summary, "detailed_results": self.test_results}

        # Save report
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2, default=str)

        logger.info("Test report saved: %s", report_path)
        return str(report_path)

    def run_full_pipeline_test(self) -> dict:
        """Run the complete pipeline integration test"""
        logger.info("Starting full pipeline integration test...")

        try:
            # Step 1: Create test model
            original_model_path = self.create_test_model()

            # Step 2: Test Stage 1 compression
            stage1_path = self.test_stage1_compression(original_model_path)

            # Step 3: Test Stage 2 compression
            stage2_path = self.test_stage2_compression(stage1_path)

            # Step 4: Test training pipeline
            training_results = self.test_training_pipeline(original_model_path)

            # Step 5: Test self-modeling
            self_modeling_results = self.test_self_modeling(original_model_path)

            # Step 6: Test deployment manifest
            deployment_bundle = self.test_deployment_manifest(stage2_path)

            # Step 7: Test model handoff integrity
            integrity_results = self.test_model_handoff_integrity()

            # Step 8: Generate test report
            report_path = self.generate_test_report()

            logger.info("Full pipeline integration test completed successfully!")

            return {
                "success": True,
                "test_results": self.test_results,
                "report_path": report_path,
                "summary": {
                    "stages_passed": sum(
                        1 for r in self.test_results.values() if r.get("success")
                    ),
                    "stages_total": len(self.test_results),
                    "overall_success": all(
                        r.get("success", False) for r in self.test_results.values()
                    ),
                },
            }

        except Exception as e:
            logger.error("Full pipeline test failed: %s", e)

            # Generate partial report
            try:
                report_path = self.generate_test_report()
            except Exception:
                report_path = None

            return {
                "success": False,
                "error": str(e),
                "test_results": self.test_results,
                "report_path": report_path,
            }


def main():
    """Main test execution function"""
    print("🚀 Starting Agent Forge Pipeline Integration Test")
    print("=" * 60)

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Test directory: {temp_dir}")

        # Create test instance
        test_runner = PipelineIntegrationTest(temp_dir)

        # Run full pipeline test
        results = test_runner.run_full_pipeline_test()

        # Display results
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        if results["success"]:
            print("✅ Overall Result: PASSED")
            summary = results["summary"]
            print(
                f"✅ Stages Passed: {summary['stages_passed']}/{summary['stages_total']}"
            )

            # Show individual stage results
            for stage_name, stage_result in test_runner.test_results.items():
                status = "✅ PASSED" if stage_result.get("success") else "❌ FAILED"
                print(f"  {stage_name}: {status}")

                if not stage_result.get("success") and "error" in stage_result:
                    print(f"    Error: {stage_result['error']}")
        else:
            print("❌ Overall Result: FAILED")
            print(f"❌ Error: {results['error']}")

            # Show partial results
            for stage_name, stage_result in test_runner.test_results.items():
                status = "✅ PASSED" if stage_result.get("success") else "❌ FAILED"
                print(f"  {stage_name}: {status}")

        # Show report location
        if results.get("report_path"):
            print(f"\n📄 Detailed report: {results['report_path']}")

        print("\n🏁 Test completed!")

        # Return appropriate exit code
        return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
