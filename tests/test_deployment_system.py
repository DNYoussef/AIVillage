"""
Test suite for deployment system components
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
import torch

from agent_forge.deployment.manifest_generator import DeploymentManifestGenerator


class TestDeploymentManifestGenerator:
    """Test deployment manifest generation functionality"""

    def test_manifest_generator_init(self):
        """Test manifest generator initialization"""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"test": "data"}, f.name)
            model_path = f.name

        try:
            generator = DeploymentManifestGenerator(model_path, "test_releases")

            # Verify initialization
            assert generator.model_path == Path(model_path)
            assert generator.output_dir == Path("test_releases")
            assert generator.output_dir.exists()

        finally:
            os.unlink(model_path)
            # Clean up output directory
            if Path("test_releases").exists():
                import shutil

                shutil.rmtree("test_releases")

    def test_manifest_generator_init_nonexistent_model(self):
        """Test manifest generator with nonexistent model"""
        with pytest.raises(FileNotFoundError):
            DeploymentManifestGenerator("nonexistent_model.pt")

    def test_calculate_file_hash(self):
        """Test file hash calculation"""
        # Create temporary file with known content
        test_content = b"test data for hashing"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            generator = DeploymentManifestGenerator(__file__)  # Use this test file
            file_hash = generator.calculate_file_hash(Path(temp_path))

            # Verify hash is correct
            expected_hash = hashlib.sha256(test_content).hexdigest()
            assert file_hash == expected_hash

        finally:
            os.unlink(temp_path)

    def test_get_file_size(self):
        """Test file size calculation"""
        # Create file with known size
        test_content = b"x" * 1000  # 1000 bytes
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            generator = DeploymentManifestGenerator(__file__)
            file_size = generator.get_file_size(Path(temp_path))

            assert file_size == 1000

        finally:
            os.unlink(temp_path)

    def test_extract_model_metadata_stage2(self):
        """Test metadata extraction from Stage 2 model"""
        # Create mock Stage 2 model data
        stage2_data = {
            "stage2_compressed_data": {"layer.weight": {"compression_ratio": 15.0}},
            "stage1_metadata": {
                "config": {
                    "bitnet_enabled": True,
                    "seedlm_enabled": True,
                    "max_sequence_length": 1024,
                }
            },
            "compression_pipeline": "BitNet -> SeedLM -> VPTQ -> HyperFn",
            "timestamp": 1234567890,
            "compression_stats": {
                "average_compression_ratio": 20.0,
                "total_reconstruction_error": 0.05,
            },
        }

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(stage2_data, f.name)
            model_path = f.name

        try:
            generator = DeploymentManifestGenerator(model_path)
            metadata = generator.extract_model_metadata(Path(model_path))

            # Verify metadata structure
            assert "compression" in metadata
            assert "training" in metadata
            assert "architecture" in metadata
            assert "timestamp" in metadata

            # Verify compression info
            assert (
                metadata["compression"]["pipeline"]
                == "BitNet -> SeedLM -> VPTQ -> HyperFn"
            )
            assert metadata["compression"]["stage"] == "stage2"
            assert metadata["compression"]["compression_ratio"] == 20.0

        finally:
            os.unlink(model_path)

    def test_extract_model_metadata_stage1(self):
        """Test metadata extraction from Stage 1 model"""
        # Create mock Stage 1 model data
        stage1_data = {
            "compressed_state": {"layer.weight": {"compression_ratio": 8.0}},
            "config": {
                "bitnet_enabled": True,
                "seedlm_enabled": True,
                "max_sequence_length": 512,
            },
            "compression_stats": {"layer.weight": {"compression_ratio": 8.0}},
            "model_info": {
                "model_path": "microsoft/DialoGPT-small",
                "tokenizer_config": {"vocab_size": 50000},
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(stage1_data, f.name)
            model_path = f.name

        try:
            generator = DeploymentManifestGenerator(model_path)
            metadata = generator.extract_model_metadata(Path(model_path))

            # Verify Stage 1 metadata
            assert metadata["compression"]["pipeline"] == "BitNet -> SeedLM"
            assert metadata["compression"]["stage"] == "stage1"
            assert metadata["compression"]["compression_ratio"] == 8.0

            # Verify training info
            assert metadata["training"]["bitnet_enabled"] == True
            assert metadata["training"]["seedlm_enabled"] == True

            # Verify architecture info
            assert metadata["architecture"]["base_model"] == "microsoft/DialoGPT-small"

        finally:
            os.unlink(model_path)

    def test_extract_model_metadata_raw(self):
        """Test metadata extraction from raw model"""
        # Create mock raw model data
        raw_data = {
            "model_state_dict": {
                "layer.weight": torch.randn(10, 10),
                "layer.bias": torch.randn(10),
            }
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(raw_data, f.name)
            model_path = f.name

        try:
            generator = DeploymentManifestGenerator(model_path)
            metadata = generator.extract_model_metadata(Path(model_path))

            # Verify raw model metadata
            assert metadata["compression"]["pipeline"] == "none"
            assert metadata["compression"]["stage"] == "raw"
            assert metadata["compression"]["compression_ratio"] == 1.0

        finally:
            os.unlink(model_path)

    def test_generate_deployment_requirements(self):
        """Test deployment requirements generation"""
        generator = DeploymentManifestGenerator(__file__)

        # Test edge tier (small model)
        requirements_edge = generator.generate_deployment_requirements(50.0)
        assert requirements_edge["deployment_tier"] == "edge"
        assert requirements_edge["hardware_requirements"]["min_ram_gb"] == 2
        assert requirements_edge["hardware_requirements"]["min_vram_gb"] == 1

        # Test mobile tier (medium model)
        requirements_mobile = generator.generate_deployment_requirements(300.0)
        assert requirements_mobile["deployment_tier"] == "mobile"
        assert requirements_mobile["hardware_requirements"]["min_ram_gb"] == 4

        # Test edge-plus tier (large model)
        requirements_edge_plus = generator.generate_deployment_requirements(1000.0)
        assert requirements_edge_plus["deployment_tier"] == "edge-plus"
        assert requirements_edge_plus["hardware_requirements"]["min_ram_gb"] == 8

        # Test server tier (very large model)
        requirements_server = generator.generate_deployment_requirements(3000.0)
        assert requirements_server["deployment_tier"] == "server"
        assert requirements_server["hardware_requirements"]["min_ram_gb"] == 16

        # Verify common structure
        for req in [
            requirements_edge,
            requirements_mobile,
            requirements_edge_plus,
            requirements_server,
        ]:
            assert "deployment_tier" in req
            assert "hardware_requirements" in req
            assert "software_requirements" in req
            assert "deployment_options" in req

            # Verify software requirements
            assert "python_version" in req["software_requirements"]
            assert "pytorch_version" in req["software_requirements"]
            assert "additional_dependencies" in req["software_requirements"]

            # Verify deployment options
            assert "docker_image" in req["deployment_options"]
            assert "huggingface_hub" in req["deployment_options"]

    def test_generate_security_info(self):
        """Test security information generation"""
        # Create test file
        test_content = b"test security data"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            generator = DeploymentManifestGenerator(temp_path)
            security_info = generator.generate_security_info(Path(temp_path))

            # Verify security info structure
            assert "sha256_hash" in security_info
            assert "file_size_bytes" in security_info
            assert "signature_verification" in security_info
            assert "vulnerability_scan" in security_info
            assert "content_verification" in security_info

            # Verify hash is correct
            expected_hash = hashlib.sha256(test_content).hexdigest()
            assert security_info["sha256_hash"] == expected_hash

            # Verify file size is correct
            assert security_info["file_size_bytes"] == len(test_content)

        finally:
            os.unlink(temp_path)

    def test_generate_usage_examples(self):
        """Test usage examples generation"""
        generator = DeploymentManifestGenerator(__file__)
        examples = generator.generate_usage_examples()

        # Verify examples structure
        assert isinstance(examples, list)
        assert len(examples) > 0

        for example in examples:
            assert "name" in example
            assert "description" in example
            assert "code" in example
            assert "expected_output" in example

            # Verify content
            assert isinstance(example["name"], str)
            assert isinstance(example["description"], str)
            assert isinstance(example["code"], str)
            assert isinstance(example["expected_output"], str)

    @patch(
        "agent_forge.deployment.manifest_generator.DeploymentManifestGenerator.run_evaluation"
    )
    def test_generate_manifest(self, mock_evaluation):
        """Test complete manifest generation"""
        # Mock evaluation results
        mock_evaluation.return_value = {
            "accuracy": 0.85,
            "perplexity": 12.5,
            "bleu_score": 0.72,
            "rouge_l": 0.68,
            "inference_time_ms": 125.0,
            "memory_usage_mb": 256.0,
            "throughput_tokens_per_sec": 42.0,
        }

        # Create test model
        test_data = {"test": "model"}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(test_data, f.name)
            model_path = f.name

        try:
            generator = DeploymentManifestGenerator(model_path)
            manifest = generator.generate_manifest(version="v1.0.0-test")

            # Verify manifest structure
            assert "manifest_version" in manifest
            assert "generated_at" in manifest
            assert "model_info" in manifest
            assert "evaluation_metrics" in manifest
            assert "deployment_requirements" in manifest
            assert "security" in manifest
            assert "usage_examples" in manifest
            assert "contact" in manifest
            assert "license" in manifest

            # Verify model info
            model_info = manifest["model_info"]
            assert model_info["name"] == "agent-forge-v1.0.0-test"
            assert model_info["version"] == "v1.0.0-test"
            assert "description" in model_info
            assert "file_size_bytes" in model_info
            assert "file_size_mb" in model_info

            # Verify evaluation metrics
            eval_metrics = manifest["evaluation_metrics"]
            assert eval_metrics["accuracy"] == 0.85
            assert eval_metrics["perplexity"] == 12.5
            assert eval_metrics["inference_time_ms"] == 125.0

            # Verify deployment requirements
            assert "deployment_tier" in manifest["deployment_requirements"]
            assert "hardware_requirements" in manifest["deployment_requirements"]

            # Verify security info
            assert "sha256_hash" in manifest["security"]
            assert "file_size_bytes" in manifest["security"]

            # Verify usage examples
            assert isinstance(manifest["usage_examples"], list)
            assert len(manifest["usage_examples"]) > 0

        finally:
            os.unlink(model_path)

    def test_save_manifest(self):
        """Test manifest saving"""
        # Create test manifest
        test_manifest = {
            "manifest_version": "1.0.0",
            "model_info": {"name": "test-model", "version": "v1.0.0"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            generator = DeploymentManifestGenerator(__file__, temp_dir)

            # Save manifest
            manifest_path = generator.save_manifest(test_manifest, "v1.0.0")

            # Verify file exists
            assert os.path.exists(manifest_path)

            # Verify content
            with open(manifest_path) as f:
                loaded_manifest = json.load(f)

            assert loaded_manifest == test_manifest

    def test_create_release_bundle(self):
        """Test release bundle creation"""
        # Create test model
        test_data = {"test": "model"}
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(test_data, f.name)
            model_path = f.name

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                generator = DeploymentManifestGenerator(model_path, temp_dir)

                # Create test manifest
                test_manifest = {
                    "manifest_version": "1.0.0",
                    "model_info": {
                        "name": "test-model",
                        "version": "v1.0.0",
                        "description": "Test model",
                        "file_size_mb": 1.0,
                        "compression_pipeline": "none",
                        "compression_ratio": 1.0,
                    },
                    "evaluation_metrics": {"accuracy": 0.85, "perplexity": 12.5},
                    "deployment_requirements": {
                        "deployment_tier": "edge",
                        "hardware_requirements": {"min_ram_gb": 2},
                        "software_requirements": {
                            "additional_dependencies": ["torch", "transformers"]
                        },
                    },
                    "security": {"sha256_hash": "abc123", "file_size_bytes": 1000},
                    "contact": {
                        "maintainer": "Test Team",
                        "repository": "https://github.com/test/repo",
                        "documentation": "https://docs.test.com",
                    },
                    "license": {
                        "name": "Apache-2.0",
                        "url": "https://www.apache.org/licenses/LICENSE-2.0",
                    },
                }

                # Create release bundle
                bundle_path = generator.create_release_bundle(test_manifest, "v1.0.0")

                # Verify bundle structure
                bundle_dir = Path(bundle_path)
                assert bundle_dir.exists()
                assert (bundle_dir / "model.pt").exists()
                assert (bundle_dir / "manifest.json").exists()
                assert (bundle_dir / "README.md").exists()
                assert (bundle_dir / "requirements.txt").exists()

                # Verify README content
                readme_content = (bundle_dir / "README.md").read_text()
                assert "test-model" in readme_content
                assert "v1.0.0" in readme_content
                assert "Accuracy" in readme_content

                # Verify requirements.txt
                requirements_content = (bundle_dir / "requirements.txt").read_text()
                assert "torch" in requirements_content
                assert "transformers" in requirements_content

        finally:
            os.unlink(model_path)

    def test_readme_generation(self):
        """Test README generation"""
        generator = DeploymentManifestGenerator(__file__)

        # Create test manifest
        test_manifest = {
            "model_info": {
                "name": "test-model",
                "version": "v1.0.0",
                "description": "A test model for unit testing",
                "file_size_mb": 256.5,
                "compression_pipeline": "BitNet -> SeedLM",
                "compression_ratio": 8.5,
            },
            "evaluation_metrics": {
                "accuracy": 0.823,
                "perplexity": 15.2,
                "bleu_score": 0.654,
                "inference_time_ms": 142.7,
            },
            "deployment_requirements": {
                "deployment_tier": "mobile",
                "hardware_requirements": {
                    "min_ram_gb": 4,
                    "min_vram_gb": 2,
                    "min_storage_gb": 2,
                },
            },
            "security": {"sha256_hash": "abc123def456", "file_size_bytes": 268435456},
            "contact": {
                "maintainer": "Test Team",
                "repository": "https://github.com/test/repo",
                "documentation": "https://docs.test.com",
            },
            "license": {
                "name": "Apache-2.0",
                "url": "https://www.apache.org/licenses/LICENSE-2.0",
            },
        }

        # Generate README
        readme_content = generator._generate_readme(test_manifest)

        # Verify README content
        assert "# test-model" in readme_content
        assert "v1.0.0" in readme_content
        assert "A test model for unit testing" in readme_content
        assert "256.5 MB" in readme_content
        assert "BitNet -> SeedLM" in readme_content
        assert "8.5" in readme_content  # compression ratio
        assert "0.823" in readme_content  # accuracy
        assert "mobile" in readme_content  # deployment tier
        assert "4 GB" in readme_content  # RAM requirement
        assert "abc123def456" in readme_content  # SHA256
        assert "Quick Start" in readme_content
        assert "Installation" in readme_content
        assert "Security" in readme_content
        assert "License" in readme_content


class TestDeploymentIntegration:
    """Integration tests for deployment system"""

    @patch(
        "agent_forge.deployment.manifest_generator.DeploymentManifestGenerator.run_evaluation"
    )
    def test_end_to_end_manifest_generation(self, mock_evaluation):
        """Test end-to-end manifest generation"""
        # Mock evaluation to avoid dependencies
        mock_evaluation.return_value = {
            "accuracy": 0.78,
            "perplexity": 18.3,
            "bleu_score": 0.61,
            "rouge_l": 0.55,
            "inference_time_ms": 167.0,
            "memory_usage_mb": 384.0,
            "throughput_tokens_per_sec": 28.5,
        }

        # Create realistic test model
        test_model_data = {
            "compressed_state": {
                "transformer.wte.weight": {
                    "compressed_blocks": [
                        {
                            "seed": 12345,
                            "coeff": torch.tensor([1, 2, 3]),
                            "exp": 2,
                            "error": 0.01,
                        }
                    ],
                    "original_shape": (50000, 768),
                    "compression_ratio": 12.5,
                },
                "transformer.h.0.attn.c_attn.weight": {
                    "compressed_blocks": [
                        {
                            "seed": 67890,
                            "coeff": torch.tensor([4, 5, 6]),
                            "exp": 1,
                            "error": 0.02,
                        }
                    ],
                    "original_shape": (768, 2304),
                    "compression_ratio": 8.2,
                },
            },
            "config": {
                "bitnet_enabled": True,
                "seedlm_enabled": True,
                "max_sequence_length": 1024,
                "target_compression_ratio": 10.0,
            },
            "compression_stats": {
                "transformer.wte.weight": {"compression_ratio": 12.5},
                "transformer.h.0.attn.c_attn.weight": {"compression_ratio": 8.2},
            },
            "model_info": {
                "model_path": "microsoft/DialoGPT-medium",
                "tokenizer_config": {"vocab_size": 50000},
            },
        }

        # Save test model
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(test_model_data, f.name)
            model_path = f.name

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create manifest generator
                generator = DeploymentManifestGenerator(model_path, temp_dir)

                # Generate manifest
                manifest = generator.generate_manifest("v0.1.0-test")

                # Create release bundle
                bundle_path = generator.create_release_bundle(manifest, "v0.1.0-test")

                # Verify complete bundle
                bundle_dir = Path(bundle_path)
                assert bundle_dir.exists()

                # Verify all files exist
                required_files = [
                    "model.pt",
                    "manifest.json",
                    "README.md",
                    "requirements.txt",
                ]
                for file_name in required_files:
                    assert (bundle_dir / file_name).exists()

                # Verify manifest content
                manifest_path = bundle_dir / "manifest.json"
                with open(manifest_path) as f:
                    loaded_manifest = json.load(f)

                # Verify key fields
                assert (
                    loaded_manifest["model_info"]["name"] == "agent-forge-v0.1.0-test"
                )
                assert (
                    loaded_manifest["model_info"]["compression_pipeline"]
                    == "BitNet -> SeedLM"
                )
                assert loaded_manifest["evaluation_metrics"]["accuracy"] == 0.78
                assert loaded_manifest["security"]["sha256_hash"] is not None

                # Verify deployment requirements are reasonable
                deployment_req = loaded_manifest["deployment_requirements"]
                assert "deployment_tier" in deployment_req
                assert "hardware_requirements" in deployment_req
                assert "software_requirements" in deployment_req

        finally:
            os.unlink(model_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
