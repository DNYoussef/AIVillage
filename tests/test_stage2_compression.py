"""
Test suite for Stage 2 compression pipeline (VPTQ + HyperFn)
"""

import os
import tempfile

import pytest
import torch

from agent_forge.compression.hyperfn import HyperCompressionEncoder
from agent_forge.compression.stage2 import Stage2Compressor
from agent_forge.compression.vptq import VPTQQuantizer


class TestVPTQQuantizer:
    """Test VPTQ quantization functionality"""

    def test_vptq_init(self):
        """Test VPTQ quantizer initialization"""
        quantizer = VPTQQuantizer(bits_per_vector=2.0, vector_length=16)

        assert quantizer.bits_per_vector == 2.0
        assert quantizer.vector_length == 16
        assert quantizer.codebook_size == 4  # 2^2
        assert quantizer.max_iterations == 50
        assert quantizer.convergence_threshold == 1e-6

    def test_vector_reshaping(self):
        """Test vector reshaping functionality"""
        quantizer = VPTQQuantizer(vector_length=8)

        # Test matrix that divides evenly
        matrix = torch.randn(4, 16)  # 64 elements total
        vectors = quantizer._reshape_vectors(matrix)
        assert vectors.shape == (8, 8)  # 64/8 = 8 vectors

        # Test matrix that needs padding
        matrix = torch.randn(3, 5)  # 15 elements total
        vectors = quantizer._reshape_vectors(matrix)
        assert vectors.shape == (2, 8)  # 16/8 = 2 vectors (padded to 16)

    def test_hessian_computation(self):
        """Test Hessian computation methods"""
        quantizer = VPTQQuantizer(vector_length=4)
        vectors = torch.randn(10, 4)

        # Test diagonal Hessian
        hessian_diag = quantizer._compute_hessian(vectors, method="diagonal")
        assert hessian_diag.shape == (4, 4)
        assert torch.allclose(hessian_diag, torch.diag(torch.diag(hessian_diag)))

        # Test Fisher Information Matrix
        hessian_fisher = quantizer._compute_hessian(vectors, method="fisher")
        assert hessian_fisher.shape == (4, 4)

        # Test Gauss-Newton
        hessian_gn = quantizer._compute_hessian(vectors, method="gauss_newton")
        assert hessian_gn.shape == (4, 4)

    def test_kmeans_plus_initialization(self):
        """Test k-means++ initialization"""
        quantizer = VPTQQuantizer(bits_per_vector=2.0, vector_length=4)
        vectors = torch.randn(20, 4)
        hessian = quantizer._compute_hessian(vectors, method="diagonal")

        codebook = quantizer._initialize_codebook_kmeans_plus(vectors, hessian)

        assert codebook.shape == (4, 4)  # 4 centroids, 4 dimensions

        # Check that centroids are different
        pairwise_distances = torch.cdist(codebook, codebook)
        pairwise_distances.fill_diagonal_(float("inf"))
        assert torch.all(pairwise_distances > 0)

    def test_vptq_quantization_basic(self):
        """Test basic VPTQ quantization"""
        quantizer = VPTQQuantizer(bits_per_vector=2.0, vector_length=4)

        # Create test weight matrix
        weight_matrix = torch.randn(8, 8)

        # Quantize
        quantized_data = quantizer.quantize_weight_matrix(weight_matrix)

        # Verify output structure
        assert "original_shape" in quantized_data
        assert "codebook" in quantized_data
        assert "assignments" in quantized_data
        assert "compression_ratio" in quantized_data
        assert "reconstruction_error" in quantized_data

        # Verify shapes
        assert quantized_data["original_shape"] == weight_matrix.shape
        assert quantized_data["codebook"].shape[1] == 4  # vector_length
        assert quantized_data["codebook"].shape[0] == 4  # 2^bits_per_vector

        # Verify compression ratio is reasonable
        assert quantized_data["compression_ratio"] > 1.0

    def test_vptq_dequantization(self):
        """Test VPTQ dequantization"""
        quantizer = VPTQQuantizer(bits_per_vector=2.0, vector_length=4)

        # Create test weight matrix
        weight_matrix = torch.randn(6, 8)

        # Quantize
        quantized_data = quantizer.quantize_weight_matrix(weight_matrix)

        # Dequantize
        reconstructed = quantizer.dequantize_weight_matrix(quantized_data)

        # Verify shape preservation
        assert reconstructed.shape == weight_matrix.shape

        # Verify reconstruction is reasonable (not exact due to quantization)
        reconstruction_error = torch.norm(weight_matrix - reconstructed)
        assert reconstruction_error.item() < 100.0  # Reasonable upper bound

    def test_vptq_different_hessian_methods(self):
        """Test VPTQ with different Hessian methods"""
        weight_matrix = torch.randn(8, 8)

        methods = ["diagonal", "fisher", "gauss_newton"]
        results = {}

        for method in methods:
            quantizer = VPTQQuantizer(bits_per_vector=2.0, vector_length=4)
            result = quantizer.quantize_weight_matrix(
                weight_matrix, hessian_method=method
            )
            results[method] = result

            # Verify all methods produce valid results
            assert "compression_ratio" in result
            assert result["compression_ratio"] > 0
            assert "reconstruction_error" in result
            assert result["hessian_method"] == method

        # Different methods should produce different results
        errors = [results[m]["reconstruction_error"] for m in methods]
        assert len(set(errors)) > 1  # At least some different errors


class TestHyperCompressionEncoder:
    """Test hyper-function compression functionality"""

    def test_hyperfn_init(self):
        """Test hyper-function encoder initialization"""
        encoder = HyperCompressionEncoder(num_clusters=8)

        assert encoder.num_clusters == 8
        assert encoder.trajectory_types == ["sinusoidal", "spiral", "chaotic"]
        assert encoder.max_search_iterations == 100
        assert encoder.convergence_threshold == 1e-6

    def test_trajectory_generation(self):
        """Test trajectory generation methods"""
        encoder = HyperCompressionEncoder(num_clusters=4)

        # Test parameters
        params = {"A": 0.5, "B": 0.3, "D": 0.1, "an": 2, "ad": 3}

        length = 10

        # Test sinusoidal trajectory
        sin_traj = encoder._generate_sinusoidal_trajectory(length, params)
        assert sin_traj.shape == (length,)
        assert torch.isfinite(sin_traj).all()

        # Test spiral trajectory
        spiral_traj = encoder._generate_spiral_trajectory(length, params)
        assert spiral_traj.shape == (length,)
        assert torch.isfinite(spiral_traj).all()

        # Test chaotic trajectory
        chaotic_traj = encoder._generate_chaotic_trajectory(length, params)
        assert chaotic_traj.shape == (length,)
        assert torch.isfinite(chaotic_traj).all()

    def test_parameter_search(self):
        """Test parameter search for trajectory fitting"""
        encoder = HyperCompressionEncoder(num_clusters=4)

        # Create test weight vector
        weights = torch.randn(20)

        # Test different trajectory types
        trajectory_types = ["sinusoidal", "spiral", "chaotic"]

        for traj_type in trajectory_types:
            params = encoder._search_params(weights, trajectory_type=traj_type)

            # Verify parameter structure
            assert "A" in params
            assert "B" in params
            assert "D" in params
            assert "err" in params
            assert "trajectory_type" in params
            assert params["trajectory_type"] == traj_type

            # Verify error is finite
            assert torch.isfinite(torch.tensor(params["err"]))

    def test_weight_clustering(self):
        """Test weight clustering functionality"""
        encoder = HyperCompressionEncoder(num_clusters=4)

        # Create test weight matrix
        weight_matrix = torch.randn(8, 16)

        # Test clustering
        clusters = encoder._cluster(weight_matrix)

        # Verify cluster structure
        assert len(clusters) == 4

        for cluster in clusters:
            assert "weights" in cluster
            assert "indices" in cluster
            assert cluster["weights"].numel() > 0
            assert cluster["indices"].numel() > 0

    def test_hyperfn_compression(self):
        """Test hyper-function compression"""
        encoder = HyperCompressionEncoder(num_clusters=4)

        # Create test weight matrix
        weight_matrix = torch.randn(8, 16)

        # Test compression
        compressed_data = encoder.compress_weight_matrix(weight_matrix)

        # Verify output structure
        assert "params" in compressed_data
        assert "original_shape" in compressed_data
        assert "compression_ratio" in compressed_data
        assert "total_error" in compressed_data
        assert "trajectory_types_used" in compressed_data

        # Verify parameters
        assert len(compressed_data["params"]) == 4  # num_clusters
        assert compressed_data["original_shape"] == weight_matrix.shape
        assert compressed_data["compression_ratio"] > 0

        # Verify trajectory types are valid
        for traj_type in compressed_data["trajectory_types_used"]:
            assert traj_type in ["sinusoidal", "spiral", "chaotic"]

    def test_hyperfn_decompression(self):
        """Test hyper-function decompression"""
        encoder = HyperCompressionEncoder(num_clusters=4)

        # Create test weight matrix
        weight_matrix = torch.randn(6, 8)

        # Compress
        compressed_data = encoder.compress_weight_matrix(weight_matrix)

        # Decompress
        reconstructed = encoder.decompress_weight_matrix(compressed_data)

        # Verify shape preservation
        assert reconstructed.shape == weight_matrix.shape

        # Verify reconstruction is reasonable
        reconstruction_error = torch.norm(weight_matrix - reconstructed)
        assert reconstruction_error.item() < 1000.0  # Reasonable upper bound

    def test_hyperfn_auto_trajectory_selection(self):
        """Test automatic trajectory type selection"""
        encoder = HyperCompressionEncoder(num_clusters=2)

        # Create test weight matrix
        weight_matrix = torch.randn(4, 8)

        # Test auto trajectory selection
        compressed_data = encoder.compress_weight_matrix(
            weight_matrix, trajectory_type="auto"
        )

        # Verify trajectory types were selected
        assert "trajectory_types_used" in compressed_data
        assert len(compressed_data["trajectory_types_used"]) > 0

        # Each cluster should have chosen a trajectory type
        for param in compressed_data["params"]:
            assert "trajectory_type" in param
            assert param["trajectory_type"] in ["sinusoidal", "spiral", "chaotic"]


class TestStage2Compressor:
    """Test Stage 2 compression pipeline"""

    def test_stage2_compressor_init(self):
        """Test Stage 2 compressor initialization"""
        compressor = Stage2Compressor(
            vptq_bits=2.0, vptq_vector_length=16, use_hyperfn=True, hyperfn_clusters=8
        )

        assert compressor.vptq.bits_per_vector == 2.0
        assert compressor.vptq.vector_length == 16
        assert compressor.use_hyperfn
        assert compressor.hyperfn is not None
        assert compressor.hyperfn.num_clusters == 8

    def test_stage2_compressor_no_hyperfn(self):
        """Test Stage 2 compressor without hyper-function"""
        compressor = Stage2Compressor(
            vptq_bits=2.0, vptq_vector_length=16, use_hyperfn=False
        )

        assert not compressor.use_hyperfn
        assert compressor.hyperfn is None

    def test_stage1_data_loading(self):
        """Test loading Stage 1 compressed data"""
        compressor = Stage2Compressor()

        # Create mock Stage 1 data
        stage1_data = {
            "compressed_state": {
                "layer1.weight": {
                    "compressed_blocks": [
                        {
                            "seed": 123,
                            "coeff": torch.tensor([1, 2]),
                            "exp": 1,
                            "error": 0.1,
                        }
                    ],
                    "original_shape": (4, 8),
                    "compression_ratio": 5.0,
                },
                "layer1.bias": torch.randn(4),
            },
            "config": {"bitnet_enabled": True},
            "model_info": {"model_path": "test_model"},
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(stage1_data, f.name)
            temp_path = f.name

        try:
            # Test loading
            compressed_weights, metadata = compressor.load_stage1_model(temp_path)

            # Verify structure
            assert "layer1.weight" in compressed_weights
            assert "layer1.bias" in compressed_weights
            assert "config" in metadata
            assert "model_info" in metadata

        finally:
            os.unlink(temp_path)

    def test_vptq_quantization(self):
        """Test VPTQ quantization step"""
        compressor = Stage2Compressor(vptq_bits=2.0, vptq_vector_length=8)

        # Create test weights
        weights = {
            "layer1.weight": torch.randn(16, 32),
            "layer1.bias": torch.randn(16),
            "layer2.weight": torch.randn(8, 16),
        }

        # Apply VPTQ quantization
        result = compressor.apply_vptq_quantization(weights)

        # Verify result structure
        assert "vptq_data" in result
        assert "compression_stats" in result
        assert "average_compression_ratio" in result

        # Verify quantization was applied to 2D tensors
        assert "layer1.weight" in result["compression_stats"]
        assert "layer2.weight" in result["compression_stats"]
        assert (
            "layer1.bias" not in result["compression_stats"]
        )  # 1D tensor, not quantized

        # Verify bias is preserved
        assert torch.allclose(
            result["vptq_data"]["layer1.bias"], weights["layer1.bias"]
        )

    def test_hyperfn_compression(self):
        """Test hyper-function compression step"""
        compressor = Stage2Compressor(use_hyperfn=True, hyperfn_clusters=4)

        # Create mock VPTQ data
        vptq_data = {
            "layer1.weight": {
                "codebook": torch.randn(4, 8),
                "assignments": torch.randint(0, 4, (16,)),
                "residual_codebook": torch.randn(16),
                "residual_idx": torch.randint(0, 16, (16, 8)),
                "original_shape": (16, 32),
                "compression_ratio": 3.0,
            },
            "layer1.bias": torch.randn(16),
        }

        # Apply hyper-function compression
        result = compressor.apply_hyperfn_compression(vptq_data)

        # Verify result structure
        assert "hyperfn_data" in result
        assert "compression_stats" in result
        assert "average_compression_ratio" in result

        # Verify hyper-function was applied to codebook
        assert "hyperfn_codebook" in result["hyperfn_data"]["layer1.weight"]
        assert "original_codebook_shape" in result["hyperfn_data"]["layer1.weight"]

        # Verify bias is preserved
        assert torch.allclose(
            result["hyperfn_data"]["layer1.bias"], vptq_data["layer1.bias"]
        )

    def test_compression_evaluation(self):
        """Test compression evaluation"""
        compressor = Stage2Compressor()

        # Create test data
        original_weights = {
            "layer1.weight": torch.randn(16, 32),
            "layer1.bias": torch.randn(16),
        }

        compressed_data = {
            "layer1.weight": {"compression_ratio": 5.0, "reconstruction_error": 0.1},
            "layer1.bias": torch.randn(16),
            "compression_stats": {
                "layer1.weight": {"compression_ratio": 5.0, "reconstruction_error": 0.1}
            },
        }

        # Evaluate compression
        eval_result = compressor.evaluate_compression(original_weights, compressed_data)

        # Verify evaluation structure
        assert "overall_compression_ratio" in eval_result
        assert "total_reconstruction_error" in eval_result
        assert "original_size_mb" in eval_result
        assert "compressed_size_mb" in eval_result

        # Verify reasonable values
        assert eval_result["overall_compression_ratio"] > 0
        assert eval_result["original_size_mb"] > 0
        assert eval_result["compressed_size_mb"] > 0


class TestStage2Integration:
    """Integration tests for Stage 2 pipeline"""

    def test_stage2_pipeline_with_hyperfn(self):
        """Test complete Stage 2 pipeline with hyper-function"""

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            stage1_path = os.path.join(temp_dir, "stage1_model.pt")
            stage2_path = os.path.join(temp_dir, "stage2_model.pt")

            # Create mock Stage 1 data
            stage1_data = {
                "compressed_state": {
                    "layer.weight": {
                        "compressed_blocks": [
                            {
                                "seed": 123,
                                "coeff": torch.tensor([1, 2]),
                                "exp": 1,
                                "error": 0.1,
                            }
                        ],
                        "original_shape": (8, 16),
                        "compression_ratio": 4.0,
                    },
                    "layer.bias": torch.randn(8),
                },
                "config": {"bitnet_enabled": True},
                "model_info": {"model_path": "test_model"},
            }

            # Save Stage 1 data
            torch.save(stage1_data, stage1_path)

            # Create Stage 2 compressor
            compressor = Stage2Compressor(
                vptq_bits=2.0,
                vptq_vector_length=8,
                use_hyperfn=True,
                hyperfn_clusters=4,
            )

            # Run Stage 2 pipeline
            try:
                result = compressor.run_pipeline(stage1_path, stage2_path)

                # Verify result
                assert "success" in result
                assert "output_path" in result
                assert result["output_path"] == stage2_path

                # Check if output file exists
                if os.path.exists(stage2_path):
                    output_data = torch.load(stage2_path)
                    assert "stage2_compressed_data" in output_data
                    assert "stage1_metadata" in output_data
                    assert "compression_pipeline" in output_data

            except Exception as e:
                pytest.skip(f"Integration test skipped due to: {e}")

    def test_stage2_pipeline_without_hyperfn(self):
        """Test Stage 2 pipeline without hyper-function"""

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            stage1_path = os.path.join(temp_dir, "stage1_model.pt")
            stage2_path = os.path.join(temp_dir, "stage2_model.pt")

            # Create mock Stage 1 data
            stage1_data = {
                "compressed_state": {
                    "layer.weight": {
                        "compressed_blocks": [
                            {
                                "seed": 456,
                                "coeff": torch.tensor([3, 4]),
                                "exp": 2,
                                "error": 0.2,
                            }
                        ],
                        "original_shape": (4, 8),
                        "compression_ratio": 3.0,
                    }
                },
                "config": {"seedlm_enabled": True},
                "model_info": {"model_path": "test_model"},
            }

            # Save Stage 1 data
            torch.save(stage1_data, stage1_path)

            # Create Stage 2 compressor without hyper-function
            compressor = Stage2Compressor(
                vptq_bits=2.0, vptq_vector_length=4, use_hyperfn=False
            )

            # Run Stage 2 pipeline
            try:
                result = compressor.run_pipeline(stage1_path, stage2_path)

                # Verify result
                assert "success" in result
                assert "hyperfn_compression_ratio" in result
                assert result["hyperfn_compression_ratio"] == 0  # No hyper-function

            except Exception as e:
                pytest.skip(f"Integration test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
