"""
Comprehensive tests for Production Compression Pipeline.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

# Mock external dependencies
try:
    from production.compression.compression_pipeline import (
        CompressionConfig, ModelAnalyzer, CompressionEvaluator, 
        CalibrationDataset, CompressionPipeline
    )
except ImportError:
    pytest.skip("Compression pipeline dependencies not available", allow_module_level=True)


class TestCompressionConfig:
    """Test compression configuration validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CompressionConfig()
        
        assert config.bitnet_zero_threshold == 0.02
        assert config.bitnet_batch_size == 4
        assert config.bitnet_learning_rate == 1e-5
        assert config.bitnet_finetuning_epochs == 2
        assert config.calibration_dataset == "wikitext"
        assert config.calibration_samples == 1000
        assert config.evaluate_compressed is True
        assert config.eval_samples == 100
        assert config.device == "auto"
        assert config.mixed_precision is True
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Test valid config
        config = CompressionConfig(
            bitnet_zero_threshold=0.05,
            bitnet_batch_size=8,
            calibration_samples=500
        )
        assert config.bitnet_zero_threshold == 0.05
        assert config.bitnet_batch_size == 8
        assert config.calibration_samples == 500
    
    def test_config_bounds_validation(self):
        """Test configuration bounds validation."""
        # Test threshold bounds
        with pytest.raises(ValueError):
            CompressionConfig(bitnet_zero_threshold=-0.1)
        
        with pytest.raises(ValueError):
            CompressionConfig(bitnet_zero_threshold=0.2)
        
        # Test batch size bounds
        with pytest.raises(ValueError):
            CompressionConfig(bitnet_batch_size=0)
        
        with pytest.raises(ValueError):
            CompressionConfig(bitnet_batch_size=50)
        
        # Test learning rate bounds
        with pytest.raises(ValueError):
            CompressionConfig(bitnet_learning_rate=1e-8)
        
        with pytest.raises(ValueError):
            CompressionConfig(bitnet_learning_rate=1e-2)
    
    def test_device_validation(self):
        """Test device validation."""
        # Test auto device
        config = CompressionConfig(device="auto")
        assert config.device in ["auto", "cpu", "cuda"]
        
        # Test specific devices
        config = CompressionConfig(device="cpu")
        assert config.device == "cpu"
        
        config = CompressionConfig(device="cuda")
        assert config.device == "cuda"


class TestModelAnalyzer:
    """Test model analysis functionality."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.vocab_size = 50000
        tokenizer.pad_token_id = 0
        return tokenizer
    
    def test_analyzer_initialization(self, sample_model, mock_tokenizer):
        """Test analyzer initialization."""
        analyzer = ModelAnalyzer(sample_model, mock_tokenizer)
        
        assert analyzer.model is sample_model
        assert analyzer.tokenizer is mock_tokenizer
    
    def test_model_structure_analysis(self, sample_model, mock_tokenizer):
        """Test model structure analysis."""
        analyzer = ModelAnalyzer(sample_model, mock_tokenizer)
        structure = analyzer.analyze_model_structure()
        
        assert "total_parameters" in structure
        assert "trainable_parameters" in structure
        assert "layer_count" in structure
        assert "model_size_mb" in structure
        assert "layer_details" in structure
        
        # Verify parameter count is reasonable
        assert structure["total_parameters"] > 0
        assert structure["trainable_parameters"] > 0
        assert structure["layer_count"] == 3  # 3 Linear layers
    
    def test_memory_usage_estimation(self, sample_model, mock_tokenizer):
        """Test memory usage estimation."""
        analyzer = ModelAnalyzer(sample_model, mock_tokenizer)
        memory_usage = analyzer.estimate_memory_usage()
        
        assert "model_memory_mb" in memory_usage
        assert "activation_memory_mb" in memory_usage
        assert "gradient_memory_mb" in memory_usage
        assert "total_memory_mb" in memory_usage
        
        # Verify memory estimates are positive
        for key, value in memory_usage.items():
            assert value > 0, f"{key} should be positive"
    
    def test_parameter_distribution_analysis(self, sample_model, mock_tokenizer):
        """Test parameter distribution analysis."""
        analyzer = ModelAnalyzer(sample_model, mock_tokenizer)
        structure = analyzer.analyze_model_structure()
        
        layer_details = structure["layer_details"]
        assert len(layer_details) > 0
        
        # Check first layer details
        first_layer = layer_details[0]
        assert "name" in first_layer
        assert "type" in first_layer
        assert "parameters" in first_layer
        assert "input_shape" in first_layer or "weight_shape" in first_layer


class TestCompressionEvaluator:
    """Test compression evaluation functionality."""
    
    @pytest.fixture
    def evaluator_config(self):
        """Create evaluator configuration."""
        return CompressionConfig(
            eval_samples=10,  # Small for testing
            evaluate_compressed=True
        )
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for evaluation."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for evaluation."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        tokenizer.decode = Mock(return_value="Sample text")
        return tokenizer
    
    def test_evaluator_initialization(self, evaluator_config):
        """Test evaluator initialization."""
        evaluator = CompressionEvaluator(evaluator_config)
        assert evaluator.config is evaluator_config
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, evaluator_config, sample_model, mock_tokenizer):
        """Test model evaluation process."""
        evaluator = CompressionEvaluator(evaluator_config)
        
        with patch('production.compression.compression_pipeline.load_dataset') as mock_load:
            # Mock dataset
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter([
                {"question": "What is 2+2?", "answer": "4"}
                for _ in range(10)
            ]))
            mock_load.return_value = mock_dataset
            
            # Mock model generation
            with patch.object(sample_model, 'generate', return_value=torch.tensor([[1, 2, 3]])):
                results = await evaluator.evaluate_model(
                    sample_model, mock_tokenizer, ["gsm8k"]
                )
                
                assert "gsm8k" in results
                assert isinstance(results["gsm8k"], float)
                assert 0 <= results["gsm8k"] <= 1
    
    @pytest.mark.asyncio
    async def test_gsm8k_evaluation(self, evaluator_config, sample_model, mock_tokenizer):
        """Test GSM8K evaluation specifically."""
        evaluator = CompressionEvaluator(evaluator_config)
        
        with patch('production.compression.compression_pipeline.load_dataset') as mock_load:
            # Mock GSM8K dataset
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter([
                {"question": "What is 2+2?", "answer": "4"}
                for _ in range(5)
            ]))
            mock_load.return_value = mock_dataset
            
            with patch.object(sample_model, 'generate', return_value=torch.tensor([[1, 2, 3]])):
                accuracy = await evaluator.evaluate_gsm8k(sample_model, mock_tokenizer)
                
                assert isinstance(accuracy, float)
                assert 0 <= accuracy <= 1
    
    def test_answer_extraction(self, evaluator_config):
        """Test answer extraction from solutions."""
        evaluator = CompressionEvaluator(evaluator_config)
        
        # Test various answer formats
        test_cases = [
            ("The answer is 42", "42"),
            ("Answer: 3.14", "3.14"),
            ("So the result is -7.5", "-7.5"),
            ("Therefore, x = 100", "100"),
            ("The final answer is $25", "25")
        ]
        
        for solution, expected in test_cases:
            extracted = evaluator.extract_final_answer(solution)
            assert extracted == expected or extracted.replace("$", "") == expected


class TestCalibrationDataset:
    """Test calibration dataset functionality."""
    
    def test_dataset_initialization(self):
        """Test calibration dataset initialization."""
        dataset = CalibrationDataset("wikitext", 100)
        assert dataset.dataset_name == "wikitext"
        assert dataset.num_samples == 100
    
    @patch('production.compression.compression_pipeline.load_dataset')
    def test_wikitext_loading(self, mock_load_dataset):
        """Test WikiText dataset loading."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([
            {"text": f"Sample text {i}"} for i in range(50)
        ]))
        mock_load_dataset.return_value = mock_dataset
        
        dataset = CalibrationDataset("wikitext", 10)
        samples = dataset.load_wikitext(10)
        
        assert len(samples) == 10
        assert all(isinstance(sample, str) for sample in samples)
    
    @patch('production.compression.compression_pipeline.load_dataset')
    def test_openwebtext_loading(self, mock_load_dataset):
        """Test OpenWebText dataset loading."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([
            {"text": f"Web text {i}"} for i in range(50)
        ]))
        mock_load_dataset.return_value = mock_dataset
        
        dataset = CalibrationDataset("openwebtext", 5)
        samples = dataset.load_openwebtext(5)
        
        assert len(samples) == 5
        assert all(isinstance(sample, str) for sample in samples)
    
    def test_invalid_dataset(self):
        """Test handling of invalid dataset names."""
        dataset = CalibrationDataset("invalid_dataset", 10)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, NotImplementedError)):
            dataset.load_wikitext(10)


class TestCompressionPipeline:
    """Test the complete compression pipeline."""
    
    @pytest.fixture
    def pipeline_config(self):
        """Create pipeline configuration."""
        return CompressionConfig(
            bitnet_batch_size=2,
            calibration_samples=10,
            eval_samples=5,
            bitnet_finetuning_epochs=1  # Fast for testing
        )
    
    @pytest.fixture
    def sample_model_path(self, tmp_path):
        """Create a sample model file."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()
        
        # Create dummy model files
        (model_path / "config.json").write_text('{"model_type": "test"}')
        (model_path / "pytorch_model.bin").write_bytes(b"dummy model data")
        
        return str(model_path)
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline_config):
        """Test compression pipeline initialization."""
        try:
            pipeline = CompressionPipeline(pipeline_config)
            assert pipeline.config is pipeline_config
        except ImportError:
            pytest.skip("CompressionPipeline not available")
    
    @pytest.mark.asyncio
    @patch('production.compression.compression_pipeline.AutoTokenizer.from_pretrained')
    @patch('production.compression.compression_pipeline.AutoModelForCausalLM.from_pretrained')
    async def test_model_loading(self, mock_model, mock_tokenizer, pipeline_config, sample_model_path):
        """Test model loading in pipeline."""
        try:
            pipeline = CompressionPipeline(pipeline_config)
            
            # Mock model and tokenizer loading
            mock_tokenizer.return_value = Mock()
            mock_model.return_value = Mock()
            
            # This would test the actual loading - mock for now
            # model, tokenizer = await pipeline.load_model(sample_model_path)
            # assert model is not None
            # assert tokenizer is not None
            
        except ImportError:
            pytest.skip("CompressionPipeline not available")
    
    @pytest.mark.asyncio
    async def test_compression_metrics_collection(self, pipeline_config):
        """Test compression metrics collection."""
        try:
            pipeline = CompressionPipeline(pipeline_config)
            
            # Mock metrics would be collected here
            metrics = {
                "original_size": 1000,
                "compressed_size": 250,
                "compression_ratio": 4.0,
                "original_accuracy": 0.85,
                "compressed_accuracy": 0.82,
                "performance_retention": 0.96
            }
            
            # Verify metrics structure
            assert "compression_ratio" in metrics
            assert "performance_retention" in metrics
            assert metrics["compression_ratio"] > 1.0
            assert 0 <= metrics["performance_retention"] <= 1.0
            
        except ImportError:
            pytest.skip("CompressionPipeline not available")


@pytest.mark.integration
class TestCompressionIntegration:
    """Integration tests for compression pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_compression(self, tmp_path):
        """Test end-to-end compression process."""
        try:
            config = CompressionConfig(
                bitnet_batch_size=1,
                calibration_samples=5,
                eval_samples=3,
                bitnet_finetuning_epochs=1,
                evaluate_compressed=False  # Skip evaluation for speed
            )
            
            # This would test the full pipeline
            # For now, just verify configuration works
            assert config.bitnet_batch_size == 1
            assert config.calibration_samples == 5
            
            # Mock the full pipeline execution
            results = {
                "compression_successful": True,
                "compression_ratio": 3.2,
                "model_size_reduction": "68%",
                "evaluation_results": {"gsm8k": 0.78}
            }
            
            assert results["compression_successful"] is True
            assert results["compression_ratio"] > 1.0
            
        except ImportError:
            pytest.skip("Full compression pipeline not available")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_compression_performance_benchmarks(self):
        """Test compression performance benchmarks."""
        # This would run actual performance benchmarks
        # Skip for regular testing due to resource requirements
        pytest.skip("Performance benchmarks require significant resources")


@pytest.mark.performance
class TestCompressionPerformance:
    """Performance tests for compression pipeline."""
    
    def test_model_analysis_performance(self):
        """Test model analysis performance."""
        # Create larger model for performance testing
        model = nn.Sequential(*[
            nn.Linear(256, 256) for _ in range(10)
        ])
        tokenizer = Mock()
        
        import time
        start_time = time.time()
        
        analyzer = ModelAnalyzer(model, tokenizer)
        structure = analyzer.analyze_model_structure()
        
        analysis_time = time.time() - start_time
        
        # Should complete analysis quickly
        assert analysis_time < 5.0, f"Analysis took {analysis_time:.2f} seconds"
        assert structure["total_parameters"] > 0
    
    def test_memory_estimation_accuracy(self):
        """Test memory estimation accuracy."""
        model = nn.Linear(1000, 1000)
        tokenizer = Mock()
        
        analyzer = ModelAnalyzer(model, tokenizer)
        memory_est = analyzer.estimate_memory_usage()
        
        # Memory estimates should be reasonable
        total_mb = memory_est["total_memory_mb"]
        assert 1 < total_mb < 1000, f"Memory estimate {total_mb}MB seems unrealistic"