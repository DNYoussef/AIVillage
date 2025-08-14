"""Tests for Model Feature Extraction Framework - Prompt G

Comprehensive validation of feature extraction and model analysis including:
- Architecture feature extraction
- Weight and activation analysis
- Model comparison and similarity metrics
- Feature visualization utilities

Integration Point: Feature extraction validation for Phase 4 testing
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ml.feature_extraction import (
    ExtractionMethod,
    FeatureExtractionConfig,
    FeatureExtractor,
    FeatureType,
    FeatureVisualizer,
    ModelComparator,
    ModelComparison,
    ModelFeatures,
    SimilarityMetric,
    compare_model_files,
    extract_and_save_features,
    load_features,
)


class DummyModel:
    """Dummy model for testing."""

    def __init__(self, depth=3, width=100):
        self.depth = depth
        self.width = width
        self.conv1 = "conv_layer"
        self.linear1 = "linear_layer"
        self.norm1 = "norm_layer"
        self.attention = "attention_layer"
        self.history = {"loss": [1.0, 0.8, 0.6, 0.5], "accuracy": [0.5, 0.7, 0.8, 0.85]}

    def state_dict(self):
        """Mock state dict for PyTorch style."""
        return {
            "conv1.weight": MagicMock(
                detach=lambda: MagicMock(
                    cpu=lambda: MagicMock(numpy=lambda: np.random.randn(64, 3, 3, 3))
                )
            ),
            "linear1.weight": MagicMock(
                detach=lambda: MagicMock(
                    cpu=lambda: MagicMock(numpy=lambda: np.random.randn(100, 50))
                )
            ),
        }


class TestModelFeatures:
    """Test ModelFeatures data structure."""

    def test_model_features_creation(self):
        """Test ModelFeatures creation."""
        features = ModelFeatures(
            model_id="test_model",
            feature_type=FeatureType.WEIGHTS,
            features={"weights": np.random.randn(100), "biases": np.random.randn(10)},
            metadata={"extraction_time": 1.5},
        )

        assert features.model_id == "test_model"
        assert features.feature_type == FeatureType.WEIGHTS
        assert "weights" in features.features
        assert "biases" in features.features
        assert features.metadata["extraction_time"] == 1.5

    def test_get_feature_dimensions(self):
        """Test getting feature dimensions."""
        features = ModelFeatures(
            model_id="test",
            feature_type=FeatureType.WEIGHTS,
            features={"layer1": np.zeros((10, 20)), "layer2": np.zeros((5, 5, 3))},
        )

        dims = features.get_feature_dimensions()

        assert dims["layer1"] == (10, 20)
        assert dims["layer2"] == (5, 5, 3)

    def test_get_feature_statistics(self):
        """Test computing feature statistics."""
        test_array = np.array([1, 2, 3, 4, 5])
        features = ModelFeatures(
            model_id="test",
            feature_type=FeatureType.WEIGHTS,
            features={"test": test_array},
        )

        stats = features.get_feature_statistics()

        assert "test" in stats
        assert stats["test"]["mean"] == 3.0
        assert stats["test"]["min"] == 1.0
        assert stats["test"]["max"] == 5.0
        assert "std" in stats["test"]
        assert "norm" in stats["test"]
        assert "sparsity" in stats["test"]


class TestFeatureExtractionConfig:
    """Test feature extraction configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = FeatureExtractionConfig()

        assert FeatureType.WEIGHTS in config.feature_types
        assert config.extraction_method == ExtractionMethod.DIRECT
        assert config.normalize is True
        assert config.compute_statistics is True
        assert config.cache_features is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = FeatureExtractionConfig(
            feature_types=[FeatureType.ARCHITECTURE, FeatureType.STATISTICS],
            extraction_method=ExtractionMethod.POOLING,
            max_features=1000,
            normalize=False,
        )

        assert FeatureType.ARCHITECTURE in config.feature_types
        assert FeatureType.STATISTICS in config.feature_types
        assert config.extraction_method == ExtractionMethod.POOLING
        assert config.max_features == 1000
        assert config.normalize is False


class TestFeatureExtractor:
    """Test feature extraction functionality."""

    def test_extractor_initialization(self):
        """Test feature extractor initialization."""
        extractor = FeatureExtractor()

        assert isinstance(extractor.config, FeatureExtractionConfig)
        assert len(extractor.feature_cache) == 0
        assert len(extractor.extraction_stats) == 0

    def test_extract_features_basic(self):
        """Test basic feature extraction."""
        extractor = FeatureExtractor()
        model = DummyModel()

        features = extractor.extract_features(model, "test_model")

        assert isinstance(features, ModelFeatures)
        assert features.model_id == "test_model"
        assert len(features.features) > 0
        assert features.extraction_method == ExtractionMethod.DIRECT

    def test_extract_architecture_features(self):
        """Test architecture feature extraction."""
        config = FeatureExtractionConfig(feature_types=[FeatureType.ARCHITECTURE])
        extractor = FeatureExtractor(config)
        model = DummyModel()

        features = extractor.extract_features(model, "arch_model")

        assert "layer_counts" in features.features
        assert "connectivity" in features.features
        assert "arch_metrics" in features.features

        # Check layer counts
        layer_counts = features.features["layer_counts"]
        assert layer_counts[0] == 1  # conv layers
        assert layer_counts[1] == 1  # linear layers

    def test_extract_weight_features(self):
        """Test weight feature extraction."""
        config = FeatureExtractionConfig(feature_types=[FeatureType.WEIGHTS])
        extractor = FeatureExtractor(config)
        model = DummyModel()

        features = extractor.extract_features(model, "weight_model")

        assert "weight_statistics" in features.features
        assert "weight_distribution" in features.features
        assert "sparsity" in features.features

    def test_extract_statistical_features(self):
        """Test statistical feature extraction."""
        config = FeatureExtractionConfig(feature_types=[FeatureType.STATISTICS])
        extractor = FeatureExtractor(config)
        model = DummyModel()

        features = extractor.extract_features(model, "stats_model")

        assert "complexity" in features.features
        assert "init_stats" in features.features

    def test_feature_caching(self):
        """Test feature caching mechanism."""
        extractor = FeatureExtractor()
        model = DummyModel()

        # First extraction
        features1 = extractor.extract_features(model, "cached_model")
        assert extractor.extraction_stats["extractions"] == 1
        assert extractor.extraction_stats.get("cache_hits", 0) == 0

        # Second extraction (should use cache)
        features2 = extractor.extract_features(model, "cached_model")
        assert extractor.extraction_stats["cache_hits"] == 1
        assert features1.model_id == features2.model_id

    def test_extraction_methods(self):
        """Test different extraction methods."""
        model = DummyModel()

        # Test pooling
        config = FeatureExtractionConfig(extraction_method=ExtractionMethod.POOLING)
        extractor = FeatureExtractor(config)
        features = extractor.extract_features(model, "pooled_model")
        assert features.extraction_method == ExtractionMethod.POOLING

        # Test projection
        config = FeatureExtractionConfig(
            extraction_method=ExtractionMethod.PROJECTION, projection_dim=50
        )
        extractor = FeatureExtractor(config)
        features = extractor.extract_features(model, "projected_model")
        assert features.extraction_method == ExtractionMethod.PROJECTION

        # Test sampling
        config = FeatureExtractionConfig(
            extraction_method=ExtractionMethod.SAMPLING, sampling_rate=0.5
        )
        extractor = FeatureExtractor(config)
        features = extractor.extract_features(model, "sampled_model")
        assert features.extraction_method == ExtractionMethod.SAMPLING

    def test_feature_normalization(self):
        """Test feature normalization."""
        config = FeatureExtractionConfig(normalize=True)
        extractor = FeatureExtractor(config)
        model = DummyModel()

        features = extractor.extract_features(model, "normalized_model")

        # Check that features are normalized
        for feature_array in features.features.values():
            if feature_array.size > 0:
                norm = np.linalg.norm(feature_array)
                assert norm == pytest.approx(1.0, rel=1e-5) or norm == 0

    def test_apply_aggregation(self):
        """Test feature aggregation."""
        config = FeatureExtractionConfig(
            extraction_method=ExtractionMethod.AGGREGATION, max_features=100
        )
        extractor = FeatureExtractor(config)
        model = DummyModel()

        features = extractor.extract_features(model, "aggregated_model")

        assert "aggregated" in features.features
        aggregated = features.features["aggregated"]
        assert aggregated.size <= 100

    def test_apply_compression(self):
        """Test feature compression."""
        config = FeatureExtractionConfig(
            extraction_method=ExtractionMethod.COMPRESSION, compression_ratio=0.5
        )
        extractor = FeatureExtractor(config)
        model = DummyModel()

        features = extractor.extract_features(model, "compressed_model")
        assert features.extraction_method == ExtractionMethod.COMPRESSION


class TestModelComparator:
    """Test model comparison functionality."""

    def test_comparator_initialization(self):
        """Test model comparator initialization."""
        comparator = ModelComparator()

        assert isinstance(comparator.feature_extractor, FeatureExtractor)
        assert len(comparator.comparison_cache) == 0

    def test_compare_models_basic(self):
        """Test basic model comparison."""
        comparator = ModelComparator()
        model_a = DummyModel(depth=3, width=100)
        model_b = DummyModel(depth=3, width=100)

        comparison = comparator.compare_models(model_a, model_b, "model_a", "model_b")

        assert isinstance(comparison, ModelComparison)
        assert comparison.model_a_id == "model_a"
        assert comparison.model_b_id == "model_b"
        assert len(comparison.similarity_scores) > 0
        assert "cosine" in comparison.similarity_scores
        assert "structural" in comparison.similarity_scores

    def test_similarity_metrics(self):
        """Test different similarity metrics."""
        comparator = ModelComparator()
        model_a = DummyModel()
        model_b = DummyModel()

        metrics = [
            SimilarityMetric.COSINE,
            SimilarityMetric.EUCLIDEAN,
            SimilarityMetric.MANHATTAN,
            SimilarityMetric.STRUCTURAL,
            SimilarityMetric.FEATURE,
            SimilarityMetric.SEMANTIC,
        ]

        comparison = comparator.compare_models(
            model_a, model_b, "model_a", "model_b", metrics=metrics
        )

        for metric in metrics:
            assert metric.value in comparison.similarity_scores
            score = comparison.similarity_scores[metric.value]
            assert 0 <= score <= 1

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        comparator = ModelComparator()

        # Create mock features
        features_a = ModelFeatures(
            model_id="a",
            feature_type=FeatureType.WEIGHTS,
            features={"test": np.array([1, 0, 0])},
        )
        features_b = ModelFeatures(
            model_id="b",
            feature_type=FeatureType.WEIGHTS,
            features={"test": np.array([1, 0, 0])},
        )

        similarity = comparator._compute_cosine_similarity(features_a, features_b)
        assert similarity == pytest.approx(1.0)  # Identical vectors

        # Different vectors
        features_b.features["test"] = np.array([0, 1, 0])
        similarity = comparator._compute_cosine_similarity(features_a, features_b)
        assert similarity == pytest.approx(0.0)  # Orthogonal vectors

    def test_structural_similarity(self):
        """Test structural similarity computation."""
        comparator = ModelComparator()

        # Same structure
        features_a = ModelFeatures(
            model_id="a",
            feature_type=FeatureType.ARCHITECTURE,
            features={"layer1": np.zeros((10, 10)), "layer2": np.zeros((5, 5))},
        )
        features_b = ModelFeatures(
            model_id="b",
            feature_type=FeatureType.ARCHITECTURE,
            features={"layer1": np.zeros((10, 10)), "layer2": np.zeros((5, 5))},
        )

        similarity = comparator._compute_structural_similarity(features_a, features_b)
        assert similarity == 1.0  # Same structure

        # Different structure
        features_b.features["layer1"] = np.zeros((20, 20))
        similarity = comparator._compute_structural_similarity(features_a, features_b)
        assert similarity < 1.0  # Different structure

    def test_comparison_caching(self):
        """Test comparison result caching."""
        comparator = ModelComparator()
        model_a = DummyModel()
        model_b = DummyModel()

        # First comparison
        comparison1 = comparator.compare_models(model_a, model_b, "model_a", "model_b")

        # Second comparison (should use cache)
        comparison2 = comparator.compare_models(model_a, model_b, "model_a", "model_b")

        assert comparison1.model_a_id == comparison2.model_a_id
        assert len(comparator.comparison_cache) == 1


class TestFeatureVisualizer:
    """Test feature visualization functionality."""

    def test_visualizer_initialization(self):
        """Test feature visualizer initialization."""
        visualizer = FeatureVisualizer()

        assert len(visualizer.visualization_cache) == 0

    def test_visualize_features(self):
        """Test feature visualization."""
        visualizer = FeatureVisualizer()

        features = ModelFeatures(
            model_id="test_model",
            feature_type=FeatureType.WEIGHTS,
            features={"weights": np.random.randn(100), "biases": np.random.randn(10)},
        )

        viz_data = visualizer.visualize_features(features)

        assert viz_data["model_id"] == "test_model"
        assert "feature_plots" in viz_data
        assert "statistics" in viz_data
        assert "dimensions" in viz_data
        assert "weights" in viz_data["feature_plots"]
        assert "biases" in viz_data["feature_plots"]

    def test_visualize_comparison(self):
        """Test comparison visualization."""
        visualizer = FeatureVisualizer()

        comparison = ModelComparison(
            model_a_id="model_a",
            model_b_id="model_b",
            similarity_scores={"cosine": 0.85, "euclidean": 0.72, "structural": 0.90},
            feature_differences={
                "layer1": np.random.randn(10),
                "layer2": np.random.randn(5),
            },
            structural_similarity=0.90,
            semantic_similarity=0.80,
        )

        viz_data = visualizer.visualize_comparison(comparison)

        assert viz_data["model_a"] == "model_a"
        assert viz_data["model_b"] == "model_b"
        assert "similarity_chart" in viz_data
        assert "difference_heatmap" in viz_data
        assert "summary" in viz_data

        # Check summary
        summary = viz_data["summary"]
        assert "overall_similarity" in summary
        assert "most_similar" in summary
        assert "least_similar" in summary

    def test_create_feature_plot(self):
        """Test feature plot creation."""
        visualizer = FeatureVisualizer()

        feature_array = np.random.randn(1000)
        plot_data = visualizer._create_feature_plot("test_feature", feature_array)

        assert plot_data["type"] == "histogram"
        assert "data" in plot_data
        assert "metadata" in plot_data
        assert "values" in plot_data["data"]
        assert "bins" in plot_data["data"]
        assert "mean" in plot_data["metadata"]
        assert "std" in plot_data["metadata"]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_extract_and_save_features(self):
        """Test extracting and saving features."""
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "features.pkl"

            features = extract_and_save_features(model, "test_model", str(save_path))

            assert isinstance(features, ModelFeatures)
            assert save_path.exists()

    def test_load_features(self):
        """Test loading features from disk."""
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "features.pkl"

            # Save features
            original = extract_and_save_features(model, "test_model", str(save_path))

            # Load features
            loaded = load_features(str(save_path))

            assert loaded.model_id == original.model_id
            assert loaded.feature_type == original.feature_type

    def test_compare_model_files(self):
        """Test comparing model files."""
        comparison = compare_model_files(
            "path/to/model_a.pt",
            "path/to/model_b.pt",
            [SimilarityMetric.COSINE, SimilarityMetric.STRUCTURAL],
        )

        assert isinstance(comparison, ModelComparison)
        assert comparison.model_a_id == "model_a"
        assert comparison.model_b_id == "model_b"
        assert "cosine" in comparison.similarity_scores
        assert "structural" in comparison.similarity_scores


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def test_end_to_end_feature_extraction(self):
        """Test complete feature extraction pipeline."""
        # Configure extraction
        config = FeatureExtractionConfig(
            feature_types=[
                FeatureType.ARCHITECTURE,
                FeatureType.WEIGHTS,
                FeatureType.STATISTICS,
            ],
            extraction_method=ExtractionMethod.POOLING,
            normalize=True,
        )

        # Extract features
        extractor = FeatureExtractor(config)
        model = DummyModel()
        features = extractor.extract_features(model, "pipeline_model")

        # Verify features
        assert len(features.features) > 0
        assert features.extraction_method == ExtractionMethod.POOLING

        # Get statistics
        stats = features.get_feature_statistics()
        assert len(stats) > 0

        # Visualize
        visualizer = FeatureVisualizer()
        viz_data = visualizer.visualize_features(features)
        assert "feature_plots" in viz_data

    def test_model_comparison_pipeline(self):
        """Test complete model comparison pipeline."""
        # Create models
        model_a = DummyModel(depth=3, width=100)
        model_b = DummyModel(depth=4, width=150)

        # Compare models
        comparator = ModelComparator()
        comparison = comparator.compare_models(
            model_a,
            model_b,
            "model_a",
            "model_b",
            metrics=[
                SimilarityMetric.COSINE,
                SimilarityMetric.STRUCTURAL,
                SimilarityMetric.SEMANTIC,
            ],
        )

        # Verify comparison
        assert len(comparison.similarity_scores) == 3
        assert comparison.structural_similarity >= 0
        assert comparison.semantic_similarity >= 0

        # Visualize comparison
        visualizer = FeatureVisualizer()
        viz_data = visualizer.visualize_comparison(comparison)
        assert "similarity_chart" in viz_data
        assert "difference_heatmap" in viz_data

    def test_feature_extraction_with_different_models(self):
        """Test feature extraction with various model types."""
        extractor = FeatureExtractor()

        # Test with different model configurations
        models = [
            DummyModel(depth=2, width=50),
            DummyModel(depth=5, width=200),
            DummyModel(depth=10, width=500),
        ]

        extracted_features = []
        for i, model in enumerate(models):
            features = extractor.extract_features(model, f"model_{i}")
            extracted_features.append(features)

        # Compare all models
        comparator = ModelComparator()
        comparisons = []

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                comparison = comparator.compare_models(
                    models[i], models[j], f"model_{i}", f"model_{j}"
                )
                comparisons.append(comparison)

        assert len(comparisons) == 3  # 3 choose 2

        # Verify comparisons show differences
        for comparison in comparisons:
            assert comparison.structural_similarity < 1.0  # Models are different


if __name__ == "__main__":
    # Run feature extraction validation
    print("=== Testing Model Feature Extraction Framework ===")

    # Test feature extraction
    print("Testing feature extraction...")
    extractor = FeatureExtractor()
    model = DummyModel()
    features = extractor.extract_features(model, "test_model")
    print(f"OK Extracted {len(features.features)} feature types")

    # Test model comparison
    print("Testing model comparison...")
    comparator = ModelComparator()
    model_a = DummyModel(depth=3)
    model_b = DummyModel(depth=4)
    comparison = comparator.compare_models(model_a, model_b, "model_a", "model_b")
    print(f"OK Similarity scores: {list(comparison.similarity_scores.keys())}")

    # Test feature visualization
    print("Testing feature visualization...")
    visualizer = FeatureVisualizer()
    viz_data = visualizer.visualize_features(features)
    print(f"OK Visualization data: {list(viz_data.keys())}")

    # Test feature statistics
    print("Testing feature statistics...")
    stats = features.get_feature_statistics()
    print(f"OK Computed statistics for {len(stats)} features")

    # Test extraction methods
    print("Testing extraction methods...")
    for method in ExtractionMethod:
        config = FeatureExtractionConfig(extraction_method=method)
        extractor = FeatureExtractor(config)
        features = extractor.extract_features(model, f"{method.value}_model")
        print(f"OK {method.value}: {len(features.features)} features")

    print("=== Model feature extraction framework validation completed ===")
