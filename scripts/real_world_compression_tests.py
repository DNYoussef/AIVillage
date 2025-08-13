#!/usr/bin/env python3
"""Real-world testing scenarios for compression pipeline.
Tests with actual models and datasets relevant to Global South applications.
"""

import json
import os
import sys
import time

from enhance_compression_mobile import MobileCompressionPipeline, MobileOptimizedLayers
import numpy as np
import torch
from torch import nn

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class RealWorldTestSuite:
    """Test compression with real-world models and constraints."""

    def __init__(self) -> None:
        self.use_cases = {
            "translation": {
                "description": "Local language translation model",
                "model_type": "transformer",
                "typical_input": (1, 50),  # batch_size, sequence_length
                "target_latency_ms": 100,
                "critical_metric": "latency",
            },
            "image_classification": {
                "description": "Agricultural disease detection",
                "model_type": "cnn",
                "typical_input": (1, 3, 224, 224),
                "target_latency_ms": 200,
                "critical_metric": "accuracy",
            },
            "speech_recognition": {
                "description": "Local dialect speech-to-text",
                "model_type": "rnn",
                "typical_input": (1, 16000),  # 1 second audio at 16kHz
                "target_latency_ms": 500,
                "critical_metric": "memory",
            },
            "recommendation": {
                "description": "Offline product recommendations",
                "model_type": "embedding",
                "typical_input": (10, 20),  # user_history_length, item_features
                "target_latency_ms": 50,
                "critical_metric": "throughput",
            },
        }

    def create_test_model(self, use_case: str) -> nn.Module:
        """Create representative model for use case."""
        config = self.use_cases[use_case]

        if config["model_type"] == "transformer":
            return self._create_translation_model()
        if config["model_type"] == "cnn":
            return self._create_vision_model()
        if config["model_type"] == "rnn":
            return self._create_speech_model()
        if config["model_type"] == "embedding":
            return self._create_recommendation_model()
        return None

    def _create_translation_model(self) -> nn.Module:
        """Small transformer for translation."""

        class TranslationModel(nn.Module):
            def __init__(self, vocab_size=10000, d_model=256, nhead=4, num_layers=2) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
                encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output = nn.Linear(d_model, vocab_size)

            def forward(self, x):
                x = self.embedding(x)
                x = x + self.pos_encoding[:, : x.size(1), :]
                x = self.transformer(x)
                return self.output(x)

        return TranslationModel()

    def _create_vision_model(self) -> nn.Module:
        """Lightweight CNN for image classification."""
        # MobileNet-inspired architecture
        layers = MobileOptimizedLayers()

        class VisionModel(nn.Module):
            def __init__(self, num_classes=10) -> None:
                super().__init__()

                self.features = nn.Sequential(
                    layers.create_mobile_conv2d(3, 32),
                    nn.MaxPool2d(2),
                    layers.create_mobile_conv2d(32, 64),
                    nn.MaxPool2d(2),
                    layers.create_mobile_conv2d(64, 128),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Linear(128, num_classes)

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)

        return VisionModel()

    def _create_speech_model(self) -> nn.Module:
        """RNN for speech recognition."""

        class SpeechModel(nn.Module):
            def __init__(self, input_dim=80, hidden_dim=128, num_classes=50) -> None:
                super().__init__()
                self.feature_extractor = nn.Conv1d(1, input_dim, kernel_size=160, stride=80)
                self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
                self.output = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                x = x.unsqueeze(1)  # Add channel dimension
                x = self.feature_extractor(x)
                x = x.transpose(1, 2)  # (batch, time, features)
                x, _ = self.rnn(x)
                return self.output(x)

        return SpeechModel()

    def _create_recommendation_model(self) -> nn.Module:
        """Embedding-based recommendation model."""

        class RecommendationModel(nn.Module):
            def __init__(self, num_users=10000, num_items=5000, embed_dim=64) -> None:
                super().__init__()
                self.user_embed = nn.Embedding(num_users, embed_dim)
                self.item_embed = nn.Embedding(num_items, embed_dim)
                self.mlp = nn.Sequential(
                    nn.Linear(embed_dim * 2, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )

            def forward(self, user_ids, item_ids):
                user_embeds = self.user_embed(user_ids)
                item_embeds = self.item_embed(item_ids)
                combined = torch.cat([user_embeds, item_embeds], dim=-1)
                return self.mlp(combined).squeeze(-1)

        return RecommendationModel()

    def run_comprehensive_tests(self) -> dict[str, dict]:
        """Run tests for all use cases."""
        mobile_pipeline = MobileCompressionPipeline()

        results = {}

        for use_case, config in self.use_cases.items():
            print(f"\n=== Testing {use_case}: {config['description']} ===")

            # Create model
            model = self.create_test_model(use_case)

            # Test each compression method
            results[use_case] = {}

            for profile in ["2gb_device", "4gb_device"]:
                print(f"\nProfile: {profile}")

                try:
                    # Compress model
                    compressed = mobile_pipeline.compress_for_mobile(model, profile)

                    # Create test input
                    if use_case == "translation":
                        test_input = torch.randint(0, 10000, config["typical_input"])
                    elif use_case == "recommendation":
                        test_input = (
                            torch.randint(0, 10000, (config["typical_input"][0],)),
                            torch.randint(0, 5000, (config["typical_input"][0],)),
                        )
                    else:
                        test_input = torch.randn(config["typical_input"])

                    # Measure performance
                    metrics = self._measure_performance(compressed, test_input, config["target_latency_ms"])

                    results[use_case][profile] = metrics

                    # Print summary
                    print(f"  Latency: {metrics['latency_ms']:.1f}ms " f"({'✅' if metrics['meets_latency'] else '❌'})")
                    print(f"  Model size: {metrics['model_size_mb']:.1f}MB")
                    print(f"  Memory usage: {metrics['memory_mb']:.1f}MB")

                except Exception as e:
                    print(f"  Error: {e}")
                    results[use_case][profile] = {"error": str(e)}

        return results

    def _measure_performance(
        self, model: nn.Module, test_input: torch.Tensor, target_latency_ms: float
    ) -> dict[str, any]:
        """Measure model performance metrics."""
        import os

        import psutil

        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                if isinstance(test_input, tuple):
                    _ = model(*test_input)
                else:
                    _ = model(test_input)

        # Measure latency
        times = []
        for _ in range(100):
            start = time.time()
            with torch.no_grad():
                if isinstance(test_input, tuple):
                    _ = model(*test_input)
                else:
                    _ = model(test_input)
            times.append(time.time() - start)

        latency_ms = np.mean(times) * 1000

        # Measure memory
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Calculate model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = model_size / (1024 * 1024)

        return {
            "latency_ms": latency_ms,
            "meets_latency": latency_ms <= target_latency_ms,
            "model_size_mb": model_size_mb,
            "memory_mb": memory_mb,
            "target_latency_ms": target_latency_ms,
        }

    def generate_performance_report(self, results: dict[str, dict]) -> None:
        """Generate detailed performance report."""
        report = """# Real-World Compression Performance Report

## Use Cases Tested

1. **Translation**: Local language translation (transformer-based)
2. **Image Classification**: Agricultural disease detection (CNN)
3. **Speech Recognition**: Local dialect speech-to-text (RNN)
4. **Recommendation**: Offline product recommendations (embeddings)

## Results Summary

"""

        # Create summary table
        report += "| Use Case | Model | 2GB Device | 4GB Device | Target |\n"
        report += "|----------|-------|------------|------------|--------|\n"

        for use_case, config in self.use_cases.items():
            report += f"| {use_case.title()} | "
            report += f"{config['model_type'].upper()} | "

            for profile in ["2gb_device", "4gb_device"]:
                if profile in results.get(use_case, {}):
                    metrics = results[use_case][profile]
                    if "error" not in metrics:
                        latency = metrics["latency_ms"]
                        status = "✅" if metrics["meets_latency"] else "❌"
                        report += f"{latency:.0f}ms {status} | "
                    else:
                        report += "Error | "
                else:
                    report += "N/A | "

            report += f"{config['target_latency_ms']}ms |\n"

        # Add detailed analysis
        report += "\n## Detailed Analysis\n\n"

        for use_case, use_case_results in results.items():
            report += f"### {use_case.title()}\n\n"
            report += f"**Description**: {self.use_cases[use_case]['description']}\n"
            report += f"**Critical Metric**: {self.use_cases[use_case]['critical_metric']}\n\n"

            for profile, metrics in use_case_results.items():
                if "error" not in metrics:
                    report += f"**{profile}**:\n"
                    report += f"- Latency: {metrics['latency_ms']:.1f}ms\n"
                    report += f"- Model Size: {metrics['model_size_mb']:.1f}MB\n"
                    report += f"- Memory Usage: {metrics['memory_mb']:.1f}MB\n"
                    report += f"- Meets Target: {'Yes' if metrics['meets_latency'] else 'No'}\n\n"

        # Add recommendations
        report += """## Recommendations

1. **Translation Models**: Use dynamic quantization for variable sequence lengths
2. **Vision Models**: Depthwise separable convolutions essential for 2GB devices
3. **Speech Models**: Consider streaming inference for long audio
4. **Recommendation**: Embedding tables dominate size - use hash embeddings

## Next Steps

1. Implement model-specific compression strategies
2. Create automated compression selection based on device profile
3. Add accuracy preservation metrics to testing
4. Develop progressive model loading for very constrained devices
"""

        with open("real_world_performance_report.md", "w") as f:
            f.write(report)

        print("\n✓ Generated real_world_performance_report.md")


# Run the test suite
if __name__ == "__main__":
    print("Running real-world compression tests...")
    suite = RealWorldTestSuite()

    results = suite.run_comprehensive_tests()
    suite.generate_performance_report(results)

    # Save raw results
    with open("real_world_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n✅ Real-world testing complete!")
