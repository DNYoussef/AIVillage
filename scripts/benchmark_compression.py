#!/usr/bin/env python3
"""Benchmark compression performance."""
from pathlib import Path
import sys
import time

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.compression import SimpleQuantizer


def create_test_models():
    models = {
        "tiny": nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ),
        "small": nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ),
        "medium": nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ),
        "large": nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ),
    }
    return models


def benchmark_compression() -> None:
    quantizer = SimpleQuantizer()
    models = create_test_models()

    print("Compression Benchmark Results")
    print("=" * 60)
    print(f"{'Model':<10} {'Original':<12} {'Compressed':<12} {'Ratio':<8} {'Time':<8}")
    print("-" * 60)

    for name, model in models.items():
        buffer = io.BytesIO()
        torch.save(model, buffer)
        original_size = buffer.tell()
        start = time.time()
        try:
            compressed = quantizer.quantize_model(model)
            elapsed = time.time() - start
            compressed_size = len(compressed)
            ratio = original_size / compressed_size
            print(
                f"{name:<10} {original_size/1024:.1f}KB {compressed_size/1024:.1f}KB {ratio:.2f}x {elapsed:.2f}s"
            )
        except Exception as e:
            print(f"{name:<10} FAILED: {e}")
    print("=" * 60)


if __name__ == "__main__":
    import io

    benchmark_compression()
