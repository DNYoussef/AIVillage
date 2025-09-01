#!/usr/bin/env python3
"""Benchmark compression performance."""

import importlib.util
import io
from pathlib import Path
import sys
import time

import torch
from torch import nn

src_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_root))
spec = importlib.util.spec_from_file_location(
    "simple_quantizer", src_root / "core" / "compression" / "simple_quantizer.py"
)
simple_quantizer = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(simple_quantizer)
SimpleQuantizer = simple_quantizer.SimpleQuantizer


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


def benchmark_compression(precision: str = "float32") -> None:
    """Benchmark compression optionally using reduced precision models."""
    quantizer = SimpleQuantizer()
    models = create_test_models()

    if precision == "float16":
        for model in models.values():
            model.half()

    print(f"Compression Benchmark Results ({precision})")
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
                f"{name:<10} {original_size / 1024:.1f}KB "
                f"{compressed_size / 1024:.1f}KB {ratio:.2f}x {elapsed:.2f}s"
            )
        except Exception as e:
            print(f"{name:<10} FAILED: {e}")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_compression("float32")
    benchmark_compression("float16")
