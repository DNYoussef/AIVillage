#!/usr/bin/env python3
"""Benchmark the advanced compression pipeline against a simple quantizer."""

from __future__ import annotations

from pathlib import Path
import sys
import time

from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.compression.advanced_pipeline import AdvancedCompressionPipeline
from core.compression.simple_quantizer import SimpleQuantizer


def create_layer(hidden: int) -> nn.Module:
    return nn.ModuleDict(
        {
            "attention": nn.Linear(hidden, hidden),
            "mlp_gate": nn.Linear(hidden, hidden * 4),
            "mlp_up": nn.Linear(hidden, hidden * 4),
            "mlp_down": nn.Linear(hidden * 4, hidden),
        }
    )


def benchmark() -> None:
    models = {
        "small": create_layer(512),
        "medium": create_layer(1024),
    }
    methods = {
        "SimpleQuantizer": SimpleQuantizer(),
        "AdvancedPipeline": AdvancedCompressionPipeline(),
    }
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        original_mb = params * 4 / 1024 / 1024
        print(f"\n{name.upper()} model ({params:,} params, {original_mb:.1f}MB)")
        for mname, comp in methods.items():
            start = time.time()
            if mname == "SimpleQuantizer":
                blob = comp.quantize_model(model)
            else:
                blob = comp.compress_model(model)
            t = time.time() - start
            compressed_mb = len(blob) / 1024 / 1024
            ratio = original_mb / compressed_mb
            print(f"  {mname}: {ratio:.1f}x in {t:.1f}s")


if __name__ == "__main__":
    benchmark()
