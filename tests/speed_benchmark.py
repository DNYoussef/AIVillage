#!/usr/bin/env python3
"""Quick speed benchmark of compression system."""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path("src").resolve()))

from src.agent_forge.compression.bitnet import BITNETCompressor
from src.agent_forge.compression.vptq import VPTQCompressor


def speed_benchmark():
    """Benchmark compression speed on different tensor sizes."""
    print("COMPRESSION SPEED BENCHMARK")
    print("=" * 40)

    sizes = [(100, 100, "Small"), (512, 512, "Medium"), (1024, 1024, "Large")]

    compressors = {"BitNet": BITNETCompressor(), "VPTQ": VPTQCompressor(bits=2)}

    for rows, cols, size_name in sizes:
        tensor = torch.randn(rows, cols)
        params = tensor.numel()
        mb = params * 4 / 1024 / 1024

        print(f"\n{size_name} tensor: {rows}x{cols} = {params:,} params ({mb:.1f}MB)")

        for comp_name, compressor in compressors.items():
            # Warm up
            compressor.compress(tensor)

            # Time compression
            start = time.time()
            compressed = compressor.compress(tensor)
            compress_time = time.time() - start

            # Time decompression
            start = time.time()
            compressor.decompress(compressed)
            decompress_time = time.time() - start

            # Calculate throughput
            compress_mbps = mb / max(compress_time, 0.001)
            decompress_mbps = mb / max(decompress_time, 0.001)

            print(
                f"  {comp_name:8}: {compress_time:6.3f}s compress ({compress_mbps:5.1f}MB/s), "
                f"{decompress_time:6.3f}s decompress ({decompress_mbps:5.1f}MB/s)"
            )


if __name__ == "__main__":
    speed_benchmark()
