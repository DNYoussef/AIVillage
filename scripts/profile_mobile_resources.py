#!/usr/bin/env python3
"""Profile memory and CPU usage on simulated mobile devices."""

from __future__ import annotations

from mobile_device_simulator import DEVICE_PROFILES, MobileSimulator
import psutil
import torch
from torch import nn


def profile_device(profile_key: str) -> dict[str, float | str]:
    """Run inference on a tiny model and capture resource usage."""
    profile = DEVICE_PROFILES[profile_key]
    simulator = MobileSimulator(profile)
    model = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    input_tensor = torch.randn(1, 256)

    with simulator.simulate():
        cpu_before = psutil.cpu_percent(interval=None)
        stats = simulator.measure_inference(model, input_tensor)
        cpu_after = psutil.cpu_percent(interval=None)

    stats["cpu_percent"] = cpu_after if cpu_after else cpu_before
    return stats


def main() -> None:
    for key in ["budget_2gb", "redmi_note_10"]:
        stats = profile_device(key)
        print(
            f"{stats['device_profile']}: CPU {stats['cpu_percent']:.1f}% | "
            f"Peak Memory {stats['memory_peak_mb']:.1f}MB"
        )


if __name__ == "__main__":
    main()
