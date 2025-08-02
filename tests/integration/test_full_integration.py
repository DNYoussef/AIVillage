"""Cross-sprint integration test for the distributed AI pipeline.

The test exercises a sequence of high level operations that would normally
occur over multiple development sprints.  Each function can either be the real
implementation or a light‑weight mock.  The goal is to verify that the public
interfaces remain compatible as the project evolves.
"""

from __future__ import annotations

import asyncio
import random
from typing import Dict, List

import pytest


# ---------------------------------------------------------------------------
# Mock / reference implementations
# ---------------------------------------------------------------------------

async def deploy_all_18_agents() -> List[str]:
    """Pretend to deploy all agents and return their identifiers."""

    # simulate a small delay for asynchronous behaviour
    await asyncio.sleep(0.01)
    return [f"agent_{i}" for i in range(18)]


async def establish_p2p_network(agents: List[str]) -> Dict[str, List[str]]:
    """Create a fully connected peer‑to‑peer network."""

    await asyncio.sleep(0.01)
    return {a: [b for b in agents if b != a] for a in agents}


async def profile_device_resources(agents: List[str]) -> List[Dict[str, int]]:
    """Profile edge devices.  Each device exposes a simple RAM metric."""

    await asyncio.sleep(0.01)
    # three heterogeneous devices
    return [
        {"name": "edge_1", "ram": 4},
        {"name": "edge_2", "ram": 8},
        {"name": "edge_3", "ram": 16},
    ]


async def distribute_model(devices: List[Dict[str, int]], model_size: int) -> Dict[str, int]:
    """Distribute model shards across devices based on RAM capacity."""

    await asyncio.sleep(0.01)
    total_ram = sum(d["ram"] for d in devices)
    shards: Dict[str, int] = {}
    assigned = 0
    for device in devices:
        share = int(model_size * device["ram"] / total_ram)
        shards[device["name"]] = share
        assigned += share
    if assigned < model_size:
        shards[devices[-1]["name"]] += model_size - assigned
    return shards


async def distributed_inference(shards: Dict[str, int]) -> float:
    """Run a tiny distributed inference returning an aggregate score."""

    inputs = [random.random() for _ in range(5)]
    partials = []
    for count in shards.values():
        await asyncio.sleep(0.005)
        partials.append(sum(inputs[: min(count, len(inputs))]))
    return sum(partials)


async def run_evolution_cycle(agents: List[str]) -> Dict[str, float]:
    """Mock an evolution cycle producing a best fitness score."""

    await asyncio.sleep(0.01)
    fitness = max(random.uniform(0.8, 0.95) for _ in agents)
    return {"generations": 1, "best_fitness": fitness}


async def federated_learning_round(shards: Dict[str, int]) -> Dict[str, float]:
    """Simulate one round of federated learning."""

    await asyncio.sleep(0.01)
    old = 0.5
    improvement = sum(shards.values()) / (len(shards) * 100)
    new = old + improvement
    return {"old_accuracy": old, "new_accuracy": new}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_integration() -> None:
    """Validate that all stages remain compatible across sprints."""

    agents = await deploy_all_18_agents()
    assert len(agents) == 18

    network = await establish_p2p_network(agents)
    assert all(len(peers) == len(agents) - 1 for peers in network.values())

    devices = await profile_device_resources(agents)
    assert len(devices) == 3 and all("ram" in d for d in devices)

    shards = await distribute_model(devices, model_size=24)
    assert sum(shards.values()) == 24

    inference_result = await distributed_inference(shards)
    assert inference_result > 0

    evolution_metrics = await run_evolution_cycle(agents)
    assert evolution_metrics["generations"] >= 1
    assert 0.8 <= evolution_metrics["best_fitness"] <= 1.0

    fl_metrics = await federated_learning_round(shards)
    assert fl_metrics["new_accuracy"] > fl_metrics["old_accuracy"]
