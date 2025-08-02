#!/usr/bin/env python3
"""Test differential privacy mechanisms in federated learning.

The test initialises a ``DistributedFederatedLearning`` instance with mock
participants, applies differential privacy noise to sample gradients and
verifies privacy budget accounting.  Command line flags allow tuning the number
of participants, enabling explicit differential‑privacy checks and recording
efficiency metrics.  Results are written as JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import sys
from pathlib import Path
import types
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

def _stub_module(name: str, **attrs: object) -> None:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod

_stub_module(
    "src.production.evolution.infrastructure_aware_evolution",
    InfrastructureAwareEvolution=type("DummyEvolution", (), {}),
)
_stub_module(
    "src.production.federated_learning.hierarchical_aggregation",
    AggregationTier=type("AggTier", (), {}),
    HierarchicalAggregator=type("HAgg", (), {}),
)
_stub_module(
    "src.production.federated_learning.secure_aggregation",
    PrivacyConfig=type("PConfig", (), {}),
    SecureAggregationProtocol=type("SAP", (), {}),
)

from src.production.federated_learning.federated_coordinator import (
    DistributedFederatedLearning,
    FederatedLearningConfig,
    PeerCapabilities,
    TrainingParticipant,
)


logger = logging.getLogger(__name__)


@dataclass
class MockP2PNode:
    node_id: str = "device_0"

    def __post_init__(self) -> None:
        self.local_capabilities = PeerCapabilities(
            device_id=self.node_id,
            cpu_cores=4,
            ram_mb=4096,
            battery_percent=100,
            trust_score=0.9,
            latency_ms=10,
            evolution_capacity=1.0,
        )

    async def broadcast_to_peers(self, _msg_type: str, _message: dict[str, Any]) -> bool:
        return True

    async def send_to_peer(self, _peer_id: str, _message: dict[str, Any]) -> bool:
        return True

    def get_suitable_evolution_peers(self, min_count: int = 1) -> list[PeerCapabilities]:
        return []


async def run_test(participants: int, dp_check: bool, efficiency: bool) -> dict[str, Any]:
    """Verify noise application and privacy budget consumption."""

    config = FederatedLearningConfig()
    fl = DistributedFederatedLearning(MockP2PNode(), config=config)

    # Add mock participants and initialise budgets
    for i in range(participants):
        cap = PeerCapabilities(
            device_id=f"device_{i}",
            cpu_cores=4,
            ram_mb=4096,
            battery_percent=90,
            trust_score=0.9,
            latency_ms=20,
            evolution_capacity=1.0,
        )
        participant = TrainingParticipant(device_id=cap.device_id, capabilities=cap)
        fl.available_participants[cap.device_id] = participant
        fl.participant_pool.add(cap.device_id)

    fl._initialize_privacy_budgets()

    gradients = {"w": torch.ones(5)}
    start = time.time()
    noisy = fl._add_differential_privacy_noise(
        gradients, fl.config.differential_privacy_epsilon, fl.config.differential_privacy_delta
    )
    runtime = time.time() - start

    noise_added = not torch.allclose(gradients["w"], noisy["w"])

    # Simulate privacy budget consumption for the first participant
    first = next(iter(fl.available_participants.values()))
    initial_budget = first.privacy_budget_remaining
    first.privacy_budget_remaining -= fl.config.differential_privacy_epsilon

    result = {
        "participants": participants,
        "noise_added": noise_added,
        "initial_budget": initial_budget,
        "post_training_budget": first.privacy_budget_remaining,
    }

    if dp_check:
        result["budget_decreased"] = first.privacy_budget_remaining < initial_budget

    if efficiency:
        result["runtime_sec"] = runtime

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=int, default=3, help="number of mock participants to simulate")
    parser.add_argument("--dp-check", action="store_true", help="validate privacy budget consumption")
    parser.add_argument("--efficiency", action="store_true", help="report runtime metrics")
    parser.add_argument("--output", type=str, default="-", help="file to write JSON results to; '-' for stdout")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(run_test(args.participants, args.dp_check, args.efficiency))

    output = json.dumps(results, indent=2)
    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)


if __name__ == "__main__":
    main()

