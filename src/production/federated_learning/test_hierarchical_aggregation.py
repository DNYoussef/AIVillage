#!/usr/bin/env python3
"""Test hierarchical aggregation for federated learning efficiency.

This test constructs a ``DistributedFederatedLearning`` instance populated with
mock participants containing pre-defined gradients.  The hierarchical
aggregation routine is executed and the resulting cluster statistics and
bandwidth savings are recorded.  Command line flags expose the participant
count, enable differential‑privacy checks (for parity with other tests) and
allow efficiency metrics to be reported.  Output is provided in JSON format.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
import time
import types
from typing import Any

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

    async def broadcast_to_peers(
        self, _msg_type: str, _message: dict[str, Any]
    ) -> bool:
        return True

    async def send_to_peer(self, _peer_id: str, _message: dict[str, Any]) -> bool:
        return True

    def get_suitable_evolution_peers(
        self, min_count: int = 1
    ) -> list[PeerCapabilities]:
        return []


async def run_test(
    participants: int, dp_check: bool, efficiency: bool
) -> dict[str, Any]:
    """Execute hierarchical aggregation and return statistics."""
    config = FederatedLearningConfig(enable_hierarchical_aggregation=True)
    fl = DistributedFederatedLearning(MockP2PNode(), config=config)

    # Populate participants with simple gradients
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
        participant.gradients = {"w": torch.ones(1) * (i + 1)}
        fl.available_participants[cap.device_id] = participant
        fl.participant_pool.add(cap.device_id)

    start = time.time()
    aggregation = await fl.implement_hierarchical_aggregation()
    runtime = time.time() - start

    result = {
        "participants": participants,
        "clusters_created": aggregation.get("clusters_created", 0),
        "bandwidth_savings": aggregation.get("bandwidth_savings", 0.0),
    }

    if dp_check:
        # DP is unrelated here but the flag is accepted for interface uniformity.
        gradients = {"w": torch.ones(1)}
        noisy = fl._add_differential_privacy_noise(
            gradients,
            fl.config.differential_privacy_epsilon,
            fl.config.differential_privacy_delta,
        )
        result["dp_noise_added"] = not torch.equal(gradients["w"], noisy["w"])

    if efficiency:
        result["runtime_sec"] = runtime

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--participants",
        type=int,
        default=6,
        help="number of mock participants to simulate",
    )
    parser.add_argument(
        "--dp-check", action="store_true", help="run a differential privacy noise check"
    )
    parser.add_argument(
        "--efficiency", action="store_true", help="report runtime metrics"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="file to write JSON results to; '-' for stdout",
    )
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
