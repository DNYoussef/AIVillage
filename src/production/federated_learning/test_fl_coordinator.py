#!/usr/bin/env python3
"""Test Distributed Federated Learning coordinator functionality.

This script simulates a basic participant selection flow of the
``DistributedFederatedLearning`` class. It provides command line flags for
specifying the number of mock participants, enabling differential privacy
checks and reporting efficiency metrics. Results are output as JSON either to
stdout or to a specified file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
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
    FederatedTrainingRound,
    PeerCapabilities,
    TrainingParticipant,
)

logger = logging.getLogger(__name__)


@dataclass
class MockP2PNode:
    """Minimal stand-in for :class:`P2PNode` used in tests."""

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
    """Execute participant selection and return metrics."""
    config = FederatedLearningConfig(
        min_participants_per_round=1,
        max_participants_per_round=participants,
    )
    fl = DistributedFederatedLearning(MockP2PNode(), config=config)

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

    training_round = FederatedTrainingRound(
        round_id="round_1",
        round_number=1,
        participants=[],
        global_model_state={},
    )

    start = time.time()
    selected = await fl._select_participants_for_round(training_round)
    runtime = time.time() - start

    result = {
        "requested_participants": participants,
        "selected_participants": len(selected),
        "selection_success": len(selected) >= 1,
    }

    if dp_check:
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
        default=5,
        help="number of mock participants to simulate",
    )
    parser.add_argument(
        "--dp-check",
        action="store_true",
        help="verify differential privacy noise application",
    )
    parser.add_argument(
        "--efficiency",
        action="store_true",
        help="report runtime metrics for the selection phase",
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
