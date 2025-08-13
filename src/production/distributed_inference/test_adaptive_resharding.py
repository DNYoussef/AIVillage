import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import types

# Stub external dependencies required by the sharding modules
sys.modules.setdefault("wandb", types.ModuleType("wandb"))
transformers_stub = types.ModuleType("transformers")


class _Dummy:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __init__(self) -> None:
        self.config = types.SimpleNamespace()


transformers_stub.AutoTokenizer = _Dummy
transformers_stub.AutoModelForCausalLM = _Dummy
sys.modules.setdefault("transformers", transformers_stub)
torch_stub = types.ModuleType("torch")
torch_stub.nn = types.SimpleNamespace()
sys.modules.setdefault("torch", torch_stub)

repo_root = Path(__file__).resolve().parents[3]
sys.path.append(str(repo_root))

package_dir = Path(__file__).resolve().parent
package_name = "src.production.distributed_inference"
pkg = types.ModuleType(package_name)
pkg.__path__ = [str(package_dir)]
sys.modules[package_name] = pkg

# Load model_sharding_engine
spec_engine = importlib.util.spec_from_file_location(
    f"{package_name}.model_sharding_engine", package_dir / "model_sharding_engine.py"
)
engine_module = importlib.util.module_from_spec(spec_engine)
sys.modules[spec_engine.name] = engine_module
spec_engine.loader.exec_module(engine_module)

ModelShardingEngine = engine_module.ModelShardingEngine
ShardingStrategy = engine_module.ShardingStrategy
DeviceProfile = engine_module.DeviceProfile

# Load adaptive_resharding
spec_reshard = importlib.util.spec_from_file_location(
    f"{package_name}.adaptive_resharding", package_dir / "adaptive_resharding.py"
)
reshard_module = importlib.util.module_from_spec(spec_reshard)
sys.modules[spec_reshard.name] = reshard_module
spec_reshard.loader.exec_module(reshard_module)

AdaptiveReshardingManager = reshard_module.AdaptiveReshardingManager
ReshardingReason = reshard_module.ReshardingReason
ReshardingStrategy = reshard_module.ReshardingStrategy
ReshardingConfig = reshard_module.ReshardingConfig

from src.production.communications.p2p.p2p_node import PeerCapabilities


class DummyP2PNode:
    """Minimal P2P node for testing."""

    def __init__(self) -> None:
        self.node_id = "local"
        self.peer_registry: dict[str, PeerCapabilities] = {}
        self.local_capabilities = PeerCapabilities(
            device_id="local",
            cpu_cores=4,
            ram_mb=8192,
            trust_score=1.0,
            evolution_capacity=1.0,
        )


class DummyResourceMonitor:
    pass


class DummyDeviceProfiler:
    current_snapshot = None


class TestModelShardingEngine(ModelShardingEngine):
    """Testing subclass used with adaptive resharding."""

    def __init__(self, device_count: int, constraint: str) -> None:
        super().__init__(DummyP2PNode(), DummyResourceMonitor(), DummyDeviceProfiler())
        self.device_count = device_count
        self.constraint = constraint

    async def _analyze_model(self, model_path: str) -> dict:
        return {
            "model_path": model_path,
            "num_layers": 8,
            "layer_memory_mb": 10.0,
            "layer_compute_score": 1.0,
        }

    async def _get_device_profiles(self, target_devices: list[str] | None = None):
        profiles: list[DeviceProfile] = []
        for i in range(self.device_count):
            ram = 1024 if self.constraint != "memory" else 256
            capabilities = PeerCapabilities(
                device_id=f"device_{i}",
                cpu_cores=4,
                ram_mb=ram,
                trust_score=0.9,
                evolution_capacity=0.8,
            )
            profiles.append(
                DeviceProfile(
                    device_id=f"device_{i}",
                    capabilities=capabilities,
                    available_memory_mb=ram * 0.75,
                    compute_score=1.0,
                    network_latency_ms=10.0,
                    reliability_score=0.9,
                )
            )
        return profiles

    async def _create_sharding_plan(
        self,
        model_analysis: dict,
        device_profiles: list[DeviceProfile],
        strategy: ShardingStrategy,
    ):
        return await self._create_sequential_plan(model_analysis, device_profiles)

    async def _activate_sharding_plan(self, plan) -> None:
        self.active_shards = {s.shard_id: s for s in plan.shards}
        self.device_assignments = {}
        for shard in plan.shards:
            self.device_assignments.setdefault(shard.device_id, []).append(
                shard.shard_id
            )


async def run_test(device_count: int, constraint: str, simulate_failures: bool) -> None:
    engine = TestModelShardingEngine(device_count, constraint)
    await engine.shard_model("dummy-model", strategy=ShardingStrategy.HYBRID)

    config = ReshardingConfig(min_resharding_interval_seconds=0)
    manager = AdaptiveReshardingManager(engine, engine.p2p_node, config=config)

    if simulate_failures:
        await manager._handle_device_left("device_0")
    else:
        await manager.trigger_resharding(
            ReshardingReason.MANUAL_TRIGGER,
            strategy=ReshardingStrategy.MINIMAL_DISRUPTION,
        )

    report = {
        "test": "adaptive_resharding",
        "resharding_events": manager.stats["total_resharding_events"],
        "successful_resharding": manager.stats["successful_resharding"],
    }
    with open("sharding_performance_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Report written to sharding_performance_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test adaptive resharding")
    parser.add_argument("--device-count", type=int, default=2)
    parser.add_argument("--constraint", type=str, default="none")
    parser.add_argument("--simulate-failures", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_test(args.device_count, args.constraint, args.simulate_failures))


if __name__ == "__main__":
    main()
