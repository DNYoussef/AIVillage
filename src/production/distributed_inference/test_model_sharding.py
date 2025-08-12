import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import time
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

spec = importlib.util.spec_from_file_location(
    f"{package_name}.model_sharding_engine", package_dir / "model_sharding_engine.py"
)
mse = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mse
spec.loader.exec_module(mse)

ModelShardingEngine = mse.ModelShardingEngine
ShardingStrategy = mse.ShardingStrategy
DeviceProfile = mse.DeviceProfile
ShardingPlan = mse.ShardingPlan

from src.core.p2p import PeerCapabilities


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
    """Lightweight testing subclass of ModelShardingEngine."""

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
    ) -> ShardingPlan:
        return await self._create_sequential_plan(model_analysis, device_profiles)

    async def _activate_sharding_plan(self, plan: ShardingPlan) -> None:
        self.active_shards = {s.shard_id: s for s in plan.shards}
        self.device_assignments = {}
        for shard in plan.shards:
            self.device_assignments.setdefault(shard.device_id, []).append(shard.shard_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test model sharding")
    parser.add_argument("--device-count", type=int, default=2)
    parser.add_argument("--constraint", type=str, default="none")
    parser.add_argument("--simulate-failures", action="store_true")
    args = parser.parse_args()

    engine = TestModelShardingEngine(args.device_count, args.constraint)

    start = time.time()
    plan = asyncio.run(engine.shard_model("dummy-model", strategy=ShardingStrategy.HYBRID))
    duration = time.time() - start

    report = {
        "test": "model_sharding",
        "device_count": args.device_count,
        "constraint": args.constraint,
        "duration_sec": duration,
        "total_shards": plan.total_shards,
        "simulate_failures": args.simulate_failures,
    }
    with open("sharding_performance_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Report written to sharding_performance_report.json")


if __name__ == "__main__":
    main()
