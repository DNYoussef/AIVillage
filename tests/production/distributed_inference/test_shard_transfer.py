import asyncio
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

# Stub heavy optional dependencies used during import of compression modules
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

click_stub = types.ModuleType("click")


def _decorator(*_args, **_kwargs):
    def wrapper(func):
        return func
    return wrapper


def _group(*_args, **_kwargs):
    def decorator(func):
        def wrapper(*a, **kw):
            return func(*a, **kw)

        def command(*c_args, **c_kwargs):
            return _decorator

        wrapper.command = command
        return wrapper

    return decorator


click_stub.group = _group
click_stub.command = _decorator
click_stub.option = _decorator
click_stub.argument = _decorator
sys.modules.setdefault("click", click_stub)

datasets_stub = types.ModuleType("datasets")
datasets_stub.load_dataset = lambda *args, **kwargs: None
sys.modules.setdefault("datasets", datasets_stub)

# Minimal transformers stub to satisfy imports
transformers_stub = types.ModuleType("transformers")


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 1)])
        self.config = types.SimpleNamespace()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - import stub
        return cls()


transformers_stub.AutoModelForCausalLM = _DummyModel
transformers_stub.AutoTokenizer = type(
    "Tok", (), {"from_pretrained": classmethod(lambda cls, *args, **kwargs: object())}
)
transformers_stub.Trainer = type("Trainer", (), {})
transformers_stub.TrainerCallback = type("TrainerCallback", (), {})
transformers_stub.TrainingArguments = type("TrainingArguments", (), {})
sys.modules.setdefault("transformers", transformers_stub)

from src.production.distributed_inference.compression_integration import (
    CompressedShard,
    DistributedCompressionManager,
)
from src.production.distributed_inference.model_sharding_engine import ModelShard


class DummyP2PNode:
    """Minimal P2P node capturing messages and files."""

    def __init__(self):
        self.node_id = "local"
        self.sent_messages: list[tuple[str, dict]] = []
        self.sent_files: list[tuple[str, str]] = []

    async def send_to_peer(self, peer_id: str, message: dict):
        self.sent_messages.append((peer_id, message))
        return True

    async def send_file(self, peer_id: str, file_path: str):
        self.sent_files.append((peer_id, file_path))
        return True


class DummyEngine:
    def __init__(self):
        self.p2p_node = DummyP2PNode()


class ToyModel(nn.Module):
    def __init__(self, num_layers: int = 5):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(num_layers)])

    def forward(self, x):  # pragma: no cover - not used
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.mark.asyncio
async def test_extract_shard_layers_saves_correct_layers(tmp_path):
    model = ToyModel()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.save(model.state_dict(), model_dir / "pytorch_model.bin")

    shard = ModelShard(
        shard_id="s1",
        device_id="local",
        layer_indices=[1, 3],
        parameters_count=0,
        memory_mb=0,
        compute_requirement=0,
    )

    manager = DistributedCompressionManager(DummyEngine())
    result_dir = await manager._extract_shard_layers(str(model_dir), shard, tmp_path)

    shard_state = torch.load(Path(result_dir) / "pytorch_model.bin")
    assert all(
        key.startswith("layers.1") or key.startswith("layers.3")
        for key in shard_state.keys()
    )
    # Each linear layer has weight and bias tensors
    assert len(shard_state) == 4


@pytest.mark.asyncio
async def test_send_compressed_shard_uses_p2p_transfer(tmp_path):
    engine = DummyEngine()
    manager = DistributedCompressionManager(engine)

    file_path = tmp_path / "compressed.bin"
    file_path.write_bytes(b"data")

    original = ModelShard(
        shard_id="s1",
        device_id="peer",
        layer_indices=[0],
        parameters_count=0,
        memory_mb=0,
        compute_requirement=0,
    )
    compressed = CompressedShard(
        shard_id="s1",
        original_shard=original,
        compressed_model_path=str(file_path),
        compression_ratio=2.0,
        compressed_size_mb=1.0,
        original_size_mb=2.0,
    )

    success = await manager._send_compressed_shard(compressed, "peer")
    assert success
    assert engine.p2p_node.sent_messages
    assert engine.p2p_node.sent_files == [("peer", str(file_path))]
