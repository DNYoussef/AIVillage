import numpy as np
import torch

from src.infrastructure.p2p.tensor_streaming import TensorStreamer


def test_safe_serialization_round_trip():
    streamer = TensorStreamer()
    array = np.arange(10, dtype=np.float32)
    data = streamer._serialize_tensor(array)
    restored = streamer._deserialize_tensor(data)
    assert np.array_equal(array, restored)


def test_safe_torch_serialization_round_trip():
    streamer = TensorStreamer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.arange(10, dtype=torch.float32, device=device)
    data = streamer._serialize_tensor(tensor)
    restored = streamer._deserialize_tensor(data)
    assert isinstance(restored, torch.Tensor)
    assert restored.device.type == tensor.device.type
    assert restored.dtype == tensor.dtype
    assert torch.equal(restored.cpu(), tensor.cpu())
