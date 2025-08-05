import numpy as np

from src.infrastructure.p2p.tensor_streaming import TensorStreamer


def test_safe_serialization_round_trip():
    streamer = TensorStreamer()
    array = np.arange(10, dtype=np.float32)
    data = streamer._serialize_tensor(array)
    restored = streamer._deserialize_tensor(data)
    assert np.array_equal(array, restored)
