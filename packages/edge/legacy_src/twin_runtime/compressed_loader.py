import torch
from src.agent_forge.compression import CompressionConfig, TwoStageCompressor


class CompressedModelLoader:
    """Reconstruct a model from multi-stage compressed weights."""

    def __init__(self, model_cls, compressed_path: str, config: CompressionConfig | None = None) -> None:
        self.model = model_cls()
        self.compressed = torch.load(compressed_path, map_location="cpu")
        self.comp = TwoStageCompressor(config or CompressionConfig(use_hyper=True))

    def assemble_model(self) -> torch.nn.Module:
        state = {}
        for name, data in self.compressed.items():
            if name == "__compression_ratio__":
                continue
            if isinstance(data, dict) and any(k in data for k in ("seedlm", "vptq", "hyper")):
                state[name] = self.comp.decompress_layer(data)
            else:
                state[name] = data
        self.model.load_state_dict(state, strict=False)
        return self.model
