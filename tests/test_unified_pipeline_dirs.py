import sys
import types

compression_stub = types.ModuleType("compression_pipeline")
compression_stub.CompressionConfig = object
compression_stub.CompressionPipeline = object
sys.modules["compression_pipeline"] = compression_stub

evomerge_stub = types.ModuleType("evomerge_pipeline")
evomerge_stub.EvolutionConfig = object
evomerge_stub.EvoMergePipeline = object
evomerge_stub.BaseModelConfig = object
sys.modules["evomerge_pipeline"] = evomerge_stub

quietstar_stub = types.ModuleType("quietstar_baker")
quietstar_stub.QuietSTaRBaker = object
quietstar_stub.QuietSTaRConfig = object
sys.modules["quietstar_baker"] = quietstar_stub

from agent_forge.unified_pipeline import UnifiedPipeline, UnifiedPipelineConfig  # noqa: E402


def test_pipeline_creates_directories(tmp_path):
    config = UnifiedPipelineConfig(
        base_output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
        enable_evomerge=False,
        enable_quietstar=False,
        enable_compression=False,
    )
    UnifiedPipeline(config)
    assert config.base_output_dir.is_dir()
    assert config.checkpoint_dir.is_dir()
