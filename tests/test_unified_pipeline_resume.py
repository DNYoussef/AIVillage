import asyncio

import pytest
import sys
import types

sys.modules.setdefault("compression_pipeline", types.ModuleType("compression_pipeline"))
sys.modules["compression_pipeline"].CompressionConfig = object
sys.modules["compression_pipeline"].CompressionPipeline = object

sys.modules.setdefault("evomerge_pipeline", types.ModuleType("evomerge_pipeline"))
sys.modules["evomerge_pipeline"].EvolutionConfig = object
sys.modules["evomerge_pipeline"].EvoMergePipeline = object

sys.modules.setdefault("quietstar_baker", types.ModuleType("quietstar_baker"))
sys.modules["quietstar_baker"].QuietSTaRBaker = object
sys.modules["quietstar_baker"].QuietSTaRConfig = object

from agent_forge.unified_pipeline import (
    PipelineState,
    UnifiedPipeline,
    UnifiedPipelineConfig,
    run_pipeline,
)


async def _noop(*args, **kwargs):
    return None


async def _report(self):
    return {"pipeline_summary": {"run_id": self.state.run_id}}


def test_resume_from_checkpoint(tmp_path, monkeypatch):
    async def fake_evomerge(self):
        self.state.evomerge_model_path = "evomerge.model"

    async def failing_quietstar(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(UnifiedPipeline, "run_evomerge_phase", fake_evomerge)
    monkeypatch.setattr(UnifiedPipeline, "run_quietstar_phase", failing_quietstar)
    monkeypatch.setattr(UnifiedPipeline, "run_compression_phase", _noop)
    monkeypatch.setattr(UnifiedPipeline, "calculate_final_metrics", _noop)
    monkeypatch.setattr(UnifiedPipeline, "generate_final_report", _report)

    config = UnifiedPipelineConfig(base_output_dir=tmp_path, checkpoint_dir=tmp_path)
    pipeline = UnifiedPipeline(config)

    with pytest.raises(RuntimeError):
        asyncio.run(pipeline.run_complete_pipeline())

    checkpoint = next(tmp_path.glob("unified_pipeline_*.json"))
    state = PipelineState.load_checkpoint(checkpoint)
    assert state.completed_phases == ["evomerge"]

    evomerge_called = False

    async def mark_evomerge(self):
        nonlocal evomerge_called
        evomerge_called = True

    async def fake_quietstar(self):
        self.state.quietstar_model_path = "quietstar.model"

    async def fake_compression(self, source_model):
        self.state.final_model_path = "final.model"

    monkeypatch.setattr(UnifiedPipeline, "run_evomerge_phase", mark_evomerge)
    monkeypatch.setattr(UnifiedPipeline, "run_quietstar_phase", fake_quietstar)
    monkeypatch.setattr(UnifiedPipeline, "run_compression_phase", fake_compression)
    monkeypatch.setattr(UnifiedPipeline, "calculate_final_metrics", _noop)
    monkeypatch.setattr(UnifiedPipeline, "generate_final_report", _report)

    run_pipeline.callback(
        config=None,
        evomerge=True,
        quietstar=True,
        compression=True,
        generations=1,
        output_dir=str(tmp_path / "out"),
        device="auto",
        resume=str(checkpoint),
    )

    assert not evomerge_called
    resumed = PipelineState.load_checkpoint(checkpoint)
    assert resumed.completed_phases == ["evomerge", "quietstar", "compression"]
