import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from production.agent_forge.evolution.metrics import EvolutionMetricsRecorder


def test_metrics_persist(tmp_path):
    storage = tmp_path / "metrics.json"
    recorder = EvolutionMetricsRecorder(storage_path=storage)

    mutation_id = recorder.record_evolution_start("mutation", "edge")
    recorder.record_fitness(mutation_id, 0.75)
    recorder.record_evolution_end(mutation_id, selected=True, compression_ratio=0.5)

    assert storage.exists()
    data = json.loads(storage.read_text())
    assert len(data) == 1
    record = data[0]
    assert record["fitness_score"] == 0.75
    assert record["selected"] is True
    assert record["compression_ratio"] == 0.5

    summary = recorder.get_metrics_summary()
    assert summary["total_rounds"] == 1
    assert summary["avg_fitness"] == 0.75
