"""Thread-safe evolution metrics recording system with Prometheus integration."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Optional dependencies -----------------------------------------------------
try:  # pragma: no cover - import guard
    from prometheus_client import Counter, Gauge

    PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    PROMETHEUS_AVAILABLE = False

    class _DummyMetric:
        """Fallback metric when prometheus_client isn't installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
            pass

        def labels(self, **_labels: Any) -> "_DummyMetric":  # type: ignore[override]
            return self

        def inc(self, amount: float = 1.0) -> None:  # pragma: no cover - no-op
            return None

        def set(self, value: float) -> None:  # pragma: no cover - no-op
            return None

    Counter = Gauge = _DummyMetric  # type: ignore

try:  # pragma: no cover - import guard
    import psutil

    PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

# Prometheus metric definitions ---------------------------------------------
EVOLUTION_ROUNDS = Counter(
    "evolution_round_total",
    "Number of evolution rounds executed",
    ["mutation_type", "node_type"],
)

FITNESS_GAUGE = Gauge(
    "fitness_score",
    "Fitness score achieved by a mutation",
    ["mutation_type", "node_type"],
)

RESOURCE_EFFICIENCY_GAUGE = Gauge(
    "resource_efficiency",
    "Estimated resource efficiency of a mutation",
    ["mutation_type", "node_type"],
)


@dataclass
class MutationMetrics:
    """Metrics captured for a single mutation/evolution round."""

    mutation_id: str
    mutation_type: str
    node_type: str
    start_time: float
    end_time: float | None = None
    fitness_score: float | None = None
    cpu_percent: float | None = None
    memory_mb: float | None = None
    energy_joules: float | None = None
    selected: bool | None = None
    compression_ratio: float | None = None

    def resource_efficiency(self) -> float | None:
        """Compute simple resource efficiency metric."""
        if self.fitness_score is None or self.cpu_percent is None:
            return None
        return self.fitness_score / (self.cpu_percent + 1e-6)


class EvolutionMetricsRecorder:
    """Thread-safe recorder persisting mutation metrics."""

    def __init__(self, storage_path: str | Path = "evolution_metrics.json") -> None:
        self.storage_path = Path(storage_path)
        self.lock = threading.Lock()
        self.active: Dict[str, MutationMetrics] = {}
        self.completed: List[MutationMetrics] = []

        # Load existing metrics if available
        if self.storage_path.exists():  # pragma: no cover - simple IO
            try:
                data = json.loads(self.storage_path.read_text())
                for item in data:
                    self.completed.append(MutationMetrics(**item))
            except Exception:
                # Corrupt file -> start fresh
                self.completed = []

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def record_evolution_start(self, mutation_type: str, node_type: str) -> str:
        """Record the start of a mutation/evolution round.

        Returns a mutation identifier that should be used for subsequent
        fitness and completion recording.
        """

        mutation_id = str(uuid.uuid4())
        start_time = time.time()
        cpu_percent = psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else 0.0
        mem_mb = (
            psutil.Process().memory_info().rss / (1024**2) if PSUTIL_AVAILABLE else 0.0
        )

        with self.lock:
            self.active[mutation_id] = MutationMetrics(
                mutation_id=mutation_id,
                mutation_type=mutation_type,
                node_type=node_type,
                start_time=start_time,
                cpu_percent=cpu_percent,
                memory_mb=mem_mb,
            )

        EVOLUTION_ROUNDS.labels(
            mutation_type=mutation_type, node_type=node_type
        ).inc()
        return mutation_id

    def record_fitness(self, mutation_id: str, fitness: float) -> None:
        """Record fitness score for an active mutation."""
        with self.lock:
            metrics = self.active.get(mutation_id)
            if metrics:
                metrics.fitness_score = fitness
                FITNESS_GAUGE.labels(
                    mutation_type=metrics.mutation_type,
                    node_type=metrics.node_type,
                ).set(fitness)

    def record_evolution_end(
        self, mutation_id: str, selected: bool, compression_ratio: float
    ) -> None:
        """Finalize mutation metrics and persist them."""
        end_time = time.time()
        cpu_percent = psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else 0.0
        mem_mb = (
            psutil.Process().memory_info().rss / (1024**2) if PSUTIL_AVAILABLE else 0.0
        )

        with self.lock:
            metrics = self.active.pop(mutation_id, None)
            if not metrics:
                return
            metrics.end_time = end_time
            metrics.cpu_percent = cpu_percent
            metrics.memory_mb = mem_mb
            duration = end_time - metrics.start_time
            metrics.energy_joules = cpu_percent * duration * 0.5  # rough estimate
            metrics.selected = selected
            metrics.compression_ratio = compression_ratio
            self.completed.append(metrics)

            efficiency = metrics.resource_efficiency() or 0.0
            RESOURCE_EFFICIENCY_GAUGE.labels(
                mutation_type=metrics.mutation_type,
                node_type=metrics.node_type,
            ).set(efficiency)

            self._persist()  # Persist after each completion

    # ------------------------------------------------------------------
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return simple aggregate statistics for persisted metrics."""
        with self.lock:
            total = len(self.completed)
            if total == 0:
                return {"total_rounds": 0, "avg_fitness": 0.0, "avg_efficiency": 0.0}

            avg_fitness = sum(
                m.fitness_score or 0.0 for m in self.completed
            ) / total
            avg_eff = sum(
                m.resource_efficiency() or 0.0 for m in self.completed
            ) / total
            return {
                "total_rounds": total,
                "avg_fitness": avg_fitness,
                "avg_efficiency": avg_eff,
            }

    # ------------------------------------------------------------------
    def _persist(self) -> None:
        """Persist completed metrics to disk."""
        data = [asdict(m) for m in self.completed]
        try:  # pragma: no cover - simple IO
            self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
