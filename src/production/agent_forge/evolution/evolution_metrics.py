"""Utilities for tracking evolution metrics."""

# ruff: noqa: I001

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:  # pragma: no cover - optional dependency
    import psutil

    PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover - no psutil available
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

from .metrics import EvolutionMetricsRecorder

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of evolution metrics."""

    PERFORMANCE = "performance"
    RESOURCE = "resource"
    QUALITY = "quality"
    DURATION = "duration"
    SUCCESS_RATE = "success_rate"
    EFFICIENCY = "efficiency"


@dataclass
class EvolutionMetrics:
    """Evolution metrics data structure."""

    timestamp: float
    agent_id: str
    evolution_type: str
    evolution_id: str

    # Performance metrics
    performance_score: float
    improvement_delta: float
    quality_score: float

    # Resource metrics
    memory_used_mb: int
    cpu_percent_avg: float
    duration_minutes: float

    # Success metrics
    success: bool
    error_count: int
    warning_count: int

    # Additional data
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "evolution_type": self.evolution_type,
            "evolution_id": self.evolution_id,
            "performance_score": self.performance_score,
            "improvement_delta": self.improvement_delta,
            "quality_score": self.quality_score,
            "memory_used_mb": self.memory_used_mb,
            "cpu_percent_avg": self.cpu_percent_avg,
            "duration_minutes": self.duration_minutes,
            "success": self.success,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "metadata": self.metadata,
        }


class EvolutionMetricsCollector:
    """Collector for evolution metrics."""

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the metrics collector."""
        self.config = config or {}
        self.metrics_history: list[EvolutionMetrics] = []
        self.active_collections: dict[str, dict[str, Any]] = {}
        self.system_events: list[dict[str, Any]] = []
        self.system_metrics_history: list[dict[str, Any]] = []
        self.mutation_recorder = EvolutionMetricsRecorder(
            self.config.get("metrics_file", "evolution_metrics.json")
        )

    async def start(self) -> None:
        """Start metrics collection."""
        logger.info("Evolution metrics collector started")

    async def stop(self) -> None:
        """Stop metrics collection."""
        logger.info("Evolution metrics collector stopped")

    async def record_evolution_start(self, evolution_event: Any) -> None:
        """Record evolution start."""
        evolution_id = f"{evolution_event.agent_id}-{int(evolution_event.timestamp)}"
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            cpu_start = process.cpu_percent(interval=None)
            mem_start = process.memory_info().rss / (1024**2)
        else:
            cpu_start = 0.0
            mem_start = 0.0

        mutation_id = self.mutation_recorder.record_evolution_start(
            mutation_type=getattr(evolution_event, "evolution_type", "unknown"),
            node_type=self.config.get("node_type", "lighthouse"),
        )

        self.active_collections[evolution_id] = {
            "start_time": evolution_event.timestamp,
            "cpu_start": cpu_start,
            "mem_start": mem_start,
            "pre_kpis": evolution_event.pre_evolution_kpis,
            "mutation_id": mutation_id,
        }
        logger.info(
            "Evolution started for agent %s (id=%s)",
            evolution_event.agent_id,
            evolution_id,
        )

    async def record_evolution_end(self, evolution_event: Any) -> None:
        """Record evolution completion."""
        evolution_id = f"{evolution_event.agent_id}-{int(evolution_event.timestamp)}"
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            cpu_now = process.cpu_percent(interval=None)
            mem_now = process.memory_info().rss / (1024**2)
        else:
            cpu_now = 0.0
            mem_now = 0.0
        start_info = self.active_collections.pop(evolution_id, {})

        pre_perf = start_info.get("pre_kpis", {}).get(
            "performance", evolution_event.pre_evolution_kpis.get("performance", 0.0)
        )
        post_kpis = evolution_event.post_evolution_kpis or {}
        performance_score = post_kpis.get("performance", pre_perf)
        quality_score = post_kpis.get("quality", 0.0)

        metrics = EvolutionMetrics(
            timestamp=evolution_event.timestamp,
            agent_id=evolution_event.agent_id,
            evolution_type=evolution_event.evolution_type,
            evolution_id=evolution_id,
            performance_score=performance_score,
            improvement_delta=performance_score - pre_perf,
            quality_score=quality_score,
            memory_used_mb=int(mem_now),
            cpu_percent_avg=cpu_now,
            duration_minutes=evolution_event.duration_seconds / 60,
            success=evolution_event.success,
            error_count=sum(1 for msg in evolution_event.insights if msg.lower().startswith("error")),
            warning_count=sum(1 for msg in evolution_event.insights if msg.lower().startswith("warn")),
            metadata={
                "trigger_reason": evolution_event.trigger_reason,
                "generation_change": evolution_event.generation_change,
                "insights": evolution_event.insights,
            },
        )

        self.metrics_history.append(metrics)
        mutation_id = start_info.get("mutation_id")
        if mutation_id is not None:
            self.mutation_recorder.record_fitness(mutation_id, performance_score)
            self.mutation_recorder.record_evolution_end(
                mutation_id,
                selected=evolution_event.success,
                compression_ratio=getattr(evolution_event, "compression_ratio", 1.0),
            )
        logger.info(
            "Evolution completed for agent %s (success=%s)",
            evolution_event.agent_id,
            evolution_event.success,
        )

    async def record_evolution_completion(self, evolution_event: Any) -> None:
        """Backward compatible wrapper for record_evolution_end."""
        await self.record_evolution_end(evolution_event)

    async def record_system_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Record system event."""
        event = {"type": event_type, "timestamp": time.time(), "data": data}
        self.system_events.append(event)
        logger.info("System event recorded: %s", event_type)

    async def record_system_metrics(self, metrics: dict[str, Any]) -> None:
        """Record system metrics."""
        metrics_entry = {"timestamp": time.time(), **metrics}
        self.system_metrics_history.append(metrics_entry)
        logger.debug("System metrics recorded: %s", metrics_entry)

    async def record_fitness(self, mutation_id: str, fitness: float) -> None:
        """Record fitness score for a mutation."""
        self.mutation_recorder.record_fitness(mutation_id, fitness)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary statistics for recorded mutation metrics."""
        return self.mutation_recorder.get_metrics_summary()
