"""Utilities for tracking evolution metrics."""

# ruff: noqa: I001

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
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

        self.db_path = self.config.get("db_path", "evolution_metrics.db")
        self._conn: sqlite3.Connection | None = None
        self._flush_task: asyncio.Task | None = None
        self._round_id: int | None = None

    async def start(self) -> None:
        """Start metrics collection."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            "PRAGMA journal_mode=WAL"
        )  # better durability for frequent writes
        self._create_tables()
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO evolution_rounds (start_time, status) VALUES (?, ?)",
            (time.time(), "running"),
        )
        self._round_id = int(cur.lastrowid)
        self._conn.commit()
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("Evolution metrics collector started")

    async def stop(self) -> None:
        """Stop metrics collection."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except Exception:  # pragma: no cover - cancellation
                pass
        await self._flush_metrics()
        if self._conn and self._round_id is not None:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE evolution_rounds SET end_time=?, status=? WHERE id=?",
                (time.time(), "completed", self._round_id),
            )
            self._conn.commit()
            self._conn.close()
            self._conn = None
            self._round_id = None
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
            error_count=sum(
                1 for msg in evolution_event.insights if msg.lower().startswith("error")
            ),
            warning_count=sum(
                1 for msg in evolution_event.insights if msg.lower().startswith("warn")
            ),
            metadata={
                "trigger_reason": evolution_event.trigger_reason,
                "generation_change": evolution_event.generation_change,
                "insights": evolution_event.insights,
                "mutation_id": start_info.get("mutation_id"),
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

    # ------------------------------------------------------------------
    def _create_tables(self) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS evolution_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL,
                end_time REAL,
                status TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fitness_metrics (
                round_id INTEGER,
                agent_id TEXT,
                fitness_score REAL,
                timestamp REAL,
                FOREIGN KEY(round_id) REFERENCES evolution_rounds(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS resource_metrics (
                round_id INTEGER,
                cpu_usage REAL,
                memory_mb REAL,
                energy_estimate REAL,
                FOREIGN KEY(round_id) REFERENCES evolution_rounds(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS selection_outcomes (
                round_id INTEGER,
                mutation_id TEXT,
                selected INTEGER,
                reason TEXT,
                FOREIGN KEY(round_id) REFERENCES evolution_rounds(id)
            )
            """
        )
        self._conn.commit()

    async def _periodic_flush(self) -> None:
        while True:
            await asyncio.sleep(30)
            await self._flush_metrics()

    async def _flush_metrics(self) -> None:
        if not self.metrics_history or self._conn is None or self._round_id is None:
            return
        cur = self._conn.cursor()
        for m in self.metrics_history:
            cur.execute(
                "INSERT INTO fitness_metrics (round_id, agent_id, fitness_score, timestamp) VALUES (?, ?, ?, ?)",
                (self._round_id, m.agent_id, m.performance_score, m.timestamp),
            )
            cur.execute(
                "INSERT INTO resource_metrics (round_id, cpu_usage, memory_mb, energy_estimate) VALUES (?, ?, ?, ?)",
                (
                    self._round_id,
                    m.cpu_percent_avg,
                    m.memory_used_mb,
                    m.metadata.get("energy_estimate", 0.0),
                ),
            )
            cur.execute(
                "INSERT INTO selection_outcomes (round_id, mutation_id, selected, reason) VALUES (?, ?, ?, ?)",
                (
                    self._round_id,
                    m.metadata.get("mutation_id"),
                    int(m.success),
                    m.metadata.get("trigger_reason", ""),
                ),
            )
        self._conn.commit()
        self.metrics_history.clear()

    # Query helpers --------------------------------------------------
    def get_rounds(self) -> list[tuple[int, float, float, str]]:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        cur = self._conn.cursor()
        cur.execute("SELECT id, start_time, end_time, status FROM evolution_rounds")
        return cur.fetchall()

    def get_fitness_metrics(self, round_id: int) -> list[tuple[str, float, float]]:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        cur = self._conn.cursor()
        cur.execute(
            "SELECT agent_id, fitness_score, timestamp FROM fitness_metrics WHERE round_id=?",
            (round_id,),
        )
        return cur.fetchall()

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
