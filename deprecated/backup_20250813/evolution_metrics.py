"""Utilities for tracking evolution metrics."""

# ruff: noqa: I001

import asyncio
import gzip
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import psutil

    PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover - no psutil available
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import redis

    from src.core.security.secure_redis_client import (
        create_secure_redis_client,
        validate_redis_url_security,
    )

    REDIS_AVAILABLE = True
except Exception:  # pragma: no cover - no redis available
    redis = None  # type: ignore
    create_secure_redis_client = None  # type: ignore
    validate_redis_url_security = None  # type: ignore
    REDIS_AVAILABLE = False

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

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EvolutionMetrics":
        """Create an EvolutionMetrics instance from a dictionary."""
        return EvolutionMetrics(
            timestamp=data["timestamp"],
            agent_id=data["agent_id"],
            evolution_type=data["evolution_type"],
            evolution_id=data["evolution_id"],
            performance_score=data["performance_score"],
            improvement_delta=data["improvement_delta"],
            quality_score=data["quality_score"],
            memory_used_mb=data["memory_used_mb"],
            cpu_percent_avg=data["cpu_percent_avg"],
            duration_minutes=data["duration_minutes"],
            success=data["success"],
            error_count=data["error_count"],
            warning_count=data["warning_count"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvolutionAnalysis:
    """Simple container for evolution analysis results."""

    improvement_rate: float
    successful_mutations: list[str]
    plateau_detected: bool
    suggestions: list[str]


class MetricsBackend:
    """Abstract backend interface for storing evolution metrics."""

    async def start(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def stop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def save_metrics(self, metrics: list[EvolutionMetrics]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def load_historical_metrics(
        self, limit: int | None = None
    ) -> list[dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError


class SQLiteMetricsBackend(MetricsBackend):
    """SQLite storage backend for evolution metrics."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._round_id: int | None = None

    async def start(self) -> None:
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO evolution_rounds (start_time, status) VALUES (?, ?)",
            (time.time(), "running"),
        )
        self._round_id = int(cur.lastrowid)
        self._conn.commit()

    async def stop(self) -> None:
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

    def _migrate(self) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
        row = cur.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            cur.execute("INSERT INTO schema_version (version) VALUES (1)")
            self._create_tables(cur)
        self._conn.commit()

    def _create_tables(self, cur: sqlite3.Cursor) -> None:
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

    async def save_metrics(self, metrics: list[EvolutionMetrics]) -> None:
        if not metrics or self._conn is None or self._round_id is None:
            return
        cur = self._conn.cursor()
        for m in metrics:
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

    async def load_historical_metrics(self, limit: int | None = None) -> list[dict[str, Any]]:
        conn = self._conn or sqlite3.connect(self.db_path)
        cur = conn.cursor()
        query = "SELECT agent_id, fitness_score, timestamp FROM fitness_metrics ORDER BY timestamp DESC"
        if limit is not None:
            query += " LIMIT ?"
            rows = cur.execute(query, (limit,)).fetchall()
        else:
            rows = cur.execute(query).fetchall()
        if conn is not self._conn:
            conn.close()
        return [{"agent_id": a, "fitness_score": f, "timestamp": t} for a, f, t in rows]


class FileMetricsBackend(MetricsBackend):
    """File based backend using JSONL with rotation and compression."""

    def __init__(self, log_dir: str, rotate_size: int = 5 * 1024 * 1024) -> None:
        self.log_dir = Path(log_dir)
        self.rotate_size = rotate_size
        self._file: os.PathLike | None = None
        self._fp = None

    async def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._open_new_file()

    def _open_new_file(self) -> None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._file = self.log_dir / f"metrics_{ts}.jsonl"
        self._fp = open(self._file, "a", encoding="utf-8")

    async def stop(self) -> None:
        if self._fp:
            self._fp.flush()
            self._fp.close()
            self._compress_current()
            self._fp = None
            self._file = None

    def _compress_current(self) -> None:
        if self._file and Path(self._file).exists():
            with (
                open(self._file, "rb") as f_in,
                gzip.open(f"{self._file}.gz", "wb") as f_out,
            ):
                f_out.writelines(f_in)
            os.remove(self._file)

    async def save_metrics(self, metrics: list[EvolutionMetrics]) -> None:
        if not self._fp:
            return
        for m in metrics:
            self._fp.write(json.dumps(m.to_dict()) + "\n")
        self._fp.flush()
        if self._file and Path(self._file).stat().st_size > self.rotate_size:
            self._fp.close()
            self._compress_current()
            self._open_new_file()

    async def load_historical_metrics(self, limit: int | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        files = sorted(self.log_dir.glob("metrics_*.jsonl*"))
        for path in files:
            if path.suffix == ".gz":
                fh = gzip.open(path, "rt", encoding="utf-8")
            else:
                fh = open(path, encoding="utf-8")
            with fh:
                for line in fh:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        if limit is not None:
            return records[-limit:]
        return records


class RedisMetricsBackend(MetricsBackend):
    """Redis based backend with SQLite persistence on shutdown."""

    def __init__(self, redis_url: str, sqlite_path: str) -> None:
        self.redis_url = redis_url
        self.redis_key = "evolution:metrics"
        self.redis_channel = "evolution_metrics"
        self._redis = None
        self.sqlite_backend = SQLiteMetricsBackend(sqlite_path)

    async def start(self) -> None:
        if not REDIS_AVAILABLE:
            msg = "redis library not available"
            raise RuntimeError(msg)

        # Validate Redis URL security
        if validate_redis_url_security:
            validate_redis_url_security(self.redis_url)

        # Create secure Redis client
        if create_secure_redis_client:
            self._redis = create_secure_redis_client(self.redis_url, decode_responses=True)
        else:
            # Fallback to basic Redis client
            self._redis = redis.Redis.from_url(self.redis_url, decode_responses=True)

        await self.sqlite_backend.start()

    async def stop(self) -> None:
        if self._redis is not None:
            stored = self._redis.lrange(self.redis_key, 0, -1)
            if stored:
                metrics = [EvolutionMetrics.from_dict(json.loads(m)) for m in stored]
                await self.sqlite_backend.save_metrics(metrics)
            self._redis.delete(self.redis_key)
            self._redis.close()
            self._redis = None
        await self.sqlite_backend.stop()

    async def save_metrics(self, metrics: list[EvolutionMetrics]) -> None:
        if not metrics:
            return
        assert self._redis is not None
        pipe = self._redis.pipeline()
        for m in metrics:
            data = json.dumps(m.to_dict())
            pipe.rpush(self.redis_key, data)
            pipe.publish(self.redis_channel, data)
        pipe.execute()
        # Persist to SQLite for durability
        await self.sqlite_backend.save_metrics(metrics)

    async def load_historical_metrics(self, limit: int | None = None) -> list[dict[str, Any]]:
        return await self.sqlite_backend.load_historical_metrics(limit)


class EvolutionMetricsCollector:
    """Collector for evolution metrics."""

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the metrics collector."""
        self.config = config or {}
        self.metrics_history: list[EvolutionMetrics] = []
        self.active_collections: dict[str, dict[str, Any]] = {}
        self.system_events: list[dict[str, Any]] = []
        self.system_metrics_history: list[dict[str, Any]] = []
        self.mutation_recorder = EvolutionMetricsRecorder(self.config.get("metrics_file", "evolution_metrics.json"))

        self.db_path = self.config.get("db_path", "evolution_metrics.db")
        backend_type = self.config.get("storage_backend", "sqlite").lower()
        if backend_type == "redis":
            redis_url = self.config.get("redis_url", "redis://localhost:6379/0")
            self._backend: MetricsBackend = RedisMetricsBackend(redis_url, self.db_path)
        elif backend_type == "file":
            log_dir = self.config.get("log_dir", "evolution_logs")
            self._backend = FileMetricsBackend(log_dir)
        else:
            self._backend = SQLiteMetricsBackend(self.db_path)
        self._flush_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start metrics collection."""
        await self._backend.start()
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info(
            "Evolution metrics collector started using %s backend",
            self.config.get("storage_backend", "sqlite"),
        )

    async def stop(self) -> None:
        """Stop metrics collection."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except BaseException:  # pragma: no cover - cancellation
                pass
        await self._flush_metrics()
        await self._backend.stop()
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
                "mutation_id": start_info.get("mutation_id"),
            },
        )

        await self.save_metrics(metrics)
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

    async def save_metrics(self, metrics: EvolutionMetrics) -> None:
        """Queue metrics for persistence."""
        self.metrics_history.append(metrics)
        threshold = int(self.config.get("flush_threshold", 50))
        if len(self.metrics_history) >= threshold:
            await self._flush_metrics()

    async def load_historical_metrics(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Load historical metrics from the configured backend."""
        await self._flush_metrics()
        return await self._backend.load_historical_metrics(limit)

    async def export_data(self, path: str, limit: int | None = None) -> None:
        """Export historical metrics to a JSON file."""
        data = await self.load_historical_metrics(limit)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    async def analyze_evolution_progress(self) -> EvolutionAnalysis:
        """Analyze stored metrics for improvement trends."""
        data = await self.load_historical_metrics()
        if not data:
            return EvolutionAnalysis(0.0, [], False, ["no data available"])

        data = sorted(data, key=lambda m: m["timestamp"])
        first = data[0]
        last = data[-1]
        hours = max((last["timestamp"] - first["timestamp"]) / 3600, 1e-9)
        improvement_rate = (last["fitness_score"] - first["fitness_score"]) / hours

        plateau = abs(improvement_rate) < 0.01
        suggestions: list[str] = []
        if plateau:
            suggestions.append("Performance plateau detected; consider exploring new mutation strategies.")

        return EvolutionAnalysis(
            improvement_rate=improvement_rate,
            successful_mutations=[],
            plateau_detected=plateau,
            suggestions=suggestions,
        )

    async def _periodic_flush(self) -> None:
        while True:
            await asyncio.sleep(30)
            await self._flush_metrics()

    async def _flush_metrics(self) -> None:
        if not self.metrics_history:
            return
        await self._backend.save_metrics(self.metrics_history)
        self.metrics_history.clear()

    # Query helpers --------------------------------------------------
    def get_rounds(self) -> list[tuple[int, float, float, str]]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, start_time, end_time, status FROM evolution_rounds")
        rows = cur.fetchall()
        conn.close()
        return rows

    def get_fitness_metrics(self, round_id: int) -> list[tuple[str, float, float]]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT agent_id, fitness_score, timestamp FROM fitness_metrics WHERE round_id=?",
            (round_id,),
        )
        rows = cur.fetchall()
        conn.close()
        return rows

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
