"""Integrated Evolution Metrics System with CODEX requirements.

This module provides comprehensive evolution metrics tracking with:
- SQLite database persistence with WAL mode
- Redis integration with automatic fallback
- 18 KPI tracking system
- Real-time metric collection and aggregation
- API endpoints for health monitoring
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
import os
from pathlib import Path
from queue import Empty, Queue
import sqlite3
import threading
import time
from typing import Any

# Try to import Redis with graceful fallback
try:
    import redis
    from redis import Redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

# Try to import psutil for resource monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


class KPIType(Enum):
    """18 Core KPIs for Evolution System."""

    PERFORMANCE_SCORE = "performance_score"
    LEARNING_RATE = "learning_rate"
    TASK_COMPLETION = "task_completion"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    MEMORY_EFFICIENCY = "memory_efficiency"
    CPU_EFFICIENCY = "cpu_efficiency"
    ADAPTATION_SPEED = "adaptation_speed"
    CREATIVITY_SCORE = "creativity_score"
    COLLABORATION_SCORE = "collaboration_score"
    SPECIALIZATION_DEPTH = "specialization_depth"
    GENERALIZATION_BREADTH = "generalization_breadth"
    ROBUSTNESS_SCORE = "robustness_score"
    ENERGY_EFFICIENCY = "energy_efficiency"
    KNOWLEDGE_RETENTION = "knowledge_retention"
    INNOVATION_RATE = "innovation_rate"
    QUALITY_CONSISTENCY = "quality_consistency"
    RESOURCE_UTILIZATION = "resource_utilization"


@dataclass
class EvolutionMetricsData:
    """Complete evolution metrics data structure with 18 KPIs."""

    # Core identifiers
    timestamp: float = field(default_factory=time.time)
    round_number: int = 0
    generation: int = 0
    agent_id: str = ""
    evolution_type: str = ""

    # 18 KPI values (all initialized to 0.0)
    performance_score: float = 0.0
    learning_rate: float = 0.0
    task_completion: float = 0.0
    error_rate: float = 0.0
    response_time: float = 0.0
    memory_efficiency: float = 0.0
    cpu_efficiency: float = 0.0
    adaptation_speed: float = 0.0
    creativity_score: float = 0.0
    collaboration_score: float = 0.0
    specialization_depth: float = 0.0
    generalization_breadth: float = 0.0
    robustness_score: float = 0.0
    energy_efficiency: float = 0.0
    knowledge_retention: float = 0.0
    innovation_rate: float = 0.0
    quality_consistency: float = 0.0
    resource_utilization: float = 0.0

    # Resource metrics
    memory_used_mb: float = 0.0
    cpu_percent: float = 0.0
    network_io_kb: float = 0.0
    disk_io_kb: float = 0.0
    gpu_usage: float = 0.0

    # Selection outcomes
    fitness_score: float = 0.0
    selected: bool = False
    selection_method: str = ""
    mutation_applied: bool = False

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionMetricsData":
        """Create from dictionary."""
        return cls(**data)


class IntegratedEvolutionMetrics:
    """Integrated Evolution Metrics system with CODEX requirements.

    Features:
    - SQLite database with WAL mode
    - Redis caching with fallback
    - 18 KPI tracking
    - Batch flushing with configurable threshold
    - Real-time metrics collection
    """

    def __init__(self) -> None:
        """Initialize the integrated metrics system."""
        # Load configuration from environment
        self.db_path = os.getenv("AIVILLAGE_DB_PATH", "./data/evolution_metrics.db")
        self.storage_backend = os.getenv("AIVILLAGE_STORAGE_BACKEND", "sqlite")
        self.flush_threshold = int(os.getenv("AIVILLAGE_METRICS_FLUSH_THRESHOLD", "50"))
        self.metrics_file = os.getenv("AIVILLAGE_METRICS_FILE", "evolution_metrics.json")
        self.log_dir = os.getenv("AIVILLAGE_LOG_DIR", "./evolution_logs")

        # Redis configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))

        # Initialize components
        self.db_conn = None
        self.redis_client = None
        self.metrics_buffer = []
        self.buffer_lock = threading.Lock()
        self.current_round_id = None

        # Metrics tracking
        self.total_metrics_collected = 0
        self.last_flush_time = time.time()

        # Background worker
        self.worker_thread = None
        self.worker_queue = Queue()
        self.running = False

        # Initialize storage
        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize storage backends."""
        # Initialize SQLite
        self._init_sqlite()

        # Try to initialize Redis
        if REDIS_AVAILABLE and self.storage_backend in ["redis", "hybrid"]:
            self._init_redis()

        # Create log directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def _init_sqlite(self) -> None:
        """Initialize SQLite database with WAL mode."""
        try:
            self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.db_conn.execute("PRAGMA journal_mode=WAL")
            self.db_conn.execute("PRAGMA synchronous=NORMAL")
            self.db_conn.execute("PRAGMA cache_size=10000")
            self.db_conn.execute("PRAGMA temp_store=MEMORY")
            self.db_conn.execute("PRAGMA mmap_size=268435456")

            logger.info(f"SQLite database initialized at {self.db_path} with WAL mode")
        except Exception as e:
            logger.exception(f"Failed to initialize SQLite: {e}")
            raise

    def _init_redis(self) -> None:
        """Initialize Redis connection with fallback."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis module not available, using SQLite only")
            return

        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis connection established at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.warning(f"Redis connection failed, falling back to SQLite: {e}")
            self.redis_client = None

    def start(self) -> None:
        """Start the metrics collection system."""
        if self.running:
            return

        self.running = True

        # Start new evolution round
        self._start_evolution_round()

        # Start background worker
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        logger.info("Evolution metrics system started")

    def stop(self) -> None:
        """Stop the metrics collection system."""
        if not self.running:
            return

        self.running = False

        # Flush remaining metrics
        self.flush()

        # Complete evolution round
        self._complete_evolution_round()

        # Wait for worker to finish
        if self.worker_thread:
            self.worker_queue.put(None)  # Signal to stop
            self.worker_thread.join(timeout=5)

        # Close connections
        if self.db_conn:
            self.db_conn.close()
        if self.redis_client:
            self.redis_client.close()

        logger.info("Evolution metrics system stopped")

    def _start_evolution_round(self) -> None:
        """Start a new evolution round in the database."""
        if not self.db_conn:
            return

        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO evolution_rounds (round_number, generation, status, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (
                    int(time.time()),  # Use timestamp as round number for uniqueness
                    1,  # Generation 1
                    "running",
                    datetime.now(),
                ),
            )
            self.current_round_id = cursor.lastrowid
            self.db_conn.commit()
            logger.info(f"Started evolution round {self.current_round_id}")
        except Exception as e:
            logger.exception(f"Failed to start evolution round: {e}")

    def _complete_evolution_round(self) -> None:
        """Mark the current evolution round as completed."""
        if not self.db_conn or not self.current_round_id:
            return

        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                UPDATE evolution_rounds
                SET status = 'completed'
                WHERE id = ?
            """,
                (self.current_round_id,),
            )
            self.db_conn.commit()
            logger.info(f"Completed evolution round {self.current_round_id}")
        except Exception as e:
            logger.exception(f"Failed to complete evolution round: {e}")

    def record_metric(self, metrics: EvolutionMetricsData) -> None:
        """Record evolution metrics.

        Args:
            metrics: EvolutionMetricsData object with KPI values
        """
        with self.buffer_lock:
            self.metrics_buffer.append(metrics)
            self.total_metrics_collected += 1

            # Check if we should flush
            if len(self.metrics_buffer) >= self.flush_threshold:
                self._flush_internal()

        # Also send to Redis for real-time access
        if self.redis_client:
            self._send_to_redis(metrics)

    def record_kpi(
        self,
        agent_id: str,
        kpi_type: KPIType,
        value: float,
        metadata: dict | None = None,
    ) -> None:
        """Record a single KPI value.

        Args:
            agent_id: Agent identifier
            kpi_type: Type of KPI from KPIType enum
            value: KPI value
            metadata: Optional metadata
        """
        metrics = EvolutionMetricsData(agent_id=agent_id, timestamp=time.time(), metadata=metadata or {})

        # Set the specific KPI value
        setattr(metrics, kpi_type.value, value)

        self.record_metric(metrics)

    def _flush_internal(self) -> None:
        """Internal flush method (assumes lock is held)."""
        if not self.metrics_buffer:
            return

        # Copy buffer and clear
        metrics_to_flush = self.metrics_buffer.copy()
        self.metrics_buffer.clear()

        # Queue for background processing
        self.worker_queue.put(("flush", metrics_to_flush))

        self.last_flush_time = time.time()

    def flush(self) -> None:
        """Manually flush metrics buffer."""
        with self.buffer_lock:
            self._flush_internal()

    def _worker_loop(self) -> None:
        """Background worker for processing metrics."""
        while self.running:
            try:
                item = self.worker_queue.get(timeout=1)
                if item is None:  # Stop signal
                    break

                action, data = item
                if action == "flush":
                    self._persist_metrics(data)

            except Empty:
                continue
            except Exception as e:
                logger.exception(f"Worker error: {e}")

    def _persist_metrics(self, metrics_list: list[EvolutionMetricsData]) -> None:
        """Persist metrics to database."""
        if not self.db_conn or not self.current_round_id:
            return

        try:
            cursor = self.db_conn.cursor()

            for metrics in metrics_list:
                # Insert fitness metrics
                cursor.execute(
                    """
                    INSERT INTO fitness_metrics
                    (round_id, agent_id, fitness_score, performance_metrics, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        self.current_round_id,
                        metrics.agent_id,
                        metrics.fitness_score,
                        json.dumps(
                            {
                                "performance_score": metrics.performance_score,
                                "learning_rate": metrics.learning_rate,
                                "task_completion": metrics.task_completion,
                                "error_rate": metrics.error_rate,
                                "response_time": metrics.response_time,
                                "adaptation_speed": metrics.adaptation_speed,
                                "creativity_score": metrics.creativity_score,
                                "collaboration_score": metrics.collaboration_score,
                                "specialization_depth": metrics.specialization_depth,
                                "generalization_breadth": metrics.generalization_breadth,
                                "robustness_score": metrics.robustness_score,
                                "innovation_rate": metrics.innovation_rate,
                                "quality_consistency": metrics.quality_consistency,
                                "knowledge_retention": metrics.knowledge_retention,
                            }
                        ),
                        datetime.fromtimestamp(metrics.timestamp),
                    ),
                )

                # Insert resource metrics
                cursor.execute(
                    """
                    INSERT INTO resource_metrics
                    (round_id, cpu_usage, memory_usage_mb, network_io_kb, disk_io_kb, gpu_usage, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        self.current_round_id,
                        metrics.cpu_percent,
                        metrics.memory_used_mb,
                        metrics.network_io_kb,
                        metrics.disk_io_kb,
                        metrics.gpu_usage,
                        datetime.fromtimestamp(metrics.timestamp),
                    ),
                )

                # Insert selection outcomes if applicable
                if metrics.selection_method:
                    cursor.execute(
                        """
                        INSERT INTO selection_outcomes
                        (round_id, parent_agent_id, selection_method, mutation_applied, survival_reason, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            self.current_round_id,
                            metrics.agent_id,
                            metrics.selection_method,
                            metrics.mutation_applied,
                            "selected" if metrics.selected else "not_selected",
                            datetime.fromtimestamp(metrics.timestamp),
                        ),
                    )

            self.db_conn.commit()
            logger.debug(f"Persisted {len(metrics_list)} metrics to database")

        except Exception as e:
            logger.exception(f"Failed to persist metrics: {e}")
            self.db_conn.rollback()

    def _send_to_redis(self, metrics: EvolutionMetricsData) -> None:
        """Send metrics to Redis for real-time access."""
        if not self.redis_client:
            return

        try:
            # Store in sorted set by timestamp
            key = f"evolution:metrics:{metrics.agent_id}"
            self.redis_client.zadd(key, {json.dumps(metrics.to_dict()): metrics.timestamp})

            # Set expiry (1 hour)
            self.redis_client.expire(key, 3600)

            # Update real-time leaderboard
            self.redis_client.zadd("evolution:leaderboard", {metrics.agent_id: metrics.fitness_score})

            # Publish to pub/sub channel
            self.redis_client.publish(
                "evolution:updates",
                json.dumps(
                    {
                        "agent_id": metrics.agent_id,
                        "fitness_score": metrics.fitness_score,
                        "timestamp": metrics.timestamp,
                    }
                ),
            )

        except Exception as e:
            logger.debug(f"Redis operation failed (non-critical): {e}")

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current metrics summary."""
        summary = {
            "total_metrics_collected": self.total_metrics_collected,
            "buffer_size": len(self.metrics_buffer),
            "last_flush_time": self.last_flush_time,
            "current_round_id": self.current_round_id,
            "redis_connected": self.redis_client is not None,
            "db_connected": self.db_conn is not None,
        }

        # Add leaderboard from Redis if available
        if self.redis_client:
            try:
                top_agents = self.redis_client.zrevrange("evolution:leaderboard", 0, 9, withscores=True)
                summary["top_agents"] = [{"agent_id": agent, "fitness_score": score} for agent, score in top_agents]
            except Exception as e:
                logger.warning(f"Failed to get top agents: {e}")
                summary["top_agents"] = []

        return summary

    def get_agent_history(self, agent_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get historical metrics for an agent."""
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                SELECT
                    fm.fitness_score,
                    fm.performance_metrics,
                    fm.timestamp,
                    rm.cpu_usage,
                    rm.memory_usage_mb
                FROM fitness_metrics fm
                LEFT JOIN resource_metrics rm ON fm.round_id = rm.round_id
                WHERE fm.agent_id = ?
                ORDER BY fm.timestamp DESC
                LIMIT ?
            """,
                (agent_id, limit),
            )

            rows = cursor.fetchall()
            history = []
            for row in rows:
                history.append(
                    {
                        "fitness_score": row[0],
                        "performance_metrics": json.loads(row[1]) if row[1] else {},
                        "timestamp": row[2],
                        "cpu_usage": row[3],
                        "memory_usage_mb": row[4],
                    }
                )

            return history

        except Exception as e:
            logger.exception(f"Failed to get agent history: {e}")
            return []

    def get_health_status(self) -> dict[str, Any]:
        """Get health status for API endpoint."""
        status = {
            "status": "healthy" if self.running else "stopped",
            "timestamp": datetime.now().isoformat(),
            "database": {
                "connected": self.db_conn is not None,
                "path": self.db_path,
                "current_round": self.current_round_id,
            },
            "redis": {"available": REDIS_AVAILABLE, "connected": False},
            "metrics": {
                "total_collected": self.total_metrics_collected,
                "buffer_size": len(self.metrics_buffer),
                "flush_threshold": self.flush_threshold,
            },
        }

        # Check Redis connection
        if self.redis_client:
            try:
                self.redis_client.ping()
                status["redis"]["connected"] = True
            except Exception as e:
                logger.debug(f"Redis connection check failed: {e}")
                status["redis"]["connected"] = False

        # Check database integrity
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM fitness_metrics")
                count = cursor.fetchone()[0]
                status["database"]["total_records"] = count
            except Exception as e:
                logger.warning(f"Database integrity check failed: {e}")
                status["status"] = "degraded"
                status["database"]["error"] = str(e)

        return status

    def collect_system_metrics(self) -> EvolutionMetricsData:
        """Collect current system resource metrics."""
        metrics = EvolutionMetricsData(agent_id="system", timestamp=time.time())

        if PSUTIL_AVAILABLE:
            try:
                # CPU metrics
                metrics.cpu_percent = psutil.cpu_percent(interval=1)
                metrics.cpu_efficiency = 100 - metrics.cpu_percent

                # Memory metrics
                mem = psutil.virtual_memory()
                metrics.memory_used_mb = mem.used / (1024 * 1024)
                metrics.memory_efficiency = 100 - mem.percent
                metrics.resource_utilization = (metrics.cpu_percent + mem.percent) / 2

                # Network I/O
                net = psutil.net_io_counters()
                metrics.network_io_kb = (net.bytes_sent + net.bytes_recv) / 1024

                # Disk I/O
                disk = psutil.disk_io_counters()
                metrics.disk_io_kb = (disk.read_bytes + disk.write_bytes) / 1024

                # Energy efficiency estimate (simplified)
                metrics.energy_efficiency = metrics.cpu_efficiency * 0.7 + metrics.memory_efficiency * 0.3

            except Exception as e:
                logger.debug(f"Failed to collect system metrics: {e}")

        return metrics


# Singleton instance
_metrics_instance = None


def get_metrics_instance() -> IntegratedEvolutionMetrics:
    """Get or create the singleton metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = IntegratedEvolutionMetrics()
    return _metrics_instance


# Convenience functions
def start_metrics():
    """Start the evolution metrics system."""
    instance = get_metrics_instance()
    instance.start()
    return instance


def stop_metrics() -> None:
    """Stop the evolution metrics system."""
    instance = get_metrics_instance()
    instance.stop()


def record_kpi(agent_id: str, kpi_type: KPIType, value: float, metadata: dict | None = None) -> None:
    """Record a KPI value."""
    instance = get_metrics_instance()
    instance.record_kpi(agent_id, kpi_type, value, metadata)


def get_health_status() -> dict[str, Any]:
    """Get health status for monitoring."""
    instance = get_metrics_instance()
    return instance.get_health_status()
