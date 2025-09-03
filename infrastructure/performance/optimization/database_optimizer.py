"""
AIVillage Database Performance Optimization
Provides connection pooling, query optimization, and performance monitoring
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time
from typing import Any

import aioredis
import asyncpg
from prometheus_client import Counter, Gauge, Histogram
import psutil
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# Performance metrics
DB_QUERY_DURATION = Histogram("db_query_duration_seconds", "Database query duration", ["database", "query_type"])
DB_CONNECTIONS_ACTIVE = Gauge("db_connections_active", "Active database connections", ["database"])
DB_CONNECTIONS_TOTAL = Counter("db_connections_total", "Total database connections", ["database", "status"])
DB_QUERY_ERRORS = Counter("db_query_errors_total", "Database query errors", ["database", "error_type"])


@dataclass
class DatabaseConfig:
    """Database optimization configuration"""

    # PostgreSQL settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "aivillage"
    postgres_user: str = "aivillage"
    postgres_password: str = "aivillage2024"
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 30
    postgres_pool_timeout: int = 30
    postgres_pool_recycle: int = 3600

    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "aivillage2024"
    redis_pool_size: int = 50
    redis_pool_timeout: int = 5

    # Query optimization
    query_timeout: int = 30
    slow_query_threshold: float = 1.0
    enable_query_cache: bool = True
    cache_ttl: int = 300

    # Connection pooling
    pool_pre_ping: bool = True
    pool_reset_on_return: str = "commit"


class PostgreSQLOptimizer:
    """PostgreSQL connection pool and query optimizer"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        self.async_pool = None
        self._query_cache = {}
        self._slow_queries = []
        # Track cache efficiency
        self._cache_hits = 0
        self._cache_misses = 0

    def initialize_sync_pool(self):
        """Initialize synchronous connection pool"""
        connection_string = (
            f"postgresql://{self.config.postgres_user}:{self.config.postgres_password}@"
            f"{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_db}"
        )

        self.engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=self.config.postgres_pool_size,
            max_overflow=self.config.postgres_pool_overflow,
            pool_timeout=self.config.postgres_pool_timeout,
            pool_recycle=self.config.postgres_pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
            pool_reset_on_return=self.config.pool_reset_on_return,
            # Performance optimizations
            connect_args={
                "command_timeout": self.config.query_timeout,
                "server_settings": {
                    "application_name": "aivillage_optimized",
                    "jit": "off",  # Disable JIT for consistent performance
                    "shared_preload_libraries": "pg_stat_statements",
                },
            },
            echo=False,  # Disable SQL logging in production
            future=True,
        )

        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        logger.info("PostgreSQL sync connection pool initialized")

    async def initialize_async_pool(self):
        """Initialize asynchronous connection pool"""
        self.async_pool = await asyncpg.create_pool(
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            database=self.config.postgres_db,
            user=self.config.postgres_user,
            password=self.config.postgres_password,
            min_size=10,
            max_size=self.config.postgres_pool_size,
            timeout=self.config.postgres_pool_timeout,
            command_timeout=self.config.query_timeout,
            # Connection settings for performance
            server_settings={
                "application_name": "aivillage_async",
                "jit": "off",
                "statement_timeout": f"{self.config.query_timeout * 1000}ms",
            },
        )
        logger.info("PostgreSQL async connection pool initialized")

    @asynccontextmanager
    async def get_async_connection(self):
        """Get async database connection with metrics"""
        start_time = time.time()
        connection = None

        try:
            connection = await self.async_pool.acquire()
            DB_CONNECTIONS_ACTIVE.labels(database="postgresql").inc()
            DB_CONNECTIONS_TOTAL.labels(database="postgresql", status="acquired").inc()

            yield connection

        except Exception as e:
            DB_CONNECTIONS_TOTAL.labels(database="postgresql", status="error").inc()
            DB_QUERY_ERRORS.labels(database="postgresql", error_type=type(e).__name__).inc()
            logger.error(f"Database connection error: {e}")
            raise

        finally:
            if connection:
                await self.async_pool.release(connection)
                DB_CONNECTIONS_ACTIVE.labels(database="postgresql").dec()

            duration = time.time() - start_time
            DB_QUERY_DURATION.labels(database="postgresql", query_type="connection").observe(duration)

    async def execute_optimized_query(
        self, query: str, params: dict | None = None, cache_key: str | None = None
    ) -> list[dict]:
        """Execute optimized query with caching and monitoring"""
        start_time = time.time()
        query_type = self._classify_query(query)

        # Check cache first
        if cache_key and self.config.enable_query_cache:
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result

        try:
            async with self.get_async_connection() as conn:
                # Execute query with performance monitoring
                result = await conn.fetch(query, *(params.values() if params else []))

                # Convert to list of dicts
                result_list = [dict(row) for row in result]

                # Cache result if enabled
                if cache_key and self.config.enable_query_cache:
                    self._cache_result(cache_key, result_list)

                duration = time.time() - start_time
                DB_QUERY_DURATION.labels(database="postgresql", query_type=query_type).observe(duration)

                # Track slow queries
                if duration > self.config.slow_query_threshold:
                    self._track_slow_query(query, duration, params)

                return result_list

        except Exception as e:
            duration = time.time() - start_time
            DB_QUERY_ERRORS.labels(database="postgresql", error_type=type(e).__name__).inc()
            logger.error(f"Query execution failed: {e}")
            raise

    def _classify_query(self, query: str) -> str:
        """Classify query type for metrics"""
        query_upper = query.strip().upper()

        if query_upper.startswith("SELECT"):
            return "select"
        elif query_upper.startswith("INSERT"):
            return "insert"
        elif query_upper.startswith("UPDATE"):
            return "update"
        elif query_upper.startswith("DELETE"):
            return "delete"
        else:
            return "other"

    def _get_cached_result(self, cache_key: str) -> list[dict] | None:
        """Get cached query result"""
        if cache_key in self._query_cache:
            cached_data, timestamp = self._query_cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                self._cache_hits += 1
                return cached_data
            del self._query_cache[cache_key]
        self._cache_misses += 1
        return None

    def _cache_result(self, cache_key: str, result: list[dict]):
        """Cache query result"""
        self._query_cache[cache_key] = (result, time.time())

        # Prevent cache from growing too large
        if len(self._query_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self._query_cache.keys(), key=lambda k: self._query_cache[k][1])[:100]
            for key in oldest_keys:
                del self._query_cache[key]

    def _track_slow_query(self, query: str, duration: float, params: dict | None):
        """Track slow queries for optimization"""
        slow_query_entry = {
            "query": query[:500],  # Truncate long queries
            "duration": duration,
            "params": params,
            "timestamp": datetime.utcnow(),
            "count": 1,
        }

        # Check if similar query exists
        for existing in self._slow_queries:
            if existing["query"] == slow_query_entry["query"]:
                existing["count"] += 1
                existing["duration"] = max(existing["duration"], duration)
                return

        self._slow_queries.append(slow_query_entry)

        # Keep only recent slow queries
        if len(self._slow_queries) > 100:
            self._slow_queries.sort(key=lambda x: x["timestamp"], reverse=True)
            self._slow_queries = self._slow_queries[:100]

        logger.warning(f"Slow query detected: {duration:.2f}s - {query[:100]}...")

    def get_optimization_report(self) -> dict[str, Any]:
        """Get database optimization report"""
        return {
            "connection_pool": {
                "size": self.config.postgres_pool_size,
                "active_connections": len(self.async_pool._queue._queue) if self.async_pool else 0,
            },
            "query_cache": {"entries": len(self._query_cache), "hit_rate": self._calculate_cache_hit_rate()},
            "slow_queries": len(self._slow_queries),
            "top_slow_queries": sorted(self._slow_queries, key=lambda x: x["duration"], reverse=True)[:5],
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate based on recorded usage"""
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0


class RedisOptimizer:
    """Redis connection pool and performance optimizer"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None
        self.redis = None

    async def initialize_pool(self):
        """Initialize Redis connection pool"""
        self.pool = aioredis.ConnectionPool.from_url(
            f"redis://:{self.config.redis_password}@{self.config.redis_host}:{self.config.redis_port}",
            max_connections=self.config.redis_pool_size,
            retry_on_timeout=True,
            socket_timeout=self.config.redis_pool_timeout,
            socket_connect_timeout=5,
            health_check_interval=30,
        )

        self.redis = aioredis.Redis(connection_pool=self.pool, decode_responses=True)
        logger.info("Redis connection pool initialized")

    @asynccontextmanager
    async def get_connection(self):
        """Get Redis connection with monitoring"""
        start_time = time.time()

        try:
            DB_CONNECTIONS_ACTIVE.labels(database="redis").inc()
            DB_CONNECTIONS_TOTAL.labels(database="redis", status="acquired").inc()

            yield self.redis

        except Exception as e:
            DB_CONNECTIONS_TOTAL.labels(database="redis", status="error").inc()
            DB_QUERY_ERRORS.labels(database="redis", error_type=type(e).__name__).inc()
            raise

        finally:
            DB_CONNECTIONS_ACTIVE.labels(database="redis").dec()
            duration = time.time() - start_time
            DB_QUERY_DURATION.labels(database="redis", query_type="operation").observe(duration)

    async def optimized_get(self, key: str) -> str | None:
        """Optimized Redis GET operation"""
        async with self.get_connection() as redis:
            return await redis.get(key)

    async def optimized_set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """Optimized Redis SET operation"""
        async with self.get_connection() as redis:
            if ttl:
                return await redis.setex(key, ttl, value)
            else:
                return await redis.set(key, value)

    async def pipeline_operations(self, operations: list[tuple[str, str, Any]]) -> list[Any]:
        """Execute multiple Redis operations in pipeline"""
        async with self.get_connection() as redis:
            pipe = redis.pipeline()

            for operation, key, value in operations:
                if operation == "get":
                    pipe.get(key)
                elif operation == "set":
                    pipe.set(key, value)
                elif operation == "delete":
                    pipe.delete(key)

            return await pipe.execute()


class DatabasePerformanceMonitor:
    """Monitor database performance and suggest optimizations"""

    def __init__(self, postgres: PostgreSQLOptimizer, redis: RedisOptimizer):
        self.postgres = postgres
        self.redis = redis
        self._metrics_history = []

    async def collect_performance_metrics(self) -> dict[str, Any]:
        """Collect comprehensive performance metrics"""
        timestamp = datetime.utcnow()

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Database-specific metrics
        postgres_report = self.postgres.get_optimization_report()

        metrics = {
            "timestamp": timestamp,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
            },
            "postgresql": postgres_report,
            "performance_recommendations": self._generate_recommendations(postgres_report, cpu_percent, memory.percent),
        }

        # Store metrics history
        self._metrics_history.append(metrics)

        # Keep only last 24 hours of metrics
        cutoff_time = timestamp - timedelta(hours=24)
        self._metrics_history = [m for m in self._metrics_history if m["timestamp"] > cutoff_time]

        return metrics

    def _generate_recommendations(self, postgres_report: dict, cpu_percent: float, memory_percent: float) -> list[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        # CPU recommendations
        if cpu_percent > 80:
            recommendations.append("High CPU usage detected. Consider connection pooling optimization.")

        # Memory recommendations
        if memory_percent > 85:
            recommendations.append("High memory usage. Consider reducing connection pool size or query cache size.")

        # PostgreSQL specific
        if postgres_report["slow_queries"] > 10:
            recommendations.append("Multiple slow queries detected. Review query performance and add indexes.")

        if postgres_report["query_cache"]["hit_rate"] < 0.5:
            recommendations.append("Low cache hit rate. Consider optimizing query cache configuration.")

        return recommendations

    def get_performance_trends(self) -> dict[str, Any]:
        """Analyze performance trends over time"""
        if len(self._metrics_history) < 2:
            return {"status": "insufficient_data"}

        recent = self._metrics_history[-1]
        previous = self._metrics_history[-2]

        return {
            "cpu_trend": recent["system"]["cpu_percent"] - previous["system"]["cpu_percent"],
            "memory_trend": recent["system"]["memory_percent"] - previous["system"]["memory_percent"],
            "slow_queries_trend": recent["postgresql"]["slow_queries"] - previous["postgresql"]["slow_queries"],
            "recommendations_count": len(recent["performance_recommendations"]),
        }


# Global database manager
class DatabaseManager:
    """Unified database performance manager"""

    def __init__(self, config: DatabaseConfig | None = None):
        self.config = config or DatabaseConfig()
        self.postgres = PostgreSQLOptimizer(self.config)
        self.redis = RedisOptimizer(self.config)
        self.monitor = DatabasePerformanceMonitor(self.postgres, self.redis)

    async def initialize(self):
        """Initialize all database systems"""
        self.postgres.initialize_sync_pool()
        await self.postgres.initialize_async_pool()
        await self.redis.initialize_pool()

        logger.info("Database performance optimization systems initialized")

    async def health_check(self) -> dict[str, bool]:
        """Check health of all database systems"""
        health = {}

        try:
            async with self.postgres.get_async_connection() as conn:
                await conn.fetchval("SELECT 1")
            health["postgresql"] = True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            health["postgresql"] = False

        try:
            async with self.redis.get_connection() as redis:
                await redis.ping()
            health["redis"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            health["redis"] = False

        return health

    async def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Get comprehensive database performance metrics"""
        return await self.monitor.collect_performance_metrics()


# Global instance
db_manager = DatabaseManager()


async def initialize_database_optimization(config: DatabaseConfig | None = None):
    """Initialize database optimization systems"""
    global db_manager
    db_manager = DatabaseManager(config)
    await db_manager.initialize()
    return db_manager
