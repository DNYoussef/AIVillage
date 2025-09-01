"""
AIVillage Performance Caching Manager
Provides optimized caching infrastructure for all services
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import logging
import time
from typing import Any

import memcache
from prometheus_client import Counter, Gauge, Histogram
import redis
from redis.exceptions import ConnectionError, RedisError
import redis.sentinel

logger = logging.getLogger(__name__)

# Metrics for cache performance
CACHE_OPERATIONS = Counter("cache_operations_total", "Total cache operations", ["operation", "cache_type", "service"])
CACHE_LATENCY = Histogram("cache_operation_duration_seconds", "Cache operation duration", ["operation", "cache_type"])
CACHE_HIT_RATE = Gauge("cache_hit_rate", "Cache hit rate by service", ["service", "cache_type"])
CACHE_MEMORY_USAGE = Gauge("cache_memory_usage_bytes", "Cache memory usage", ["cache_type", "instance"])


@dataclass
class CacheConfig:
    """Cache configuration settings"""

    redis_sentinels: list[tuple] = None
    redis_master_name: str = "mymaster"
    redis_password: str = "aivillage2024"
    memcached_servers: list[str] = None
    default_ttl: int = 3600
    max_retries: int = 3
    retry_delay: float = 0.1

    def __post_init__(self):
        if self.redis_sentinels is None:
            self.redis_sentinels = [("localhost", 26379), ("localhost", 26380), ("localhost", 26381)]
        if self.memcached_servers is None:
            self.memcached_servers = ["localhost:11211"]


class CacheKey:
    """Standardized cache key generation"""

    @staticmethod
    def agent_forge_key(phase: str, model: str, params_hash: str) -> str:
        """Generate cache key for Agent Forge pipeline results"""
        return f"af:phase:{phase}:model:{model}:params:{params_hash}"

    @staticmethod
    def hyperrag_key(query_type: str, query_hash: str, collection: str = "") -> str:
        """Generate cache key for HyperRAG query results"""
        base = f"hyperrag:query:{query_type}:hash:{query_hash}"
        return f"{base}:collection:{collection}" if collection else base

    @staticmethod
    def p2p_key(message_type: str, peer_id: str, content_hash: str) -> str:
        """Generate cache key for P2P message routing"""
        return f"p2p:msg:{message_type}:peer:{peer_id}:hash:{content_hash}"

    @staticmethod
    def api_key(endpoint: str, params_hash: str, user_id: str = "") -> str:
        """Generate cache key for API responses"""
        base = f"api:endpoint:{endpoint}:params:{params_hash}"
        return f"{base}:user:{user_id}" if user_id else base

    @staticmethod
    def edge_key(device_id: str, task_type: str, resource_hash: str) -> str:
        """Generate cache key for edge computing tasks"""
        return f"edge:device:{device_id}:task:{task_type}:resource:{resource_hash}"

    @staticmethod
    def hash_params(params: dict[str, Any]) -> str:
        """Create deterministic hash of parameters"""
        serialized = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class RedisCacheManager:
    """High-performance Redis cache manager with Sentinel support"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.sentinel = redis.sentinel.Sentinel(
            config.redis_sentinels, password=config.redis_password, socket_timeout=1.0, socket_connect_timeout=1.0
        )
        self._master = None
        self._replica = None
        self._stats = {"hits": 0, "misses": 0, "errors": 0}

    def _get_master(self) -> redis.Redis:
        """Get Redis master connection with automatic failover"""
        if self._master is None:
            try:
                self._master = self.sentinel.master_for(
                    self.config.redis_master_name,
                    password=self.config.redis_password,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0,
                    db=0,
                )
            except Exception as e:
                logger.error(f"Failed to connect to Redis master: {e}")
                raise ConnectionError(f"Redis master unavailable: {e}")
        return self._master

    def _get_replica(self) -> redis.Redis:
        """Get Redis replica for read operations"""
        if self._replica is None:
            try:
                self._replica = self.sentinel.slave_for(
                    self.config.redis_master_name,
                    password=self.config.redis_password,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0,
                    db=0,
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis replica, using master: {e}")
                return self._get_master()
        return self._replica

    async def get(self, key: str, use_replica: bool = True) -> Any | None:
        """Get value from cache with automatic retry"""
        start_time = time.time()

        for attempt in range(self.config.max_retries):
            try:
                client = self._get_replica() if use_replica else self._get_master()
                value = client.get(key)

                # Record metrics
                CACHE_LATENCY.labels(operation="get", cache_type="redis").observe(time.time() - start_time)

                if value is not None:
                    self._stats["hits"] += 1
                    CACHE_OPERATIONS.labels(operation="hit", cache_type="redis", service="unknown").inc()

                    try:
                        return json.loads(value.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        return value.decode("utf-8")
                else:
                    self._stats["misses"] += 1
                    CACHE_OPERATIONS.labels(operation="miss", cache_type="redis", service="unknown").inc()
                    return None

            except (RedisError, ConnectionError) as e:
                self._stats["errors"] += 1
                logger.warning(f"Redis get error (attempt {attempt + 1}): {e}")

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))
                    # Reset connections on error
                    self._master = None
                    self._replica = None
                else:
                    CACHE_OPERATIONS.labels(operation="error", cache_type="redis", service="unknown").inc()
                    return None

        return None

    async def set(self, key: str, value: Any, ttl: int | None = None, service: str = "unknown") -> bool:
        """Set value in cache with TTL"""
        start_time = time.time()
        ttl = ttl or self.config.default_ttl

        for attempt in range(self.config.max_retries):
            try:
                client = self._get_master()

                # Serialize value
                if isinstance(value, dict | list):
                    serialized_value = json.dumps(value, separators=(",", ":"))
                else:
                    serialized_value = str(value)

                result = client.setex(key, ttl, serialized_value)

                # Record metrics
                CACHE_LATENCY.labels(operation="set", cache_type="redis").observe(time.time() - start_time)
                CACHE_OPERATIONS.labels(operation="set", cache_type="redis", service=service).inc()

                return result

            except (RedisError, ConnectionError) as e:
                self._stats["errors"] += 1
                logger.warning(f"Redis set error (attempt {attempt + 1}): {e}")

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))
                    self._master = None
                else:
                    CACHE_OPERATIONS.labels(operation="error", cache_type="redis", service=service).inc()
                    return False

        return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            client = self._get_master()
            result = client.delete(key)
            CACHE_OPERATIONS.labels(operation="delete", cache_type="redis", service="unknown").inc()
            return bool(result)
        except (RedisError, ConnectionError) as e:
            logger.warning(f"Redis delete error: {e}")
            CACHE_OPERATIONS.labels(operation="error", cache_type="redis", service="unknown").inc()
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_ops = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total_ops) if total_ops > 0 else 0.0

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "errors": self._stats["errors"],
            "hit_rate": hit_rate,
            "total_operations": total_ops,
        }


class MemcachedManager:
    """Memcached manager for HyperRAG query caching"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.client = memcache.Client(config.memcached_servers, debug=0)
        self._stats = {"hits": 0, "misses": 0, "errors": 0}

    async def get(self, key: str) -> Any | None:
        """Get value from Memcached"""
        start_time = time.time()

        try:
            value = self.client.get(key)
            CACHE_LATENCY.labels(operation="get", cache_type="memcached").observe(time.time() - start_time)

            if value is not None:
                self._stats["hits"] += 1
                CACHE_OPERATIONS.labels(operation="hit", cache_type="memcached", service="hyperrag").inc()
                return value
            else:
                self._stats["misses"] += 1
                CACHE_OPERATIONS.labels(operation="miss", cache_type="memcached", service="hyperrag").inc()
                return None

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Memcached get error: {e}")
            CACHE_OPERATIONS.labels(operation="error", cache_type="memcached", service="hyperrag").inc()
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in Memcached"""
        start_time = time.time()
        ttl = ttl or self.config.default_ttl

        try:
            result = self.client.set(key, value, time=ttl)
            CACHE_LATENCY.labels(operation="set", cache_type="memcached").observe(time.time() - start_time)
            CACHE_OPERATIONS.labels(operation="set", cache_type="memcached", service="hyperrag").inc()
            return result

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Memcached set error: {e}")
            CACHE_OPERATIONS.labels(operation="error", cache_type="memcached", service="hyperrag").inc()
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get Memcached statistics"""
        total_ops = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total_ops) if total_ops > 0 else 0.0

        stats = self.client.get_stats()

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "errors": self._stats["errors"],
            "hit_rate": hit_rate,
            "memcached_stats": stats,
        }


class UnifiedCacheManager:
    """Unified cache manager for all AIVillage services"""

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self.redis = RedisCacheManager(self.config)
        self.memcached = MemcachedManager(self.config)

        # Service-specific cache assignments
        self.service_cache_map = {
            "agent_forge": self.redis,  # High-performance structured caching
            "hyperrag": self.memcached,  # Fast query result caching
            "p2p-mesh": self.redis,  # Message routing and peer data
            "api-gateway": self.redis,  # API response caching
            "edge-computing": self.redis,  # Edge task and resource caching
        }

    async def get(self, key: str, service: str = "default") -> Any | None:
        """Get value using appropriate cache for service"""
        cache_manager = self.service_cache_map.get(service, self.redis)
        return await cache_manager.get(key)

    async def set(self, key: str, value: Any, ttl: int | None = None, service: str = "default") -> bool:
        """Set value using appropriate cache for service"""
        cache_manager = self.service_cache_map.get(service, self.redis)
        return await cache_manager.set(key, value, ttl, service)

    async def delete(self, key: str, service: str = "default") -> bool:
        """Delete key using appropriate cache"""
        cache_manager = self.service_cache_map.get(service, self.redis)
        if hasattr(cache_manager, "delete"):
            return await cache_manager.delete(key)
        return False

    @asynccontextmanager
    async def cached_operation(self, key: str, ttl: int | None = None, service: str = "default"):
        """Context manager for cached operations"""
        # Try to get cached result first
        cached_result = await self.get(key, service)
        if cached_result is not None:
            yield cached_result, True  # (result, was_cached)
            return

        # Provide mechanism to cache result
        result_container = {"result": None}

        def cache_result(result: Any):
            result_container["result"] = result

        yield cache_result, False  # (cache_function, was_cached)

        # Cache the result if one was provided
        if result_container["result"] is not None:
            await self.set(key, result_container["result"], ttl, service)

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get statistics from all cache systems"""
        return {
            "redis": self.redis.get_stats(),
            "memcached": self.memcached.get_stats(),
            "timestamp": datetime.utcnow().isoformat(),
            "config": asdict(self.config),
        }

    async def health_check(self) -> dict[str, bool]:
        """Check health of all cache systems"""
        health = {}

        # Test Redis
        try:
            await self.redis.set("health_check", "ok", ttl=10)
            result = await self.redis.get("health_check")
            health["redis"] = result == "ok"
            await self.redis.delete("health_check")
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            health["redis"] = False

        # Test Memcached
        try:
            await self.memcached.set("health_check", "ok", ttl=10)
            result = await self.memcached.get("health_check")
            health["memcached"] = result == "ok"
        except Exception as e:
            logger.error(f"Memcached health check failed: {e}")
            health["memcached"] = False

        return health


# Global cache manager instance
cache_manager = UnifiedCacheManager()


# Convenience functions for each service
async def cache_agent_forge_result(
    phase: str, model: str, params: dict[str, Any], result: Any, ttl: int = 1800
) -> bool:
    """Cache Agent Forge pipeline phase result"""
    params_hash = CacheKey.hash_params(params)
    key = CacheKey.agent_forge_key(phase, model, params_hash)
    return await cache_manager.set(key, result, ttl, "agent_forge")


async def get_cached_agent_forge_result(phase: str, model: str, params: dict[str, Any]) -> Any | None:
    """Get cached Agent Forge pipeline result"""
    params_hash = CacheKey.hash_params(params)
    key = CacheKey.agent_forge_key(phase, model, params_hash)
    return await cache_manager.get(key, "agent_forge")


async def cache_hyperrag_query(query_type: str, query: str, collection: str, result: Any, ttl: int = 600) -> bool:
    """Cache HyperRAG query result"""
    query_hash = CacheKey.hash_params({"query": query})
    key = CacheKey.hyperrag_key(query_type, query_hash, collection)
    return await cache_manager.set(key, result, ttl, "hyperrag")


async def get_cached_hyperrag_query(query_type: str, query: str, collection: str = "") -> Any | None:
    """Get cached HyperRAG query result"""
    query_hash = CacheKey.hash_params({"query": query})
    key = CacheKey.hyperrag_key(query_type, query_hash, collection)
    return await cache_manager.get(key, "hyperrag")


# Initialize cache manager
def initialize_cache_manager(config: CacheConfig | None = None):
    """Initialize global cache manager"""
    global cache_manager
    cache_manager = UnifiedCacheManager(config)
    logger.info("Cache manager initialized for AIVillage performance optimization")
    return cache_manager
