"""
AIVillage Unified Linting Cache System
Advanced caching with Redis, Memcached, and in-memory fallbacks
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Available cache backend types"""
    REDIS = "redis"
    MEMCACHED = "memcached"
    MEMORY = "memory"
    FILE = "file"


class CacheStrategy(Enum):
    """Cache strategy types"""
    LRU = "lru"           # Least Recently Used
    TTL = "ttl"           # Time To Live
    ADAPTIVE = "adaptive" # Adaptive based on usage patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[int] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access time and count"""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    backend_type: str = "unknown"
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheKey:
    """Utility class for generating cache keys"""
    
    @staticmethod
    def hash_params(params: Dict[str, Any]) -> str:
        """Generate deterministic hash from parameters"""
        # Serialize parameters in a consistent way
        serialized = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    @staticmethod
    def generate_key(service: str, operation: str, params: Dict[str, Any]) -> str:
        """Generate a standardized cache key"""
        param_hash = CacheKey.hash_params(params)
        return f"{service}:{operation}:{param_hash}"
    
    @staticmethod
    def linting_key(tool: str, target_paths: List[str], config_hash: str) -> str:
        """Generate cache key for linting operations"""
        params = {
            "tool": tool,
            "paths": sorted(target_paths),
            "config_hash": config_hash
        }
        return CacheKey.generate_key("linting", tool, params)


class CacheBackendInterface(ABC):
    """Abstract interface for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        pass


class RedisCacheBackend(CacheBackendInterface):
    """Redis-based cache backend with advanced features"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 password: str = None, db: int = 0, connection_pool_size: int = 10):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.connection_pool_size = connection_pool_size
        self.client = None
        self.stats = CacheStats(backend_type="redis")
        
    async def connect(self):
        """Initialize Redis connection"""
        try:
            import redis.asyncio as redis
            
            # Create connection pool
            pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                max_connections=self.connection_pool_size,
                decode_responses=False  # We'll handle encoding/decoding
            )
            
            self.client = redis.Redis(connection_pool=pool)
            
            # Test connection
            await self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
            
        except ImportError:
            logger.error("Redis package not available. Install with: pip install redis")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            if not self.client:
                return None
                
            data = await self.client.get(key)
            if data is None:
                self.stats.misses += 1
                return None
            
            # Deserialize data
            value = pickle.loads(data)
            self.stats.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis"""
        try:
            if not self.client:
                return False
            
            # Serialize data
            data = pickle.dumps(value)
            
            # Set with TTL if specified
            if ttl:
                await self.client.setex(key, ttl, data)
            else:
                await self.client.set(key, data)
            
            self.stats.sets += 1
            self.stats.total_size_bytes += len(data)
            return True
            
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis"""
        try:
            if not self.client:
                return False
                
            result = await self.client.delete(key)
            if result > 0:
                self.stats.deletes += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear Redis database"""
        try:
            if not self.client:
                return False
                
            await self.client.flushdb()
            logger.info("Redis cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            if not self.client:
                return False
                
            result = await self.client.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def get_stats(self) -> CacheStats:
        """Get Redis cache statistics"""
        try:
            if not self.client:
                return self.stats
            
            # Get Redis info
            info = await self.client.info()
            redis_stats = info.get('stats', {})
            
            # Update stats from Redis
            self.stats.hits = redis_stats.get('keyspace_hits', self.stats.hits)
            self.stats.misses = redis_stats.get('keyspace_misses', self.stats.misses)
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return self.stats


class MemcachedCacheBackend(CacheBackendInterface):
    """Memcached-based cache backend"""
    
    def __init__(self, servers: List[str] = None):
        self.servers = servers or ["127.0.0.1:11211"]
        self.client = None
        self.stats = CacheStats(backend_type="memcached")
    
    async def connect(self):
        """Initialize Memcached connection"""
        try:
            import memcache
            
            # Create Memcached client
            self.client = memcache.Client(self.servers, debug=False)
            
            # Test connection
            stats = self.client.get_stats()
            if not stats:
                raise Exception("No Memcached servers responding")
                
            logger.info(f"Connected to Memcached servers: {self.servers}")
            return True
            
        except ImportError:
            logger.error("Memcached package not available. Install with: pip install python-memcached")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Memcached: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Memcached"""
        try:
            if not self.client:
                return None
            
            # Memcached operations are synchronous, run in thread pool
            value = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get, key
            )
            
            if value is None:
                self.stats.misses += 1
                return None
            
            self.stats.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Memcached get error for key {key}: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Memcached"""
        try:
            if not self.client:
                return False
            
            # Memcached operations are synchronous
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.client.set, key, value, ttl or 0
            )
            
            if result:
                self.stats.sets += 1
                # Estimate size (Memcached doesn't provide exact size)
                self.stats.total_size_bytes += len(pickle.dumps(value))
            
            return result
            
        except Exception as e:
            logger.error(f"Memcached set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Memcached"""
        try:
            if not self.client:
                return False
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.client.delete, key
            )
            
            if result:
                self.stats.deletes += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Memcached delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear Memcached"""
        try:
            if not self.client:
                return False
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.client.flush_all
            )
            
            logger.info("Memcached cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Memcached clear error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Memcached"""
        try:
            value = await self.get(key)
            return value is not None
        except Exception:
            return False
    
    async def get_stats(self) -> CacheStats:
        """Get Memcached statistics"""
        try:
            if not self.client:
                return self.stats
            
            # Get Memcached stats
            stats = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_stats
            )
            
            if stats:
                # Parse stats from first server
                server_stats = stats[0][1]
                self.stats.hits = int(server_stats.get('get_hits', 0))
                self.stats.misses = int(server_stats.get('get_misses', 0))
                self.stats.total_size_bytes = int(server_stats.get('bytes', 0))
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Memcached stats error: {e}")
            return self.stats


class InMemoryCacheBackend(CacheBackendInterface):
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats(backend_type="memory")
        self._lock = asyncio.Lock()
    
    async def connect(self):
        """Initialize in-memory cache (always succeeds)"""
        logger.info(f"In-memory cache initialized with max_size={self.max_size}")
        return True
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        async with self._lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.stats.misses += 1
                return None
            
            if entry.is_expired():
                del self.cache[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
            
            entry.touch()
            self.stats.hits += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache"""
        async with self._lock:
            # Check if we need to evict entries
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()
            
            # Calculate size
            size_bytes = len(pickle.dumps(value))
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=1,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self.stats.sets += 1
            self.stats.total_size_bytes += size_bytes
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache"""
        async with self._lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.stats.deletes += 1
                self.stats.total_size_bytes -= entry.size_bytes
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear memory cache"""
        async with self._lock:
            self.cache.clear()
            self.stats.total_size_bytes = 0
            logger.info("In-memory cache cleared")
            return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache"""
        async with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                return False
            
            if entry.is_expired():
                del self.cache[key]
                self.stats.evictions += 1
                return False
            
            return True
    
    async def get_stats(self) -> CacheStats:
        """Get memory cache statistics"""
        async with self._lock:
            # Clean up expired entries first
            await self._cleanup_expired()
            
            self.stats.total_size_bytes = sum(entry.size_bytes for entry in self.cache.values())
            return self.stats
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        # Find LRU entry
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].accessed_at)
        entry = self.cache.pop(lru_key)
        
        self.stats.evictions += 1
        self.stats.total_size_bytes -= entry.size_bytes
        
        logger.debug(f"Evicted LRU entry: {lru_key}")
    
    async def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self.cache.pop(key)
            self.stats.evictions += 1
            self.stats.total_size_bytes -= entry.size_bytes


class UnifiedCacheManager:
    """
    Unified cache manager with multiple backend support and intelligent fallback
    """
    
    def __init__(self, 
                 preferred_backends: List[CacheBackend] = None,
                 redis_config: Dict[str, Any] = None,
                 memcached_config: Dict[str, Any] = None,
                 memory_config: Dict[str, Any] = None):
        
        self.preferred_backends = preferred_backends or [
            CacheBackend.REDIS, CacheBackend.MEMCACHED, CacheBackend.MEMORY
        ]
        
        self.redis_config = redis_config or {}
        self.memcached_config = memcached_config or {}
        self.memory_config = memory_config or {"max_size": 1000, "default_ttl": 3600}
        
        self.active_backend: Optional[CacheBackendInterface] = None
        self.backend_type: Optional[CacheBackend] = None
        self.fallback_chain: List[CacheBackendInterface] = []
        
        # Performance monitoring
        self.operation_times: List[float] = []
        self.max_operation_history = 1000
    
    async def initialize(self) -> bool:
        """Initialize cache manager with the best available backend"""
        
        logger.info(f"Initializing cache manager with preferred backends: {self.preferred_backends}")
        
        # Try each backend in order of preference
        for backend_type in self.preferred_backends:
            success = await self._try_initialize_backend(backend_type)
            if success:
                self.backend_type = backend_type
                logger.info(f"Successfully initialized cache with {backend_type.value} backend")
                return True
        
        # If all else fails, use in-memory cache
        logger.warning("All preferred backends failed, falling back to in-memory cache")
        return await self._try_initialize_backend(CacheBackend.MEMORY)
    
    async def _try_initialize_backend(self, backend_type: CacheBackend) -> bool:
        """Try to initialize a specific backend"""
        
        try:
            if backend_type == CacheBackend.REDIS:
                backend = RedisCacheBackend(**self.redis_config)
            elif backend_type == CacheBackend.MEMCACHED:
                backend = MemcachedCacheBackend(**self.memcached_config)
            elif backend_type == CacheBackend.MEMORY:
                backend = InMemoryCacheBackend(**self.memory_config)
            else:
                return False
            
            success = await backend.connect()
            if success:
                self.active_backend = backend
                return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {backend_type.value} backend: {e}")
        
        return False
    
    async def get(self, key: str, service: str = "default") -> Optional[Any]:
        """Get value from cache"""
        if not self.active_backend:
            logger.warning("No active cache backend")
            return None
        
        full_key = f"{service}:{key}"
        start_time = time.time()
        
        try:
            value = await self.active_backend.get(full_key)
            self._record_operation_time(time.time() - start_time)
            return value
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, service: str = "default") -> bool:
        """Set value in cache"""
        if not self.active_backend:
            logger.warning("No active cache backend")
            return False
        
        full_key = f"{service}:{key}"
        start_time = time.time()
        
        try:
            success = await self.active_backend.set(full_key, value, ttl)
            self._record_operation_time(time.time() - start_time)
            return success
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str, service: str = "default") -> bool:
        """Delete value from cache"""
        if not self.active_backend:
            logger.warning("No active cache backend")
            return False
        
        full_key = f"{service}:{key}"
        
        try:
            return await self.active_backend.delete(full_key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self, service: str = None) -> bool:
        """Clear cache (optionally for specific service)"""
        if not self.active_backend:
            return False
        
        try:
            if service:
                # Clear only keys for specific service (implementation depends on backend)
                logger.warning(f"Service-specific clear feature disabled for {self.backend_type}")
                return False
            else:
                return await self.active_backend.clear()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def exists(self, key: str, service: str = "default") -> bool:
        """Check if key exists in cache"""
        if not self.active_backend:
            return False
        
        full_key = f"{service}:{key}"
        
        try:
            return await self.active_backend.exists(full_key)
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if not self.active_backend:
            return {"error": "No active backend"}
        
        try:
            backend_stats = await self.active_backend.get_stats()
            
            # Calculate performance metrics
            avg_operation_time = (
                sum(self.operation_times) / len(self.operation_times) 
                if self.operation_times else 0
            )
            
            return {
                "backend_type": self.backend_type.value if self.backend_type else "unknown",
                "backend_stats": asdict(backend_stats),
                "performance": {
                    "average_operation_time_ms": avg_operation_time * 1000,
                    "total_operations": len(self.operation_times)
                },
                "configuration": {
                    "preferred_backends": [b.value for b in self.preferred_backends],
                    "active_backend": self.backend_type.value if self.backend_type else None
                }
            }
            
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"error": str(e)}
    
    def _record_operation_time(self, operation_time: float):
        """Record operation time for performance monitoring"""
        self.operation_times.append(operation_time)
        
        # Keep only recent operations
        if len(self.operation_times) > self.max_operation_history:
            self.operation_times = self.operation_times[-self.max_operation_history:]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        if not self.active_backend:
            return {
                "status": "unhealthy",
                "error": "No active backend",
                "backend_type": None
            }
        
        try:
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            test_value = {"timestamp": time.time(), "test": True}
            
            # Test set operation
            set_success = await self.set(test_key, test_value, ttl=60, service="health_check")
            if not set_success:
                return {
                    "status": "unhealthy",
                    "error": "Cache set operation failed",
                    "backend_type": self.backend_type.value
                }
            
            # Test get operation
            retrieved_value = await self.get(test_key, service="health_check")
            if retrieved_value != test_value:
                return {
                    "status": "unhealthy",
                    "error": "Cache get operation failed",
                    "backend_type": self.backend_type.value
                }
            
            # Test delete operation
            delete_success = await self.delete(test_key, service="health_check")
            
            # Get backend stats
            stats = await self.get_stats()
            
            return {
                "status": "healthy",
                "backend_type": self.backend_type.value,
                "operations_test": {
                    "set": set_success,
                    "get": retrieved_value == test_value,
                    "delete": delete_success
                },
                "performance": stats.get("performance", {}),
                "backend_stats": stats.get("backend_stats", {})
            }
            
        except Exception as e:
            logger.error(f"Cache health check error: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "backend_type": self.backend_type.value if self.backend_type else None
            }


# Global cache manager instance
cache_manager = UnifiedCacheManager()


# Convenience functions for linting operations
class LintingCacheService:
    """Specialized caching service for linting operations"""
    
    def __init__(self, cache_manager: UnifiedCacheManager):
        self.cache_manager = cache_manager
        self.service_name = "linting"
        self.default_ttl = 3600  # 1 hour
    
    async def get_linting_result(self, tool: str, target_paths: List[str], 
                               config_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached linting result"""
        key = CacheKey.linting_key(tool, target_paths, config_hash)
        return await self.cache_manager.get(key, self.service_name)
    
    async def set_linting_result(self, tool: str, target_paths: List[str], 
                               config_hash: str, result: Dict[str, Any], 
                               ttl: Optional[int] = None) -> bool:
        """Cache linting result"""
        key = CacheKey.linting_key(tool, target_paths, config_hash)
        return await self.cache_manager.set(
            key, result, ttl or self.default_ttl, self.service_name
        )
    
    async def invalidate_tool_cache(self, tool: str) -> bool:
        """Invalidate all cache entries for a specific tool"""
        # This is a simplified implementation
        # In production, would need pattern matching support
        logger.info(f"Cache invalidation for tool {tool} requested (feature disabled)")
        return True


# Global linting cache service
linting_cache = LintingCacheService(cache_manager)


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize cache manager
        success = await cache_manager.initialize()
        print(f"Cache initialization: {success}")
        
        if success:
            # Test basic operations
            await cache_manager.set("test_key", {"data": "test_value"}, ttl=60)
            value = await cache_manager.get("test_key")
            print(f"Retrieved value: {value}")
            
            # Test linting cache
            result = await linting_cache.get_linting_result(
                "ruff", ["src/"], "config_hash_123"
            )
            print(f"Linting cache result: {result}")
            
            # Health check
            health = await cache_manager.health_check()
            print(f"Cache health: {health}")
            
            # Statistics
            stats = await cache_manager.get_stats()
            print(f"Cache stats: {stats}")
    
    asyncio.run(main())