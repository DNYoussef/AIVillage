"""Redis connection manager with comprehensive fallback support.

This module provides Redis connection management with automatic fallbacks to:
- SQLite for persistence
- In-memory storage for caching
- File-based storage for queue operations
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import redis
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    aioredis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Redis connection configuration."""

    url: str
    db: int = 0
    max_connections: int = 10
    retry_on_timeout: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_attempts: int = 3
    retry_delay: float = 1.0
    fallback_enabled: bool = True
    fallback_storage: str = "sqlite"  # sqlite, memory, file


class RedisFallbackStorage:
    """Fallback storage implementation for Redis operations."""

    def __init__(self, storage_type: str, storage_path: str = "./data/redis_fallback"):
        self.storage_type = storage_type
        self.storage_path = Path(storage_path)
        self._memory_store: dict[str, Any] = {}
        self._sqlite_conn: sqlite3.Connection | None = None

        if storage_type == "sqlite":
            self._init_sqlite()
        elif storage_type == "file":
            self.storage_path.mkdir(parents=True, exist_ok=True)

    def _init_sqlite(self):
        """Initialize SQLite fallback database."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        db_path = self.storage_path.parent / "redis_fallback.db"

        self._sqlite_conn = sqlite3.connect(str(db_path), timeout=30)
        self._sqlite_conn.execute("PRAGMA journal_mode=WAL")

        # Create fallback tables
        self._sqlite_conn.executescript(
            """
        CREATE TABLE IF NOT EXISTS redis_fallback (
            key TEXT PRIMARY KEY,
            value BLOB NOT NULL,
            value_type TEXT NOT NULL,
            expires_at REAL NULL,
            created_at REAL DEFAULT (datetime('now','unixepoch'))
        );

        CREATE INDEX IF NOT EXISTS idx_redis_fallback_expires ON redis_fallback(expires_at);

        CREATE TABLE IF NOT EXISTS redis_lists (
            key TEXT NOT NULL,
            index_pos INTEGER NOT NULL,
            value BLOB NOT NULL,
            created_at REAL DEFAULT (datetime('now','unixepoch')),
            PRIMARY KEY (key, index_pos)
        );

        CREATE TABLE IF NOT EXISTS redis_sets (
            key TEXT NOT NULL,
            member BLOB NOT NULL,
            created_at REAL DEFAULT (datetime('now','unixepoch')),
            PRIMARY KEY (key, member)
        );

        CREATE TABLE IF NOT EXISTS redis_hashes (
            key TEXT NOT NULL,
            field TEXT NOT NULL,
            value BLOB NOT NULL,
            created_at REAL DEFAULT (datetime('now','unixepoch')),
            PRIMARY KEY (key, field)
        );
        """
        )
        self._sqlite_conn.commit()

    async def get(self, key: str) -> Any | None:
        """Get value from fallback storage."""
        if self.storage_type == "memory":
            return self._memory_store.get(key)

        if self.storage_type == "sqlite" and self._sqlite_conn:
            cursor = self._sqlite_conn.cursor()
            cursor.execute(
                """
            SELECT value, value_type, expires_at
            FROM redis_fallback
            WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
            """,
                (key, time.time()),
            )

            row = cursor.fetchone()
            if row:
                value_blob, value_type, expires_at = row
                if value_type == "json":
                    return json.loads(value_blob.decode())
                if value_type == "pickle":
                    return pickle.loads(value_blob)
                return value_blob.decode()
            return None

        if self.storage_type == "file":
            file_path = self.storage_path / f"{key}.json"
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        data = json.load(f)

                    # Check expiration
                    if "expires_at" in data and data["expires_at"] < time.time():
                        file_path.unlink()
                        return None

                    return data.get("value")
                except Exception as e:
                    logger.warning(f"Failed to read fallback file {file_path}: {e}")
                    return None

        return None

    async def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        """Set value in fallback storage."""
        expires_at = time.time() + ex if ex else None

        if self.storage_type == "memory":
            self._memory_store[key] = value
            # TODO: Implement expiration for memory storage
            return True

        if self.storage_type == "sqlite" and self._sqlite_conn:
            try:
                # Determine value type and serialize
                if isinstance(value, (dict, list)):
                    value_blob = json.dumps(value).encode()
                    value_type = "json"
                elif isinstance(value, (str, int, float, bool)):
                    value_blob = str(value).encode()
                    value_type = "string"
                else:
                    value_blob = pickle.dumps(value)
                    value_type = "pickle"

                self._sqlite_conn.execute(
                    """
                INSERT OR REPLACE INTO redis_fallback (key, value, value_type, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                    (key, value_blob, value_type, expires_at),
                )

                self._sqlite_conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to set fallback value for {key}: {e}")
                return False

        elif self.storage_type == "file":
            try:
                file_path = self.storage_path / f"{key}.json"
                data = {
                    "value": value,
                    "created_at": time.time(),
                    "expires_at": expires_at,
                }

                with open(file_path, "w") as f:
                    json.dump(data, f, default=str)

                return True
            except Exception as e:
                logger.error(f"Failed to write fallback file for {key}: {e}")
                return False

        return False

    async def delete(self, key: str) -> bool:
        """Delete value from fallback storage."""
        if self.storage_type == "memory":
            return self._memory_store.pop(key, None) is not None

        if self.storage_type == "sqlite" and self._sqlite_conn:
            cursor = self._sqlite_conn.cursor()
            cursor.execute("DELETE FROM redis_fallback WHERE key = ?", (key,))
            self._sqlite_conn.commit()
            return cursor.rowcount > 0

        if self.storage_type == "file":
            file_path = self.storage_path / f"{key}.json"
            if file_path.exists():
                file_path.unlink()
                return True
            return False

        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in fallback storage."""
        value = await self.get(key)
        return value is not None

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key."""
        if self.storage_type == "sqlite" and self._sqlite_conn:
            expires_at = time.time() + seconds
            cursor = self._sqlite_conn.cursor()
            cursor.execute(
                """
            UPDATE redis_fallback
            SET expires_at = ?
            WHERE key = ?
            """,
                (expires_at, key),
            )
            self._sqlite_conn.commit()
            return cursor.rowcount > 0

        # For other storage types, would need to implement expiration logic
        return False

    async def cleanup_expired(self) -> int:
        """Clean up expired keys."""
        if self.storage_type == "sqlite" and self._sqlite_conn:
            cursor = self._sqlite_conn.cursor()
            cursor.execute(
                """
            DELETE FROM redis_fallback
            WHERE expires_at IS NOT NULL AND expires_at < ?
            """,
                (time.time(),),
            )
            self._sqlite_conn.commit()
            return cursor.rowcount

        if self.storage_type == "file":
            expired_count = 0
            for file_path in self.storage_path.glob("*.json"):
                try:
                    with open(file_path) as f:
                        data = json.load(f)

                    if "expires_at" in data and data["expires_at"] < time.time():
                        file_path.unlink()
                        expired_count += 1
                except Exception:
                    pass  # Skip corrupted files

            return expired_count

        return 0

    def close(self):
        """Close fallback storage connections."""
        if self._sqlite_conn:
            self._sqlite_conn.close()


class RedisManager:
    """Redis connection manager with fallback support."""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.pools: dict[str, Any] = {}
        self.fallback_stores: dict[str, RedisFallbackStorage] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Redis connections and fallback storage."""
        if self._initialized:
            return

        logger.info("Initializing Redis manager...")

        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using fallbacks only")
            await self._init_fallback_stores()
            self._initialized = True
            return

        # Initialize Redis connection pools
        await self._init_redis_pools()

        # Initialize fallback stores
        await self._init_fallback_stores()

        self._initialized = True
        logger.info("Redis manager initialized successfully")

    async def _init_redis_pools(self):
        """Initialize Redis connection pools."""
        if not self.config_manager:
            logger.warning("No config manager provided, skipping Redis initialization")
            return

        redis_configs = {
            "evolution_metrics": RedisConfig(
                url=self.config_manager.get(
                    "AIVILLAGE_REDIS_URL", "redis://localhost:6379/0"
                ),
                db=0,
                fallback_storage="sqlite",
            ),
            "rag_cache": RedisConfig(
                url=self.config_manager.get(
                    "RAG_REDIS_URL", "redis://localhost:6379/1"
                ),
                db=1,
                fallback_storage="memory",
            ),
            "p2p_discovery": RedisConfig(
                url="redis://localhost:6379/2", db=2, fallback_storage="file"
            ),
            "session_store": RedisConfig(
                url="redis://localhost:6379/3", db=3, fallback_storage="sqlite"
            ),
        }

        for pool_name, config in redis_configs.items():
            try:
                # Create connection pool
                pool = aioredis.ConnectionPool.from_url(
                    config.url,
                    max_connections=config.max_connections,
                    retry_on_timeout=config.retry_on_timeout,
                    socket_timeout=config.socket_timeout,
                    socket_connect_timeout=config.socket_connect_timeout,
                )

                # Test connection
                redis_client = aioredis.Redis(connection_pool=pool)
                await redis_client.ping()
                await redis_client.close()

                self.pools[pool_name] = {
                    "pool": pool,
                    "config": config,
                    "available": True,
                }

                logger.info(f"Redis pool {pool_name} initialized successfully")

            except Exception as e:
                logger.warning(f"Failed to initialize Redis pool {pool_name}: {e}")
                self.pools[pool_name] = {
                    "pool": None,
                    "config": config,
                    "available": False,
                }

    async def _init_fallback_stores(self):
        """Initialize fallback storage systems."""
        fallback_configs = {
            "evolution_metrics": ("sqlite", "./data/fallback/evolution_metrics"),
            "rag_cache": ("memory", "./data/fallback/rag_cache"),
            "p2p_discovery": ("file", "./data/fallback/p2p_discovery"),
            "session_store": ("sqlite", "./data/fallback/sessions"),
        }

        for store_name, (storage_type, storage_path) in fallback_configs.items():
            try:
                fallback_store = RedisFallbackStorage(storage_type, storage_path)
                self.fallback_stores[store_name] = fallback_store
                logger.info(f"Fallback store {store_name} ({storage_type}) initialized")
            except Exception as e:
                logger.error(f"Failed to initialize fallback store {store_name}: {e}")

    @asynccontextmanager
    async def get_connection(self, pool_name: str):
        """Get Redis connection with automatic fallback."""
        # Try Redis first
        if pool_name in self.pools and self.pools[pool_name]["available"]:
            pool = self.pools[pool_name]["pool"]
            if pool:
                redis_client = aioredis.Redis(connection_pool=pool)
                try:
                    # Test connection with a quick ping
                    await asyncio.wait_for(redis_client.ping(), timeout=1.0)
                    yield RedisConnection(redis_client, None, "redis")
                    return
                except Exception as e:
                    logger.warning(f"Redis connection failed for {pool_name}: {e}")
                    await redis_client.close()

        # Use fallback storage
        if pool_name in self.fallback_stores:
            fallback_store = self.fallback_stores[pool_name]
            yield RedisConnection(None, fallback_store, "fallback")
        else:
            # Create temporary in-memory fallback
            temp_fallback = RedisFallbackStorage("memory")
            yield RedisConnection(None, temp_fallback, "memory")

    async def check_connections(self) -> dict[str, dict[str, Any]]:
        """Check status of all Redis connections."""
        status = {}

        for pool_name, pool_info in self.pools.items():
            pool_status = {
                "redis_available": False,
                "fallback_available": False,
                "last_error": None,
            }

            # Test Redis connection
            if pool_info["available"] and pool_info["pool"]:
                try:
                    redis_client = aioredis.Redis(connection_pool=pool_info["pool"])
                    await redis_client.ping()
                    await redis_client.close()
                    pool_status["redis_available"] = True
                except Exception as e:
                    pool_status["last_error"] = str(e)

            # Test fallback
            if pool_name in self.fallback_stores:
                try:
                    fallback_store = self.fallback_stores[pool_name]
                    await fallback_store.set("health_check", "ok", ex=1)
                    result = await fallback_store.get("health_check")
                    pool_status["fallback_available"] = result == "ok"
                except Exception as e:
                    if not pool_status["last_error"]:
                        pool_status["last_error"] = f"Fallback error: {e!s}"

            status[pool_name] = pool_status

        return status

    async def cleanup_expired_keys(self) -> dict[str, int]:
        """Cleanup expired keys from fallback stores."""
        cleanup_results = {}

        for store_name, fallback_store in self.fallback_stores.items():
            try:
                expired_count = await fallback_store.cleanup_expired()
                cleanup_results[store_name] = expired_count
                if expired_count > 0:
                    logger.info(
                        f"Cleaned up {expired_count} expired keys from {store_name}"
                    )
            except Exception as e:
                logger.error(f"Failed to cleanup expired keys from {store_name}: {e}")
                cleanup_results[store_name] = 0

        return cleanup_results

    async def close(self):
        """Close all Redis connections and fallback stores."""
        logger.info("Closing Redis manager...")

        # Close Redis pools
        for pool_name, pool_info in self.pools.items():
            if pool_info["pool"]:
                try:
                    await pool_info["pool"].disconnect()
                    logger.debug(f"Closed Redis pool: {pool_name}")
                except Exception as e:
                    logger.error(f"Error closing Redis pool {pool_name}: {e}")

        # Close fallback stores
        for store_name, fallback_store in self.fallback_stores.items():
            try:
                fallback_store.close()
                logger.debug(f"Closed fallback store: {store_name}")
            except Exception as e:
                logger.error(f"Error closing fallback store {store_name}: {e}")

        self.pools.clear()
        self.fallback_stores.clear()
        self._initialized = False

        logger.info("Redis manager closed")


class RedisConnection:
    """Unified interface for Redis and fallback storage operations."""

    def __init__(self, redis_client, fallback_store, connection_type):
        self.redis_client = redis_client
        self.fallback_store = fallback_store
        self.connection_type = connection_type

    async def get(self, key: str) -> Any | None:
        """Get value from Redis or fallback storage."""
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    # Try to decode JSON first, then return as string
                    try:
                        return json.loads(value.decode())
                    except (json.JSONDecodeError, AttributeError):
                        return value.decode() if isinstance(value, bytes) else value
                return None
            except Exception as e:
                logger.warning(f"Redis get failed for {key}: {e}")
                # Fall through to fallback

        if self.fallback_store:
            return await self.fallback_store.get(key)

        return None

    async def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        """Set value in Redis or fallback storage."""
        if self.redis_client:
            try:
                # Serialize complex types to JSON
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)

                if ex:
                    await self.redis_client.setex(key, ex, value)
                else:
                    await self.redis_client.set(key, value)
                return True
            except Exception as e:
                logger.warning(f"Redis set failed for {key}: {e}")
                # Fall through to fallback

        if self.fallback_store:
            return await self.fallback_store.set(key, value, ex)

        return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis or fallback storage."""
        if self.redis_client:
            try:
                result = await self.redis_client.delete(key)
                return result > 0
            except Exception as e:
                logger.warning(f"Redis delete failed for {key}: {e}")
                # Fall through to fallback

        if self.fallback_store:
            return await self.fallback_store.delete(key)

        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis or fallback storage."""
        if self.redis_client:
            try:
                return bool(await self.redis_client.exists(key))
            except Exception as e:
                logger.warning(f"Redis exists failed for {key}: {e}")
                # Fall through to fallback

        if self.fallback_store:
            return await self.fallback_store.exists(key)

        return False

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key."""
        if self.redis_client:
            try:
                return bool(await self.redis_client.expire(key, seconds))
            except Exception as e:
                logger.warning(f"Redis expire failed for {key}: {e}")
                # Fall through to fallback

        if self.fallback_store:
            return await self.fallback_store.expire(key, seconds)

        return False

    async def close(self):
        """Close the connection."""
        if self.redis_client:
            await self.redis_client.close()


# Global Redis manager instance
_redis_manager: RedisManager | None = None


async def get_redis_manager(config_manager=None) -> RedisManager:
    """Get global Redis manager instance."""
    global _redis_manager

    if _redis_manager is None:
        _redis_manager = RedisManager(config_manager)
        await _redis_manager.initialize()

    return _redis_manager


async def initialize_redis(config_manager=None) -> RedisManager:
    """Initialize Redis manager with configuration."""
    redis_manager = RedisManager(config_manager)
    await redis_manager.initialize()
    return redis_manager


if __name__ == "__main__":

    async def main():
        """Test Redis manager with fallbacks."""
        # Initialize Redis manager
        redis_manager = RedisManager()
        await redis_manager.initialize()

        # Test connections
        connection_status = await redis_manager.check_connections()
        print("Connection status:", connection_status)

        # Test operations
        async with redis_manager.get_connection("rag_cache") as conn:
            # Test set/get
            await conn.set("test_key", {"message": "Hello World"}, ex=60)
            value = await conn.get("test_key")
            print(f"Retrieved value: {value}")

            # Test exists
            exists = await conn.exists("test_key")
            print(f"Key exists: {exists}")

            # Test delete
            deleted = await conn.delete("test_key")
            print(f"Key deleted: {deleted}")

        # Cleanup expired keys
        cleanup_results = await redis_manager.cleanup_expired_keys()
        print("Cleanup results:", cleanup_results)

        await redis_manager.close()

    asyncio.run(main())
