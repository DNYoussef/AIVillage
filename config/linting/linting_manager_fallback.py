"""
AIVillage Unified Linting Manager - Fallback Implementation
Provides caching fallback when external dependencies are not available
"""

import asyncio
import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml

logger = logging.getLogger(__name__)


class CacheKey:
    """Simplified cache key generation for fallback"""
    
    @staticmethod
    def hash_params(params: Dict[str, Any]) -> str:
        """Create deterministic hash of parameters"""
        serialized = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class FallbackCacheManager:
    """Simple in-memory cache manager as fallback when Redis/Memcached unavailable"""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl: Dict[str, float] = {}
        self._stats = {"hits": 0, "misses": 0, "sets": 0}
    
    async def get(self, key: str, service: str = "default") -> Any:
        """Get value from in-memory cache"""
        current_time = time.time()
        
        # Check if key exists and hasn't expired
        if key in self._cache and key in self._ttl:
            if current_time < self._ttl[key]:
                self._stats["hits"] += 1
                logger.debug(f"Cache hit for key: {key[:20]}...")
                return self._cache[key]
            else:
                # Remove expired key
                del self._cache[key]
                del self._ttl[key]
        
        self._stats["misses"] += 1
        logger.debug(f"Cache miss for key: {key[:20]}...")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600, service: str = "default") -> bool:
        """Set value in in-memory cache"""
        try:
            self._cache[key] = value
            self._ttl[key] = time.time() + ttl
            self._stats["sets"] += 1
            logger.debug(f"Cached value for key: {key[:20]}... (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to cache value: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self._cache:
            del self._cache[key]
        if key in self._ttl:
            del self._ttl[key]
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_ops = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total_ops) if total_ops > 0 else 0.0
        
        return {
            "cache_type": "fallback_memory",
            "total_keys": len(self._cache),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "hit_rate": hit_rate,
            "total_operations": total_ops
        }


# Global fallback cache manager
cache_manager = FallbackCacheManager()