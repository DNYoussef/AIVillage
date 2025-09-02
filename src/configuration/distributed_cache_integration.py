"""
Distributed Configuration Cache Integration with Context7 MCP
Provides distributed caching capabilities for configuration management
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib
import redis
from threading import RLock

logger = logging.getLogger(__name__)

@dataclass
class CacheNode:
    """Represents a cache node in the distributed system"""
    node_id: str
    host: str
    port: int
    weight: int
    healthy: bool = True
    last_heartbeat: Optional[datetime] = None

@dataclass
class CacheEntry:
    """Represents a cached configuration entry"""
    key: str
    value: Dict[str, Any]
    ttl: int
    created_at: datetime
    checksum: str
    replicated_nodes: Set[str]

class DistributedCacheManager:
    """Manages distributed configuration caching across multiple nodes"""
    
    def __init__(self, 
                 nodes: Optional[List[CacheNode]] = None,
                 replication_factor: int = 2,
                 consistency_level: str = "eventual"):
        self.nodes = nodes or []
        self.replication_factor = replication_factor
        self.consistency_level = consistency_level  # "strong", "eventual", "weak"
        self._local_cache: Dict[str, CacheEntry] = {}
        self._lock = RLock()
        
        # Connection pools for Redis nodes
        self._redis_pools: Dict[str, redis.Redis] = {}
        
        # Health monitoring
        self._health_check_interval = 30  # seconds
        self._heartbeat_timeout = 60  # seconds
        
    async def initialize(self):
        """Initialize distributed cache system"""
        logger.info("Initializing distributed configuration cache")
        
        # Initialize Redis connections for each node
        for node in self.nodes:
            try:
                redis_client = redis.Redis(
                    host=node.host,
                    port=node.port,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                redis_client.ping()
                self._redis_pools[node.node_id] = redis_client
                node.healthy = True
                node.last_heartbeat = datetime.now()
                logger.info(f"Connected to cache node: {node.node_id}")
                
            except Exception as e:
                logger.error(f"Failed to connect to cache node {node.node_id}: {e}")
                node.healthy = False
                
        # Start health monitoring
        asyncio.create_task(self._health_monitor())
        
    async def cache_configuration(self, 
                                key: str, 
                                config: Dict[str, Any], 
                                ttl: int = 3600,
                                replicate: bool = True) -> bool:
        """Cache configuration with optional replication"""
        
        # Calculate checksum
        config_json = json.dumps(config, sort_keys=True)
        checksum = hashlib.sha256(config_json.encode()).hexdigest()
        
        # Create cache entry
        cache_entry = CacheEntry(
            key=key,
            value=config,
            ttl=ttl,
            created_at=datetime.now(),
            checksum=checksum,
            replicated_nodes=set()
        )
        
        # Store in local cache
        with self._lock:
            self._local_cache[key] = cache_entry
            
        success = True
        
        # Replicate to distributed nodes
        if replicate and self.nodes:
            target_nodes = self._select_replication_nodes(key)
            
            replication_tasks = []
            for node in target_nodes:
                if node.healthy and node.node_id in self._redis_pools:
                    task = asyncio.create_task(
                        self._replicate_to_node(node, key, cache_entry)
                    )
                    replication_tasks.append(task)
                    
            # Wait for replication based on consistency level
            if self.consistency_level == "strong":
                # Wait for all replications to complete
                results = await asyncio.gather(*replication_tasks, return_exceptions=True)
                success = all(isinstance(r, bool) and r for r in results)
                
            elif self.consistency_level == "eventual":
                # Fire and forget - let them complete in background
                pass
                
        logger.info(f"Cached configuration '{key}' with {len(cache_entry.replicated_nodes)} replicas")
        return success
        
    async def get_cached_configuration(self, key: str, prefer_local: bool = True) -> Optional[Dict[str, Any]]:
        """Retrieve cached configuration with fallback strategy"""
        
        # Try local cache first
        if prefer_local:
            with self._lock:
                if key in self._local_cache:
                    entry = self._local_cache[key]
                    if not self._is_expired(entry):
                        return entry.value
                    else:
                        # Remove expired entry
                        del self._local_cache[key]
                        
        # Try distributed nodes
        if self.nodes:
            for node in self._get_healthy_nodes():
                try:
                    redis_client = self._redis_pools.get(node.node_id)
                    if redis_client:
                        cached_data = redis_client.get(f"config:{key}")
                        if cached_data:
                            data = json.loads(cached_data)
                            
                            # Validate TTL
                            if self._validate_distributed_entry(data):
                                config_value = data.get("value", {})
                                
                                # Update local cache
                                cache_entry = CacheEntry(
                                    key=key,
                                    value=config_value,
                                    ttl=data.get("ttl", 3600),
                                    created_at=datetime.fromisoformat(data.get("created_at")),
                                    checksum=data.get("checksum", ""),
                                    replicated_nodes=set(data.get("replicated_nodes", []))
                                )
                                
                                with self._lock:
                                    self._local_cache[key] = cache_entry
                                    
                                return config_value
                                
                except Exception as e:
                    logger.warning(f"Failed to retrieve from node {node.node_id}: {e}")
                    continue
                    
        return None
        
    async def invalidate_configuration(self, key: str, propagate: bool = True) -> bool:
        """Invalidate cached configuration"""
        
        # Remove from local cache
        with self._lock:
            removed_locally = self._local_cache.pop(key, None) is not None
            
        # Propagate to distributed nodes
        if propagate and self.nodes:
            invalidation_tasks = []
            for node in self._get_healthy_nodes():
                if node.node_id in self._redis_pools:
                    task = asyncio.create_task(
                        self._invalidate_on_node(node, key)
                    )
                    invalidation_tasks.append(task)
                    
            # Wait for invalidations
            await asyncio.gather(*invalidation_tasks, return_exceptions=True)
            
        logger.info(f"Invalidated configuration '{key}' (local: {removed_locally})")
        return True
        
    async def flush_cache_pattern(self, pattern: str) -> bool:
        """Flush all cache entries matching pattern"""
        
        # Flush local cache
        with self._lock:
            keys_to_remove = [k for k in self._local_cache.keys() if self._matches_pattern(k, pattern)]
            for key in keys_to_remove:
                del self._local_cache[key]
                
        # Flush on distributed nodes
        flush_tasks = []
        for node in self._get_healthy_nodes():
            if node.node_id in self._redis_pools:
                task = asyncio.create_task(
                    self._flush_pattern_on_node(node, pattern)
                )
                flush_tasks.append(task)
                
        await asyncio.gather(*flush_tasks, return_exceptions=True)
        
        logger.info(f"Flushed cache pattern: {pattern}")
        return True
        
    def _select_replication_nodes(self, key: str) -> List[CacheNode]:
        """Select nodes for replication using consistent hashing"""
        healthy_nodes = self._get_healthy_nodes()
        
        if not healthy_nodes:
            return []
            
        # Simple selection based on key hash for now
        key_hash = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        
        # Select nodes based on replication factor
        selected = []
        for i in range(min(self.replication_factor, len(healthy_nodes))):
            node_index = (key_hash + i) % len(healthy_nodes)
            selected.append(healthy_nodes[node_index])
            
        return selected
        
    async def _replicate_to_node(self, node: CacheNode, key: str, entry: CacheEntry) -> bool:
        """Replicate cache entry to a specific node"""
        try:
            redis_client = self._redis_pools.get(node.node_id)
            if not redis_client:
                return False
                
            # Prepare data for storage
            data = {
                "key": entry.key,
                "value": entry.value,
                "ttl": entry.ttl,
                "created_at": entry.created_at.isoformat(),
                "checksum": entry.checksum,
                "replicated_nodes": list(entry.replicated_nodes)
            }
            
            # Store with TTL
            redis_client.setex(
                f"config:{key}",
                entry.ttl,
                json.dumps(data)
            )
            
            # Update replicated nodes
            entry.replicated_nodes.add(node.node_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to replicate to node {node.node_id}: {e}")
            return False
            
    async def _invalidate_on_node(self, node: CacheNode, key: str) -> bool:
        """Invalidate cache entry on a specific node"""
        try:
            redis_client = self._redis_pools.get(node.node_id)
            if redis_client:
                redis_client.delete(f"config:{key}")
                return True
        except Exception as e:
            logger.error(f"Failed to invalidate on node {node.node_id}: {e}")
        return False
        
    async def _flush_pattern_on_node(self, node: CacheNode, pattern: str) -> bool:
        """Flush pattern on a specific node"""
        try:
            redis_client = self._redis_pools.get(node.node_id)
            if redis_client:
                # Convert simple pattern to Redis pattern
                redis_pattern = f"config:{pattern}"
                keys = redis_client.keys(redis_pattern)
                if keys:
                    redis_client.delete(*keys)
                return True
        except Exception as e:
            logger.error(f"Failed to flush pattern on node {node.node_id}: {e}")
        return False
        
    def _get_healthy_nodes(self) -> List[CacheNode]:
        """Get list of healthy nodes"""
        return [node for node in self.nodes if node.healthy]
        
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        age = datetime.now() - entry.created_at
        return age.total_seconds() > entry.ttl
        
    def _validate_distributed_entry(self, data: Dict[str, Any]) -> bool:
        """Validate distributed cache entry"""
        try:
            created_at = datetime.fromisoformat(data.get("created_at", ""))
            ttl = data.get("ttl", 0)
            age = datetime.now() - created_at
            return age.total_seconds() <= ttl
        except:
            return False
            
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching"""
        if "*" in pattern:
            pattern_parts = pattern.split("*")
            return key.startswith(pattern_parts[0])
        return key == pattern
        
    async def _health_monitor(self):
        """Monitor node health continuously"""
        while True:
            try:
                for node in self.nodes:
                    try:
                        redis_client = self._redis_pools.get(node.node_id)
                        if redis_client:
                            redis_client.ping()
                            node.healthy = True
                            node.last_heartbeat = datetime.now()
                        else:
                            node.healthy = False
                    except Exception as e:
                        logger.warning(f"Health check failed for node {node.node_id}: {e}")
                        node.healthy = False
                        
                await asyncio.sleep(self._health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self._health_check_interval)
                
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            local_count = len(self._local_cache)
            
        # Get stats from distributed nodes
        node_stats = {}
        for node in self.nodes:
            try:
                redis_client = self._redis_pools.get(node.node_id)
                if redis_client and node.healthy:
                    info = redis_client.info()
                    node_stats[node.node_id] = {
                        "keys": info.get("db0", {}).get("keys", 0),
                        "memory_used": info.get("used_memory_human", "0B"),
                        "connected_clients": info.get("connected_clients", 0)
                    }
            except Exception as e:
                node_stats[node.node_id] = {"error": str(e)}
                
        return {
            "local_cache_entries": local_count,
            "healthy_nodes": len(self._get_healthy_nodes()),
            "total_nodes": len(self.nodes),
            "replication_factor": self.replication_factor,
            "consistency_level": self.consistency_level,
            "node_stats": node_stats
        }

class Context7MCPIntegration:
    """Integration layer for Context7 MCP distributed caching"""
    
    def __init__(self, cache_manager: DistributedCacheManager):
        self.cache_manager = cache_manager
        self._session_cache: Dict[str, Any] = {}
        
    async def cache_with_session_persistence(self, 
                                           key: str, 
                                           config: Dict[str, Any], 
                                           session_id: Optional[str] = None,
                                           ttl: int = 3600) -> bool:
        """Cache configuration with session persistence"""
        
        # Cache in distributed system
        success = await self.cache_manager.cache_configuration(key, config, ttl)
        
        # Store session-specific data if provided
        if session_id:
            session_key = f"session:{session_id}:{key}"
            self._session_cache[session_key] = {
                "config": config,
                "timestamp": datetime.now(),
                "ttl": ttl
            }
            
        # Reference implementation: Integrate with actual Context7 MCP
        # await context7.session.store(f"config/{key}", config, session_id=session_id)
        
        return success
        
    async def get_session_configuration(self, 
                                      key: str, 
                                      session_id: str) -> Optional[Dict[str, Any]]:
        """Get session-specific configuration"""
        
        session_key = f"session:{session_id}:{key}"
        
        # Check local session cache
        if session_key in self._session_cache:
            entry = self._session_cache[session_key]
            age = datetime.now() - entry["timestamp"]
            if age.total_seconds() <= entry["ttl"]:
                return entry["config"]
            else:
                del self._session_cache[session_key]
                
        # Fall back to distributed cache
        return await self.cache_manager.get_cached_configuration(key)
        
    async def replicate_across_regions(self, 
                                     key: str, 
                                     config: Dict[str, Any],
                                     regions: List[str]) -> Dict[str, bool]:
        """Replicate configuration across geographic regions"""
        
        results = {}
        
        # Reference implementation: Implement region-aware replication via Context7 MCP
        # For now, simulate with node selection
        for region in regions:
            region_nodes = [node for node in self.cache_manager.nodes 
                          if node.node_id.startswith(region)]
            
            if region_nodes:
                success = True
                for node in region_nodes:
                    try:
                        cache_entry = CacheEntry(
                            key=key,
                            value=config,
                            ttl=3600,
                            created_at=datetime.now(),
                            checksum=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                            replicated_nodes=set()
                        )
                        node_success = await self.cache_manager._replicate_to_node(node, key, cache_entry)
                        success = success and node_success
                    except Exception as e:
                        logger.error(f"Region {region} replication failed: {e}")
                        success = False
                        
                results[region] = success
            else:
                results[region] = False
                
        return results

# Factory function for creating distributed cache manager
async def create_distributed_cache_manager(
    redis_nodes: Optional[List[Dict[str, Any]]] = None,
    replication_factor: int = 2,
    consistency_level: str = "eventual"
) -> DistributedCacheManager:
    """Create and initialize distributed cache manager"""
    
    # Default nodes for development
    if not redis_nodes:
        redis_nodes = [
            {"node_id": "primary", "host": "localhost", "port": 6379, "weight": 100},
            {"node_id": "replica1", "host": "localhost", "port": 6380, "weight": 50},
        ]
        
    # Create cache nodes
    nodes = []
    for node_config in redis_nodes:
        node = CacheNode(
            node_id=node_config["node_id"],
            host=node_config["host"],
            port=node_config["port"],
            weight=node_config["weight"]
        )
        nodes.append(node)
        
    # Create manager
    manager = DistributedCacheManager(
        nodes=nodes,
        replication_factor=replication_factor,
        consistency_level=consistency_level
    )
    
    await manager.initialize()
    return manager

if __name__ == "__main__":
    async def test_distributed_cache():
        # Test the distributed cache system
        manager = await create_distributed_cache_manager()
        
        # Test configuration caching
        test_config = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"rate_limit": 1000}
        }
        
        await manager.cache_configuration("test_config", test_config)
        
        # Retrieve configuration
        cached = await manager.get_cached_configuration("test_config")
        print("Cached configuration:", cached)
        
        # Get cache stats
        stats = await manager.get_cache_stats()
        print("Cache stats:", stats)
        
    asyncio.run(test_distributed_cache())