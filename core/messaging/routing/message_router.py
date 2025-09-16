"""
Message Router Implementation

Unified message routing logic consolidating routing from all communication systems.
Handles service discovery, load balancing, and message delivery routing.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Message routing strategies"""
    DIRECT = "direct"              # Direct routing to specific target
    BROADCAST = "broadcast"        # Broadcast to all available targets
    ROUND_ROBIN = "round_robin"    # Round-robin load balancing
    RANDOM = "random"              # Random selection
    LEAST_LOADED = "least_loaded"  # Route to least loaded target


class RouteInfo:
    """Information about a message route"""
    
    def __init__(self, target_id: str, transport_name: str, priority: int = 1):
        self.target_id = target_id
        self.transport_name = transport_name
        self.priority = priority
        self.last_used = datetime.now(timezone.utc)
        self.success_count = 0
        self.failure_count = 0
        self.average_latency_ms = 0.0
        self.is_available = True
    
    def record_success(self, latency_ms: float) -> None:
        """Record successful routing"""
        self.success_count += 1
        self.last_used = datetime.now(timezone.utc)
        
        # Update average latency (simple moving average)
        if self.average_latency_ms == 0.0:
            self.average_latency_ms = latency_ms
        else:
            self.average_latency_ms = (self.average_latency_ms * 0.8) + (latency_ms * 0.2)
    
    def record_failure(self) -> None:
        """Record routing failure"""
        self.failure_count += 1
        
        # Mark as unavailable after consecutive failures
        if self.failure_count > 3:
            self.is_available = False
    
    def get_success_rate(self) -> float:
        """Get success rate percentage"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 100.0
        return (self.success_count / total) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "transport_name": self.transport_name,
            "priority": self.priority,
            "last_used": self.last_used.isoformat(),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.get_success_rate(),
            "average_latency_ms": self.average_latency_ms,
            "is_available": self.is_available
        }


class MessageRouter:
    """Unified message routing logic"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Routing tables
        self.static_routes: Dict[str, List[RouteInfo]] = {}  # target_id -> routes
        self.dynamic_routes: Dict[str, List[RouteInfo]] = {}  # discovered routes
        
        # Load balancing state
        self.round_robin_index: Dict[str, int] = {}  # target_id -> index
        
        # Route cache for performance
        self.route_cache: Dict[str, RouteInfo] = {}  # cache_key -> route
        self.cache_ttl_seconds = 300  # 5 minutes
        
        # Metrics
        self.routing_metrics = {
            "routes_resolved": 0,
            "routes_failed": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info(f"Message router initialized for node: {node_id}")
    
    async def start(self) -> None:
        """Start the message router"""
        if self.running:
            return
        
        self.running = True
        
        # Start cleanup task for route cache
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Message router started")
    
    async def stop(self) -> None:
        """Stop the message router"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Message router stopped")
    
    def add_static_route(self, target_id: str, transport_name: str, priority: int = 1) -> None:
        """Add static route for a target"""
        if target_id not in self.static_routes:
            self.static_routes[target_id] = []
        
        route_info = RouteInfo(target_id, transport_name, priority)
        self.static_routes[target_id].append(route_info)
        
        # Sort by priority (higher priority first)
        self.static_routes[target_id].sort(key=lambda r: r.priority, reverse=True)
        
        logger.info(f"Added static route: {target_id} -> {transport_name} (priority: {priority})")
    
    def remove_static_route(self, target_id: str, transport_name: str) -> bool:
        """Remove static route"""
        if target_id not in self.static_routes:
            return False
        
        routes = self.static_routes[target_id]
        original_count = len(routes)
        
        self.static_routes[target_id] = [
            r for r in routes if r.transport_name != transport_name
        ]
        
        # Remove empty target entries
        if not self.static_routes[target_id]:
            del self.static_routes[target_id]
        
        removed = len(routes) != original_count
        if removed:
            logger.info(f"Removed static route: {target_id} -> {transport_name}")
        
        return removed
    
    def add_dynamic_route(self, target_id: str, transport_name: str, priority: int = 1) -> None:
        """Add dynamically discovered route"""
        if target_id not in self.dynamic_routes:
            self.dynamic_routes[target_id] = []
        
        # Check if route already exists
        for route in self.dynamic_routes[target_id]:
            if route.transport_name == transport_name:
                route.priority = priority  # Update priority
                route.is_available = True   # Mark as available again
                logger.debug(f"Updated dynamic route: {target_id} -> {transport_name}")
                return
        
        # Add new route
        route_info = RouteInfo(target_id, transport_name, priority)
        self.dynamic_routes[target_id].append(route_info)
        
        # Sort by priority
        self.dynamic_routes[target_id].sort(key=lambda r: r.priority, reverse=True)
        
        logger.info(f"Added dynamic route: {target_id} -> {transport_name} (priority: {priority})")
    
    def resolve_route(self, target_id: str, strategy: RoutingStrategy = RoutingStrategy.DIRECT) -> Optional[RouteInfo]:
        """Resolve route for target using specified strategy"""
        try:
            self.routing_metrics["routes_resolved"] += 1
            
            # Check cache first
            cache_key = f"{target_id}:{strategy.value}"
            cached_route = self._get_cached_route(cache_key)
            if cached_route:
                self.routing_metrics["cache_hits"] += 1
                return cached_route
            
            self.routing_metrics["cache_misses"] += 1
            
            # Get available routes
            available_routes = self._get_available_routes(target_id)
            if not available_routes:
                logger.warning(f"No routes available for target: {target_id}")
                self.routing_metrics["routes_failed"] += 1
                return None
            
            # Apply routing strategy
            selected_route = self._apply_routing_strategy(target_id, available_routes, strategy)
            
            if selected_route:
                # Cache the result
                self._cache_route(cache_key, selected_route)
                logger.debug(f"Resolved route: {target_id} -> {selected_route.transport_name} ({strategy.value})")
            
            return selected_route
            
        except Exception as e:
            logger.error(f"Error resolving route for {target_id}: {e}")
            self.routing_metrics["routes_failed"] += 1
            return None
    
    def record_route_success(self, target_id: str, transport_name: str, latency_ms: float) -> None:
        """Record successful route usage"""
        route = self._find_route(target_id, transport_name)
        if route:
            route.record_success(latency_ms)
            logger.debug(f"Recorded route success: {target_id} -> {transport_name} ({latency_ms}ms)")
    
    def record_route_failure(self, target_id: str, transport_name: str) -> None:
        """Record route failure"""
        route = self._find_route(target_id, transport_name)
        if route:
            route.record_failure()
            logger.warning(f"Recorded route failure: {target_id} -> {transport_name}")
            
            # Invalidate cache entries for this target
            self._invalidate_cache_for_target(target_id)
    
    def get_route_stats(self, target_id: str) -> List[Dict[str, Any]]:
        """Get statistics for routes to a target"""
        routes = self._get_all_routes(target_id)
        return [route.to_dict() for route in routes]
    
    def get_all_targets(self) -> Set[str]:
        """Get all known targets"""
        targets = set(self.static_routes.keys())
        targets.update(self.dynamic_routes.keys())
        return targets
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics"""
        return {
            **self.routing_metrics,
            "static_routes_count": len(self.static_routes),
            "dynamic_routes_count": len(self.dynamic_routes),
            "cache_size": len(self.route_cache),
            "total_routes": sum(len(routes) for routes in self.static_routes.values()) +
                           sum(len(routes) for routes in self.dynamic_routes.values())
        }
    
    def clear_cache(self) -> None:
        """Clear route cache"""
        self.route_cache.clear()
        logger.info("Route cache cleared")
    
    # Private methods
    
    def _get_available_routes(self, target_id: str) -> List[RouteInfo]:
        """Get all available routes for a target"""
        routes = []
        
        # Add static routes (higher priority)
        if target_id in self.static_routes:
            routes.extend([r for r in self.static_routes[target_id] if r.is_available])
        
        # Add dynamic routes
        if target_id in self.dynamic_routes:
            routes.extend([r for r in self.dynamic_routes[target_id] if r.is_available])
        
        # Sort by priority and success rate
        routes.sort(key=lambda r: (r.priority, r.get_success_rate()), reverse=True)
        
        return routes
    
    def _get_all_routes(self, target_id: str) -> List[RouteInfo]:
        """Get all routes (including unavailable ones) for a target"""
        routes = []
        
        if target_id in self.static_routes:
            routes.extend(self.static_routes[target_id])
        
        if target_id in self.dynamic_routes:
            routes.extend(self.dynamic_routes[target_id])
        
        return routes
    
    def _apply_routing_strategy(self, target_id: str, routes: List[RouteInfo], 
                              strategy: RoutingStrategy) -> Optional[RouteInfo]:
        """Apply routing strategy to select route"""
        if not routes:
            return None
        
        if strategy == RoutingStrategy.DIRECT:
            # Return highest priority route
            return routes[0]
        
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Round-robin selection
            if target_id not in self.round_robin_index:
                self.round_robin_index[target_id] = 0
            
            index = self.round_robin_index[target_id] % len(routes)
            self.round_robin_index[target_id] += 1
            
            return routes[index]
        
        elif strategy == RoutingStrategy.RANDOM:
            # Random selection
            import random
            return random.choice(routes)
        
        elif strategy == RoutingStrategy.LEAST_LOADED:
            # Select route with best performance metrics
            return min(routes, key=lambda r: (r.average_latency_ms, -r.get_success_rate()))
        
        else:
            # Default to direct
            return routes[0]
    
    def _find_route(self, target_id: str, transport_name: str) -> Optional[RouteInfo]:
        """Find specific route by target and transport"""
        for route in self._get_all_routes(target_id):
            if route.transport_name == transport_name:
                return route
        return None
    
    def _get_cached_route(self, cache_key: str) -> Optional[RouteInfo]:
        """Get route from cache if not expired"""
        if cache_key in self.route_cache:
            route = self.route_cache[cache_key]
            
            # Check if cache entry is still valid
            elapsed = (datetime.now(timezone.utc) - route.last_used).total_seconds()
            if elapsed < self.cache_ttl_seconds and route.is_available:
                return route
            else:
                # Remove expired cache entry
                del self.route_cache[cache_key]
        
        return None
    
    def _cache_route(self, cache_key: str, route: RouteInfo) -> None:
        """Cache route result"""
        self.route_cache[cache_key] = route
    
    def _invalidate_cache_for_target(self, target_id: str) -> None:
        """Invalidate all cache entries for a target"""
        keys_to_remove = [key for key in self.route_cache.keys() if key.startswith(f"{target_id}:")]
        for key in keys_to_remove:
            del self.route_cache[key]
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired cache entries and unavailable routes"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Cleanup expired cache entries
                current_time = datetime.now(timezone.utc)
                expired_keys = []
                
                for key, route in self.route_cache.items():
                    elapsed = (current_time - route.last_used).total_seconds()
                    if elapsed >= self.cache_ttl_seconds:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.route_cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                # Reset availability for routes that haven't been used recently
                for routes_dict in [self.static_routes, self.dynamic_routes]:
                    for routes in routes_dict.values():
                        for route in routes:
                            if not route.is_available:
                                elapsed = (current_time - route.last_used).total_seconds()
                                if elapsed >= 300:  # 5 minutes
                                    route.is_available = True
                                    route.failure_count = 0  # Reset failure count
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in router cleanup loop: {e}")
