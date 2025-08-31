"""
Service Communication Layer - Event Bus and Message Routing

This module provides the communication infrastructure for microservices:
- Event publishing and subscription
- Message routing and filtering
- Service discovery and health checking
- Circuit breaker and retry mechanisms

"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from interfaces.service_contracts import Event

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for service resilience."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure = 0
        self.state = CircuitBreakerState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure > self.timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
    
    def on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")


class EventBus(ABC):
    """Abstract event bus interface."""
    
    @abstractmethod
    async def publish(self, event: Event) -> bool:
        """Publish an event."""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        """Subscribe to events of a specific type."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: str, handler: Callable[[Event], None]):
        """Unsubscribe from events."""
        pass


class InMemoryEventBus(EventBus):
    """In-memory event bus implementation for development."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = 1000
    
    async def publish(self, event: Event) -> bool:
        """Publish an event to all subscribers."""
        try:
            # Store in history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
            
            # Notify subscribers
            handlers = self.subscribers.get(event.event_type, [])
            handlers.extend(self.subscribers.get("*", []))  # Wildcard subscribers
            
            if handlers:
                # Execute handlers concurrently
                tasks = [self._safe_handle_event(handler, event) for handler in handlers]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                logger.debug(f"Published {event.event_type} to {len(handlers)} subscribers")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_type}: {e}")
            return False
    
    async def _safe_handle_event(self, handler: Callable, event: Event):
        """Safely execute event handler."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Event handler failed for {event.event_type}: {e}")
    
    async def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        """Subscribe to events of a specific type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        if handler not in self.subscribers[event_type]:
            self.subscribers[event_type].append(handler)
            logger.debug(f"Subscribed to {event_type}")
    
    async def unsubscribe(self, event_type: str, handler: Callable[[Event], None]):
        """Unsubscribe from events."""
        if event_type in self.subscribers:
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed from {event_type}")


class RedisEventBus(EventBus):
    """Redis-based event bus implementation for production."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.subscribers: Dict[str, List[Callable]] = {}
        self.subscription_task = None
    
    async def connect(self):
        """Connect to Redis."""
        try:
            import aioredis
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis event bus")
            
            # Start subscription task
            self.subscription_task = asyncio.create_task(self._subscription_loop())
            
        except ImportError:
            logger.error("aioredis not installed, falling back to in-memory event bus")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def publish(self, event: Event) -> bool:
        """Publish event to Redis channel."""
        try:
            if not self.redis_client:
                return False
            
            channel = f"events:{event.event_type}"
            message = json.dumps(event.dict(), default=str)
            
            await self.redis_client.publish(channel, message)
            logger.debug(f"Published {event.event_type} to Redis")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event to Redis: {e}")
            return False
    
    async def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        """Subscribe to events via Redis channels."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        if handler not in self.subscribers[event_type]:
            self.subscribers[event_type].append(handler)
            
            # Subscribe to Redis channel
            if self.redis_client:
                channel = f"events:{event_type}"
                await self.redis_client.subscribe(channel)
                logger.debug(f"Subscribed to Redis channel: {channel}")
    
    async def unsubscribe(self, event_type: str, handler: Callable[[Event], None]):
        """Unsubscribe from Redis channels."""
        if event_type in self.subscribers and handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            
            # If no more handlers, unsubscribe from Redis
            if not self.subscribers[event_type] and self.redis_client:
                channel = f"events:{event_type}"
                await self.redis_client.unsubscribe(channel)
                logger.debug(f"Unsubscribed from Redis channel: {channel}")
    
    async def _subscription_loop(self):
        """Handle incoming Redis messages."""
        try:
            pubsub = self.redis_client.pubsub()
            
            # Subscribe to all event channels
            for event_type in self.subscribers.keys():
                await pubsub.subscribe(f"events:{event_type}")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        event_data = json.loads(message["data"])
                        event = Event(**event_data)
                        
                        # Handle event
                        handlers = self.subscribers.get(event.event_type, [])
                        tasks = [self._safe_handle_event(handler, event) for handler in handlers]
                        if tasks:
                            await asyncio.gather(*tasks, return_exceptions=True)
                            
                    except Exception as e:
                        logger.error(f"Failed to handle Redis message: {e}")
                        
        except Exception as e:
            logger.error(f"Redis subscription loop error: {e}")
    
    async def _safe_handle_event(self, handler: Callable, event: Event):
        """Safely execute event handler."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(f"Event handler failed: {e}")
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.subscription_task:
            self.subscription_task.cancel()
        if self.redis_client:
            await self.redis_client.close()


class ServiceDiscovery:
    """Service discovery and health checking."""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def register_service(self, name: str, endpoint: str, health_endpoint: str = None):
        """Register a service endpoint."""
        self.services[name] = {
            "endpoint": endpoint,
            "health_endpoint": health_endpoint or f"{endpoint}/health",
            "registered_at": datetime.now(),
            "last_health_check": None,
            "is_healthy": True
        }
        
        # Create circuit breaker for service
        self.circuit_breakers[name] = CircuitBreaker()
        
        logger.info(f"Registered service {name} at {endpoint}")
    
    def get_service_endpoint(self, name: str) -> Optional[str]:
        """Get service endpoint by name."""
        service = self.services.get(name)
        return service["endpoint"] if service and service["is_healthy"] else None
    
    async def call_service(self, service_name: str, func, *args, **kwargs):
        """Call service with circuit breaker protection."""
        if service_name not in self.circuit_breakers:
            raise ValueError(f"Unknown service: {service_name}")
        
        circuit_breaker = self.circuit_breakers[service_name]
        return await circuit_breaker.call(func, *args, **kwargs)
    
    async def health_check(self, service_name: str) -> bool:
        """Check service health."""
        service = self.services.get(service_name)
        if not service:
            return False
        
        try:
            # In real implementation, make HTTP request to health endpoint
            # For now, simulate health check
            await asyncio.sleep(0.01)
            
            # Update health status
            service["last_health_check"] = datetime.now()
            service["is_healthy"] = True
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            service["is_healthy"] = False
            return False
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered services."""
        results = {}
        tasks = []
        
        for service_name in self.services.keys():
            task = asyncio.create_task(self.health_check(service_name))
            tasks.append((service_name, task))
        
        for service_name, task in tasks:
            try:
                results[service_name] = await task
            except Exception:
                results[service_name] = False
        
        return results
    
    def get_healthy_services(self) -> List[str]:
        """Get list of healthy services."""
        return [name for name, info in self.services.items() if info["is_healthy"]]


class MessageRouter:
    """Routes messages between services based on patterns."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.routes: Dict[str, List[str]] = {}  # event_type -> [service_names]
        self.filters: Dict[str, Callable] = {}  # event_type -> filter_function
    
    def add_route(self, event_type: str, target_services: List[str]):
        """Add routing rule for an event type."""
        self.routes[event_type] = target_services
        logger.debug(f"Added route: {event_type} -> {target_services}")
    
    def add_filter(self, event_type: str, filter_func: Callable[[Event], bool]):
        """Add filter for an event type."""
        self.filters[event_type] = filter_func
        logger.debug(f"Added filter for {event_type}")
    
    async def route_event(self, event: Event) -> bool:
        """Route event to appropriate services."""
        try:
            # Apply filter if exists
            if event.event_type in self.filters:
                filter_func = self.filters[event.event_type]
                if not filter_func(event):
                    logger.debug(f"Event {event.event_type} filtered out")
                    return True
            
            # Route to specific services or publish globally
            target_services = self.routes.get(event.event_type, [])
            
            if target_services:
                # Route to specific services (implementation would send to service queues)
                logger.debug(f"Routing {event.event_type} to {target_services}")
            else:
                # Publish globally
                await self.event_bus.publish(event)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to route event {event.event_type}: {e}")
            return False


async def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"All retry attempts failed: {e}")
                raise
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            await asyncio.sleep(delay)


# Factory functions for creating communication components

def create_event_bus(bus_type: str = "memory", **kwargs) -> EventBus:
    """Create event bus instance."""
    if bus_type == "redis":
        redis_url = kwargs.get("redis_url", "redis://localhost:6379")
        return RedisEventBus(redis_url)
    else:
        return InMemoryEventBus()


def create_service_discovery() -> ServiceDiscovery:
    """Create service discovery instance."""
    return ServiceDiscovery()


def create_message_router(event_bus: EventBus) -> MessageRouter:
    """Create message router instance."""
    return MessageRouter(event_bus)


# Communication layer setup for microservices
class CommunicationLayer:
    """Central communication layer for microservices."""
    
    def __init__(self, 
                 bus_type: str = "memory",
                 redis_url: str = "redis://localhost:6379"):
        self.event_bus = create_event_bus(bus_type, redis_url=redis_url)
        self.service_discovery = create_service_discovery()
        self.message_router = create_message_router(self.event_bus)
    
    async def initialize(self):
        """Initialize communication layer."""
        if isinstance(self.event_bus, RedisEventBus):
            await self.event_bus.connect()
        logger.info("Communication layer initialized")
    
    async def shutdown(self):
        """Shutdown communication layer."""
        if isinstance(self.event_bus, RedisEventBus):
            await self.event_bus.disconnect()
        logger.info("Communication layer shutdown")
    
    def get_event_bus(self) -> EventBus:
        return self.event_bus
    
    def get_service_discovery(self) -> ServiceDiscovery:
        return self.service_discovery
    
    def get_message_router(self) -> MessageRouter:
        return self.message_router