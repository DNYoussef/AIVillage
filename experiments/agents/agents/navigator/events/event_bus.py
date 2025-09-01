"""Event Bus Implementation - Inter-service communication for routing system

This module provides the event bus infrastructure that enables loose coupling
between routing services while maintaining coordinated decision-making.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional
import time
import uuid

from ..interfaces.routing_interfaces import RoutingEvent

logger = logging.getLogger(__name__)


@dataclass
class Subscription:
    """Event subscription details"""

    subscription_id: str
    event_type: str
    callback: Callable[[RoutingEvent], None]
    service_name: str
    created_at: float = field(default_factory=time.time)
    active: bool = True


class EventBusService:
    """Event bus for inter-service communication in the Navigator system"""

    def __init__(self, max_event_history: int = 1000):
        self.max_event_history = max_event_history

        # Event handling
        self.subscriptions: Dict[str, Subscription] = {}
        self.event_type_subscriptions: Dict[str, List[str]] = defaultdict(list)
        self.event_history: List[RoutingEvent] = []

        # Performance tracking
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.processing_times: Dict[str, float] = defaultdict(float)
        self.failed_deliveries: Dict[str, int] = defaultdict(int)

        # Async processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.running = False

        logger.info("EventBusService initialized")

    async def start(self) -> None:
        """Start the event bus processing"""
        if self.running:
            return

        self.running = True
        self.processing_task = asyncio.create_task(self._process_events())
        logger.info("EventBusService started")

    async def stop(self) -> None:
        """Stop the event bus processing"""
        self.running = False

        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        logger.info("EventBusService stopped")

    def publish(self, event: RoutingEvent) -> None:
        """Publish event to interested services"""
        if not self.running:
            logger.warning(f"Event bus not running, dropping event: {event.event_type}")
            return

        # Add to processing queue
        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.error(f"Event queue full, dropping event: {event.event_type}")

    async def publish_async(self, event: RoutingEvent) -> None:
        """Publish event asynchronously"""
        if not self.running:
            logger.warning(f"Event bus not running, dropping event: {event.event_type}")
            return

        await self.event_queue.put(event)

    def subscribe(self, event_type: str, callback: Callable[[RoutingEvent], None], service_name: str) -> str:
        """Subscribe to specific event types"""
        subscription_id = f"{service_name}_{event_type}_{uuid.uuid4().hex[:8]}"

        subscription = Subscription(
            subscription_id=subscription_id, event_type=event_type, callback=callback, service_name=service_name
        )

        self.subscriptions[subscription_id] = subscription
        self.event_type_subscriptions[event_type].append(subscription_id)

        logger.info(f"Service '{service_name}' subscribed to '{event_type}' events")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions[subscription_id]
        subscription.active = False

        # Remove from event type mapping
        event_type_subs = self.event_type_subscriptions[subscription.event_type]
        if subscription_id in event_type_subs:
            event_type_subs.remove(subscription_id)

        del self.subscriptions[subscription_id]

        logger.info(f"Service '{subscription.service_name}' unsubscribed from " f"'{subscription.event_type}' events")
        return True

    async def _process_events(self) -> None:
        """Process events from the queue"""
        logger.info("Event processing started")

        while self.running:
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                await self._deliver_event(event)

            except Exception as e:
                logger.error(f"Error in event processing: {e}")
                await asyncio.sleep(0.1)  # Brief pause to prevent tight loop

    async def _deliver_event(self, event: RoutingEvent) -> None:
        """Deliver event to all subscribers"""
        start_time = time.time()

        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_event_history:
            self.event_history.pop(0)

        # Track event counts
        self.event_counts[event.event_type] += 1

        # Get subscribers for this event type
        subscription_ids = self.event_type_subscriptions.get(event.event_type, [])

        if not subscription_ids:
            logger.debug(f"No subscribers for event type: {event.event_type}")
            return

        # Deliver to each subscriber
        delivery_tasks = []
        for sub_id in subscription_ids:
            if sub_id in self.subscriptions and self.subscriptions[sub_id].active:
                task = asyncio.create_task(self._deliver_to_subscriber(event, self.subscriptions[sub_id]))
                delivery_tasks.append(task)

        # Wait for all deliveries to complete
        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)

        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times[event.event_type] += processing_time

        logger.debug(
            f"Event '{event.event_type}' delivered to {len(delivery_tasks)} "
            f"subscribers in {processing_time*1000:.2f}ms"
        )

    async def _deliver_to_subscriber(self, event: RoutingEvent, subscription: Subscription) -> None:
        """Deliver event to a single subscriber"""
        try:
            # Call subscriber callback (may be sync or async)
            if asyncio.iscoroutinefunction(subscription.callback):
                await subscription.callback(event)
            else:
                subscription.callback(event)

        except Exception as e:
            self.failed_deliveries[subscription.service_name] += 1
            logger.error(
                f"Failed to deliver event '{event.event_type}' to " f"service '{subscription.service_name}': {e}"
            )

    def get_event_history(
        self, event_type: Optional[str] = None, service_name: Optional[str] = None, limit: int = 100
    ) -> List[RoutingEvent]:
        """Get recent event history with optional filtering"""
        events = self.event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if service_name:
            events = [e for e in events if e.source_service == service_name]

        return events[-limit:] if limit > 0 else events

    def get_subscription_info(self) -> Dict[str, Any]:
        """Get information about current subscriptions"""
        active_subs = [s for s in self.subscriptions.values() if s.active]

        service_counts = defaultdict(int)
        event_type_counts = defaultdict(int)

        for sub in active_subs:
            service_counts[sub.service_name] += 1
            event_type_counts[sub.event_type] += 1

        return {
            "total_subscriptions": len(active_subs),
            "subscriptions_by_service": dict(service_counts),
            "subscriptions_by_event_type": dict(event_type_counts),
            "active_services": list(service_counts.keys()),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get event bus performance metrics"""
        return {
            "event_counts": dict(self.event_counts),
            "processing_times": dict(self.processing_times),
            "failed_deliveries": dict(self.failed_deliveries),
            "queue_size": self.event_queue.qsize(),
            "event_history_size": len(self.event_history),
            "total_events_processed": sum(self.event_counts.values()),
            "total_failed_deliveries": sum(self.failed_deliveries.values()),
        }

    def clear_history(self) -> None:
        """Clear event history"""
        self.event_history.clear()
        logger.info("Event history cleared")

    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        self.event_counts.clear()
        self.processing_times.clear()
        self.failed_deliveries.clear()
        logger.info("Event bus metrics reset")


# Global event bus instance
_event_bus: Optional[EventBusService] = None


def get_event_bus() -> EventBusService:
    """Get the global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBusService()
    return _event_bus


async def initialize_event_bus() -> EventBusService:
    """Initialize and start the global event bus"""
    event_bus = get_event_bus()
    if not event_bus.running:
        await event_bus.start()
    return event_bus


async def shutdown_event_bus() -> None:
    """Shutdown the global event bus"""
    global _event_bus
    if _event_bus and _event_bus.running:
        await _event_bus.stop()
        _event_bus = None
