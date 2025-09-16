"""
Data Flow Coordinator for Phase 6 Integration Pipeline

This module manages robust data flow between all Phase 6 components,
ensuring consistent state management, error recovery, and reliable
cross-component communication.
"""

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import uuid

from .serialization_utils import (
    SafeJSONSerializer, HybridSerializer, SerializationConfig,
    SerializationError
)
from .state_manager import StateManager, Phase, StateStatus

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Message types for data flow communication"""
    DATA_TRANSFER = "data_transfer"
    STATE_UPDATE = "state_update"
    ERROR_NOTIFICATION = "error_notification"
    HEALTH_CHECK = "health_check"
    SYNCHRONIZATION = "synchronization"
    CHECKPOINT = "checkpoint"
    RECOVERY = "recovery"

class ComponentStatus(Enum):
    """Component status enumeration"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class DataFlowMessage:
    """Message structure for data flow communication"""
    message_id: str
    timestamp: datetime
    message_type: MessageType
    source_component: str
    target_component: Optional[str]
    data: Dict[str, Any]
    priority: int = 5  # 1-10, higher = more priority
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 30.0

@dataclass
class ComponentInfo:
    """Information about a registered component"""
    component_id: str
    component_type: str
    status: ComponentStatus
    last_heartbeat: datetime
    message_queue: deque
    error_count: int
    total_messages_processed: int
    handler_function: Optional[Callable]

@dataclass
class DataFlowMetrics:
    """Metrics for data flow monitoring"""
    total_messages_sent: int = 0
    total_messages_delivered: int = 0
    total_messages_failed: int = 0
    average_delivery_time_ms: float = 0.0
    active_components: int = 0
    error_rate: float = 0.0
    throughput_msgs_per_sec: float = 0.0

class DataFlowCoordinator:
    """
    Central coordinator for managing data flow between Phase 6 components.

    Provides:
    - Reliable message passing between components
    - State synchronization across components
    - Error recovery and retry mechanisms
    - Health monitoring and diagnostics
    - Serialization handling for complex data types
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.coordinator_id = str(uuid.uuid4())

        # Initialize serialization
        serialization_config = SerializationConfig(
            use_compression=config.get('use_compression', True),
            include_metadata=config.get('include_metadata', True),
            fail_on_unsupported=config.get('fail_on_unsupported', False)
        )
        self.serializer = HybridSerializer(serialization_config)

        # Component management
        self.components: Dict[str, ComponentInfo] = {}
        self.component_lock = threading.RLock()

        # Message management
        self.message_queue = asyncio.Queue()
        self.message_history = deque(maxlen=config.get('message_history_size', 1000))
        self.pending_messages: Dict[str, DataFlowMessage] = {}

        # State management
        self.state_manager = StateManager(config.get('state_config', {}))

        # Metrics and monitoring
        self.metrics = DataFlowMetrics()
        self.metrics_lock = threading.Lock()
        self.start_time = time.time()

        # Configuration
        self.heartbeat_interval = config.get('heartbeat_interval_seconds', 10)
        self.message_timeout = config.get('message_timeout_seconds', 30)
        self.max_queue_size = config.get('max_queue_size', 1000)
        self.cleanup_interval = config.get('cleanup_interval_seconds', 60)

        # Background tasks
        self.running = False
        self.background_tasks = []

        logger.info(f"DataFlowCoordinator initialized with ID: {self.coordinator_id}")

    async def start(self):
        """Start the data flow coordinator"""
        if self.running:
            logger.warning("DataFlowCoordinator already running")
            return

        self.running = True
        logger.info("Starting DataFlowCoordinator...")

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._cleanup_manager()),
            asyncio.create_task(self._metrics_updater())
        ]

        logger.info("DataFlowCoordinator started successfully")

    async def stop(self):
        """Stop the data flow coordinator"""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping DataFlowCoordinator...")

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

        logger.info("DataFlowCoordinator stopped")

    def register_component(self, component_id: str, component_type: str,
                          handler_function: Optional[Callable] = None) -> bool:
        """Register a component with the data flow coordinator"""
        try:
            with self.component_lock:
                if component_id in self.components:
                    logger.warning(f"Component {component_id} already registered")
                    return False

                component_info = ComponentInfo(
                    component_id=component_id,
                    component_type=component_type,
                    status=ComponentStatus.INITIALIZING,
                    last_heartbeat=datetime.now(),
                    message_queue=deque(maxlen=self.max_queue_size),
                    error_count=0,
                    total_messages_processed=0,
                    handler_function=handler_function
                )

                self.components[component_id] = component_info

                with self.metrics_lock:
                    self.metrics.active_components += 1

                logger.info(f"Registered component: {component_id} ({component_type})")
                return True

        except Exception as e:
            logger.error(f"Failed to register component {component_id}: {e}")
            return False

    def unregister_component(self, component_id: str) -> bool:
        """Unregister a component from the data flow coordinator"""
        try:
            with self.component_lock:
                if component_id not in self.components:
                    logger.warning(f"Component {component_id} not found")
                    return False

                del self.components[component_id]

                with self.metrics_lock:
                    self.metrics.active_components -= 1

                logger.info(f"Unregistered component: {component_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to unregister component {component_id}: {e}")
            return False

    async def send_message(self, message: DataFlowMessage) -> bool:
        """Send a message through the data flow system"""
        try:
            # Validate message
            if not self._validate_message(message):
                return False

            # Serialize the message data
            serialized_data, format_used = self.serializer.serialize(
                message.data, prefer_json=True
            )
            message.data['__serialization_format__'] = format_used
            message.data['__serialized_size__'] = len(serialized_data)

            # Add to queue
            await self.message_queue.put(message)

            # Track in pending messages
            self.pending_messages[message.message_id] = message

            # Update metrics
            with self.metrics_lock:
                self.metrics.total_messages_sent += 1

            logger.debug(f"Message {message.message_id} queued for delivery")
            return True

        except Exception as e:
            logger.error(f"Failed to send message {message.message_id}: {e}")
            return False

    async def send_data(self, source_component: str, target_component: str,
                       data: Dict[str, Any], message_type: MessageType = MessageType.DATA_TRANSFER,
                       priority: int = 5) -> str:
        """Convenience method to send data between components"""
        message = DataFlowMessage(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            message_type=message_type,
            source_component=source_component,
            target_component=target_component,
            data=data,
            priority=priority
        )

        success = await self.send_message(message)
        return message.message_id if success else ""

    async def broadcast_message(self, source_component: str, data: Dict[str, Any],
                               message_type: MessageType = MessageType.DATA_TRANSFER,
                               exclude_components: Optional[List[str]] = None) -> List[str]:
        """Broadcast a message to all registered components"""
        message_ids = []
        exclude_components = exclude_components or []

        with self.component_lock:
            for component_id in self.components:
                if component_id != source_component and component_id not in exclude_components:
                    message_id = await self.send_data(
                        source_component, component_id, data, message_type
                    )
                    if message_id:
                        message_ids.append(message_id)

        return message_ids

    def update_component_status(self, component_id: str, status: ComponentStatus):
        """Update component status"""
        with self.component_lock:
            if component_id in self.components:
                self.components[component_id].status = status
                self.components[component_id].last_heartbeat = datetime.now()
                logger.debug(f"Component {component_id} status updated to {status.value}")

    def send_heartbeat(self, component_id: str) -> bool:
        """Send heartbeat for a component"""
        try:
            with self.component_lock:
                if component_id in self.components:
                    self.components[component_id].last_heartbeat = datetime.now()
                    return True
                return False

        except Exception as e:
            logger.error(f"Failed to send heartbeat for {component_id}: {e}")
            return False

    async def synchronize_state(self, component_id: str, state_data: Dict[str, Any]) -> bool:
        """Synchronize state across components"""
        try:
            # Create state in state manager
            state_id = f"sync_{component_id}_{int(time.time())}"
            success = self.state_manager.create_state(
                state_id=state_id,
                phase=Phase.PHASE6_BAKING,
                data=state_data,
                tags={'component': component_id, 'type': 'synchronization'}
            )

            if success:
                # Broadcast synchronization message
                await self.broadcast_message(
                    source_component=component_id,
                    data={
                        'state_id': state_id,
                        'synchronization_data': state_data
                    },
                    message_type=MessageType.SYNCHRONIZATION,
                    exclude_components=[component_id]
                )

            return success

        except Exception as e:
            logger.error(f"Failed to synchronize state for {component_id}: {e}")
            return False

    async def create_checkpoint(self, checkpoint_name: str,
                               include_components: Optional[List[str]] = None) -> bool:
        """Create a checkpoint of current system state"""
        try:
            # Collect state from all components
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'coordinator_id': self.coordinator_id,
                'metrics': asdict(self.metrics),
                'components': {}
            }

            include_components = include_components or list(self.components.keys())

            with self.component_lock:
                for component_id in include_components:
                    if component_id in self.components:
                        component = self.components[component_id]
                        checkpoint_data['components'][component_id] = {
                            'component_type': component.component_type,
                            'status': component.status.value,
                            'last_heartbeat': component.last_heartbeat.isoformat(),
                            'error_count': component.error_count,
                            'total_messages_processed': component.total_messages_processed,
                            'queue_size': len(component.message_queue)
                        }

            # Create checkpoint in state manager
            success = self.state_manager.create_checkpoint(
                checkpoint_name=checkpoint_name,
                phases=[Phase.PHASE6_BAKING]
            )

            if success:
                # Save coordinator checkpoint
                checkpoint_file = Path(self.state_manager.storage_dir) / 'checkpoints' / checkpoint_name / 'coordinator_state.json'
                self.serializer.serialize_to_file(checkpoint_data, checkpoint_file)

            return success

        except Exception as e:
            logger.error(f"Failed to create checkpoint {checkpoint_name}: {e}")
            return False

    async def restore_checkpoint(self, checkpoint_name: str) -> bool:
        """Restore system state from checkpoint"""
        try:
            # Restore state manager checkpoint
            success = self.state_manager.restore_checkpoint(
                checkpoint_name=checkpoint_name,
                overwrite_existing=True
            )

            if success:
                # Restore coordinator state
                checkpoint_file = Path(self.state_manager.storage_dir) / 'checkpoints' / checkpoint_name / 'coordinator_state.json'
                if checkpoint_file.exists():
                    checkpoint_data = self.serializer.deserialize_from_file(checkpoint_file)

                    # Restore metrics
                    if 'metrics' in checkpoint_data:
                        metrics_data = checkpoint_data['metrics']
                        with self.metrics_lock:
                            for key, value in metrics_data.items():
                                if hasattr(self.metrics, key):
                                    setattr(self.metrics, key, value)

                    logger.info(f"Restored coordinator state from checkpoint {checkpoint_name}")

            return success

        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_name}: {e}")
            return False

    def get_component_status(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a component"""
        with self.component_lock:
            if component_id not in self.components:
                return None

            component = self.components[component_id]
            return {
                'component_id': component_id,
                'component_type': component.component_type,
                'status': component.status.value,
                'last_heartbeat': component.last_heartbeat.isoformat(),
                'queue_size': len(component.message_queue),
                'error_count': component.error_count,
                'total_messages_processed': component.total_messages_processed,
                'is_healthy': self._is_component_healthy(component)
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        with self.component_lock, self.metrics_lock:
            component_statuses = {}
            healthy_components = 0

            for component_id, component in self.components.items():
                is_healthy = self._is_component_healthy(component)
                component_statuses[component_id] = {
                    'type': component.component_type,
                    'status': component.status.value,
                    'healthy': is_healthy,
                    'last_heartbeat': component.last_heartbeat.isoformat(),
                    'error_count': component.error_count
                }
                if is_healthy:
                    healthy_components += 1

            return {
                'coordinator_id': self.coordinator_id,
                'uptime_seconds': time.time() - self.start_time,
                'total_components': len(self.components),
                'healthy_components': healthy_components,
                'system_health': 'HEALTHY' if healthy_components == len(self.components) else 'DEGRADED',
                'metrics': asdict(self.metrics),
                'pending_messages': len(self.pending_messages),
                'queue_size': self.message_queue.qsize(),
                'components': component_statuses
            }

    def validate_data_flow_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of the data flow system"""
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'HEALTHY',
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'component_issues': {},
            'metrics_summary': {}
        }

        try:
            with self.component_lock, self.metrics_lock:
                # Check component health
                unhealthy_components = []
                for component_id, component in self.components.items():
                    if not self._is_component_healthy(component):
                        unhealthy_components.append(component_id)
                        validation_results['component_issues'][component_id] = {
                            'status': component.status.value,
                            'error_count': component.error_count,
                            'last_heartbeat_age_seconds': (datetime.now() - component.last_heartbeat).total_seconds()
                        }

                if unhealthy_components:
                    validation_results['overall_health'] = 'DEGRADED'
                    validation_results['issues'].append(f"Unhealthy components: {unhealthy_components}")

                # Check message queue health
                if self.message_queue.qsize() > self.max_queue_size * 0.8:
                    validation_results['warnings'].append("Message queue near capacity")

                # Check error rates
                if self.metrics.error_rate > 0.1:  # 10% error rate threshold
                    validation_results['issues'].append(f"High error rate: {self.metrics.error_rate:.2%}")

                # Check delivery performance
                if self.metrics.average_delivery_time_ms > 1000:  # 1 second threshold
                    validation_results['warnings'].append(f"Slow message delivery: {self.metrics.average_delivery_time_ms:.1f}ms")

                # Generate recommendations
                if unhealthy_components:
                    validation_results['recommendations'].append("Restart unhealthy components")

                if len(validation_results['issues']) == 0:
                    validation_results['overall_health'] = 'HEALTHY'
                elif len(validation_results['issues']) > 2:
                    validation_results['overall_health'] = 'CRITICAL'

                validation_results['metrics_summary'] = {
                    'total_messages_sent': self.metrics.total_messages_sent,
                    'delivery_success_rate': (self.metrics.total_messages_delivered / max(1, self.metrics.total_messages_sent)),
                    'error_rate': self.metrics.error_rate,
                    'throughput_msgs_per_sec': self.metrics.throughput_msgs_per_sec,
                    'average_delivery_time_ms': self.metrics.average_delivery_time_ms
                }

        except Exception as e:
            validation_results['overall_health'] = 'ERROR'
            validation_results['issues'].append(f"Validation error: {e}")

        return validation_results

    # Private methods

    def _validate_message(self, message: DataFlowMessage) -> bool:
        """Validate message structure and contents"""
        try:
            # Check required fields
            if not message.message_id or not message.source_component:
                logger.error("Message missing required fields")
                return False

            # Check if source component exists
            with self.component_lock:
                if message.source_component not in self.components:
                    logger.error(f"Unknown source component: {message.source_component}")
                    return False

                # Check if target component exists (if specified)
                if (message.target_component and
                    message.target_component not in self.components):
                    logger.error(f"Unknown target component: {message.target_component}")
                    return False

            # Check data serialization
            try:
                self.serializer.serialize(message.data)
            except SerializationError:
                logger.error(f"Message data cannot be serialized: {message.message_id}")
                return False

            return True

        except Exception as e:
            logger.error(f"Message validation error: {e}")
            return False

    def _is_component_healthy(self, component: ComponentInfo) -> bool:
        """Check if a component is healthy"""
        # Check heartbeat age
        heartbeat_age = (datetime.now() - component.last_heartbeat).total_seconds()
        if heartbeat_age > self.heartbeat_interval * 3:  # 3x heartbeat interval threshold
            return False

        # Check error rate
        if component.total_messages_processed > 0:
            error_rate = component.error_count / component.total_messages_processed
            if error_rate > 0.2:  # 20% error rate threshold
                return False

        # Check status
        if component.status in [ComponentStatus.ERROR, ComponentStatus.OFFLINE]:
            return False

        return True

    async def _message_processor(self):
        """Background task to process messages"""
        while self.running:
            try:
                # Get message from queue with timeout
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the message
                await self._process_single_message(message)

            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(1)

    async def _process_single_message(self, message: DataFlowMessage):
        """Process a single message"""
        start_time = time.time()

        try:
            # Check if message has expired
            age_seconds = (datetime.now() - message.timestamp).total_seconds()
            if age_seconds > message.timeout_seconds:
                logger.warning(f"Message {message.message_id} expired")
                self._mark_message_failed(message.message_id)
                return

            # Deliver message to target component(s)
            if message.target_component:
                success = await self._deliver_to_component(message, message.target_component)
            else:
                # Broadcast message
                success = await self._broadcast_to_all_components(message)

            # Update metrics
            delivery_time_ms = (time.time() - start_time) * 1000

            with self.metrics_lock:
                if success:
                    self.metrics.total_messages_delivered += 1
                    # Update rolling average
                    if self.metrics.average_delivery_time_ms == 0:
                        self.metrics.average_delivery_time_ms = delivery_time_ms
                    else:
                        self.metrics.average_delivery_time_ms = (
                            self.metrics.average_delivery_time_ms * 0.9 +
                            delivery_time_ms * 0.1
                        )
                else:
                    self.metrics.total_messages_failed += 1

            # Remove from pending messages
            self.pending_messages.pop(message.message_id, None)

            # Add to history
            self.message_history.append({
                'message_id': message.message_id,
                'timestamp': message.timestamp.isoformat(),
                'source': message.source_component,
                'target': message.target_component,
                'type': message.message_type.value,
                'success': success,
                'delivery_time_ms': delivery_time_ms
            })

        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            self._mark_message_failed(message.message_id)

    async def _deliver_to_component(self, message: DataFlowMessage, target_component: str) -> bool:
        """Deliver message to a specific component"""
        try:
            with self.component_lock:
                if target_component not in self.components:
                    logger.error(f"Target component not found: {target_component}")
                    return False

                component = self.components[target_component]

                # Add message to component queue
                component.message_queue.append(message)

                # Call handler function if available
                if component.handler_function:
                    try:
                        # Call handler in a separate thread to avoid blocking
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            component.handler_function,
                            message
                        )
                    except Exception as e:
                        logger.error(f"Error in component handler for {target_component}: {e}")
                        component.error_count += 1
                        return False

                component.total_messages_processed += 1
                logger.debug(f"Message {message.message_id} delivered to {target_component}")
                return True

        except Exception as e:
            logger.error(f"Error delivering message to {target_component}: {e}")
            return False

    async def _broadcast_to_all_components(self, message: DataFlowMessage) -> bool:
        """Broadcast message to all components except sender"""
        success_count = 0
        total_components = 0

        with self.component_lock:
            for component_id in self.components:
                if component_id != message.source_component:
                    total_components += 1
                    if await self._deliver_to_component(message, component_id):
                        success_count += 1

        # Consider broadcast successful if delivered to majority
        return success_count >= (total_components / 2) if total_components > 0 else True

    def _mark_message_failed(self, message_id: str):
        """Mark a message as failed"""
        with self.metrics_lock:
            self.metrics.total_messages_failed += 1

        self.pending_messages.pop(message_id, None)

        # Add to history
        self.message_history.append({
            'message_id': message_id,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'failure_reason': 'Processing failed'
        })

    async def _health_monitor(self):
        """Background task to monitor component health"""
        while self.running:
            try:
                current_time = datetime.now()

                with self.component_lock:
                    for component_id, component in self.components.items():
                        # Check heartbeat age
                        heartbeat_age = (current_time - component.last_heartbeat).total_seconds()

                        if heartbeat_age > self.heartbeat_interval * 2:
                            if component.status != ComponentStatus.OFFLINE:
                                logger.warning(f"Component {component_id} appears offline (no heartbeat for {heartbeat_age:.1f}s)")
                                component.status = ComponentStatus.OFFLINE

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.heartbeat_interval)

    async def _cleanup_manager(self):
        """Background task to clean up old data"""
        while self.running:
            try:
                # Clean up old pending messages
                current_time = datetime.now()
                expired_messages = []

                for message_id, message in self.pending_messages.items():
                    age_seconds = (current_time - message.timestamp).total_seconds()
                    if age_seconds > message.timeout_seconds:
                        expired_messages.append(message_id)

                for message_id in expired_messages:
                    self._mark_message_failed(message_id)
                    logger.debug(f"Cleaned up expired message: {message_id}")

                await asyncio.sleep(self.cleanup_interval)

            except Exception as e:
                logger.error(f"Error in cleanup manager: {e}")
                await asyncio.sleep(self.cleanup_interval)

    async def _metrics_updater(self):
        """Background task to update metrics"""
        while self.running:
            try:
                with self.metrics_lock:
                    # Calculate error rate
                    total_messages = self.metrics.total_messages_sent
                    if total_messages > 0:
                        self.metrics.error_rate = self.metrics.total_messages_failed / total_messages

                    # Calculate throughput
                    uptime = time.time() - self.start_time
                    if uptime > 0:
                        self.metrics.throughput_msgs_per_sec = total_messages / uptime

                await asyncio.sleep(5)  # Update metrics every 5 seconds

            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(5)

# Factory functions
def create_data_flow_coordinator(config: Dict[str, Any]) -> DataFlowCoordinator:
    """Factory function to create data flow coordinator"""
    return DataFlowCoordinator(config)

# Testing utilities
async def test_data_flow_coordination():
    """Test data flow coordination functionality"""
    config = {
        'use_compression': True,
        'heartbeat_interval_seconds': 5,
        'message_timeout_seconds': 30,
        'max_queue_size': 100
    }

    coordinator = DataFlowCoordinator(config)

    try:
        await coordinator.start()

        # Register test components
        coordinator.register_component("test_component_1", "baking_agent")
        coordinator.register_component("test_component_2", "optimization_agent")

        # Send test message
        message_id = await coordinator.send_data(
            "test_component_1", "test_component_2",
            {"test_data": "hello world", "timestamp": datetime.now()}
        )

        print(f"Test message sent: {message_id}")

        # Check system status
        status = coordinator.get_system_status()
        print(f"System status: {status}")

        # Validate data flow
        validation = coordinator.validate_data_flow_integrity()
        print(f"Validation results: {validation}")

        await asyncio.sleep(2)  # Let messages process

    finally:
        await coordinator.stop()

if __name__ == "__main__":
    asyncio.run(test_data_flow_coordination())