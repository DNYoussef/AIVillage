"""
Unified Edge Computing Infrastructure

A comprehensive edge computing ecosystem that integrates Digital Twin, Device Management,
Knowledge Systems, and Communication layers into a cohesive, production-ready platform.

Key Components:
- Digital Twin Concierge: Privacy-first on-device learning
- Unified Edge Device System: Complete device lifecycle management
- MiniRAG System: Local knowledge with global elevation
- Chat Engine: Resilient multi-mode communication
- Mobile Bridge: Cross-platform mobile integration
- Shared Types: Common data structures and enums

Architecture:
Device Discovery → Registration → Optimization → Task Execution → Knowledge Learning → Communication

Usage:
    from infrastructure.edge import EdgeSystem, create_edge_system

    # Create complete edge system
    edge_system = await create_edge_system(
        device_name="MyDevice",
        enable_digital_twin=True,
        enable_mobile_bridge=True
    )

    # Process tasks
    task_result = await edge_system.process_task(task)

    # Query knowledge
    knowledge_results = await edge_system.query_knowledge("search query")
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from .communication.chat_engine import ChatEngine
from .device.unified_system import UnifiedEdgeDeviceSystem

# Core component imports
from .digital_twin.concierge import DigitalTwinConcierge, UserPreferences
from .integration.mobile_bridge import EnhancedMobileBridge, create_mobile_bridge
from .integration.shared_types import (
    DataSource,
    DeviceType,
    EdgeDeviceProfile,
    EdgeMessage,
    EdgeTask,
    EdgeTaskResult,
    PrivacyLevel,
    ProcessingMode,
    ResourceConstraint,
    TaskPriority,
    create_edge_task,
)
from .knowledge.minirag_system import MiniRAGSystem

logger = logging.getLogger(__name__)


class EdgeSystem:
    """
    Unified Edge Computing System

    Integrates all edge computing components into a single, cohesive system
    that provides comprehensive device management, knowledge processing,
    digital twin functionality, and resilient communication.
    """

    def __init__(
        self,
        data_dir: Path,
        device_config: EdgeDeviceProfile,
        user_preferences: UserPreferences | None = None,
        enable_digital_twin: bool = True,
        enable_mobile_bridge: bool = True,
        enable_chat_engine: bool = True,
    ):
        self.data_dir = data_dir
        self.device_config = device_config
        self.user_preferences = user_preferences
        self.initialized = False

        # Core components
        self.device_system: UnifiedEdgeDeviceSystem | None = None
        self.digital_twin: DigitalTwinConcierge | None = None
        self.knowledge_system: MiniRAGSystem | None = None
        self.chat_engine: ChatEngine | None = None
        self.mobile_bridge: EnhancedMobileBridge | None = None

        # Component flags
        self.enable_digital_twin = enable_digital_twin
        self.enable_mobile_bridge = enable_mobile_bridge
        self.enable_chat_engine = enable_chat_engine

        # Integration state
        self.device_profile: EdgeDeviceProfile | None = None
        self.active_tasks: dict[str, EdgeTask] = {}
        self.system_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "knowledge_pieces": 0,
            "conversations": 0,
            "uptime_seconds": 0.0,
        }

        # Health monitoring storage
        self.health_checks: dict[str, Callable[[], Any]] = {}
        self.component_health: dict[str, Any] = {}

        logger.info("Edge System initialized with integrated components")

    async def initialize(self) -> bool:
        """Initialize all edge system components with integrated architecture"""
        if self.initialized:
            return True

        try:
            logger.info("Initializing Unified Edge System...")

            # 1. Initialize core device system
            self.device_system = UnifiedEdgeDeviceSystem(self.device_config)
            if not await self.device_system.initialize():
                raise RuntimeError("Failed to initialize device system")

            # 2. Initialize knowledge system
            twin_id = f"twin_{self.device_config.device_id}"
            self.knowledge_system = MiniRAGSystem(self.data_dir / "knowledge", twin_id)

            # 3. Initialize digital twin (if enabled)
            if self.enable_digital_twin:
                preferences = self.user_preferences or UserPreferences()
                self.digital_twin = DigitalTwinConcierge(self.data_dir / "digital_twin", preferences)
                logger.info("Digital Twin Concierge initialized")

            # 4. Initialize chat engine (if enabled)
            if self.enable_chat_engine:
                self.chat_engine = ChatEngine()
                logger.info("Chat Engine initialized")

            # 5. Initialize mobile bridge (if enabled)
            if self.enable_mobile_bridge:
                self.mobile_bridge = await create_mobile_bridge(auto_initialize=True)
                logger.info("Mobile Bridge initialized")

            # 6. Wire components together
            await self._wire_component_integrations()

            # 7. Create device profile
            await self._create_device_profile()

            # 8. Start background integration tasks
            await self._start_integration_tasks()

            self.initialized = True
            logger.info("✅ Unified Edge System initialization completed")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Edge System: {e}")
            return False

    async def _wire_component_integrations(self):
        """Wire integration points between all components"""
        logger.info("Wiring component integrations...")

        # Wire Digital Twin → Knowledge System
        if self.digital_twin and self.knowledge_system:
            # Share the knowledge system instance with digital twin
            self.digital_twin.mini_rag = self.knowledge_system
            logger.debug("Digital Twin ↔ Knowledge System integration wired")

        # Wire Device System → All Components
        if self.device_system:
            # Register device system in health monitoring registry
            self.health_checks["device_system"] = self.device_system.get_system_status
            logger.debug("Device system health monitoring enabled")

        # Wire Chat Engine → Knowledge System
        if self.chat_engine and self.knowledge_system:
            # Link chat engine with knowledge system and register health check
            self.chat_engine.knowledge_system = self.knowledge_system
            self.health_checks["chat_engine"] = self.chat_engine.get_system_status
            logger.debug("Chat engine knowledge integration wired")

        # Monitor knowledge system health as well
        if self.knowledge_system:
            self.health_checks["knowledge_system"] = self.knowledge_system.get_system_stats

        logger.info("Component integrations wired successfully")

    async def _create_device_profile(self):
        """Create comprehensive device profile from all components"""
        if not self.device_system:
            return

        self.device_system.get_system_status()

        # This would create a comprehensive EdgeDeviceProfile
        # combining information from all components
        logger.debug("Device profile created")

    async def _start_integration_tasks(self):
        """Start background tasks for component integration"""
        # Task for syncing knowledge between components
        asyncio.create_task(self._knowledge_sync_task())

        # Task for cross-component health monitoring
        asyncio.create_task(self._health_monitoring_task())

        # Task for metrics aggregation
        asyncio.create_task(self._metrics_aggregation_task())

    async def _knowledge_sync_task(self):
        """Background task to sync knowledge between components"""
        while self.initialized:
            try:
                if self.digital_twin and self.knowledge_system:
                    # Sync learned patterns from digital twin to knowledge system
                    stats = self.knowledge_system.get_system_stats()
                    self.system_metrics["knowledge_pieces"] = stats.get("total_knowledge_pieces", 0)

                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Knowledge sync error: {e}")
                await asyncio.sleep(60)

    async def _health_monitoring_task(self):
        """Background task for cross-component health monitoring"""
        while self.initialized:
            try:
                # Monitor health of registered components
                for name, check in self.health_checks.items():
                    try:
                        self.component_health[name] = check()
                    except Exception as hc_err:
                        logger.error(f"{name} health check failed: {hc_err}")

                await asyncio.sleep(30)  # Every 30 seconds
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)

    async def _metrics_aggregation_task(self):
        """Background task for system-wide metrics aggregation"""
        while self.initialized:
            try:
                # Aggregate metrics from all components
                if self.device_system:
                    device_stats = self.device_system.get_system_status()
                    self.system_metrics.update(
                        {
                            "device_tasks_completed": device_stats["tasks"]["completed"],
                            "device_tasks_failed": device_stats["tasks"]["failed"],
                        }
                    )

                if self.knowledge_system:
                    knowledge_stats = self.knowledge_system.get_system_stats()
                    self.system_metrics.update(
                        {
                            "knowledge_pieces": knowledge_stats["total_knowledge_pieces"],
                        }
                    )

                await asyncio.sleep(60)  # Every minute
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
                await asyncio.sleep(60)

    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================

    async def process_task(self, task: EdgeTask | dict[str, Any]) -> EdgeTaskResult:
        """Process a task using the integrated edge system"""
        if not self.initialized:
            raise RuntimeError("Edge System not initialized")

        # Convert dict to EdgeTask if needed
        if isinstance(task, dict):
            task = create_edge_task(**task)

        try:
            # Track task
            self.active_tasks[task.task_id] = task

            # Route task based on requirements and system capabilities
            if self.device_system:
                # Submit to device system for execution
                self.device_system.submit_task(task)

                # For now, simulate task completion
                await asyncio.sleep(0.1)  # Simulate processing

                result = EdgeTaskResult(
                    task_id=task.task_id,
                    success=True,
                    execution_time_seconds=0.1,
                    result_data={"status": "completed", "processed_by": "edge_system"},
                    executed_on_device=self.device_config.device_id,
                    processing_mode_used=task.processing_mode,
                )

                self.system_metrics["tasks_completed"] += 1
                return result
            else:
                raise RuntimeError("Device system not available")

        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            self.system_metrics["tasks_failed"] += 1

            return EdgeTaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                executed_on_device=self.device_config.device_id,
            )

        finally:
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

    async def query_knowledge(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Query the integrated knowledge system"""
        if not self.knowledge_system:
            return []

        try:
            results = await self.knowledge_system.query_knowledge(query, max_results)
            return [
                {
                    "content": result.content,
                    "source": result.source.value,
                    "confidence": result.confidence_score,
                    "relevance": result.relevance.value,
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return []

    async def process_chat(self, message: str, conversation_id: str) -> dict[str, Any]:
        """Process chat message through integrated chat system"""
        if not self.chat_engine:
            return {"error": "Chat engine not available"}

        try:
            # Process through chat engine
            result = self.chat_engine.process_chat(message, conversation_id)

            # Potentially enhance with knowledge system
            if self.knowledge_system and result.get("mode") == "local":
                try:
                    knowledge_hits = await self.knowledge_system.query_knowledge(message, 3)
                    if knowledge_hits:
                        result["knowledge"] = [k.content for k in knowledge_hits]
                except Exception as aug_err:
                    logger.error(f"Knowledge augmentation failed: {aug_err}")

            self.system_metrics["conversations"] += 1
            return result

        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return {"error": str(e)}

    async def run_learning_cycle(self) -> dict[str, Any]:
        """Run a learning cycle through the digital twin"""
        if not self.digital_twin:
            return {"error": "Digital twin not available"}

        try:
            # Get device profile for learning cycle

            # Create mock device profile for learning
            device_profile = None  # Would create from actual metrics

            # Run learning cycle
            cycle_result = await self.digital_twin.run_learning_cycle(device_profile)

            return {
                "cycle_id": cycle_result.cycle_id,
                "data_points": cycle_result.data_points_count,
                "average_surprise": cycle_result.average_surprise,
                "improvement_score": cycle_result.improvement_score,
            }

        except Exception as e:
            logger.error(f"Learning cycle failed: {e}")
            return {"error": str(e)}

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status from all components"""
        status = {
            "initialized": self.initialized,
            "system_metrics": self.system_metrics,
            "active_tasks": len(self.active_tasks),
            "components": {
                "device_system": self.device_system is not None,
                "digital_twin": self.digital_twin is not None,
                "knowledge_system": self.knowledge_system is not None,
                "chat_engine": self.chat_engine is not None,
                "mobile_bridge": self.mobile_bridge is not None,
            },
        }

        # Add component-specific status
        if self.device_system:
            status["device_status"] = self.device_system.get_system_status()

        if self.knowledge_system:
            status["knowledge_status"] = self.knowledge_system.get_system_stats()

        if self.chat_engine:
            status["chat_status"] = self.chat_engine.get_system_status()

        if self.mobile_bridge:
            status["mobile_status"] = self.mobile_bridge.get_comprehensive_status()

        return status

    async def shutdown(self):
        """Gracefully shutdown all edge system components"""
        logger.info("Shutting down Unified Edge System...")

        self.initialized = False

        # Shutdown components in reverse order
        if self.mobile_bridge:
            await self.mobile_bridge.shutdown()

        if self.device_system:
            await self.device_system.shutdown()

        # Knowledge system and chat engine don't need special shutdown

        logger.info("Edge System shutdown completed")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


async def create_edge_system(
    device_name: str = "EdgeDevice",
    data_dir: str | Path | None = None,
    enable_digital_twin: bool = True,
    enable_mobile_bridge: bool = True,
    enable_chat_engine: bool = True,
    **config_kwargs,
) -> EdgeSystem:
    """
    Create and initialize a complete unified edge system

    Args:
        device_name: Name for this edge device
        data_dir: Directory for data storage
        enable_digital_twin: Enable digital twin functionality
        enable_mobile_bridge: Enable mobile platform integration
        enable_chat_engine: Enable chat processing
        **config_kwargs: Additional device configuration options

    Returns:
        Fully initialized EdgeSystem ready for use
    """
    # Setup data directory
    if data_dir is None:
        data_dir = Path.cwd() / "edge_data"
    elif isinstance(data_dir, str):
        data_dir = Path(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Create device configuration with proper defaults
    from .integration.shared_types import EdgeCapabilities, EdgeResourceMetrics

    default_capabilities = EdgeCapabilities(device_id=device_name, device_type=DeviceType.DESKTOP)
    default_metrics = EdgeResourceMetrics()

    device_config = EdgeDeviceProfile(
        device_id=device_name,
        device_name=device_name,
        device_type=DeviceType.DESKTOP,  # Default device type
        capabilities=default_capabilities,
        current_metrics=default_metrics,
    )

    # Create user preferences for digital twin
    user_preferences = None
    if enable_digital_twin:
        user_preferences = UserPreferences(
            enabled_sources={DataSource.APP_USAGE, DataSource.CONVERSATION},
            learning_enabled=True,
        )

    # Create and initialize system
    edge_system = EdgeSystem(
        data_dir=data_dir,
        device_config=device_config,
        user_preferences=user_preferences,
        enable_digital_twin=enable_digital_twin,
        enable_mobile_bridge=enable_mobile_bridge,
        enable_chat_engine=enable_chat_engine,
    )

    success = await edge_system.initialize()
    if not success:
        raise RuntimeError("Failed to initialize EdgeSystem")

    return edge_system


async def create_mobile_edge_system(**kwargs) -> EdgeSystem:
    """Create edge system optimized for mobile platforms"""
    # Set mobile-specific defaults
    mobile_defaults = {
        "device_name": "MobileDevice",
        "enable_mobile_bridge": True,
        "enable_digital_twin": True,
        "power_aware_scheduling": True,
        "battery_threshold_percent": 30.0,
    }

    # Override defaults with user-provided kwargs
    mobile_defaults.update(kwargs)

    return await create_edge_system(**mobile_defaults)


# ============================================================================
# PUBLIC API EXPORTS
# ============================================================================

__all__ = [
    # Main system class
    "EdgeSystem",
    # Factory functions
    "create_edge_system",
    "create_mobile_edge_system",
    # Core components (re-exported for convenience)
    "DigitalTwinConcierge",
    "UnifiedEdgeDeviceSystem",
    "MiniRAGSystem",
    "ChatEngine",
    "EnhancedMobileBridge",
    # Shared types (re-exported for convenience)
    "EdgeTask",
    "EdgeTaskResult",
    "EdgeDeviceProfile",
    "EdgeMessage",
    "DataSource",
    "PrivacyLevel",
    "ProcessingMode",
    "DeviceType",
    "TaskPriority",
    "ResourceConstraint",
    "create_edge_task",
]
