#!/usr/bin/env python3
"""
Digital Twin Integration - External system interfaces

Extracted from UnifiedDigitalTwinSystem to handle P2P, fog computing,
and external service integrations following Single Responsibility Principle.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IntegrationStatus:
    """Status information for external integrations."""

    service_name: str
    is_connected: bool
    last_sync: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None


class DigitalTwinIntegration:
    """
    Digital Twin External Integration Management

    Handles P2P networks, fog computing, chat engines, and other
    external service integrations with proper isolation.
    """

    def __init__(self, twin_id: str, enable_p2p: bool = True):
        self.twin_id = twin_id
        self.enable_p2p = enable_p2p

        # Integration components
        self.p2p_system = None
        self.fog_system = None
        self.chat_engine = None
        self.config_manager = None

        # Integration status tracking
        self.integrations: Dict[str, IntegrationStatus] = {}

        # Integration metrics
        self.integration_metrics = {
            "p2p_sync_operations": 0,
            "fog_tasks_executed": 0,
            "chat_interactions": 0,
            "config_updates": 0,
        }

    async def initialize_integrations(self, data_dir: Path):
        """Initialize all external integrations."""

        logger.info("Initializing Digital Twin Integrations...")

        # Initialize P2P integration
        if self.enable_p2p:
            await self._initialize_p2p_integration()

        # Initialize chat engine
        await self._initialize_chat_engine(data_dir)

        # Initialize configuration management
        await self._initialize_configuration(data_dir)

        # Initialize fog computing if available
        await self._initialize_fog_computing()

        logger.info("✅ Digital Twin Integrations initialized")

    async def _initialize_p2p_integration(self):
        """Initialize P2P system integration with error handling."""

        try:
            # Import P2P system with dependency injection
            from core.decentralized_architecture.unified_p2p_system import create_decentralized_system

            self.p2p_system = create_decentralized_system(f"twin-{self.twin_id}")
            await self.p2p_system.start()

            # Register message handlers for twin operations
            self.p2p_system.register_message_handler("twin_sync", self._handle_p2p_sync)
            self.p2p_system.register_message_handler("twin_backup", self._handle_p2p_backup)
            self.p2p_system.register_message_handler("twin_query", self._handle_p2p_query)

            self.integrations["p2p"] = IntegrationStatus(
                service_name="P2P Network", is_connected=True, last_sync=time.time()
            )

            logger.info("P2P integration initialized successfully")

        except ImportError as e:
            logger.warning(f"P2P integration not available: {e}")
            self.integrations["p2p"] = IntegrationStatus(
                service_name="P2P Network", is_connected=False, last_error=str(e)
            )
        except Exception as e:
            logger.error(f"P2P integration failed: {e}")
            self.integrations["p2p"] = IntegrationStatus(
                service_name="P2P Network", is_connected=False, error_count=1, last_error=str(e)
            )

    async def _initialize_chat_engine(self, data_dir: Path):
        """Initialize chat engine integration."""

        try:
            # Try external chat engine first
            from infrastructure.twin.chat_engine import ChatEngine

            self.chat_engine = ChatEngine(twin_id=self.twin_id, data_dir=data_dir)

            await self.chat_engine.initialize()

            self.integrations["chat"] = IntegrationStatus(
                service_name="Chat Engine", is_connected=True, last_sync=time.time()
            )

            logger.info("External chat engine initialized")

        except ImportError:
            # Use built-in chat capabilities
            self.chat_engine = self

            self.integrations["chat"] = IntegrationStatus(
                service_name="Built-in Chat", is_connected=True, last_sync=time.time()
            )

            logger.info("Using built-in chat engine")

    async def _initialize_configuration(self, data_dir: Path):
        """Initialize configuration management integration."""

        try:
            from infrastructure.twin.config_manager import ConfigManager

            self.config_manager = ConfigManager(config_dir=data_dir / "config")
            await self.config_manager.load_configuration()

            self.integrations["config"] = IntegrationStatus(
                service_name="Config Manager", is_connected=True, last_sync=time.time()
            )

            logger.info("External configuration manager initialized")

        except ImportError:
            # Use built-in configuration
            self.config_manager = self

            self.integrations["config"] = IntegrationStatus(
                service_name="Built-in Config", is_connected=True, last_sync=time.time()
            )

            logger.info("Using built-in configuration manager")

    async def _initialize_fog_computing(self):
        """Initialize fog computing integration."""

        try:
            from infrastructure.fog.fog_computing import FogComputingNode

            self.fog_system = FogComputingNode(node_id=f"twin-{self.twin_id}")
            await self.fog_system.initialize()

            self.integrations["fog"] = IntegrationStatus(
                service_name="Fog Computing", is_connected=True, last_sync=time.time()
            )

            logger.info("Fog computing integration initialized")

        except ImportError:
            logger.info("Fog computing not available")
            self.integrations["fog"] = IntegrationStatus(
                service_name="Fog Computing", is_connected=False, last_error="Module not available"
            )

    async def generate_ai_response(
        self, conversation_id: str, user_message: str, user_id: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate AI response using available chat integration."""

        start_time = time.time()

        try:
            # Use external chat engine if available
            if hasattr(self.chat_engine, "generate_response"):
                response = await self.chat_engine.generate_response(
                    conversation_id=conversation_id, user_message=user_message, user_id=user_id, context=context or {}
                )
            else:
                # Fallback to built-in response generation
                response = await self._simulate_ai_response(user_message, context)

            processing_time = (time.time() - start_time) * 1000

            self.integration_metrics["chat_interactions"] += 1

            return {
                "content": response.get("content", ""),
                "tokens_used": response.get("tokens_used", 0),
                "processing_time_ms": processing_time,
                "model_used": response.get("model", "built-in"),
            }

        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return {
                "content": "I apologize, but I'm experiencing technical difficulties. Please try again.",
                "tokens_used": 0,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "error": str(e),
            }

    async def _simulate_ai_response(self, user_message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Built-in AI response simulation for fallback."""

        # Simple response patterns for demonstration
        response_templates = [
            "I understand you mentioned: '{message}'. Could you provide more details?",
            "That's an interesting point about '{message}'. How can I assist you further?",
            "Thank you for sharing: '{message}'. What would you like to know more about?",
        ]

        import random

        template = random.choice(response_templates)  # nosec B311
        response_content = template.format(message=user_message[:100])

        # Simulate token usage
        tokens_used = len(user_message.split()) * 2 + 50

        return {
            "content": response_content,
            "tokens_used": tokens_used,
            "model": "built-in-simulation",
        }

    async def sync_with_p2p_network(self, data_to_sync: Dict[str, Any]) -> bool:
        """Synchronize data with P2P network."""

        if not self.p2p_system or not self.integrations["p2p"].is_connected:
            logger.warning("P2P system not available for synchronization")
            return False

        try:
            # Create sync message
            sync_message = {
                "twin_id": self.twin_id,
                "sync_type": data_to_sync.get("type", "general"),
                "data": data_to_sync,
                "timestamp": time.time(),
            }

            # Send to P2P network
            await self.p2p_system.broadcast_message("twin_sync", json.dumps(sync_message))

            self.integration_metrics["p2p_sync_operations"] += 1
            self.integrations["p2p"].last_sync = time.time()

            logger.debug(f"Synced data with P2P network: {data_to_sync.get('type', 'unknown')}")
            return True

        except Exception as e:
            logger.error(f"P2P synchronization failed: {e}")
            self.integrations["p2p"].error_count += 1
            self.integrations["p2p"].last_error = str(e)
            return False

    async def _handle_p2p_sync(self, message):
        """Handle P2P synchronization requests."""

        try:
            payload = json.loads(message.payload.decode("utf-8"))
            sync_type = payload.get("sync_type")
            sender_twin = payload.get("twin_id")

            logger.debug(f"Received P2P sync from twin {sender_twin}: {sync_type}")

            # Process different sync types
            if sync_type == "user_data":
                await self._process_user_data_sync(payload)
            elif sync_type == "conversation":
                await self._process_conversation_sync(payload)
            elif sync_type == "configuration":
                await self._process_config_sync(payload)
            else:
                logger.warning(f"Unknown P2P sync type: {sync_type}")

            self.integration_metrics["p2p_sync_operations"] += 1

        except Exception as e:
            logger.error(f"P2P sync handling error: {e}")

    async def _handle_p2p_backup(self, message):
        """Handle P2P backup requests."""

        try:
            payload = json.loads(message.payload.decode("utf-8"))
            backup_type = payload.get("backup_type", "full")

            logger.info(f"P2P backup request: {backup_type} from {message.sender_id}")

            # Create backup response
            backup_data = await self._create_backup_data(backup_type)

            # Send backup response
            if backup_data:
                await self.p2p_system.send_direct_message(
                    message.sender_id, "twin_backup_response", json.dumps(backup_data)
                )

        except Exception as e:
            logger.error(f"P2P backup handling error: {e}")

    async def _handle_p2p_query(self, message):
        """Handle P2P query requests."""

        try:
            payload = json.loads(message.payload.decode("utf-8"))
            query_type = payload.get("query_type")
            query_data = payload.get("data", {})

            logger.debug(f"P2P query: {query_type} from {message.sender_id}")

            # Process query and generate response
            response = await self._process_p2p_query(query_type, query_data)

            # Send response back
            if response:
                await self.p2p_system.send_direct_message(
                    message.sender_id, "twin_query_response", json.dumps(response)
                )

        except Exception as e:
            logger.error(f"P2P query handling error: {e}")

    async def execute_fog_task(self, task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute task using fog computing if available."""

        if not self.fog_system or not self.integrations["fog"].is_connected:
            logger.debug("Fog computing not available for task execution")
            return None

        try:
            result = await self.fog_system.execute_task(task_data)

            self.integration_metrics["fog_tasks_executed"] += 1
            self.integrations["fog"].last_sync = time.time()

            return result

        except Exception as e:
            logger.error(f"Fog task execution failed: {e}")
            self.integrations["fog"].error_count += 1
            self.integrations["fog"].last_error = str(e)
            return None

    async def _process_user_data_sync(self, payload: Dict[str, Any]):
        """Process user data synchronization."""

        user_data = payload.get("data", {})
        user_id = user_data.get("user_id")

        if user_id:
            logger.debug(f"Processing user data sync for user {user_id}")
            # Additional user data sync logic would go here

    async def _process_conversation_sync(self, payload: Dict[str, Any]):
        """Process conversation synchronization."""

        conv_data = payload.get("data", {})
        conversation_id = conv_data.get("conversation_id")

        if conversation_id:
            logger.debug(f"Processing conversation sync for {conversation_id}")
            # Additional conversation sync logic would go here

    async def _process_config_sync(self, payload: Dict[str, Any]):
        """Process configuration synchronization."""

        config_data = payload.get("data", {})
        config_key = config_data.get("key")

        if config_key:
            logger.debug(f"Processing config sync for {config_key}")
            # Additional config sync logic would go here

    async def _create_backup_data(self, backup_type: str) -> Dict[str, Any]:
        """Create backup data for P2P sharing."""

        backup_data = {"twin_id": self.twin_id, "backup_type": backup_type, "timestamp": time.time(), "data": {}}

        # Add relevant data based on backup type
        if backup_type in ["full", "user_data"]:
            backup_data["data"]["user_count"] = len(getattr(self, "cached_users", []))

        if backup_type in ["full", "conversations"]:
            backup_data["data"]["conversation_count"] = len(getattr(self, "conversations", []))

        return backup_data

    async def _process_p2p_query(self, query_type: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process P2P query and generate response."""

        response = {"twin_id": self.twin_id, "query_type": query_type, "timestamp": time.time(), "result": {}}

        if query_type == "status":
            response["result"] = await self.get_integration_status()
        elif query_type == "metrics":
            response["result"] = self.get_integration_metrics()
        else:
            response["result"] = {"error": f"Unknown query type: {query_type}"}

        return response

    async def stop_integrations(self):
        """Stop all external integrations gracefully."""

        logger.info("Stopping Digital Twin Integrations...")

        # Stop P2P system
        if self.p2p_system:
            try:
                await self.p2p_system.stop()
                self.integrations["p2p"].is_connected = False
                logger.info("P2P system stopped")
            except Exception as e:
                logger.error(f"Error stopping P2P system: {e}")

        # Stop fog computing
        if self.fog_system:
            try:
                await self.fog_system.stop()
                self.integrations["fog"].is_connected = False
                logger.info("Fog computing stopped")
            except Exception as e:
                logger.error(f"Error stopping fog computing: {e}")

        # Stop chat engine
        if hasattr(self.chat_engine, "stop"):
            try:
                await self.chat_engine.stop()
                self.integrations["chat"].is_connected = False
                logger.info("Chat engine stopped")
            except Exception as e:
                logger.error(f"Error stopping chat engine: {e}")

    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""

        return {
            integration_name: {
                "service_name": status.service_name,
                "is_connected": status.is_connected,
                "last_sync": status.last_sync,
                "error_count": status.error_count,
                "last_error": status.last_error,
            }
            for integration_name, status in self.integrations.items()
        }

    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics."""

        return {
            "p2p_sync_operations": self.integration_metrics["p2p_sync_operations"],
            "fog_tasks_executed": self.integration_metrics["fog_tasks_executed"],
            "chat_interactions": self.integration_metrics["chat_interactions"],
            "config_updates": self.integration_metrics["config_updates"],
            "active_integrations": sum(1 for status in self.integrations.values() if status.is_connected),
            "total_integrations": len(self.integrations),
        }
