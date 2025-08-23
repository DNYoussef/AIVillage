#!/usr/bin/env python3
"""
Digital Twin Orchestrator - Facade coordination

Orchestrates the interaction between Core, Storage, and Integration components
following the Facade pattern to maintain API compatibility.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..unified_digital_twin_system import TwinAccessLevel, TwinDataType
from .core import DigitalTwinCore
from .integration import DigitalTwinIntegration
from .storage import DigitalTwinStorage

logger = logging.getLogger(__name__)


class DigitalTwinOrchestrator:
    """
    Digital Twin System Orchestrator

    Provides the unified interface for the digital twin system while
    delegating responsibilities to specialized components following
    the Facade pattern and maintaining API compatibility.
    """

    def __init__(
        self, twin_id: str, data_dir: Optional[str] = None, enable_encryption: bool = True, enable_p2p: bool = True
    ):
        self.twin_id = twin_id
        self.data_dir = Path(data_dir) if data_dir else Path("data/twins")
        self.enable_encryption = enable_encryption
        self.enable_p2p = enable_p2p

        # Initialize components with dependency injection
        self.core = DigitalTwinCore(twin_id)
        self.storage = DigitalTwinStorage(self.data_dir, enable_encryption)
        self.integration = DigitalTwinIntegration(twin_id, enable_p2p)

        # System state
        self._running = False
        self._background_tasks: List[asyncio.Task] = []

    async def start(self) -> bool:
        """Start the unified digital twin system."""

        if self._running:
            return True

        logger.info("Starting unified digital twin system...")

        try:
            # Initialize components in proper order
            await self.storage.initialize()
            await self.integration.initialize_integrations(self.data_dir)

            # Load cached data from storage into core
            await self._load_cached_data()

            # Start background maintenance tasks
            self._background_tasks = [
                asyncio.create_task(self._maintenance_loop()),
                asyncio.create_task(self._metrics_loop()),
                asyncio.create_task(self._sync_loop()),
            ]

            self._running = True
            logger.info("✅ Unified digital twin system started successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start digital twin system: {e}")
            await self.stop()  # Cleanup on failure
            raise

    async def stop(self):
        """Stop the unified digital twin system gracefully."""

        self._running = False

        logger.info("Stopping unified digital twin system...")

        # Cancel background tasks
        for task in self._background_tasks:
            if not task.cancelled():
                task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Save cached data before shutdown
        await self._save_cached_data()

        # Stop components in reverse order
        await self.integration.stop_integrations()
        await self.storage.close()

        logger.info("✅ Unified digital twin system stopped")

    # User Management API (delegated to Core)

    async def create_user(self, username: str, email: str, password: str, **user_data) -> str:
        """Create new digital twin user."""

        user_id = await self.core.create_user(username, email, password, **user_data)

        # Persist to storage
        user = await self.core._get_user_by_id(user_id)
        if user:
            await self.storage.save_user(user)

            # Sync with P2P network if enabled
            await self.integration.sync_with_p2p_network(
                {"type": "user_created", "user_id": user_id, "username": username}
            )

        return user_id

    async def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return access token."""

        token = await self.core.authenticate_user(username, password)

        if token:
            # Update user in storage
            user = await self.core._get_user_by_username(username)
            if user:
                await self.storage.save_user(user)

        return token

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile data."""

        # Try core cache first
        profile = await self.core.get_user_profile(user_id)

        if not profile:
            # Load from storage if not in cache
            user = await self.storage.load_user(user_id)
            if user:
                self.core.users[user_id] = user
                profile = await self.core.get_user_profile(user_id)

        return profile

    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile data."""

        success = await self.core.update_user_profile(user_id, updates)

        if success:
            # Persist changes to storage
            user = await self.core._get_user_by_id(user_id)
            if user:
                await self.storage.save_user(user)

                # Sync with P2P network
                await self.integration.sync_with_p2p_network(
                    {"type": "user_updated", "user_id": user_id, "updates": updates}
                )

        return success

    # Conversation Management API (coordinated between Core and Storage)

    async def create_conversation(self, user_id: str, title: str = "", **conv_data) -> str:
        """Create new conversation session."""

        conversation_id = await self.core.create_conversation(user_id, title, **conv_data)

        # Persist to storage
        conversation = self.core.active_conversations.get(conversation_id)
        if conversation:
            await self.storage.save_conversation(conversation)

        return conversation_id

    async def add_message(
        self, conversation_id: str, user_id: str, content: str, message_type: str = "user", **msg_data
    ) -> str:
        """Add message to conversation."""

        message_id = await self.core.add_message(conversation_id, user_id, content, message_type, **msg_data)

        # Create message object for storage
        from ..unified_digital_twin_system import TwinMessage

        message = TwinMessage(
            message_id=message_id,
            conversation_id=conversation_id,
            user_id=user_id,
            content=content,
            message_type=message_type,
            **msg_data,
        )

        # Persist message and update conversation
        await self.storage.save_message(message)

        conversation = self.core.active_conversations.get(conversation_id)
        if conversation:
            await self.storage.save_conversation(conversation)

        return message_id

    async def get_conversation_history(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation message history."""

        return await self.storage.get_conversation_messages(conversation_id, limit)

    async def generate_ai_response(self, conversation_id: str, user_message: str, user_id: str) -> Dict[str, Any]:
        """Generate AI response for conversation."""

        # Add user message first
        await self.add_message(
            conversation_id=conversation_id, user_id=user_id, content=user_message, message_type="user"
        )

        # Get conversation history for context
        history = await self.get_conversation_history(conversation_id, limit=20)

        # Generate response using integration layer
        ai_response = await self.integration.generate_ai_response(
            conversation_id=conversation_id, user_message=user_message, user_id=user_id, context={"history": history}
        )

        # Add AI response message
        response_id = await self.add_message(
            conversation_id=conversation_id,
            user_id="assistant",
            content=ai_response["content"],
            message_type="assistant",
            tokens_used=ai_response.get("tokens_used", 0),
            processing_time_ms=ai_response.get("processing_time_ms", 0),
        )

        return {
            "response_id": response_id,
            "content": ai_response["content"],
            "tokens_used": ai_response.get("tokens_used", 0),
            "processing_time_ms": ai_response.get("processing_time_ms", 0),
            "model_used": ai_response.get("model_used", "unknown"),
        }

    # Data Management API (delegated to Storage)

    async def store_twin_data(
        self,
        user_id: str,
        data_type: str,
        data_key: str,
        data_value: Any,
        access_level: TwinAccessLevel = TwinAccessLevel.PRIVATE,
        expires_at: Optional[float] = None,
    ) -> Optional[str]:
        """Store arbitrary data in digital twin."""

        data_id = await self.storage.store_twin_data(user_id, data_type, data_key, data_value, access_level, expires_at)

        if data_id:
            # Sync with P2P network for shared data
            if access_level in [TwinAccessLevel.PUBLIC, TwinAccessLevel.RESTRICTED]:
                await self.integration.sync_with_p2p_network(
                    {
                        "type": "data_stored",
                        "user_id": user_id,
                        "data_type": data_type,
                        "data_key": data_key,
                        "access_level": access_level.value,
                    }
                )

        return data_id

    async def get_twin_data(self, user_id: str, data_type: str, data_key: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve data from digital twin."""

        return await self.storage.get_twin_data(user_id, data_type, data_key)

    # System Management and Monitoring

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""

        core_metrics = self.core.get_core_metrics()
        storage_metrics = self.storage.get_storage_metrics()
        integration_metrics = self.integration.get_integration_metrics()

        return {
            "twin_id": self.twin_id,
            "running": self._running,
            "core": core_metrics,
            "storage": storage_metrics,
            "integration": integration_metrics,
            "features": {
                "encryption_enabled": self.enable_encryption,
                "p2p_enabled": self.enable_p2p,
                "external_integrations": len(self.integration.integrations),
            },
            "background_tasks": len([t for t in self._background_tasks if not t.done()]),
        }

    def get_user_statistics(self) -> Dict[str, Any]:
        """Get user activity statistics."""

        core_metrics = self.core.get_core_metrics()

        return {
            "total_users": core_metrics["performance"]["total_users"],
            "active_users": len([u for u in self.core.users.values() if u.is_active]),
            "total_conversations": core_metrics["performance"]["total_conversations"],
            "total_messages": core_metrics["performance"]["total_messages"],
            "cached_users": core_metrics["cached_users"],
            "active_sessions": core_metrics["active_sessions"],
        }

    # Background Tasks

    async def _maintenance_loop(self):
        """Background maintenance tasks."""

        while self._running:
            try:
                # Clean up expired data
                await self.storage.cleanup_expired_data()

                # Clean up expired sessions
                await self.core.cleanup_expired_sessions()

                # Clean up old conversations based on retention policy
                await self._cleanup_old_conversations()

                # Update user activity status
                await self._update_user_activity_status()

            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")

            await asyncio.sleep(300)  # Run every 5 minutes

    async def _metrics_loop(self):
        """Background metrics collection."""

        while self._running:
            try:
                # Update cached metrics from storage
                with self.storage._get_db_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute("SELECT COUNT(*) FROM users")
                    self.core.metrics["total_users"] = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM conversations")
                    self.core.metrics["total_conversations"] = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM messages")
                    self.core.metrics["total_messages"] = cursor.fetchone()[0]

                # Log metrics periodically
                if self.core.metrics["total_users"] > 0:
                    logger.debug(f"Twin metrics: {self.core.metrics}")

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")

            await asyncio.sleep(60)  # Update every minute

    async def _sync_loop(self):
        """Background P2P synchronization."""

        while self._running:
            try:
                if self.integration.integrations.get("p2p", {}).get("is_connected", False):
                    # Perform periodic sync operations
                    sync_data = {
                        "type": "heartbeat",
                        "metrics": self.get_system_metrics(),
                        "timestamp": time.time(),
                    }

                    await self.integration.sync_with_p2p_network(sync_data)

            except Exception as e:
                logger.error(f"Sync loop error: {e}")

            await asyncio.sleep(600)  # Sync every 10 minutes

    async def _load_cached_data(self):
        """Load frequently accessed data into cache."""

        try:
            # Load active users into core cache
            with self.storage._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT user_id FROM users WHERE is_active = 1 LIMIT 100")

                for row in cursor.fetchall():
                    user = await self.storage.load_user(row["user_id"])
                    if user:
                        self.core.users[user.user_id] = user

            logger.info(f"Loaded {len(self.core.users)} users into cache")

        except Exception as e:
            logger.error(f"Error loading cached data: {e}")

    async def _save_cached_data(self):
        """Save cached data before shutdown."""

        try:
            # Save all cached users
            for user in self.core.users.values():
                await self.storage.save_user(user)

            # Save all cached conversations
            for conversation in self.core.active_conversations.values():
                await self.storage.save_conversation(conversation)

            logger.info("Cached data saved to storage")

        except Exception as e:
            logger.error(f"Error saving cached data: {e}")

    async def _cleanup_old_conversations(self):
        """Clean up old conversations based on retention policy."""

        try:
            with self.storage._get_db_connection() as conn:
                cursor = conn.cursor()

                # Get conversations past their retention period
                cursor.execute(
                    """
                    SELECT conversation_id, retention_days, last_message_at
                    FROM conversations
                    WHERE retention_days > 0
                """
                )

                expired_conversations = []
                current_time = time.time()

                for row in cursor.fetchall():
                    retention_seconds = row["retention_days"] * 24 * 3600
                    if current_time - row["last_message_at"] > retention_seconds:
                        expired_conversations.append(row["conversation_id"])

                # Delete expired conversations and their messages
                for conv_id in expired_conversations:
                    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
                    cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conv_id,))

                    # Remove from cache
                    self.core.active_conversations.pop(conv_id, None)

                if expired_conversations:
                    conn.commit()
                    logger.info(f"Cleaned up {len(expired_conversations)} expired conversations")

        except Exception as e:
            logger.error(f"Error cleaning up conversations: {e}")

    async def _update_user_activity_status(self):
        """Update user activity status based on last active time."""

        try:
            # Mark users as inactive if not seen for 30 days
            inactive_threshold = time.time() - (30 * 24 * 3600)

            for user in self.core.users.values():
                if user.is_active and user.last_active < inactive_threshold:
                    user.is_active = False
                    await self.storage.save_user(user)

        except Exception as e:
            logger.error(f"Error updating user activity: {e}")


# Factory function for backward compatibility
def create_digital_twin_system(twin_id: str, **kwargs) -> DigitalTwinOrchestrator:
    """
    Create unified digital twin system.

    Args:
        twin_id: Unique digital twin identifier
        data_dir: Directory for data storage (default: data/twins)
        enable_encryption: Enable data encryption (default: True)
        enable_p2p: Enable P2P integration (default: True)

    Returns:
        DigitalTwinOrchestrator instance
    """
    return DigitalTwinOrchestrator(twin_id, **kwargs)
