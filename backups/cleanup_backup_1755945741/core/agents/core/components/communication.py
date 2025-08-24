"""Agent Communication Component.

Handles all inter-agent communication through channels, P2P messaging,
and broadcast systems. Follows single responsibility principle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CommunicationConfig:
    """Configuration for agent communication channels."""

    p2p_enabled: bool = True
    broadcast_enabled: bool = True
    group_channels_enabled: bool = True
    emergency_channel_enabled: bool = True
    max_message_size_kb: int = 1024
    message_timeout_seconds: int = 30
    retry_attempts: int = 3


@dataclass
class MessageMetrics:
    """Message delivery and performance metrics."""

    sent_count: int = 0
    received_count: int = 0
    failed_count: int = 0
    average_latency_ms: float = 0.0
    last_activity: datetime | None = None


class AgentCommunication:
    """Handles agent communication with P2P, broadcast, and group channels.

    This component encapsulates all communication logic, reducing coupling
    between communication concerns and other agent responsibilities.
    """

    def __init__(self, agent_id: str, config: CommunicationConfig | None = None):
        """Initialize communication component.

        Args:
            agent_id: Unique identifier for the agent
            config: Communication configuration, uses defaults if None
        """
        self.agent_id = agent_id
        self.config = config or CommunicationConfig()
        self.metrics = MessageMetrics()

        # Communication channels organized by type (CoN - Connascence of Name)
        self._channels = {"direct": [], "broadcast": [], "group": {}, "emergency": []}

        # External clients injected during initialization (DI pattern)
        self._p2p_client = None
        self._mcp_tools = {}

        logger.debug(f"Communication component initialized for agent {agent_id}")

    def inject_dependencies(self, p2p_client: Any, mcp_tools: dict[str, Any]) -> None:
        """Inject external dependencies (Dependency Inversion Principle)."""
        self._p2p_client = p2p_client
        self._mcp_tools = mcp_tools
        logger.debug("Communication dependencies injected")

    async def send_direct_message(
        self, recipient: str, message: str, priority: int = 5, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send direct message to specific agent.

        Args:
            recipient: Target agent ID
            message: Message content
            priority: Message priority (1-10, 10=highest)
            metadata: Additional message metadata

        Returns:
            Dict with delivery status and message details
        """
        if not self._p2p_client:
            return self._create_error_response("P2P client not available")

        try:
            result = await self._execute_communication_tool(
                "communicate",
                {
                    "recipient": recipient,
                    "message": message,
                    "channel_type": "direct",
                    "priority": priority,
                    "sender_id": self.agent_id,
                    "metadata": metadata or {},
                },
            )

            self._update_metrics(sent=True, latency_ms=0)  # Would measure actual latency
            logger.debug(f"Direct message sent from {self.agent_id} to {recipient}")

            return result

        except Exception as e:
            self._update_metrics(failed=True)
            logger.error(f"Failed to send direct message: {e}")
            return self._create_error_response(f"Send failed: {str(e)}")

    async def broadcast_message(
        self, message: str, priority: int = 5, exclude_agents: list[str] | None = None
    ) -> dict[str, Any]:
        """Broadcast message to all agents in network.

        Args:
            message: Message content to broadcast
            priority: Message priority
            exclude_agents: Agent IDs to exclude from broadcast

        Returns:
            Dict with broadcast status and delivery statistics
        """
        if not self.config.broadcast_enabled:
            return self._create_error_response("Broadcast disabled in configuration")

        try:
            result = await self._execute_communication_tool(
                "communicate",
                {
                    "recipient": "*",
                    "message": message,
                    "channel_type": "broadcast",
                    "priority": priority,
                    "sender_id": self.agent_id,
                    "metadata": {"exclude_agents": exclude_agents or []},
                },
            )

            self._update_metrics(sent=True)
            logger.info(f"Broadcast message sent from {self.agent_id}")

            return result

        except Exception as e:
            self._update_metrics(failed=True)
            logger.error(f"Failed to broadcast message: {e}")
            return self._create_error_response(f"Broadcast failed: {str(e)}")

    async def join_group_channel(self, channel_name: str) -> bool:
        """Join a topic-based group communication channel.

        Args:
            channel_name: Name of the group channel to join

        Returns:
            True if successfully joined, False otherwise
        """
        if not self.config.group_channels_enabled:
            logger.warning(f"Group channels disabled, cannot join {channel_name}")
            return False

        if channel_name not in self._channels["group"]:
            self._channels["group"][channel_name] = []

        logger.info(f"Agent {self.agent_id} joined group channel: {channel_name}")
        return True

    async def leave_group_channel(self, channel_name: str) -> bool:
        """Leave a group communication channel.

        Args:
            channel_name: Name of the group channel to leave

        Returns:
            True if successfully left, False otherwise
        """
        if channel_name in self._channels["group"]:
            del self._channels["group"][channel_name]
            logger.info(f"Agent {self.agent_id} left group channel: {channel_name}")
            return True

        return False

    async def send_group_message(self, channel_name: str, message: str, priority: int = 5) -> dict[str, Any]:
        """Send message to specific group channel.

        Args:
            channel_name: Target group channel
            message: Message content
            priority: Message priority

        Returns:
            Dict with delivery status
        """
        if channel_name not in self._channels["group"]:
            return self._create_error_response(f"Not member of group channel: {channel_name}")

        try:
            result = await self._execute_communication_tool(
                "communicate",
                {
                    "recipient": f"group:{channel_name}",
                    "message": message,
                    "channel_type": "group",
                    "priority": priority,
                    "sender_id": self.agent_id,
                    "metadata": {"channel": channel_name},
                },
            )

            self._update_metrics(sent=True)
            logger.debug(f"Group message sent to {channel_name} from {self.agent_id}")

            return result

        except Exception as e:
            self._update_metrics(failed=True)
            logger.error(f"Failed to send group message: {e}")
            return self._create_error_response(f"Group send failed: {str(e)}")

    async def send_emergency_message(self, message: str, recipients: list[str] | None = None) -> dict[str, Any]:
        """Send high-priority emergency message.

        Args:
            message: Emergency message content
            recipients: Specific recipients, or None for broadcast

        Returns:
            Dict with delivery status
        """
        if not self.config.emergency_channel_enabled:
            return self._create_error_response("Emergency channel disabled")

        # Emergency messages always use highest priority
        if recipients is None:
            return await self.broadcast_message(message, priority=10)
        else:
            # Send to each recipient individually for reliability
            results = {}
            for recipient in recipients:
                result = await self.send_direct_message(recipient, message, priority=10)
                results[recipient] = result.get("status") == "success"

            return {"status": "success", "emergency_delivery": results, "timestamp": datetime.now().isoformat()}

    def get_communication_metrics(self) -> dict[str, Any]:
        """Get communication performance metrics.

        Returns:
            Dict containing message statistics and performance data
        """
        return {
            "messages_sent": self.metrics.sent_count,
            "messages_received": self.metrics.received_count,
            "messages_failed": self.metrics.failed_count,
            "average_latency_ms": self.metrics.average_latency_ms,
            "success_rate": self._calculate_success_rate(),
            "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None,
            "active_channels": {
                "direct": len(self._channels["direct"]),
                "broadcast": len(self._channels["broadcast"]),
                "group": list(self._channels["group"].keys()),
                "emergency": len(self._channels["emergency"]),
            },
        }

    def get_channel_status(self) -> dict[str, Any]:
        """Get status of all communication channels.

        Returns:
            Dict with channel status and configuration
        """
        return {
            "config": {
                "p2p_enabled": self.config.p2p_enabled,
                "broadcast_enabled": self.config.broadcast_enabled,
                "group_channels_enabled": self.config.group_channels_enabled,
                "emergency_channel_enabled": self.config.emergency_channel_enabled,
            },
            "connections": {
                "p2p_client_connected": self._p2p_client is not None,
                "mcp_tools_available": len(self._mcp_tools) > 0,
            },
            "channels": {
                "group_memberships": list(self._channels["group"].keys()),
                "total_channels": sum(len(ch) if isinstance(ch, list) else len(ch) for ch in self._channels.values()),
            },
        }

    async def _execute_communication_tool(self, tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute MCP communication tool (internal helper)."""
        if tool_name not in self._mcp_tools:
            raise ValueError(f"Communication tool {tool_name} not available")

        tool = self._mcp_tools[tool_name]
        return await tool.execute(parameters)

    def _update_metrics(
        self, sent: bool = False, received: bool = False, failed: bool = False, latency_ms: float = 0
    ) -> None:
        """Update internal communication metrics (CoI - Connascence of Identity for metrics object)."""
        if sent:
            self.metrics.sent_count += 1
        if received:
            self.metrics.received_count += 1
        if failed:
            self.metrics.failed_count += 1

        # Update running average latency
        if latency_ms > 0 and self.metrics.sent_count > 0:
            current_avg = self.metrics.average_latency_ms
            new_avg = ((current_avg * (self.metrics.sent_count - 1)) + latency_ms) / self.metrics.sent_count
            self.metrics.average_latency_ms = new_avg

        self.metrics.last_activity = datetime.now()

    def _calculate_success_rate(self) -> float:
        """Calculate message delivery success rate."""
        total_attempts = self.metrics.sent_count + self.metrics.failed_count
        if total_attempts == 0:
            return 1.0
        return self.metrics.sent_count / total_attempts

    def _create_error_response(self, error_message: str) -> dict[str, Any]:
        """Create standardized error response (CoA - single algorithm for error handling)."""
        return {
            "status": "error",
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
        }
