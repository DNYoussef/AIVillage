"""Mock transport implementations for testing."""

import random
from typing import Any


class MockTransport:
    """Mock transport for testing without real network dependencies."""

    def __init__(self, transport_type: str = "mock"):
        self.transport_type = transport_type
        self.is_connected = True
        self.message_queue = []
        self.sent_count = 0
        self.received_count = 0

    def send(self, data: bytes, destination: str) -> bool:
        """Simulate sending data."""
        self.sent_count += 1
        # Simulate success based on transport type
        if self.transport_type == "bitchat":
            return random.random() > 0.15  # 85% success
        elif self.transport_type == "betanet":
            return random.random() > 0.1  # 90% success
        else:
            return random.random() > 0.2  # 80% success

    def receive(self) -> bytes | None:
        """Simulate receiving data."""
        if self.message_queue:
            self.received_count += 1
            return self.message_queue.pop(0)
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get transport statistics."""
        return {
            "type": self.transport_type,
            "sent": self.sent_count,
            "received": self.received_count,
            "queued": len(self.message_queue),
            "connected": self.is_connected,
        }


class MockNode:
    """Mock P2P node for testing."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.transports = {
            "bitchat": MockTransport("bitchat"),
            "betanet": MockTransport("betanet"),
        }
        self.routing_table = {}
        self.message_cache = {}

    def select_transport(
        self, message_size: int, urgency: str, battery_level: int, network_type: str
    ) -> str:
        """Select transport based on conditions."""
        # Policy simulation
        if battery_level < 20:
            return "bitchat"
        elif message_size > 2000 and urgency == "high" and network_type == "internet":
            return "betanet"
        elif network_type == "local" or message_size < 500:
            return "bitchat"
        else:
            return "betanet"

    def send_message(
        self,
        destination: str,
        data: bytes,
        urgency: str = "normal",
        battery_level: int = 50,
    ) -> bool:
        """Send message using appropriate transport."""
        transport_name = self.select_transport(
            len(data),
            urgency,
            battery_level,
            "local" if destination.startswith("local") else "internet",
        )

        transport = self.transports[transport_name]
        return transport.send(data, destination)
