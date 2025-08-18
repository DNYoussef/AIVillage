"""
Access Ticket System for BetaNet HTX

Implements the access ticket authentication system for BetaNet,
providing controlled access to mixnodes and transport services.
"""

import json
import logging
import secrets
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AccessTicket:
    """BetaNet access ticket for authentication."""

    ticket_id: str
    device_id: str
    service_type: str = "htx"
    created_at: float = 0
    expires_at: float = 0
    signature: bytes | None = None

    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()
        if self.expires_at == 0:
            self.expires_at = self.created_at + 3600  # 1 hour default

    @property
    def is_valid(self) -> bool:
        """Check if ticket is still valid."""
        return time.time() < self.expires_at

    @property
    def is_expired(self) -> bool:
        """Check if ticket has expired."""
        return not self.is_valid

    def to_dict(self) -> dict:
        """Convert ticket to dictionary."""
        return {
            "ticket_id": self.ticket_id,
            "device_id": self.device_id,
            "service_type": self.service_type,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "signature": self.signature.hex() if self.signature else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AccessTicket":
        """Create ticket from dictionary."""
        signature = None
        if data.get("signature"):
            signature = bytes.fromhex(data["signature"])

        return cls(
            ticket_id=data["ticket_id"],
            device_id=data["device_id"],
            service_type=data.get("service_type", "htx"),
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            signature=signature,
        )


class TicketManager:
    """Manager for BetaNet access tickets."""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.tickets: dict[str, AccessTicket] = {}
        self.signing_key = secrets.token_bytes(32)  # Placeholder

    def create_ticket(self, service_type: str = "htx", valid_duration: int = 3600) -> AccessTicket:
        """Create a new access ticket."""
        ticket_id = secrets.token_hex(16)

        ticket = AccessTicket(
            ticket_id=ticket_id,
            device_id=self.device_id,
            service_type=service_type,
            expires_at=time.time() + valid_duration,
        )

        # Sign ticket (placeholder)
        ticket.signature = self._sign_ticket(ticket)

        self.tickets[ticket_id] = ticket
        logger.debug(f"Created access ticket: {ticket_id}")

        return ticket

    def _sign_ticket(self, ticket: AccessTicket) -> bytes:
        """Sign access ticket (placeholder implementation)."""
        # In production, this would use proper cryptographic signing
        json.dumps(
            {
                "ticket_id": ticket.ticket_id,
                "device_id": ticket.device_id,
                "service_type": ticket.service_type,
                "expires_at": ticket.expires_at,
            }
        ).encode()

        # Placeholder signature
        return secrets.token_bytes(64)

    def verify_ticket(self, ticket: AccessTicket) -> bool:
        """Verify access ticket signature and validity."""
        if ticket.is_expired:
            logger.warning(f"Ticket {ticket.ticket_id} has expired")
            return False

        # Verify signature (placeholder)
        self._sign_ticket(ticket)

        # In placeholder mode, just check if signature exists
        if not ticket.signature:
            logger.warning(f"Ticket {ticket.ticket_id} has no signature")
            return False

        logger.debug(f"Verified ticket: {ticket.ticket_id}")
        return True

    def get_ticket(self, ticket_id: str) -> AccessTicket | None:
        """Get ticket by ID."""
        return self.tickets.get(ticket_id)

    def cleanup_expired_tickets(self):
        """Remove expired tickets."""
        expired_tickets = []

        for ticket_id, ticket in self.tickets.items():
            if ticket.is_expired:
                expired_tickets.append(ticket_id)

        for ticket_id in expired_tickets:
            del self.tickets[ticket_id]
            logger.debug(f"Removed expired ticket: {ticket_id}")

    def get_valid_tickets(self) -> list[AccessTicket]:
        """Get all valid (non-expired) tickets."""
        return [ticket for ticket in self.tickets.values() if ticket.is_valid]


def create_ticket_manager(device_id: str) -> TicketManager:
    """Factory function to create ticket manager."""
    return TicketManager(device_id)
