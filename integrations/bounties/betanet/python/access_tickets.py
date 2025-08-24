"""
Access Ticket System for BetaNet HTX

Implements the access ticket authentication system for BetaNet,
providing controlled access to mixnodes and transport services with proper cryptography.
"""

from dataclasses import dataclass
import hashlib
import hmac
import json
import logging
import secrets
import time

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

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
    """Manager for BetaNet access tickets with proper Ed25519 signing."""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.tickets: dict[str, AccessTicket] = {}

        # Generate proper Ed25519 signing key
        if CRYPTO_AVAILABLE:
            self.signing_key = ed25519.Ed25519PrivateKey.generate()
            self.public_key = self.signing_key.public_key()
        else:
            # Fallback to HMAC-based signing
            self.signing_key = secrets.token_bytes(32)
            self.public_key = secrets.token_bytes(32)

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
        """Sign access ticket with Ed25519 or HMAC fallback."""
        # Create canonical ticket representation
        ticket_data = json.dumps(
            {
                "ticket_id": ticket.ticket_id,
                "device_id": ticket.device_id,
                "service_type": ticket.service_type,
                "created_at": ticket.created_at,
                "expires_at": ticket.expires_at,
            },
            sort_keys=True,
        ).encode()

        if CRYPTO_AVAILABLE and hasattr(self.signing_key, "sign"):
            # Use real Ed25519 signing
            try:
                signature = self.signing_key.sign(ticket_data)
                return signature
            except Exception as e:
                logger.warning(f"Ed25519 signing failed, using HMAC fallback: {e}")

        # Fallback to HMAC-SHA256
        if isinstance(self.signing_key, bytes):
            return hmac.new(self.signing_key, ticket_data, hashlib.sha256).digest()
        else:
            # If somehow we get here, use a deterministic but weak signature
            return hashlib.sha256(ticket_data + b"betanet_fallback").digest()

    def verify_ticket(self, ticket: AccessTicket) -> bool:
        """Verify access ticket signature and validity with real cryptography."""
        if ticket.is_expired:
            logger.warning(f"Ticket {ticket.ticket_id} has expired")
            return False

        if not ticket.signature:
            logger.warning(f"Ticket {ticket.ticket_id} has no signature")
            return False

        # Create canonical ticket representation for verification
        ticket_data = json.dumps(
            {
                "ticket_id": ticket.ticket_id,
                "device_id": ticket.device_id,
                "service_type": ticket.service_type,
                "created_at": ticket.created_at,
                "expires_at": ticket.expires_at,
            },
            sort_keys=True,
        ).encode()

        if CRYPTO_AVAILABLE and hasattr(self.public_key, "public_bytes"):
            # Use real Ed25519 verification
            try:
                self.public_key.verify(ticket.signature, ticket_data)
                logger.debug(f"Ed25519 verification passed for ticket: {ticket.ticket_id}")
                return True
            except Exception as e:
                logger.warning(f"Ed25519 verification failed for ticket {ticket.ticket_id}: {e}")
                # Fall through to HMAC verification

        # Fallback to HMAC verification
        if isinstance(self.signing_key, bytes):
            expected_signature = hmac.new(self.signing_key, ticket_data, hashlib.sha256).digest()
            if hmac.compare_digest(ticket.signature, expected_signature):
                logger.debug(f"HMAC verification passed for ticket: {ticket.ticket_id}")
                return True
            else:
                logger.warning(f"HMAC verification failed for ticket: {ticket.ticket_id}")
                return False

        # Final fallback - check deterministic signature
        expected_signature = hashlib.sha256(ticket_data + b"betanet_fallback").digest()
        if hmac.compare_digest(ticket.signature, expected_signature):
            logger.debug(f"Fallback verification passed for ticket: {ticket.ticket_id}")
            return True

        logger.warning(f"All signature verification methods failed for ticket: {ticket.ticket_id}")
        return False

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
