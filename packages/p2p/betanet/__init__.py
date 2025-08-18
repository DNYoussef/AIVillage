"""
BetaNet encrypted internet transport implementation.

Based on the production-ready betanet bounty implementation, providing:
- HTX v1.1 compliant transport protocol
- Noise XK encryption with forward secrecy
- Access ticket authentication system
- uTLS fingerprinting for traffic analysis resistance
- Mobile-optimized performance and battery management

This implementation serves as the Python bridge to the Rust betanet-htx crates
while providing standalone Python functionality where Rust bindings are unavailable.
"""

from .access_tickets import AccessTicket, TicketManager
from .htx_transport import HtxClient, HtxFrame, HtxServer
from .mixnode_client import MixnodeClient
from .noise_protocol import NoiseXKHandshake

__all__ = [
    "HtxClient",
    "HtxServer",
    "HtxFrame",
    "NoiseXKHandshake",
    "AccessTicket",
    "TicketManager",
    "MixnodeClient",
]
