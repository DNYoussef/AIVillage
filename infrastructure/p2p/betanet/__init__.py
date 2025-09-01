"""BetaNet Anonymous Routing Package

Privacy-focused anonymous routing with mixnode network and covert transport.

Archaeological Enhancement: Preserved anonymous communication innovations
from deprecated branches with enhanced privacy and security features.

Innovation Score: 9.2/10 - Advanced anonymous routing

Features:
- Mixnode network for traffic anonymization
- Noise protocol for forward secrecy
- HTX (HTTP Transport eXtension) for covert channels
- Access tickets for secure authentication
- Circuit-based routing with traffic analysis resistance

Version: 2.0.0
"""

import logging
from typing import TYPE_CHECKING

# Package information
__version__ = "2.0.0"
__author__ = "AI Village Team"

# Configure logging
logger = logging.getLogger(__name__)

# Import components with graceful fallback
try:
    from .mixnode_client import MixnodeClient

    logger.info("BetaNet mixnode client loaded")
except ImportError as e:
    logger.warning(f"BetaNet mixnode client not available: {e}")
    if not TYPE_CHECKING:
        MixnodeClient = None

try:
    from .noise_protocol import NoiseProtocol, NoiseState

    logger.debug("BetaNet noise protocol loaded")
except ImportError as e:
    logger.debug(f"Noise protocol not available: {e}")
    if not TYPE_CHECKING:
        NoiseProtocol = None
        NoiseState = None

try:
    from .htx_transport import CovertChannel, HTXTransport

    logger.debug("BetaNet HTX transport loaded")
except ImportError as e:
    logger.debug(f"HTX transport not available: {e}")
    if not TYPE_CHECKING:
        HTXTransport = None
        CovertChannel = None

try:
    from .access_tickets import AccessTicket, AccessTicketManager

    logger.debug("BetaNet access tickets loaded")
except ImportError as e:
    logger.debug(f"Access tickets not available: {e}")
    if not TYPE_CHECKING:
        AccessTicketManager = None
        AccessTicket = None

__all__ = [
    # Core anonymous routing
    "MixnodeClient",
    # Cryptographic protocols
    "NoiseProtocol",
    "NoiseState",
    # Covert transport
    "HTXTransport",
    "CovertChannel",
    # Authentication
    "AccessTicketManager",
    "AccessTicket",
]
