"""
BetaNet - Encrypted Internet Protocol

Handles secure internet-based communications:
- HTX transport with fingerprint camouflage
- Cover traffic and privacy protection
- Mobile-optimized data usage
- Covert channel establishment
"""

from .covert_transport import BetaNetCovertTransport
from .htx_transport import BetaNetHTXTransport

__all__ = ["BetaNetHTXTransport", "BetaNetCovertTransport"]
