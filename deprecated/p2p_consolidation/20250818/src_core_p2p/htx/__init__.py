"""HTX Transport Layer - Modular Betanet v1.1 Implementation

This package provides a modular implementation of the HTX (HTTP/2 over TLS eXtended)
cover transport protocol for Betanet v1.1 compliance.

Components:
- frame_format: HTX frame encoding/decoding
- utls_fingerprinting: JA3/JA4 fingerprint calibration
- noise_protocol: Noise XK inner security protocol
- access_tickets: Authentication and rate limiting
- transport: Main HTX transport coordinator

Architecture:
Each module has a single responsibility and clean interfaces, enabling
professional-grade modularity for bounty submission requirements.
"""

from .access_tickets import (
    AccessTicket,
    AccessTicketManager,
    TicketStatus,
    TicketType,
    generate_issuer_keypair,
)
from .frame_format import (
    HTXFrame,
    HTXFrameBuffer,
    HTXFrameCodec,
    HTXFrameType,
    create_data_frame,
    create_ping_frame,
    create_window_update_frame,
)
from .noise_protocol import NoiseHandshakeState, NoiseKeys, NoiseXKProtocol
from .utls_fingerprinting import (
    ClientHelloFingerprint,
    FingerprintTemplate,
    uTLSFingerprintCalibrator,
)

__all__ = [
    # Access Tickets
    "AccessTicket",
    "AccessTicketManager",
    "TicketStatus",
    "TicketType",
    "generate_issuer_keypair",
    # Frame Format
    "HTXFrame",
    "HTXFrameCodec",
    "HTXFrameType",
    "HTXFrameBuffer",
    "create_data_frame",
    "create_ping_frame",
    "create_window_update_frame",
    # Noise Protocol
    "NoiseXKProtocol",
    "NoiseHandshakeState",
    "NoiseKeys",
    # uTLS Fingerprinting
    "ClientHelloFingerprint",
    "FingerprintTemplate",
    "uTLSFingerprintCalibrator",
]

__version__ = "1.1.0"
