"""
P2P compatibility bridges and adapters.

Provides compatibility layers for existing P2P implementations and
bridges to external systems like mobile platforms and Rust FFI.
"""

from .compatibility import LegacyTransportBridge, create_legacy_bridge
from .rust_ffi import RustFFIBridge

__all__ = [
    "create_legacy_bridge",
    "LegacyTransportBridge",
    "RustFFIBridge",
]
