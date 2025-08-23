"""
Rust FFI Bridge for BetaNet

Provides bridge to the Rust betanet FFI library for high-performance
transport operations when available.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RustFFIBridge:
    """Bridge to Rust betanet FFI library."""

    def __init__(self):
        self.available = False
        self.ffi_lib = None

    def initialize(self) -> bool:
        """Initialize Rust FFI bridge with real library loading."""
        try:
            # Try to load betanet FFI library
            import ctypes
            import os
            import platform

            logger.info("Attempting to load betanet FFI library...")

            # Determine library name based on platform
            system = platform.system().lower()
            if system == "windows":
                lib_name = "betanet_ffi.dll"
            elif system == "darwin":
                lib_name = "libbetanet_ffi.dylib"
            else:
                lib_name = "libbetanet_ffi.so"

            # Look for library in common locations
            possible_paths = [
                f"./target/debug/{lib_name}",
                f"./target/release/{lib_name}",
                f"./{lib_name}",
                os.path.join(os.getcwd(), "target", "debug", lib_name),
                os.path.join(os.getcwd(), "target", "release", lib_name),
            ]

            for lib_path in possible_paths:
                if os.path.exists(lib_path):
                    try:
                        self.ffi_lib = ctypes.CDLL(lib_path)
                        self.available = True
                        logger.info(f"Loaded betanet FFI library from: {lib_path}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load {lib_path}: {e}")
                        continue

            if not self.available:
                logger.info("Rust FFI library not found, using Python fallback")

            return True

        except Exception as e:
            logger.warning(f"Failed to initialize Rust FFI bridge: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get FFI bridge status."""
        return {
            "available": self.available,
            "library_loaded": self.ffi_lib is not None,
        }
