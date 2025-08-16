"""
Betanet Python Bridge

Integrates the Rust Betanet components with the existing AI Village Python infrastructure.
This bridge connects to the existing dual-path transport system and Navigator agent.
"""

import ctypes
import json
import logging
import platform
from ctypes import CDLL, POINTER, Structure, c_char_p, c_int, c_uint, c_void_p
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BetanetResult(ctypes.c_int):
    """Result codes from C API"""
    SUCCESS = 0
    ERROR = 1
    INVALID_PARAMETER = 2
    NETWORK_ERROR = 3
    CRYPTO_ERROR = 4


class BetanetConfig(Structure):
    """C configuration structure"""
    _fields_ = [
        ("listen_addr", c_char_p),
        ("enable_tcp", c_int),
        ("enable_quic", c_int),
        ("enable_noise_xk", c_int),
        ("max_connections", c_uint),
        ("connection_timeout_secs", c_uint),
    ]


class BetanetBridge:
    """Bridge between Rust Betanet components and Python infrastructure"""

    def __init__(self, library_path: Path | None = None):
        """Initialize the Betanet bridge"""
        self.library_path = library_path or self._find_library()
        self.lib = None
        self._clients = {}
        self._mixnodes = {}

    def _find_library(self) -> Path:
        """Find the Betanet C library"""
        system = platform.system().lower()

        if system == "windows":
            lib_name = "betanet_c.dll"
        elif system == "darwin":
            lib_name = "libbetanet_c.dylib"
        else:
            lib_name = "libbetanet_c.so"

        # Look in common locations
        search_paths = [
            Path(__file__).parent.parent / "target" / "release" / lib_name,
            Path(__file__).parent.parent / "target" / "debug" / lib_name,
            Path.cwd() / lib_name,
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError(f"Could not find Betanet library: {lib_name}")

    def initialize(self) -> bool:
        """Initialize the Betanet library"""
        try:
            self.lib = CDLL(str(self.library_path))

            # Define function signatures
            self.lib.betanet_init.restype = BetanetResult

            self.lib.betanet_htx_client_create.argtypes = [POINTER(BetanetConfig)]
            self.lib.betanet_htx_client_create.restype = c_void_p

            self.lib.betanet_htx_client_connect.argtypes = [c_void_p, c_char_p]
            self.lib.betanet_htx_client_connect.restype = BetanetResult

            self.lib.betanet_htx_client_send.argtypes = [c_void_p, POINTER(ctypes.c_uint8), c_uint]
            self.lib.betanet_htx_client_send.restype = BetanetResult

            self.lib.betanet_utls_generate_chrome_template.argtypes = [c_char_p, c_char_p, c_uint]
            self.lib.betanet_utls_generate_chrome_template.restype = BetanetResult

            self.lib.betanet_get_version.restype = c_char_p

            # Initialize library
            result = self.lib.betanet_init()
            if result != BetanetResult.SUCCESS:
                logger.error("Failed to initialize Betanet library")
                return False

            version = self.lib.betanet_get_version().decode('utf-8')
            logger.info(f"Initialized Betanet library version: {version}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Betanet library: {e}")
            return False

    def create_htx_client(self, config: dict[str, Any]) -> str | None:
        """Create an HTX client with integration to existing transport layer"""
        try:
            # Convert Python config to C config
            c_config = BetanetConfig()
            c_config.listen_addr = config.get("listen_addr", "127.0.0.1:9000").encode('utf-8')
            c_config.enable_tcp = 1 if config.get("enable_tcp", True) else 0
            c_config.enable_quic = 1 if config.get("enable_quic", False) else 0
            c_config.enable_noise_xk = 1 if config.get("enable_noise_xk", True) else 0
            c_config.max_connections = config.get("max_connections", 1000)
            c_config.connection_timeout_secs = config.get("connection_timeout_secs", 30)

            client_ptr = self.lib.betanet_htx_client_create(ctypes.byref(c_config))
            if not client_ptr:
                logger.error("Failed to create HTX client")
                return None

            client_id = f"htx_client_{len(self._clients)}"
            self._clients[client_id] = client_ptr

            logger.info(f"Created HTX client: {client_id}")
            return client_id

        except Exception as e:
            logger.error(f"Failed to create HTX client: {e}")
            return None

    def generate_chrome_fingerprint(self, hostname: str) -> dict[str, Any] | None:
        """Generate Chrome fingerprint template for uTLS"""
        try:
            buffer_size = 4096
            output_buffer = ctypes.create_string_buffer(buffer_size)

            result = self.lib.betanet_utls_generate_chrome_template(
                hostname.encode('utf-8'),
                output_buffer,
                buffer_size
            )

            if result != BetanetResult.SUCCESS:
                logger.error(f"Failed to generate Chrome fingerprint for {hostname}")
                return None

            fingerprint_json = output_buffer.value.decode('utf-8')
            fingerprint_data = json.loads(fingerprint_json)

            logger.info(f"Generated Chrome fingerprint for {hostname}: {fingerprint_data['fingerprint'][:16]}...")
            return fingerprint_data

        except Exception as e:
            logger.error(f"Failed to generate Chrome fingerprint: {e}")
            return None

    def integrate_with_dual_path_transport(self, transport_manager):
        """Integrate with existing dual-path transport system"""
        try:
            # This would integrate with the existing src/core/p2p/dual_path_transport.py
            # by registering our Rust HTX client as a transport option

            logger.info("Integrating Betanet HTX with dual-path transport")

            # Register HTX transport with the navigator
            transport_config = {
                "name": "betanet_htx",
                "type": "internet",
                "priority": 3,  # Lower than BitChat for mobile-first
                "capabilities": ["encrypted", "anonymous", "chrome_mimicry"],
                "resource_usage": "medium",
                "battery_impact": "low",
            }

            if hasattr(transport_manager, 'register_transport'):
                transport_manager.register_transport("betanet_htx", self, transport_config)
                logger.info("Registered Betanet HTX transport with dual-path manager")

            return True

        except Exception as e:
            logger.error(f"Failed to integrate with dual-path transport: {e}")
            return False

    def send_message(self, client_id: str, data: bytes) -> bool:
        """Send message via HTX client"""
        try:
            if client_id not in self._clients:
                logger.error(f"Unknown client: {client_id}")
                return False

            client_ptr = self._clients[client_id]

            # Convert bytes to C array
            data_array = (ctypes.c_uint8 * len(data))(*data)

            result = self.lib.betanet_htx_client_send(
                client_ptr,
                data_array,
                len(data)
            )

            if result != BetanetResult.SUCCESS:
                logger.error(f"Failed to send message via {client_id}")
                return False

            logger.debug(f"Sent {len(data)} bytes via {client_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def cleanup(self):
        """Clean up resources"""
        try:
            # Clean up clients
            for client_id, client_ptr in self._clients.items():
                self.lib.betanet_htx_client_destroy(client_ptr)
                logger.debug(f"Cleaned up client: {client_id}")

            # Clean up mixnodes
            for mixnode_id, mixnode_ptr in self._mixnodes.items():
                self.lib.betanet_mixnode_destroy(mixnode_ptr)
                logger.debug(f"Cleaned up mixnode: {mixnode_id}")

            self._clients.clear()
            self._mixnodes.clear()

            logger.info("Betanet bridge cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Integration helpers for existing AI Village infrastructure

async def integrate_betanet_with_aivillage():
    """
    Integration function to connect Betanet components with existing AI Village infrastructure.
    This should be called during system initialization.
    """
    try:
        # Initialize Betanet bridge
        bridge = BetanetBridge()
        if not bridge.initialize():
            logger.error("Failed to initialize Betanet bridge")
            return None

        # Try to integrate with existing transport system
        try:
            # Import existing dual-path transport
            from ...core.p2p.dual_path_transport import DualPathTransportManager

            transport_manager = DualPathTransportManager()
            bridge.integrate_with_dual_path_transport(transport_manager)

            logger.info("Successfully integrated Betanet with AI Village dual-path transport")

        except ImportError as e:
            logger.warning(f"Could not integrate with dual-path transport: {e}")

        # Create default HTX client for testing
        htx_config = {
            "listen_addr": "127.0.0.1:9000",
            "enable_tcp": True,
            "enable_noise_xk": True,
            "max_connections": 100,
        }

        client_id = bridge.create_htx_client(htx_config)
        if client_id:
            logger.info(f"Created default HTX client: {client_id}")

        return bridge

    except Exception as e:
        logger.error(f"Failed to integrate Betanet with AI Village: {e}")
        return None


def test_chrome_fingerprint_generation():
    """Test Chrome fingerprint generation"""
    bridge = BetanetBridge()
    if not bridge.initialize():
        print("Failed to initialize Betanet bridge")
        return

    # Test fingerprint generation for common sites
    test_hostnames = [
        "google.com",
        "github.com",
        "cloudflare.com",
        "example.com"
    ]

    for hostname in test_hostnames:
        fingerprint = bridge.generate_chrome_fingerprint(hostname)
        if fingerprint:
            print(f"{hostname}: {fingerprint['fingerprint']} (Chrome {fingerprint['chrome_version']})")
        else:
            print(f"Failed to generate fingerprint for {hostname}")

    bridge.cleanup()


if __name__ == "__main__":
    # Test the bridge
    test_chrome_fingerprint_generation()
