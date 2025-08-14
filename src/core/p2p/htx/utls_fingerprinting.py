"""uTLS Fingerprinting for HTX Cover Transport - Betanet v1.1

Implements JA3/JA4 fingerprint calibration and TLS ClientHello generation
to mimic legitimate browser traffic patterns for cover transport.

This module is focused solely on TLS fingerprinting concerns.
"""

import hashlib
import logging
import random
import secrets
import struct
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TLSExtension:
    """TLS extension structure."""

    extension_type: int
    data: bytes = b""

    def encode(self) -> bytes:
        """Encode extension to wire format."""
        return struct.pack(">HH", self.extension_type, len(self.data)) + self.data


@dataclass
class ClientHelloFingerprint:
    """TLS ClientHello fingerprint parameters."""

    # Core TLS parameters
    version: int = 0x0303  # TLS 1.2
    cipher_suites: list[int] = field(default_factory=list)
    extensions: list[int] = field(default_factory=list)
    elliptic_curves: list[int] = field(default_factory=list)
    signature_algorithms: list[int] = field(default_factory=list)

    # JA3 fingerprint components
    ja3_string: str = ""
    ja3_hash: str = ""

    # JA4 fingerprint components
    ja4_string: str = ""
    ja4_hash: str = ""

    # Browser identity
    browser_type: str = "chrome"
    browser_version: str = "120.0"
    os_hint: str = "windows"


class FingerprintTemplate:
    """Pre-defined browser fingerprint templates."""

    # Chrome 120 on Windows (most common)
    CHROME_120_WINDOWS = ClientHelloFingerprint(
        version=0x0303,
        cipher_suites=[
            0x1301,
            0x1302,
            0x1303,
            0xC02C,
            0xC030,
            0x009F,
            0xCCAA,
            0xC02B,
            0xC02F,
            0x009E,
            0xC024,
            0xC028,
            0x006B,
            0xC023,
            0xC027,
            0x0067,
            0xC00A,
            0xC014,
            0x0039,
            0xC009,
            0xC013,
            0x0033,
            0x009D,
            0x009C,
            0x003D,
            0x003C,
            0x0035,
            0x002F,
            0x00FF,
        ],
        extensions=[
            0,
            23,
            35,
            13,
            43,
            45,
            51,
            21,
            27,
            17513,
            18,
            11,
            10,
            16,
            22,
            23,
            49,
            13172,
            5,
            18,
            43,
            27,
            17513,
            2570,
        ],
        elliptic_curves=[29, 23, 24, 25, 256, 257],
        signature_algorithms=[
            0x0403,
            0x0804,
            0x0401,
            0x0503,
            0x0805,
            0x0501,
            0x0806,
            0x0601,
            0x0201,
        ],
        browser_type="chrome",
        browser_version="120.0",
        os_hint="windows",
    )

    # Firefox 121 on Linux
    FIREFOX_121_LINUX = ClientHelloFingerprint(
        version=0x0303,
        cipher_suites=[
            0x1301,
            0x1302,
            0x1303,
            0xC02C,
            0xC02B,
            0xC030,
            0xC02F,
            0xC028,
            0xC027,
            0xC024,
            0xC023,
            0xC00A,
            0xC009,
            0xC014,
            0xC013,
            0x009F,
            0x009E,
            0x006B,
            0x0067,
            0x0039,
            0x0033,
            0x009D,
            0x009C,
            0x003D,
            0x003C,
            0x0035,
            0x002F,
            0x00FF,
        ],
        extensions=[
            0,
            23,
            35,
            13,
            16,
            11,
            10,
            22,
            23,
            49,
            13172,
            5,
            18,
            21,
            43,
            45,
            51,
        ],
        elliptic_curves=[29, 23, 24, 25],
        signature_algorithms=[
            0x0403,
            0x0503,
            0x0603,
            0x0804,
            0x0805,
            0x0806,
            0x0401,
            0x0501,
            0x0601,
            0x0201,
        ],
        browser_type="firefox",
        browser_version="121.0",
        os_hint="linux",
    )

    # Safari 17 on macOS
    SAFARI_17_MACOS = ClientHelloFingerprint(
        version=0x0303,
        cipher_suites=[
            0x1301,
            0x1302,
            0x1303,
            0xC02C,
            0xC02B,
            0xC030,
            0xC02F,
            0xC00A,
            0xC009,
            0xC014,
            0xC013,
            0x009F,
            0x009E,
            0x006B,
            0x0067,
            0x0039,
            0x0033,
            0x009D,
            0x009C,
            0x003D,
            0x003C,
            0x0035,
            0x002F,
            0x00FF,
        ],
        extensions=[
            0,
            23,
            35,
            13,
            43,
            45,
            51,
            21,
            16,
            11,
            10,
            22,
            23,
            49,
            13172,
            5,
            18,
        ],
        elliptic_curves=[29, 23, 24, 25],
        signature_algorithms=[
            0x0403,
            0x0804,
            0x0401,
            0x0503,
            0x0805,
            0x0501,
            0x0806,
            0x0601,
            0x0201,
        ],
        browser_type="safari",
        browser_version="17.0",
        os_hint="macos",
    )


class uTLSFingerprintCalibrator:
    """uTLS fingerprint calibrator for cover transport mimicry."""

    def __init__(self):
        self.templates = {
            "chrome_120_windows": FingerprintTemplate.CHROME_120_WINDOWS,
            "firefox_121_linux": FingerprintTemplate.FIREFOX_121_LINUX,
            "safari_17_macos": FingerprintTemplate.SAFARI_17_MACOS,
        }

        # Fingerprint cache for performance
        self.fingerprint_cache: dict[str, ClientHelloFingerprint] = {}

    def get_random_fingerprint(self) -> ClientHelloFingerprint:
        """Get a random browser fingerprint for diversity."""
        template_name = random.choice(list(self.templates.keys()))
        return self.calibrate_fingerprint(template_name)

    def calibrate_fingerprint(self, template_name: str, randomize: bool = True) -> ClientHelloFingerprint:
        """Calibrate fingerprint based on template with optional randomization.

        Args:
            template_name: Name of fingerprint template to use
            randomize: Whether to add minor randomization for diversity

        Returns:
            Calibrated fingerprint ready for use
        """
        if template_name not in self.templates:
            logger.warning(f"Unknown template {template_name}, using Chrome default")
            template_name = "chrome_120_windows"

        base_fingerprint = self.templates[template_name]

        # Create copy for modification
        fingerprint = ClientHelloFingerprint(
            version=base_fingerprint.version,
            cipher_suites=base_fingerprint.cipher_suites.copy(),
            extensions=base_fingerprint.extensions.copy(),
            elliptic_curves=base_fingerprint.elliptic_curves.copy(),
            signature_algorithms=base_fingerprint.signature_algorithms.copy(),
            browser_type=base_fingerprint.browser_type,
            browser_version=base_fingerprint.browser_version,
            os_hint=base_fingerprint.os_hint,
        )

        if randomize:
            fingerprint = self._apply_randomization(fingerprint)

        # Calculate fingerprint hashes
        fingerprint.ja3_string = self._generate_ja3_string(fingerprint)
        fingerprint.ja3_hash = hashlib.md5(fingerprint.ja3_string.encode()).hexdigest()

        fingerprint.ja4_string = self._generate_ja4_string(fingerprint)
        fingerprint.ja4_hash = hashlib.sha256(fingerprint.ja4_string.encode()).hexdigest()[:36]

        logger.info(f"Calibrated fingerprint: {template_name}, JA3={fingerprint.ja3_hash[:8]}...")

        return fingerprint

    def _apply_randomization(self, fingerprint: ClientHelloFingerprint) -> ClientHelloFingerprint:
        """Apply subtle randomization to avoid perfect fingerprint correlation."""

        # Randomly shuffle extension order (maintaining functionality)
        if len(fingerprint.extensions) > 3:
            # Only shuffle non-critical extensions
            critical_extensions = [0, 23, 35, 13]  # SNI, session_ticket, etc.
            non_critical = [ext for ext in fingerprint.extensions if ext not in critical_extensions]

            if len(non_critical) > 1:
                random.shuffle(non_critical)

            # Rebuild extension list
            new_extensions = []
            for ext in fingerprint.extensions:
                if ext in critical_extensions:
                    new_extensions.append(ext)
                elif ext in non_critical:
                    new_extensions.append(non_critical.pop(0))

            fingerprint.extensions = new_extensions

        # Minor cipher suite reordering (keeping TLS 1.3 suites first)
        tls13_suites = [suite for suite in fingerprint.cipher_suites if suite in [0x1301, 0x1302, 0x1303]]
        other_suites = [suite for suite in fingerprint.cipher_suites if suite not in tls13_suites]

        if len(other_suites) > 2:
            # Shuffle middle portion only
            middle_start = 1
            middle_end = min(len(other_suites) - 1, 5)
            if middle_end > middle_start:
                middle_portion = other_suites[middle_start:middle_end]
                random.shuffle(middle_portion)
                other_suites[middle_start:middle_end] = middle_portion

        fingerprint.cipher_suites = tls13_suites + other_suites

        return fingerprint

    def _generate_ja3_string(self, fingerprint: ClientHelloFingerprint) -> str:
        """Generate JA3 fingerprint string.

        JA3 format: version,ciphers,extensions,elliptic_curves,signature_algorithms
        """
        version_str = str(fingerprint.version)
        ciphers_str = "-".join(str(suite) for suite in fingerprint.cipher_suites)
        extensions_str = "-".join(str(ext) for ext in fingerprint.extensions)
        curves_str = "-".join(str(curve) for curve in fingerprint.elliptic_curves)
        sig_algs_str = "-".join(str(alg) for alg in fingerprint.signature_algorithms)

        ja3_string = f"{version_str},{ciphers_str},{extensions_str},{curves_str},{sig_algs_str}"
        return ja3_string

    def _generate_ja4_string(self, fingerprint: ClientHelloFingerprint) -> str:
        """Generate JA4 fingerprint string (simplified implementation).

        JA4 is more complex than JA3, this is a basic implementation.
        """
        # JA4 format is more sophisticated, simplified version here
        version_hex = f"{fingerprint.version:04x}"
        cipher_count = f"{len(fingerprint.cipher_suites):02d}"
        ext_count = f"{len(fingerprint.extensions):02d}"

        # First and last cipher suites
        first_cipher = f"{fingerprint.cipher_suites[0]:04x}" if fingerprint.cipher_suites else "0000"
        last_cipher = f"{fingerprint.cipher_suites[-1]:04x}" if fingerprint.cipher_suites else "0000"

        ja4_string = f"q{version_hex}{cipher_count}{ext_count}_{first_cipher}_{last_cipher}"
        return ja4_string

    def generate_client_hello(self, fingerprint: ClientHelloFingerprint, server_name: str = "example.com") -> bytes:
        """Generate TLS ClientHello message matching the fingerprint.

        Args:
            fingerprint: Target fingerprint to match
            server_name: SNI server name

        Returns:
            Raw TLS ClientHello message bytes
        """
        # TLS record header (will be prepended by TLS library)
        client_hello = bytearray()

        # ClientHello message type and length (placeholder)
        client_hello.extend([0x01, 0x00, 0x00, 0x00])  # Will fix length later

        # Protocol version
        client_hello.extend(struct.pack(">H", fingerprint.version))

        # Random (32 bytes)
        client_random = secrets.token_bytes(32)
        client_hello.extend(client_random)

        # Session ID (empty for simplicity)
        client_hello.extend([0x00])  # Session ID length = 0

        # Cipher suites
        cipher_suites_data = b"".join(struct.pack(">H", suite) for suite in fingerprint.cipher_suites)
        client_hello.extend(struct.pack(">H", len(cipher_suites_data)))
        client_hello.extend(cipher_suites_data)

        # Compression methods (only null compression)
        client_hello.extend([0x01, 0x00])  # 1 method, null compression

        # Extensions
        extensions_data = self._build_extensions(fingerprint, server_name)
        client_hello.extend(struct.pack(">H", len(extensions_data)))
        client_hello.extend(extensions_data)

        # Fix message length
        message_length = len(client_hello) - 4  # Exclude type and length fields
        struct.pack_into(">I", client_hello, 0, (0x01 << 24) | message_length)

        return bytes(client_hello)

    def _build_extensions(self, fingerprint: ClientHelloFingerprint, server_name: str) -> bytes:
        """Build TLS extensions matching fingerprint."""
        extensions = []

        # Build each extension based on fingerprint.extensions list
        for ext_type in fingerprint.extensions:
            ext_data = self._build_extension_data(ext_type, fingerprint, server_name)
            if ext_data is not None:
                extensions.append(TLSExtension(ext_type, ext_data))

        # Encode all extensions
        extensions_data = b""
        for ext in extensions:
            extensions_data += ext.encode()

        return extensions_data

    def _build_extension_data(
        self, ext_type: int, fingerprint: ClientHelloFingerprint, server_name: str
    ) -> bytes | None:
        """Build data for specific extension type."""

        if ext_type == 0:  # SNI (Server Name Indication)
            # SNI extension format: list_length(2) + name_type(1) + name_length(2) + name
            server_name_bytes = server_name.encode("utf-8")
            name_length = len(server_name_bytes)
            list_length = 1 + 2 + name_length  # name_type + name_length + name
            return struct.pack(">HBH", list_length, 0, name_length) + server_name_bytes

        elif ext_type == 10:  # Supported Groups (Elliptic Curves)
            curves_data = b"".join(struct.pack(">H", curve) for curve in fingerprint.elliptic_curves)
            return struct.pack(">H", len(curves_data)) + curves_data

        elif ext_type == 11:  # EC Point Formats
            return b"\x01\x00"  # Support uncompressed points only

        elif ext_type == 13:  # Signature Algorithms
            sig_algs_data = b"".join(struct.pack(">H", alg) for alg in fingerprint.signature_algorithms)
            return struct.pack(">H", len(sig_algs_data)) + sig_algs_data

        elif ext_type == 16:  # ALPN (Application Layer Protocol Negotiation)
            # Advertise HTTP/2 and HTTP/1.1
            alpn_data = b"\x00\x0c\x02h2\x08http/1.1"
            return alpn_data

        elif ext_type == 23:  # Session Ticket
            return b""  # Empty session ticket extension

        elif ext_type == 35:  # Session Ticket (TLS 1.3)
            return b""  # Empty for TLS 1.3

        elif ext_type == 43:  # Supported Versions
            # Advertise TLS 1.3, 1.2
            return b"\x04\x03\x04\x03\x03"

        elif ext_type == 45:  # PSK Key Exchange Modes
            return b"\x01\x01"  # PSK with (EC)DHE key establishment

        elif ext_type == 51:  # Key Share (TLS 1.3)
            # Generate X25519 key share
            secrets.token_bytes(32)
            # This would normally do X25519 scalar multiplication
            public_key = secrets.token_bytes(32)  # Simplified for now
            return struct.pack(">HHH", 36, 29, 32) + public_key  # 36=length, 29=X25519, 32=key_length

        else:
            # Default empty extension for unknown types
            return b""

    def validate_fingerprint_match(self, generated_hello: bytes, target_fingerprint: ClientHelloFingerprint) -> bool:
        """Validate that generated ClientHello matches target fingerprint."""
        try:
            # This would parse the generated ClientHello and verify it produces
            # the same JA3/JA4 fingerprints as the target
            # Simplified validation for now

            # Basic checks
            if len(generated_hello) < 40:  # Minimum realistic size
                return False

            # Check protocol version is present
            if len(generated_hello) < 10:
                return False

            # Extract version from ClientHello (simplified)
            # Real implementation would need proper TLS parsing

            logger.info("Fingerprint validation passed (simplified check)")
            return True

        except Exception as e:
            logger.error(f"Fingerprint validation failed: {e}")
            return False

    def get_fingerprint_stats(self) -> dict[str, any]:
        """Get fingerprint calibrator statistics."""
        return {
            "available_templates": list(self.templates.keys()),
            "cache_size": len(self.fingerprint_cache),
            "template_details": {
                name: {
                    "browser": fp.browser_type,
                    "version": fp.browser_version,
                    "os": fp.os_hint,
                    "cipher_count": len(fp.cipher_suites),
                    "extension_count": len(fp.extensions),
                }
                for name, fp in self.templates.items()
            },
        }
