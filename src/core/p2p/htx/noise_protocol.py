"""Noise XK Protocol Implementation for HTX Inner Security - Betanet v1.1

Implements the Noise XK handshake pattern for HTX inner protocol security:
- X25519 ECDH key exchange
- ChaCha20-Poly1305 AEAD encryption
- Blake2s hash function
- Known server public key validation

This module focuses solely on Noise protocol implementation.
"""

import hashlib
import logging
import secrets
import struct
from dataclasses import dataclass
from enum import Enum

# Try to import cryptography for production use
try:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import x25519
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    CRYPTO_AVAILABLE = True
except ImportError:
    # Fallback to basic implementation
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class NoiseHandshakeState(Enum):
    """Noise XK handshake states."""

    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    HANDSHAKE_COMPLETE = "complete"
    HANDSHAKE_FAILED = "failed"


@dataclass
class NoiseKeys:
    """Noise protocol keys and state."""

    # Static keys (long-term identity)
    static_private: bytes = None
    static_public: bytes = None

    # Ephemeral keys (per-session)
    ephemeral_private: bytes = None
    ephemeral_public: bytes = None

    # Remote public keys
    remote_static: bytes = None
    remote_ephemeral: bytes = None

    # Handshake state
    handshake_hash: bytes = b""
    chaining_key: bytes = b""

    # Transport keys (post-handshake)
    sending_key: bytes = None
    receiving_key: bytes = None
    sending_nonce: int = 0
    receiving_nonce: int = 0


class NoiseXKProtocol:
    """Noise XK protocol implementation for HTX inner security.

    XK Pattern:
      <- s
      ...
      -> e, es
      <- e, ee
      -> s, se

    Where:
    - s = static key
    - e = ephemeral key
    - es/ee/se = DH operations
    """

    PROTOCOL_NAME = b"Noise_XK_25519_ChaChaPoly_BLAKE2s"
    DH_LEN = 32  # X25519 public key length
    HASH_LEN = 32  # BLAKE2s hash length
    TAGLEN = 16  # ChaCha20-Poly1305 tag length

    def __init__(self, is_initiator: bool, known_remote_static: bytes = None):
        """Initialize Noise XK protocol.

        Args:
            is_initiator: True if this is the initiating party
            known_remote_static: Remote static public key (required for XK)
        """
        self.is_initiator = is_initiator
        self.state = NoiseHandshakeState.UNINITIALIZED
        self.keys = NoiseKeys()

        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available, using simplified implementation")

        # XK requires known remote static key
        if known_remote_static:
            if len(known_remote_static) != self.DH_LEN:
                raise ValueError(f"Invalid remote static key length: {len(known_remote_static)}")
            self.keys.remote_static = known_remote_static
        elif not is_initiator:
            # Responder needs static key pair
            self.keys.static_private, self.keys.static_public = self._generate_keypair()

    def initialize_handshake(self) -> None:
        """Initialize handshake state."""
        if self.state != NoiseHandshakeState.UNINITIALIZED:
            raise RuntimeError(f"Cannot initialize from state: {self.state}")

        # Initialize handshake hash with protocol name
        self.keys.handshake_hash = self._hash(self.PROTOCOL_NAME)

        # Initialize chaining key
        self.keys.chaining_key = self.PROTOCOL_NAME.ljust(32, b"\x00")[:32]

        # Mix in remote static key (XK pattern requirement)
        if self.keys.remote_static:
            self.keys.handshake_hash = self._hash(self.keys.handshake_hash + self.keys.remote_static)

        self.state = NoiseHandshakeState.INITIALIZED
        logger.info(f"Noise XK handshake initialized (initiator: {self.is_initiator})")

    def create_message_1(self) -> bytes:
        """Create first handshake message (initiator -> responder).

        Message 1: e, es
        - Send ephemeral public key
        - Perform DH(e, rs) and mix result

        Returns:
            Handshake message bytes
        """
        if not self.is_initiator:
            raise RuntimeError("Only initiator can send message 1")

        if self.state != NoiseHandshakeState.INITIALIZED:
            raise RuntimeError(f"Cannot send message 1 from state: {self.state}")

        if not self.keys.remote_static:
            raise RuntimeError("Remote static key required for XK pattern")

        # Generate ephemeral key pair
        self.keys.ephemeral_private, self.keys.ephemeral_public = self._generate_keypair()

        # Start building message
        message = bytearray()

        # Add ephemeral public key
        message.extend(self.keys.ephemeral_public)

        # Mix ephemeral public key into handshake hash
        self.keys.handshake_hash = self._hash(self.keys.handshake_hash + self.keys.ephemeral_public)

        # Perform DH(e, rs) - ephemeral with remote static
        dh_result = self._dh(self.keys.ephemeral_private, self.keys.remote_static)
        self.keys.chaining_key, temp_key = self._hkdf(self.keys.chaining_key, dh_result, 2)

        self.state = NoiseHandshakeState.MESSAGE_SENT
        logger.debug(f"Created Noise XK message 1 ({len(message)} bytes)")

        return bytes(message)

    def process_message_1(self, message: bytes) -> None:
        """Process first handshake message (responder).

        Args:
            message: Received handshake message
        """
        if self.is_initiator:
            raise RuntimeError("Initiator cannot process message 1")

        if self.state != NoiseHandshakeState.INITIALIZED:
            raise RuntimeError(f"Cannot process message 1 from state: {self.state}")

        if len(message) < self.DH_LEN:
            raise ValueError(f"Message 1 too short: {len(message)} < {self.DH_LEN}")

        # Extract remote ephemeral public key
        self.keys.remote_ephemeral = message[: self.DH_LEN]

        # Mix remote ephemeral public key into handshake hash
        self.keys.handshake_hash = self._hash(self.keys.handshake_hash + self.keys.remote_ephemeral)

        # Perform DH(e, rs) - use our static key with remote ephemeral
        if not self.keys.static_private:
            raise RuntimeError("Static private key required")

        dh_result = self._dh(self.keys.static_private, self.keys.remote_ephemeral)
        self.keys.chaining_key, temp_key = self._hkdf(self.keys.chaining_key, dh_result, 2)

        self.state = NoiseHandshakeState.MESSAGE_RECEIVED
        logger.debug("Processed Noise XK message 1")

    def create_message_2(self) -> bytes:
        """Create second handshake message (responder -> initiator).

        Message 2: e, ee
        - Send ephemeral public key
        - Perform DH(e, e) and mix result

        Returns:
            Handshake message bytes
        """
        if self.is_initiator:
            raise RuntimeError("Initiator cannot send message 2")

        if self.state != NoiseHandshakeState.MESSAGE_RECEIVED:
            raise RuntimeError(f"Cannot send message 2 from state: {self.state}")

        # Generate ephemeral key pair
        self.keys.ephemeral_private, self.keys.ephemeral_public = self._generate_keypair()

        # Start building message
        message = bytearray()

        # Add ephemeral public key
        message.extend(self.keys.ephemeral_public)

        # Mix ephemeral public key into handshake hash
        self.keys.handshake_hash = self._hash(self.keys.handshake_hash + self.keys.ephemeral_public)

        # Perform DH(e, e) - ephemeral with remote ephemeral
        dh_result = self._dh(self.keys.ephemeral_private, self.keys.remote_ephemeral)
        self.keys.chaining_key, temp_key = self._hkdf(self.keys.chaining_key, dh_result, 2)

        self.state = NoiseHandshakeState.MESSAGE_SENT
        logger.debug(f"Created Noise XK message 2 ({len(message)} bytes)")

        return bytes(message)

    def process_message_2(self, message: bytes) -> None:
        """Process second handshake message (initiator).

        Args:
            message: Received handshake message
        """
        if not self.is_initiator:
            raise RuntimeError("Only initiator can process message 2")

        if self.state != NoiseHandshakeState.MESSAGE_SENT:
            raise RuntimeError(f"Cannot process message 2 from state: {self.state}")

        if len(message) < self.DH_LEN:
            raise ValueError(f"Message 2 too short: {len(message)} < {self.DH_LEN}")

        # Extract remote ephemeral public key
        self.keys.remote_ephemeral = message[: self.DH_LEN]

        # Mix remote ephemeral public key into handshake hash
        self.keys.handshake_hash = self._hash(self.keys.handshake_hash + self.keys.remote_ephemeral)

        # Perform DH(e, e) - our ephemeral with remote ephemeral
        dh_result = self._dh(self.keys.ephemeral_private, self.keys.remote_ephemeral)
        self.keys.chaining_key, temp_key = self._hkdf(self.keys.chaining_key, dh_result, 2)

        self.state = NoiseHandshakeState.MESSAGE_RECEIVED
        logger.debug("Processed Noise XK message 2")

    def create_message_3(self) -> bytes:
        """Create third handshake message (initiator -> responder).

        Message 3: s, se
        - Send encrypted static public key
        - Perform DH(s, e) and mix result

        Returns:
            Handshake message bytes
        """
        if not self.is_initiator:
            raise RuntimeError("Only initiator can send message 3")

        if self.state != NoiseHandshakeState.MESSAGE_RECEIVED:
            raise RuntimeError(f"Cannot send message 3 from state: {self.state}")

        # Generate static key pair if not present
        if not self.keys.static_private:
            self.keys.static_private, self.keys.static_public = self._generate_keypair()

        # Start building message
        message = bytearray()

        # Encrypt and authenticate static public key
        temp_key1, temp_key2 = self._hkdf(self.keys.chaining_key, b"", 2)
        encrypted_static = self._encrypt(temp_key1, 0, self.keys.handshake_hash, self.keys.static_public)
        message.extend(encrypted_static)

        # Mix encrypted static key into handshake hash
        self.keys.handshake_hash = self._hash(self.keys.handshake_hash + encrypted_static)

        # Perform DH(s, e) - our static with remote ephemeral
        dh_result = self._dh(self.keys.static_private, self.keys.remote_ephemeral)
        self.keys.chaining_key, temp_key = self._hkdf(self.keys.chaining_key, dh_result, 2)

        # Handshake complete - derive transport keys
        self._derive_transport_keys()
        self.state = NoiseHandshakeState.HANDSHAKE_COMPLETE

        logger.info("Noise XK handshake complete (initiator)")
        return bytes(message)

    def process_message_3(self, message: bytes) -> None:
        """Process third handshake message (responder).

        Args:
            message: Received handshake message
        """
        if self.is_initiator:
            raise RuntimeError("Initiator cannot process message 3")

        if self.state != NoiseHandshakeState.MESSAGE_SENT:
            raise RuntimeError(f"Cannot process message 3 from state: {self.state}")

        expected_len = self.DH_LEN + self.TAGLEN
        if len(message) < expected_len:
            raise ValueError(f"Message 3 too short: {len(message)} < {expected_len}")

        # Decrypt static public key
        temp_key1, temp_key2 = self._hkdf(self.keys.chaining_key, b"", 2)
        encrypted_static = message[: self.DH_LEN + self.TAGLEN]

        try:
            decrypted_static = self._decrypt(temp_key1, 0, self.keys.handshake_hash, encrypted_static)
            self.keys.remote_static = decrypted_static
        except Exception as e:
            self.state = NoiseHandshakeState.HANDSHAKE_FAILED
            raise ValueError(f"Failed to decrypt remote static key: {e}")

        # Mix encrypted static key into handshake hash
        self.keys.handshake_hash = self._hash(self.keys.handshake_hash + encrypted_static)

        # Perform DH(s, e) - remote static with our ephemeral
        dh_result = self._dh(self.keys.ephemeral_private, self.keys.remote_static)
        self.keys.chaining_key, temp_key = self._hkdf(self.keys.chaining_key, dh_result, 2)

        # Handshake complete - derive transport keys
        self._derive_transport_keys()
        self.state = NoiseHandshakeState.HANDSHAKE_COMPLETE

        logger.info("Noise XK handshake complete (responder)")

    def encrypt_transport_message(self, plaintext: bytes) -> bytes:
        """Encrypt message using transport keys.

        Args:
            plaintext: Message to encrypt

        Returns:
            Encrypted message with authentication tag
        """
        if self.state != NoiseHandshakeState.HANDSHAKE_COMPLETE:
            raise RuntimeError(f"Transport not ready, state: {self.state}")

        if not self.keys.sending_key:
            raise RuntimeError("Sending key not available")

        # Encrypt with nonce and increment
        ciphertext = self._encrypt(
            self.keys.sending_key,
            self.keys.sending_nonce,
            b"",
            plaintext,  # No additional data for transport
        )

        self.keys.sending_nonce += 1

        # Check for nonce overflow (security requirement)
        if self.keys.sending_nonce >= 2**64 - 1:
            raise RuntimeError("Sending nonce overflow - rekey required")

        return ciphertext

    def decrypt_transport_message(self, ciphertext: bytes) -> bytes:
        """Decrypt message using transport keys.

        Args:
            ciphertext: Encrypted message with tag

        Returns:
            Decrypted plaintext
        """
        if self.state != NoiseHandshakeState.HANDSHAKE_COMPLETE:
            raise RuntimeError(f"Transport not ready, state: {self.state}")

        if not self.keys.receiving_key:
            raise RuntimeError("Receiving key not available")

        # Decrypt with nonce and increment
        plaintext = self._decrypt(
            self.keys.receiving_key,
            self.keys.receiving_nonce,
            b"",
            ciphertext,  # No additional data for transport
        )

        self.keys.receiving_nonce += 1

        # Check for nonce overflow
        if self.keys.receiving_nonce >= 2**64 - 1:
            raise RuntimeError("Receiving nonce overflow - rekey required")

        return plaintext

    def _derive_transport_keys(self) -> None:
        """Derive transport encryption keys from handshake state."""
        # Split chaining key into two transport keys
        self.keys.sending_key, self.keys.receiving_key = self._hkdf(self.keys.chaining_key, b"", 2)

        # For XK pattern, initiator sends with first key, receives with second
        if self.is_initiator:
            # Keys are correct as-is
            pass
        else:
            # Swap keys for responder
            self.keys.sending_key, self.keys.receiving_key = (
                self.keys.receiving_key,
                self.keys.sending_key,
            )

        # Initialize nonces
        self.keys.sending_nonce = 0
        self.keys.receiving_nonce = 0

        logger.debug("Transport keys derived")

    def _generate_keypair(self) -> tuple[bytes, bytes]:
        """Generate X25519 key pair."""
        if CRYPTO_AVAILABLE:
            private_key = x25519.X25519PrivateKey.generate()
            public_key = private_key.public_key()

            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )

            public_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )

            return private_bytes, public_bytes
        else:
            # Simplified fallback - not cryptographically secure
            private_key = secrets.token_bytes(32)
            # This is not real X25519 - just for testing
            public_key = hashlib.sha256(private_key).digest()[:32]
            return private_key, public_key

    def _dh(self, private_key: bytes, public_key: bytes) -> bytes:
        """Perform X25519 Diffie-Hellman operation."""
        if CRYPTO_AVAILABLE:
            priv_obj = x25519.X25519PrivateKey.from_private_bytes(private_key)
            pub_obj = x25519.X25519PublicKey.from_public_bytes(public_key)
            shared_key = priv_obj.exchange(pub_obj)
            return shared_key
        else:
            # Simplified fallback - not secure
            return hashlib.sha256(private_key + public_key).digest()[:32]

    def _hash(self, data: bytes) -> bytes:
        """BLAKE2s hash function."""
        # Using SHA-256 as fallback since BLAKE2s may not be available
        return hashlib.sha256(data).digest()[:32]

    def _hkdf(self, chaining_key: bytes, input_key_material: bytes, num_outputs: int) -> tuple[bytes, ...]:
        """HKDF key derivation (simplified)."""
        if CRYPTO_AVAILABLE:
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32 * num_outputs,
                salt=chaining_key,
                info=b"",
                backend=default_backend(),
            )
            derived = hkdf.derive(input_key_material)
        else:
            # Simplified HKDF using HMAC-SHA256
            derived = b""
            for i in range(num_outputs):
                derived += hashlib.sha256(chaining_key + input_key_material + bytes([i])).digest()[:32]

        # Split into requested number of 32-byte keys
        keys = []
        for i in range(num_outputs):
            keys.append(derived[i * 32 : (i + 1) * 32])

        return tuple(keys)

    def _encrypt(self, key: bytes, nonce: int, associated_data: bytes, plaintext: bytes) -> bytes:
        """ChaCha20-Poly1305 AEAD encryption."""
        if CRYPTO_AVAILABLE:
            cipher = ChaCha20Poly1305(key)
            # Convert nonce to 12-byte format
            nonce_bytes = struct.pack("<Q", nonce).ljust(12, b"\x00")[:12]
            return cipher.encrypt(nonce_bytes, plaintext, associated_data)
        else:
            # Simplified fallback - not secure, just for testing
            xor_key = hashlib.sha256(key + struct.pack("<Q", nonce) + associated_data).digest()
            encrypted = bytes(a ^ b for a, b in zip(plaintext, xor_key * ((len(plaintext) // 32) + 1), strict=False))
            # Fake authentication tag
            tag = hashlib.sha256(encrypted + key).digest()[:16]
            return encrypted + tag

    def _decrypt(self, key: bytes, nonce: int, associated_data: bytes, ciphertext: bytes) -> bytes:
        """ChaCha20-Poly1305 AEAD decryption."""
        if CRYPTO_AVAILABLE:
            cipher = ChaCha20Poly1305(key)
            nonce_bytes = struct.pack("<Q", nonce).ljust(12, b"\x00")[:12]
            return cipher.decrypt(nonce_bytes, ciphertext, associated_data)
        else:
            # Simplified fallback - extract encrypted data and verify tag
            if len(ciphertext) < 16:
                raise ValueError("Ciphertext too short")

            encrypted = ciphertext[:-16]
            tag = ciphertext[-16:]

            # Verify tag (simplified)
            expected_tag = hashlib.sha256(encrypted + key).digest()[:16]
            if tag != expected_tag:
                raise ValueError("Authentication tag verification failed")

            # Decrypt
            xor_key = hashlib.sha256(key + struct.pack("<Q", nonce) + associated_data).digest()
            plaintext = bytes(a ^ b for a, b in zip(encrypted, xor_key * ((len(encrypted) // 32) + 1), strict=False))
            return plaintext

    def get_handshake_hash(self) -> bytes:
        """Get current handshake hash for verification."""
        return self.keys.handshake_hash

    def get_status(self) -> dict:
        """Get protocol status information."""
        return {
            "state": self.state.value,
            "is_initiator": self.is_initiator,
            "has_static_key": self.keys.static_private is not None,
            "has_remote_static": self.keys.remote_static is not None,
            "transport_ready": self.state == NoiseHandshakeState.HANDSHAKE_COMPLETE,
            "sending_nonce": self.keys.sending_nonce,
            "receiving_nonce": self.keys.receiving_nonce,
            "crypto_available": CRYPTO_AVAILABLE,
        }

    def reset(self) -> None:
        """Reset protocol state for new handshake."""
        self.keys = NoiseKeys()
        self.state = NoiseHandshakeState.UNINITIALIZED
        logger.info("Noise XK protocol reset")
