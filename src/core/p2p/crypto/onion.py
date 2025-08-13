"""Onion Routing Cryptography for Betanet

Provides real layered encryption using X25519 + ChaCha20-Poly1305 for each hop.
Replaces the JSON-based placeholder encryption with actual cryptographic layers.

Architecture:
- Each hop has an X25519 keypair
- Sender builds layers from inside-out (destination to first hop)
- Each layer encrypts: next_hop_id + encrypted_inner_payload
- Intermediate nodes peel one layer, forward to next hop
- Final hop gets the original payload
"""

import logging
import os
import struct
from dataclasses import dataclass

# Cryptographic primitives
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
        X25519PublicKey,
    )
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning(
        "Cryptography library not available. Onion routing will be disabled."
    )

logger = logging.getLogger(__name__)


@dataclass
class OnionLayer:
    """Single layer of onion encryption"""

    next_hop: str  # Peer ID of next hop
    encrypted_payload: bytes  # Encrypted inner content
    layer_mac: bytes  # Authentication tag for this layer


@dataclass
class OnionHop:
    """Configuration for one hop in the onion route"""

    hop_id: str  # Peer ID
    public_key: bytes  # X25519 public key (32 bytes)


class OnionCrypto:
    """Onion routing cryptography implementation"""

    def __init__(self):
        if not CRYPTO_AVAILABLE:
            raise ImportError("Cryptography library required for onion routing")

        # Generate our own X25519 keypair for this node
        self.private_key = X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

        # Cache for derived shared secrets (performance optimization)
        self.shared_secret_cache: dict[str, bytes] = {}

        logger.debug("OnionCrypto initialized with X25519 keypair")

    def get_public_key_bytes(self) -> bytes:
        """Get our public key as bytes"""
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

    def build_layers(self, payload: bytes, hop_route: list[OnionHop]) -> bytes:
        """Build onion layers from inside-out

        Args:
            payload: Original message payload
            hop_route: List of hops (first = destination, last = first hop to send to)

        Returns:
            Encrypted onion ready to send to first hop
        """
        if not hop_route:
            raise ValueError("Empty hop route")

        # Start with the original payload
        current_payload = payload

        # Build layers from inside-out (destination first)
        for i, hop in enumerate(reversed(hop_route)):
            # For all but the last layer, include next hop info
            if i == 0:
                # Final layer: just the payload
                inner_data = current_payload
            else:
                # Intermediate layer: next_hop + payload
                next_hop = hop_route[len(hop_route) - i]
                inner_data = self._pack_layer_data(next_hop.hop_id, current_payload)

            # Encrypt this layer
            current_payload = self._encrypt_layer(inner_data, hop.public_key)
            logger.debug(
                f"Built layer {i+1} for hop {hop.hop_id}, size: {len(current_payload)}"
            )

        logger.info(
            f"Built onion with {len(hop_route)} layers, final size: {len(current_payload)}"
        )
        return current_payload

    def peel_layer(
        self, encrypted_onion: bytes, our_private_key: X25519PrivateKey | None = None
    ) -> tuple[str | None, bytes]:
        """Peel one layer of the onion

        Args:
            encrypted_onion: Encrypted onion data
            our_private_key: Private key to use (defaults to our key)

        Returns:
            (next_hop_id, decrypted_inner_payload) or (None, original_payload) if final hop
        """
        if our_private_key is None:
            our_private_key = self.private_key

        try:
            # Decrypt the outer layer
            inner_data = self._decrypt_layer(encrypted_onion, our_private_key)

            # Try to unpack as intermediate layer (next_hop + payload)
            try:
                next_hop, inner_payload = self._unpack_layer_data(inner_data)
                logger.debug(
                    f"Peeled layer, next hop: {next_hop}, payload size: {len(inner_payload)}"
                )
                return next_hop, inner_payload
            except ValueError:
                # Not an intermediate layer, must be the final payload
                logger.debug(f"Reached final layer, payload size: {len(inner_data)}")
                return None, inner_data

        except Exception as e:
            logger.error(f"Failed to peel onion layer: {e}")
            raise

    def _encrypt_layer(self, data: bytes, recipient_pubkey: bytes) -> bytes:
        """Encrypt data for a specific recipient using X25519 + ChaCha20-Poly1305"""
        # Generate ephemeral keypair for this layer
        ephemeral_private = X25519PrivateKey.generate()
        ephemeral_public = ephemeral_private.public_key()

        # Perform ECDH with recipient's public key
        recipient_x25519_key = X25519PublicKey.from_public_bytes(recipient_pubkey)
        shared_secret = ephemeral_private.exchange(recipient_x25519_key)

        # Derive encryption key using HKDF
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,  # ChaCha20 key size
            salt=b"betanet-onion-v1",
            info=b"layer-encryption",
        ).derive(shared_secret)

        # Encrypt with ChaCha20-Poly1305
        cipher = ChaCha20Poly1305(derived_key)
        nonce = os.urandom(12)  # 96-bit nonce for ChaCha20-Poly1305
        ciphertext = cipher.encrypt(nonce, data, None)

        # Format: ephemeral_pubkey (32) + nonce (12) + ciphertext
        ephemeral_pubkey_bytes = ephemeral_public.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

        return ephemeral_pubkey_bytes + nonce + ciphertext

    def _decrypt_layer(
        self, encrypted_data: bytes, our_private_key: X25519PrivateKey
    ) -> bytes:
        """Decrypt a layer using our private key"""
        if len(encrypted_data) < 32 + 12 + 16:  # pubkey + nonce + min_ciphertext
            raise ValueError("Invalid encrypted layer format")

        # Parse components
        ephemeral_pubkey_bytes = encrypted_data[:32]
        nonce = encrypted_data[32:44]
        ciphertext = encrypted_data[44:]

        # Reconstruct ephemeral public key
        ephemeral_public = X25519PublicKey.from_public_bytes(ephemeral_pubkey_bytes)

        # Perform ECDH
        shared_secret = our_private_key.exchange(ephemeral_public)

        # Derive the same key used for encryption
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"betanet-onion-v1",
            info=b"layer-encryption",
        ).derive(shared_secret)

        # Decrypt
        cipher = ChaCha20Poly1305(derived_key)
        plaintext = cipher.decrypt(nonce, ciphertext, None)

        return plaintext

    def _pack_layer_data(self, next_hop: str, payload: bytes) -> bytes:
        """Pack next hop info and payload for intermediate layers"""
        # Format: next_hop_len (4) + next_hop + payload
        next_hop_bytes = next_hop.encode("utf-8")
        return struct.pack("!I", len(next_hop_bytes)) + next_hop_bytes + payload

    def _unpack_layer_data(self, data: bytes) -> tuple[str, bytes]:
        """Unpack next hop info and payload from intermediate layers"""
        if len(data) < 4:
            raise ValueError("Invalid layer data")

        # Parse next hop length
        next_hop_len = struct.unpack("!I", data[:4])[0]

        if len(data) < 4 + next_hop_len:
            raise ValueError("Invalid layer data length")

        # Extract next hop and payload
        next_hop = data[4 : 4 + next_hop_len].decode("utf-8")
        payload = data[4 + next_hop_len :]

        return next_hop, payload


# Global onion crypto instance (lazy initialization)
_onion_crypto: OnionCrypto | None = None


def get_onion_crypto() -> OnionCrypto:
    """Get global onion crypto instance"""
    global _onion_crypto
    if _onion_crypto is None:
        _onion_crypto = OnionCrypto()
    return _onion_crypto


def build_onion_layers(payload: bytes, hop_pubkeys: list[tuple[str, bytes]]) -> bytes:
    """Build onion layers for a route

    Args:
        payload: Message payload to encrypt
        hop_pubkeys: List of (hop_id, pubkey_bytes) tuples

    Returns:
        Encrypted onion ready to send to first hop
    """
    if not CRYPTO_AVAILABLE:
        logger.warning("Crypto unavailable, returning plaintext (INSECURE)")
        return payload

    crypto = get_onion_crypto()
    hop_route = [OnionHop(hop_id, pubkey) for hop_id, pubkey in hop_pubkeys]
    return crypto.build_layers(payload, hop_route)


def peel_onion_layer(encrypted_onion: bytes) -> tuple[str | None, bytes]:
    """Peel one layer from an onion

    Args:
        encrypted_onion: Encrypted onion data

    Returns:
        (next_hop_id, inner_payload) or (None, final_payload)
    """
    if not CRYPTO_AVAILABLE:
        logger.warning("Crypto unavailable, returning as-is (INSECURE)")
        return None, encrypted_onion

    crypto = get_onion_crypto()
    return crypto.peel_layer(encrypted_onion)


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate X25519 keypair for onion routing

    Returns:
        (private_key_bytes, public_key_bytes)
    """
    if not CRYPTO_AVAILABLE:
        # Return dummy keys
        return b"dummy_private_key_32_bytes_long", b"dummy_public_key_32_bytes_long!"

    private_key = X25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )

    return private_bytes, public_bytes


# Test utilities
def test_onion_routing():
    """Test the onion routing implementation"""
    if not CRYPTO_AVAILABLE:
        print("Crypto not available, skipping test")
        return False

    print("Testing onion routing...")

    # Generate test keypairs for 3 hops
    hop_keys = []
    for i in range(3):
        private_bytes, public_bytes = generate_keypair()
        hop_keys.append((f"hop_{i}", private_bytes, public_bytes))

    # Build onion
    original_payload = b"Hello through 3 hops!"
    hop_pubkeys = [(hop_id, pub_key) for hop_id, _, pub_key in hop_keys]

    onion = build_onion_layers(original_payload, hop_pubkeys)
    print(f"Built onion: {len(onion)} bytes")

    # Test peeling layers
    current_onion = onion
    for i, (_hop_id, priv_key, _) in enumerate(hop_keys):
        # Reconstruct private key
        x25519_private = X25519PrivateKey.from_private_bytes(priv_key)

        crypto = get_onion_crypto()
        next_hop, inner = crypto.peel_layer(current_onion, x25519_private)

        if i < len(hop_keys) - 1:
            print(f"Hop {i}: next_hop={next_hop}, payload_size={len(inner)}")
            assert next_hop is not None
            current_onion = inner
        else:
            print(f"Final hop: payload={inner}")
            assert next_hop is None
            assert inner == original_payload

    print("✓ Onion routing test passed!")
    return True


if __name__ == "__main__":
    test_onion_routing()
