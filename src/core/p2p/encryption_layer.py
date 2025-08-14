"""Encryption Layer for Secure P2P Communication."""

import asyncio
import base64
import contextlib
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration for encryption layer."""

    key_size: int = 2048  # RSA key size
    aes_key_size: int = 32  # 256-bit AES
    iv_size: int = 16  # 128-bit IV
    salt_size: int = 32  # 256-bit salt
    mac_size: int = 32  # 256-bit HMAC
    key_rotation_interval: float = 3600.0  # 1 hour
    max_message_age: float = 300.0  # 5 minutes
    enable_forward_secrecy: bool = True


class EncryptionLayer:
    """Secure encryption layer for P2P communications."""

    def __init__(self, node_id: str, config: SecurityConfig | None = None) -> None:
        self.node_id = node_id
        self.config = config or SecurityConfig()

        # Cryptographic keys
        self.private_key: rsa.RSAPrivateKey | None = None
        self.public_key: rsa.RSAPublicKey | None = None

        # Peer public keys
        self.peer_public_keys: dict[str, rsa.RSAPublicKey] = {}

        # Session keys for symmetric encryption
        self.session_keys: dict[str, bytes] = {}  # peer_id -> session_key
        self.session_created: dict[str, float] = {}  # peer_id -> creation_time

        # Message nonces for replay protection
        self.used_nonces: dict[str, float] = {}  # nonce -> timestamp

        # Statistics
        self.stats = {
            "messages_encrypted": 0,
            "messages_decrypted": 0,
            "key_exchanges": 0,
            "encryption_failures": 0,
            "decryption_failures": 0,
            "key_rotations": 0,
        }

        # Background tasks
        self.cleanup_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize encryption system."""
        try:
            # Generate RSA key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.config.key_size,
                backend=default_backend(),
            )
            self.public_key = self.private_key.public_key()

            # Start background cleanup
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info(f"Encryption layer initialized for node {self.node_id}")

        except Exception as e:
            logger.exception(f"Failed to initialize encryption: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown encryption system."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.cleanup_task

        # Clear sensitive data
        self.session_keys.clear()
        self.used_nonces.clear()

        logger.info("Encryption layer shutdown")

    async def encrypt_message(
        self, message_data: str, recipient_id: str | None = None
    ) -> bytes:
        """Encrypt message for transmission."""
        try:
            # Convert to bytes
            plaintext = message_data.encode("utf-8")

            if recipient_id:
                # Encrypted message for specific peer
                encrypted_data = await self._encrypt_for_peer(plaintext, recipient_id)
            else:
                # Broadcast message - use lighter encryption
                encrypted_data = await self._encrypt_broadcast(plaintext)

            self.stats["messages_encrypted"] += 1
            return encrypted_data

        except Exception as e:
            logger.exception(f"Encryption failed: {e}")
            self.stats["encryption_failures"] += 1
            raise

    async def decrypt_message(
        self, encrypted_data: bytes, sender_id: str | None = None
    ) -> str:
        """Decrypt received message."""
        try:
            if sender_id:
                # Decrypt from specific peer
                plaintext = await self._decrypt_from_peer(encrypted_data, sender_id)
            else:
                # Try to decrypt as broadcast message
                plaintext = await self._decrypt_broadcast(encrypted_data)

            self.stats["messages_decrypted"] += 1
            return plaintext.decode("utf-8")

        except Exception as e:
            logger.exception(f"Decryption failed: {e}")
            self.stats["decryption_failures"] += 1
            raise

    async def _encrypt_for_peer(self, plaintext: bytes, peer_id: str) -> bytes:
        """Encrypt message for specific peer using session key."""
        # Get or create session key
        session_key = await self._get_session_key(peer_id)

        # Generate IV and nonce
        iv = os.urandom(self.config.iv_size)
        nonce = secrets.token_bytes(16)
        timestamp = int(time.time()).to_bytes(8, "big")

        # Create authenticated data
        auth_data = peer_id.encode("utf-8") + timestamp + nonce

        # Encrypt with AES-GCM
        cipher = Cipher(
            algorithms.AES(session_key), modes.GCM(iv), backend=default_backend()
        )

        encryptor = cipher.encryptor()
        encryptor.authenticate_additional_data(auth_data)
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        # Construct final message: [auth_data][iv][tag][ciphertext]
        encrypted_message = auth_data + iv + encryptor.tag + ciphertext

        return encrypted_message

    async def _decrypt_from_peer(self, encrypted_data: bytes, sender_id: str) -> bytes:
        """Decrypt message from specific peer using session key."""
        try:
            # Parse message components
            auth_data_size = (
                len(sender_id.encode("utf-8")) + 8 + 16
            )  # sender + timestamp + nonce
            auth_data = encrypted_data[:auth_data_size]
            iv = encrypted_data[auth_data_size : auth_data_size + self.config.iv_size]
            tag = encrypted_data[
                auth_data_size + self.config.iv_size : auth_data_size
                + self.config.iv_size
                + 16
            ]
            ciphertext = encrypted_data[auth_data_size + self.config.iv_size + 16 :]

            # Extract timestamp and nonce for replay protection
            sender_bytes = sender_id.encode("utf-8")
            timestamp_bytes = auth_data[len(sender_bytes) : len(sender_bytes) + 8]
            nonce = auth_data[len(sender_bytes) + 8 :]

            # Check timestamp
            timestamp = int.from_bytes(timestamp_bytes, "big")
            if abs(time.time() - timestamp) > self.config.max_message_age:
                msg = "Message too old"
                raise ValueError(msg)

            # Check nonce for replay protection
            nonce_key = base64.b64encode(nonce).decode("utf-8")
            if nonce_key in self.used_nonces:
                msg = "Replay attack detected"
                raise ValueError(msg)
            self.used_nonces[nonce_key] = time.time()

            # Get session key
            session_key = await self._get_session_key(sender_id)

            # Decrypt
            cipher = Cipher(
                algorithms.AES(session_key),
                modes.GCM(iv, tag),
                backend=default_backend(),
            )

            decryptor = cipher.decryptor()
            decryptor.authenticate_additional_data(auth_data)
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            return plaintext

        except Exception as e:
            logger.exception(f"Failed to decrypt message from {sender_id}: {e}")
            raise

    async def _encrypt_broadcast(self, plaintext: bytes) -> bytes:
        """Encrypt broadcast message (simpler encryption for efficiency)."""
        # Use node's private key for signing and simple symmetric encryption
        # This is less secure but more efficient for broadcast messages

        # Generate key from node private key (deterministic)
        key_material = self.private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        # Derive broadcast key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"broadcast_salt_" + self.node_id.encode("utf-8")[:16],
            iterations=10000,
            backend=default_backend(),
        )
        broadcast_key = kdf.derive(key_material[:32])

        # Encrypt with AES-CTR (simpler than GCM for broadcast)
        iv = os.urandom(16)
        cipher = Cipher(
            algorithms.AES(broadcast_key), modes.CTR(iv), backend=default_backend()
        )

        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        # Add HMAC for integrity
        mac = hmac.new(broadcast_key, iv + ciphertext, hashlib.sha256).digest()

        # Format: [iv][mac][ciphertext]
        return iv + mac + ciphertext

    async def _decrypt_broadcast(self, encrypted_data: bytes) -> bytes:
        """Decrypt broadcast message."""
        try:
            # Parse components
            encrypted_data[:16]
            encrypted_data[16:48]
            encrypted_data[48:]

            # We don't know the sender, so we can't decrypt
            # This is a limitation of the broadcast encryption approach
            # In a real implementation, we'd need a different strategy
            msg = "Cannot decrypt broadcast message without sender information"
            raise ValueError(msg)

        except Exception as e:
            logger.exception(f"Failed to decrypt broadcast message: {e}")
            raise

    async def _get_session_key(self, peer_id: str) -> bytes:
        """Get or create session key for peer."""
        current_time = time.time()

        # Check if we need key rotation
        if (
            peer_id in self.session_keys
            and peer_id in self.session_created
            and current_time - self.session_created[peer_id]
            > self.config.key_rotation_interval
        ):
            # Rotate key
            await self._rotate_session_key(peer_id)

        # Create new session key if needed
        if peer_id not in self.session_keys:
            await self._create_session_key(peer_id)

        return self.session_keys[peer_id]

    async def _create_session_key(self, peer_id: str) -> None:
        """Create new session key for peer."""
        try:
            # Generate random session key
            session_key = os.urandom(self.config.aes_key_size)

            # Store session key
            self.session_keys[peer_id] = session_key
            self.session_created[peer_id] = time.time()

            # In a real implementation, we'd exchange this key securely with the peer
            # using RSA public key encryption or a key exchange protocol like ECDH

            self.stats["key_exchanges"] += 1
            logger.debug(f"Created session key for peer {peer_id}")

        except Exception as e:
            logger.exception(f"Failed to create session key for {peer_id}: {e}")
            raise

    async def _rotate_session_key(self, peer_id: str) -> None:
        """Rotate session key for peer."""
        try:
            # Remove old key
            if peer_id in self.session_keys:
                del self.session_keys[peer_id]
            if peer_id in self.session_created:
                del self.session_created[peer_id]

            # Create new key
            await self._create_session_key(peer_id)

            self.stats["key_rotations"] += 1
            logger.info(f"Rotated session key for peer {peer_id}")

        except Exception as e:
            logger.exception(f"Failed to rotate session key for {peer_id}: {e}")

    async def exchange_public_keys(
        self, peer_id: str, peer_public_key_data: bytes
    ) -> None:
        """Exchange public keys with peer."""
        try:
            # Parse peer's public key
            peer_public_key = serialization.load_der_public_key(
                peer_public_key_data, backend=default_backend()
            )

            # Verify it's an RSA key
            if not isinstance(peer_public_key, rsa.RSAPublicKey):
                msg = "Invalid public key type"
                raise ValueError(msg)

            # Store peer's public key
            self.peer_public_keys[peer_id] = peer_public_key

            logger.info(f"Exchanged public keys with peer {peer_id}")

        except Exception as e:
            logger.exception(f"Failed to exchange keys with {peer_id}: {e}")
            raise

    def get_public_key_data(self) -> bytes:
        """Get our public key in DER format for sharing."""
        if not self.public_key:
            msg = "Encryption not initialized"
            raise ValueError(msg)

        return self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired data."""
        while True:
            try:
                current_time = time.time()

                # Clean up old nonces
                expired_nonces = [
                    nonce
                    for nonce, timestamp in self.used_nonces.items()
                    if current_time - timestamp > self.config.max_message_age * 2
                ]

                for nonce in expired_nonces:
                    del self.used_nonces[nonce]

                # Clean up expired session keys
                expired_sessions = [
                    peer_id
                    for peer_id, timestamp in self.session_created.items()
                    if current_time - timestamp > self.config.key_rotation_interval * 2
                ]

                for peer_id in expired_sessions:
                    if peer_id in self.session_keys:
                        del self.session_keys[peer_id]
                    del self.session_created[peer_id]
                    logger.debug(f"Cleaned up expired session for {peer_id}")

                # Sleep for cleanup interval
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.exception(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)

    def verify_message_integrity(
        self, message_data: bytes, signature: bytes, sender_id: str
    ) -> bool:
        """Verify message integrity using sender's public key."""
        try:
            if sender_id not in self.peer_public_keys:
                logger.warning(f"No public key for peer {sender_id}")
                return False

            peer_public_key = self.peer_public_keys[sender_id]

            # Verify signature
            peer_public_key.verify(
                signature,
                message_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            return True

        except Exception as e:
            logger.exception(f"Signature verification failed for {sender_id}: {e}")
            return False

    def sign_message(self, message_data: bytes) -> bytes:
        """Sign message with our private key."""
        try:
            if not self.private_key:
                msg = "Encryption not initialized"
                raise ValueError(msg)

            signature = self.private_key.sign(
                message_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            return signature

        except Exception as e:
            logger.exception(f"Message signing failed: {e}")
            raise

    def get_encryption_stats(self) -> dict[str, Any]:
        """Get encryption statistics."""
        return {
            **self.stats,
            "session_keys_active": len(self.session_keys),
            "peer_public_keys": len(self.peer_public_keys),
            "used_nonces": len(self.used_nonces),
            "encryption_enabled": self.private_key is not None,
        }
