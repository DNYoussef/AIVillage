"""Enhanced Digital Twin Encryption for AIVillage SCION production systems.

Provides AES-256-GCM encryption with backward compatibility for Fernet,
automated key rotation, and comprehensive security features.
"""

import base64
from datetime import datetime, timedelta
import json
import logging
import os
import secrets
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class DigitalTwinEncryptionError(Exception):
    """Raised when digital twin encryption operations fail."""

    pass


class KeyRotationManager:
    """Manages automatic key rotation with versioning."""

    def __init__(self, rotation_days: int = 30):
        self.rotation_days = rotation_days
        self.key_versions = {}

    def generate_key_version(self) -> str:
        """Generate a new key version identifier."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}_{secrets.token_hex(4)}"

    def is_rotation_needed(self, key_created_at: datetime) -> bool:
        """Check if key rotation is needed."""
        return datetime.utcnow() - key_created_at > timedelta(days=self.rotation_days)

    def get_active_keys(self) -> dict[str, dict[str, Any]]:
        """Get all active key versions for decryption compatibility."""
        current_time = datetime.utcnow()
        active_keys = {}

        for version, key_data in self.key_versions.items():
            # Keep keys active for grace period after rotation
            grace_period = timedelta(days=self.rotation_days)
            if current_time - key_data["created_at"] <= grace_period * 2:
                active_keys[version] = key_data

        return active_keys


class EnhancedDigitalTwinEncryption:
    """Enhanced encryption system with AES-256-GCM and backward compatibility."""

    def __init__(self):
        """Initialize enhanced encryption system."""
        self.key_rotation_manager = KeyRotationManager()
        self.backend = default_backend()

        # Initialize encryption keys
        self._initialize_keys()

        # Backward compatibility with Fernet
        self._initialize_legacy_fernet()

    def _initialize_keys(self):
        """Initialize AES-256-GCM encryption keys."""
        master_key = os.environ.get("DIGITAL_TWIN_MASTER_KEY")
        if not master_key:
            logger.warning("DIGITAL_TWIN_MASTER_KEY not found, generating new key")
            master_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            os.environ["DIGITAL_TWIN_MASTER_KEY"] = master_key

        self.master_key = base64.urlsafe_b64decode(master_key.encode())

        # Initialize current encryption key
        self.current_key_version = self._get_or_create_current_key()

    def _initialize_legacy_fernet(self):
        """Initialize legacy Fernet cipher for backward compatibility."""
        legacy_key = os.environ.get("DIGITAL_TWIN_ENCRYPTION_KEY")
        if legacy_key:
            try:
                self.legacy_cipher = Fernet(legacy_key.encode())
                logger.info("Legacy Fernet cipher initialized for backward compatibility")
            except Exception as e:
                logger.warning(f"Failed to initialize legacy cipher: {e}")
                self.legacy_cipher = None
        else:
            self.legacy_cipher = None

    def _get_or_create_current_key(self) -> str:
        """Get or create the current encryption key version."""
        # Check if current key needs rotation
        current_version = os.environ.get("CURRENT_KEY_VERSION")

        if current_version and current_version in self.key_rotation_manager.key_versions:
            key_data = self.key_rotation_manager.key_versions[current_version]
            if not self.key_rotation_manager.is_rotation_needed(key_data["created_at"]):
                return current_version

        # Generate new key version
        new_version = self.key_rotation_manager.generate_key_version()
        key_data = {
            "key": self._derive_key_from_master(new_version),
            "created_at": datetime.utcnow(),
            "algorithm": "AES-256-GCM",
        }

        self.key_rotation_manager.key_versions[new_version] = key_data
        os.environ["CURRENT_KEY_VERSION"] = new_version

        logger.info(f"Generated new encryption key version: {new_version}")
        return new_version

    def _derive_key_from_master(self, version: str) -> bytes:
        """Derive a specific key from master key using version as salt."""
        salt = version.encode() + b"digital_twin_salt"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=self.backend  # AES-256
        )
        return kdf.derive(self.master_key)

    def encrypt_sensitive_field(self, data: Any, field_name: str) -> bytes:
        """Encrypt sensitive field data using AES-256-GCM."""
        try:
            data_str = json.dumps(data) if not isinstance(data, str) else data
            data_bytes = data_str.encode("utf-8")

            # Get current key
            current_key_data = self.key_rotation_manager.key_versions[self.current_key_version]
            key = current_key_data["key"]

            # Generate random IV/nonce
            iv = secrets.token_bytes(12)  # 96 bits for GCM

            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
            encryptor = cipher.encryptor()

            # Encrypt data
            ciphertext = encryptor.update(data_bytes) + encryptor.finalize()

            # Combine version, IV, tag, and ciphertext
            encrypted_package = {
                "version": self.current_key_version,
                "iv": base64.b64encode(iv).decode(),
                "tag": base64.b64encode(encryptor.tag).decode(),
                "data": base64.b64encode(ciphertext).decode(),
                "algorithm": "AES-256-GCM",
            }

            return json.dumps(encrypted_package).encode()

        except Exception as e:
            logger.error(f"Failed to encrypt {field_name}: {e}")
            raise DigitalTwinEncryptionError(f"Failed to encrypt {field_name}: {e}")

    def decrypt_sensitive_field(self, encrypted_data: bytes, field_name: str) -> str:
        """Decrypt sensitive field data with backward compatibility."""
        try:
            # Try to parse as JSON (new format)
            try:
                encrypted_str = encrypted_data.decode("utf-8")
                encrypted_package = json.loads(encrypted_str)
                return self._decrypt_gcm_package(encrypted_package)
            except (json.JSONDecodeError, KeyError):
                # Fall back to legacy Fernet decryption
                return self._decrypt_legacy_fernet(encrypted_data)

        except Exception as e:
            logger.error(f"Failed to decrypt {field_name}: {e}")
            raise DigitalTwinEncryptionError(f"Failed to decrypt {field_name}: {e}")

    def _decrypt_gcm_package(self, package: dict[str, Any]) -> str:
        """Decrypt AES-256-GCM encrypted package."""
        version = package["version"]
        iv = base64.b64decode(package["iv"].encode())
        tag = base64.b64decode(package["tag"].encode())
        ciphertext = base64.b64decode(package["data"].encode())

        # Get key for this version
        if version not in self.key_rotation_manager.key_versions:
            # Try to recreate key from master
            key_data = {
                "key": self._derive_key_from_master(version),
                "created_at": datetime.utcnow(),
                "algorithm": "AES-256-GCM",
            }
            self.key_rotation_manager.key_versions[version] = key_data

        key = self.key_rotation_manager.key_versions[version]["key"]

        # Decrypt
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext.decode("utf-8")

    def _decrypt_legacy_fernet(self, encrypted_data: bytes) -> str:
        """Decrypt legacy Fernet encrypted data."""
        if not self.legacy_cipher:
            raise DigitalTwinEncryptionError("Legacy encryption key not available")

        decrypted = self.legacy_cipher.decrypt(encrypted_data)
        return decrypted.decode("utf-8")

    def rotate_keys(self) -> str:
        """Manually trigger key rotation."""
        old_version = self.current_key_version
        self.current_key_version = self._get_or_create_current_key()

        logger.info(f"Key rotated from {old_version} to {self.current_key_version}")
        return self.current_key_version

    def get_key_status(self) -> dict[str, Any]:
        """Get current key rotation status."""
        current_key_data = self.key_rotation_manager.key_versions[self.current_key_version]
        created_at = current_key_data["created_at"]
        rotation_needed = self.key_rotation_manager.is_rotation_needed(created_at)

        return {
            "current_version": self.current_key_version,
            "created_at": created_at.isoformat(),
            "rotation_needed": rotation_needed,
            "days_until_rotation": (self.key_rotation_manager.rotation_days - (datetime.utcnow() - created_at).days),
            "active_versions": list(self.key_rotation_manager.get_active_keys().keys()),
            "algorithm": "AES-256-GCM",
        }


# Backward compatibility alias
DigitalTwinEncryption = EnhancedDigitalTwinEncryption
