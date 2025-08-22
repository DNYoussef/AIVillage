#!/usr/bin/env python3
"""Digital Twin Data Encryption System.

Provides CODEX-compliant encryption for Digital Twin sensitive data fields
with 32-byte base64 key requirement and GDPR/COPPA/FERPA compliance.
"""

import base64
import hashlib
import logging
import os
from datetime import datetime, timedelta
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class DigitalTwinEncryptionError(Exception):
    """Raised when digital twin encryption operations fail."""


class ComplianceViolationError(Exception):
    """Raised when compliance requirements are violated."""


class DigitalTwinEncryption:
    """Secure encryption for Digital Twin sensitive data fields."""

    def __init__(self, encryption_key: str | None = None) -> None:
        """Initialize encryption with CODEX-compliant configuration.

        Args:
            encryption_key: Base64-encoded 32-byte key from environment
                          (DIGITAL_TWIN_ENCRYPTION_KEY)
        """
        self.encryption_key = encryption_key or os.getenv("DIGITAL_TWIN_ENCRYPTION_KEY")

        if not self.encryption_key:
            msg = "DIGITAL_TWIN_ENCRYPTION_KEY environment variable is required"
            raise DigitalTwinEncryptionError(msg)

        # Validate and derive Fernet key
        self.fernet_key = self._validate_and_derive_key(self.encryption_key)
        self.cipher = Fernet(self.fernet_key)

        # Compliance flags from environment
        self.coppa_compliant = os.getenv("DIGITAL_TWIN_COPPA_COMPLIANT", "true").lower() == "true"
        self.ferpa_compliant = os.getenv("DIGITAL_TWIN_FERPA_COMPLIANT", "true").lower() == "true"
        self.gdpr_compliant = os.getenv("DIGITAL_TWIN_GDPR_COMPLIANT", "true").lower() == "true"

        # Data retention settings
        self.profile_ttl_days = int(os.getenv("DIGITAL_TWIN_PROFILE_TTL_DAYS", "365"))

        logger.info(
            "Digital Twin encryption initialized with compliance flags: "
            f"COPPA={self.coppa_compliant}, FERPA={self.ferpa_compliant}, "
            f"GDPR={self.gdpr_compliant}"
        )

    def _validate_and_derive_key(self, base64_key: str) -> bytes:
        """Validate and derive Fernet key from base64 encoded key.

        Args:
            base64_key: Base64-encoded 32-byte key

        Returns:
            Fernet-compatible key

        Raises:
            DigitalTwinEncryptionError: If key is invalid
        """
        try:
            # Decode base64 key
            key_bytes = base64.b64decode(base64_key)

            # Validate key is exactly 32 bytes as required by CODEX specs
            if len(key_bytes) != 32:
                msg = f"Encryption key must be exactly 32 bytes after base64 decoding, " f"got {len(key_bytes)} bytes"
                raise DigitalTwinEncryptionError(msg)

            # Derive Fernet-compatible key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"digital_twin_salt_aivillage_2025",
                iterations=100000,
                backend=default_backend(),
            )

            fernet_key = base64.urlsafe_b64encode(kdf.derive(key_bytes))

            return fernet_key

        except Exception as e:
            msg = f"Invalid encryption key: {e}"
            raise DigitalTwinEncryptionError(msg)

    def encrypt_sensitive_field(self, plaintext_value: Any, field_name: str = "") -> bytes:
        """Encrypt sensitive field value for database storage.

        Args:
            plaintext_value: Value to encrypt
            field_name: Field name for audit logging

        Returns:
            Encrypted bytes suitable for BLOB storage
        """
        try:
            # Convert to string if needed
            if not isinstance(plaintext_value, str):
                plaintext_value = str(plaintext_value)

            # Add metadata for compliance
            metadata = {
                "field": field_name,
                "encrypted_at": datetime.utcnow().isoformat(),
                "compliance_flags": {
                    "coppa": self.coppa_compliant,
                    "ferpa": self.ferpa_compliant,
                    "gdpr": self.gdpr_compliant,
                },
            }

            # Create payload with metadata
            payload = {"data": plaintext_value, "meta": metadata}

            # Serialize and encrypt
            import json

            json_data = json.dumps(payload).encode("utf-8")
            encrypted_data = self.cipher.encrypt(json_data)

            logger.debug(f"Encrypted sensitive field: {field_name}")
            return encrypted_data

        except Exception as e:
            logger.exception(f"Failed to encrypt field {field_name}: {e}")
            msg = f"Encryption failed for {field_name}: {e}"
            raise DigitalTwinEncryptionError(msg)

    def decrypt_sensitive_field(self, encrypted_data: bytes, field_name: str = "") -> Any:
        """Decrypt sensitive field value from database.

        Args:
            encrypted_data: Encrypted bytes from database
            field_name: Field name for audit logging

        Returns:
            Decrypted original value
        """
        try:
            # Decrypt data
            decrypted_bytes = self.cipher.decrypt(encrypted_data)

            # Deserialize payload
            import json

            payload = json.loads(decrypted_bytes.decode("utf-8"))

            # Validate compliance requirements
            self._validate_compliance_access(payload.get("meta", {}), field_name)

            # Extract data
            original_value = payload["data"]

            logger.debug(f"Decrypted sensitive field: {field_name}")
            return original_value

        except Exception as e:
            logger.exception(f"Failed to decrypt field {field_name}: {e}")
            msg = f"Decryption failed for {field_name}: {e}"
            raise DigitalTwinEncryptionError(msg)

    def _validate_compliance_access(self, metadata: dict[str, Any], field_name: str) -> None:
        """Validate compliance requirements for data access.

        Args:
            metadata: Encrypted data metadata
            field_name: Field being accessed

        Raises:
            ComplianceViolationError: If compliance requirements not met
        """
        if not metadata:
            return

        compliance_flags = metadata.get("compliance_flags", {})

        # Check if current compliance settings allow access
        if compliance_flags.get("coppa") and not self.coppa_compliant:
            msg = f"COPPA compliance required to access {field_name}"
            raise ComplianceViolationError(msg)

        if compliance_flags.get("ferpa") and not self.ferpa_compliant:
            msg = f"FERPA compliance required to access {field_name}"
            raise ComplianceViolationError(msg)

        if compliance_flags.get("gdpr") and not self.gdpr_compliant:
            msg = f"GDPR compliance required to access {field_name}"
            raise ComplianceViolationError(msg)

    def hash_user_id(self, user_id: str) -> str:
        """Create privacy-compliant hash of user ID.

        Args:
            user_id: Original user identifier

        Returns:
            SHA-256 hash for privacy compliance
        """
        # Add salt for additional security
        salt = b"aivillage_user_id_salt_2025"
        data = user_id.encode("utf-8") + salt

        hash_obj = hashlib.sha256(data)
        return hash_obj.hexdigest()

    def check_data_retention_compliance(self, created_at: datetime) -> dict[str, Any]:
        """Check if data meets retention policy requirements.

        Args:
            created_at: When the data was created

        Returns:
            Compliance status and actions needed
        """
        now = datetime.utcnow()
        age_days = (now - created_at).days

        status = {
            "age_days": age_days,
            "ttl_days": self.profile_ttl_days,
            "expires_at": created_at + timedelta(days=self.profile_ttl_days),
            "is_expired": age_days > self.profile_ttl_days,
            "days_until_expiry": self.profile_ttl_days - age_days,
            "requires_deletion": False,
            "compliance_actions": [],
        }

        # GDPR right to be forgotten
        if self.gdpr_compliant and age_days > self.profile_ttl_days:
            status["requires_deletion"] = True
            status["compliance_actions"].append("GDPR_DELETE_EXPIRED_DATA")

        # COPPA special handling for minors
        if self.coppa_compliant and age_days > 180:  # 6 months for COPPA
            status["compliance_actions"].append("COPPA_REVIEW_MINOR_DATA")

        # FERPA educational record retention
        if self.ferpa_compliant:
            status["compliance_actions"].append("FERPA_MAINTAIN_EDUCATIONAL_RECORD")

        return status

    def create_audit_log_entry(self, action: str, field_name: str, user_id_hash: str) -> dict[str, Any]:
        """Create audit log entry for compliance tracking.

        Args:
            action: Action performed (encrypt, decrypt, access, delete)
            field_name: Field name accessed
            user_id_hash: Hashed user identifier

        Returns:
            Audit log entry data
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "field_name": field_name,
            "user_id_hash": user_id_hash,
            "compliance_flags": {
                "coppa_enabled": self.coppa_compliant,
                "ferpa_enabled": self.ferpa_compliant,
                "gdpr_enabled": self.gdpr_compliant,
            },
            "node_id": os.getenv("LIBP2P_PEER_ID_FILE", "unknown"),
            "session_id": os.urandom(8).hex(),
        }

    def encrypt_profile_data(self, profile_data: dict[str, Any]) -> dict[str, Any]:
        """Encrypt entire learning profile with field-level encryption.

        Args:
            profile_data: Complete profile dictionary

        Returns:
            Profile with encrypted sensitive fields
        """
        # Define sensitive fields that require encryption
        sensitive_fields = {
            "learning_style",
            "knowledge_domains",
            "learning_goals",
            "performance_metrics",
            "engagement_score",
            "learning_trajectory",
        }

        encrypted_profile = profile_data.copy()

        # Encrypt sensitive fields individually
        for field_name in sensitive_fields:
            if field_name in profile_data:
                original_value = profile_data[field_name]
                encrypted_profile[f"{field_name}_encrypted"] = self.encrypt_sensitive_field(original_value, field_name)
                # Remove plaintext version
                del encrypted_profile[field_name]

        # Add compliance metadata
        encrypted_profile["encryption_metadata"] = {
            "encrypted_at": datetime.utcnow().isoformat(),
            "encryption_version": "1.0",
            "compliance_status": {
                "coppa_compliant": self.coppa_compliant,
                "ferpa_compliant": self.ferpa_compliant,
                "gdpr_compliant": self.gdpr_compliant,
            },
        }

        return encrypted_profile

    def decrypt_profile_data(self, encrypted_profile: dict[str, Any]) -> dict[str, Any]:
        """Decrypt learning profile with field-level decryption.

        Args:
            encrypted_profile: Profile with encrypted fields

        Returns:
            Profile with decrypted sensitive fields
        """
        decrypted_profile = encrypted_profile.copy()

        # Find and decrypt encrypted fields
        encrypted_field_names = [field for field in encrypted_profile if field.endswith("_encrypted")]

        for encrypted_field in encrypted_field_names:
            # Extract original field name
            original_field = encrypted_field.replace("_encrypted", "")

            # Decrypt the field
            encrypted_data = encrypted_profile[encrypted_field]
            decrypted_value = self.decrypt_sensitive_field(encrypted_data, original_field)

            # Add decrypted value
            decrypted_profile[original_field] = decrypted_value

            # Remove encrypted version
            del decrypted_profile[encrypted_field]

        return decrypted_profile


def generate_encryption_key() -> str:
    """Generate a new 32-byte base64-encoded encryption key.

    Returns:
        Base64-encoded encryption key suitable for environment variable
    """
    key_bytes = os.urandom(32)
    return base64.b64encode(key_bytes).decode("utf-8")


# Example usage and testing
if __name__ == "__main__":
    # Generate a new key for testing
    test_key = generate_encryption_key()
    print(f"Generated test key: {test_key}")

    # Set environment for testing
    os.environ["DIGITAL_TWIN_ENCRYPTION_KEY"] = test_key
    os.environ["DIGITAL_TWIN_COPPA_COMPLIANT"] = "true"
    os.environ["DIGITAL_TWIN_FERPA_COMPLIANT"] = "true"
    os.environ["DIGITAL_TWIN_GDPR_COMPLIANT"] = "true"

    # Test encryption system
    encryption = DigitalTwinEncryption()

    # Test field encryption
    sensitive_data = "learning_style: visual, preferred_subjects: [math, science]"
    encrypted = encryption.encrypt_sensitive_field(sensitive_data, "learning_preferences")
    decrypted = encryption.decrypt_sensitive_field(encrypted, "learning_preferences")

    print(f"Original: {sensitive_data}")
    print(f"Encrypted length: {len(encrypted)} bytes")
    print(f"Decrypted: {decrypted}")
    print(f"Encryption successful: {sensitive_data == decrypted}")
