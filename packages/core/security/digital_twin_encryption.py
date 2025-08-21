"""Digital Twin Encryption for AIVillage SCION production systems."""

import base64
import os
from cryptography.fernet import Fernet
from typing import Any


class DigitalTwinEncryptionError(Exception):
    """Raised when digital twin encryption operations fail."""
    pass


class DigitalTwinEncryption:
    """Handles encryption/decryption for digital twin sensitive data."""
    
    def __init__(self):
        """Initialize encryption with key from environment or generate new one."""
        key_b64 = os.environ.get("DIGITAL_TWIN_ENCRYPTION_KEY")
        if not key_b64:
            raise DigitalTwinEncryptionError("DIGITAL_TWIN_ENCRYPTION_KEY environment variable required")
        
        try:
            # Fernet keys are already base64-encoded bytes, not needing additional decoding
            self.cipher = Fernet(key_b64.encode())
        except Exception as e:
            raise DigitalTwinEncryptionError(f"Invalid encryption key: {e}")
    
    def encrypt_sensitive_field(self, data: Any, field_name: str) -> bytes:
        """Encrypt sensitive field data."""
        try:
            data_str = str(data)
            return self.cipher.encrypt(data_str.encode())
        except Exception as e:
            raise DigitalTwinEncryptionError(f"Failed to encrypt {field_name}: {e}")
    
    def decrypt_sensitive_field(self, encrypted_data: bytes, field_name: str) -> str:
        """Decrypt sensitive field data."""
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            return decrypted.decode()
        except Exception as e:
            raise DigitalTwinEncryptionError(f"Failed to decrypt {field_name}: {e}")