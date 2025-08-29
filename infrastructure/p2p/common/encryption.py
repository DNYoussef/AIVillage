"""
Encryption utilities for P2P communications.

Provides standardized encryption/decryption capabilities with support
for multiple encryption algorithms and key management.
"""

import os
import hashlib
import secrets
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

try:
    import nacl.secret
    import nacl.public
    import nacl.utils
    from nacl.signing import SigningKey, VerifyKey
    HAS_NACL = True
except ImportError:
    HAS_NACL = False

logger = logging.getLogger(__name__)


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    NOISE_PROTOCOL = "noise_protocol"
    RSA_OAEP = "rsa_oaep"


@dataclass
class KeyPair:
    """Encryption key pair."""
    public_key: bytes
    private_key: bytes
    algorithm: EncryptionAlgorithm
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EncryptedData:
    """Encrypted data with metadata."""
    ciphertext: bytes
    algorithm: EncryptionAlgorithm
    nonce: Optional[bytes] = None
    tag: Optional[bytes] = None
    key_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EncryptionManager(ABC):
    """Abstract base class for encryption managers."""
    
    @property
    @abstractmethod
    def algorithm(self) -> EncryptionAlgorithm:
        """Get the encryption algorithm."""
        pass
    
    @abstractmethod
    def generate_key(self) -> bytes:
        """Generate a new encryption key."""
        pass
    
    @abstractmethod
    def encrypt(self, plaintext: bytes, key: bytes, **kwargs) -> EncryptedData:
        """Encrypt plaintext data."""
        pass
    
    @abstractmethod
    def decrypt(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt encrypted data."""
        pass


class AESEncryption(EncryptionManager):
    """AES-256-GCM encryption manager."""
    
    def __init__(self):
        if not HAS_CRYPTOGRAPHY:
            raise ImportError("cryptography library required for AES encryption")
    
    @property
    def algorithm(self) -> EncryptionAlgorithm:
        return EncryptionAlgorithm.AES_256_GCM
    
    def generate_key(self) -> bytes:
        """Generate a 256-bit AES key."""
        return os.urandom(32)  # 256 bits
    
    def encrypt(self, plaintext: bytes, key: bytes, **kwargs) -> EncryptedData:
        """Encrypt data using AES-256-GCM."""
        if len(key) != 32:
            raise ValueError("AES-256 requires a 32-byte key")
        
        # Generate random nonce
        nonce = os.urandom(12)  # 96 bits for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=self.algorithm,
            nonce=nonce,
            tag=encryptor.tag
        )
    
    def decrypt(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt AES-256-GCM encrypted data."""
        if encrypted_data.algorithm != self.algorithm:
            raise ValueError(f"Wrong algorithm: expected {self.algorithm.value}")
        
        if len(key) != 32:
            raise ValueError("AES-256 requires a 32-byte key")
        
        if encrypted_data.nonce is None or encrypted_data.tag is None:
            raise ValueError("Nonce and tag required for AES-GCM decryption")
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(encrypted_data.nonce, encrypted_data.tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt data
        return decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()


class ChaCha20Encryption(EncryptionManager):
    """ChaCha20-Poly1305 encryption manager."""
    
    def __init__(self):
        if not HAS_NACL:
            raise ImportError("PyNaCl library required for ChaCha20 encryption")
    
    @property
    def algorithm(self) -> EncryptionAlgorithm:
        return EncryptionAlgorithm.CHACHA20_POLY1305
    
    def generate_key(self) -> bytes:
        """Generate a ChaCha20 key."""
        return nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)
    
    def encrypt(self, plaintext: bytes, key: bytes, **kwargs) -> EncryptedData:
        """Encrypt data using ChaCha20-Poly1305."""
        box = nacl.secret.SecretBox(key)
        encrypted = box.encrypt(plaintext)
        
        # PyNaCl combines nonce + ciphertext + tag
        nonce = encrypted.nonce
        ciphertext = encrypted.ciphertext
        
        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=self.algorithm,
            nonce=nonce
        )
    
    def decrypt(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt ChaCha20-Poly1305 encrypted data."""
        if encrypted_data.algorithm != self.algorithm:
            raise ValueError(f"Wrong algorithm: expected {self.algorithm.value}")
        
        box = nacl.secret.SecretBox(key)
        
        # Reconstruct the encrypted message
        if encrypted_data.nonce:
            # Reconstruct the full encrypted message
            encrypted_message = encrypted_data.nonce + encrypted_data.ciphertext
        else:
            encrypted_message = encrypted_data.ciphertext
        
        return box.decrypt(encrypted_message)


class NoiseProtocolEncryption(EncryptionManager):
    """Noise Protocol encryption manager."""
    
    def __init__(self):
        # This would integrate with the existing noise protocol implementation
        self.noise_protocol = None
        logger.warning("Noise Protocol encryption is a placeholder")
    
    @property
    def algorithm(self) -> EncryptionAlgorithm:
        return EncryptionAlgorithm.NOISE_PROTOCOL
    
    def generate_key(self) -> bytes:
        """Generate a Noise Protocol key."""
        # This would use the actual noise protocol key generation
        return os.urandom(32)
    
    def encrypt(self, plaintext: bytes, key: bytes, **kwargs) -> EncryptedData:
        """Encrypt using Noise Protocol."""
        # This would integrate with the actual noise protocol
        # For now, fall back to AES for compatibility
        aes = AESEncryption() if HAS_CRYPTOGRAPHY else None
        if aes:
            return aes.encrypt(plaintext, key)
        
        raise NotImplementedError("Noise Protocol encryption not fully implemented")
    
    def decrypt(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """Decrypt using Noise Protocol."""
        # This would integrate with the actual noise protocol
        # For now, fall back to AES for compatibility
        aes = AESEncryption() if HAS_CRYPTOGRAPHY else None
        if aes:
            return aes.decrypt(encrypted_data, key)
        
        raise NotImplementedError("Noise Protocol decryption not fully implemented")


# Default encryption managers
_encryption_managers: Dict[EncryptionAlgorithm, EncryptionManager] = {}

if HAS_CRYPTOGRAPHY:
    _encryption_managers[EncryptionAlgorithm.AES_256_GCM] = AESEncryption()

if HAS_NACL:
    _encryption_managers[EncryptionAlgorithm.CHACHA20_POLY1305] = ChaCha20Encryption()

# Always register noise protocol (may fall back to AES)
_encryption_managers[EncryptionAlgorithm.NOISE_PROTOCOL] = NoiseProtocolEncryption()


def get_encryption_manager(algorithm: EncryptionAlgorithm) -> EncryptionManager:
    """Get encryption manager for specified algorithm."""
    if algorithm not in _encryption_managers:
        raise ValueError(f"Encryption manager not available for: {algorithm.value}")
    
    return _encryption_managers[algorithm]


def register_encryption_manager(algorithm: EncryptionAlgorithm, manager: EncryptionManager) -> None:
    """Register a custom encryption manager."""
    _encryption_managers[algorithm] = manager


def encrypt_data(plaintext: bytes, key: bytes, 
                algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM) -> EncryptedData:
    """Encrypt data using specified algorithm."""
    manager = get_encryption_manager(algorithm)
    return manager.encrypt(plaintext, key)


def decrypt_data(encrypted_data: EncryptedData, key: bytes) -> bytes:
    """Decrypt data using the algorithm specified in the encrypted data."""
    manager = get_encryption_manager(encrypted_data.algorithm)
    return manager.decrypt(encrypted_data, key)


def generate_keypair(algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM) -> KeyPair:
    """Generate a key pair for specified algorithm."""
    if algorithm == EncryptionAlgorithm.RSA_OAEP and HAS_CRYPTOGRAPHY:
        # Generate RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return KeyPair(
            private_key=private_pem,
            public_key=public_pem,
            algorithm=EncryptionAlgorithm.RSA_OAEP
        )
    
    elif algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.CHACHA20_POLY1305]:
        # Generate symmetric key
        manager = get_encryption_manager(algorithm)
        key = manager.generate_key()
        
        return KeyPair(
            private_key=key,
            public_key=key,  # Same key for symmetric encryption
            algorithm=algorithm
        )
    
    else:
        raise ValueError(f"Key pair generation not supported for: {algorithm.value}")


def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Derive encryption key from password using PBKDF2."""
    if not HAS_CRYPTOGRAPHY:
        raise ImportError("cryptography library required for key derivation")
    
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256 bits
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    
    key = kdf.derive(password.encode('utf-8'))
    return key, salt


def create_secure_hash(data: bytes, algorithm: str = "sha256") -> bytes:
    """Create a secure hash of data."""
    if algorithm == "sha256":
        return hashlib.sha256(data).digest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).digest()
    elif algorithm == "blake2b":
        return hashlib.blake2b(data).digest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def generate_secure_random(length: int) -> bytes:
    """Generate cryptographically secure random bytes."""
    return secrets.token_bytes(length)
