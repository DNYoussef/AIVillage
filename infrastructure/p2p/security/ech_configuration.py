"""
Encrypted Client Hello (ECH) Configuration Parser

Based on archaeological findings from codex/add-ech-config-parsing-and-validation.
Implements ECH configuration parsing, validation, and management for SNI protection
in P2P communications.

Archaeological Integration Status: ACTIVE  
Innovation Score: 8.3/10 (CRITICAL)
Implementation Date: 2025-08-29
"""

import hashlib
import json
import logging
import secrets
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ECHCipherSuite(Enum):
    """Supported ECH cipher suites from archaeological findings."""
    CHACHA20_POLY1305_SHA256 = "chacha20_poly1305_sha256"
    AES_256_GCM_SHA384 = "aes_256_gcm_sha384"
    AES_128_GCM_SHA256 = "aes_128_gcm_sha256"


class ECHConfigError(Exception):
    """ECH Configuration specific errors."""
    pass


@dataclass
class ECHConfig:
    """ECH Configuration based on archaeological ECH parser implementation.
    
    Provides configuration management for Encrypted Client Hello to prevent
    SNI leakage and enhance privacy in P2P communications.
    """
    
    public_key: bytes
    cipher_suites: List[ECHCipherSuite] = field(default_factory=lambda: [ECHCipherSuite.CHACHA20_POLY1305_SHA256])
    max_name_len: int = 64
    public_name: str = "example.com"
    extensions: Dict[str, bytes] = field(default_factory=dict)
    config_id: Optional[int] = None
    creation_time: Optional[int] = None
    
    def __post_init__(self):
        """Initialize computed fields after dataclass creation."""
        if self.config_id is None:
            self.config_id = self._generate_config_id()
        if self.creation_time is None:
            self.creation_time = int(time.time())
    
    def _generate_config_id(self) -> int:
        """Generate deterministic config ID from public key."""
        if not self.public_key:
            raise ECHConfigError("Cannot generate config ID without public key")
        
        hash_data = hashlib.sha256(self.public_key).digest()
        return struct.unpack('>I', hash_data[:4])[0] & 0xFFFF
    
    def validate(self) -> bool:
        """Validate ECH configuration parameters."""
        try:
            # Validate public key
            if not self.public_key or len(self.public_key) != 32:
                logger.error("Invalid public key length")
                return False
            
            # Validate cipher suites
            if not self.cipher_suites:
                logger.error("At least one cipher suite required")
                return False
            
            # Validate max name length
            if self.max_name_len < 1 or self.max_name_len > 255:
                logger.error("Invalid max_name_len: must be 1-255")
                return False
            
            # Validate public name
            if not self.public_name or len(self.public_name.encode('utf-8')) > self.max_name_len:
                logger.error("Public name too long for max_name_len")
                return False
            
            # Validate config ID range
            if self.config_id is not None and (self.config_id < 0 or self.config_id > 0xFFFF):
                logger.error("Config ID out of valid range")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"ECH config validation error: {e}")
            return False
    
    def serialize(self) -> bytes:
        """Serialize ECH config for wire transmission."""
        if not self.validate():
            raise ECHConfigError("Cannot serialize invalid ECH config")
        
        try:
            config_data = {
                'config_id': self.config_id,
                'public_key': self.public_key.hex(),
                'cipher_suites': [cs.value for cs in self.cipher_suites],
                'max_name_len': self.max_name_len,
                'public_name': self.public_name,
                'extensions': {k: v.hex() for k, v in self.extensions.items()},
                'creation_time': self.creation_time,
                'version': '1.0'
            }
            return json.dumps(config_data, separators=(',', ':')).encode('utf-8')
        except Exception as e:
            logger.error(f"ECH config serialization failed: {e}")
            raise ECHConfigError(f"Serialization error: {e}")
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'ECHConfig':
        """Deserialize ECH config from wire format."""
        try:
            if not data:
                raise ECHConfigError("Empty ECH config data")
            
            config_data = json.loads(data.decode('utf-8'))
            
            # Validate required fields
            required_fields = ['config_id', 'public_key', 'cipher_suites']
            for field in required_fields:
                if field not in config_data:
                    raise ECHConfigError(f"Missing required field: {field}")
            
            # Parse cipher suites
            cipher_suites = []
            for cs_value in config_data['cipher_suites']:
                try:
                    cipher_suites.append(ECHCipherSuite(cs_value))
                except ValueError:
                    logger.warning(f"Unknown cipher suite: {cs_value}, skipping")
            
            if not cipher_suites:
                raise ECHConfigError("No valid cipher suites found")
            
            # Create config instance
            config = cls(
                public_key=bytes.fromhex(config_data['public_key']),
                cipher_suites=cipher_suites,
                max_name_len=config_data.get('max_name_len', 64),
                public_name=config_data.get('public_name', 'example.com'),
                extensions={
                    k: bytes.fromhex(v) 
                    for k, v in config_data.get('extensions', {}).items()
                },
                config_id=config_data['config_id'],
                creation_time=config_data.get('creation_time')
            )
            
            if not config.validate():
                raise ECHConfigError("Deserialized config failed validation")
            
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"ECH config JSON decode error: {e}")
            raise ECHConfigError(f"Invalid JSON format: {e}")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"ECH config data error: {e}")
            raise ECHConfigError(f"Invalid config data: {e}")
        except Exception as e:
            logger.error(f"ECH config deserialization error: {e}")
            raise ECHConfigError(f"Deserialization failed: {e}")
    
    def is_expired(self, max_age_seconds: int = 86400) -> bool:
        """Check if ECH config is expired (default: 24 hours)."""
        if self.creation_time is None:
            return True
        
        return (int(time.time()) - self.creation_time) > max_age_seconds
    
    def get_preferred_cipher_suite(self) -> ECHCipherSuite:
        """Get the preferred cipher suite for this configuration."""
        # Prefer ChaCha20-Poly1305 for better performance on most platforms
        preferred_order = [
            ECHCipherSuite.CHACHA20_POLY1305_SHA256,
            ECHCipherSuite.AES_256_GCM_SHA384,
            ECHCipherSuite.AES_128_GCM_SHA256
        ]
        
        for preferred in preferred_order:
            if preferred in self.cipher_suites:
                return preferred
        
        # Fallback to first available
        return self.cipher_suites[0]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/debugging."""
        return {
            'config_id': self.config_id,
            'cipher_suites': [cs.value for cs in self.cipher_suites],
            'max_name_len': self.max_name_len,
            'public_name': self.public_name,
            'creation_time': self.creation_time,
            'is_valid': self.validate(),
            'is_expired': self.is_expired()
        }


class ECHConfigManager:
    """Manages ECH configurations with caching and validation."""
    
    def __init__(self):
        self._configs: Dict[int, ECHConfig] = {}
        self._default_config: Optional[ECHConfig] = None
    
    def add_config(self, config: ECHConfig) -> bool:
        """Add an ECH configuration to the manager."""
        if not config.validate():
            logger.error("Attempted to add invalid ECH config")
            return False
        
        if config.is_expired():
            logger.warning(f"Adding expired ECH config {config.config_id}")
        
        self._configs[config.config_id] = config
        
        # Set as default if it's the first config
        if self._default_config is None:
            self._default_config = config
            
        logger.info(f"Added ECH config {config.config_id}")
        return True
    
    def get_config(self, config_id: int) -> Optional[ECHConfig]:
        """Get ECH configuration by ID."""
        config = self._configs.get(config_id)
        
        if config and config.is_expired():
            logger.warning(f"Retrieved expired ECH config {config_id}")
            
        return config
    
    def get_default_config(self) -> Optional[ECHConfig]:
        """Get the default ECH configuration."""
        return self._default_config
    
    def remove_config(self, config_id: int) -> bool:
        """Remove ECH configuration by ID."""
        if config_id in self._configs:
            removed_config = self._configs.pop(config_id)
            
            # Update default if we removed it
            if self._default_config and self._default_config.config_id == config_id:
                self._default_config = next(iter(self._configs.values()), None)
            
            logger.info(f"Removed ECH config {config_id}")
            return True
        
        return False
    
    def cleanup_expired(self, max_age_seconds: int = 86400) -> int:
        """Remove expired configurations."""
        expired_ids = [
            config_id for config_id, config in self._configs.items()
            if config.is_expired(max_age_seconds)
        ]
        
        for config_id in expired_ids:
            self.remove_config(config_id)
        
        logger.info(f"Cleaned up {len(expired_ids)} expired ECH configs")
        return len(expired_ids)
    
    def get_config_count(self) -> int:
        """Get the number of stored configurations."""
        return len(self._configs)
    
    def list_configs(self) -> List[Dict]:
        """List all configurations as dictionaries."""
        return [config.to_dict() for config in self._configs.values()]


def create_ech_config(
    public_key: bytes,
    cipher_suites: Optional[List[ECHCipherSuite]] = None,
    public_name: str = "p2p.aivillage.internal"
) -> ECHConfig:
    """Factory function to create ECH configuration."""
    if cipher_suites is None:
        cipher_suites = [ECHCipherSuite.CHACHA20_POLY1305_SHA256]
    
    return ECHConfig(
        public_key=public_key,
        cipher_suites=cipher_suites,
        public_name=public_name
    )


def generate_ech_keypair() -> Tuple[bytes, bytes]:
    """Generate X25519 keypair for ECH configuration."""
    try:
        # Try to use proper cryptography
        from cryptography.hazmat.primitives.asymmetric import x25519
        
        private_key = x25519.X25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        return private_bytes, public_bytes
        
    except ImportError:
        # Fallback for development/testing
        logger.warning("Using fallback keypair generation - not for production")
        private_key = secrets.token_bytes(32)
        # Simple public key derivation (not cryptographically secure)
        public_key = hashlib.sha256(private_key).digest()[:32]
        return private_key, public_key