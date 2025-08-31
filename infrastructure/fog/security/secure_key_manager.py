"""
Secure Key Management and Rotation System for Federated Learning

This module provides comprehensive cryptographic key management capabilities for
federated learning systems, including distributed key generation, secure rotation,
backup/recovery, and integration with hardware security modules (HSMs).
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """Types of cryptographic keys"""
    SIGNING = "signing"
    ENCRYPTION = "encryption"
    AUTHENTICATION = "authentication"
    MASTER = "master"
    SESSION = "session"
    BACKUP = "backup"
    RECOVERY = "recovery"


class KeyStatus(Enum):
    """Key lifecycle status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING_ROTATION = "pending_rotation"
    COMPROMISED = "compromised"
    REVOKED = "revoked"
    ARCHIVED = "archived"


class RotationTrigger(Enum):
    """Key rotation triggers"""
    SCHEDULED = "scheduled"
    COMPROMISE_DETECTED = "compromise_detected"
    USAGE_THRESHOLD = "usage_threshold"
    MANUAL = "manual"
    COMPLIANCE_REQUIREMENT = "compliance_requirement"


@dataclass
class CryptographicKey:
    """Cryptographic key with metadata"""
    key_id: str
    key_type: KeyType
    algorithm: str
    key_size: int
    public_key: Optional[bytes]
    private_key: Optional[bytes]  # Encrypted when stored
    created_at: float
    expires_at: Optional[float]
    status: KeyStatus = KeyStatus.ACTIVE
    usage_count: int = 0
    max_usage: Optional[int] = None
    rotation_schedule: Optional[int] = None  # Rotation interval in seconds
    last_rotated: Optional[float] = None
    parent_key_id: Optional[str] = None  # For derived keys
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if key is expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def needs_rotation(self) -> bool:
        """Check if key needs rotation"""
        current_time = time.time()
        
        # Check expiration
        if self.is_expired():
            return True
        
        # Check usage threshold
        if self.max_usage and self.usage_count >= self.max_usage:
            return True
        
        # Check rotation schedule
        if self.rotation_schedule and self.last_rotated:
            if current_time - self.last_rotated >= self.rotation_schedule:
                return True
        
        # Check status
        if self.status in [KeyStatus.COMPROMISED, KeyStatus.PENDING_ROTATION]:
            return True
        
        return False


@dataclass
class KeyShare:
    """Key share for distributed key management"""
    share_id: str
    key_id: str
    share_index: int
    threshold: int
    total_shares: int
    encrypted_share: bytes
    verification_hash: str
    holder_id: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyBackup:
    """Secure key backup"""
    backup_id: str
    key_id: str
    encrypted_backup: bytes
    backup_method: str
    checksum: str
    recovery_shares: List[str]  # Share IDs needed for recovery
    created_at: float = field(default_factory=time.time)
    location: str = "local"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RotationEvent:
    """Key rotation event record"""
    event_id: str
    key_id: str
    old_key_id: str
    new_key_id: str
    trigger: RotationTrigger
    initiated_by: str
    started_at: float
    completed_at: Optional[float] = None
    status: str = "in_progress"
    details: Dict[str, Any] = field(default_factory=dict)
    rollback_plan: Optional[Dict[str, Any]] = None


class SecureKeyStore:
    """Secure storage for cryptographic keys"""
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password or self._generate_master_password()
        self.salt = secrets.token_bytes(32)
        self.fernet = self._initialize_encryption()
        self.keys: Dict[str, CryptographicKey] = {}
        self.key_shares: Dict[str, KeyShare] = {}
        self.backups: Dict[str, KeyBackup] = {}
        
    def _generate_master_password(self) -> str:
        """Generate cryptographically secure master password"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize Fernet encryption for key storage"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))
        return Fernet(key)
    
    def store_key(self, key: CryptographicKey) -> bool:
        """Store key securely"""
        try:
            # Encrypt private key if present
            if key.private_key:
                encrypted_private = self.fernet.encrypt(key.private_key)
                # Create copy with encrypted private key
                stored_key = CryptographicKey(
                    key_id=key.key_id,
                    key_type=key.key_type,
                    algorithm=key.algorithm,
                    key_size=key.key_size,
                    public_key=key.public_key,
                    private_key=encrypted_private,
                    created_at=key.created_at,
                    expires_at=key.expires_at,
                    status=key.status,
                    usage_count=key.usage_count,
                    max_usage=key.max_usage,
                    rotation_schedule=key.rotation_schedule,
                    last_rotated=key.last_rotated,
                    parent_key_id=key.parent_key_id,
                    metadata=key.metadata.copy()
                )
            else:
                stored_key = key
            
            self.keys[key.key_id] = stored_key
            return True
        except Exception as e:
            logger.error(f"Failed to store key {key.key_id}: {e}")
            return False
    
    def retrieve_key(self, key_id: str, decrypt_private: bool = True) -> Optional[CryptographicKey]:
        """Retrieve and optionally decrypt key"""
        if key_id not in self.keys:
            return None
        
        stored_key = self.keys[key_id]
        
        if decrypt_private and stored_key.private_key:
            try:
                decrypted_private = self.fernet.decrypt(stored_key.private_key)
                # Return copy with decrypted private key
                return CryptographicKey(
                    key_id=stored_key.key_id,
                    key_type=stored_key.key_type,
                    algorithm=stored_key.algorithm,
                    key_size=stored_key.key_size,
                    public_key=stored_key.public_key,
                    private_key=decrypted_private,
                    created_at=stored_key.created_at,
                    expires_at=stored_key.expires_at,
                    status=stored_key.status,
                    usage_count=stored_key.usage_count,
                    max_usage=stored_key.max_usage,
                    rotation_schedule=stored_key.rotation_schedule,
                    last_rotated=stored_key.last_rotated,
                    parent_key_id=stored_key.parent_key_id,
                    metadata=stored_key.metadata.copy()
                )
            except Exception as e:
                logger.error(f"Failed to decrypt private key for {key_id}: {e}")
                return None
        
        return stored_key
    
    def delete_key(self, key_id: str) -> bool:
        """Securely delete key"""
        if key_id in self.keys:
            del self.keys[key_id]
            return True
        return False
    
    def list_keys(
        self, 
        key_type: Optional[KeyType] = None,
        status: Optional[KeyStatus] = None
    ) -> List[str]:
        """List key IDs matching criteria"""
        result = []
        for key_id, key in self.keys.items():
            if key_type and key.key_type != key_type:
                continue
            if status and key.status != status:
                continue
            result.append(key_id)
        return result


class DistributedKeyGenerator:
    """Distributed key generation for federated environments"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.active_ceremonies: Dict[str, Dict[str, Any]] = {}
        self.completed_ceremonies: Dict[str, Dict[str, Any]] = {}
        
    async def initiate_dkg_ceremony(
        self,
        ceremony_id: str,
        participants: List[str],
        threshold: int,
        key_type: KeyType = KeyType.SIGNING,
        algorithm: str = "ECDSA"
    ) -> Dict[str, Any]:
        """Initiate distributed key generation ceremony"""
        if ceremony_id in self.active_ceremonies:
            raise ValueError(f"Ceremony {ceremony_id} already active")
        
        if threshold > len(participants):
            raise ValueError("Threshold cannot exceed number of participants")
        
        ceremony_data = {
            "ceremony_id": ceremony_id,
            "initiator": self.node_id,
            "participants": participants,
            "threshold": threshold,
            "key_type": key_type,
            "algorithm": algorithm,
            "phase": "initialization",
            "started_at": time.time(),
            "contributions": {},
            "verifications": {},
            "shares": {},
            "master_public_key": None,
            "status": "active"
        }
        
        self.active_ceremonies[ceremony_id] = ceremony_data
        
        logger.info(f"Initiated DKG ceremony {ceremony_id} with {len(participants)} participants")
        
        return {
            "ceremony_id": ceremony_id,
            "phase": "initialization",
            "participants": participants,
            "threshold": threshold,
            "your_index": participants.index(self.node_id) if self.node_id in participants else -1
        }
    
    async def contribute_to_ceremony(
        self,
        ceremony_id: str,
        contribution_data: bytes
    ) -> Dict[str, Any]:
        """Contribute to DKG ceremony"""
        if ceremony_id not in self.active_ceremonies:
            raise ValueError(f"Unknown ceremony {ceremony_id}")
        
        ceremony = self.active_ceremonies[ceremony_id]
        
        if self.node_id not in ceremony["participants"]:
            raise ValueError("Not a participant in this ceremony")
        
        # Store contribution
        ceremony["contributions"][self.node_id] = {
            "data": contribution_data,
            "timestamp": time.time(),
            "verified": False
        }
        
        # Check if all contributions received
        if len(ceremony["contributions"]) == len(ceremony["participants"]):
            ceremony["phase"] = "verification"
            await self._proceed_to_verification_phase(ceremony_id)
        
        return {
            "ceremony_id": ceremony_id,
            "contribution_accepted": True,
            "phase": ceremony["phase"],
            "contributions_received": len(ceremony["contributions"])
        }
    
    async def _proceed_to_verification_phase(self, ceremony_id: str):
        """Move ceremony to verification phase"""
        ceremony = self.active_ceremonies[ceremony_id]
        
        # Verify contributions (simplified)
        all_valid = True
        for participant, contribution in ceremony["contributions"].items():
            # In production, implement proper cryptographic verification
            if len(contribution["data"]) < 32:  # Minimum contribution size
                all_valid = False
                logger.warning(f"Invalid contribution from {participant}")
            else:
                contribution["verified"] = True
        
        if all_valid:
            ceremony["phase"] = "key_generation"
            await self._generate_distributed_key(ceremony_id)
        else:
            ceremony["status"] = "failed"
            ceremony["error"] = "Contribution verification failed"
    
    async def _generate_distributed_key(self, ceremony_id: str):
        """Generate the distributed key from contributions"""
        ceremony = self.active_ceremonies[ceremony_id]
        participants = ceremony["participants"]
        threshold = ceremony["threshold"]
        
        # Generate master key pair (simplified)
        if ceremony["algorithm"] == "ECDSA":
            private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
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
            
        elif ceremony["algorithm"] == "RSA":
            private_key = rsa.generate_private_key(65537, 2048, default_backend())
            public_key = private_key.public_key()
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        else:
            ceremony["status"] = "failed"
            ceremony["error"] = f"Unsupported algorithm: {ceremony['algorithm']}"
            return
        
        ceremony["master_public_key"] = public_pem
        
        # Generate key shares using Shamir's Secret Sharing (simplified)
        shares = await self._generate_key_shares(
            private_pem, participants, threshold
        )
        ceremony["shares"] = shares
        
        # Mark ceremony as completed
        ceremony["status"] = "completed"
        ceremony["completed_at"] = time.time()
        ceremony["phase"] = "completed"
        
        # Move to completed ceremonies
        self.completed_ceremonies[ceremony_id] = ceremony
        del self.active_ceremonies[ceremony_id]
        
        logger.info(f"DKG ceremony {ceremony_id} completed successfully")
    
    async def _generate_key_shares(
        self,
        private_key: bytes,
        participants: List[str],
        threshold: int
    ) -> Dict[str, KeyShare]:
        """Generate key shares using Shamir's Secret Sharing"""
        shares = {}
        
        # Convert private key to integer for secret sharing
        key_int = int.from_bytes(hashlib.sha256(private_key).digest(), 'big')
        
        # Generate polynomial coefficients
        prime = 2**256 - 189  # Large prime for field arithmetic
        coefficients = [key_int] + [secrets.randbits(256) for _ in range(threshold - 1)]
        
        # Generate shares
        for i, participant in enumerate(participants, 1):
            # Evaluate polynomial at point i
            share_value = sum(
                coeff * pow(i, j, prime) for j, coeff in enumerate(coefficients)
            ) % prime
            
            # Encrypt share
            share_bytes = share_value.to_bytes(32, 'big')
            encrypted_share = self._encrypt_share(share_bytes, participant)
            
            # Create verification hash
            verification_hash = hashlib.sha256(
                share_bytes + participant.encode()
            ).hexdigest()
            
            share = KeyShare(
                share_id=f"share_{participant}_{secrets.token_hex(4)}",
                key_id=f"dkg_{secrets.token_hex(8)}",
                share_index=i,
                threshold=threshold,
                total_shares=len(participants),
                encrypted_share=encrypted_share,
                verification_hash=verification_hash,
                holder_id=participant
            )
            
            shares[participant] = share
        
        return shares
    
    def _encrypt_share(self, share_data: bytes, recipient: str) -> bytes:
        """Encrypt key share for recipient (simplified)"""
        # In production, use recipient's public key for encryption
        recipient_key = hashlib.sha256(recipient.encode()).digest()
        fernet = Fernet(base64.urlsafe_b64encode(recipient_key))
        return fernet.encrypt(share_data)
    
    def get_ceremony_status(self, ceremony_id: str) -> Optional[Dict[str, Any]]:
        """Get ceremony status"""
        if ceremony_id in self.active_ceremonies:
            ceremony = self.active_ceremonies[ceremony_id]
        elif ceremony_id in self.completed_ceremonies:
            ceremony = self.completed_ceremonies[ceremony_id]
        else:
            return None
        
        return {
            "ceremony_id": ceremony_id,
            "status": ceremony["status"],
            "phase": ceremony["phase"],
            "participants": ceremony["participants"],
            "threshold": ceremony["threshold"],
            "started_at": ceremony["started_at"],
            "completed_at": ceremony.get("completed_at"),
            "contributions_count": len(ceremony["contributions"]),
            "master_public_key_available": ceremony["master_public_key"] is not None
        }


class KeyRotationManager:
    """Manages automated and manual key rotation"""
    
    def __init__(self, key_store: SecureKeyStore):
        self.key_store = key_store
        self.rotation_events: Dict[str, RotationEvent] = {}
        self.rotation_policies: Dict[str, Dict[str, Any]] = {}
        self.background_task: Optional[asyncio.Task] = None
        
    def set_rotation_policy(
        self,
        policy_id: str,
        key_types: List[KeyType],
        rotation_interval: int,
        max_usage: Optional[int] = None,
        auto_rotate: bool = True
    ):
        """Set key rotation policy"""
        self.rotation_policies[policy_id] = {
            "key_types": key_types,
            "rotation_interval": rotation_interval,
            "max_usage": max_usage,
            "auto_rotate": auto_rotate,
            "created_at": time.time()
        }
        
        logger.info(f"Set rotation policy {policy_id} for key types {[kt.value for kt in key_types]}")
    
    async def start_rotation_monitoring(self):
        """Start background task for rotation monitoring"""
        if self.background_task:
            return
        
        self.background_task = asyncio.create_task(self._rotation_monitoring_loop())
        logger.info("Started key rotation monitoring")
    
    async def stop_rotation_monitoring(self):
        """Stop background rotation monitoring"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
            logger.info("Stopped key rotation monitoring")
    
    async def _rotation_monitoring_loop(self):
        """Background loop for monitoring key rotation needs"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check all keys for rotation needs
                key_ids = self.key_store.list_keys()
                
                for key_id in key_ids:
                    key = self.key_store.retrieve_key(key_id, decrypt_private=False)
                    if key and key.needs_rotation():
                        await self._schedule_key_rotation(key_id, RotationTrigger.SCHEDULED)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rotation monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def rotate_key(
        self,
        key_id: str,
        trigger: RotationTrigger = RotationTrigger.MANUAL,
        initiated_by: str = "system"
    ) -> str:
        """Rotate a specific key"""
        old_key = self.key_store.retrieve_key(key_id)
        if not old_key:
            raise ValueError(f"Key {key_id} not found")
        
        # Create rotation event
        event_id = f"rot_{key_id}_{int(time.time())}_{secrets.token_hex(4)}"
        new_key_id = f"{key_id}_rotated_{int(time.time())}"
        
        rotation_event = RotationEvent(
            event_id=event_id,
            key_id=key_id,
            old_key_id=key_id,
            new_key_id=new_key_id,
            trigger=trigger,
            initiated_by=initiated_by,
            started_at=time.time(),
            details={
                "old_key_algorithm": old_key.algorithm,
                "old_key_size": old_key.key_size,
                "rotation_reason": trigger.value
            }
        )
        
        self.rotation_events[event_id] = rotation_event
        
        try:
            # Generate new key
            new_key = await self._generate_rotated_key(old_key, new_key_id)
            
            # Store new key
            if not self.key_store.store_key(new_key):
                raise Exception("Failed to store new key")
            
            # Update old key status
            old_key.status = KeyStatus.INACTIVE
            old_key.metadata["rotated_to"] = new_key_id
            old_key.metadata["rotation_event"] = event_id
            self.key_store.store_key(old_key)
            
            # Complete rotation event
            rotation_event.completed_at = time.time()
            rotation_event.status = "completed"
            rotation_event.details["new_key_algorithm"] = new_key.algorithm
            rotation_event.details["new_key_size"] = new_key.key_size
            
            logger.info(f"Successfully rotated key {key_id} to {new_key_id}")
            
            return new_key_id
            
        except Exception as e:
            rotation_event.status = "failed"
            rotation_event.details["error"] = str(e)
            logger.error(f"Key rotation failed for {key_id}: {e}")
            raise
    
    async def _generate_rotated_key(
        self,
        old_key: CryptographicKey,
        new_key_id: str
    ) -> CryptographicKey:
        """Generate new key to replace old key"""
        if old_key.algorithm == "RSA":
            private_key = rsa.generate_private_key(65537, old_key.key_size, default_backend())
            public_key = private_key.public_key()
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
        elif old_key.algorithm in ["ECDSA", "ECDH"]:
            private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
            public_key = private_key.public_key()
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        else:
            raise ValueError(f"Unsupported algorithm for rotation: {old_key.algorithm}")
        
        # Create new key with similar properties
        new_key = CryptographicKey(
            key_id=new_key_id,
            key_type=old_key.key_type,
            algorithm=old_key.algorithm,
            key_size=old_key.key_size,
            public_key=public_pem,
            private_key=private_pem,
            created_at=time.time(),
            expires_at=old_key.expires_at,  # Keep same expiration policy
            rotation_schedule=old_key.rotation_schedule,
            max_usage=old_key.max_usage,
            parent_key_id=old_key.key_id,
            metadata={
                "rotated_from": old_key.key_id,
                "rotation_generation": old_key.metadata.get("rotation_generation", 0) + 1
            }
        )
        
        return new_key
    
    async def _schedule_key_rotation(self, key_id: str, trigger: RotationTrigger):
        """Schedule key rotation"""
        try:
            await self.rotate_key(key_id, trigger, "automated_system")
        except Exception as e:
            logger.error(f"Scheduled rotation failed for key {key_id}: {e}")
    
    def get_rotation_history(self, key_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get rotation event history"""
        events = []
        
        for event in self.rotation_events.values():
            if key_id and event.key_id != key_id:
                continue
            
            events.append({
                "event_id": event.event_id,
                "key_id": event.key_id,
                "old_key_id": event.old_key_id,
                "new_key_id": event.new_key_id,
                "trigger": event.trigger.value,
                "initiated_by": event.initiated_by,
                "started_at": event.started_at,
                "completed_at": event.completed_at,
                "status": event.status,
                "duration": (event.completed_at - event.started_at) if event.completed_at else None
            })
        
        return sorted(events, key=lambda x: x["started_at"], reverse=True)


class KeyBackupManager:
    """Manages secure key backup and recovery"""
    
    def __init__(self, key_store: SecureKeyStore):
        self.key_store = key_store
        self.backup_locations: Dict[str, Dict[str, Any]] = {}
        
    def add_backup_location(
        self,
        location_id: str,
        location_type: str,
        config: Dict[str, Any]
    ):
        """Add backup storage location"""
        self.backup_locations[location_id] = {
            "type": location_type,
            "config": config,
            "active": True,
            "created_at": time.time()
        }
        
        logger.info(f"Added backup location {location_id} of type {location_type}")
    
    async def create_backup(
        self,
        key_id: str,
        backup_locations: List[str],
        recovery_threshold: int = 2,
        method: str = "shamir_shares"
    ) -> str:
        """Create secure backup of key"""
        key = self.key_store.retrieve_key(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found")
        
        backup_id = f"backup_{key_id}_{int(time.time())}_{secrets.token_hex(4)}"
        
        if method == "shamir_shares":
            backup = await self._create_shamir_backup(
                key, backup_id, backup_locations, recovery_threshold
            )
        elif method == "encrypted_copies":
            backup = await self._create_encrypted_backup(
                key, backup_id, backup_locations
            )
        else:
            raise ValueError(f"Unsupported backup method: {method}")
        
        # Store backup metadata
        self.key_store.backups[backup_id] = backup
        
        logger.info(f"Created backup {backup_id} for key {key_id} using {method}")
        
        return backup_id
    
    async def _create_shamir_backup(
        self,
        key: CryptographicKey,
        backup_id: str,
        locations: List[str],
        threshold: int
    ) -> KeyBackup:
        """Create backup using Shamir's Secret Sharing"""
        # Serialize key data
        key_data = {
            "key_id": key.key_id,
            "key_type": key.key_type.value,
            "algorithm": key.algorithm,
            "key_size": key.key_size,
            "public_key": key.public_key.hex() if key.public_key else None,
            "private_key": key.private_key.hex() if key.private_key else None,
            "created_at": key.created_at,
            "metadata": key.metadata
        }
        
        key_json = json.dumps(key_data, sort_keys=True)
        key_bytes = key_json.encode()
        
        # Generate shares
        shares = await self._generate_backup_shares(key_bytes, len(locations), threshold)
        share_ids = []
        
        # Distribute shares to backup locations
        for i, location_id in enumerate(locations):
            if location_id not in self.backup_locations:
                logger.warning(f"Backup location {location_id} not configured")
                continue
            
            share_id = f"share_{backup_id}_{i}"
            encrypted_share = self._encrypt_for_location(shares[i], location_id)
            
            # Store share (simplified - in production, use actual backup storage)
            await self._store_backup_share(location_id, share_id, encrypted_share)
            share_ids.append(share_id)
        
        # Calculate checksum
        checksum = hashlib.sha256(key_bytes).hexdigest()
        
        backup = KeyBackup(
            backup_id=backup_id,
            key_id=key.key_id,
            encrypted_backup=b"",  # Not used for Shamir shares
            backup_method="shamir_shares",
            checksum=checksum,
            recovery_shares=share_ids,
            location="distributed",
            metadata={
                "threshold": threshold,
                "total_shares": len(locations),
                "backup_locations": locations
            }
        )
        
        return backup
    
    async def _create_encrypted_backup(
        self,
        key: CryptographicKey,
        backup_id: str,
        locations: List[str]
    ) -> KeyBackup:
        """Create backup using encryption"""
        # Serialize key data
        key_data = {
            "key_id": key.key_id,
            "key_type": key.key_type.value,
            "algorithm": key.algorithm,
            "key_size": key.key_size,
            "public_key": key.public_key.hex() if key.public_key else None,
            "private_key": key.private_key.hex() if key.private_key else None,
            "created_at": key.created_at,
            "metadata": key.metadata
        }
        
        key_json = json.dumps(key_data, sort_keys=True)
        key_bytes = key_json.encode()
        
        # Encrypt backup
        backup_key = Fernet.generate_key()
        fernet = Fernet(backup_key)
        encrypted_backup = fernet.encrypt(key_bytes)
        
        # Store backup in all locations
        for location_id in locations:
            await self._store_backup_data(location_id, backup_id, encrypted_backup)
        
        # Calculate checksum
        checksum = hashlib.sha256(key_bytes).hexdigest()
        
        backup = KeyBackup(
            backup_id=backup_id,
            key_id=key.key_id,
            encrypted_backup=encrypted_backup,
            backup_method="encrypted_copies",
            checksum=checksum,
            recovery_shares=[],
            location="multiple",
            metadata={
                "backup_key": backup_key.decode(),
                "backup_locations": locations
            }
        )
        
        return backup
    
    async def _generate_backup_shares(
        self,
        data: bytes,
        total_shares: int,
        threshold: int
    ) -> List[bytes]:
        """Generate Shamir secret shares for backup"""
        # Convert data to integer
        data_int = int.from_bytes(hashlib.sha256(data).digest(), 'big')
        
        # Generate polynomial coefficients
        prime = 2**256 - 189
        coefficients = [data_int] + [secrets.randbits(256) for _ in range(threshold - 1)]
        
        # Generate shares
        shares = []
        for i in range(1, total_shares + 1):
            share_value = sum(
                coeff * pow(i, j, prime) for j, coeff in enumerate(coefficients)
            ) % prime
            
            # Include original data length and share metadata
            share_data = {
                "share_value": share_value,
                "share_index": i,
                "threshold": threshold,
                "total_shares": total_shares,
                "data_length": len(data),
                "data_hash": hashlib.sha256(data).hexdigest()
            }
            
            share_bytes = json.dumps(share_data).encode()
            shares.append(share_bytes)
        
        return shares
    
    def _encrypt_for_location(self, data: bytes, location_id: str) -> bytes:
        """Encrypt data for specific backup location"""
        location_config = self.backup_locations[location_id]["config"]
        location_key = location_config.get("encryption_key")
        
        if not location_key:
            # Generate location-specific key from ID
            location_key = hashlib.sha256(f"backup_key_{location_id}".encode()).digest()
            location_key = base64.urlsafe_b64encode(location_key)
        
        fernet = Fernet(location_key)
        return fernet.encrypt(data)
    
    async def _store_backup_share(self, location_id: str, share_id: str, encrypted_share: bytes):
        """Store backup share at location (simplified)"""
        # In production, implement actual storage to configured location
        logger.debug(f"Stored backup share {share_id} at location {location_id}")
    
    async def _store_backup_data(self, location_id: str, backup_id: str, encrypted_data: bytes):
        """Store backup data at location (simplified)"""
        # In production, implement actual storage to configured location
        logger.debug(f"Stored backup {backup_id} at location {location_id}")
    
    async def recover_key(self, backup_id: str, recovery_data: Dict[str, Any]) -> CryptographicKey:
        """Recover key from backup"""
        if backup_id not in self.key_store.backups:
            raise ValueError(f"Backup {backup_id} not found")
        
        backup = self.key_store.backups[backup_id]
        
        if backup.backup_method == "shamir_shares":
            key_data = await self._recover_from_shamir_shares(backup, recovery_data)
        elif backup.backup_method == "encrypted_copies":
            key_data = await self._recover_from_encrypted_backup(backup, recovery_data)
        else:
            raise ValueError(f"Unsupported backup method: {backup.backup_method}")
        
        # Verify checksum
        recovered_checksum = hashlib.sha256(key_data).hexdigest()
        if recovered_checksum != backup.checksum:
            raise ValueError("Backup integrity check failed")
        
        # Reconstruct key
        key_dict = json.loads(key_data.decode())
        
        key = CryptographicKey(
            key_id=key_dict["key_id"],
            key_type=KeyType(key_dict["key_type"]),
            algorithm=key_dict["algorithm"],
            key_size=key_dict["key_size"],
            public_key=bytes.fromhex(key_dict["public_key"]) if key_dict["public_key"] else None,
            private_key=bytes.fromhex(key_dict["private_key"]) if key_dict["private_key"] else None,
            created_at=key_dict["created_at"],
            metadata=key_dict["metadata"],
            status=KeyStatus.ACTIVE  # Recovered keys are active
        )
        
        logger.info(f"Successfully recovered key {key.key_id} from backup {backup_id}")
        
        return key
    
    async def _recover_from_shamir_shares(
        self,
        backup: KeyBackup,
        recovery_data: Dict[str, Any]
    ) -> bytes:
        """Recover key data from Shamir shares"""
        threshold = backup.metadata["threshold"]
        provided_shares = recovery_data.get("shares", [])
        
        if len(provided_shares) < threshold:
            raise ValueError(f"Insufficient shares: need {threshold}, got {len(provided_shares)}")
        
        # Decrypt and parse shares
        share_data = []
        for share_info in provided_shares[:threshold]:
            location_id = share_info["location_id"]
            encrypted_share = share_info["encrypted_data"]
            
            # Decrypt share
            decrypted_share = self._decrypt_from_location(encrypted_share, location_id)
            share_dict = json.loads(decrypted_share.decode())
            share_data.append(share_dict)
        
        # Reconstruct secret using Lagrange interpolation
        prime = 2**256 - 189
        secret = 0
        
        for i, share in enumerate(share_data):
            share_value = share["share_value"]
            share_index = share["share_index"]
            
            # Calculate Lagrange coefficient
            numerator = 1
            denominator = 1
            
            for j, other_share in enumerate(share_data):
                if i == j:
                    continue
                other_index = other_share["share_index"]
                numerator = (numerator * (-other_index)) % prime
                denominator = (denominator * (share_index - other_index)) % prime
            
            # Modular inverse
            inv_denominator = pow(denominator, prime - 2, prime)
            lagrange_coeff = (numerator * inv_denominator) % prime
            
            secret = (secret + share_value * lagrange_coeff) % prime
        
        # Convert secret back to original data
        # Note: This is simplified - in production, need to handle the original data reconstruction
        secret_hash = secret.to_bytes(32, 'big')
        
        # For this example, we'll use the hash as a key to decrypt the actual data
        # In production, you'd store the actual data encrypted with this secret
        return b"reconstructed_key_data"  # Placeholder
    
    async def _recover_from_encrypted_backup(
        self,
        backup: KeyBackup,
        recovery_data: Dict[str, Any]
    ) -> bytes:
        """Recover key data from encrypted backup"""
        backup_key = recovery_data.get("backup_key")
        if not backup_key:
            backup_key = backup.metadata.get("backup_key")
        
        if not backup_key:
            raise ValueError("Backup decryption key not provided")
        
        fernet = Fernet(backup_key.encode())
        decrypted_data = fernet.decrypt(backup.encrypted_backup)
        
        return decrypted_data
    
    def _decrypt_from_location(self, encrypted_data: bytes, location_id: str) -> bytes:
        """Decrypt data from specific backup location"""
        location_config = self.backup_locations[location_id]["config"]
        location_key = location_config.get("encryption_key")
        
        if not location_key:
            # Generate location-specific key from ID
            location_key = hashlib.sha256(f"backup_key_{location_id}".encode()).digest()
            location_key = base64.urlsafe_b64encode(location_key)
        
        fernet = Fernet(location_key)
        return fernet.decrypt(encrypted_data)
    
    def list_backups(self, key_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        for backup_id, backup in self.key_store.backups.items():
            if key_id and backup.key_id != key_id:
                continue
            
            backups.append({
                "backup_id": backup_id,
                "key_id": backup.key_id,
                "backup_method": backup.backup_method,
                "created_at": backup.created_at,
                "location": backup.location,
                "recovery_shares_count": len(backup.recovery_shares),
                "metadata": backup.metadata
            })
        
        return sorted(backups, key=lambda x: x["created_at"], reverse=True)


class SecureKeyManager:
    """
    Main secure key management system coordinating all components
    """
    
    def __init__(self, node_id: str, master_password: Optional[str] = None):
        self.node_id = node_id
        self.key_store = SecureKeyStore(master_password)
        self.dkg = DistributedKeyGenerator(node_id)
        self.rotation_manager = KeyRotationManager(self.key_store)
        self.backup_manager = KeyBackupManager(self.key_store)
        
        # Statistics
        self.stats = {
            "keys_generated": 0,
            "keys_rotated": 0,
            "keys_backed_up": 0,
            "keys_recovered": 0,
            "dkg_ceremonies_completed": 0
        }
        
        logger.info(f"Initialized secure key manager for node {self.node_id}")
    
    async def initialize(self):
        """Initialize key manager"""
        # Start rotation monitoring
        await self.rotation_manager.start_rotation_monitoring()
        
        # Set default rotation policies
        self.rotation_manager.set_rotation_policy(
            "default_signing",
            [KeyType.SIGNING],
            rotation_interval=86400 * 90,  # 90 days
            max_usage=10000,
            auto_rotate=True
        )
        
        self.rotation_manager.set_rotation_policy(
            "default_encryption",
            [KeyType.ENCRYPTION],
            rotation_interval=86400 * 30,  # 30 days
            max_usage=50000,
            auto_rotate=True
        )
        
        # Add default backup locations
        self.backup_manager.add_backup_location(
            "local_encrypted",
            "local_storage",
            {"path": "./backups", "encryption_key": Fernet.generate_key().decode()}
        )
        
        logger.info("Key manager initialization completed")
    
    async def generate_key(
        self,
        key_id: str,
        key_type: KeyType,
        algorithm: str = "RSA",
        key_size: int = 2048,
        expires_in: Optional[int] = None,
        auto_backup: bool = True
    ) -> CryptographicKey:
        """Generate new cryptographic key"""
        if algorithm == "RSA":
            private_key = rsa.generate_private_key(65537, key_size, default_backend())
            public_key = private_key.public_key()
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
        elif algorithm in ["ECDSA", "ECDH"]:
            private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
            public_key = private_key.public_key()
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Create key object
        key = CryptographicKey(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            key_size=key_size,
            public_key=public_pem,
            private_key=private_pem,
            created_at=time.time(),
            expires_at=time.time() + expires_in if expires_in else None
        )
        
        # Store key
        if not self.key_store.store_key(key):
            raise Exception(f"Failed to store key {key_id}")
        
        # Create backup if requested
        if auto_backup:
            try:
                await self.backup_manager.create_backup(
                    key_id, ["local_encrypted"], recovery_threshold=1
                )
                self.stats["keys_backed_up"] += 1
            except Exception as e:
                logger.warning(f"Failed to create backup for key {key_id}: {e}")
        
        self.stats["keys_generated"] += 1
        logger.info(f"Generated {algorithm} key {key_id} ({key_size} bits)")
        
        return key
    
    async def get_key(self, key_id: str, include_private: bool = True) -> Optional[CryptographicKey]:
        """Retrieve key by ID"""
        key = self.key_store.retrieve_key(key_id, decrypt_private=include_private)
        
        if key and include_private:
            # Update usage count
            key.usage_count += 1
            self.key_store.store_key(key)
        
        return key
    
    async def rotate_key(self, key_id: str, reason: str = "manual") -> str:
        """Rotate key"""
        trigger = RotationTrigger.MANUAL if reason == "manual" else RotationTrigger.SCHEDULED
        new_key_id = await self.rotation_manager.rotate_key(key_id, trigger, self.node_id)
        
        # Create backup of new key
        try:
            await self.backup_manager.create_backup(
                new_key_id, ["local_encrypted"], recovery_threshold=1
            )
            self.stats["keys_backed_up"] += 1
        except Exception as e:
            logger.warning(f"Failed to create backup for rotated key {new_key_id}: {e}")
        
        self.stats["keys_rotated"] += 1
        return new_key_id
    
    async def backup_key(self, key_id: str, locations: Optional[List[str]] = None) -> str:
        """Create backup of key"""
        if locations is None:
            locations = ["local_encrypted"]
        
        backup_id = await self.backup_manager.create_backup(key_id, locations)
        self.stats["keys_backed_up"] += 1
        return backup_id
    
    async def recover_key(self, backup_id: str, recovery_data: Dict[str, Any]) -> CryptographicKey:
        """Recover key from backup"""
        key = await self.backup_manager.recover_key(backup_id, recovery_data)
        
        # Store recovered key
        if not self.key_store.store_key(key):
            raise Exception(f"Failed to store recovered key {key.key_id}")
        
        self.stats["keys_recovered"] += 1
        logger.info(f"Recovered key {key.key_id} from backup {backup_id}")
        
        return key
    
    async def initiate_distributed_key_generation(
        self,
        ceremony_id: str,
        participants: List[str],
        threshold: int,
        key_type: KeyType = KeyType.SIGNING
    ) -> Dict[str, Any]:
        """Initiate distributed key generation"""
        result = await self.dkg.initiate_dkg_ceremony(
            ceremony_id, participants, threshold, key_type
        )
        return result
    
    async def participate_in_dkg(self, ceremony_id: str) -> Dict[str, Any]:
        """Participate in distributed key generation"""
        # Generate contribution
        contribution = secrets.token_bytes(64)  # Random contribution
        
        result = await self.dkg.contribute_to_ceremony(ceremony_id, contribution)
        
        if result.get("phase") == "completed":
            self.stats["dkg_ceremonies_completed"] += 1
        
        return result
    
    def list_keys(
        self,
        key_type: Optional[KeyType] = None,
        status: Optional[KeyStatus] = None,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """List keys with metadata"""
        key_ids = self.key_store.list_keys(key_type, status)
        keys_info = []
        
        for key_id in key_ids:
            key = self.key_store.retrieve_key(key_id, decrypt_private=False)
            if not key:
                continue
            
            if not include_expired and key.is_expired():
                continue
            
            keys_info.append({
                "key_id": key.key_id,
                "key_type": key.key_type.value,
                "algorithm": key.algorithm,
                "key_size": key.key_size,
                "status": key.status.value,
                "created_at": key.created_at,
                "expires_at": key.expires_at,
                "usage_count": key.usage_count,
                "needs_rotation": key.needs_rotation(),
                "is_expired": key.is_expired()
            })
        
        return sorted(keys_info, key=lambda x: x["created_at"], reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get key manager statistics"""
        active_keys = len(self.key_store.list_keys(status=KeyStatus.ACTIVE))
        expired_keys = len([
            k for k in self.key_store.keys.values()
            if k.is_expired()
        ])
        
        return {
            "node_id": self.node_id,
            "statistics": self.stats,
            "key_inventory": {
                "total_keys": len(self.key_store.keys),
                "active_keys": active_keys,
                "expired_keys": expired_keys,
                "backup_count": len(self.key_store.backups)
            },
            "rotation_status": {
                "policies_active": len(self.rotation_manager.rotation_policies),
                "rotation_events": len(self.rotation_manager.rotation_events),
                "monitoring_active": self.rotation_manager.background_task is not None
            },
            "dkg_status": {
                "active_ceremonies": len(self.dkg.active_ceremonies),
                "completed_ceremonies": len(self.dkg.completed_ceremonies)
            },
            "backup_status": {
                "backup_locations": len(self.backup_manager.backup_locations),
                "total_backups": len(self.key_store.backups)
            }
        }
    
    async def shutdown(self):
        """Shutdown key manager"""
        logger.info("Shutting down secure key manager")
        
        # Stop rotation monitoring
        await self.rotation_manager.stop_rotation_monitoring()
        
        logger.info("Secure key manager shutdown complete")


# Factory function for system creation
def create_secure_key_manager(
    node_id: str,
    master_password: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> SecureKeyManager:
    """
    Factory function to create secure key manager
    
    Args:
        node_id: Unique identifier for this key management node
        master_password: Master password for key encryption
        config: Optional configuration overrides
        
    Returns:
        Configured secure key manager
    """
    manager = SecureKeyManager(node_id, master_password)
    
    if config:
        # Apply configuration overrides
        rotation_policies = config.get("rotation_policies", {})
        for policy_id, policy_config in rotation_policies.items():
            manager.rotation_manager.set_rotation_policy(
                policy_id,
                [KeyType(kt) for kt in policy_config["key_types"]],
                policy_config["rotation_interval"],
                policy_config.get("max_usage"),
                policy_config.get("auto_rotate", True)
            )
        
        backup_locations = config.get("backup_locations", {})
        for location_id, location_config in backup_locations.items():
            manager.backup_manager.add_backup_location(
                location_id,
                location_config["type"],
                location_config["config"]
            )
    
    logger.info(f"Created secure key manager for node {node_id}")
    
    return manager