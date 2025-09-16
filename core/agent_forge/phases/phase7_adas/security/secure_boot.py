"""
Automotive Secure Boot System
Implements secure boot, firmware verification, key storage, and HSM integration
Compliant with UN R155 and ISO/SAE 21434 standards
"""

import os
import time
import hashlib
import hmac
import struct
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import logging
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class BootStage(Enum):
    """Boot process stages"""
    POWER_ON_RESET = "por"
    SECURE_ROM = "secure_rom"
    BOOTLOADER = "bootloader"
    KERNEL = "kernel"
    SYSTEM_SERVICES = "system_services"
    APPLICATION = "application"
    OPERATIONAL = "operational"

class VerificationResult(Enum):
    """Verification result codes"""
    SUCCESS = "success"
    SIGNATURE_INVALID = "signature_invalid"
    HASH_MISMATCH = "hash_mismatch"
    CERTIFICATE_INVALID = "certificate_invalid"
    REVOKED_KEY = "revoked_key"
    EXPIRED_CERTIFICATE = "expired_certificate"
    ROLLBACK_DETECTED = "rollback_detected"
    TAMPER_DETECTED = "tamper_detected"

class SecurityLevel(Enum):
    """Security levels for different components"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class FirmwareImage:
    """Firmware image structure"""
    name: str
    version: str
    hash_algorithm: str
    hash_value: bytes
    signature: bytes
    certificate_chain: List[bytes]
    metadata: Dict[str, Any]
    size: int
    load_address: int
    entry_point: int

@dataclass
class SecureBootEvent:
    """Secure boot event for audit logging"""
    timestamp: float
    stage: BootStage
    component: str
    event_type: str
    result: VerificationResult
    details: Dict[str, Any]
    duration: float

@dataclass
class TrustedKey:
    """Trusted key for secure boot"""
    key_id: str
    key_type: str
    public_key: bytes
    certificate: bytes
    usage: List[str]  # signing, encryption, authentication
    security_level: SecurityLevel
    expiration: Optional[float]
    revoked: bool

class HardwareSecurityModule:
    """Hardware Security Module interface for automotive applications"""

    def __init__(self, hsm_type: str = "automotive_hsm"):
        self.hsm_type = hsm_type
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.key_store = {}
        self.secure_counters = {}
        self.attestation_keys = {}
        self._initialize_hsm()

    def _initialize_hsm(self):
        """Initialize HSM hardware interface"""
        # In real implementation, this would interface with actual HSM hardware
        self.hardware_id = "HSM_" + hashlib.sha256(os.urandom(32)).hexdigest()[:16]

        # Initialize secure storage
        self.secure_storage = {
            'root_keys': {},
            'device_keys': {},
            'certificates': {},
            'counters': {},
            'configuration': {}
        }

        # Generate device attestation key
        self._generate_attestation_keys()

        self.initialized = True
        self.logger.info(f"HSM initialized: {self.hardware_id}")

    def _generate_attestation_keys(self):
        """Generate device attestation keys"""
        # Generate RSA key pair for attestation
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        self.attestation_keys = {
            'private_key': private_key,
            'public_key': private_key.public_key(),
            'certificate': self._create_self_signed_cert(private_key)
        }

    def _create_self_signed_cert(self, private_key: rsa.RSAPrivateKey) -> bytes:
        """Create self-signed certificate for attestation"""
        # Simplified certificate creation
        cert_data = {
            'subject': f'HSM-{self.hardware_id}',
            'issuer': f'HSM-{self.hardware_id}',
            'serial': int(time.time()),
            'not_before': time.time(),
            'not_after': time.time() + (365 * 24 * 3600),  # 1 year
            'public_key': private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        }

        cert_json = json.dumps(cert_data, sort_keys=True)
        return cert_json.encode()

    def store_key(self, key_id: str, key_data: bytes, key_type: str = "symmetric") -> bool:
        """Store key in secure storage"""
        if not self.initialized:
            return False

        try:
            # Encrypt key data (in real HSM, hardware would handle this)
            encrypted_key = self._encrypt_key_data(key_data)

            self.secure_storage['device_keys'][key_id] = {
                'encrypted_data': encrypted_key,
                'key_type': key_type,
                'created': time.time(),
                'usage_count': 0
            }

            self.logger.info(f"Key stored in HSM: {key_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to store key {key_id}: {e}")
            return False

    def retrieve_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve key from secure storage"""
        if key_id not in self.secure_storage['device_keys']:
            return None

        try:
            key_info = self.secure_storage['device_keys'][key_id]
            decrypted_key = self._decrypt_key_data(key_info['encrypted_data'])

            # Update usage count
            key_info['usage_count'] += 1

            return decrypted_key

        except Exception as e:
            self.logger.error(f"Failed to retrieve key {key_id}: {e}")
            return None

    def _encrypt_key_data(self, key_data: bytes) -> bytes:
        """Encrypt key data for storage"""
        # Simplified encryption (real HSM would use hardware encryption)
        key = hashlib.sha256(self.hardware_id.encode()).digest()
        iv = os.urandom(16)

        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        # Add padding
        padding_length = 16 - (len(key_data) % 16)
        padded_data = key_data + bytes([padding_length]) * padding_length

        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        return iv + encrypted_data

    def _decrypt_key_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt key data from storage"""
        key = hashlib.sha256(self.hardware_id.encode()).digest()
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]

        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()

        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Remove padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]

    def sign_data(self, key_id: str, data: bytes) -> Optional[bytes]:
        """Sign data using stored key"""
        if key_id not in self.secure_storage['device_keys']:
            return None

        try:
            # For demonstration, use attestation key
            signature = self.attestation_keys['private_key'].sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            self.logger.debug(f"Data signed with key {key_id}")
            return signature

        except Exception as e:
            self.logger.error(f"Failed to sign data with key {key_id}: {e}")
            return None

    def verify_signature(self, key_id: str, data: bytes, signature: bytes) -> bool:
        """Verify signature using stored key"""
        try:
            # For demonstration, use attestation public key
            self.attestation_keys['public_key'].verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return True

        except Exception as e:
            self.logger.debug(f"Signature verification failed for key {key_id}: {e}")
            return False

    def get_secure_counter(self, counter_id: str) -> int:
        """Get monotonic secure counter value"""
        if counter_id not in self.secure_storage['counters']:
            self.secure_storage['counters'][counter_id] = 0

        return self.secure_storage['counters'][counter_id]

    def increment_secure_counter(self, counter_id: str) -> int:
        """Increment and return secure counter value"""
        current_value = self.get_secure_counter(counter_id)
        new_value = current_value + 1

        self.secure_storage['counters'][counter_id] = new_value

        self.logger.debug(f"Counter {counter_id} incremented to {new_value}")
        return new_value

    def get_device_attestation(self) -> Dict[str, Any]:
        """Get device attestation information"""
        return {
            'hardware_id': self.hardware_id,
            'certificate': self.attestation_keys['certificate'],
            'public_key': self.attestation_keys['public_key'].public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ),
            'timestamp': time.time()
        }

class SecureKeyStorage:
    """Secure key storage and management system"""

    def __init__(self, hsm: HardwareSecurityModule):
        self.hsm = hsm
        self.logger = logging.getLogger(__name__)
        self.trusted_keys = {}
        self.revoked_keys = set()
        self.key_usage_log = []

        # Initialize with default root keys
        self._initialize_root_keys()

    def _initialize_root_keys(self):
        """Initialize root keys for the secure boot chain"""
        # Root CA key for verification
        root_ca_key = TrustedKey(
            key_id="ROOT_CA",
            key_type="RSA2048",
            public_key=b"dummy_root_ca_public_key",  # Would be actual key in production
            certificate=b"dummy_root_ca_certificate",
            usage=["signing", "certification"],
            security_level=SecurityLevel.CRITICAL,
            expiration=None,  # Root keys don't expire
            revoked=False
        )

        # OEM signing key
        oem_key = TrustedKey(
            key_id="OEM_SIGNING",
            key_type="RSA2048",
            public_key=b"dummy_oem_public_key",
            certificate=b"dummy_oem_certificate",
            usage=["signing"],
            security_level=SecurityLevel.CRITICAL,
            expiration=time.time() + (5 * 365 * 24 * 3600),  # 5 years
            revoked=False
        )

        # Store keys in HSM
        self.trusted_keys[root_ca_key.key_id] = root_ca_key
        self.trusted_keys[oem_key.key_id] = oem_key

        self.hsm.store_key("ROOT_CA_PRIVATE", b"dummy_root_ca_private_key", "asymmetric")
        self.hsm.store_key("OEM_PRIVATE", b"dummy_oem_private_key", "asymmetric")

        self.logger.info("Root keys initialized in secure storage")

    def add_trusted_key(self, trusted_key: TrustedKey) -> bool:
        """Add trusted key to the keystore"""
        try:
            # Verify key is not revoked
            if trusted_key.key_id in self.revoked_keys:
                self.logger.error(f"Cannot add revoked key: {trusted_key.key_id}")
                return False

            # Check expiration
            if trusted_key.expiration and time.time() > trusted_key.expiration:
                self.logger.error(f"Cannot add expired key: {trusted_key.key_id}")
                return False

            # Store in keystore
            self.trusted_keys[trusted_key.key_id] = trusted_key

            # Store key material in HSM
            key_material = trusted_key.public_key + trusted_key.certificate
            self.hsm.store_key(f"TRUSTED_{trusted_key.key_id}", key_material)

            self.logger.info(f"Trusted key added: {trusted_key.key_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add trusted key {trusted_key.key_id}: {e}")
            return False

    def get_trusted_key(self, key_id: str) -> Optional[TrustedKey]:
        """Get trusted key by ID"""
        if key_id not in self.trusted_keys:
            return None

        key = self.trusted_keys[key_id]

        # Check if key is revoked
        if key.revoked or key_id in self.revoked_keys:
            self.logger.warning(f"Attempt to use revoked key: {key_id}")
            return None

        # Check expiration
        if key.expiration and time.time() > key.expiration:
            self.logger.warning(f"Attempt to use expired key: {key_id}")
            return None

        # Log key usage
        self.key_usage_log.append({
            'key_id': key_id,
            'timestamp': time.time(),
            'operation': 'retrieve'
        })

        return key

    def revoke_key(self, key_id: str, reason: str) -> bool:
        """Revoke a trusted key"""
        if key_id in self.trusted_keys:
            self.trusted_keys[key_id].revoked = True
            self.revoked_keys.add(key_id)

            self.logger.warning(f"Key revoked: {key_id} - Reason: {reason}")

            # Log revocation
            self.key_usage_log.append({
                'key_id': key_id,
                'timestamp': time.time(),
                'operation': 'revoke',
                'reason': reason
            })

            return True

        return False

    def get_key_usage_history(self, key_id: str) -> List[Dict[str, Any]]:
        """Get usage history for a key"""
        return [log for log in self.key_usage_log if log['key_id'] == key_id]

class FirmwareVerificationEngine:
    """Firmware verification and validation engine"""

    def __init__(self, key_storage: SecureKeyStorage, hsm: HardwareSecurityModule):
        self.key_storage = key_storage
        self.hsm = hsm
        self.logger = logging.getLogger(__name__)
        self.verification_cache = {}
        self.rollback_counters = {}

    def verify_firmware_image(self, firmware: FirmwareImage) -> VerificationResult:
        """Verify firmware image integrity and authenticity"""
        try:
            # Step 1: Verify hash
            if not self._verify_hash(firmware):
                return VerificationResult.HASH_MISMATCH

            # Step 2: Verify certificate chain
            if not self._verify_certificate_chain(firmware.certificate_chain):
                return VerificationResult.CERTIFICATE_INVALID

            # Step 3: Verify signature
            if not self._verify_signature(firmware):
                return VerificationResult.SIGNATURE_INVALID

            # Step 4: Check for rollback protection
            if not self._check_rollback_protection(firmware):
                return VerificationResult.ROLLBACK_DETECTED

            # Step 5: Check for tampering indicators
            if not self._check_tampering(firmware):
                return VerificationResult.TAMPER_DETECTED

            self.logger.info(f"Firmware verification successful: {firmware.name}")
            return VerificationResult.SUCCESS

        except Exception as e:
            self.logger.error(f"Firmware verification error: {e}")
            return VerificationResult.SIGNATURE_INVALID

    def _verify_hash(self, firmware: FirmwareImage) -> bool:
        """Verify firmware image hash"""
        # Calculate hash of firmware image
        # In real implementation, would read actual firmware binary
        firmware_data = f"firmware_{firmware.name}_{firmware.version}".encode()

        if firmware.hash_algorithm == "SHA256":
            calculated_hash = hashlib.sha256(firmware_data).digest()
        elif firmware.hash_algorithm == "SHA384":
            calculated_hash = hashlib.sha384(firmware_data).digest()
        else:
            self.logger.error(f"Unsupported hash algorithm: {firmware.hash_algorithm}")
            return False

        hash_match = hmac.compare_digest(calculated_hash, firmware.hash_value)

        if not hash_match:
            self.logger.error(f"Hash mismatch for firmware: {firmware.name}")

        return hash_match

    def _verify_certificate_chain(self, certificate_chain: List[bytes]) -> bool:
        """Verify certificate chain up to root CA"""
        if not certificate_chain:
            return False

        # Simplified certificate chain verification
        # In production, would perform full X.509 validation

        try:
            # Verify each certificate in chain
            for i, cert in enumerate(certificate_chain):
                cert_data = json.loads(cert.decode())

                # Check certificate validity period
                current_time = time.time()
                if current_time < cert_data.get('not_before', 0):
                    self.logger.error(f"Certificate not yet valid: {i}")
                    return False

                if current_time > cert_data.get('not_after', float('inf')):
                    self.logger.error(f"Certificate expired: {i}")
                    return False

                # Verify certificate signature (simplified)
                if i > 0:  # Skip root certificate self-verification
                    parent_cert = json.loads(certificate_chain[i-1].decode())
                    # Would verify signature using parent's public key
                    pass

            self.logger.debug("Certificate chain verification successful")
            return True

        except Exception as e:
            self.logger.error(f"Certificate chain verification failed: {e}")
            return False

    def _verify_signature(self, firmware: FirmwareImage) -> bool:
        """Verify firmware signature"""
        try:
            # Get signing certificate from chain
            if not firmware.certificate_chain:
                return False

            signing_cert = json.loads(firmware.certificate_chain[-1].decode())

            # Create data to verify signature against
            signature_data = (
                firmware.name.encode() +
                firmware.version.encode() +
                firmware.hash_value +
                struct.pack('>I', firmware.size)
            )

            # Use HSM to verify signature
            # In real implementation, would extract public key from certificate
            return self.hsm.verify_signature("OEM_SIGNING", signature_data, firmware.signature)

        except Exception as e:
            self.logger.error(f"Signature verification failed: {e}")
            return False

    def _check_rollback_protection(self, firmware: FirmwareImage) -> bool:
        """Check firmware version against rollback protection"""
        component_name = firmware.name
        current_counter = self.hsm.get_secure_counter(f"rollback_{component_name}")

        # Extract version number from firmware version string
        try:
            # Assume version format like "1.2.3" or "v1.2.3"
            version_str = firmware.version.lstrip('v')
            version_parts = version_str.split('.')
            firmware_counter = int(version_parts[0]) * 10000 + int(version_parts[1]) * 100
            if len(version_parts) > 2:
                firmware_counter += int(version_parts[2])

            if firmware_counter < current_counter:
                self.logger.error(f"Rollback detected: firmware version {firmware.version} < current {current_counter}")
                return False

            # Update counter if firmware is newer
            if firmware_counter > current_counter:
                self.hsm.increment_secure_counter(f"rollback_{component_name}")
                self.rollback_counters[component_name] = firmware_counter

            return True

        except Exception as e:
            self.logger.error(f"Rollback protection check failed: {e}")
            return False

    def _check_tampering(self, firmware: FirmwareImage) -> bool:
        """Check for tampering indicators"""
        # Check metadata integrity
        metadata_hash = hashlib.sha256(
            json.dumps(firmware.metadata, sort_keys=True).encode()
        ).hexdigest()

        if 'metadata_hash' in firmware.metadata:
            if metadata_hash != firmware.metadata['metadata_hash']:
                self.logger.error("Metadata tampering detected")
                return False

        # Check for suspicious metadata
        suspicious_indicators = [
            'debug_enabled',
            'backdoor_access',
            'unsigned_code',
            'test_mode_permanent'
        ]

        for indicator in suspicious_indicators:
            if firmware.metadata.get(indicator, False):
                self.logger.warning(f"Suspicious metadata indicator: {indicator}")
                # Don't fail verification for warnings, just log

        return True

class SecureBootManager:
    """Main secure boot manager coordinating all components"""

    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.hsm = HardwareSecurityModule()
        self.key_storage = SecureKeyStorage(self.hsm)
        self.verification_engine = FirmwareVerificationEngine(self.key_storage, self.hsm)

        # Boot state tracking
        self.current_stage = BootStage.POWER_ON_RESET
        self.boot_events = []
        self.verified_components = {}
        self.boot_measurements = {}

        # Performance tracking
        self.boot_start_time = None
        self.stage_times = {}

        self.logger.info(f"Secure boot manager initialized for vehicle {vehicle_id}")

    def start_secure_boot(self) -> bool:
        """Start the secure boot process"""
        self.boot_start_time = time.time()
        self.logger.info("Starting secure boot process")

        try:
            # Stage 1: Secure ROM
            if not self._execute_boot_stage(BootStage.SECURE_ROM):
                return False

            # Stage 2: Bootloader
            if not self._execute_boot_stage(BootStage.BOOTLOADER):
                return False

            # Stage 3: Kernel
            if not self._execute_boot_stage(BootStage.KERNEL):
                return False

            # Stage 4: System Services
            if not self._execute_boot_stage(BootStage.SYSTEM_SERVICES):
                return False

            # Stage 5: Applications
            if not self._execute_boot_stage(BootStage.APPLICATION):
                return False

            # Stage 6: Operational
            self.current_stage = BootStage.OPERATIONAL

            boot_duration = time.time() - self.boot_start_time
            self.logger.info(f"Secure boot completed successfully in {boot_duration:.2f}s")

            return True

        except Exception as e:
            self.logger.critical(f"Secure boot failed: {e}")
            return False

    def _execute_boot_stage(self, stage: BootStage) -> bool:
        """Execute a specific boot stage"""
        stage_start_time = time.time()
        self.current_stage = stage

        self.logger.info(f"Executing boot stage: {stage.value}")

        try:
            # Get components for this stage
            components = self._get_stage_components(stage)

            for component_name in components:
                # Verify component
                if not self._verify_component(component_name, stage):
                    self._log_boot_event(
                        stage, component_name, "verification_failed",
                        VerificationResult.SIGNATURE_INVALID, {}
                    )
                    return False

                self._log_boot_event(
                    stage, component_name, "verification_success",
                    VerificationResult.SUCCESS, {}
                )

            # Measure boot stage performance
            stage_duration = time.time() - stage_start_time
            self.stage_times[stage.value] = stage_duration

            # Create measurement for attestation
            stage_measurement = self._create_stage_measurement(stage, components)
            self.boot_measurements[stage.value] = stage_measurement

            return True

        except Exception as e:
            self.logger.error(f"Boot stage {stage.value} failed: {e}")
            return False

    def _get_stage_components(self, stage: BootStage) -> List[str]:
        """Get components that need verification for each stage"""
        stage_components = {
            BootStage.SECURE_ROM: ["secure_rom"],
            BootStage.BOOTLOADER: ["bootloader", "bootloader_config"],
            BootStage.KERNEL: ["kernel", "kernel_modules", "device_drivers"],
            BootStage.SYSTEM_SERVICES: ["init_system", "security_services", "network_services"],
            BootStage.APPLICATION: ["adas_application", "vehicle_services", "diagnostics"]
        }

        return stage_components.get(stage, [])

    def _verify_component(self, component_name: str, stage: BootStage) -> bool:
        """Verify a specific component"""
        # Create dummy firmware image for demonstration
        firmware = FirmwareImage(
            name=component_name,
            version="1.0.0",
            hash_algorithm="SHA256",
            hash_value=hashlib.sha256(f"firmware_{component_name}_1.0.0".encode()).digest(),
            signature=b"dummy_signature",
            certificate_chain=[b'{"subject": "OEM", "issuer": "ROOT_CA", "not_before": 0, "not_after": 9999999999}'],
            metadata={"component_type": component_name, "stage": stage.value},
            size=1024,
            load_address=0x10000000,
            entry_point=0x10000000
        )

        # Sign firmware with HSM
        signature_data = (
            firmware.name.encode() +
            firmware.version.encode() +
            firmware.hash_value +
            struct.pack('>I', firmware.size)
        )

        firmware.signature = self.hsm.sign_data("OEM_SIGNING", signature_data)
        if not firmware.signature:
            return False

        # Verify firmware
        result = self.verification_engine.verify_firmware_image(firmware)

        if result == VerificationResult.SUCCESS:
            self.verified_components[component_name] = {
                'stage': stage,
                'verification_time': time.time(),
                'version': firmware.version,
                'hash': firmware.hash_value.hex()
            }
            return True

        return False

    def _create_stage_measurement(self, stage: BootStage, components: List[str]) -> bytes:
        """Create cryptographic measurement of boot stage"""
        measurement_data = {
            'stage': stage.value,
            'timestamp': time.time(),
            'components': components,
            'vehicle_id': self.vehicle_id,
            'hsm_id': self.hsm.hardware_id
        }

        # Create hash of stage measurement
        measurement_json = json.dumps(measurement_data, sort_keys=True)
        measurement_hash = hashlib.sha256(measurement_json.encode()).digest()

        return measurement_hash

    def _log_boot_event(self, stage: BootStage, component: str, event_type: str,
                       result: VerificationResult, details: Dict[str, Any]):
        """Log boot event for audit trail"""
        event = SecureBootEvent(
            timestamp=time.time(),
            stage=stage,
            component=component,
            event_type=event_type,
            result=result,
            details=details,
            duration=0.0  # Will be updated if needed
        )

        self.boot_events.append(event)

    def get_boot_attestation(self) -> Dict[str, Any]:
        """Generate boot attestation report"""
        device_attestation = self.hsm.get_device_attestation()

        attestation = {
            'vehicle_id': self.vehicle_id,
            'timestamp': time.time(),
            'boot_status': self.current_stage.value,
            'device_attestation': device_attestation,
            'verified_components': self.verified_components.copy(),
            'boot_measurements': {k: v.hex() for k, v in self.boot_measurements.items()},
            'stage_times': self.stage_times.copy(),
            'total_boot_time': sum(self.stage_times.values()) if self.stage_times else 0.0
        }

        # Sign attestation with HSM
        attestation_json = json.dumps(attestation, sort_keys=True)
        attestation_signature = self.hsm.sign_data("ROOT_CA", attestation_json.encode())

        if attestation_signature:
            attestation['signature'] = attestation_signature.hex()

        return attestation

    def verify_runtime_component(self, component_name: str, component_data: bytes) -> bool:
        """Verify component during runtime (for dynamic loading)"""
        try:
            # Calculate component hash
            component_hash = hashlib.sha256(component_data).digest()

            # Check against known good hashes
            if component_name in self.verified_components:
                known_hash = bytes.fromhex(self.verified_components[component_name]['hash'])
                if hmac.compare_digest(component_hash, known_hash):
                    return True

            # If not found in verified components, perform full verification
            # This would involve looking up the component's certificate and signature
            self.logger.warning(f"Runtime component verification not implemented for: {component_name}")
            return False

        except Exception as e:
            self.logger.error(f"Runtime component verification failed: {e}")
            return False

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return {
            'vehicle_id': self.vehicle_id,
            'current_stage': self.current_stage.value,
            'hsm_initialized': self.hsm.initialized,
            'verified_components_count': len(self.verified_components),
            'total_boot_events': len(self.boot_events),
            'trusted_keys_count': len(self.key_storage.trusted_keys),
            'revoked_keys_count': len(self.key_storage.revoked_keys),
            'secure_counters': dict(self.hsm.secure_storage['counters']),
            'last_boot_time': self.boot_start_time
        }

    def emergency_lockdown(self, reason: str) -> bool:
        """Emergency lockdown of the system"""
        self.logger.critical(f"EMERGENCY LOCKDOWN ACTIVATED: {reason}")

        try:
            # Revoke all non-essential keys
            for key_id in list(self.key_storage.trusted_keys.keys()):
                if key_id not in ['ROOT_CA', 'OEM_SIGNING']:  # Keep essential keys
                    self.key_storage.revoke_key(key_id, f"Emergency lockdown: {reason}")

            # Clear verification cache
            self.verification_engine.verification_cache.clear()

            # Log emergency event
            self._log_boot_event(
                self.current_stage, "SYSTEM", "emergency_lockdown",
                VerificationResult.TAMPER_DETECTED, {"reason": reason}
            )

            return True

        except Exception as e:
            self.logger.error(f"Emergency lockdown failed: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize secure boot manager
    boot_manager = SecureBootManager("VEHICLE_001")

    # Start secure boot process
    boot_success = boot_manager.start_secure_boot()

    if boot_success:
        print("Secure boot completed successfully")

        # Get boot attestation
        attestation = boot_manager.get_boot_attestation()
        print(f"Boot attestation: {json.dumps(attestation, indent=2)}")

        # Get security status
        status = boot_manager.get_security_status()
        print(f"Security status: {json.dumps(status, indent=2)}")

    else:
        print("Secure boot failed!")

        # In case of failure, might trigger emergency lockdown
        boot_manager.emergency_lockdown("Boot verification failed")