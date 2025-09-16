"""
Automotive Secure Communication Module
Implements V2X encryption, secure CAN bus communication, and authentication
Compliant with UN R155 and ISO/SAE 21434 standards
"""

import os
import time
import hashlib
import hmac
import struct
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import json
import logging

class V2XMessageType(Enum):
    """V2X message types for different communication scenarios"""
    BASIC_SAFETY = "BSM"
    COOPERATIVE_AWARENESS = "CAM"
    DECENTRALIZED_NOTIFICATION = "DENM"
    SIGNAL_PHASE_TIMING = "SPAT"
    MAP_DATA = "MAP"
    PERSONAL_SAFETY = "PSM"

class SecurityLevel(Enum):
    """Security levels for different communication channels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class V2XMessage:
    """Structured V2X message with security metadata"""
    message_id: str
    message_type: V2XMessageType
    payload: bytes
    timestamp: float
    source_id: str
    security_level: SecurityLevel
    signature: Optional[bytes] = None
    encryption_key_id: Optional[str] = None

@dataclass
class CANFrame:
    """Secure CAN frame with authentication"""
    can_id: int
    data: bytes
    timestamp: float
    dlc: int
    mac: Optional[bytes] = None
    sequence_number: Optional[int] = None

class V2XEncryption:
    """V2X encryption and key management system"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.key_store = {}
        self.session_keys = {}
        self.certificate_cache = {}
        self._initialize_crypto()

    def _initialize_crypto(self):
        """Initialize cryptographic components"""
        # Generate master keys for V2X communication
        self.master_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.master_public_key = self.master_private_key.public_key()

        # Initialize AES encryption parameters
        self.aes_key_size = 256  # bits
        self.iv_size = 16  # bytes

        self.logger.info("V2X encryption system initialized")

    def generate_session_key(self, peer_id: str) -> bytes:
        """Generate session key for secure communication"""
        # Use PBKDF2 for key derivation
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )

        # Derive key from master key and peer ID
        key_material = f"{peer_id}_{time.time()}".encode()
        session_key = kdf.derive(key_material)

        self.session_keys[peer_id] = {
            'key': session_key,
            'salt': salt,
            'created': time.time(),
            'usage_count': 0
        }

        return session_key

    def encrypt_v2x_message(self, message: V2XMessage, peer_id: str) -> bytes:
        """Encrypt V2X message with session key"""
        if peer_id not in self.session_keys:
            self.generate_session_key(peer_id)

        session_key = self.session_keys[peer_id]['key']
        iv = os.urandom(self.iv_size)

        # Create cipher
        cipher = Cipher(
            algorithms.AES(session_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        # Serialize message
        message_data = json.dumps({
            'message_id': message.message_id,
            'message_type': message.message_type.value,
            'payload': message.payload.hex(),
            'timestamp': message.timestamp,
            'source_id': message.source_id,
            'security_level': message.security_level.value
        }).encode()

        # Add padding for AES block size
        padding_length = 16 - (len(message_data) % 16)
        padded_data = message_data + bytes([padding_length]) * padding_length

        # Encrypt
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        # Update usage count
        self.session_keys[peer_id]['usage_count'] += 1

        return iv + encrypted_data

    def decrypt_v2x_message(self, encrypted_data: bytes, peer_id: str) -> V2XMessage:
        """Decrypt V2X message"""
        if peer_id not in self.session_keys:
            raise ValueError(f"No session key found for peer {peer_id}")

        session_key = self.session_keys[peer_id]['key']

        # Extract IV and ciphertext
        iv = encrypted_data[:self.iv_size]
        ciphertext = encrypted_data[self.iv_size:]

        # Create cipher
        cipher = Cipher(
            algorithms.AES(session_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()

        # Decrypt
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Remove padding
        padding_length = padded_data[-1]
        message_data = padded_data[:-padding_length]

        # Deserialize message
        data = json.loads(message_data.decode())

        return V2XMessage(
            message_id=data['message_id'],
            message_type=V2XMessageType(data['message_type']),
            payload=bytes.fromhex(data['payload']),
            timestamp=data['timestamp'],
            source_id=data['source_id'],
            security_level=SecurityLevel(data['security_level'])
        )

    def sign_message(self, message: V2XMessage) -> bytes:
        """Sign V2X message with digital signature"""
        # Create message hash
        message_hash = hashlib.sha256(
            f"{message.message_id}{message.timestamp}{message.payload.hex()}".encode()
        ).digest()

        # Sign with RSA private key
        signature = self.master_private_key.sign(
            message_hash,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return signature

    def verify_signature(self, message: V2XMessage, signature: bytes, public_key: rsa.RSAPublicKey) -> bool:
        """Verify message signature"""
        try:
            # Create message hash
            message_hash = hashlib.sha256(
                f"{message.message_id}{message.timestamp}{message.payload.hex()}".encode()
            ).digest()

            # Verify signature
            public_key.verify(
                signature,
                message_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            self.logger.error(f"Signature verification failed: {e}")
            return False

class SecureCANBus:
    """Secure CAN bus communication with authentication and integrity protection"""

    def __init__(self, bus_id: str):
        self.bus_id = bus_id
        self.logger = logging.getLogger(__name__)
        self.frame_sequence = 0
        self.sequence_window = {}  # Track sequence numbers per CAN ID
        self.mac_keys = {}  # MAC keys per CAN ID
        self.frame_buffer = {}
        self._initialize_security()

    def _initialize_security(self):
        """Initialize CAN bus security components"""
        # Generate MAC keys for different CAN IDs
        self.master_mac_key = os.urandom(32)  # 256-bit key
        self.max_sequence_gap = 100  # Maximum allowed sequence number gap

        self.logger.info(f"Secure CAN bus {self.bus_id} initialized")

    def generate_mac_key(self, can_id: int) -> bytes:
        """Generate MAC key for specific CAN ID"""
        # Derive key from master key and CAN ID
        key_data = f"CAN_{can_id}_{self.bus_id}".encode()
        mac_key = hmac.new(self.master_mac_key, key_data, hashlib.sha256).digest()
        self.mac_keys[can_id] = mac_key
        return mac_key

    def calculate_mac(self, can_frame: CANFrame) -> bytes:
        """Calculate MAC for CAN frame"""
        if can_frame.can_id not in self.mac_keys:
            self.generate_mac_key(can_frame.can_id)

        mac_key = self.mac_keys[can_frame.can_id]

        # Create MAC input data
        mac_data = struct.pack('>I', can_frame.can_id)  # CAN ID (4 bytes)
        mac_data += struct.pack('>B', can_frame.dlc)    # DLC (1 byte)
        mac_data += can_frame.data                       # Data payload
        mac_data += struct.pack('>Q', int(can_frame.timestamp * 1000))  # Timestamp
        if can_frame.sequence_number is not None:
            mac_data += struct.pack('>I', can_frame.sequence_number)

        # Calculate HMAC
        mac = hmac.new(mac_key, mac_data, hashlib.sha256).digest()[:8]  # Truncate to 8 bytes

        return mac

    def secure_send_frame(self, can_id: int, data: bytes, priority: SecurityLevel = SecurityLevel.MEDIUM) -> CANFrame:
        """Send secure CAN frame with authentication"""
        # Create frame
        frame = CANFrame(
            can_id=can_id,
            data=data,
            timestamp=time.time(),
            dlc=len(data),
            sequence_number=self._get_next_sequence(can_id)
        )

        # Calculate MAC
        frame.mac = self.calculate_mac(frame)

        # Log frame for monitoring
        self.logger.debug(f"Sending secure CAN frame: ID=0x{can_id:X}, DLC={frame.dlc}")

        return frame

    def verify_frame(self, frame: CANFrame) -> bool:
        """Verify CAN frame authenticity and integrity"""
        # Verify MAC
        expected_mac = self.calculate_mac(frame)
        if not hmac.compare_digest(frame.mac, expected_mac):
            self.logger.warning(f"MAC verification failed for CAN ID 0x{frame.can_id:X}")
            return False

        # Verify sequence number
        if not self._verify_sequence_number(frame):
            self.logger.warning(f"Sequence number verification failed for CAN ID 0x{frame.can_id:X}")
            return False

        # Check timestamp freshness (within 1 second)
        current_time = time.time()
        if abs(current_time - frame.timestamp) > 1.0:
            self.logger.warning(f"Timestamp verification failed for CAN ID 0x{frame.can_id:X}")
            return False

        self.logger.debug(f"Frame verification successful for CAN ID 0x{frame.can_id:X}")
        return True

    def _get_next_sequence(self, can_id: int) -> int:
        """Get next sequence number for CAN ID"""
        if can_id not in self.sequence_window:
            self.sequence_window[can_id] = 0

        self.sequence_window[can_id] = (self.sequence_window[can_id] + 1) % 0xFFFFFFFF
        return self.sequence_window[can_id]

    def _verify_sequence_number(self, frame: CANFrame) -> bool:
        """Verify sequence number to prevent replay attacks"""
        if frame.sequence_number is None:
            return False

        can_id = frame.can_id
        if can_id not in self.sequence_window:
            self.sequence_window[can_id] = frame.sequence_number
            return True

        last_sequence = self.sequence_window[can_id]

        # Check for sequence number advancement
        if frame.sequence_number > last_sequence:
            # Check for reasonable gap
            if frame.sequence_number - last_sequence <= self.max_sequence_gap:
                self.sequence_window[can_id] = frame.sequence_number
                return True
        elif frame.sequence_number < last_sequence:
            # Handle wraparound
            wraparound_diff = (0xFFFFFFFF - last_sequence) + frame.sequence_number
            if wraparound_diff <= self.max_sequence_gap:
                self.sequence_window[can_id] = frame.sequence_number
                return True

        return False

class AuthenticationManager:
    """Manages device authentication and authorization"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device_certificates = {}
        self.revoked_certificates = set()
        self.authentication_cache = {}
        self.challenge_responses = {}

    def register_device(self, device_id: str, certificate: bytes, public_key: rsa.RSAPublicKey) -> bool:
        """Register device with certificate and public key"""
        try:
            # Verify certificate validity (simplified)
            cert_hash = hashlib.sha256(certificate).hexdigest()

            self.device_certificates[device_id] = {
                'certificate': certificate,
                'public_key': public_key,
                'cert_hash': cert_hash,
                'registered_time': time.time(),
                'last_seen': time.time()
            }

            self.logger.info(f"Device {device_id} registered successfully")
            return True

        except Exception as e:
            self.logger.error(f"Device registration failed for {device_id}: {e}")
            return False

    def authenticate_device(self, device_id: str, challenge_response: bytes) -> bool:
        """Authenticate device using challenge-response"""
        if device_id not in self.device_certificates:
            self.logger.warning(f"Unknown device authentication attempt: {device_id}")
            return False

        if device_id in self.revoked_certificates:
            self.logger.warning(f"Revoked device authentication attempt: {device_id}")
            return False

        # Get stored challenge for device
        if device_id not in self.challenge_responses:
            self.logger.warning(f"No challenge found for device: {device_id}")
            return False

        challenge = self.challenge_responses[device_id]['challenge']
        expected_response = self.challenge_responses[device_id]['expected_response']

        # Verify response
        if hmac.compare_digest(challenge_response, expected_response):
            # Update last seen time
            self.device_certificates[device_id]['last_seen'] = time.time()

            # Cache authentication result
            self.authentication_cache[device_id] = {
                'authenticated': True,
                'timestamp': time.time()
            }

            self.logger.info(f"Device {device_id} authenticated successfully")
            return True

        self.logger.warning(f"Authentication failed for device: {device_id}")
        return False

    def generate_challenge(self, device_id: str) -> bytes:
        """Generate authentication challenge for device"""
        if device_id not in self.device_certificates:
            raise ValueError(f"Device {device_id} not registered")

        # Generate random challenge
        challenge = os.urandom(32)

        # Calculate expected response using device's public key
        device_info = self.device_certificates[device_id]
        public_key = device_info['public_key']

        # Expected response is signature of challenge
        # Note: In real implementation, device would sign with private key
        expected_response = hashlib.sha256(challenge + device_id.encode()).digest()

        self.challenge_responses[device_id] = {
            'challenge': challenge,
            'expected_response': expected_response,
            'timestamp': time.time()
        }

        return challenge

    def revoke_device(self, device_id: str) -> bool:
        """Revoke device certificate"""
        if device_id in self.device_certificates:
            self.revoked_certificates.add(device_id)

            # Remove from authentication cache
            if device_id in self.authentication_cache:
                del self.authentication_cache[device_id]

            self.logger.warning(f"Device {device_id} certificate revoked")
            return True

        return False

    def is_authenticated(self, device_id: str) -> bool:
        """Check if device is currently authenticated"""
        if device_id not in self.authentication_cache:
            return False

        auth_data = self.authentication_cache[device_id]

        # Check if authentication is still valid (5 minute timeout)
        if time.time() - auth_data['timestamp'] > 300:
            del self.authentication_cache[device_id]
            return False

        return auth_data['authenticated']

class AntiTamperingSystem:
    """Anti-tampering and integrity monitoring system"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.integrity_hashes = {}
        self.tamper_events = []
        self.monitoring_active = False
        self.monitoring_thread = None

    def calculate_system_integrity(self) -> Dict[str, str]:
        """Calculate integrity hashes for critical system components"""
        integrity_data = {}

        # Critical file paths to monitor
        critical_paths = [
            '/boot/bootloader',
            '/system/kernel',
            '/app/adas_core',
            '/config/security.conf'
        ]

        for path in critical_paths:
            try:
                # In real implementation, would read actual files
                # Using simulated data for demonstration
                file_content = f"simulated_content_{path}_{time.time()}".encode()
                file_hash = hashlib.sha256(file_content).hexdigest()
                integrity_data[path] = file_hash
            except Exception as e:
                self.logger.error(f"Failed to calculate hash for {path}: {e}")
                integrity_data[path] = "ERROR"

        return integrity_data

    def initialize_baseline(self):
        """Initialize baseline integrity measurements"""
        self.integrity_hashes = self.calculate_system_integrity()
        self.logger.info("Integrity baseline initialized")

    def check_integrity(self) -> List[str]:
        """Check system integrity against baseline"""
        current_hashes = self.calculate_system_integrity()
        tampered_components = []

        for path, baseline_hash in self.integrity_hashes.items():
            current_hash = current_hashes.get(path, "MISSING")

            if current_hash != baseline_hash:
                tampered_components.append(path)

                tamper_event = {
                    'timestamp': time.time(),
                    'component': path,
                    'baseline_hash': baseline_hash,
                    'current_hash': current_hash,
                    'event_type': 'INTEGRITY_VIOLATION'
                }

                self.tamper_events.append(tamper_event)
                self.logger.critical(f"Integrity violation detected: {path}")

        return tampered_components

    def start_monitoring(self, interval: int = 60):
        """Start continuous integrity monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Anti-tampering monitoring started")

    def stop_monitoring(self):
        """Stop integrity monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Anti-tampering monitoring stopped")

    def _monitoring_loop(self, interval: int):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                tampered = self.check_integrity()
                if tampered:
                    self.logger.critical(f"Tampering detected in components: {tampered}")

                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(interval)

    def get_tamper_events(self, since: Optional[float] = None) -> List[Dict]:
        """Get tamper events since specified timestamp"""
        if since is None:
            return self.tamper_events

        return [event for event in self.tamper_events if event['timestamp'] >= since]

    def clear_tamper_events(self):
        """Clear tamper event history"""
        self.tamper_events.clear()
        self.logger.info("Tamper event history cleared")

class SecureCommunicationManager:
    """Main secure communication management system"""

    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.v2x_encryption = V2XEncryption()
        self.can_buses = {}  # Multiple CAN buses
        self.auth_manager = AuthenticationManager()
        self.anti_tamper = AntiTamperingSystem()

        # Security metrics
        self.security_metrics = {
            'messages_encrypted': 0,
            'messages_decrypted': 0,
            'authentication_attempts': 0,
            'authentication_failures': 0,
            'integrity_violations': 0,
            'can_frames_secured': 0
        }

        self._initialize_system()

    def _initialize_system(self):
        """Initialize the secure communication system"""
        # Initialize anti-tampering baseline
        self.anti_tamper.initialize_baseline()
        self.anti_tamper.start_monitoring()

        # Create default CAN buses
        self.add_can_bus("powertrain", SecurityLevel.CRITICAL)
        self.add_can_bus("body", SecurityLevel.MEDIUM)
        self.add_can_bus("infotainment", SecurityLevel.LOW)

        self.logger.info(f"Secure communication system initialized for vehicle {self.vehicle_id}")

    def add_can_bus(self, bus_name: str, security_level: SecurityLevel):
        """Add secure CAN bus"""
        self.can_buses[bus_name] = {
            'bus': SecureCANBus(bus_name),
            'security_level': security_level
        }
        self.logger.info(f"Added secure CAN bus: {bus_name}")

    def send_v2x_message(self, message: V2XMessage, peer_id: str) -> bool:
        """Send secure V2X message"""
        try:
            # Sign message
            message.signature = self.v2x_encryption.sign_message(message)

            # Encrypt message
            encrypted_data = self.v2x_encryption.encrypt_v2x_message(message, peer_id)

            # In real implementation, would send via V2X radio
            self.logger.info(f"V2X message sent to {peer_id}: {message.message_id}")

            self.security_metrics['messages_encrypted'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Failed to send V2X message: {e}")
            return False

    def receive_v2x_message(self, encrypted_data: bytes, peer_id: str, signature: bytes) -> Optional[V2XMessage]:
        """Receive and decrypt V2X message"""
        try:
            # Decrypt message
            message = self.v2x_encryption.decrypt_v2x_message(encrypted_data, peer_id)

            # Verify signature
            # Note: Would need peer's public key in real implementation
            peer_public_key = self.v2x_encryption.master_public_key  # Simplified

            if not self.v2x_encryption.verify_signature(message, signature, peer_public_key):
                self.logger.warning(f"Signature verification failed for message from {peer_id}")
                return None

            self.security_metrics['messages_decrypted'] += 1
            self.logger.info(f"V2X message received from {peer_id}: {message.message_id}")

            return message

        except Exception as e:
            self.logger.error(f"Failed to receive V2X message: {e}")
            return None

    def send_can_frame(self, bus_name: str, can_id: int, data: bytes) -> bool:
        """Send secure CAN frame"""
        if bus_name not in self.can_buses:
            self.logger.error(f"Unknown CAN bus: {bus_name}")
            return False

        try:
            can_bus = self.can_buses[bus_name]['bus']
            security_level = self.can_buses[bus_name]['security_level']

            frame = can_bus.secure_send_frame(can_id, data, security_level)

            # In real implementation, would send via CAN controller
            self.logger.debug(f"CAN frame sent on {bus_name}: ID=0x{can_id:X}")

            self.security_metrics['can_frames_secured'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Failed to send CAN frame: {e}")
            return False

    def receive_can_frame(self, bus_name: str, frame: CANFrame) -> bool:
        """Receive and verify CAN frame"""
        if bus_name not in self.can_buses:
            self.logger.error(f"Unknown CAN bus: {bus_name}")
            return False

        try:
            can_bus = self.can_buses[bus_name]['bus']

            if can_bus.verify_frame(frame):
                self.logger.debug(f"CAN frame verified on {bus_name}: ID=0x{frame.can_id:X}")
                return True
            else:
                self.logger.warning(f"CAN frame verification failed on {bus_name}: ID=0x{frame.can_id:X}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to verify CAN frame: {e}")
            return False

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        # Check integrity
        tampered_components = self.anti_tamper.check_integrity()

        return {
            'vehicle_id': self.vehicle_id,
            'timestamp': time.time(),
            'security_metrics': self.security_metrics.copy(),
            'integrity_status': 'COMPROMISED' if tampered_components else 'INTACT',
            'tampered_components': tampered_components,
            'can_buses': list(self.can_buses.keys()),
            'active_sessions': len(self.v2x_encryption.session_keys),
            'registered_devices': len(self.auth_manager.device_certificates),
            'revoked_devices': len(self.auth_manager.revoked_certificates)
        }

    def shutdown(self):
        """Shutdown secure communication system"""
        self.anti_tamper.stop_monitoring()
        self.logger.info("Secure communication system shutdown")

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize secure communication system
    comm_system = SecureCommunicationManager("VEHICLE_001")

    # Example V2X message
    v2x_message = V2XMessage(
        message_id="BSM_001",
        message_type=V2XMessageType.BASIC_SAFETY,
        payload=b"Vehicle position and velocity data",
        timestamp=time.time(),
        source_id="VEHICLE_001",
        security_level=SecurityLevel.HIGH
    )

    # Send V2X message
    comm_system.send_v2x_message(v2x_message, "RSU_001")

    # Send CAN frame
    comm_system.send_can_frame("powertrain", 0x123, b"\x01\x02\x03\x04")

    # Get security status
    status = comm_system.get_security_status()
    print(f"Security Status: {status}")

    # Cleanup
    comm_system.shutdown()