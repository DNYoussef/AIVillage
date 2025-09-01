"""
ECH + Noise Protocol Architecture Design

Clean architecture implementation with proper connascence management
and zero-breaking-change integration with existing P2P infrastructure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union
import logging
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# ============================================================================
# DOMAIN ENTITIES (Strong Connascence Internal Only)
# ============================================================================

class ECHVersion(Enum):
    """ECH version enumeration - prevents magic numbers"""
    VERSION_1 = 0x0001
    VERSION_2 = 0x0002

class CipherSuite(Enum):
    """Supported cipher suites for ECH and Noise"""
    CHACHA20_POLY1305_SHA256 = ("ChaCha20Poly1305", "SHA256")
    AES_256_GCM_SHA384 = ("AES256GCM", "SHA384")
    AES_128_GCM_SHA256 = ("AES128GCM", "SHA256")

class HandshakeStatus(Enum):
    """Handshake status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    FALLBACK = "fallback"

@dataclass(frozen=True)
class ECHConfig:
    """Immutable ECH configuration - prevents temporal coupling"""
    version: ECHVersion
    config_id: int
    kem_id: int
    public_key: bytes
    cipher_suites: List[CipherSuite]
    extensions: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validation to prevent invalid state"""
        if not self.cipher_suites:
            raise ValueError("At least one cipher suite required")
        if len(self.public_key) == 0:
            raise ValueError("Public key cannot be empty")
        if not 0 <= self.config_id <= 255:
            raise ValueError("Config ID must be 0-255")

@dataclass
class HandshakeResult:
    """Result of enhanced handshake operation"""
    success: bool
    connection_id: str
    encryption_key: Optional[bytes] = None
    decryption_key: Optional[bytes] = None
    ech_enabled: bool = False
    forward_secrecy: bool = False
    cipher_suite: Optional[CipherSuite] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class SecurityMetrics:
    """Security metrics collection"""
    handshake_duration_ms: float
    ech_success_rate: float
    fallback_rate: float
    threat_detections: int
    timestamp: float = field(default_factory=time.time)

# ============================================================================
# INTERFACES (Weak Connascence Boundaries)
# ============================================================================

class ECHConfigParser(Protocol):
    """ECH configuration parser interface - dependency inversion"""
    
    def parse_config(self, config_bytes: bytes) -> ECHConfig:
        """Parse ECH configuration from bytes"""
        ...
    
    def validate_config(self, config: ECHConfig) -> bool:
        """Validate ECH configuration"""
        ...

class NoiseHandshakeInterface(Protocol):
    """Noise protocol handshake interface"""
    
    async def initiate_handshake(self, peer_id: str) -> HandshakeResult:
        """Initiate handshake with peer"""
        ...
    
    async def process_handshake_response(self, response: bytes) -> HandshakeResult:
        """Process handshake response from peer"""
        ...

class TransportInterface(Protocol):
    """Transport layer interface - stable abstraction"""
    
    async def establish_connection(self, peer_id: str, **options) -> Any:
        """Establish connection to peer"""
        ...
    
    async def send_message(self, connection: Any, message: bytes) -> bool:
        """Send message through connection"""
        ...
    
    async def close_connection(self, connection: Any) -> None:
        """Close connection to peer"""
        ...

class SecurityMonitorInterface(Protocol):
    """Security monitoring interface"""
    
    def record_handshake_attempt(self, peer_id: str, success: bool) -> None:
        """Record handshake attempt"""
        ...
    
    def detect_threats(self, peer_id: str, metadata: Dict[str, Any]) -> List[str]:
        """Detect security threats"""
        ...

# ============================================================================
# ECH IMPLEMENTATION (Strong Connascence Localized)
# ============================================================================

class ECHError(Exception):
    """ECH-specific errors"""
    pass

class ECHConfigParserImpl:
    """ECH configuration parser implementation"""
    
    MIN_CONFIG_SIZE = 32
    MAX_PUBLIC_KEY_SIZE = 2048
    
    def parse_config(self, config_bytes: bytes) -> ECHConfig:
        """Parse ECH configuration with validation"""
        if len(config_bytes) < self.MIN_CONFIG_SIZE:
            raise ECHError(f"Configuration too small: {len(config_bytes)} bytes")
        
        try:
            # Parse header fields
            version_raw = int.from_bytes(config_bytes[0:2], 'big')
            version = ECHVersion(version_raw)
            
            config_id = config_bytes[2]
            kem_id = int.from_bytes(config_bytes[3:5], 'big')
            public_key_length = int.from_bytes(config_bytes[5:7], 'big')
            
            if public_key_length > self.MAX_PUBLIC_KEY_SIZE:
                raise ECHError(f"Public key too large: {public_key_length}")
            
            public_key = config_bytes[7:7+public_key_length]
            
            # Parse cipher suites
            offset = 7 + public_key_length
            cipher_suites_length = int.from_bytes(config_bytes[offset:offset+2], 'big')
            offset += 2
            
            cipher_suites = []
            for i in range(0, cipher_suites_length, 4):
                if offset + i + 4 <= len(config_bytes):
                    kdf_id = int.from_bytes(config_bytes[offset+i:offset+i+2], 'big')
                    aead_id = int.from_bytes(config_bytes[offset+i+2:offset+i+4], 'big')
                    
                    # Map to our cipher suite enum
                    if (kdf_id, aead_id) == (0x0001, 0x0001):
                        cipher_suites.append(CipherSuite.AES_128_GCM_SHA256)
                    elif (kdf_id, aead_id) == (0x0003, 0x0003):
                        cipher_suites.append(CipherSuite.CHACHA20_POLY1305_SHA256)
            
            if not cipher_suites:
                raise ECHError("No supported cipher suites found")
            
            return ECHConfig(
                version=version,
                config_id=config_id,
                kem_id=kem_id,
                public_key=public_key,
                cipher_suites=cipher_suites
            )
            
        except (ValueError, IndexError) as e:
            raise ECHError(f"Configuration parsing failed: {e}")
    
    def validate_config(self, config: ECHConfig) -> bool:
        """Validate ECH configuration integrity"""
        try:
            # Version validation
            if config.version not in ECHVersion:
                return False
            
            # Key validation
            if len(config.public_key) == 0:
                return False
            
            # Cipher suite validation
            if not config.cipher_suites:
                return False
            
            return True
        except Exception:
            return False

class ECHKeyDeriver:
    """ECH key derivation implementation"""
    
    def derive_ech_keys(self, config: ECHConfig, client_random: bytes) -> Dict[str, bytes]:
        """Derive ECH encryption keys"""
        try:
            # Simplified key derivation for demonstration
            # In production, use proper HKDF with ECH specification
            import hashlib
            
            key_material = config.public_key + client_random
            master_key = hashlib.sha256(key_material).digest()
            
            return {
                'encryption_key': master_key[:16],
                'authentication_key': master_key[16:32],
                'nonce': client_random[:12]
            }
        except Exception as e:
            raise ECHError(f"Key derivation failed: {e}")

# ============================================================================
# ENHANCED NOISE PROTOCOL (Extends Existing)
# ============================================================================

class ECHEnhancedNoiseHandshake:
    """Enhanced Noise XK handshake with ECH support"""
    
    def __init__(
        self, 
        base_handshake: NoiseHandshakeInterface,
        ech_config: Optional[ECHConfig] = None,
        config_parser: Optional[ECHConfigParser] = None
    ):
        """Initialize with dependency injection - weak coupling"""
        self._base_handshake = base_handshake
        self._ech_config = ech_config
        self._config_parser = config_parser or ECHConfigParserImpl()
        self._key_deriver = ECHKeyDeriver()
        self._metrics = SecurityMetrics(
            handshake_duration_ms=0.0,
            ech_success_rate=0.0,
            fallback_rate=0.0,
            threat_detections=0
        )
    
    @property
    def ech_enabled(self) -> bool:
        """Check if ECH is enabled"""
        return self._ech_config is not None
    
    async def initiate_handshake(self, peer_id: str) -> HandshakeResult:
        """Enhanced handshake with ECH support and fallback"""
        start_time = time.time()
        
        try:
            if self.ech_enabled:
                result = await self._ech_enhanced_handshake(peer_id)
                if result.success:
                    self._update_metrics(start_time, ech_success=True)
                    return result
                
                logger.warning(f"ECH handshake failed for {peer_id}, falling back")
            
            # Fallback to standard handshake
            result = await self._base_handshake.initiate_handshake(peer_id)
            result.ech_enabled = False
            self._update_metrics(start_time, ech_success=False, fallback=True)
            
            return result
            
        except Exception as e:
            error_result = HandshakeResult(
                success=False,
                connection_id="",
                error_message=str(e)
            )
            self._update_metrics(start_time, ech_success=False, error=True)
            return error_result
    
    async def _ech_enhanced_handshake(self, peer_id: str) -> HandshakeResult:
        """ECH-specific handshake implementation"""
        if not self._ech_config:
            raise ECHError("ECH config not available")
        
        try:
            # Generate client random for ECH key derivation
            import secrets
            client_random = secrets.token_bytes(32)
            
            # Derive ECH keys
            ech_keys = self._key_deriver.derive_ech_keys(self._ech_config, client_random)
            
            # Create encrypted Client Hello Inner (simplified)
            encrypted_sni = self._encrypt_sni(peer_id, ech_keys)
            
            # Construct handshake message
            handshake_data = {
                'type': 'NOISE_XK_ECH_INIT',
                'encrypted_sni': encrypted_sni,
                'config_id': self._ech_config.config_id,
                'client_random': client_random
            }
            
            # Simulate handshake success for architecture demonstration
            # In production, this would involve actual network communication
            connection_id = f"ech_conn_{peer_id}_{int(time.time())}"
            
            return HandshakeResult(
                success=True,
                connection_id=connection_id,
                encryption_key=ech_keys['encryption_key'],
                decryption_key=ech_keys['encryption_key'],  # Simplified
                ech_enabled=True,
                forward_secrecy=True,
                cipher_suite=self._ech_config.cipher_suites[0]
            )
            
        except Exception as e:
            raise ECHError(f"ECH handshake failed: {e}")
    
    def _encrypt_sni(self, sni: str, ech_keys: Dict[str, bytes]) -> bytes:
        """Encrypt Server Name Indication"""
        try:
            # Simplified ECH SNI encryption
            # In production, use proper AEAD encryption
            import hashlib
            
            sni_bytes = sni.encode('utf-8')
            key = ech_keys['encryption_key']
            nonce = ech_keys['nonce']
            
            # XOR encryption for demonstration
            encrypted = bytes(a ^ b for a, b in zip(sni_bytes, key * ((len(sni_bytes) // len(key)) + 1)))
            return nonce + encrypted
            
        except Exception as e:
            raise ECHError(f"SNI encryption failed: {e}")
    
    def _update_metrics(self, start_time: float, ech_success: bool = False, 
                       fallback: bool = False, error: bool = False) -> None:
        """Update performance and security metrics"""
        duration = (time.time() - start_time) * 1000
        self._metrics.handshake_duration_ms = duration
        
        if ech_success:
            self._metrics.ech_success_rate = min(1.0, self._metrics.ech_success_rate + 0.1)
        elif fallback:
            self._metrics.fallback_rate = min(1.0, self._metrics.fallback_rate + 0.1)

# ============================================================================
# TRANSPORT INTEGRATION (Zero-Breaking-Change Decorator)
# ============================================================================

class ECHTransportWrapper:
    """ECH-aware transport wrapper - decorator pattern"""
    
    def __init__(
        self, 
        base_transport: TransportInterface,
        security_monitor: Optional[SecurityMonitorInterface] = None
    ):
        """Initialize with base transport - dependency injection"""
        self._base_transport = base_transport
        self._security_monitor = security_monitor
        self._ech_configs: Dict[str, ECHConfig] = {}
        self._enhanced_handshakes: Dict[str, ECHEnhancedNoiseHandshake] = {}
        
    def register_ech_config(self, peer_id: str, config: ECHConfig) -> None:
        """Register ECH configuration for peer - named parameters"""
        self._ech_configs[peer_id] = config
        logger.info(f"Registered ECH config for {peer_id}")
    
    async def establish_connection(self, peer_id: str, **options) -> Any:
        """Enhanced connection establishment with ECH"""
        try:
            # Check if ECH should be used
            use_ech = options.get('use_ech', True) and peer_id in self._ech_configs
            
            if use_ech:
                # Create ECH-enhanced handshake
                base_handshake = options.get('handshake_provider')
                if base_handshake:
                    ech_config = self._ech_configs[peer_id]
                    enhanced_handshake = ECHEnhancedNoiseHandshake(
                        base_handshake=base_handshake,
                        ech_config=ech_config
                    )
                    
                    # Attempt ECH handshake
                    handshake_result = await enhanced_handshake.initiate_handshake(peer_id)
                    
                    if handshake_result.success and handshake_result.ech_enabled:
                        # ECH success - enhance connection options
                        enhanced_options = {
                            **options,
                            'encryption_level': 'ECH_ENHANCED',
                            'handshake_result': handshake_result
                        }
                        
                        if self._security_monitor:
                            self._security_monitor.record_handshake_attempt(peer_id, True)
                        
                        return await self._base_transport.establish_connection(
                            peer_id, **enhanced_options
                        )
            
            # Fallback to standard connection
            if self._security_monitor:
                self._security_monitor.record_handshake_attempt(peer_id, True)
            
            return await self._base_transport.establish_connection(peer_id, **options)
            
        except Exception as e:
            logger.error(f"Enhanced connection establishment failed for {peer_id}: {e}")
            if self._security_monitor:
                self._security_monitor.record_handshake_attempt(peer_id, False)
            raise
    
    async def send_message(self, connection: Any, message: bytes) -> bool:
        """Enhanced message sending with ECH awareness"""
        # Check if connection has ECH encryption
        if hasattr(connection, 'encryption_level') and connection.encryption_level == 'ECH_ENHANCED':
            # Apply additional ECH-specific message processing
            logger.debug("Sending message through ECH-enhanced connection")
        
        return await self._base_transport.send_message(connection, message)
    
    async def close_connection(self, connection: Any) -> None:
        """Connection cleanup with ECH state management"""
        # Clean up ECH-specific state if needed
        return await self._base_transport.close_connection(connection)
    
    def get_ech_status(self) -> Dict[str, Any]:
        """Get ECH status information"""
        return {
            'registered_configs': len(self._ech_configs),
            'ech_peers': list(self._ech_configs.keys()),
            'enhanced_handshakes': len(self._enhanced_handshakes)
        }

# ============================================================================
# SECURITY INTEGRATION (Extension Pattern)
# ============================================================================

class ECHSecurityExtension:
    """ECH security extensions for existing security manager"""
    
    def __init__(self, base_security_manager):
        """Initialize with existing security manager"""
        self._base_security = base_security_manager
        self._ech_metrics = {}
        self._threat_patterns = []
    
    def enhance_peer_authentication(self, peer_id: str, auth_data: Dict[str, Any]) -> bool:
        """Enhanced authentication with ECH capabilities"""
        # Use existing authentication
        base_result = self._base_security.authenticate_peer(peer_id, auth_data)
        
        if base_result:
            # Check for ECH capability
            if 'ech_support' in auth_data:
                logger.info(f"Peer {peer_id} supports ECH")
                self._enable_ech_for_peer(peer_id)
        
        return base_result
    
    def detect_ech_threats(self, peer_id: str, handshake_data: Dict[str, Any]) -> List[str]:
        """ECH-specific threat detection"""
        threats = []
        
        # Check for ECH downgrade attacks
        if self._is_ech_capable(peer_id) and not handshake_data.get('ech_used', False):
            threats.append('POTENTIAL_ECH_DOWNGRADE')
        
        # Check for malformed ECH data
        if 'ech_config' in handshake_data:
            if not self._validate_ech_config(handshake_data['ech_config']):
                threats.append('MALFORMED_ECH_CONFIG')
        
        return threats
    
    def _enable_ech_for_peer(self, peer_id: str) -> None:
        """Enable ECH for specific peer"""
        # Implementation would update peer capabilities
        pass
    
    def _is_ech_capable(self, peer_id: str) -> bool:
        """Check if peer supports ECH"""
        # Implementation would check peer capabilities
        return False
    
    def _validate_ech_config(self, config_data: Any) -> bool:
        """Validate ECH configuration data"""
        # Implementation would validate ECH config
        return True

# ============================================================================
# FACTORY AND COORDINATION (Dependency Injection)
# ============================================================================

class ECHSystemFactory:
    """Factory for creating ECH system components"""
    
    @staticmethod
    def create_enhanced_transport(
        base_transport: TransportInterface,
        security_monitor: Optional[SecurityMonitorInterface] = None
    ) -> ECHTransportWrapper:
        """Create ECH-enhanced transport wrapper"""
        return ECHTransportWrapper(base_transport, security_monitor)
    
    @staticmethod
    def create_ech_handshake(
        base_handshake: NoiseHandshakeInterface,
        ech_config_bytes: Optional[bytes] = None
    ) -> ECHEnhancedNoiseHandshake:
        """Create ECH-enhanced handshake"""
        config_parser = ECHConfigParserImpl()
        ech_config = None
        
        if ech_config_bytes:
            try:
                ech_config = config_parser.parse_config(ech_config_bytes)
            except ECHError as e:
                logger.warning(f"ECH config parsing failed: {e}")
        
        return ECHEnhancedNoiseHandshake(
            base_handshake=base_handshake,
            ech_config=ech_config,
            config_parser=config_parser
        )
    
    @staticmethod
    def create_security_extension(base_security_manager) -> ECHSecurityExtension:
        """Create ECH security extension"""
        return ECHSecurityExtension(base_security_manager)

# ============================================================================
# CONTEXT MANAGER FOR RESOURCE MANAGEMENT
# ============================================================================

@asynccontextmanager
async def ech_enhanced_system(
    base_transport: TransportInterface,
    base_handshake: NoiseHandshakeInterface,
    security_manager,
    ech_configs: Dict[str, bytes]
):
    """Context manager for ECH-enhanced system setup"""
    try:
        # Create enhanced components
        transport_wrapper = ECHSystemFactory.create_enhanced_transport(base_transport)
        security_extension = ECHSystemFactory.create_security_extension(security_manager)
        
        # Register ECH configurations
        config_parser = ECHConfigParserImpl()
        for peer_id, config_bytes in ech_configs.items():
            try:
                ech_config = config_parser.parse_config(config_bytes)
                transport_wrapper.register_ech_config(peer_id, ech_config)
            except ECHError as e:
                logger.warning(f"Failed to register ECH config for {peer_id}: {e}")
        
        logger.info("ECH-enhanced system initialized successfully")
        
        yield {
            'transport': transport_wrapper,
            'security': security_extension,
            'factory': ECHSystemFactory
        }
        
    finally:
        logger.info("ECH-enhanced system cleanup completed")

# ============================================================================
# EXAMPLE USAGE AND INTEGRATION
# ============================================================================

async def example_integration():
    """Example of how to integrate ECH with existing systems"""
    
    # Mock existing components
    class MockTransport:
        async def establish_connection(self, peer_id: str, **options):
            return f"connection_to_{peer_id}"
        
        async def send_message(self, connection, message: bytes):
            return True
            
        async def close_connection(self, connection):
            pass
    
    class MockHandshake:
        async def initiate_handshake(self, peer_id: str):
            return HandshakeResult(
                success=True,
                connection_id=f"handshake_{peer_id}",
                forward_secrecy=True
            )
        
        async def process_handshake_response(self, response: bytes):
            return HandshakeResult(success=True, connection_id="test")
    
    class MockSecurityManager:
        def authenticate_peer(self, peer_id: str, auth_data: Dict[str, Any]):
            return True
    
    # Example ECH configuration bytes (mock)
    mock_ech_config = bytes(range(64))  # Simplified for example
    
    # Integration example
    base_transport = MockTransport()
    base_handshake = MockHandshake()
    security_manager = MockSecurityManager()
    
    ech_configs = {
        "peer1": mock_ech_config,
        "peer2": mock_ech_config
    }
    
    async with ech_enhanced_system(
        base_transport, 
        base_handshake, 
        security_manager, 
        ech_configs
    ) as system:
        
        transport = system['transport']
        
        # Establish ECH-enhanced connection
        connection = await transport.establish_connection("peer1", use_ech=True)
        print(f"Established connection: {connection}")
        
        # Send message through enhanced transport
        success = await transport.send_message(connection, b"Hello, ECH!")
        print(f"Message sent: {success}")
        
        # Get ECH status
        status = transport.get_ech_status()
        print(f"ECH Status: {status}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_integration())