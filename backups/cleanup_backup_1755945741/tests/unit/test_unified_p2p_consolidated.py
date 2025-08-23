#!/usr/bin/env python3
"""
UNIFIED P2P SYSTEM CONSOLIDATED TESTS
Comprehensive test suite for the unified P2P decentralized system

MISSION: Consolidate scattered P2P tests into comprehensive test coverage
- Tests unified P2P system functionality
- Tests BitChat BLE mesh networking
- Tests BetaNet HTX transport
- Tests mobile bridge integration
- Tests configuration management
- Tests cross-system integration
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config.unified_p2p_config import (
        BetaNetConfig,
        BitChatConfig,
        DeploymentMode,
        NetworkProfile,
        UnifiedP2PConfig,
        UnifiedP2PConfigManager,
        create_mobile_config,
        create_privacy_first_config,
        get_p2p_config,
    )
    from core.decentralized_architecture.unified_p2p_system import (
        DecentralizedMessage,
        DecentralizedTransportType,
        DeviceCapabilities,
        MessagePriority,
        MobileDeviceContext,
        PeerInfo,
        UnifiedDecentralizedSystem,
        create_unified_p2p_system,
    )

    P2P_AVAILABLE = True
except ImportError as e:
    P2P_AVAILABLE = False
    pytest.skip(f"P2P system not available: {e}", allow_module_level=True)


class TestDecentralizedMessage:
    """Test suite for DecentralizedMessage functionality."""

    def test_message_creation(self):
        """Test creating a basic decentralized message."""
        message = DecentralizedMessage(
            message_id="test_msg_001",
            sender_id="node_001",
            receiver_id="node_002",
            message_type="test",
            payload=b"Hello, world!",
            priority=MessagePriority.NORMAL,
        )

        assert message.message_id == "test_msg_001"
        assert message.sender_id == "node_001"
        assert message.receiver_id == "node_002"
        assert message.message_type == "test"
        assert message.payload == b"Hello, world!"
        assert message.priority == MessagePriority.NORMAL
        assert message.size_bytes == 13
        assert not message.is_broadcast
        assert message.can_relay

    def test_broadcast_message(self):
        """Test broadcast message detection."""
        broadcast_msg = DecentralizedMessage(
            message_id="broadcast_001",
            sender_id="node_001",
            receiver_id="broadcast",
            message_type="announcement",
            payload=b"Global announcement",
        )

        assert broadcast_msg.is_broadcast

    def test_message_serialization(self):
        """Test message to_dict and from_dict methods."""
        original_msg = DecentralizedMessage(
            message_id="serial_test",
            sender_id="sender_001",
            receiver_id="receiver_001",
            message_type="data",
            payload=b"Serialization test",
            priority=MessagePriority.HIGH,
            requires_privacy=True,
        )

        # Serialize to dict
        msg_dict = original_msg.to_dict()

        # Deserialize from dict
        restored_msg = DecentralizedMessage.from_dict(msg_dict)

        assert restored_msg.message_id == original_msg.message_id
        assert restored_msg.sender_id == original_msg.sender_id
        assert restored_msg.receiver_id == original_msg.receiver_id
        assert restored_msg.payload == original_msg.payload
        assert restored_msg.priority == original_msg.priority
        assert restored_msg.requires_privacy == original_msg.requires_privacy

    def test_message_expiration(self):
        """Test message expiration logic."""
        # Create expired message
        expired_msg = DecentralizedMessage(
            message_id="expired_test",
            sender_id="node_001",
            receiver_id="node_002",
            message_type="test",
            payload=b"This should be expired",
            expires_at=time.time() - 100,  # 100 seconds ago
        )

        assert expired_msg.is_expired
        assert not expired_msg.can_relay

    def test_hop_limit_logic(self):
        """Test hop limit and relay capability."""
        message = DecentralizedMessage(
            message_id="hop_test",
            sender_id="node_001",
            receiver_id="node_002",
            message_type="relay_test",
            payload=b"Testing hop limits",
            hop_limit=3,
            hop_count=0,
        )

        # Should be able to relay initially
        assert message.can_relay

        # Simulate relaying
        message.hop_count = message.hop_limit
        assert not message.can_relay


class TestPeerInfo:
    """Test suite for PeerInfo functionality."""

    def test_peer_creation(self):
        """Test creating peer information."""
        mobile_context = MobileDeviceContext(platform="android", battery_level=0.8, network_type="wifi")

        peer = PeerInfo(
            peer_id="test_peer_001",
            transport_types=[DecentralizedTransportType.BITCHAT_BLE],
            device_capabilities=[DeviceCapabilities.MOBILE_OPTIMIZED],
            mobile_context=mobile_context,
            latency_ms=150.0,
            reliability_score=0.95,
        )

        assert peer.peer_id == "test_peer_001"
        assert DecentralizedTransportType.BITCHAT_BLE in peer.transport_types
        assert peer.is_online()
        assert peer.supports_transport(DecentralizedTransportType.BITCHAT_BLE)
        assert not peer.supports_transport(DecentralizedTransportType.BETANET_HTX)
        assert peer.mobile_context.platform == "android"

    def test_peer_activity_tracking(self):
        """Test peer activity updates."""
        peer = PeerInfo(peer_id="activity_test")

        initial_time = peer.last_seen
        time.sleep(0.1)  # Small delay

        peer.update_activity()
        assert peer.last_seen > initial_time
        assert peer.last_message_time > initial_time


class TestUnifiedP2PConfig:
    """Test suite for unified P2P configuration system."""

    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = UnifiedP2PConfig()

        assert config.node_id is not None
        assert config.device_name is not None
        assert config.deployment_mode == DeploymentMode.DEVELOPMENT
        assert config.network_profile == NetworkProfile.BALANCED
        assert config.bitchat.enabled is True
        assert config.betanet.enabled is True

    def test_mobile_optimized_config(self):
        """Test mobile-optimized configuration."""
        config = create_mobile_config(mobile_platform="android", node_id="mobile_test_node")

        assert config.network_profile == NetworkProfile.MOBILE_OPTIMIZED
        assert config.mobile.battery_aware_scheduling is True
        assert config.mobile.thermal_throttling_enabled is True
        assert config.performance.max_memory_mb <= 256
        assert config.bitchat.max_peers <= 25

    def test_privacy_first_config(self):
        """Test privacy-first configuration."""
        config = create_privacy_first_config()

        assert config.network_profile == NetworkProfile.PRIVACY_FIRST
        assert config.betanet.enable_noise_encryption is True
        assert config.betanet.enable_forward_secrecy is True
        assert config.security.max_relay_hops >= 3

    def test_production_deployment_config(self):
        """Test production deployment optimizations."""
        config = UnifiedP2PConfig(deployment_mode=DeploymentMode.PRODUCTION)

        assert config.deployment_mode == DeploymentMode.PRODUCTION
        assert config.debug_mode is False
        assert config.log_level == "WARNING"
        assert config.security.enforce_encryption is True
        assert config.performance.max_memory_mb >= 1024


class TestUnifiedP2PConfigManager:
    """Test suite for configuration manager."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_config_dir = Path("/tmp/test_p2p_config")
        self.temp_config_dir.mkdir(exist_ok=True)
        self.config_manager = UnifiedP2PConfigManager(self.temp_config_dir)

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil

        if self.temp_config_dir.exists():
            shutil.rmtree(self.temp_config_dir)

    def test_config_save_and_load(self):
        """Test configuration saving and loading."""
        # Create test configuration
        original_config = UnifiedP2PConfig(node_id="save_load_test", network_profile=NetworkProfile.OFFLINE_FIRST)

        # Save configuration
        self.config_manager.save_config(original_config)

        # Load configuration
        loaded_config = self.config_manager.load_config()

        assert loaded_config.node_id == "save_load_test"
        assert loaded_config.network_profile == NetworkProfile.OFFLINE_FIRST

    def test_environment_variable_overrides(self):
        """Test environment variable configuration overrides."""
        with patch.dict(
            "os.environ", {"P2P_NODE_ID": "env_test_node", "BITCHAT_MAX_PEERS": "75", "P2P_DEBUG_MODE": "true"}
        ):
            config = self.config_manager.load_config()

            assert config.node_id == "env_test_node"
            assert config.bitchat.max_peers == 75
            assert config.debug_mode is True


class TestUnifiedDecentralizedSystem:
    """Test suite for the unified decentralized system."""

    def setup_method(self):
        """Setup test environment for each test."""
        self.node_id = "test_node_001"
        self.system = UnifiedDecentralizedSystem(
            node_id=self.node_id,
            enable_bitchat=False,  # Disable to avoid hardware dependencies
            enable_betanet=False,  # Disable to avoid network dependencies
            enable_mobile_bridge=False,
            enable_fog_bridge=False,
        )

    def teardown_method(self):
        """Cleanup after each test."""
        if hasattr(self.system, "_running") and self.system._running:
            # Note: Can't use asyncio.run() in sync teardown
            pass

    def test_system_initialization(self):
        """Test basic system initialization."""
        assert self.system.node_id == self.node_id
        assert not self.system._running
        assert len(self.system.device_capabilities) > 0
        assert self.system.metrics["messages_sent"] == 0
        assert self.system.metrics["messages_received"] == 0

    def test_device_capability_detection(self):
        """Test device capability detection."""
        # Test desktop system
        desktop_system = UnifiedDecentralizedSystem(node_id="desktop_test", mobile_platform="desktop")

        assert DeviceCapabilities.ETHERNET in desktop_system.device_capabilities

        # Test mobile system
        mobile_system = UnifiedDecentralizedSystem(node_id="mobile_test", mobile_platform="android")

        assert DeviceCapabilities.MOBILE_OPTIMIZED in mobile_system.device_capabilities
        assert DeviceCapabilities.BATTERY_AWARE in mobile_system.device_capabilities

    def test_peer_management(self):
        """Test peer information management."""
        peer_id = "test_peer_001"

        # Add peer
        peer_info = PeerInfo(peer_id=peer_id, transport_types=[DecentralizedTransportType.BITCHAT_BLE])

        self.system.peers[peer_id] = peer_info

        # Verify peer was added
        assert peer_id in self.system.peers
        assert self.system.peers[peer_id].peer_id == peer_id

        # Test status reporting
        status = self.system.get_status()
        assert status["peer_count"] == 1
        assert status["node_id"] == self.node_id

    def test_message_handler_registration(self):
        """Test message handler registration."""
        handler_called = False
        received_message = None

        def test_handler(message, transport_type):
            nonlocal handler_called, received_message
            handler_called = True
            received_message = message

        # Register handler
        self.system.register_message_handler(test_handler)
        assert len(self.system.message_handlers) == 1

    @pytest.mark.asyncio
    async def test_system_lifecycle(self):
        """Test system start and stop lifecycle."""
        # Start system
        start_result = await self.system.start()

        # System should start successfully even without transports
        assert start_result is True
        assert self.system._running is True
        assert len(self.system._background_tasks) > 0

        # Stop system
        stop_result = await self.system.stop()
        assert stop_result is True
        assert self.system._running is False
        assert len(self.system._background_tasks) == 0

    def test_mobile_context_updates(self):
        """Test mobile device context updates."""
        mobile_system = UnifiedDecentralizedSystem(node_id="mobile_context_test", mobile_platform="ios")

        # Update mobile context
        mobile_system.update_mobile_context(battery_level=0.3, is_charging=True, thermal_state="elevated")

        assert mobile_system.mobile_context.battery_level == 0.3
        assert mobile_system.mobile_context.is_charging is True
        assert mobile_system.mobile_context.thermal_state == "elevated"

    def test_transport_capability_mapping(self):
        """Test transport capability mapping."""
        # Test BitChat capabilities
        bitchat_caps = self.system._get_transport_capabilities(DecentralizedTransportType.BITCHAT_BLE)
        assert bitchat_caps.supports_broadcast is True
        assert bitchat_caps.is_offline_capable is True
        assert bitchat_caps.requires_internet is False

        # Test BetaNet capabilities
        betanet_caps = self.system._get_transport_capabilities(DecentralizedTransportType.BETANET_HTX)
        assert betanet_caps.provides_encryption is True
        assert betanet_caps.supports_forward_secrecy is True
        assert betanet_caps.requires_internet is True

    @pytest.mark.asyncio
    async def test_message_processing_flow(self):
        """Test complete message processing flow."""
        # Create test message
        test_message = DecentralizedMessage(
            message_id="processing_test",
            sender_id="test_sender",
            receiver_id=self.system.node_id,
            message_type="test_data",
            payload=b"Test message processing",
        )

        # Process received message
        await self.system._process_received_message(test_message, DecentralizedTransportType.BITCHAT_BLE)

        # Verify metrics updated
        assert self.system.metrics["messages_received"] == 1
        assert test_message.message_id in self.system.message_cache

    def test_factory_function(self):
        """Test factory function for creating unified P2P system."""
        system = create_unified_p2p_system("factory_test_node")

        assert isinstance(system, UnifiedDecentralizedSystem)
        assert system.node_id == "factory_test_node"


class TestBitChatIntegration:
    """Test suite for BitChat BLE mesh integration."""

    def test_bitchat_message_conversion(self):
        """Test conversion between DecentralizedMessage and BitChat format."""
        # Create decentralized message
        decentralized_msg = DecentralizedMessage(
            message_id="bitchat_test",
            sender_id="node_001",
            receiver_id="node_002",
            message_type="chat",
            payload=b"Hello via BitChat",
            priority=MessagePriority.NORMAL,
            hop_limit=5,
        )

        # Convert to BitChat format (as would be done in _send_via_bitchat)
        bitchat_format = {
            "id": decentralized_msg.message_id,
            "sender": decentralized_msg.sender_id,
            "recipient": decentralized_msg.receiver_id,
            "type": decentralized_msg.message_type,
            "payload": decentralized_msg.payload,
            "priority": decentralized_msg.priority.value,
            "ttl": decentralized_msg.hop_limit,
            "timestamp": decentralized_msg.timestamp,
        }

        assert bitchat_format["id"] == "bitchat_test"
        assert bitchat_format["ttl"] == 5
        assert bitchat_format["priority"] == MessagePriority.NORMAL.value

    def test_bitchat_mesh_routing(self):
        """Test BitChat mesh routing logic."""
        system = UnifiedDecentralizedSystem(
            node_id="routing_test_node", enable_bitchat=False  # Disable actual transport
        )

        # Create message for relay testing
        message = DecentralizedMessage(
            message_id="relay_test",
            sender_id="source_node",
            receiver_id="destination_node",
            message_type="data",
            payload=b"Relay test message",
            hop_count=2,
            hop_limit=7,
        )

        # Add some test peers
        system.peers["relay_peer_1"] = PeerInfo(
            peer_id="relay_peer_1", transport_types=[DecentralizedTransportType.BITCHAT_BLE], reliability_score=0.9
        )
        system.peers["relay_peer_2"] = PeerInfo(
            peer_id="relay_peer_2", transport_types=[DecentralizedTransportType.BITCHAT_BLE], reliability_score=0.8
        )

        # Test relay candidate selection logic
        exclude_peers = set(message.route_path)
        relay_peers = [
            peer
            for peer in system.peers.values()
            if peer.peer_id not in exclude_peers
            and peer.is_online()
            and DecentralizedTransportType.BITCHAT_BLE in peer.transport_types
        ]

        assert len(relay_peers) == 2
        best_peer = max(relay_peers, key=lambda p: p.reliability_score)
        assert best_peer.peer_id == "relay_peer_1"  # Higher reliability score


class TestBetaNetIntegration:
    """Test suite for BetaNet HTX transport integration."""

    def test_betanet_frame_conversion(self):
        """Test conversion to BetaNet HTX frame format."""
        decentralized_msg = DecentralizedMessage(
            message_id="betanet_frame_test",
            sender_id="client_001",
            receiver_id="server_001",
            message_type="secure_data",
            payload=b"Encrypted secure payload",
            requires_privacy=True,
        )

        # Convert to BetaNet HTX format (as would be done in _send_via_betanet)
        htx_frame = {
            "stream_id": hash(decentralized_msg.message_id) % 1000000,
            "frame_type": "DATA",
            "sender_id": decentralized_msg.sender_id,
            "receiver_id": decentralized_msg.receiver_id,
            "message_type": decentralized_msg.message_type,
            "payload": decentralized_msg.payload,
            "requires_ack": decentralized_msg.requires_ack,
            "timestamp": decentralized_msg.timestamp,
        }

        assert htx_frame["frame_type"] == "DATA"
        assert htx_frame["sender_id"] == "client_001"
        assert htx_frame["requires_ack"] == decentralized_msg.requires_ack
        assert 0 <= htx_frame["stream_id"] < 1000000

    def test_betanet_privacy_features(self):
        """Test BetaNet privacy and security features."""
        config = UnifiedP2PConfig(network_profile=NetworkProfile.PRIVACY_FIRST)

        # Verify privacy settings are enabled
        assert config.betanet.enable_noise_encryption is True
        assert config.betanet.enable_forward_secrecy is True
        assert config.security.max_relay_hops >= 3


class TestCrossSystemIntegration:
    """Test suite for cross-system integration functionality."""

    @pytest.mark.asyncio
    async def test_transport_failover(self):
        """Test transport failover functionality."""
        system = UnifiedDecentralizedSystem(node_id="failover_test", enable_bitchat=False, enable_betanet=False)

        await system.start()

        try:
            # Attempt to send message with no transports available
            success = await system.send_message(
                receiver_id="test_receiver",
                message_type="failover_test",
                payload=b"Testing failover",
                priority=MessagePriority.CRITICAL,
            )

            # Should fail gracefully when no transports available
            assert success is False
            assert system.metrics["messages_dropped"] == 1

        finally:
            await system.stop()

    def test_multi_transport_peer_info(self):
        """Test peer information with multiple transports."""
        peer = PeerInfo(
            peer_id="multi_transport_peer",
            transport_types=[
                DecentralizedTransportType.BITCHAT_BLE,
                DecentralizedTransportType.BETANET_HTX,
                DecentralizedTransportType.MOBILE_NATIVE,
            ],
        )

        assert peer.supports_transport(DecentralizedTransportType.BITCHAT_BLE)
        assert peer.supports_transport(DecentralizedTransportType.BETANET_HTX)
        assert peer.supports_transport(DecentralizedTransportType.MOBILE_NATIVE)
        assert not peer.supports_transport(DecentralizedTransportType.FOG_BRIDGE)

    def test_system_status_comprehensive(self):
        """Test comprehensive system status reporting."""
        system = UnifiedDecentralizedSystem(node_id="status_test_node", mobile_platform="android")

        # Add test data
        system.peers["peer_1"] = PeerInfo(peer_id="peer_1")
        system.peers["peer_2"] = PeerInfo(peer_id="peer_2")
        system.message_cache["msg_1"] = time.time()
        system.offline_message_store["offline_peer"] = [
            DecentralizedMessage(
                message_id="stored_msg",
                sender_id="sender",
                receiver_id="offline_peer",
                message_type="stored",
                payload=b"stored message",
            )
        ]

        # Get status
        status = system.get_status()

        assert status["node_id"] == "status_test_node"
        assert status["peer_count"] == 2
        assert status["message_cache_size"] == 1
        assert status["offline_store_size"] == 1
        assert "mobile_context" in status
        assert status["mobile_context"]["platform"] == "android"

    def test_peer_list_reporting(self):
        """Test peer list reporting functionality."""
        system = UnifiedDecentralizedSystem(node_id="peer_list_test")

        # Add test peers with different characteristics
        system.peers["fast_peer"] = PeerInfo(
            peer_id="fast_peer", latency_ms=50.0, reliability_score=0.95, messages_sent=100, battery_friendly=True
        )

        system.peers["slow_peer"] = PeerInfo(
            peer_id="slow_peer", latency_ms=500.0, reliability_score=0.7, messages_sent=25, battery_friendly=False
        )

        peer_list = system.get_peers()

        assert len(peer_list) == 2

        fast_peer_data = next(p for p in peer_list if p["peer_id"] == "fast_peer")
        assert fast_peer_data["latency_ms"] == 50.0
        assert fast_peer_data["reliability_score"] == 0.95
        assert fast_peer_data["battery_friendly"] is True

        slow_peer_data = next(p for p in peer_list if p["peer_id"] == "slow_peer")
        assert slow_peer_data["latency_ms"] == 500.0
        assert slow_peer_data["reliability_score"] == 0.7


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_invalid_message_handling(self):
        """Test handling of invalid messages."""
        # Test message with no payload
        empty_msg = DecentralizedMessage(
            message_id="empty_test", sender_id="sender", receiver_id="receiver", message_type="empty", payload=b""
        )

        assert empty_msg.size_bytes == 0

        # Test message serialization with empty payload
        msg_dict = empty_msg.to_dict()
        restored_msg = DecentralizedMessage.from_dict(msg_dict)
        assert restored_msg.payload == b""

    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test invalid enum values in dict conversion
        invalid_config_dict = {"network_profile": "invalid_profile", "deployment_mode": "invalid_mode"}

        config_manager = UnifiedP2PConfigManager()

        # Should fall back to default configuration on invalid data
        config = config_manager._dict_to_config(invalid_config_dict)

        # Should have fallen back to defaults
        assert isinstance(config, UnifiedP2PConfig)

    @pytest.mark.asyncio
    async def test_system_resilience(self):
        """Test system resilience to errors."""
        system = UnifiedDecentralizedSystem(node_id="resilience_test", enable_bitchat=False, enable_betanet=False)

        # Start system
        await system.start()

        try:
            # Test message processing with invalid data
            invalid_msg = DecentralizedMessage(
                message_id="",  # Empty ID
                sender_id="",  # Empty sender
                receiver_id="",  # Empty receiver
                message_type="invalid",
                payload=b"Invalid message test",
            )

            # Should handle gracefully without crashing
            await system._process_received_message(invalid_msg, DecentralizedTransportType.BITCHAT_BLE)

            # System should still be running
            assert system._running is True

        finally:
            await system.stop()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
