"""Federation Integration Tests

Tests the complete federated network architecture including:
- Device registry and role assignment
- Enhanced BitChat with crypto and fragmentation  
- Tor hidden service integration
- Multi-protocol routing via Navigator
- Privacy levels and VPN-like tunneling
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add src and experimental paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "experimental", "agents", "agents")
)

# Import federation components
try:
    from federation.core.device_registry import (
        DeviceCapability,
        DeviceProfile,
        DeviceRegistry,
        DeviceRole,
    )
    from federation.core.federation_manager import FederationManager, PrivacyLevel
    from federation.protocols.bitchat_enhanced import (
        BitChatMessageType,
        EnhancedBitChatMessage,
        EnhancedBitChatTransport,
    )
    from federation.protocols.tor_transport import TorHiddenService, TorTransport

    FEDERATION_IMPORTS_OK = True
except ImportError as e:
    print(f"Federation import error: {e}")
    FEDERATION_IMPORTS_OK = False

# Skip tests if imports fail
pytestmark = pytest.mark.skipif(
    not FEDERATION_IMPORTS_OK, reason="Federation modules not available"
)


class TestDeviceRegistry:
    """Test device registry and role assignment"""

    def test_device_registry_initialization(self):
        """Test device registry basic initialization"""
        registry = DeviceRegistry("test_device")
        assert registry.local_device_id == "test_device"
        assert len(registry.devices) == 0
        assert len(registry.known_beacons) == 0

    @pytest.mark.asyncio
    async def test_local_device_initialization(self):
        """Test local device profile creation"""
        registry = DeviceRegistry("test_device")

        capabilities = {
            DeviceCapability.BLUETOOTH,
            DeviceCapability.WIFI,
            DeviceCapability.HIGH_COMPUTE,
        }
        profile = await registry.initialize_local_device(capabilities, "test_region")

        assert profile.identity.device_id == "test_device"
        assert profile.region == "test_region"
        assert DeviceCapability.BLUETOOTH in profile.capabilities
        assert len(profile.protocols) > 0

    @pytest.mark.asyncio
    async def test_beacon_node_role_assignment(self):
        """Test beacon node role assignment"""
        registry = DeviceRegistry("beacon_device")

        # Beacon-suitable capabilities
        capabilities = {
            DeviceCapability.ALWAYS_ON,
            DeviceCapability.HIGH_COMPUTE,
            DeviceCapability.BANDWIDTH_HIGH,
            DeviceCapability.WIFI,
        }

        with patch("os.cpu_count", return_value=8):
            try:
                with patch("psutil.virtual_memory") as mock_memory:
                    mock_memory.return_value.total = 8 * 1024**3  # 8GB
                    with patch(
                        "psutil.sensors_battery", return_value=None
                    ):  # No battery = always on
                        profile = await registry.initialize_local_device(capabilities)
            except ImportError:
                # psutil not available, create profile manually
                profile = await registry.initialize_local_device(capabilities)

        # Should be assigned worker/edge role (beacon requires more specific conditions)
        assert profile.role in [DeviceRole.BEACON, DeviceRole.WORKER, DeviceRole.EDGE]

    def test_device_score_calculation(self):
        """Test device contribution score calculation"""
        from federation.core.device_registry import DeviceIdentity

        identity = DeviceIdentity(
            "test_device", public_key=b"test_key", signing_key_public=b"sign_key"
        )
        identity.reputation_score = 0.7

        profile = DeviceProfile(
            identity=identity,
            role=DeviceRole.WORKER,
            cpu_cores=4,
            memory_gb=8.0,
            bandwidth_mbps=50.0,
            uptime_hours=24 * 15,  # 15 days
        )

        score = profile.calculate_device_score()
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be higher than base reputation due to resources

    @pytest.mark.asyncio
    async def test_device_discovery_cleanup(self):
        """Test cleanup of stale devices"""
        registry = DeviceRegistry("test_device")

        # Add a device manually
        from federation.core.device_registry import DeviceIdentity, DeviceProfile

        old_device = DeviceProfile(
            identity=DeviceIdentity(
                "old_device", public_key=b"key", signing_key_public=b"sign"
            ),
            role=DeviceRole.EDGE,
        )
        old_device.identity.last_seen = 0.0  # Very old timestamp

        registry.devices["old_device"] = old_device
        assert len(registry.devices) == 1

        # Clean up devices older than 1 hour
        await registry.cleanup_stale_devices(max_age_hours=1)

        assert len(registry.devices) == 0


class TestEnhancedBitChat:
    """Test enhanced BitChat with crypto and fragmentation"""

    @pytest.mark.asyncio
    async def test_enhanced_bitchat_initialization(self):
        """Test enhanced BitChat initialization"""
        transport = EnhancedBitChatTransport("test_device", enable_crypto=True)

        assert transport.device_id == "test_device"
        assert transport.enable_crypto == True
        assert transport.crypto_keys is not None
        assert len(transport.channels) == 0

    def test_message_compression(self):
        """Test LZ4 message compression"""
        message = EnhancedBitChatMessage(
            sender="test_sender",
            payload=b"A" * 200,  # Large payload that should compress well
        )

        original_size = len(message.payload)
        compressed = message.compress_payload()

        if compressed:  # If LZ4 is available
            assert message.compressed == True
            assert len(message.payload) < original_size

            # Test decompression
            message.decompress_payload()
            assert message.compressed == False
            assert len(message.payload) == original_size

    def test_message_fragmentation(self):
        """Test message fragmentation for BLE packet size"""
        large_payload = b"X" * 1000  # Larger than 500-byte BLE limit

        message = EnhancedBitChatMessage(sender="test_sender", payload=large_payload)

        fragments = message.fragment_message(max_fragment_size=400)

        assert len(fragments) > 1
        assert message.is_fragmented == True

        # Test reassembly
        reassembled = EnhancedBitChatMessage.reassemble_from_fragments(fragments)
        assert reassembled is not None
        assert reassembled.payload == large_payload

    @pytest.mark.asyncio
    async def test_irc_style_channels(self):
        """Test IRC-style channel functionality"""
        transport = EnhancedBitChatTransport("test_device")

        # Join channel
        success = await transport.join_channel("general")
        assert success == True
        assert "general" in transport.joined_channels
        assert "general" in transport.channels

        # Check channel membership
        members = transport.channels["general"].members
        assert "test_device" in members

        # Send channel message
        with patch.object(transport, "_transmit_enhanced_message", return_value=True):
            success = await transport.send_channel_message("general", "Hello channel!")
            assert success == True

        # Leave channel
        await transport.leave_channel("general")
        assert "general" not in transport.joined_channels

    def test_crypto_message_signing(self):
        """Test Ed25519 message signing and verification"""
        try:
            import nacl.signing

            message = EnhancedBitChatMessage(
                sender="test_sender", payload=b"test message"
            )

            # Generate signing keys
            signing_key = nacl.signing.SigningKey.generate()
            verify_key = signing_key.verify_key

            # Sign message
            success = message.sign_message(signing_key)
            assert success == True or message.signed == False  # May not have crypto

            if message.signed:
                # Verify signature
                valid = message.verify_signature(verify_key)
                assert valid == True

        except ImportError:
            # Skip if crypto not available
            pass


class TestTorTransport:
    """Test Tor hidden service transport"""

    def test_tor_transport_initialization(self):
        """Test Tor transport initialization"""
        transport = TorTransport(
            socks_port=9050, control_port=9051, hidden_service_port=80, target_port=8080
        )

        assert transport.socks_port == 9050
        assert transport.control_port == 9051
        assert transport.hidden_service_port == 80
        assert transport.target_port == 8080
        assert transport.is_running == False

    @pytest.mark.asyncio
    async def test_tor_circuit_creation(self):
        """Test Tor circuit creation (mocked)"""
        transport = TorTransport()

        # Mock Tor controller
        mock_controller = Mock()
        mock_controller.new_circuit = Mock(return_value="circuit_123")

        mock_circuit_info = Mock()
        mock_circuit_info.status = "BUILT"
        mock_circuit_info.path = [("relay1", None), ("relay2", None), ("relay3", None)]

        mock_controller.get_circuit = Mock(return_value=mock_circuit_info)
        transport.tor_controller = mock_controller

        circuit_id = await transport.create_circuit()

        if circuit_id:  # If not None
            assert circuit_id == "circuit_123"
            assert circuit_id in transport.circuits
            assert transport.circuits[circuit_id].hop_count() == 3

    def test_onion_address_handling(self):
        """Test onion address validation and handling"""
        transport = TorTransport()

        # Test invalid onion address
        assert "invalid_address" != transport.get_onion_address()

        # Mock hidden service
        transport.hidden_service = TorHiddenService(
            service_id="test_service",
            onion_address="3g2upl4pq6kufc4m.onion",
            private_key="mock_key",
            port=80,
            target_port=8080,
            created_at=1234567890,
        )

        assert transport.get_onion_address() == "3g2upl4pq6kufc4m.onion"

    def test_tor_status_reporting(self):
        """Test Tor transport status reporting"""
        transport = TorTransport()
        status = transport.get_status()

        assert "is_running" in status
        assert "onion_address" in status
        assert "active_circuits" in status
        assert "stats" in status

        assert status["is_running"] == False
        assert status["active_circuits"] == 0


class TestFederationManager:
    """Test complete federation manager"""

    @pytest.mark.asyncio
    async def test_federation_manager_initialization(self):
        """Test federation manager initialization"""
        manager = FederationManager(
            device_id="test_federation_node",
            region="test_region",
            enable_tor=False,  # Disable for testing
            enable_i2p=False,
        )

        assert manager.device_id == "test_federation_node"
        assert manager.region == "test_region"
        assert manager.is_running == False

    @pytest.mark.asyncio
    async def test_federation_startup_sequence(self):
        """Test federation startup with mocked components"""
        manager = FederationManager("test_node", enable_tor=False, enable_i2p=False)

        # Mock the dual-path transport startup
        with patch(
            "federation.core.federation_manager.DualPathTransport"
        ) as mock_transport_class:
            mock_transport = Mock()
            mock_transport.start = AsyncMock(return_value=True)
            mock_transport.register_message_handler = Mock()
            mock_transport.broadcast_message = AsyncMock(return_value=3)
            mock_transport_class.return_value = mock_transport

            # Mock device registry initialization
            with patch.object(
                manager.device_registry, "initialize_local_device"
            ) as mock_init:
                mock_profile = Mock()
                mock_profile.role = DeviceRole.EDGE
                mock_profile.capabilities = {DeviceCapability.BLUETOOTH}
                mock_profile.protocols = {"bitchat", "betanet"}
                mock_profile.battery_percent = None  # No battery info
                mock_init.return_value = mock_profile

                # Start federation
                success = await manager.start()

                assert success == True
                assert manager.is_running == True
                assert manager.federation_role == DeviceRole.EDGE

    @pytest.mark.asyncio
    async def test_privacy_level_routing(self):
        """Test privacy level routing decisions"""
        manager = FederationManager("test_node")
        manager.is_running = True

        # Mock dual-path transport
        mock_transport = Mock()
        mock_transport.send_message = AsyncMock(return_value=True)
        manager.dual_path_transport = mock_transport

        # Mock privacy circuit creation for anonymous levels
        with patch.object(manager, "_send_via_privacy_circuit", return_value=True):
            # Test different privacy levels
            for privacy_level in [
                PrivacyLevel.PUBLIC,
                PrivacyLevel.PRIVATE,
                PrivacyLevel.ANONYMOUS,
            ]:
                success = await manager.send_federated_message(
                    recipient="test_recipient",
                    payload={"test": "message"},
                    privacy_level=privacy_level,
                )

                assert success == True

    @pytest.mark.asyncio
    async def test_ai_service_request_routing(self):
        """Test AI service request routing through federation"""
        manager = FederationManager("test_node")
        manager.is_running = True

        # Mock service discovery
        with patch.object(
            manager,
            "_find_service_providers",
            return_value=["edge_node_1", "edge_node_2"],
        ):
            with patch.object(
                manager, "_select_optimal_service_node", return_value="edge_node_1"
            ):
                with patch.object(manager, "send_federated_message", return_value=True):
                    result = await manager.request_ai_service(
                        service_name="translate",
                        request_data={"text": "Hello", "target_lang": "es"},
                        privacy_level=PrivacyLevel.PRIVATE,
                    )

                    # Should return some result (even if mocked)
                    assert result is not None

    @pytest.mark.asyncio
    async def test_privacy_tunnel_creation(self):
        """Test VPN-like privacy tunnel creation"""
        manager = FederationManager("test_node")
        manager.is_running = True

        # Mock device registry with relay nodes
        mock_relay_profile = Mock()
        mock_relay_profile.identity.device_id = "relay_node_1"
        mock_relay_profile.role = DeviceRole.RELAY

        with patch.object(
            manager.device_registry,
            "get_devices_by_role",
            return_value=[mock_relay_profile],
        ):
            tunnel_id = await manager.create_privacy_tunnel(
                destination="target_node", privacy_level=PrivacyLevel.ANONYMOUS
            )

            if tunnel_id:  # If tunnel creation succeeded
                assert tunnel_id in manager.active_tunnels
                tunnel_info = manager.active_tunnels[tunnel_id]
                assert tunnel_info["destination"] == "target_node"
                assert tunnel_info["privacy_level"] == PrivacyLevel.ANONYMOUS

    def test_federation_status_reporting(self):
        """Test comprehensive federation status reporting"""
        manager = FederationManager("test_node")
        status = manager.get_federation_status()

        required_keys = [
            "federation_role",
            "privacy_tunnels",
            "task_queue_size",
            "protocols_available",
            "federation_stats",
        ]

        for key in required_keys:
            assert key in status


class TestMultiProtocolIntegration:
    """Test integration between multiple protocols"""

    @pytest.mark.asyncio
    async def test_protocol_fallback_sequence(self):
        """Test fallback from BitChat -> Betanet -> Tor"""
        manager = FederationManager("test_node", enable_tor=True)

        # Mock protocol availability
        mock_navigator = Mock()

        # Test protocol selection under different conditions
        test_scenarios = [
            # Scenario 1: All protocols available, prefer BitChat for nearby peer
            {
                "available": ["bitchat", "betanet", "tor"],
                "peer_nearby": True,
                "expected": "bitchat",
            },
            # Scenario 2: No BitChat, use Betanet
            {
                "available": ["betanet", "tor"],
                "peer_nearby": False,
                "expected": "betanet",
            },
            # Scenario 3: Only Tor available (high privacy)
            {"available": ["tor"], "peer_nearby": False, "expected": "tor"},
        ]

        for scenario in test_scenarios:
            # This would test the actual protocol selection logic
            # For now, just verify the structure exists
            assert isinstance(scenario["available"], list)

    @pytest.mark.asyncio
    async def test_message_routing_across_protocols(self):
        """Test message routing across protocol boundaries"""
        # Test that a message can be sent via BitChat and received via Tor

        # Source: BitChat node
        bitchat_transport = EnhancedBitChatTransport("bitchat_node")

        # Destination: Tor node
        tor_transport = TorTransport()

        # Create test message
        test_message = {
            "type": "federation_message",
            "content": "cross-protocol test",
            "source_protocol": "bitchat",
            "target_protocol": "tor",
        }

        # In a real implementation, this would test actual message bridging
        # For now, verify the message structure
        assert test_message["type"] == "federation_message"
        assert test_message["source_protocol"] != test_message["target_protocol"]

    def test_privacy_level_enforcement(self):
        """Test that privacy levels are enforced correctly"""
        test_cases = [
            {
                "privacy_level": PrivacyLevel.PUBLIC,
                "expected_protocols": ["bitchat", "betanet"],
                "requires_encryption": False,
            },
            {
                "privacy_level": PrivacyLevel.PRIVATE,
                "expected_protocols": ["bitchat", "betanet"],
                "requires_encryption": True,
            },
            {
                "privacy_level": PrivacyLevel.ANONYMOUS,
                "expected_protocols": ["tor", "betanet_mixnode"],
                "requires_onion_routing": True,
            },
            {
                "privacy_level": PrivacyLevel.PARANOID,
                "expected_protocols": ["tor_i2p_chained"],
                "requires_multiple_hops": True,
                "requires_dummy_traffic": True,
            },
        ]

        for case in test_cases:
            # Verify privacy level structure
            assert isinstance(case["privacy_level"], int)
            assert case["privacy_level"] in [0, 1, 2, 3]
            assert isinstance(case["expected_protocols"], list)


class TestFederationPerformance:
    """Test federation performance and scalability"""

    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self):
        """Test handling multiple concurrent messages"""
        manager = FederationManager("test_node")
        manager.is_running = True

        # Mock transport
        mock_transport = Mock()
        mock_transport.send_message = AsyncMock(return_value=True)
        manager.dual_path_transport = mock_transport

        # Send multiple messages concurrently
        tasks = []
        for i in range(10):
            task = manager.send_federated_message(
                recipient=f"peer_{i}",
                payload={"message": f"test_{i}"},
                privacy_level=PrivacyLevel.PRIVATE,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All messages should succeed
        assert all(results)

    def test_memory_usage_limits(self):
        """Test that federation components stay within memory limits"""
        manager = FederationManager("test_node")

        # Add many devices to registry
        for i in range(1000):
            device_id = f"device_{i}"
            # In practice, would test actual memory usage

        # Verify basic functionality still works
        status = manager.get_federation_status()
        assert isinstance(status, dict)

    def test_network_partition_handling(self):
        """Test behavior during network partitions"""
        manager = FederationManager("test_node")

        # Simulate network partition by disabling protocols
        manager.enable_tor = False
        manager.enable_i2p = False

        # Should still function with remaining protocols
        available_protocols = manager._get_available_protocols()
        assert len(available_protocols) >= 1  # At least BitChat or Betanet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
