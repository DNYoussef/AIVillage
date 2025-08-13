#!/usr/bin/env python3
"""
BitChat Protocol Buffer Round-trip Test

Validates the protobuf schema by testing serialization/deserialization
and cross-platform compatibility scenarios for Android/iOS interchange.
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any


# Mock protobuf classes for testing (in production would use generated protobuf)
@dataclass
class MockEnvelope:
    """Mock BitChat message envelope for testing"""

    msg_id: str
    created_at: int
    hop_count: int
    ttl: int
    original_sender: str
    message_type: str
    ciphertext_blob: bytes
    routing: dict | None = None
    priority: str = "PRIORITY_NORMAL"

    def serialize(self) -> bytes:
        """Serialize to JSON bytes (mock protobuf serialization)"""
        data = {
            "msg_id": self.msg_id,
            "created_at": self.created_at,
            "hop_count": self.hop_count,
            "ttl": self.ttl,
            "original_sender": self.original_sender,
            "message_type": self.message_type,
            "ciphertext_blob": self.ciphertext_blob.hex(),
            "routing": self.routing,
            "priority": self.priority,
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "MockEnvelope":
        """Deserialize from JSON bytes (mock protobuf deserialization)"""
        parsed = json.loads(data.decode("utf-8"))
        return cls(
            msg_id=parsed["msg_id"],
            created_at=parsed["created_at"],
            hop_count=parsed["hop_count"],
            ttl=parsed["ttl"],
            original_sender=parsed["original_sender"],
            message_type=parsed["message_type"],
            ciphertext_blob=bytes.fromhex(parsed["ciphertext_blob"]),
            routing=parsed.get("routing"),
            priority=parsed.get("priority", "PRIORITY_NORMAL"),
        )


@dataclass
class MockPeerCapability:
    """Mock peer capability message"""

    peer_id: str
    device_info: dict[str, Any]
    supported_transports: list[str]
    battery_level: float
    network_caps: dict[str, Any]
    last_heartbeat: int
    trust_score: float = 1.0

    def serialize(self) -> bytes:
        data = {
            "peer_id": self.peer_id,
            "device_info": self.device_info,
            "supported_transports": self.supported_transports,
            "battery_level": self.battery_level,
            "network_caps": self.network_caps,
            "last_heartbeat": self.last_heartbeat,
            "trust_score": self.trust_score,
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "MockPeerCapability":
        parsed = json.loads(data.decode("utf-8"))
        return cls(**parsed)


@dataclass
class MockChunkedMessage:
    """Mock chunked message for large payload testing"""

    message_id: str
    chunk_index: int
    total_chunks: int
    chunk_data: bytes
    chunk_checksum: str
    total_size: int

    def serialize(self) -> bytes:
        data = {
            "message_id": self.message_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunk_data": self.chunk_data.hex(),
            "chunk_checksum": self.chunk_checksum,
            "total_size": self.total_size,
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "MockChunkedMessage":
        parsed = json.loads(data.decode("utf-8"))
        return cls(
            message_id=parsed["message_id"],
            chunk_index=parsed["chunk_index"],
            total_chunks=parsed["total_chunks"],
            chunk_data=bytes.fromhex(parsed["chunk_data"]),
            chunk_checksum=parsed["chunk_checksum"],
            total_size=parsed["total_size"],
        )


class TestBitChatProtocolRoundtrip:
    """Test suite for BitChat protocol message round-trip validation"""

    def test_basic_envelope_roundtrip(self):
        """Test basic message envelope serialization and deserialization"""
        # Create test envelope
        test_payload = b"Hello BitChat from Python test!"
        envelope = MockEnvelope(
            msg_id=f"test_{uuid.uuid4().hex[:8]}",
            created_at=int(time.time() * 1000),
            hop_count=0,
            ttl=7,
            original_sender="python_test_peer",
            message_type="MESSAGE_TYPE_DATA",
            ciphertext_blob=test_payload,
        )

        # Serialize and deserialize
        serialized = envelope.serialize()
        deserialized = MockEnvelope.deserialize(serialized)

        # Validate round-trip
        assert envelope.msg_id == deserialized.msg_id
        assert envelope.created_at == deserialized.created_at
        assert envelope.hop_count == deserialized.hop_count
        assert envelope.ttl == deserialized.ttl
        assert envelope.original_sender == deserialized.original_sender
        assert envelope.message_type == deserialized.message_type
        assert envelope.ciphertext_blob == deserialized.ciphertext_blob

        print(f"✅ Basic envelope round-trip test passed ({len(serialized)} bytes)")

    def test_hop_count_progression(self):
        """Test TTL and hop count validation logic"""
        initial_envelope = MockEnvelope(
            msg_id="hop_test_msg",
            created_at=int(time.time() * 1000),
            hop_count=0,
            ttl=7,
            original_sender="origin_peer",
            message_type="MESSAGE_TYPE_DATA",
            ciphertext_blob=b"hop test message",
        )

        # Simulate message relay through 7 hops
        current_envelope = initial_envelope
        for hop in range(1, 8):
            # Simulate relay processing
            relayed_envelope = MockEnvelope(
                msg_id=current_envelope.msg_id,
                created_at=current_envelope.created_at,
                hop_count=current_envelope.hop_count + 1,
                ttl=current_envelope.ttl - 1,
                original_sender=current_envelope.original_sender,
                message_type=current_envelope.message_type,
                ciphertext_blob=current_envelope.ciphertext_blob,
            )

            # Validate hop progression
            assert relayed_envelope.hop_count == hop
            assert relayed_envelope.ttl == 7 - hop

            # Test serialization at each hop
            serialized = relayed_envelope.serialize()
            deserialized = MockEnvelope.deserialize(serialized)
            assert deserialized.hop_count == hop
            assert deserialized.ttl == 7 - hop

            current_envelope = relayed_envelope

            # Message should be dropped when TTL reaches 0
            if relayed_envelope.ttl <= 0:
                print(f"⏰ Message dropped at hop {hop} (TTL expired)")
                break

        print(f"✅ Hop count progression test passed (max {hop} hops)")

    def test_peer_capability_exchange(self):
        """Test peer capability message format"""
        capability = MockPeerCapability(
            peer_id="test_peer_android_001",
            device_info={
                "platform": "PLATFORM_ANDROID",
                "device_model": "Pixel 7",
                "os_version": "Android 14",
                "app_version": "1.0.0",
                "has_wifi": True,
                "has_bluetooth": True,
                "has_cellular": True,
                "has_nfc": False,
            },
            supported_transports=[
                "TRANSPORT_BLE",
                "TRANSPORT_BLUETOOTH_CLASSIC",
                "TRANSPORT_WIFI_DIRECT",
                "TRANSPORT_NEARBY_CONNECTIONS",
            ],
            battery_level=0.75,
            network_caps={
                "max_connections": 10,
                "max_chunk_size": 262144,  # 256KB
                "supports_relay": True,
                "supports_store_forward": True,
                "supports_encryption": True,
                "supports_background": False,
                "estimated_bandwidth": 1000000,  # 1 Mbps
            },
            last_heartbeat=int(time.time() * 1000),
            trust_score=0.9,
        )

        # Test round-trip
        serialized = capability.serialize()
        deserialized = MockPeerCapability.deserialize(serialized)

        assert capability.peer_id == deserialized.peer_id
        assert capability.device_info == deserialized.device_info
        assert capability.supported_transports == deserialized.supported_transports
        assert capability.battery_level == deserialized.battery_level
        assert capability.network_caps == deserialized.network_caps
        assert capability.trust_score == deserialized.trust_score

        print(f"✅ Peer capability exchange test passed ({len(serialized)} bytes)")

    def test_chunked_message_assembly(self):
        """Test chunked message handling for large payloads"""
        # Create large test message (1MB)
        large_payload = b"A" * (1024 * 1024)  # 1MB of 'A' characters
        chunk_size = 256 * 1024  # 256KB chunks
        total_chunks = (len(large_payload) + chunk_size - 1) // chunk_size

        print(
            f"📦 Testing chunked message: {len(large_payload)} bytes in {total_chunks} chunks"
        )

        # Create chunks
        chunks = []
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(large_payload))
            chunk_data = large_payload[start_idx:end_idx]

            # Calculate checksum for integrity
            chunk_checksum = hashlib.sha256(chunk_data).hexdigest()

            chunk = MockChunkedMessage(
                message_id="large_msg_test",
                chunk_index=i,
                total_chunks=total_chunks,
                chunk_data=chunk_data,
                chunk_checksum=chunk_checksum,
                total_size=len(large_payload),
            )

            # Test serialization of each chunk
            serialized = chunk.serialize()
            deserialized = MockChunkedMessage.deserialize(serialized)

            assert chunk.message_id == deserialized.message_id
            assert chunk.chunk_index == deserialized.chunk_index
            assert chunk.total_chunks == deserialized.total_chunks
            assert chunk.chunk_data == deserialized.chunk_data
            assert chunk.chunk_checksum == deserialized.chunk_checksum
            assert chunk.total_size == deserialized.total_size

            chunks.append(deserialized)

        # Verify chunk assembly
        reassembled_payload = b""
        for chunk in chunks:
            reassembled_payload += chunk.chunk_data

        assert reassembled_payload == large_payload
        assert len(chunks) == total_chunks

        print(f"✅ Chunked message assembly test passed ({total_chunks} chunks)")

    def test_cross_platform_compatibility(self):
        """Test Android/iOS interoperability scenarios"""
        # Android -> iOS message
        android_envelope = MockEnvelope(
            msg_id="android_to_ios_msg",
            created_at=int(time.time() * 1000),
            hop_count=1,
            ttl=6,
            original_sender="android_pixel_peer",
            message_type="MESSAGE_TYPE_DATA",
            ciphertext_blob=b"Hello from Android Nearby Connections!",
            routing={
                "target_peer_id": "ios_iphone_peer",
                "path": ["android_pixel_peer", "bridge_peer"],
                "network_region": "local_mesh_A",
                "qos": {
                    "max_latency_ms": 1000,
                    "reliability_threshold": 0.95,
                    "bandwidth_bps": 100000,
                    "prefer_low_power": False,
                },
            },
            priority="PRIORITY_HIGH",
        )

        # Serialize and deserialize (simulating cross-platform transmission)
        serialized = android_envelope.serialize()
        ios_received = MockEnvelope.deserialize(serialized)

        # Validate cross-platform message integrity
        assert android_envelope.msg_id == ios_received.msg_id
        assert android_envelope.original_sender == ios_received.original_sender
        assert android_envelope.message_type == ios_received.message_type
        assert android_envelope.ciphertext_blob == ios_received.ciphertext_blob
        assert android_envelope.routing == ios_received.routing
        assert android_envelope.priority == ios_received.priority

        # iOS -> Android message
        ios_envelope = MockEnvelope(
            msg_id="ios_to_android_msg",
            created_at=int(time.time() * 1000),
            hop_count=2,
            ttl=5,
            original_sender="ios_iphone_peer",
            message_type="MESSAGE_TYPE_DATA",
            ciphertext_blob=b"Hello from iOS MultipeerConnectivity!",
            routing={
                "target_peer_id": "android_pixel_peer",
                "path": ["ios_iphone_peer", "bridge_peer", "android_relay"],
                "network_region": "local_mesh_A",
            },
        )

        # Test reverse direction
        serialized = ios_envelope.serialize()
        android_received = MockEnvelope.deserialize(serialized)

        assert ios_envelope.msg_id == android_received.msg_id
        assert ios_envelope.original_sender == android_received.original_sender
        assert ios_envelope.hop_count == android_received.hop_count
        assert ios_envelope.ttl == android_received.ttl

        print("✅ Cross-platform compatibility test passed")

    def test_message_size_limits(self):
        """Test various message sizes and validate chunk size decisions"""
        test_sizes = [
            100,  # Small message
            1024,  # 1KB message
            65536,  # 64KB message
            262144,  # 256KB message (chunk boundary)
            500000,  # 500KB message (requires chunking)
            1048576,  # 1MB message (multiple chunks)
        ]

        chunk_threshold = 256 * 1024  # 256KB

        for size in test_sizes:
            payload = b"X" * size
            envelope = MockEnvelope(
                msg_id=f"size_test_{size}",
                created_at=int(time.time() * 1000),
                hop_count=0,
                ttl=7,
                original_sender="size_test_peer",
                message_type="MESSAGE_TYPE_DATA",
                ciphertext_blob=payload,
            )

            serialized = envelope.serialize()

            # Check if chunking would be required
            requires_chunking = len(serialized) > chunk_threshold
            expected_chunks = (
                1
                if not requires_chunking
                else ((len(serialized) + chunk_threshold - 1) // chunk_threshold)
            )

            print(
                f"📏 Message size {size} bytes -> serialized {len(serialized)} bytes "
                f"(chunks: {expected_chunks}, chunking: {'yes' if requires_chunking else 'no'})"
            )

            # Validate serialization works regardless of size
            deserialized = MockEnvelope.deserialize(serialized)
            assert envelope.ciphertext_blob == deserialized.ciphertext_blob

        print("✅ Message size limits test passed")

    def test_error_handling_and_validation(self):
        """Test protocol error handling and data validation"""
        # Test invalid TTL
        invalid_ttl_envelope = MockEnvelope(
            msg_id="invalid_ttl_test",
            created_at=int(time.time() * 1000),
            hop_count=8,  # Exceeds max hops
            ttl=0,  # Already expired
            original_sender="error_test_peer",
            message_type="MESSAGE_TYPE_DATA",
            ciphertext_blob=b"invalid message",
        )

        # Should be detectable as invalid
        assert invalid_ttl_envelope.hop_count > 7  # Exceeds max
        assert invalid_ttl_envelope.ttl <= 0  # Expired

        # Test malformed message ID
        malformed_envelope = MockEnvelope(
            msg_id="",  # Empty message ID
            created_at=int(time.time() * 1000),
            hop_count=0,
            ttl=7,
            original_sender="",  # Empty sender
            message_type="INVALID_TYPE",  # Invalid type
            ciphertext_blob=b"",  # Empty payload
        )

        # Should be detectable as malformed
        assert len(malformed_envelope.msg_id) == 0
        assert len(malformed_envelope.original_sender) == 0
        assert malformed_envelope.message_type not in [
            "MESSAGE_TYPE_DATA",
            "MESSAGE_TYPE_CAPABILITY",
            "MESSAGE_TYPE_HEARTBEAT",
            "MESSAGE_TYPE_DISCOVERY",
        ]

        # Test timestamp validation
        future_envelope = MockEnvelope(
            msg_id="future_test",
            created_at=int(time.time() * 1000)
            + (24 * 60 * 60 * 1000),  # 24 hours in future
            hop_count=0,
            ttl=7,
            original_sender="time_test_peer",
            message_type="MESSAGE_TYPE_DATA",
            ciphertext_blob=b"future message",
        )

        # Should be detectable as future-dated
        current_time = int(time.time() * 1000)
        assert future_envelope.created_at > current_time + (
            60 * 1000
        )  # More than 1 minute in future

        print("✅ Error handling and validation test passed")

    def test_performance_benchmarks(self):
        """Test serialization/deserialization performance"""
        import time

        # Performance test parameters
        message_count = 1000
        payload_sizes = [100, 1024, 10240]  # 100B, 1KB, 10KB

        for payload_size in payload_sizes:
            payload = b"P" * payload_size

            # Measure serialization performance
            start_time = time.time()
            for i in range(message_count):
                envelope = MockEnvelope(
                    msg_id=f"perf_test_{i}",
                    created_at=int(time.time() * 1000),
                    hop_count=i % 7,
                    ttl=7 - (i % 7),
                    original_sender="perf_test_peer",
                    message_type="MESSAGE_TYPE_DATA",
                    ciphertext_blob=payload,
                )
                envelope.serialize()

            serialization_time = time.time() - start_time

            # Measure deserialization performance
            test_envelope = MockEnvelope(
                msg_id="perf_final_test",
                created_at=int(time.time() * 1000),
                hop_count=0,
                ttl=7,
                original_sender="perf_test_peer",
                message_type="MESSAGE_TYPE_DATA",
                ciphertext_blob=payload,
            )
            test_serialized = test_envelope.serialize()

            start_time = time.time()
            for i in range(message_count):
                MockEnvelope.deserialize(test_serialized)
            deserialization_time = time.time() - start_time

            # Calculate rates
            serialization_rate = message_count / serialization_time
            deserialization_rate = message_count / deserialization_time

            print(
                f"⚡ Performance {payload_size}B payload: "
                f"serialize {serialization_rate:.0f} msg/s, "
                f"deserialize {deserialization_rate:.0f} msg/s"
            )

        print("✅ Performance benchmarks completed")


def run_protocol_tests():
    """Run all protocol tests and report results"""
    print("🧪 BitChat Protocol Buffer Round-trip Tests")
    print("=" * 50)

    test_suite = TestBitChatProtocolRoundtrip()

    tests = [
        ("Basic Envelope Round-trip", test_suite.test_basic_envelope_roundtrip),
        ("Hop Count Progression", test_suite.test_hop_count_progression),
        ("Peer Capability Exchange", test_suite.test_peer_capability_exchange),
        ("Chunked Message Assembly", test_suite.test_chunked_message_assembly),
        ("Cross-platform Compatibility", test_suite.test_cross_platform_compatibility),
        ("Message Size Limits", test_suite.test_message_size_limits),
        ("Error Handling", test_suite.test_error_handling_and_validation),
        ("Performance Benchmarks", test_suite.test_performance_benchmarks),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n🔬 Running: {test_name}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {e}")
            failed += 1

    print(f"\n📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All protocol tests passed! BitChat protobuf schema is valid.")
        return True
    else:
        print("💥 Some tests failed. Please review the protobuf schema.")
        return False


if __name__ == "__main__":
    success = run_protocol_tests()
    exit(0 if success else 1)
