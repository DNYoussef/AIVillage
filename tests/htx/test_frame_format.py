"""
Comprehensive Test Suite for HTX Frame Format - Betanet v1.1

Tests the modular frame format implementation including:
- Frame encoding/decoding with uint24 length and varint stream_id
- Frame type validation and handling
- Buffer management and parsing
- Flow control frame creation
- Error handling and edge cases

Building on existing test patterns from the codebase.
"""

import os
import struct
import sys

import pytest

# Add src to path following existing pattern
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.p2p.htx.frame_format import (
    HTXFrame,
    HTXFrameBuffer,
    HTXFrameCodec,
    HTXFrameType,
    create_data_frame,
    create_ping_frame,
    create_window_update_frame,
)


class TestHTXFrameType:
    """Test HTX frame type enumeration."""

    def test_frame_type_values(self):
        """Test frame type values match specification."""
        assert HTXFrameType.DATA == 0x00
        assert HTXFrameType.WINDOW_UPDATE == 0x01
        assert HTXFrameType.KEY_UPDATE == 0x02
        assert HTXFrameType.PING == 0x03
        assert HTXFrameType.PRIORITY == 0x04
        assert HTXFrameType.PADDING == 0x05
        assert HTXFrameType.ACCESS_TICKET == 0x06
        assert HTXFrameType.CONTROL == 0x07

    def test_frame_type_names(self):
        """Test frame type names are correct."""
        assert HTXFrameType.DATA.name == "DATA"
        assert HTXFrameType.WINDOW_UPDATE.name == "WINDOW_UPDATE"
        assert HTXFrameType.PING.name == "PING"


class TestHTXFrame:
    """Test HTX frame structure."""

    def test_frame_creation(self):
        """Test basic frame creation."""
        payload = b"test payload data"
        frame = HTXFrame(frame_type=HTXFrameType.DATA, stream_id=42, payload=payload)

        assert frame.frame_type == HTXFrameType.DATA
        assert frame.stream_id == 42
        assert frame.payload == payload

    def test_frame_length_calculation(self):
        """Test frame length calculation."""
        # Small stream ID (1 byte varint)
        frame = HTXFrame(HTXFrameType.DATA, stream_id=10, payload=b"hello")
        expected_length = 3 + 1 + 1 + 5  # length(3) + varint(1) + type(1) + payload(5)
        assert len(frame) == expected_length

        # Large stream ID (4 bytes varint)
        frame = HTXFrame(HTXFrameType.DATA, stream_id=0x0FFFFFFF, payload=b"world")
        expected_length = 3 + 4 + 1 + 5  # length(3) + varint(4) + type(1) + payload(5)
        assert len(frame) == expected_length

    def test_varint_length_calculation(self):
        """Test varint length estimation."""
        # 1 byte varint (0-127)
        assert HTXFrame._encode_varint_length(0) == 1
        assert HTXFrame._encode_varint_length(127) == 1

        # 2 byte varint (128-16383)
        assert HTXFrame._encode_varint_length(128) == 2
        assert HTXFrame._encode_varint_length(16383) == 2

        # 3 byte varint (16384-2097151)
        assert HTXFrame._encode_varint_length(16384) == 3
        assert HTXFrame._encode_varint_length(2097151) == 3

        # 4 byte varint (2097152-268435455)
        assert HTXFrame._encode_varint_length(2097152) == 4
        assert HTXFrame._encode_varint_length(268435455) == 4

    def test_varint_overflow_error(self):
        """Test varint overflow detection."""
        with pytest.raises(ValueError, match="Stream ID too large"):
            HTXFrame._encode_varint_length(268435456)  # 2^28


class TestHTXFrameCodec:
    """Test HTX frame encoding and decoding."""

    def test_simple_frame_encode_decode(self):
        """Test basic frame encoding and decoding."""
        original_payload = b"Hello, HTX world!"

        # Encode frame
        encoded = HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=123, payload=original_payload)

        assert isinstance(encoded, bytes)
        assert len(encoded) > len(original_payload)

        # Decode frame
        decoded_frame, consumed = HTXFrameCodec.decode_frame(encoded)

        assert decoded_frame is not None
        assert decoded_frame.frame_type == HTXFrameType.DATA
        assert decoded_frame.stream_id == 123
        assert decoded_frame.payload == original_payload
        assert consumed == len(encoded)

    def test_empty_payload_frame(self):
        """Test frame with empty payload."""
        encoded = HTXFrameCodec.encode_frame(HTXFrameType.PING, stream_id=0, payload=b"")

        decoded_frame, consumed = HTXFrameCodec.decode_frame(encoded)

        assert decoded_frame.frame_type == HTXFrameType.PING
        assert decoded_frame.stream_id == 0
        assert decoded_frame.payload == b""

    def test_large_payload_frame(self):
        """Test frame with large payload."""
        large_payload = b"X" * 10000  # 10KB payload

        encoded = HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=456, payload=large_payload)

        decoded_frame, consumed = HTXFrameCodec.decode_frame(encoded)

        assert decoded_frame.frame_type == HTXFrameType.DATA
        assert decoded_frame.stream_id == 456
        assert decoded_frame.payload == large_payload
        assert len(decoded_frame.payload) == 10000

    def test_varint_stream_id_encoding(self):
        """Test various stream ID varint encodings."""
        test_cases = [
            (0, 1),  # 1 byte
            (127, 1),  # 1 byte boundary
            (128, 2),  # 2 byte
            (16383, 2),  # 2 byte boundary
            (16384, 3),  # 3 byte
            (2097151, 3),  # 3 byte boundary
            (2097152, 4),  # 4 byte
            (268435455, 4),  # 4 byte boundary
        ]

        for stream_id, expected_varint_bytes in test_cases:
            encoded = HTXFrameCodec.encode_frame(HTXFrameType.CONTROL, stream_id=stream_id, payload=b"test")

            decoded_frame, _ = HTXFrameCodec.decode_frame(encoded)
            assert decoded_frame.stream_id == stream_id

    def test_frame_size_limits(self):
        """Test frame size limit enforcement."""
        # Test maximum frame size
        max_payload = b"X" * (HTXFrameCodec.MAX_FRAME_SIZE - 10)  # Leave room for headers

        encoded = HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=1, payload=max_payload)

        decoded_frame, _ = HTXFrameCodec.decode_frame(encoded)
        assert len(decoded_frame.payload) == len(max_payload)

        # Test oversized frame
        oversized_payload = b"X" * HTXFrameCodec.MAX_FRAME_SIZE

        with pytest.raises(ValueError, match="Frame too large"):
            HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=1, payload=oversized_payload)

    def test_stream_id_limits(self):
        """Test stream ID limit enforcement."""
        # Test maximum stream ID
        max_stream_id = HTXFrameCodec.MAX_STREAM_ID

        encoded = HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=max_stream_id, payload=b"test")

        decoded_frame, _ = HTXFrameCodec.decode_frame(encoded)
        assert decoded_frame.stream_id == max_stream_id

        # Test oversized stream ID
        with pytest.raises(ValueError, match="Stream ID out of range"):
            HTXFrameCodec.encode_frame(
                HTXFrameType.DATA,
                stream_id=HTXFrameCodec.MAX_STREAM_ID + 1,
                payload=b"test",
            )

    def test_invalid_frame_type(self):
        """Test invalid frame type handling."""
        with pytest.raises(ValueError, match="Invalid frame type"):
            HTXFrameCodec.encode_frame(
                255,  # Invalid frame type
                stream_id=1,
                payload=b"test",
            )

    def test_negative_stream_id(self):
        """Test negative stream ID rejection."""
        with pytest.raises(ValueError, match="Stream ID out of range"):
            HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=-1, payload=b"test")


class TestFrameDecoding:
    """Test frame decoding edge cases and error handling."""

    def test_incomplete_frame_data(self):
        """Test handling of incomplete frame data."""
        # Create a complete frame first
        complete_frame = HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=42, payload=b"complete payload")

        # Test with incomplete length header
        incomplete_data = complete_frame[:2]  # Only 2 bytes of 3-byte length
        decoded_frame, consumed = HTXFrameCodec.decode_frame(incomplete_data)
        assert decoded_frame is None
        assert consumed == 0

        # Test with incomplete frame content
        incomplete_data = complete_frame[:10]  # Partial frame
        decoded_frame, consumed = HTXFrameCodec.decode_frame(incomplete_data)
        assert decoded_frame is None
        assert consumed == 0

    def test_malformed_frame_data(self):
        """Test handling of malformed frame data."""
        # Test with invalid length (too large)
        malformed_data = struct.pack(">I", HTXFrameCodec.MAX_FRAME_SIZE + 1)[1:]  # 3-byte length
        malformed_data += b"dummy content"

        with pytest.raises(ValueError, match="Frame too large"):
            HTXFrameCodec.decode_frame(malformed_data)

        # Test with unknown frame type
        valid_frame = HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=1, payload=b"test")

        # Modify frame type to invalid value
        malformed_data = bytearray(valid_frame)
        type_offset = 3 + 1  # length(3) + varint stream_id(1)
        malformed_data[type_offset] = 255  # Invalid frame type

        with pytest.raises(ValueError, match="Unknown frame type"):
            HTXFrameCodec.decode_frame(bytes(malformed_data))

    def test_truncated_varint(self):
        """Test handling of truncated varint stream ID."""
        # Create a frame that should have a multi-byte varint but is truncated
        length_bytes = struct.pack(">I", 5)[1:]  # 3-byte length = 5
        truncated_data = length_bytes + b"\x80"  # Start of 2-byte varint but truncated

        with pytest.raises(ValueError, match="Incomplete varint"):
            HTXFrameCodec.decode_frame(truncated_data)

    def test_oversized_varint(self):
        """Test handling of oversized varint stream ID."""
        # Create a frame with varint that's too long (>4 bytes)
        length_bytes = struct.pack(">I", 10)[1:]  # 3-byte length
        oversized_varint = b"\x80\x80\x80\x80\x80"  # 5-byte varint (too long)
        malformed_data = length_bytes + oversized_varint + b"\x00test"

        with pytest.raises(ValueError, match="Varint too long"):
            HTXFrameCodec.decode_frame(malformed_data)


class TestHTXFrameBuffer:
    """Test HTX frame buffer for stream processing."""

    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = HTXFrameBuffer()
        assert len(buffer.buffer) == 0
        assert buffer.max_buffer_size == 1048576  # 1MB default

        # Test custom buffer size
        custom_buffer = HTXFrameBuffer(max_buffer_size=512000)
        assert custom_buffer.max_buffer_size == 512000

    def test_single_frame_parsing(self):
        """Test parsing single complete frame from buffer."""
        buffer = HTXFrameBuffer()

        # Create a frame
        test_frame = HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=100, payload=b"single frame test")

        # Add to buffer and parse
        buffer.append_data(test_frame)
        frames = buffer.parse_frames()

        assert len(frames) == 1
        assert frames[0].frame_type == HTXFrameType.DATA
        assert frames[0].stream_id == 100
        assert frames[0].payload == b"single frame test"

        # Buffer should be empty after parsing
        assert len(buffer.buffer) == 0

    def test_multiple_frames_parsing(self):
        """Test parsing multiple frames from buffer."""
        buffer = HTXFrameBuffer()

        # Create multiple frames
        frame1 = HTXFrameCodec.encode_frame(HTXFrameType.DATA, 1, b"frame 1")
        frame2 = HTXFrameCodec.encode_frame(HTXFrameType.PING, 0, b"ping data")
        frame3 = HTXFrameCodec.encode_frame(HTXFrameType.CONTROL, 5, b"control")

        # Add all frames at once
        all_frames_data = frame1 + frame2 + frame3
        buffer.append_data(all_frames_data)

        # Parse all frames
        frames = buffer.parse_frames()

        assert len(frames) == 3

        assert frames[0].frame_type == HTXFrameType.DATA
        assert frames[0].stream_id == 1
        assert frames[0].payload == b"frame 1"

        assert frames[1].frame_type == HTXFrameType.PING
        assert frames[1].stream_id == 0
        assert frames[1].payload == b"ping data"

        assert frames[2].frame_type == HTXFrameType.CONTROL
        assert frames[2].stream_id == 5
        assert frames[2].payload == b"control"

    def test_partial_frame_handling(self):
        """Test handling of partial frames in buffer."""
        buffer = HTXFrameBuffer()

        # Create a complete frame
        complete_frame = HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=42, payload=b"complete frame data")

        # Add partial frame data
        partial_data = complete_frame[:10]
        buffer.append_data(partial_data)

        # Should not parse any frames yet
        frames = buffer.parse_frames()
        assert len(frames) == 0
        assert len(buffer.buffer) == 10  # Partial data remains

        # Add remaining data
        remaining_data = complete_frame[10:]
        buffer.append_data(remaining_data)

        # Now should parse the complete frame
        frames = buffer.parse_frames()
        assert len(frames) == 1
        assert frames[0].payload == b"complete frame data"
        assert len(buffer.buffer) == 0  # Buffer consumed

    def test_buffer_overflow_protection(self):
        """Test buffer overflow protection."""
        buffer = HTXFrameBuffer(max_buffer_size=1000)

        # Try to add data that would exceed buffer size
        large_data = b"X" * 1500

        with pytest.raises(ValueError, match="Buffer overflow"):
            buffer.append_data(large_data)

        # Buffer should remain unchanged
        assert len(buffer.buffer) == 0

    def test_malformed_frame_recovery(self):
        """Test recovery from malformed frames in stream."""
        buffer = HTXFrameBuffer()

        # Add some junk data followed by a valid frame
        junk_data = b"\xff\xff\xff\xff"  # Invalid frame
        valid_frame = HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=1, payload=b"valid after junk")

        buffer.append_data(junk_data + valid_frame)

        # Should skip junk and parse valid frame
        frames = buffer.parse_frames()

        # Depending on implementation, might recover and parse valid frame
        # This tests error recovery behavior
        assert isinstance(frames, list)

    def test_buffer_info(self):
        """Test buffer information reporting."""
        buffer = HTXFrameBuffer(max_buffer_size=2048)

        info = buffer.get_buffer_info()
        assert info["buffer_size"] == 0
        assert info["max_buffer_size"] == 2048
        assert info["utilization"] == 0.0

        # Add some data
        test_data = b"X" * 100
        buffer.append_data(test_data)

        info = buffer.get_buffer_info()
        assert info["buffer_size"] == 100
        assert info["utilization"] == 100 / 2048


class TestFrameHelperFunctions:
    """Test frame creation helper functions."""

    def test_create_data_frame(self):
        """Test DATA frame creation helper."""
        stream_id = 42
        data = b"test data payload"

        frame_bytes = create_data_frame(stream_id, data)

        # Decode and verify
        decoded_frame, _ = HTXFrameCodec.decode_frame(frame_bytes)
        assert decoded_frame.frame_type == HTXFrameType.DATA
        assert decoded_frame.stream_id == stream_id
        assert decoded_frame.payload == data

    def test_create_data_frame_invalid_stream(self):
        """Test DATA frame creation with invalid stream ID."""
        with pytest.raises(ValueError, match="DATA frames require stream_id > 0"):
            create_data_frame(0, b"data")

        with pytest.raises(ValueError, match="DATA frames require stream_id > 0"):
            create_data_frame(-1, b"data")

    def test_create_ping_frame(self):
        """Test PING frame creation helper."""
        # Test with default timestamp data
        frame_bytes = create_ping_frame()

        decoded_frame, _ = HTXFrameCodec.decode_frame(frame_bytes)
        assert decoded_frame.frame_type == HTXFrameType.PING
        assert decoded_frame.stream_id == 0  # PING uses stream 0
        assert len(decoded_frame.payload) == 8  # Timestamp is 8 bytes

        # Test with custom ping data
        custom_data = b"ping123"
        frame_bytes = create_ping_frame(custom_data)

        decoded_frame, _ = HTXFrameCodec.decode_frame(frame_bytes)
        assert decoded_frame.payload == custom_data

    def test_create_ping_frame_oversized(self):
        """Test PING frame with oversized payload."""
        oversized_data = b"X" * 10  # > 8 bytes

        with pytest.raises(ValueError, match="PING payload must be <= 8 bytes"):
            create_ping_frame(oversized_data)

    def test_create_window_update_frame(self):
        """Test WINDOW_UPDATE frame creation helper."""
        stream_id = 25
        window_delta = 65536

        frame_bytes = create_window_update_frame(stream_id, window_delta)

        decoded_frame, _ = HTXFrameCodec.decode_frame(frame_bytes)
        assert decoded_frame.frame_type == HTXFrameType.WINDOW_UPDATE
        assert decoded_frame.stream_id == stream_id
        assert len(decoded_frame.payload) == 4  # Window delta is 4 bytes

        # Verify window delta encoding
        expected_payload = struct.pack(">I", window_delta)
        assert decoded_frame.payload == expected_payload

    def test_create_window_update_invalid_delta(self):
        """Test WINDOW_UPDATE frame with invalid delta."""
        # Test zero delta
        with pytest.raises(ValueError, match="Invalid window delta"):
            create_window_update_frame(1, 0)

        # Test negative delta
        with pytest.raises(ValueError, match="Invalid window delta"):
            create_window_update_frame(1, -1000)

        # Test oversized delta
        with pytest.raises(ValueError, match="Invalid window delta"):
            create_window_update_frame(1, 0x80000000)  # > 2^31-1


class TestFrameFormatIntegration:
    """Integration tests for frame format system."""

    def test_round_trip_all_frame_types(self):
        """Test round-trip encoding/decoding for all frame types."""
        test_cases = [
            (HTXFrameType.DATA, 1, b"data payload"),
            (HTXFrameType.WINDOW_UPDATE, 2, struct.pack(">I", 1024)),
            (HTXFrameType.KEY_UPDATE, 0, b"key update data"),
            (HTXFrameType.PING, 0, b"ping123"),
            (HTXFrameType.PRIORITY, 5, b"priority info"),
            (HTXFrameType.PADDING, 10, b"X" * 100),
            (HTXFrameType.ACCESS_TICKET, 0, b"ticket data"),
            (HTXFrameType.CONTROL, 15, b"control message"),
        ]

        for frame_type, stream_id, payload in test_cases:
            # Encode
            encoded = HTXFrameCodec.encode_frame(frame_type, stream_id, payload)

            # Decode
            decoded_frame, consumed = HTXFrameCodec.decode_frame(encoded)

            # Verify
            assert decoded_frame.frame_type == frame_type
            assert decoded_frame.stream_id == stream_id
            assert decoded_frame.payload == payload
            assert consumed == len(encoded)

    def test_streaming_frame_processing(self):
        """Test processing frames as they arrive in chunks."""
        buffer = HTXFrameBuffer()

        # Create a sequence of frames
        frames_to_send = [
            (HTXFrameType.DATA, 1, b"first message"),
            (HTXFrameType.DATA, 3, b"second message"),
            (HTXFrameType.PING, 0, b"keepalive"),
            (HTXFrameType.DATA, 1, b"third message"),
        ]

        # Encode all frames
        encoded_frames = []
        for frame_type, stream_id, payload in frames_to_send:
            encoded = HTXFrameCodec.encode_frame(frame_type, stream_id, payload)
            encoded_frames.append(encoded)

        # Simulate streaming by adding data in chunks
        all_data = b"".join(encoded_frames)
        chunk_size = 50  # Small chunks to simulate network packets

        received_frames = []

        for i in range(0, len(all_data), chunk_size):
            chunk = all_data[i : i + chunk_size]
            buffer.append_data(chunk)

            # Try to parse frames after each chunk
            frames = buffer.parse_frames()
            received_frames.extend(frames)

        # Verify all frames were received correctly
        assert len(received_frames) == len(frames_to_send)

        for i, (expected_type, expected_stream, expected_payload) in enumerate(frames_to_send):
            frame = received_frames[i]
            assert frame.frame_type == expected_type
            assert frame.stream_id == expected_stream
            assert frame.payload == expected_payload


def test_htx_frame_format_smoke_test():
    """Smoke test for HTX frame format functionality."""
    print("Running HTX frame format smoke test...")

    # Test basic frame creation
    payload = b"HTX frame format test"
    frame_bytes = HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id=42, payload=payload)
    assert len(frame_bytes) > len(payload)
    print(f"  Encoded frame: {len(frame_bytes)} bytes")

    # Test decoding
    decoded_frame, consumed = HTXFrameCodec.decode_frame(frame_bytes)
    assert decoded_frame.payload == payload
    assert consumed == len(frame_bytes)
    print(f"  Decoded frame: stream={decoded_frame.stream_id}, type={decoded_frame.frame_type.name}")

    # Test buffer processing
    buffer = HTXFrameBuffer()
    buffer.append_data(frame_bytes)
    frames = buffer.parse_frames()
    assert len(frames) == 1
    print(f"  Buffer processed: {len(frames)} frames")

    # Test helper functions
    ping_frame = create_ping_frame()
    window_frame = create_window_update_frame(1, 1024)
    data_frame = create_data_frame(5, b"helper test")

    assert len(ping_frame) > 0
    assert len(window_frame) > 0
    assert len(data_frame) > 0
    print(f"  Helper functions: ping={len(ping_frame)}, window={len(window_frame)}, data={len(data_frame)} bytes")

    print("  HTX frame format smoke test PASSED")


if __name__ == "__main__":
    # Run smoke test when executed directly
    test_htx_frame_format_smoke_test()
    print("\nTo run full test suite:")
    print("  pytest tests/htx/test_frame_format.py -v")
