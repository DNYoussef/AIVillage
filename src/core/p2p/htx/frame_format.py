"""HTX Frame Format Implementation - Betanet v1.1

Implements the HTX frame format specification:
- uint24 length field (big-endian)
- varint stream_id encoding
- uint8 frame type
- variable payload
- WINDOW_UPDATE flow control frames

This module is focused solely on frame encoding/decoding.
"""

import logging
import struct
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class HTXFrameType(IntEnum):
    """HTX frame types per Betanet v1.1 specification."""

    DATA = 0x00
    WINDOW_UPDATE = 0x01
    KEY_UPDATE = 0x02
    PING = 0x03
    PRIORITY = 0x04
    PADDING = 0x05
    ACCESS_TICKET = 0x06
    CONTROL = 0x07


@dataclass
class HTXFrame:
    """Parsed HTX frame structure."""

    frame_type: HTXFrameType
    stream_id: int
    payload: bytes

    def __len__(self) -> int:
        """Get total frame size including headers."""
        return 3 + self._encode_varint_length(self.stream_id) + 1 + len(self.payload)

    @staticmethod
    def _encode_varint_length(value: int) -> int:
        """Get length of varint encoding."""
        if value < 0x80:
            return 1
        elif value < 0x4000:
            return 2
        elif value < 0x200000:
            return 3
        elif value < 0x10000000:
            return 4
        else:
            raise ValueError(f"Stream ID too large: {value}")


class HTXFrameCodec:
    """HTX frame encoder/decoder with strict v1.1 compliance."""

    # Frame size limits per spec
    MAX_FRAME_SIZE = 16_777_215  # 2^24 - 1 (uint24 max)
    MAX_STREAM_ID = 268_435_455  # 2^28 - 1 (varint max)

    @staticmethod
    def encode_frame(frame_type: HTXFrameType, stream_id: int, payload: bytes) -> bytes:
        """Encode HTX frame with proper v1.1 format.

        Format: uint24 length | varint stream_id | uint8 type | payload

        Args:
            frame_type: HTX frame type
            stream_id: Stream identifier (0 for connection-level frames)
            payload: Frame payload data

        Returns:
            Encoded frame bytes

        Raises:
            ValueError: If parameters violate spec limits
        """
        if stream_id < 0 or stream_id > HTXFrameCodec.MAX_STREAM_ID:
            raise ValueError(f"Stream ID out of range: {stream_id}")

        if not isinstance(frame_type, HTXFrameType):
            raise ValueError(f"Invalid frame type: {frame_type}")

        # Encode stream ID as varint
        stream_bytes = HTXFrameCodec._encode_varint(stream_id)

        # Calculate frame length: stream_id + type + payload
        frame_content_length = len(stream_bytes) + 1 + len(payload)

        if frame_content_length > HTXFrameCodec.MAX_FRAME_SIZE:
            raise ValueError(f"Frame too large: {frame_content_length} > {HTXFrameCodec.MAX_FRAME_SIZE}")

        # Encode uint24 length (big-endian)
        length_bytes = struct.pack(">I", frame_content_length)[1:]  # Take last 3 bytes

        # Assemble frame
        frame = length_bytes + stream_bytes + bytes([frame_type]) + payload

        logger.debug(
            f"Encoded HTX frame: type={frame_type.name}, stream={stream_id}, "
            f"payload_len={len(payload)}, total_len={len(frame)}"
        )

        return frame

    @staticmethod
    def decode_frame(data: bytes) -> tuple[HTXFrame | None, int]:
        """Decode HTX frame from bytes.

        Args:
            data: Raw frame data (may contain partial frames)

        Returns:
            Tuple of (parsed_frame, bytes_consumed) or (None, 0) if incomplete

        Raises:
            ValueError: If frame format is invalid
        """
        if len(data) < 4:  # Minimum: 3-byte length + 1-byte stream_id + 1-byte type
            return None, 0

        # Decode uint24 length (big-endian)
        length = struct.unpack(">I", b"\x00" + data[:3])[0]

        if length > HTXFrameCodec.MAX_FRAME_SIZE:
            raise ValueError(f"Frame too large: {length} > {HTXFrameCodec.MAX_FRAME_SIZE}")

        # Check if we have the complete frame
        total_frame_size = 3 + length  # 3-byte length + content
        if len(data) < total_frame_size:
            return None, 0  # Need more data

        # Parse frame content starting after length field
        content_start = 3
        content_data = data[content_start : content_start + length]

        # Decode varint stream_id
        stream_id, varint_len = HTXFrameCodec._decode_varint(content_data)

        # Decode frame type
        if len(content_data) < varint_len + 1:
            raise ValueError("Truncated frame: missing frame type")

        frame_type_byte = content_data[varint_len]
        try:
            frame_type = HTXFrameType(frame_type_byte)
        except ValueError:
            raise ValueError(f"Unknown frame type: 0x{frame_type_byte:02x}")

        # Extract payload
        payload_start = varint_len + 1
        payload = content_data[payload_start:]

        frame = HTXFrame(frame_type=frame_type, stream_id=stream_id, payload=payload)

        logger.debug(f"Decoded HTX frame: type={frame_type.name}, stream={stream_id}, " f"payload_len={len(payload)}")

        return frame, total_frame_size

    @staticmethod
    def _encode_varint(value: int) -> bytes:
        """Encode integer as varint (LEB128-style).

        Args:
            value: Integer to encode (0 to 2^28-1)

        Returns:
            Varint-encoded bytes
        """
        if value < 0:
            raise ValueError(f"Varint cannot be negative: {value}")

        result = bytearray()

        while value >= 0x80:
            result.append((value & 0x7F) | 0x80)
            value >>= 7

        result.append(value & 0x7F)
        return bytes(result)

    @staticmethod
    def _decode_varint(data: bytes) -> tuple[int, int]:
        """Decode varint from bytes.

        Args:
            data: Bytes containing varint

        Returns:
            Tuple of (decoded_value, bytes_consumed)

        Raises:
            ValueError: If varint is invalid or incomplete
        """
        if not data:
            raise ValueError("Empty data for varint decode")

        value = 0
        shift = 0
        bytes_consumed = 0

        for byte in data:
            bytes_consumed += 1

            value |= (byte & 0x7F) << shift

            if (byte & 0x80) == 0:
                # End of varint
                break

            shift += 7
            if shift >= 28:  # Prevent overflow beyond 28 bits
                raise ValueError("Varint too long")

        else:
            # Loop completed without finding end byte
            raise ValueError("Incomplete varint")

        if value > HTXFrameCodec.MAX_STREAM_ID:
            raise ValueError(f"Stream ID too large: {value}")

        return value, bytes_consumed


class HTXFrameBuffer:
    """Buffer for accumulating and parsing HTX frames from stream data."""

    def __init__(self, max_buffer_size: int = 1048576):  # 1MB default
        self.buffer = bytearray()
        self.max_buffer_size = max_buffer_size

    def append_data(self, data: bytes) -> None:
        """Append new data to buffer."""
        if len(self.buffer) + len(data) > self.max_buffer_size:
            raise ValueError(f"Buffer overflow: {len(self.buffer) + len(data)} > {self.max_buffer_size}")

        self.buffer.extend(data)

    def parse_frames(self) -> list[HTXFrame]:
        """Parse all complete frames from buffer.

        Returns:
            List of parsed frames (buffer is consumed for parsed frames)
        """
        frames = []
        offset = 0

        while offset < len(self.buffer):
            try:
                remaining_data = bytes(self.buffer[offset:])
                frame, consumed = HTXFrameCodec.decode_frame(remaining_data)

                if frame is None:
                    # Incomplete frame, need more data
                    break

                frames.append(frame)
                offset += consumed

            except ValueError as e:
                logger.error(f"Frame decode error at offset {offset}: {e}")
                # Skip malformed data - could implement more sophisticated error recovery
                offset += 1

        # Remove consumed bytes from buffer
        if offset > 0:
            self.buffer = self.buffer[offset:]

        return frames

    def get_buffer_info(self) -> dict:
        """Get buffer statistics."""
        return {
            "buffer_size": len(self.buffer),
            "max_buffer_size": self.max_buffer_size,
            "utilization": len(self.buffer) / self.max_buffer_size,
        }


def create_window_update_frame(stream_id: int, window_delta: int) -> bytes:
    """Create WINDOW_UPDATE frame for flow control.

    Args:
        stream_id: Stream to update (0 for connection-level)
        window_delta: Number of bytes to add to window

    Returns:
        Encoded WINDOW_UPDATE frame
    """
    if window_delta <= 0 or window_delta > 0x7FFFFFFF:
        raise ValueError(f"Invalid window delta: {window_delta}")

    payload = struct.pack(">I", window_delta)
    return HTXFrameCodec.encode_frame(HTXFrameType.WINDOW_UPDATE, stream_id, payload)


def create_ping_frame(ping_data: bytes = None) -> bytes:
    """Create PING frame for keepalive.

    Args:
        ping_data: Optional ping payload (defaults to timestamp)

    Returns:
        Encoded PING frame
    """
    if ping_data is None:
        ping_data = struct.pack(">Q", int(time.time() * 1000))  # Timestamp in ms
    elif len(ping_data) > 8:
        raise ValueError("PING payload must be <= 8 bytes")

    return HTXFrameCodec.encode_frame(HTXFrameType.PING, 0, ping_data)


def create_data_frame(stream_id: int, data: bytes) -> bytes:
    """Create DATA frame for payload transmission.

    Args:
        stream_id: Target stream ID
        data: Payload data

    Returns:
        Encoded DATA frame
    """
    if stream_id <= 0:
        raise ValueError("DATA frames require stream_id > 0")

    return HTXFrameCodec.encode_frame(HTXFrameType.DATA, stream_id, data)
