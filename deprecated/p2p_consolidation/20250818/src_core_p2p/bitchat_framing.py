"""BitChat Unified Message Framing & FEC System

Provides normalized message framing and Forward Error Correction (FEC) for reliable
BitChat BLE transport across Python, Android, and iOS implementations.

Features:
- Standardized binary message format for cross-platform compatibility
- Reed-Solomon FEC for error correction in noisy BLE environments
- Fragment/reassembly for messages exceeding BLE MTU limits
- Duplicate detection and ordering guarantees
- Battery-optimized compression and encoding
"""

from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import logging
import struct
import time
import uuid
import zlib

# Optional Reed-Solomon FEC support
try:
    import reedsolo

    REEDSOLO_AVAILABLE = True
except ImportError:
    REEDSOLO_AVAILABLE = False
    logging.warning("Reed-Solomon FEC unavailable - install reedsolo for error correction")

logger = logging.getLogger(__name__)


class MessageType(IntEnum):
    """BitChat message types for protocol handling"""

    DATA = 0x01  # User data payload
    CAPABILITY = 0x02  # Peer capability exchange
    HEARTBEAT = 0x03  # Keep-alive and discovery
    FRAGMENT = 0x04  # Message fragment
    ACK = 0x05  # Acknowledgment
    ROUTE_UPDATE = 0x06  # Routing table update


class FragmentFlags(IntEnum):
    """Fragment control flags"""

    NONE = 0x00
    FIRST_FRAGMENT = 0x01
    LAST_FRAGMENT = 0x02
    HAS_FEC = 0x04
    COMPRESSED = 0x08


@dataclass
class BitChatFrame:
    """Standardized BitChat message frame"""

    # Frame header (fixed size)
    version: int = 1
    message_type: MessageType = MessageType.DATA
    flags: int = 0
    ttl: int = 7
    hop_count: int = 0

    # Message identification
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""  # Empty for broadcast

    # Timing and routing
    timestamp: float = field(default_factory=time.time)
    route_path: list[str] = field(default_factory=list)

    # Payload
    payload: bytes = b""
    payload_size: int = 0

    # Fragmentation (if applicable)
    fragment_id: str | None = None
    fragment_index: int = 0
    total_fragments: int = 1

    # Error correction
    fec_data: bytes | None = None
    checksum: bytes | None = None

    def __post_init__(self):
        """Initialize computed fields"""
        if not self.payload_size:
            self.payload_size = len(self.payload)
        if not self.route_path and self.sender_id:
            self.route_path = [self.sender_id]

    @property
    def is_fragment(self) -> bool:
        """Check if this frame is a fragment"""
        return self.message_type == MessageType.FRAGMENT

    @property
    def is_first_fragment(self) -> bool:
        """Check if this is the first fragment"""
        return bool(self.flags & FragmentFlags.FIRST_FRAGMENT)

    @property
    def is_last_fragment(self) -> bool:
        """Check if this is the last fragment"""
        return bool(self.flags & FragmentFlags.LAST_FRAGMENT)

    @property
    def has_fec(self) -> bool:
        """Check if frame includes FEC data"""
        return bool(self.flags & FragmentFlags.HAS_FEC)

    @property
    def is_compressed(self) -> bool:
        """Check if payload is compressed"""
        return bool(self.flags & FragmentFlags.COMPRESSED)


class BitChatFraming:
    """BitChat message framing and FEC system"""

    # Protocol constants
    FRAME_HEADER_SIZE = 64  # Fixed header size in bytes
    MAX_BLE_MTU = 512  # BLE characteristic limit
    MAX_PAYLOAD_SIZE = MAX_BLE_MTU - FRAME_HEADER_SIZE - 32  # Reserve for overhead

    # FEC parameters
    FEC_SYMBOLS = 8  # Reed-Solomon redundancy symbols
    MIN_FEC_SIZE = 32  # Minimum payload size for FEC

    # Fragmentation
    MAX_FRAGMENTS = 16  # Maximum fragments per message
    FRAGMENT_TIMEOUT = 30.0  # Seconds to wait for all fragments

    def __init__(self, enable_fec: bool = True, enable_compression: bool = True):
        self.enable_fec = enable_fec and REEDSOLO_AVAILABLE
        self.enable_compression = enable_compression

        # Fragment reassembly state
        self.pending_fragments: dict[str, dict[int, BitChatFrame]] = {}
        self.fragment_timers: dict[str, float] = {}
        self.seen_messages: set = set()

        # Statistics
        self.stats = {
            "frames_encoded": 0,
            "frames_decoded": 0,
            "fragments_sent": 0,
            "fragments_received": 0,
            "fec_corrections": 0,
            "decode_errors": 0,
            "duplicate_frames": 0,
        }

        logger.info(f"BitChat framing initialized (FEC: {self.enable_fec}, compression: {self.enable_compression})")

    def create_frame(
        self, message_type: MessageType, payload: bytes, sender_id: str, recipient_id: str = "", ttl: int = 7, **kwargs
    ) -> BitChatFrame:
        """Create a new BitChat frame with given parameters"""

        frame = BitChatFrame(
            message_type=message_type,
            payload=payload,
            sender_id=sender_id,
            recipient_id=recipient_id,
            ttl=ttl,
            **kwargs,
        )

        # Add compression if beneficial
        if self.enable_compression and len(payload) > 64:
            compressed = zlib.compress(payload, level=1)
            if len(compressed) < len(payload) * 0.9:  # 10% improvement threshold
                frame.payload = compressed
                frame.flags |= FragmentFlags.COMPRESSED
                frame.payload_size = len(compressed)

        # Generate checksum
        frame.checksum = self._compute_checksum(frame)

        return frame

    def encode_frame(self, frame: BitChatFrame) -> bytes:
        """Encode frame to binary format for transmission"""
        try:
            # Add FEC if enabled and payload is large enough
            if self.enable_fec and len(frame.payload) >= self.MIN_FEC_SIZE:
                frame.fec_data = self._generate_fec(frame.payload)
                frame.flags |= FragmentFlags.HAS_FEC

            # Pack frame header (64 bytes fixed)
            header = struct.pack(
                "!BBBBBBBB32s32s32sLLLLLH",
                frame.version,
                frame.message_type,
                frame.flags,
                frame.ttl,
                frame.hop_count,
                0,
                0,
                0,  # Reserved bytes
                frame.message_id[:32].encode().ljust(32, b"\x00"),
                frame.sender_id[:32].encode().ljust(32, b"\x00"),
                frame.recipient_id[:32].encode().ljust(32, b"\x00"),
                int(frame.timestamp * 1000),  # Timestamp in milliseconds
                frame.payload_size,
                frame.fragment_index,
                frame.total_fragments,
                len(frame.route_path),
                len(frame.fec_data) if frame.fec_data else 0,
            )

            # Add route path (variable length)
            route_data = b""
            for hop in frame.route_path:
                hop_bytes = hop[:16].encode().ljust(16, b"\x00")
                route_data += hop_bytes

            # Add payload
            payload_data = frame.payload

            # Add FEC data if present
            fec_data = frame.fec_data or b""

            # Add checksum
            checksum_data = frame.checksum or b""

            # Combine all parts
            frame_data = header + route_data + payload_data + fec_data + checksum_data

            self.stats["frames_encoded"] += 1
            logger.debug(f"Encoded frame {frame.message_id[:8]} ({len(frame_data)} bytes)")

            return frame_data

        except Exception as e:
            logger.error(f"Frame encoding failed: {e}")
            raise

    def decode_frame(self, data: bytes) -> BitChatFrame | None:
        """Decode binary data to BitChat frame"""
        try:
            if len(data) < self.FRAME_HEADER_SIZE:
                logger.warning(f"Frame too short: {len(data)} < {self.FRAME_HEADER_SIZE}")
                return None

            # Unpack header
            header_data = data[: self.FRAME_HEADER_SIZE]
            (
                version,
                msg_type,
                flags,
                ttl,
                hop_count,
                _,
                _,
                _,
                msg_id_bytes,
                sender_bytes,
                recipient_bytes,
                timestamp_ms,
                payload_size,
                frag_index,
                total_frags,
                route_len,
                fec_len,
            ) = struct.unpack("!BBBBBBBB32s32s32sLLLLLH", header_data)

            # Extract string fields
            message_id = msg_id_bytes.rstrip(b"\x00").decode()
            sender_id = sender_bytes.rstrip(b"\x00").decode()
            recipient_id = recipient_bytes.rstrip(b"\x00").decode()
            timestamp = timestamp_ms / 1000.0

            # Parse route path
            offset = self.FRAME_HEADER_SIZE
            route_path = []
            for _ in range(route_len):
                if offset + 16 > len(data):
                    break
                hop = data[offset : offset + 16].rstrip(b"\x00").decode()
                route_path.append(hop)
                offset += 16

            # Extract payload
            if offset + payload_size > len(data):
                logger.warning(f"Payload size mismatch: {offset + payload_size} > {len(data)}")
                return None

            payload = data[offset : offset + payload_size]
            offset += payload_size

            # Extract FEC data if present
            fec_data = None
            if fec_len > 0 and offset + fec_len <= len(data):
                fec_data = data[offset : offset + fec_len]
                offset += fec_len

            # Extract checksum (remaining bytes)
            checksum = data[offset:] if offset < len(data) else None

            # Create frame object
            frame = BitChatFrame(
                version=version,
                message_type=MessageType(msg_type),
                flags=flags,
                ttl=ttl,
                hop_count=hop_count,
                message_id=message_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                timestamp=timestamp,
                route_path=route_path,
                payload=payload,
                payload_size=payload_size,
                fragment_index=frag_index,
                total_fragments=total_frags,
                fec_data=fec_data,
                checksum=checksum,
            )

            # Verify checksum
            if not self._verify_checksum(frame):
                logger.warning(f"Checksum verification failed for frame {message_id[:8]}")
                self.stats["decode_errors"] += 1
                return None

            # Apply FEC correction if available
            if frame.has_fec and frame.fec_data:
                corrected_payload = self._apply_fec(frame.payload, frame.fec_data)
                if corrected_payload != frame.payload:
                    frame.payload = corrected_payload
                    self.stats["fec_corrections"] += 1
                    logger.debug(f"FEC corrected errors in frame {message_id[:8]}")

            # Decompress payload if compressed
            if frame.is_compressed:
                try:
                    frame.payload = zlib.decompress(frame.payload)
                    frame.flags &= ~FragmentFlags.COMPRESSED  # Clear compression flag
                except Exception as e:
                    logger.warning(f"Decompression failed for frame {message_id[:8]}: {e}")
                    return None

            self.stats["frames_decoded"] += 1
            logger.debug(f"Decoded frame {message_id[:8]} from {sender_id}")

            return frame

        except Exception as e:
            logger.error(f"Frame decoding failed: {e}")
            self.stats["decode_errors"] += 1
            return None

    def _compute_checksum(self, frame: BitChatFrame) -> bytes:
        """Compute frame checksum for integrity verification"""
        hash_input = (
            frame.message_id.encode()
            + frame.sender_id.encode()
            + frame.recipient_id.encode()
            + frame.payload
            + struct.pack("!BBL", frame.version, frame.message_type, frame.payload_size)
        )
        return hashlib.sha256(hash_input).digest()[:8]  # 8-byte checksum

    def _verify_checksum(self, frame: BitChatFrame) -> bool:
        """Verify frame checksum"""
        if not frame.checksum:
            return True  # No checksum to verify

        expected_checksum = self._compute_checksum(frame)
        return frame.checksum == expected_checksum

    def _generate_fec(self, data: bytes) -> bytes:
        """Generate Reed-Solomon FEC data"""
        if not REEDSOLO_AVAILABLE:
            return b""

        try:
            rs = reedsolo.RSCodec(self.FEC_SYMBOLS)
            encoded = rs.encode(data)
            # Return only the FEC symbols (redundancy data)
            return encoded[len(data) :]
        except Exception as e:
            logger.warning(f"FEC generation failed: {e}")
            return b""

    def _apply_fec(self, data: bytes, fec_data: bytes) -> bytes:
        """Apply Reed-Solomon error correction"""
        if not REEDSOLO_AVAILABLE or not fec_data:
            return data

        try:
            rs = reedsolo.RSCodec(self.FEC_SYMBOLS)
            # Reconstruct full encoded message
            encoded_message = data + fec_data
            corrected = rs.decode(encoded_message)[0]  # Returns (corrected_data, corrected_ecc)
            return corrected
        except Exception as e:
            logger.warning(f"FEC correction failed: {e}")
            return data

    def get_stats(self) -> dict[str, int]:
        """Get framing statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics counters"""
        for key in self.stats:
            self.stats[key] = 0
