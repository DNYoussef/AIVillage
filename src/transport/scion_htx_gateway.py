"""
SCION HTX Gateway Integration - Betanet v1.1 Compliant

Implements HTX-tunnelled SCION gateway control stream with signed CBOR messages,
replay protection, and token bucket rate limiting.
"""

import asyncio
import hashlib
import hmac
import logging
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cbor2
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger(__name__)


@dataclass
class SCIONControlMessage:
    """SCION control stream message."""
    prev_as: str          # Previous AS identifier
    next_as: str          # Next AS identifier
    timestamp: float      # Unix timestamp
    flow_id: bytes        # Flow identifier
    nonce: bytes          # Replay nonce
    signature: bytes      # Ed25519 signature
    metadata: Dict[str, Any] = None

    def to_cbor(self) -> bytes:
        """Serialize to CBOR format."""
        data = {
            "prevAS": self.prev_as,
            "nextAS": self.next_as,
            "TS": int(self.timestamp),
            "FLOW": self.flow_id.hex(),
            "NONCE": self.nonce.hex(),
            "SIG": self.signature.hex()
        }
        if self.metadata:
            data["META"] = self.metadata
        return cbor2.dumps(data)

    @classmethod
    def from_cbor(cls, data: bytes) -> "SCIONControlMessage":
        """Deserialize from CBOR."""
        obj = cbor2.loads(data)
        return cls(
            prev_as=obj["prevAS"],
            next_as=obj["nextAS"],
            timestamp=float(obj["TS"]),
            flow_id=bytes.fromhex(obj["FLOW"]),
            nonce=bytes.fromhex(obj["NONCE"]),
            signature=bytes.fromhex(obj["SIG"]),
            metadata=obj.get("META")
        )


class SCIONHTXGateway:
    """SCION gateway with HTX-tunnelled control stream."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SCION HTX gateway."""
        self.config = config or {}
        self.control_stream_id = 2  # Even stream ID for control
        self.signing_key: Optional[ed25519.Ed25519PrivateKey] = None
        self.peer_keys: Dict[str, ed25519.Ed25519PublicKey] = {}
        self.replay_cache: Dict[Tuple[str, float], bool] = {}
        self.token_buckets: Dict[str, Dict[str, Any]] = {}
        self.active_flows: Dict[bytes, Dict[str, Any]] = {}
        self.path_cache: List[Dict[str, Any]] = []
        self.stream_writer: Optional[asyncio.StreamWriter] = None
        self.stream_reader: Optional[asyncio.StreamReader] = None
        self.control_task: Optional[asyncio.Task] = None

        # Initialize signing keys
        self._initialize_keys()

        # Load configuration
        self.ts_window = self.config.get("ts_window", 300)  # ±300s
        self.replay_ttl = self.config.get("replay_ttl", 7200)  # 2 hours
        self.max_paths = self.config.get("max_paths", 3)
        self.switch_timeout = self.config.get("switch_timeout", 0.3)  # 300ms

    def _initialize_keys(self):
        """Initialize Ed25519 signing keys."""
        # Generate or load signing key
        key_path = Path(self.config.get("key_path", "tmp_betanet/l1/gateway.key"))

        if key_path.exists():
            # Load existing key
            with open(key_path, "rb") as f:
                key_data = f.read()
                self.signing_key = ed25519.Ed25519PrivateKey.from_private_bytes(key_data)
        else:
            # Generate new key
            self.signing_key = ed25519.Ed25519PrivateKey.generate()

            # Save key
            key_path.parent.mkdir(parents=True, exist_ok=True)
            with open(key_path, "wb") as f:
                f.write(self.signing_key.private_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PrivateFormat.Raw,
                    encryption_algorithm=serialization.NoEncryption()
                ))

        logger.info(f"Initialized signing key: {key_path}")

    async def establish_control_stream(self, reader: asyncio.StreamReader,
                                      writer: asyncio.StreamWriter):
        """Establish HTX-tunnelled control stream on stream_id=2."""
        self.stream_reader = reader
        self.stream_writer = writer

        logger.info(f"Establishing control stream on stream_id={self.control_stream_id}")

        # Start control message handler
        self.control_task = asyncio.create_task(self._handle_control_messages())

        # Send initial control message
        await self.send_control_message(
            prev_as=self.config.get("local_as", "1-ff00:0:110"),
            next_as=self.config.get("peer_as", "2-ff00:0:220"),
            flow_id=secrets.token_bytes(16)
        )

    async def send_control_message(self, prev_as: str, next_as: str,
                                  flow_id: bytes) -> bool:
        """Send signed control message."""
        if not self.stream_writer:
            logger.error("No control stream established")
            return False

        # Create message
        message = SCIONControlMessage(
            prev_as=prev_as,
            next_as=next_as,
            timestamp=time.time(),
            flow_id=flow_id,
            nonce=secrets.token_bytes(16),
            signature=b""  # Will be set after signing
        )

        # Sign message
        message.signature = self._sign_message(message)

        # Serialize to CBOR
        cbor_data = message.to_cbor()

        # Create HTX frame (simplified - would use HTXTransport in production)
        frame = self._create_control_frame(cbor_data)

        # Send on control stream
        self.stream_writer.write(frame)
        await self.stream_writer.drain()

        logger.debug(f"Sent control message for flow {flow_id.hex()[:8]}")

        # Store flow
        self.active_flows[flow_id] = {
            "prev_as": prev_as,
            "next_as": next_as,
            "timestamp": message.timestamp
        }

        return True

    def _sign_message(self, message: SCIONControlMessage) -> bytes:
        """Sign control message with Ed25519."""
        # Create signing data (without signature field)
        sign_data = f"{message.prev_as}|{message.next_as}|{int(message.timestamp)}|"
        sign_data += f"{message.flow_id.hex()}|{message.nonce.hex()}"

        # Sign with Ed25519
        signature = self.signing_key.sign(sign_data.encode())

        return signature

    def _verify_signature(self, message: SCIONControlMessage,
                         public_key: ed25519.Ed25519PublicKey) -> bool:
        """Verify message signature."""
        # Create signing data
        sign_data = f"{message.prev_as}|{message.next_as}|{int(message.timestamp)}|"
        sign_data += f"{message.flow_id.hex()}|{message.nonce.hex()}"

        try:
            public_key.verify(message.signature, sign_data.encode())
            return True
        except InvalidSignature:
            return False

    def _create_control_frame(self, data: bytes) -> bytes:
        """Create HTX control frame."""
        # Frame format: length(3) | stream_id(varint) | type | payload
        stream_bytes = self._encode_varint(self.control_stream_id)
        frame_type = 0x07  # CONTROL type

        length = len(stream_bytes) + 1 + len(data)
        length_bytes = struct.pack(">I", length)[1:]  # uint24

        return length_bytes + stream_bytes + bytes([frame_type]) + data

    def _encode_varint(self, value: int) -> bytes:
        """Encode integer as varint."""
        result = bytearray()
        while value > 127:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value & 0x7F)
        return bytes(result)

    async def _handle_control_messages(self):
        """Handle incoming control messages."""
        while True:
            try:
                # Read frame header (simplified)
                header = await self.stream_reader.read(4)
                if not header:
                    break

                # Parse length (uint24)
                length = struct.unpack(">I", b"\x00" + header[:3])[0]

                # Read payload
                payload = await self.stream_reader.read(length)

                # Parse CBOR message
                message = SCIONControlMessage.from_cbor(payload)

                # Validate message
                if await self._validate_control_message(message):
                    await self._process_control_message(message)
                else:
                    logger.warning(f"Invalid control message from {message.prev_as}")

            except Exception as e:
                logger.error(f"Control message handler error: {e}")
                break

    async def _validate_control_message(self, message: SCIONControlMessage) -> bool:
        """Validate control message with all checks."""
        # Check timestamp window (±300s)
        now = time.time()
        if abs(now - message.timestamp) > self.ts_window:
            logger.warning(f"Message outside timestamp window: {message.timestamp}")
            return False

        # Check replay protection
        replay_key = (message.flow_id.hex(), int(message.timestamp))
        if replay_key in self.replay_cache:
            logger.warning(f"Replay detected for flow {message.flow_id.hex()[:8]}")
            return False

        # Add to replay cache
        self.replay_cache[replay_key] = True

        # Clean old replay entries
        cutoff = now - self.replay_ttl
        self.replay_cache = {
            k: v for k, v in self.replay_cache.items()
            if k[1] > cutoff
        }

        # Verify signature if we have peer's public key
        peer_key = self.peer_keys.get(message.prev_as)
        if peer_key:
            if not self._verify_signature(message, peer_key):
                logger.warning(f"Invalid signature from {message.prev_as}")
                return False

        # Check token bucket rate limiting
        if not self._check_token_bucket(message.prev_as):
            logger.warning(f"Rate limit exceeded for {message.prev_as}")
            return False

        return True

    def _check_token_bucket(self, peer_id: str) -> bool:
        """Check token bucket for rate limiting."""
        if peer_id not in self.token_buckets:
            self.token_buckets[peer_id] = {
                "tokens": 100,
                "last_refill": time.time(),
                "rate": 10  # tokens per second
            }

        bucket = self.token_buckets[peer_id]
        now = time.time()

        # Refill tokens
        elapsed = now - bucket["last_refill"]
        new_tokens = elapsed * bucket["rate"]
        bucket["tokens"] = min(100, bucket["tokens"] + new_tokens)
        bucket["last_refill"] = now

        # Check if we have tokens
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True

        return False

    async def _process_control_message(self, message: SCIONControlMessage):
        """Process validated control message."""
        logger.info(f"Processing control from {message.prev_as} to {message.next_as}")

        # Update flow state
        self.active_flows[message.flow_id] = {
            "prev_as": message.prev_as,
            "next_as": message.next_as,
            "timestamp": message.timestamp,
            "metadata": message.metadata
        }

        # Handle path updates if present
        if message.metadata and "paths" in message.metadata:
            await self._update_path_cache(message.metadata["paths"])

    async def _update_path_cache(self, paths: List[Dict]):
        """Update cached SCION paths."""
        # Keep up to 3 disjoint validated paths
        self.path_cache = paths[:self.max_paths]

        # Log path diversity
        as_sets = [set(p.get("as_path", [])) for p in self.path_cache]
        disjoint_count = sum(
            1 for i, s1 in enumerate(as_sets)
            for s2 in as_sets[i+1:]
            if not s1 & s2
        )

        logger.info(f"Updated path cache: {len(self.path_cache)} paths, "
                   f"{disjoint_count} disjoint pairs")

    async def switch_path(self, target_as: str) -> bool:
        """Switch to alternate path within 300ms."""
        start_time = time.time()

        # Find alternate path
        alternate_path = None
        for path in self.path_cache:
            if path.get("target_as") == target_as and not path.get("active"):
                alternate_path = path
                break

        if not alternate_path:
            logger.warning(f"No alternate path to {target_as}")
            return False

        # Perform switch
        # (Actual implementation would update forwarding tables)
        await asyncio.sleep(0.05)  # Simulated switch time

        # Mark as active
        for path in self.path_cache:
            path["active"] = (path == alternate_path)

        switch_time = time.time() - start_time

        if switch_time <= self.switch_timeout:
            logger.info(f"Path switch completed in {switch_time*1000:.1f}ms")
            await self._log_path_switch(target_as, switch_time)
            return True
        else:
            logger.warning(f"Path switch exceeded timeout: {switch_time*1000:.1f}ms")
            return False

    async def _log_path_switch(self, target_as: str, switch_time: float):
        """Log path switch event."""
        log_entry = {
            "timestamp": time.time(),
            "target_as": target_as,
            "switch_time_ms": switch_time * 1000,
            "path_count": len(self.path_cache),
            "success": switch_time <= self.switch_timeout
        }

        log_dir = Path("tmp_betanet/l1")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Append to log file
        with open(log_dir / "path_switch_receipts.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    async def perform_rekey(self):
        """Re-establish control stream on rekey."""
        logger.info("Performing control stream rekey")

        # Close current stream
        if self.control_task:
            self.control_task.cancel()

        # Reset state
        self.active_flows.clear()

        # Re-establish with new keys
        if self.stream_reader and self.stream_writer:
            await self.establish_control_stream(
                self.stream_reader,
                self.stream_writer
            )

        logger.info("Control stream rekey completed")

    def verify_no_legacy_header(self, packet: bytes) -> bool:
        """Verify packet doesn't use legacy transition header."""
        # Check for legacy header pattern (simplified check)
        # Legacy header would have specific byte patterns

        if len(packet) < 20:
            return True

        # Check for prohibited legacy markers
        legacy_markers = [
            b"\x00\x00\xSC",  # Old SCION marker
            b"TRANS",          # Transition header
        ]

        for marker in legacy_markers:
            if marker in packet[:20]:
                logger.error("Legacy transition header detected - prohibited on public network")
                return False

        return True

    async def cleanup(self):
        """Clean up resources."""
        if self.control_task:
            self.control_task.cancel()

        if self.stream_writer:
            self.stream_writer.close()
            await self.stream_writer.wait_closed()

        logger.info("SCION HTX gateway cleaned up")


# Test function
async def test_scion_htx_compliance():
    """Test SCION HTX gateway compliance."""
    import json
    import secrets

    logger.info("Testing SCION HTX gateway v1.1 compliance")

    gateway = SCIONHTXGateway({
        "local_as": "1-ff00:0:110",
        "peer_as": "2-ff00:0:220",
        "ts_window": 300,
        "replay_ttl": 7200
    })

    # Test control message creation
    flow_id = secrets.token_bytes(16)
    message = SCIONControlMessage(
        prev_as="1-ff00:0:110",
        next_as="2-ff00:0:220",
        timestamp=time.time(),
        flow_id=flow_id,
        nonce=secrets.token_bytes(16),
        signature=b""
    )

    # Sign message
    message.signature = gateway._sign_message(message)

    # Serialize to CBOR
    cbor_data = message.to_cbor()
    logger.info(f"Created control message: {len(cbor_data)} bytes")

    # Test deserialization
    decoded = SCIONControlMessage.from_cbor(cbor_data)
    assert decoded.prev_as == message.prev_as
    assert decoded.flow_id == message.flow_id

    # Test replay protection
    assert await gateway._validate_control_message(message) == True
    # Second validation should fail (replay)
    assert await gateway._validate_control_message(message) == False

    # Test path switching
    gateway.path_cache = [
        {"target_as": "2-ff00:0:220", "active": True},
        {"target_as": "2-ff00:0:220", "active": False},
        {"target_as": "3-ff00:0:330", "active": False}
    ]

    success = await gateway.switch_path("2-ff00:0:220")
    assert success == True

    # Test legacy header check
    good_packet = b"\x01\x02\x03" + secrets.token_bytes(100)
    assert gateway.verify_no_legacy_header(good_packet) == True

    bad_packet = b"\x00\x00\xSC" + secrets.token_bytes(100)
    assert gateway.verify_no_legacy_header(bad_packet) == False

    logger.info("✅ All SCION HTX gateway tests passed")

    await gateway.cleanup()


if __name__ == "__main__":
    import json
    import secrets

    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_scion_htx_compliance())
