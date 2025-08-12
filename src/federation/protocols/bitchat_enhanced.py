"""Enhanced BitChat Extensions - Builds on existing implementation

Extends existing BitChatTransport with Jack Dorsey's BitChat specification:
- X25519/Ed25519 cryptography
- Message fragmentation for 500-byte BLE limit  
- LZ4 compression for messages >100 bytes
- Dummy traffic injection for privacy
- IRC-style channels
- 12-hour store-and-forward TTL
"""

import asyncio
import json
import logging
import os
import random
import struct
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

# Import existing BitChat implementation
from core.p2p.bitchat_transport import BitChatMessage, BitChatPeer, BitChatTransport

# Compression
try:
    import lz4.frame

    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    logging.warning("LZ4 not available - BitChat compression disabled")

# Enhanced cryptography
try:
    import nacl.encoding
    import nacl.public
    import nacl.secret
    import nacl.signing
    import nacl.utils

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("PyNaCl not available - BitChat crypto disabled")

logger = logging.getLogger(__name__)


class BitChatMessageType(Enum):
    """Enhanced BitChat message types"""

    DATA = 0x01
    FRAGMENT = 0x02
    ACK = 0x03
    HELLO = 0x04
    CHANNEL_JOIN = 0x05
    CHANNEL_LEAVE = 0x06
    CHANNEL_MSG = 0x07
    DUMMY = 0x08
    KEY_EXCHANGE = 0x09


@dataclass
class BitChatCryptoKeys:
    """Cryptographic keys for enhanced BitChat"""

    # X25519 for encryption
    private_key: bytes = field(default_factory=lambda: os.urandom(32))
    public_key: Optional[bytes] = None

    # Ed25519 for signing
    signing_key: Optional["nacl.signing.SigningKey"] = None
    verify_key: Optional["nacl.signing.VerifyKey"] = None

    def __post_init__(self):
        if CRYPTO_AVAILABLE:
            # Generate X25519 keypair
            private_key_obj = nacl.public.PrivateKey(self.private_key)
            self.public_key = bytes(private_key_obj.public_key)

            # Generate Ed25519 keypair
            if not self.signing_key:
                self.signing_key = nacl.signing.SigningKey.generate()
            self.verify_key = self.signing_key.verify_key
        else:
            # Fallback for testing
            self.public_key = b"mock_public_key_32_bytes_long___"


@dataclass
class BitChatFragment:
    """Message fragment for BLE 500-byte limit"""

    fragment_id: str
    total_fragments: int
    fragment_index: int
    data: bytes
    checksum: int

    def to_bytes(self) -> bytes:
        """Serialize fragment for BLE transmission"""
        # Pack: 16-byte fragment_id + 2-byte total + 2-byte index + 4-byte checksum
        header = struct.pack(
            "!16sHHI",
            self.fragment_id.encode()[:16].ljust(16, b"\x00"),
            self.total_fragments,
            self.fragment_index,
            self.checksum,
        )
        return header + self.data

    @classmethod
    def from_bytes(cls, data: bytes) -> "BitChatFragment":
        """Deserialize fragment from BLE data"""
        if len(data) < 24:  # Minimum header size
            raise ValueError("Fragment data too short")

        header = struct.unpack("!16sHHI", data[:24])
        fragment_id = header[0].rstrip(b"\x00").decode()

        return cls(
            fragment_id=fragment_id,
            total_fragments=header[1],
            fragment_index=header[2],
            checksum=header[3],
            data=data[24:],
        )


@dataclass
class EnhancedBitChatMessage(BitChatMessage):
    """Enhanced BitChat message with crypto and fragmentation"""

    message_type: BitChatMessageType = BitChatMessageType.DATA
    channel: str = ""
    compressed: bool = False
    encrypted: bool = False
    signed: bool = False
    expires_at: Optional[float] = None
    fragments: List[BitChatFragment] = field(default_factory=list)
    is_fragmented: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Set 12-hour expiration for store-and-forward
        if not self.expires_at:
            self.expires_at = time.time() + (12 * 3600)

    def compress_payload(self) -> bool:
        """Compress payload if >100 bytes using LZ4"""
        if not LZ4_AVAILABLE or self.compressed or len(self.payload) <= 100:
            return False

        try:
            compressed = lz4.frame.compress(self.payload)
            # Only use if compression saves space
            if len(compressed) < len(self.payload):
                original_size = len(self.payload)
                self.payload = compressed
                self.compressed = True
                compression_ratio = (1 - len(compressed) / original_size) * 100
                logger.debug(
                    f"Compressed message {self.id[:8]}: {original_size} -> {len(compressed)} bytes ({compression_ratio:.1f}%)"
                )
                return True
        except Exception as e:
            logger.warning(f"LZ4 compression failed: {e}")

        return False

    def decompress_payload(self) -> bool:
        """Decompress LZ4 compressed payload"""
        if not LZ4_AVAILABLE or not self.compressed:
            return True

        try:
            self.payload = lz4.frame.decompress(self.payload)
            self.compressed = False
            return True
        except Exception as e:
            logger.error(f"LZ4 decompression failed: {e}")
            return False

    def encrypt_payload(
        self, recipient_public_key: bytes, sender_private_key: bytes
    ) -> bool:
        """Encrypt payload using X25519"""
        if not CRYPTO_AVAILABLE or self.encrypted:
            return False

        try:
            sender_key = nacl.public.PrivateKey(sender_private_key)
            recipient_key = nacl.public.PublicKey(recipient_public_key)
            box = nacl.public.Box(sender_key, recipient_key)

            self.payload = box.encrypt(self.payload)
            self.encrypted = True
            logger.debug(f"Encrypted message {self.id[:8]} with X25519")
            return True
        except Exception as e:
            logger.error(f"X25519 encryption failed: {e}")
            return False

    def decrypt_payload(
        self, sender_public_key: bytes, recipient_private_key: bytes
    ) -> bool:
        """Decrypt payload using X25519"""
        if not CRYPTO_AVAILABLE or not self.encrypted:
            return False

        try:
            recipient_key = nacl.public.PrivateKey(recipient_private_key)
            sender_key = nacl.public.PublicKey(sender_public_key)
            box = nacl.public.Box(recipient_key, sender_key)

            self.payload = box.decrypt(self.payload)
            self.encrypted = False
            logger.debug(f"Decrypted message {self.id[:8]} with X25519")
            return True
        except Exception as e:
            logger.error(f"X25519 decryption failed: {e}")
            return False

    def sign_message(self, signing_key: "nacl.signing.SigningKey") -> bool:
        """Sign message using Ed25519"""
        if not CRYPTO_AVAILABLE or self.signed:
            return False

        try:
            # Create message digest for signing
            signable_data = self._get_signable_data()
            signature = signing_key.sign(signable_data).signature

            # Prepend 64-byte signature to payload
            self.payload = signature + self.payload
            self.signed = True
            logger.debug(f"Signed message {self.id[:8]} with Ed25519")
            return True
        except Exception as e:
            logger.error(f"Ed25519 signing failed: {e}")
            return False

    def verify_signature(self, verify_key: "nacl.signing.VerifyKey") -> bool:
        """Verify Ed25519 signature"""
        if not CRYPTO_AVAILABLE or not self.signed:
            return True  # No signature to verify

        try:
            if len(self.payload) < 64:
                return False

            # Extract signature and original payload
            signature = self.payload[:64]
            original_payload = self.payload[64:]

            # Temporarily restore payload for verification
            temp_payload = self.payload
            temp_signed = self.signed

            self.payload = original_payload
            self.signed = False

            # Verify signature against message digest
            signable_data = self._get_signable_data()
            verify_key.verify(signable_data, signature)

            # Restore original state
            self.payload = temp_payload
            self.signed = temp_signed

            logger.debug(f"Verified signature for message {self.id[:8]}")
            return True
        except Exception as e:
            logger.error(f"Ed25519 verification failed: {e}")
            return False

    def fragment_message(self, max_fragment_size: int = 450) -> List[BitChatFragment]:
        """Fragment message for 500-byte BLE limit"""
        if not self.payload:
            return []

        # Calculate usable fragment size (account for 24-byte header + BLE overhead)
        fragment_data_size = max_fragment_size - 24
        total_size = len(self.payload)

        if total_size <= fragment_data_size:
            return []  # No fragmentation needed

        total_fragments = (total_size + fragment_data_size - 1) // fragment_data_size
        fragment_id = str(uuid.uuid4())
        fragments = []

        for i in range(total_fragments):
            start = i * fragment_data_size
            end = min(start + fragment_data_size, total_size)
            chunk = self.payload[start:end]

            fragment = BitChatFragment(
                fragment_id=fragment_id,
                total_fragments=total_fragments,
                fragment_index=i,
                data=chunk,
                checksum=hash(chunk) & 0xFFFFFFFF,
            )
            fragments.append(fragment)

        self.fragments = fragments
        self.is_fragmented = True

        logger.debug(f"Fragmented message {self.id[:8]} into {total_fragments} pieces")
        return fragments

    @classmethod
    def reassemble_from_fragments(
        cls, fragments: List[BitChatFragment]
    ) -> Optional["EnhancedBitChatMessage"]:
        """Reassemble message from fragments"""
        if not fragments:
            return None

        # Sort by fragment index
        fragments.sort(key=lambda f: f.fragment_index)

        # Verify we have all fragments
        expected_count = fragments[0].total_fragments
        if len(fragments) != expected_count:
            logger.warning(
                f"Missing fragments: have {len(fragments)}, need {expected_count}"
            )
            return None

        # Verify fragment sequence and checksums
        payload = b""
        for i, fragment in enumerate(fragments):
            if fragment.fragment_index != i:
                logger.error(
                    f"Fragment sequence error: expected {i}, got {fragment.fragment_index}"
                )
                return None

            # Verify checksum
            if hash(fragment.data) & 0xFFFFFFFF != fragment.checksum:
                logger.error(f"Fragment checksum mismatch for fragment {i}")
                return None

            payload += fragment.data

        # Create reassembled message
        message = cls(id=fragments[0].fragment_id, payload=payload, is_fragmented=True)

        logger.debug(
            f"Reassembled message {message.id[:8]} from {len(fragments)} fragments"
        )
        return message

    def _get_signable_data(self) -> bytes:
        """Get data to be signed (message metadata + payload)"""
        signable = f"{self.id}{self.sender}{self.recipient}{self.channel}{self.timestamp}".encode()
        return signable + self.payload


@dataclass
class BitChatChannel:
    """IRC-style channel for group messaging"""

    name: str
    creator: str
    members: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    message_history: List[EnhancedBitChatMessage] = field(default_factory=list)
    max_history: int = 100

    def __post_init__(self):
        self.members.add(self.creator)

    def add_member(self, device_id: str) -> bool:
        """Add member to channel"""
        if device_id not in self.members:
            self.members.add(device_id)
            logger.info(f"Added {device_id} to channel #{self.name}")
            return True
        return False

    def remove_member(self, device_id: str) -> bool:
        """Remove member from channel"""
        if device_id in self.members:
            self.members.remove(device_id)
            logger.info(f"Removed {device_id} from channel #{self.name}")
            return True
        return False

    def add_message(self, message: EnhancedBitChatMessage):
        """Add message to channel history"""
        self.message_history.append(message)

        # Trim history if too long
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)


class EnhancedBitChatTransport(BitChatTransport):
    """Enhanced BitChat transport extending existing implementation"""

    def __init__(self, device_id: str = None, enable_crypto: bool = True):
        super().__init__(device_id)

        self.enable_crypto = enable_crypto

        # Cryptographic keys
        self.crypto_keys = BitChatCryptoKeys()

        # Enhanced message handling
        self.enhanced_handlers: Dict[BitChatMessageType, Callable] = {}
        self.pending_fragments: Dict[str, Dict[int, BitChatFragment]] = defaultdict(
            dict
        )

        # Store-and-forward with 12-hour TTL
        self.store_forward_cache: Dict[str, List[EnhancedBitChatMessage]] = defaultdict(
            list
        )
        self.cache_max_messages = 1000

        # IRC-style channels
        self.channels: Dict[str, BitChatChannel] = {}
        self.joined_channels: Set[str] = set()

        # Dummy traffic for privacy
        self.dummy_traffic_enabled = True
        self.dummy_traffic_task: Optional[asyncio.Task] = None

        # Peer key exchange
        self.peer_keys: Dict[
            str, Dict[str, bytes]
        ] = {}  # device_id -> {public_key, verify_key}

        # Enhanced statistics
        self.enhanced_stats = {
            "fragments_sent": 0,
            "fragments_received": 0,
            "messages_compressed": 0,
            "messages_encrypted": 0,
            "dummy_traffic_sent": 0,
            "channels_joined": 0,
        }

        logger.info(f"Enhanced BitChat initialized with crypto: {enable_crypto}")

    async def start(self) -> bool:
        """Start enhanced BitChat transport"""
        success = await super().start()

        if not success:
            return False

        # Register enhanced message handlers
        self._register_enhanced_handlers()

        # Start dummy traffic generation
        if self.dummy_traffic_enabled:
            self.dummy_traffic_task = asyncio.create_task(self._dummy_traffic_loop())

        # Start maintenance tasks
        asyncio.create_task(self._enhanced_maintenance_loop())

        # Send initial hello with crypto keys
        await self._send_crypto_hello()

        logger.info("Enhanced BitChat transport started")
        return True

    async def stop(self):
        """Stop enhanced BitChat transport"""
        if self.dummy_traffic_task:
            self.dummy_traffic_task.cancel()

        await super().stop()
        logger.info("Enhanced BitChat transport stopped")

    async def send_enhanced_message(
        self,
        recipient: str,
        payload: bytes,
        channel: str = "",
        encrypt: bool = True,
        compress: bool = True,
        priority: int = 5,
    ) -> bool:
        """Send message with all enhancements"""

        if not self.is_running:
            return False

        # Create enhanced message
        message = EnhancedBitChatMessage(
            sender=self.device_id,
            recipient=recipient,
            channel=channel,
            payload=payload,
            priority=priority,
        )

        # Apply compression if beneficial
        if compress:
            if message.compress_payload():
                self.enhanced_stats["messages_compressed"] += 1

        # Apply encryption if enabled and we have recipient's key
        if encrypt and self.enable_crypto and recipient:
            peer_keys = self.peer_keys.get(recipient)
            if peer_keys and "public_key" in peer_keys:
                if message.encrypt_payload(
                    peer_keys["public_key"], self.crypto_keys.private_key
                ):
                    self.enhanced_stats["messages_encrypted"] += 1

        # Sign message if crypto is enabled
        if self.enable_crypto and self.crypto_keys.signing_key:
            message.sign_message(self.crypto_keys.signing_key)

        # Handle fragmentation for large messages
        if len(message.payload) > 450:  # BLE packet size limit
            fragments = message.fragment_message()
            if fragments:
                return await self._send_fragments(fragments, recipient)

        # Send as single message
        return await self._transmit_enhanced_message(message)

    async def join_channel(self, channel_name: str) -> bool:
        """Join IRC-style channel"""
        if channel_name in self.joined_channels:
            return True

        # Create channel if it doesn't exist
        if channel_name not in self.channels:
            self.channels[channel_name] = BitChatChannel(channel_name, self.device_id)

        # Add self to channel
        channel = self.channels[channel_name]
        channel.add_member(self.device_id)
        self.joined_channels.add(channel_name)

        # Broadcast join message
        join_msg = EnhancedBitChatMessage(
            sender=self.device_id,
            channel=channel_name,
            message_type=BitChatMessageType.CHANNEL_JOIN,
            payload=f"/join {channel_name}".encode(),
        )

        await self._transmit_enhanced_message(join_msg)
        self.enhanced_stats["channels_joined"] += 1
        logger.info(f"Joined channel #{channel_name}")
        return True

    async def leave_channel(self, channel_name: str) -> bool:
        """Leave IRC-style channel"""
        if channel_name not in self.joined_channels:
            return True

        # Remove self from channel
        if channel_name in self.channels:
            self.channels[channel_name].remove_member(self.device_id)

        self.joined_channels.remove(channel_name)

        # Broadcast leave message
        leave_msg = EnhancedBitChatMessage(
            sender=self.device_id,
            channel=channel_name,
            message_type=BitChatMessageType.CHANNEL_LEAVE,
            payload=f"/leave {channel_name}".encode(),
        )

        await self._transmit_enhanced_message(leave_msg)
        logger.info(f"Left channel #{channel_name}")
        return True

    async def send_channel_message(self, channel_name: str, message: str) -> bool:
        """Send message to IRC-style channel"""
        if channel_name not in self.joined_channels:
            await self.join_channel(channel_name)

        channel_msg = EnhancedBitChatMessage(
            sender=self.device_id,
            channel=channel_name,
            message_type=BitChatMessageType.CHANNEL_MSG,
            payload=message.encode(),
        )

        return await self._transmit_enhanced_message(channel_msg)

    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced BitChat status"""
        base_status = self.get_status()

        base_status.update(
            {
                "crypto_enabled": self.enable_crypto,
                "peer_keys_exchanged": len(self.peer_keys),
                "channels_joined": list(self.joined_channels),
                "store_forward_cache_size": sum(
                    len(msgs) for msgs in self.store_forward_cache.values()
                ),
                "pending_fragments": len(self.pending_fragments),
                "enhanced_stats": self.enhanced_stats.copy(),
            }
        )

        return base_status

    async def _transmit_enhanced_message(self, message: EnhancedBitChatMessage) -> bool:
        """Transmit enhanced message"""
        try:
            # Convert to base BitChatMessage for transmission
            base_message = BitChatMessage(
                id=message.id,
                sender=message.sender,
                recipient=message.recipient,
                payload=message.payload,
                ttl=message.ttl,
                hop_count=message.hop_count,
                timestamp=message.timestamp,
                route_path=message.route_path.copy(),
                priority=message.priority,
            )

            # Use parent class transmission
            success = await super().send_message(
                message.recipient, message.payload, message.priority, message.ttl
            )

            if success:
                logger.debug(f"Transmitted enhanced message {message.id[:8]}")

            return success

        except Exception as e:
            logger.error(f"Enhanced message transmission failed: {e}")
            return False

    async def _send_fragments(
        self, fragments: List[BitChatFragment], recipient: str
    ) -> bool:
        """Send message fragments"""
        success_count = 0

        for fragment in fragments:
            fragment_msg = EnhancedBitChatMessage(
                sender=self.device_id,
                recipient=recipient,
                message_type=BitChatMessageType.FRAGMENT,
                payload=fragment.to_bytes(),
            )

            if await self._transmit_enhanced_message(fragment_msg):
                success_count += 1
                self.enhanced_stats["fragments_sent"] += 1

        logger.debug(f"Sent {success_count}/{len(fragments)} fragments")
        return success_count == len(fragments)

    def _register_enhanced_handlers(self):
        """Register enhanced message handlers"""
        self.enhanced_handlers[BitChatMessageType.DATA] = self._handle_enhanced_data
        self.enhanced_handlers[BitChatMessageType.FRAGMENT] = self._handle_fragment
        self.enhanced_handlers[BitChatMessageType.HELLO] = self._handle_crypto_hello
        self.enhanced_handlers[
            BitChatMessageType.CHANNEL_JOIN
        ] = self._handle_channel_join
        self.enhanced_handlers[
            BitChatMessageType.CHANNEL_LEAVE
        ] = self._handle_channel_leave
        self.enhanced_handlers[
            BitChatMessageType.CHANNEL_MSG
        ] = self._handle_channel_message
        self.enhanced_handlers[BitChatMessageType.DUMMY] = self._handle_dummy_traffic

    async def _handle_enhanced_data(self, message: EnhancedBitChatMessage):
        """Handle enhanced data message"""
        # Verify signature if signed
        if message.signed:
            peer_keys = self.peer_keys.get(message.sender)
            if peer_keys and "verify_key" in peer_keys:
                verify_key = nacl.signing.VerifyKey(peer_keys["verify_key"])
                if not message.verify_signature(verify_key):
                    logger.warning(f"Invalid signature from {message.sender}")
                    return

        # Decrypt if encrypted
        if message.encrypted:
            peer_keys = self.peer_keys.get(message.sender)
            if peer_keys and "public_key" in peer_keys:
                message.decrypt_payload(
                    peer_keys["public_key"], self.crypto_keys.private_key
                )

        # Decompress if compressed
        message.decompress_payload()

        logger.debug(f"Processed enhanced data message from {message.sender}")

    async def _handle_fragment(self, message: EnhancedBitChatMessage):
        """Handle message fragment"""
        try:
            fragment = BitChatFragment.from_bytes(message.payload)

            # Store fragment
            self.pending_fragments[fragment.fragment_id][
                fragment.fragment_index
            ] = fragment
            self.enhanced_stats["fragments_received"] += 1

            # Check if we have all fragments
            fragments_dict = self.pending_fragments[fragment.fragment_id]
            if len(fragments_dict) == fragment.total_fragments:
                # Attempt reassembly
                fragments_list = [
                    fragments_dict[i] for i in range(fragment.total_fragments)
                ]
                reassembled = EnhancedBitChatMessage.reassemble_from_fragments(
                    fragments_list
                )

                if reassembled:
                    # Handle reassembled message
                    await self._handle_enhanced_data(reassembled)

                # Clean up fragments
                del self.pending_fragments[fragment.fragment_id]

        except Exception as e:
            logger.error(f"Fragment handling failed: {e}")

    async def _handle_crypto_hello(self, message: EnhancedBitChatMessage):
        """Handle hello message with crypto key exchange"""
        try:
            peer_data = json.loads(message.payload.decode())
            peer_id = peer_data.get("device_id", message.sender)

            # Store peer's public keys
            if "public_key" in peer_data and "verify_key" in peer_data:
                self.peer_keys[peer_id] = {
                    "public_key": bytes.fromhex(peer_data["public_key"]),
                    "verify_key": bytes.fromhex(peer_data["verify_key"]),
                }
                logger.info(f"Exchanged crypto keys with {peer_id}")

            # Update peer info
            if peer_id not in self.discovered_peers:
                self.discovered_peers[peer_id] = BitChatPeer(
                    device_id=peer_id,
                    bluetooth_addr="unknown",
                    device_name=peer_data.get("device_name", peer_id),
                )

        except Exception as e:
            logger.warning(f"Invalid crypto hello from {message.sender}: {e}")

    async def _handle_channel_join(self, message: EnhancedBitChatMessage):
        """Handle channel join"""
        channel_name = message.channel
        if channel_name:
            if channel_name not in self.channels:
                self.channels[channel_name] = BitChatChannel(
                    channel_name, message.sender
                )

            self.channels[channel_name].add_member(message.sender)

    async def _handle_channel_leave(self, message: EnhancedBitChatMessage):
        """Handle channel leave"""
        channel_name = message.channel
        if channel_name in self.channels:
            self.channels[channel_name].remove_member(message.sender)

    async def _handle_channel_message(self, message: EnhancedBitChatMessage):
        """Handle channel message"""
        channel_name = message.channel
        if channel_name in self.channels:
            self.channels[channel_name].add_message(message)

    async def _handle_dummy_traffic(self, message: EnhancedBitChatMessage):
        """Handle dummy traffic (privacy padding)"""
        logger.debug(f"Received dummy traffic from {message.sender}")

    async def _send_crypto_hello(self):
        """Send hello message with crypto keys"""
        if not self.enable_crypto:
            return

        hello_payload = json.dumps(
            {
                "device_id": self.device_id,
                "device_name": f"BitChat_{self.device_id[:8]}",
                "public_key": self.crypto_keys.public_key.hex()
                if self.crypto_keys.public_key
                else "",
                "verify_key": bytes(self.crypto_keys.verify_key).hex()
                if self.crypto_keys.verify_key
                else "",
                "timestamp": time.time(),
                "capabilities": ["bitchat", "crypto", "channels", "fragmentation"],
                "channels": list(self.joined_channels),
            }
        ).encode()

        hello_msg = EnhancedBitChatMessage(
            sender=self.device_id,
            message_type=BitChatMessageType.HELLO,
            payload=hello_payload,
        )

        await self._transmit_enhanced_message(hello_msg)
        logger.debug("Sent crypto hello message")

    async def _dummy_traffic_loop(self):
        """Generate dummy traffic for privacy (30-120 second intervals)"""
        while self.is_running:
            try:
                # Random interval as per spec
                interval = random.uniform(30, 120)
                await asyncio.sleep(interval)

                if not self.is_running:
                    break

                # Create dummy message with random payload (50-500 bytes)
                dummy_size = random.randint(50, 500)
                dummy_payload = os.urandom(dummy_size)

                dummy_msg = EnhancedBitChatMessage(
                    sender=self.device_id,
                    message_type=BitChatMessageType.DUMMY,
                    payload=dummy_payload,
                )

                # Add random delay (50-500ms as per spec)
                delay = random.uniform(0.05, 0.5)
                await asyncio.sleep(delay)

                await self._transmit_enhanced_message(dummy_msg)
                self.enhanced_stats["dummy_traffic_sent"] += 1
                logger.debug("Sent dummy traffic for privacy")

            except Exception as e:
                logger.error(f"Dummy traffic generation error: {e}")
                await asyncio.sleep(60)

    async def _enhanced_maintenance_loop(self):
        """Enhanced maintenance tasks"""
        while self.is_running:
            try:
                # Clean expired store-and-forward messages (12-hour TTL)
                await self._cleanup_expired_store_forward()

                # Clean up stale fragments (5 minutes)
                await self._cleanup_stale_fragments()

                # Send periodic crypto hello
                await self._send_crypto_hello()

                # Maintenance every 5 minutes
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Enhanced maintenance error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_store_forward(self):
        """Clean up expired store-and-forward messages"""
        current_time = time.time()
        cleaned_count = 0

        for recipient, messages in list(self.store_forward_cache.items()):
            # Filter out messages older than 12 hours
            valid_messages = [
                msg
                for msg in messages
                if msg.expires_at and msg.expires_at > current_time
            ]

            cleaned = len(messages) - len(valid_messages)
            if cleaned > 0:
                cleaned_count += cleaned

                if valid_messages:
                    self.store_forward_cache[recipient] = valid_messages
                else:
                    del self.store_forward_cache[recipient]

        if cleaned_count > 0:
            logger.debug(f"Cleaned {cleaned_count} expired store-and-forward messages")

    async def _cleanup_stale_fragments(self):
        """Clean up incomplete fragments older than 5 minutes"""
        current_time = time.time()
        stale_fragments = []

        # Remove fragment sets that haven't been updated in 5 minutes
        for fragment_id in list(self.pending_fragments.keys()):
            # This is simplified - in practice we'd track fragment timestamps
            if len(list(self.pending_fragments.keys())) > 100:  # Prevent memory bloat
                stale_fragments.append(fragment_id)

        for fragment_id in stale_fragments[:10]:  # Clean up to 10 at a time
            del self.pending_fragments[fragment_id]

        if stale_fragments:
            logger.debug(f"Cleaned {len(stale_fragments)} stale fragment sets")
