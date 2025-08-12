"""Tensor Streaming for Efficient Model Weight Transfer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import io
import json
import logging
import os
import time
from typing import Any, Callable
import uuid
import zlib
import base64

# For compression (using existing compression pipeline)
import lz4.frame

# For tensor operations
import numpy as np

# We'll handle torch import gracefully in case it's not available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch may not be installed in all environments
    TORCH_AVAILABLE = False

# For key exchange
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.fernet import Fernet

from .p2p_node import MessageType, P2PMessage, P2PNode

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Tensor compression methods."""

    NONE = "none"
    LZ4 = "lz4"
    ZLIB = "zlib"
    QUANTIZED_8BIT = "quantized_8bit"
    QUANTIZED_4BIT = "quantized_4bit"
    DELTA_ENCODING = "delta_encoding"


class TensorFormat(Enum):
    """Tensor serialization formats."""

    NUMPY = "numpy"
    PICKLE = "pickle"
    JSON = "json"
    CUSTOM_BINARY = "custom_binary"


@dataclass
class StreamingConfig:
    """Configuration for tensor streaming."""

    chunk_size: int = 64 * 1024  # 64KB chunks
    compression: CompressionType = CompressionType.LZ4
    tensor_format: TensorFormat = TensorFormat.NUMPY
    max_retries: int = 3
    retry_delay: float = 1.0
    checksum_verification: bool = True
    bandwidth_limit_kbps: int | None = None  # Bandwidth throttling
    priority_queue: bool = True
    resume_capability: bool = True


@dataclass
class TensorMetadata:
    """Metadata for tensor transfer."""

    tensor_id: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    total_chunks: int
    compression: CompressionType
    format: TensorFormat
    checksum: str
    timestamp: float = field(default_factory=time.time)
    source_node: str = ""
    tags: dict[str, Any] = field(default_factory=dict)
    device: str | None = None
    is_torch: bool = False
    requires_grad: bool = False


@dataclass
class TensorChunk:
    """Individual chunk of tensor data."""

    tensor_id: str
    chunk_index: int
    total_chunks: int
    data: bytes
    checksum: str
    timestamp: float = field(default_factory=time.time)
    is_compressed: bool = False
    compression_type: CompressionType | None = None


@dataclass
class TransferProgress:
    """Progress tracking for tensor transfers."""

    tensor_id: str
    total_chunks: int
    received_chunks: int
    missing_chunks: list[int] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    bytes_transferred: int = 0
    estimated_total_bytes: int = 0
    transfer_rate_kbps: float = 0.0

    @property
    def progress_percent(self) -> float:
        """Calculate transfer progress percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.received_chunks / self.total_chunks) * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if transfer is complete."""
        return self.received_chunks >= self.total_chunks


class TensorStreaming:
    """High-performance tensor streaming for model weight distribution."""

    def __init__(
        self,
        node: P2PNode,
        config: StreamingConfig | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.node = node
        self.config = config or StreamingConfig()
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Transfer tracking
        self.active_transfers: dict[str, TransferProgress] = {}
        self.pending_chunks: dict[str, dict[int, TensorChunk]] = {}  # tensor_id -> chunk_index -> chunk
        self.tensor_metadata: dict[str, TensorMetadata] = {}

        # Bandwidth management is coordinated through a singleton controller
        self.bandwidth_controller = BandwidthController.get_instance(
            self.config.bandwidth_limit_kbps
        )

        # Priority queue for chunk requests
        self.priority_queue: list[tuple[float, str, int]] = []  # (priority, tensor_id, chunk_index)

        # Statistics
        self.stats = {
            "tensors_sent": 0,
            "tensors_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "chunks_sent": 0,
            "chunks_received": 0,
            "compression_ratio": 0.0,
            "avg_transfer_rate_kbps": 0.0,
            "failed_transfers": 0,
        }

        # Diffie-Hellman key exchange
        self._dh_private_key = x25519.X25519PrivateKey.generate()
        self._dh_public_key = self._dh_private_key.public_key()
        self._key_cache: dict[str, bytes] = {}
        self._fernet = Fernet(Fernet.generate_key())

        # Register message handlers
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register tensor streaming message handlers."""
        self.node.register_handler(MessageType.TENSOR_CHUNK, self._handle_tensor_chunk)
        if MessageType.DATA not in self.node.message_handlers:
            self.node.register_handler(MessageType.DATA, self._handle_tensor_chunk)

    async def _initiate_key_exchange(self, peer_id: str) -> None:
        """Start Diffie-Hellman key exchange with a peer."""
        payload = {
            "action": "dh_key",
            "key": self._dh_public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            ).hex(),
        }
        await self.node.send_message(peer_id, MessageType.DATA, payload)

    def _derive_shared_key(self, peer_public_bytes: bytes) -> bytes:
        """Derive shared encryption key from peer's public key."""
        peer_public = x25519.X25519PublicKey.from_public_bytes(peer_public_bytes)
        shared = self._dh_private_key.exchange(peer_public)
        derived = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"tensor_streaming",
        ).derive(shared)
        return base64.urlsafe_b64encode(derived)

    async def _ensure_key(self, peer_id: str) -> None:
        """Ensure a shared key exists for a peer."""
        if peer_id in self._key_cache:
            return
        await self._initiate_key_exchange(peer_id)
        for _ in range(self.config.max_retries * 10):
            if peer_id in self._key_cache:
                return
            await asyncio.sleep(self.config.retry_delay / 10)
        raise RuntimeError(f"Key exchange failed with {peer_id}")

    async def send_tensor(
        self,
        tensor_data: np.ndarray | Any,
        tensor_name: str,
        destination: str,
        metadata_tags: dict[str, Any] | None = None,
        priority: float = 1.0,
    ) -> str:
        """Send a tensor to a peer node."""
        tensor_id = str(uuid.uuid4())
        metadata_tags = metadata_tags or {}

        logger.info(f"Starting tensor transfer: {tensor_name} -> {destination}")

        try:
            await self._ensure_key(destination)
            # Serialize tensor
            serialized_data, tensor_metadata = await self._serialize_tensor(
                tensor_data, tensor_id, tensor_name, metadata_tags
            )

            # Compress if configured
            if self.config.compression != CompressionType.NONE:
                compressed_data = await self._compress_tensor(serialized_data, self.config.compression)
                compression_ratio = len(serialized_data) / len(compressed_data)
                logger.info(f"Compression ratio: {compression_ratio:.2f}x")
            else:
                compressed_data = serialized_data
                compression_ratio = 1.0

            # Update stats
            self.stats["compression_ratio"] = (
                self.stats["compression_ratio"] * self.stats["tensors_sent"] + compression_ratio
            ) / (self.stats["tensors_sent"] + 1)

            # Split into chunks
            chunks = self._split_into_chunks(compressed_data, tensor_id)
            tensor_metadata.total_chunks = len(chunks)

            # Send metadata first
            await self._send_tensor_metadata(destination, tensor_metadata)

            # Send chunks with bandwidth throttling
            start_time = time.time()
            bytes_sent = 0

            for chunk in chunks:
                # Apply bandwidth throttling. The controller will no-op if
                # no limit is configured.
                await self.bandwidth_controller.throttle(len(chunk.data))

                # Send chunk
                success = await self._send_tensor_chunk(destination, chunk)

                if success:
                    bytes_sent += len(chunk.data)
                    self.stats["chunks_sent"] += 1
                else:
                    # Retry failed chunks
                    for retry in range(self.config.max_retries):
                        await asyncio.sleep(self.config.retry_delay * (retry + 1))
                        if await self._send_tensor_chunk(destination, chunk):
                            bytes_sent += len(chunk.data)
                            self.stats["chunks_sent"] += 1
                            break
                    else:
                        logger.error(f"Failed to send chunk {chunk.chunk_index} after retries")
                        self.stats["failed_transfers"] += 1
                        return tensor_id

            # Calculate transfer rate
            transfer_time = time.time() - start_time
            transfer_rate = (bytes_sent / 1024) / transfer_time if transfer_time > 0 else 0

            self.stats["tensors_sent"] += 1
            self.stats["bytes_sent"] += bytes_sent
            self.stats["avg_transfer_rate_kbps"] = (
                self.stats["avg_transfer_rate_kbps"] * (self.stats["tensors_sent"] - 1) + transfer_rate
            ) / self.stats["tensors_sent"]

            logger.info(f"Tensor {tensor_name} sent successfully in {transfer_time:.2f}s at {transfer_rate:.2f} KB/s")
            return tensor_id

        except Exception as e:
            logger.exception(f"Failed to send tensor {tensor_name}: {e}")
            self.stats["failed_transfers"] += 1
            raise

    async def receive_tensor(
        self,
        tensor_id: str,
        timeout: float = 300.0,
        progress_callback: Callable | None = None,
    ) -> tuple[Any, TensorMetadata] | None:
        """Receive a tensor from a peer node."""
        if tensor_id not in self.tensor_metadata:
            logger.error(f"No metadata found for tensor {tensor_id}")
            return None

        metadata = self.tensor_metadata[tensor_id]

        # Initialize transfer tracking
        if tensor_id not in self.active_transfers:
            self.active_transfers[tensor_id] = TransferProgress(
                tensor_id=tensor_id,
                total_chunks=metadata.total_chunks,
                received_chunks=0,
                estimated_total_bytes=metadata.size_bytes,
            )

        progress = self.active_transfers[tensor_id]
        start_time = time.time()

        logger.info(f"Receiving tensor {metadata.name} ({metadata.total_chunks} chunks)")

        try:
            # Wait for all chunks with timeout
            while not progress.is_complete and (time.time() - start_time) < timeout:
                if progress_callback:
                    progress_callback(progress)

                await asyncio.sleep(0.1)  # Check every 100ms

            if not progress.is_complete:
                logger.error(f"Tensor {tensor_id} transfer timed out")
                return None

            # Reconstruct tensor from chunks
            tensor_data = await self._reconstruct_tensor(tensor_id)

            if tensor_data is not None:
                self.stats["tensors_received"] += 1
                self.stats["bytes_received"] += metadata.size_bytes

                # Clean up active transfer
                self.active_transfers.pop(tensor_id, None)

                logger.info(f"Successfully received tensor {metadata.name}")
                return tensor_data, metadata
            logger.error(f"Failed to reconstruct tensor {tensor_id}")
            return None

        except Exception as e:
            logger.exception(f"Error receiving tensor {tensor_id}: {e}")
            return None

    async def request_missing_chunks(
        self,
        tensor_id: str,
        source_peer: str,
    ) -> bool:
        """Request missing chunks for an incomplete transfer."""
        if tensor_id not in self.active_transfers:
            return False

        self.active_transfers[tensor_id]
        missing_chunks = self._find_missing_chunks(tensor_id)

        if not missing_chunks:
            return True  # No missing chunks

        logger.info(f"Requesting {len(missing_chunks)} missing chunks for {tensor_id}")

        # Send chunk requests
        request_payload = {
            "action": "request_chunks",
            "tensor_id": tensor_id,
            "chunk_indices": missing_chunks,
        }

        return await self.node.send_message(source_peer, MessageType.DATA, request_payload)

    def get_transfer_progress(self, tensor_id: str) -> TransferProgress | None:
        """Get progress information for an active transfer."""
        return self.active_transfers.get(tensor_id)

    def list_active_transfers(self) -> list[TransferProgress]:
        """List all active transfers."""
        return list(self.active_transfers.values())

    def get_streaming_stats(self) -> dict[str, Any]:
        """Get comprehensive streaming statistics."""
        return {
            **self.stats,
            "active_transfers": len(self.active_transfers),
            "cached_chunks": sum(len(chunks) for chunks in self.pending_chunks.values()),
            "bandwidth_limit_kbps": self.config.bandwidth_limit_kbps,
            "compression_type": self.config.compression.value,
        }

    async def _serialize_tensor(
        self,
        tensor_data: Any,
        tensor_id: str,
        tensor_name: str,
        tags: dict[str, Any],
    ) -> tuple[bytes, TensorMetadata]:
        """Serialize tensor to bytes."""
        if self.config.tensor_format == TensorFormat.NUMPY:
            if TORCH_AVAILABLE and isinstance(tensor_data, torch.Tensor):
                np_array = tensor_data.detach().cpu().numpy()
                buffer = io.BytesIO()
                np.save(buffer, np_array)
                serialized = buffer.getvalue()

                metadata = TensorMetadata(
                    tensor_id=tensor_id,
                    name=tensor_name,
                    shape=np_array.shape,
                    dtype=str(np_array.dtype),
                    size_bytes=len(serialized),
                    total_chunks=0,  # Will be set later
                    compression=self.config.compression,
                    format=self.config.tensor_format,
                    checksum=hashlib.sha256(serialized).hexdigest(),
                    source_node=self.node.node_id,
                    tags=tags,
                    device=str(tensor_data.device),
                    is_torch=True,
                    requires_grad=tensor_data.requires_grad,
                )

                return serialized, metadata
            if isinstance(tensor_data, np.ndarray):
                buffer = io.BytesIO()
                np.save(buffer, tensor_data)
                serialized = buffer.getvalue()

                metadata = TensorMetadata(
                    tensor_id=tensor_id,
                    name=tensor_name,
                    shape=tensor_data.shape,
                    dtype=str(tensor_data.dtype),
                    size_bytes=len(serialized),
                    total_chunks=0,  # Will be set later
                    compression=self.config.compression,
                    format=self.config.tensor_format,
                    checksum=hashlib.sha256(serialized).hexdigest(),
                    source_node=self.node.node_id,
                    tags=tags,
                )

                return serialized, metadata
            # Convert to numpy if possible
            try:
                array = np.array(tensor_data)
                return await self._serialize_tensor(array, tensor_id, tensor_name, tags)
            except Exception as e:
                logger.warning(f"Failed to convert to numpy array: {e}")

        # Fallback: attempt JSON serialization
        try:
            serialized = json.dumps(tensor_data).encode("utf-8")
            format_type = TensorFormat.JSON
        except TypeError:
            logger.exception("Tensor data is not serializable without pickle")
            raise

        metadata = TensorMetadata(
            tensor_id=tensor_id,
            name=tensor_name,
            shape=(),
            dtype="json",
            size_bytes=len(serialized),
            total_chunks=0,
            compression=self.config.compression,
            format=format_type,
            checksum=hashlib.sha256(serialized).hexdigest(),
            source_node=self.node.node_id,
            tags=tags,
        )

        return serialized, metadata

    async def _compress_tensor(
        self,
        data: bytes,
        compression_type: CompressionType,
    ) -> bytes:
        """Compress tensor data."""
        if compression_type == CompressionType.LZ4:
            return lz4.frame.compress(data)
        if compression_type == CompressionType.ZLIB:
            return zlib.compress(data, level=6)
        if compression_type == CompressionType.QUANTIZED_8BIT:
            # Simplified quantization - in practice would use proper quantization
            return data  # Placeholder
        return data

    async def _decompress_tensor(
        self,
        data: bytes,
        compression_type: CompressionType,
    ) -> bytes:
        """Decompress tensor data."""
        if compression_type == CompressionType.LZ4:
            return lz4.frame.decompress(data)
        if compression_type == CompressionType.ZLIB:
            return zlib.decompress(data)
        return data

    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data for disk caching."""
        return self._fernet.encrypt(data)

    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt cached data."""
        return self._fernet.decrypt(data)

    def _split_into_chunks(self, data: bytes, tensor_id: str) -> list[TensorChunk]:
        """Split tensor data into chunks."""
        chunks = []
        total_chunks = (len(data) + self.config.chunk_size - 1) // self.config.chunk_size

        for i in range(total_chunks):
            start_idx = i * self.config.chunk_size
            end_idx = min(start_idx + self.config.chunk_size, len(data))
            chunk_data = data[start_idx:end_idx]

            chunk = TensorChunk(
                tensor_id=tensor_id,
                chunk_index=i,
                total_chunks=total_chunks,
                data=chunk_data,
                checksum=hashlib.md5(chunk_data).hexdigest(),
                is_compressed=(self.config.compression != CompressionType.NONE),
                compression_type=self.config.compression,
            )

            chunks.append(chunk)

        return chunks

    async def _send_tensor_metadata(
        self,
        destination: str,
        metadata: TensorMetadata,
    ) -> bool:
        """Send tensor metadata to destination."""
        payload = {
            "action": "tensor_metadata",
            "metadata": {
                "tensor_id": metadata.tensor_id,
                "name": metadata.name,
                "shape": metadata.shape,
                "dtype": metadata.dtype,
                "size_bytes": metadata.size_bytes,
                "total_chunks": metadata.total_chunks,
                "compression": metadata.compression.value,
                "format": metadata.format.value,
                "checksum": metadata.checksum,
                "timestamp": metadata.timestamp,
                "source_node": metadata.source_node,
                "tags": metadata.tags,
                "device": metadata.device,
                "is_torch": metadata.is_torch,
                "requires_grad": metadata.requires_grad,
            },
        }

        return await self.node.send_message(destination, MessageType.DATA, payload)

    async def _send_tensor_chunk(
        self,
        destination: str,
        chunk: TensorChunk,
    ) -> bool:
        """Send a tensor chunk to destination."""
        # Create chunk message
        chunk_message = P2PMessage(
            message_type=MessageType.TENSOR_CHUNK,
            sender_id=self.node.node_id,
            receiver_id=destination,
            payload={
                "tensor_id": chunk.tensor_id,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "data": chunk.data.hex(),  # Hex encode for JSON serialization
                "checksum": chunk.checksum,
                "timestamp": chunk.timestamp,
                "is_compressed": chunk.is_compressed,
                "compression_type": (chunk.compression_type.value if chunk.compression_type else None),
            },
        )

        # Send through P2P node
        return await self.node.send_message(destination, MessageType.TENSOR_CHUNK, chunk_message.payload)

    async def _handle_tensor_chunk(
        self,
        message: P2PMessage,
        writer: asyncio.StreamWriter | None = None,
    ) -> None:
        """Handle incoming tensor chunk."""
        payload = message.payload

        if payload.get("action") == "dh_key":
            peer_bytes = bytes.fromhex(payload["key"])
            self._key_cache[message.sender_id] = self._derive_shared_key(peer_bytes)
            response = {
                "action": "dh_key_response",
                "key": self._dh_public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                ).hex(),
            }
            await self.node.send_message(message.sender_id, MessageType.DATA, response)
            return
        if payload.get("action") == "dh_key_response":
            peer_bytes = bytes.fromhex(payload["key"])
            self._key_cache[message.sender_id] = self._derive_shared_key(peer_bytes)
            return
        if payload.get("action") == "tensor_metadata":
            await self._handle_tensor_metadata(payload["metadata"])
        elif payload.get("action") == "request_chunks":
            await self._handle_chunk_request(message.sender_id, payload)
        else:
            # Regular chunk data
            await self._handle_chunk_data(payload)

    async def _handle_tensor_metadata(self, metadata_dict: dict[str, Any]) -> None:
        """Handle tensor metadata message."""
        tensor_id = metadata_dict["tensor_id"]

        metadata = TensorMetadata(
            tensor_id=tensor_id,
            name=metadata_dict["name"],
            shape=tuple(metadata_dict["shape"]),
            dtype=metadata_dict["dtype"],
            size_bytes=metadata_dict["size_bytes"],
            total_chunks=metadata_dict["total_chunks"],
            compression=CompressionType(metadata_dict["compression"]),
            format=TensorFormat(metadata_dict["format"]),
            checksum=metadata_dict["checksum"],
            timestamp=metadata_dict["timestamp"],
            source_node=metadata_dict["source_node"],
            tags=metadata_dict["tags"],
            device=metadata_dict.get("device"),
            is_torch=metadata_dict.get("is_torch", False),
            requires_grad=metadata_dict.get("requires_grad", False),
        )

        self.tensor_metadata[tensor_id] = metadata
        self.pending_chunks[tensor_id] = {}

        logger.info(f"Received metadata for tensor {metadata.name} ({metadata.total_chunks} chunks)")

    async def _handle_chunk_data(self, payload: dict[str, Any]) -> None:
        """Handle incoming chunk data."""
        tensor_id = payload["tensor_id"]
        chunk_index = payload["chunk_index"]

        # Decode hex data
        chunk_data = bytes.fromhex(payload["data"])

        # Verify checksum
        if self.config.checksum_verification:
            expected_checksum = payload["checksum"]
            actual_checksum = hashlib.md5(chunk_data).hexdigest()

            if expected_checksum != actual_checksum:
                logger.error(f"Checksum mismatch for chunk {chunk_index} of tensor {tensor_id}")
                return

        # Create chunk object
        chunk = TensorChunk(
            tensor_id=tensor_id,
            chunk_index=chunk_index,
            total_chunks=payload["total_chunks"],
            data=chunk_data,
            checksum=payload["checksum"],
            timestamp=payload["timestamp"],
            is_compressed=payload["is_compressed"],
            compression_type=(CompressionType(payload["compression_type"]) if payload["compression_type"] else None),
        )

        # Store chunk (encrypted on disk if cache_dir is set)
        if tensor_id not in self.pending_chunks:
            self.pending_chunks[tensor_id] = {}

        if self.cache_dir:
            file_path = os.path.join(self.cache_dir, f"{tensor_id}_{chunk_index}.chk")
            with open(file_path, "wb") as f:
                f.write(self._encrypt_data(chunk_data))
            chunk.data = b""
            chunk.file_path = file_path

        self.pending_chunks[tensor_id][chunk_index] = chunk

        # Update progress
        if tensor_id in self.active_transfers:
            progress = self.active_transfers[tensor_id]
            progress.received_chunks = len(self.pending_chunks[tensor_id])
            progress.last_update = time.time()
            progress.bytes_transferred += len(chunk_data)

            # Calculate transfer rate
            elapsed = progress.last_update - progress.start_time
            if elapsed > 0:
                progress.transfer_rate_kbps = (progress.bytes_transferred / 1024) / elapsed

        self.stats["chunks_received"] += 1

        logger.debug(f"Received chunk {chunk_index}/{chunk.total_chunks} for tensor {tensor_id}")

    async def _handle_chunk_request(
        self,
        requester: str,
        payload: dict[str, Any],
    ) -> None:
        """Handle request for specific chunks."""
        tensor_id = payload["tensor_id"]
        chunk_indices = payload["chunk_indices"]

        logger.info(f"Received request for {len(chunk_indices)} chunks of tensor {tensor_id}")

        # This would require caching sent tensors to respond to requests
        # For now, just log the request
        logger.warning("Chunk re-request functionality not yet implemented")

    async def _reconstruct_tensor(self, tensor_id: str, cleanup: bool = True) -> Any | None:
        """Reconstruct tensor from received chunks."""
        if tensor_id not in self.pending_chunks or tensor_id not in self.tensor_metadata:
            return None

        chunks = self.pending_chunks[tensor_id]
        metadata = self.tensor_metadata[tensor_id]

        # Check if all chunks are received
        if len(chunks) != metadata.total_chunks:
            logger.error(f"Missing chunks for tensor {tensor_id}: {len(chunks)}/{metadata.total_chunks}")
            return None

        # Sort chunks by index
        sorted_chunks = [chunks[i] for i in range(metadata.total_chunks)]

        # Concatenate chunk data
        parts = []
        for chunk in sorted_chunks:
            if self.cache_dir and getattr(chunk, "file_path", None):
                with open(chunk.file_path, "rb") as f:
                    enc = f.read()
                chunk_data = self._decrypt_data(enc)
            else:
                chunk_data = chunk.data
            parts.append(chunk_data)
        combined_data = b"".join(parts)

        # Decompress if needed
        if metadata.compression != CompressionType.NONE:
            combined_data = await self._decompress_tensor(combined_data, metadata.compression)

        # Verify overall checksum
        if self.config.checksum_verification:
            actual_checksum = hashlib.sha256(combined_data).hexdigest()
            if actual_checksum != metadata.checksum:
                logger.error(f"Overall checksum mismatch for tensor {tensor_id}")
                return None

        # Deserialize tensor
        if metadata.format == TensorFormat.NUMPY:
            buffer = io.BytesIO(combined_data)
            np_array = np.load(buffer)
            if metadata.is_torch:
                if not TORCH_AVAILABLE:
                    logger.error("Torch not available for tensor reconstruction")
                    return None
                tensor = torch.from_numpy(np_array)
                tensor = tensor.to(metadata.device or "cpu")
                if metadata.requires_grad:
                    tensor.requires_grad_(True)
                result: Any = tensor
            else:
                result = np_array
        elif metadata.format == TensorFormat.JSON:
            result = json.loads(combined_data.decode("utf-8"))
        else:
            logger.error(f"Unsupported tensor format: {metadata.format}")
            return None

        if cleanup:
            if self.cache_dir:
                for chunk in sorted_chunks:
                    if getattr(chunk, "file_path", None):
                        try:
                            os.remove(chunk.file_path)
                        except OSError:
                            pass
            self.pending_chunks.pop(tensor_id, None)
            self.tensor_metadata.pop(tensor_id, None)

        return result

    def _find_missing_chunks(self, tensor_id: str) -> list[int]:
        """Find missing chunk indices for a tensor."""
        if tensor_id not in self.pending_chunks or tensor_id not in self.tensor_metadata:
            return []

        chunks = self.pending_chunks[tensor_id]
        metadata = self.tensor_metadata[tensor_id]

        received_indices = set(chunks.keys())
        all_indices = set(range(metadata.total_chunks))

        return list(all_indices - received_indices)


class BandwidthController:
    """Singleton controller enforcing bandwidth limits across instances."""

    _instance: BandwidthController | None = None

    def __init__(self, limit_kbps: int | None = None) -> None:
        self.limit_kbps = limit_kbps
        self.bytes_sent = 0
        self.last_reset = time.time()
        self.reset_interval = 1.0  # seconds
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls, limit_kbps: int | None = None) -> BandwidthController:
        if cls._instance is None:
            cls._instance = cls(limit_kbps)
        else:
            cls._instance.limit_kbps = limit_kbps
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for tests)."""
        cls._instance = None

    async def throttle(self, bytes_to_send: int) -> None:
        """Throttle bandwidth if a limit is configured."""
        if not self.limit_kbps:
            return

        while True:
            async with self._lock:
                current_time = time.time()

                if current_time - self.last_reset >= self.reset_interval:
                    self.bytes_sent = 0
                    self.last_reset = current_time

                limit_bytes = self.limit_kbps * 1024
                if self.bytes_sent + bytes_to_send <= limit_bytes:
                    self.bytes_sent += bytes_to_send
                    return

                delay = self.reset_interval - (current_time - self.last_reset)

            if delay > 0:
                await asyncio.sleep(delay)
