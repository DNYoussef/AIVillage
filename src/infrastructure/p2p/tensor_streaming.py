"""Efficient tensor streaming for model distribution."""

import asyncio
import base64
from collections.abc import Iterator
import hashlib
import io
import json
import logging
import time
from typing import Any

# Import compression
import lz4.frame

# We'll handle torch import gracefully in case it's not available
try:
    import numpy as np
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Create dummy classes for type hints
    class torch:
        class Tensor:
            pass

    class np:
        class ndarray:
            pass


logger = logging.getLogger(__name__)


class TensorChunk:
    """Represents a chunk of tensor data."""

    def __init__(
        self,
        chunk_id: int,
        total_chunks: int,
        data: bytes,
        checksum: str,
        metadata: dict | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.total_chunks = total_chunks
        self.data = data
        self.checksum = checksum
        self.metadata = metadata or {}
        self.timestamp = time.time()


class TensorStreamer:
    """Stream large tensors across P2P network with compression and resume."""

    def __init__(self, chunk_size: int = 1024 * 1024) -> None:  # 1MB chunks
        self.chunk_size = chunk_size
        self.compression_level = 9  # Max compression for bandwidth
        self.active_transfers: dict[str, dict] = {}  # transfer_id -> state
        self.bandwidth_limit_kbps: int | None = None  # No limit by default

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, tensor streaming will be limited")

    def set_bandwidth_limit(self, kbps: int) -> None:
        """Set bandwidth limit in kilobits per second."""
        self.bandwidth_limit_kbps = kbps

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()

    def _serialize_tensor(self, tensor: Any) -> bytes:
        """Serialize tensor with metadata."""
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            np_array = tensor.cpu().numpy()
            metadata = {
                "shape": list(np_array.shape),
                "dtype": str(np_array.dtype),
                "device": str(tensor.device),
                "requires_grad": tensor.requires_grad,
                "is_torch_tensor": True,
            }
        elif isinstance(tensor, np.ndarray):
            np_array = tensor
            metadata = {
                "shape": list(np_array.shape),
                "dtype": str(np_array.dtype),
                "is_torch_tensor": False,
            }
        else:
            # Attempt conversion to numpy
            np_array = np.array(tensor)
            metadata = {
                "shape": list(np_array.shape),
                "dtype": str(np_array.dtype),
                "is_torch_tensor": False,
            }

        buffer = io.BytesIO()
        np.save(buffer, np_array)
        payload = {
            "metadata": metadata,
            "data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        }
        return json.dumps(payload).encode("utf-8")

    def _deserialize_tensor(self, data: bytes) -> Any:
        """Deserialize tensor from bytes."""
        obj = json.loads(data.decode("utf-8"))
        metadata = obj["metadata"]
        np_bytes = base64.b64decode(obj["data"])
        buffer = io.BytesIO(np_bytes)
        np_array = np.load(buffer)

        if TORCH_AVAILABLE and metadata.get("is_torch_tensor", False):
            tensor = torch.from_numpy(np_array)
            if metadata.get("requires_grad", False):
                tensor.requires_grad_(True)
            return tensor
        return np_array

    def compress_and_chunk_tensor(
        self, tensor: Any, transfer_id: str | None = None
    ) -> Iterator[TensorChunk]:
        """Compress tensor and yield chunks."""
        if not transfer_id:
            transfer_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

        try:
            # Serialize tensor
            serialized = self._serialize_tensor(tensor)

            # Compress
            compressed = lz4.frame.compress(
                serialized, compression_level=self.compression_level
            )

            logger.info(
                f"Compressed tensor: {len(serialized)} -> {len(compressed)} bytes "
                f"({len(compressed) / len(serialized):.2%} ratio)"
            )

            # Calculate total chunks
            total_chunks = (len(compressed) + self.chunk_size - 1) // self.chunk_size

            # Track transfer
            self.active_transfers[transfer_id] = {
                "total_size": len(compressed),
                "total_chunks": total_chunks,
                "chunks_sent": 0,
                "start_time": time.time(),
                "completed": False,
            }

            # Yield chunks
            for i in range(0, len(compressed), self.chunk_size):
                chunk_data = compressed[i : i + self.chunk_size]
                chunk_id = i // self.chunk_size
                checksum = self._calculate_checksum(chunk_data)

                chunk = TensorChunk(
                    chunk_id=chunk_id,
                    total_chunks=total_chunks,
                    data=chunk_data,
                    checksum=checksum,
                    metadata={
                        "transfer_id": transfer_id,
                        "original_size": len(serialized),
                        "compressed_size": len(compressed),
                    },
                )

                # Update transfer state
                self.active_transfers[transfer_id]["chunks_sent"] += 1

                yield chunk

            # Mark transfer as completed
            self.active_transfers[transfer_id]["completed"] = True

        except Exception as e:
            logger.exception(f"Failed to compress and chunk tensor: {e}")
            if transfer_id in self.active_transfers:
                del self.active_transfers[transfer_id]
            raise

    async def stream_tensor_to_peer(
        self, p2p_node, peer_id: str, tensor: Any, transfer_id: str | None = None
    ) -> bool:
        """Stream tensor to specific peer with bandwidth limiting."""
        try:
            total_bytes = 0
            start_time = time.time()

            for chunk in self.compress_and_chunk_tensor(tensor, transfer_id):
                # Prepare message
                message = {
                    "type": "TENSOR_CHUNK",
                    "transfer_id": chunk.metadata["transfer_id"],
                    "chunk_id": chunk.chunk_id,
                    "total_chunks": chunk.total_chunks,
                    "data": chunk.data.hex(),  # Convert to hex for JSON safety
                    "checksum": chunk.checksum,
                    "metadata": chunk.metadata,
                }

                # Send chunk
                success = await p2p_node.send_to_peer(peer_id, message)
                if not success:
                    logger.error(f"Failed to send chunk {chunk.chunk_id} to {peer_id}")
                    return False

                total_bytes += len(chunk.data)

                # Bandwidth throttling
                if self.bandwidth_limit_kbps:
                    await self._throttle_bandwidth(
                        len(chunk.data), start_time, total_bytes
                    )

                logger.debug(
                    f"Sent chunk {chunk.chunk_id}/{chunk.total_chunks} to {peer_id}"
                )

            logger.info(
                f"Successfully streamed tensor to {peer_id} "
                f"({total_bytes} bytes in {time.time() - start_time:.2f}s)"
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to stream tensor to {peer_id}: {e}")
            return False

    async def _throttle_bandwidth(
        self, bytes_sent: int, start_time: float, total_bytes: int
    ) -> None:
        """Apply bandwidth throttling."""
        if not self.bandwidth_limit_kbps:
            return

        elapsed = time.time() - start_time
        target_bytes_per_second = (
            self.bandwidth_limit_kbps * 1024
        ) / 8  # Convert to bytes/sec
        target_total_bytes = elapsed * target_bytes_per_second

        if total_bytes > target_total_bytes:
            # We're sending too fast, sleep
            delay = (total_bytes - target_total_bytes) / target_bytes_per_second
            await asyncio.sleep(delay)

    async def stream_model_to_peers(
        self,
        p2p_node,
        model_state_dict: dict[str, Any],
        peer_ids: list[str],
        parallel: bool = True,
    ) -> dict[str, bool]:
        """Stream entire model to multiple peers."""
        results = {}

        if parallel:
            # Stream to all peers in parallel
            tasks = []
            for peer_id in peer_ids:
                task = self._stream_model_to_single_peer(
                    p2p_node, model_state_dict, peer_id
                )
                tasks.append(task)

            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            for peer_id, result in zip(peer_ids, completed_results, strict=False):
                if isinstance(result, Exception):
                    logger.error(f"Failed to stream to {peer_id}: {result}")
                    results[peer_id] = False
                else:
                    results[peer_id] = result
        else:
            # Stream sequentially
            for peer_id in peer_ids:
                results[peer_id] = await self._stream_model_to_single_peer(
                    p2p_node, model_state_dict, peer_id
                )

        return results

    async def _stream_model_to_single_peer(
        self, p2p_node, model_state_dict: dict[str, Any], peer_id: str
    ) -> bool:
        """Stream model to single peer."""
        try:
            total_params = 0
            successful_layers = 0

            for layer_name, tensor in model_state_dict.items():
                if TORCH_AVAILABLE and hasattr(tensor, "numel"):
                    param_count = tensor.numel()
                else:
                    param_count = getattr(tensor, "size", 0)

                total_params += param_count

                logger.info(
                    f"Streaming layer {layer_name} to {peer_id} ({param_count} parameters)"
                )

                success = await self.stream_tensor_to_peer(
                    p2p_node,
                    peer_id,
                    tensor,
                    transfer_id=f"{layer_name}_{peer_id}_{int(time.time())}",
                )

                if success:
                    successful_layers += 1
                else:
                    logger.error(f"Failed to stream layer {layer_name} to {peer_id}")
                    return False

            logger.info(
                f"Successfully streamed {successful_layers} layers "
                f"({total_params} total parameters) to {peer_id}"
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to stream model to {peer_id}: {e}")
            return False

    def create_tensor_receiver(self, p2p_node) -> "TensorReceiver":
        """Create a tensor receiver for this streamer."""
        return TensorReceiver(p2p_node, self)

    def get_transfer_status(self, transfer_id: str) -> dict | None:
        """Get status of active transfer."""
        return self.active_transfers.get(transfer_id)

    def get_active_transfers(self) -> dict[str, dict]:
        """Get all active transfers."""
        return self.active_transfers.copy()


class TensorReceiver:
    """Receives and reconstructs streamed tensors."""

    def __init__(self, p2p_node, streamer: TensorStreamer) -> None:
        self.p2p_node = p2p_node
        self.streamer = streamer
        self.pending_transfers: dict[str, dict] = {}  # transfer_id -> chunks

        # Register handler for tensor chunks
        self.p2p_node.register_handler("TENSOR_CHUNK", self._handle_tensor_chunk)

    async def _handle_tensor_chunk(self, message: dict, writer) -> None:
        """Handle incoming tensor chunk."""
        try:
            transfer_id = message.get("transfer_id")
            chunk_id = message.get("chunk_id")
            total_chunks = message.get("total_chunks")
            data_hex = message.get("data")
            checksum = message.get("checksum")
            metadata = message.get("metadata", {})

            if not all([transfer_id, data_hex, checksum]):
                logger.error("Invalid tensor chunk message")
                return

            # Convert hex back to bytes
            chunk_data = bytes.fromhex(data_hex)

            # Verify checksum
            if self.streamer._calculate_checksum(chunk_data) != checksum:
                logger.error(f"Checksum mismatch for chunk {chunk_id}")
                return

            # Initialize transfer if needed
            if transfer_id not in self.pending_transfers:
                self.pending_transfers[transfer_id] = {
                    "chunks": {},
                    "total_chunks": total_chunks,
                    "metadata": metadata,
                    "start_time": time.time(),
                }

            # Store chunk
            self.pending_transfers[transfer_id]["chunks"][chunk_id] = chunk_data

            logger.debug(
                f"Received chunk {chunk_id}/{total_chunks} for transfer {transfer_id}"
            )

            # Check if transfer is complete
            chunks = self.pending_transfers[transfer_id]["chunks"]
            if len(chunks) == total_chunks:
                await self._complete_transfer(transfer_id)

            # Send acknowledgment
            ack_message = {
                "type": "CHUNK_ACK",
                "transfer_id": transfer_id,
                "chunk_id": chunk_id,
                "status": "received",
            }

            if writer:
                await self.p2p_node._send_message(ack_message, writer)

        except Exception as e:
            logger.exception(f"Error handling tensor chunk: {e}")

    async def _complete_transfer(self, transfer_id: str) -> None:
        """Complete a tensor transfer by reconstructing the tensor."""
        try:
            transfer_info = self.pending_transfers[transfer_id]
            chunks = transfer_info["chunks"]
            total_chunks = transfer_info["total_chunks"]

            # Reconstruct compressed data
            compressed_data = b""
            for chunk_id in range(total_chunks):
                if chunk_id not in chunks:
                    logger.error(f"Missing chunk {chunk_id} for transfer {transfer_id}")
                    return
                compressed_data += chunks[chunk_id]

            # Decompress
            try:
                serialized_data = lz4.frame.decompress(compressed_data)
            except Exception as e:
                logger.exception(f"Failed to decompress transfer {transfer_id}: {e}")
                return

            # Deserialize tensor
            tensor = self.streamer._deserialize_tensor(serialized_data)

            # Store reconstructed tensor
            transfer_info["tensor"] = tensor
            transfer_info["completed"] = True
            transfer_info["completion_time"] = time.time()

            duration = transfer_info["completion_time"] - transfer_info["start_time"]
            logger.info(
                f"Successfully reconstructed tensor for transfer {transfer_id} "
                f"in {duration:.2f}s"
            )

        except Exception as e:
            logger.exception(f"Failed to complete transfer {transfer_id}: {e}")

    def get_completed_tensor(self, transfer_id: str) -> Any | None:
        """Get completed tensor by transfer ID."""
        transfer_info = self.pending_transfers.get(transfer_id)
        if transfer_info and transfer_info.get("completed"):
            return transfer_info.get("tensor")
        return None

    def get_pending_transfers(self) -> list[str]:
        """Get list of pending transfer IDs."""
        return [
            tid
            for tid, info in self.pending_transfers.items()
            if not info.get("completed", False)
        ]

    def cleanup_completed_transfers(self, max_age_seconds: int = 3600) -> None:
        """Clean up old completed transfers."""
        current_time = time.time()
        to_remove = []

        for transfer_id, info in self.pending_transfers.items():
            if info.get("completed"):
                age = current_time - info.get("completion_time", 0)
                if age > max_age_seconds:
                    to_remove.append(transfer_id)

        for transfer_id in to_remove:
            del self.pending_transfers[transfer_id]
            logger.debug(f"Cleaned up old transfer {transfer_id}")
