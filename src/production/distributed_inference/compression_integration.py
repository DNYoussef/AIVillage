"""Compression Integration for Distributed Inference.

This module integrates the existing compression pipeline with distributed sharding,
enabling compressed model shards to be distributed efficiently across devices.
"""

import asyncio
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn

from AIVillage.src.production.compression.compression_pipeline import CompressionConfig, CompressionPipeline

from .model_sharding_engine import ModelShard, ModelShardingEngine, ShardingPlan

logger = logging.getLogger(__name__)


@dataclass
class CompressedShard:
    """Represents a compressed model shard."""

    shard_id: str
    original_shard: ModelShard
    compressed_model_path: str
    compression_ratio: float
    compressed_size_mb: float
    original_size_mb: float
    compression_method: str = "BitNet"
    decompression_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedCompressionConfig:
    """Configuration for distributed compression."""

    # Compression settings
    bitnet_zero_threshold: float = 0.02
    enable_compression: bool = True
    compression_cache_dir: str = "./compressed_shards"

    # Distribution settings
    parallel_compression: bool = True
    max_compression_workers: int = 3
    compression_timeout_seconds: float = 300.0

    # Decompression settings
    cache_decompressed_layers: bool = True
    max_cache_size_mb: float = 1024.0  # 1GB cache
    preload_next_shard: bool = True

    # Performance settings
    use_streaming_decompression: bool = True
    chunk_size_mb: float = 32.0


class DistributedCompressionManager:
    """Manages compression/decompression of distributed model shards."""

    def __init__(
        self,
        sharding_engine: ModelShardingEngine,
        config: DistributedCompressionConfig | None = None,
    ) -> None:
        self.sharding_engine = sharding_engine
        self.config = config or DistributedCompressionConfig()

        # Compression state
        self.compressed_shards: dict[str, CompressedShard] = {}
        self.compression_cache: dict[str, Any] = {}  # Decompressed model cache
        self.active_compressions: dict[str, asyncio.Task] = {}

        # Performance tracking
        self.stats = {
            "shards_compressed": 0,
            "shards_decompressed": 0,
            "total_compression_time": 0.0,
            "total_decompression_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "compression_ratio_avg": 0.0,
            "storage_saved_mb": 0.0,
        }

        # Ensure cache directory exists
        Path(self.config.compression_cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info("DistributedCompressionManager initialized")

    async def compress_sharding_plan(self, plan: ShardingPlan) -> dict[str, CompressedShard]:
        """Compress all shards in a sharding plan."""
        logger.info(f"Starting compression of {len(plan.shards)} shards")
        start_time = time.time()

        try:
            if self.config.parallel_compression:
                # Compress shards in parallel
                compression_tasks = []
                for shard in plan.shards:
                    task = asyncio.create_task(self._compress_shard(shard, plan.model_name))
                    compression_tasks.append(task)
                    self.active_compressions[shard.shard_id] = task

                # Wait for all compressions with timeout
                compressed_shards = await asyncio.wait_for(
                    asyncio.gather(*compression_tasks, return_exceptions=True),
                    timeout=self.config.compression_timeout_seconds,
                )

                # Process results
                successful_compressions = {}
                for _i, result in enumerate(compressed_shards):
                    if isinstance(result, Exception):
                        logger.error(f"Shard compression failed: {result}")
                    else:
                        successful_compressions[result.shard_id] = result
                        self.compressed_shards[result.shard_id] = result

            else:
                # Sequential compression
                successful_compressions = {}
                for shard in plan.shards:
                    try:
                        compressed_shard = await self._compress_shard(shard, plan.model_name)
                        successful_compressions[compressed_shard.shard_id] = compressed_shard
                        self.compressed_shards[compressed_shard.shard_id] = compressed_shard
                    except Exception as e:
                        logger.exception(f"Failed to compress shard {shard.shard_id}: {e}")

            # Update statistics
            duration = time.time() - start_time
            self.stats["total_compression_time"] += duration
            self.stats["shards_compressed"] += len(successful_compressions)

            # Calculate average compression ratio
            if successful_compressions:
                total_ratio = sum(cs.compression_ratio for cs in successful_compressions.values())
                self.stats["compression_ratio_avg"] = total_ratio / len(successful_compressions)

                total_savings = sum(
                    cs.original_size_mb - cs.compressed_size_mb for cs in successful_compressions.values()
                )
                self.stats["storage_saved_mb"] += total_savings

            logger.info(
                f"Compression completed: {len(successful_compressions)}/{len(plan.shards)} shards in {duration:.2f}s"
            )
            return successful_compressions

        except asyncio.TimeoutError:
            logger.exception("Compression timeout exceeded")
            raise
        except Exception as e:
            logger.exception(f"Compression failed: {e}")
            raise
        finally:
            # Clean up active compression tasks
            for task in self.active_compressions.values():
                if not task.done():
                    task.cancel()
            self.active_compressions.clear()

    async def _compress_shard(self, shard: ModelShard, model_name: str) -> CompressedShard:
        """Compress a single model shard."""
        logger.debug(f"Compressing shard {shard.shard_id} with layers {shard.layer_indices}")
        start_time = time.time()

        try:
            # Create shard-specific model path
            shard_cache_dir = Path(self.config.compression_cache_dir) / f"shard_{shard.shard_id}"
            shard_cache_dir.mkdir(exist_ok=True)

            # Extract shard layers from full model (simplified implementation)
            # In a real implementation, this would load only the required layers
            shard_model_path = await self._extract_shard_layers(model_name, shard, shard_cache_dir)

            # Configure compression for this shard
            compression_config = CompressionConfig(
                input_model_path=shard_model_path,
                output_model_path=str(shard_cache_dir / "compressed"),
                bitnet_zero_threshold=self.config.bitnet_zero_threshold,
                eval_before_after=False,  # Skip evaluation for speed
                calibration_samples=100,  # Reduced for faster compression
                device="auto",
            )

            # Run compression pipeline
            pipeline = CompressionPipeline(compression_config)
            compression_results = await pipeline.run_compression_pipeline()

            if not compression_results["success"]:
                msg = f"Compression failed for shard {shard.shard_id}"
                raise RuntimeError(msg)

            # Create compressed shard metadata
            compressed_shard = CompressedShard(
                shard_id=shard.shard_id,
                original_shard=shard,
                compressed_model_path=compression_results["model_path"],
                compression_ratio=compression_results["compression_ratio"],
                compressed_size_mb=compression_results.get(
                    "compressed_size_mb",
                    shard.memory_mb / compression_results["compression_ratio"],
                ),
                original_size_mb=shard.memory_mb,
                decompression_time_ms=0.0,  # Will be measured during decompression
                metadata={
                    "compression_method": "BitNet",
                    "layers": shard.layer_indices,
                    "device_target": shard.device_id,
                    "compression_duration": time.time() - start_time,
                },
            )

            logger.debug(f"Shard compression completed: {compressed_shard.compression_ratio:.1f}x reduction")
            return compressed_shard

        except Exception as e:
            logger.exception(f"Shard compression failed: {e}")
            raise

    async def _extract_shard_layers(self, model_name: str, shard: ModelShard, output_dir: Path) -> str:
        """Extract specific layers for a shard (simplified implementation).

        This method loads the model weights, selects only the parameters that
        correspond to the requested ``layer_indices`` and writes them to a new
        directory.  The returned path can then be used as an input to the
        compression pipeline.  Only a very small subset of the full HuggingFace
        save format is implemented – just enough for our unit tests – but the
        logic mirrors what a real implementation would do.
        """
        # Destination for the shard-specific model
        shard_model_path = output_dir / "shard_model"
        shard_model_path.mkdir(exist_ok=True)

        logger.debug(f"Extracting layers {shard.layer_indices} from {model_name} into {shard_model_path}")

        # ------------------------------------------------------------------
        # Load the model weights.  We first attempt to load from a local path
        # to avoid network calls during testing.  If the path does not exist we
        # fall back to the HuggingFace ``from_pretrained`` helper.
        # ------------------------------------------------------------------
        model_path = Path(model_name)
        state_dict: dict[str, torch.Tensor]
        if model_path.exists():
            # If a directory is provided, assume the weights are stored in
            # ``pytorch_model.bin`` – mirroring the HF layout.
            weight_file = model_path / "pytorch_model.bin" if model_path.is_dir() else model_path
            state_dict = torch.load(weight_file, map_location="cpu")
        else:  # pragma: no cover - network loading is not used in tests
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(model_name)
            state_dict = model.state_dict()

        # ------------------------------------------------------------------
        # Filter out parameters that do not belong to the requested layers.
        # We match common naming patterns used by transformer implementations
        # (``layers.<idx>``, ``model.layers.<idx>``, ``transformer.h.<idx>``, ...)
        # ------------------------------------------------------------------
        selected_state: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            for idx in shard.layer_indices:
                patterns = (
                    f"layers.{idx}.",
                    f"model.layers.{idx}.",
                    f"transformer.h.{idx}.",
                )
                if any(p in key for p in patterns):
                    selected_state[key] = tensor
                    break

        # Save the filtered state dict to the shard directory
        torch.save(selected_state, shard_model_path / "pytorch_model.bin")

        return str(shard_model_path)

    async def decompress_shard(self, shard_id: str) -> nn.Module:
        """Decompress a shard for inference."""
        if shard_id not in self.compressed_shards:
            msg = f"Compressed shard not found: {shard_id}"
            raise ValueError(msg)

        # Check cache first
        if shard_id in self.compression_cache:
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache hit for shard {shard_id}")
            return self.compression_cache[shard_id]

        self.stats["cache_misses"] += 1
        compressed_shard = self.compressed_shards[shard_id]

        logger.debug(f"Decompressing shard {shard_id}")
        start_time = time.time()

        try:
            # Load compressed model
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                compressed_shard.compressed_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            # Measure decompression time
            decompression_time = (time.time() - start_time) * 1000  # Convert to ms
            compressed_shard.decompression_time_ms = decompression_time

            # Cache if enabled and space available
            if self.config.cache_decompressed_layers:
                if self._get_cache_size_mb() < self.config.max_cache_size_mb:
                    self.compression_cache[shard_id] = model
                else:
                    # Evict oldest cached shard
                    self._evict_oldest_cache_entry()
                    self.compression_cache[shard_id] = model

            # Update statistics
            self.stats["shards_decompressed"] += 1
            self.stats["total_decompression_time"] += decompression_time

            logger.debug(f"Shard decompression completed in {decompression_time:.1f}ms")
            return model

        except Exception as e:
            logger.exception(f"Shard decompression failed: {e}")
            raise

    async def decompress_for_inference(self, shard_ids: list[str]) -> list[nn.Module]:
        """Decompress multiple shards for inference pipeline."""
        logger.debug(f"Decompressing {len(shard_ids)} shards for inference")

        # Preload shards if streaming decompression is enabled
        if self.config.use_streaming_decompression:
            return await self._streaming_decompress(shard_ids)
        # Load all shards at once
        decompressed_models = []
        for shard_id in shard_ids:
            model = await self.decompress_shard(shard_id)
            decompressed_models.append(model)
        return decompressed_models

    async def _streaming_decompress(self, shard_ids: list[str]) -> list[nn.Module]:
        """Stream decompression of shards to minimize memory usage."""
        decompressed_models = []

        for i, shard_id in enumerate(shard_ids):
            # Decompress current shard
            model = await self.decompress_shard(shard_id)
            decompressed_models.append(model)

            # Preload next shard if enabled
            if (
                self.config.preload_next_shard
                and i + 1 < len(shard_ids)
                and shard_ids[i + 1] not in self.compression_cache
            ):
                next_shard_id = shard_ids[i + 1]
                asyncio.create_task(self.decompress_shard(next_shard_id))

        return decompressed_models

    def _get_cache_size_mb(self) -> float:
        """Estimate current cache size in MB."""
        # Simplified estimation based on number of cached models
        # In reality, you'd want to track actual memory usage
        return len(self.compression_cache) * 100  # Assume 100MB per cached shard

    def _evict_oldest_cache_entry(self) -> None:
        """Evict the oldest entry from the cache."""
        if self.compression_cache:
            oldest_key = next(iter(self.compression_cache))
            del self.compression_cache[oldest_key]
            logger.debug(f"Evicted cached shard: {oldest_key}")

    async def distribute_compressed_shards(self, compressed_shards: dict[str, CompressedShard]) -> dict[str, bool]:
        """Distribute compressed shards to their target devices."""
        logger.info(f"Distributing {len(compressed_shards)} compressed shards")

        distribution_results = {}

        for shard_id, compressed_shard in compressed_shards.items():
            try:
                # Get target device
                target_device = compressed_shard.original_shard.device_id

                if target_device == self.sharding_engine.p2p_node.node_id:
                    # Local device - shard is already available
                    distribution_results[shard_id] = True
                else:
                    # Remote device - send compressed shard
                    success = await self._send_compressed_shard(compressed_shard, target_device)
                    distribution_results[shard_id] = success

            except Exception as e:
                logger.exception(f"Failed to distribute shard {shard_id}: {e}")
                distribution_results[shard_id] = False

        successful_distributions = sum(1 for success in distribution_results.values() if success)
        logger.info(f"Shard distribution completed: {successful_distributions}/{len(compressed_shards)} successful")

        return distribution_results

    async def _send_compressed_shard(self, compressed_shard: CompressedShard, target_device: str) -> bool:
        """Send compressed shard to target device."""
        try:
            # Create shard transfer message
            transfer_message = {
                "type": "COMPRESSED_SHARD_TRANSFER",
                "shard_id": compressed_shard.shard_id,
                "compression_ratio": compressed_shard.compression_ratio,
                "compressed_size_mb": compressed_shard.compressed_size_mb,
                "layer_indices": compressed_shard.original_shard.layer_indices,
                "metadata": compressed_shard.metadata,
            }

            # Send metadata first
            success = await self.sharding_engine.p2p_node.send_to_peer(target_device, transfer_message)

            if success:
                # After metadata acknowledgement, stream the actual file
                file_path = Path(compressed_shard.compressed_model_path)
                if not file_path.exists():
                    logger.error(f"Compressed shard file not found: {file_path}")
                    return False

                file_sent = await self.sharding_engine.p2p_node.send_file(target_device, str(file_path))

                if file_sent:
                    logger.debug(f"Compressed shard {compressed_shard.shard_id} sent to {target_device}")
                    return True

                logger.error(f"Failed to send shard file for {compressed_shard.shard_id} to {target_device}")
                return False

            logger.error(f"Failed to send shard metadata to {target_device}")
            return False

        except Exception as e:
            logger.exception(f"Shard transfer failed: {e}")
            return False

    def get_compression_stats(self) -> dict[str, Any]:
        """Get compression statistics."""
        return {
            "statistics": self.stats.copy(),
            "compressed_shards": len(self.compressed_shards),
            "cached_shards": len(self.compression_cache),
            "cache_size_mb": self._get_cache_size_mb(),
            "active_compressions": len(self.active_compressions),
            "config": {
                "compression_enabled": self.config.enable_compression,
                "parallel_compression": self.config.parallel_compression,
                "max_workers": self.config.max_compression_workers,
                "cache_enabled": self.config.cache_decompressed_layers,
                "max_cache_size_mb": self.config.max_cache_size_mb,
                "streaming_decompression": self.config.use_streaming_decompression,
            },
            "compression_methods": {
                "bitnet": {
                    "threshold": self.config.bitnet_zero_threshold,
                    "shards_compressed": sum(
                        1 for cs in self.compressed_shards.values() if cs.compression_method == "BitNet"
                    ),
                }
            },
        }

    async def cleanup_compression_cache(self) -> None:
        """Clean up compression cache and temporary files."""
        logger.info("Cleaning up compression cache")

        # Clear memory cache
        self.compression_cache.clear()

        # Clean up temporary files
        cache_dir = Path(self.config.compression_cache_dir)
        if cache_dir.exists():
            try:
                import shutil

                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Compression cache cleaned up")
            except Exception as e:
                logger.exception(f"Failed to clean up cache directory: {e}")

        # Reset state
        self.compressed_shards.clear()

        # Cancel active compressions
        for task in self.active_compressions.values():
            if not task.done():
                task.cancel()
        self.active_compressions.clear()

    async def optimize_compression_settings(self, performance_target: float = 0.8) -> dict[str, Any]:
        """Optimize compression settings based on performance requirements."""
        logger.info(f"Optimizing compression settings for performance target: {performance_target}")

        # Analyze current performance
        current_stats = self.get_compression_stats()

        optimization_suggestions = {
            "current_performance": current_stats,
            "suggested_changes": [],
            "estimated_improvement": 0.0,
        }

        # Suggest optimizations based on statistics
        avg_compression_time = self.stats["total_compression_time"] / max(1, self.stats["shards_compressed"])

        if avg_compression_time > 60.0:  # If compression takes more than 1 minute per shard
            optimization_suggestions["suggested_changes"].append(
                {
                    "setting": "parallel_compression",
                    "current": self.config.parallel_compression,
                    "suggested": True,
                    "reason": "High compression time detected",
                }
            )

        cache_hit_ratio = self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])

        if cache_hit_ratio < 0.7:  # Low cache hit ratio
            optimization_suggestions["suggested_changes"].append(
                {
                    "setting": "max_cache_size_mb",
                    "current": self.config.max_cache_size_mb,
                    "suggested": self.config.max_cache_size_mb * 1.5,
                    "reason": "Low cache hit ratio detected",
                }
            )

        logger.info(f"Generated {len(optimization_suggestions['suggested_changes'])} optimization suggestions")
        return optimization_suggestions
