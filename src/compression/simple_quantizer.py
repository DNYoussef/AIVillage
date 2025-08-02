"""Simple quantizer for mobile compression.

Forget complex compression - just make quantization work.
Target: Reduce model size by 4x for 2GB phones.
"""

import io
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CompressionError(Exception):
    """Raised when compression doesn't meet requirements."""


class SimpleQuantizer:
    """Real quantizer implementation using PyTorch dynamic quantization."""

    def __init__(self, target_compression_ratio: float = 4.0):
        """Initialize quantizer.
        
        Args:
            target_compression_ratio: Minimum compression ratio required
        """
        self.target_compression_ratio = target_compression_ratio
        self.compression_stats = {}

    def quantize_model(self, model_path: str) -> bytes:
        """Quantize a PyTorch model for mobile deployment.
        
        Args:
            model_path: Path to the PyTorch model file
            
        Returns:
            Compressed model as bytes
            
        Raises:
            CompressionError: If compression ratio is insufficient
        """
        try:
            import torch
            import torch.quantization
        except ImportError:
            raise CompressionError("PyTorch not available - cannot quantize")

        if not os.path.exists(model_path):
            raise CompressionError(f"Model file not found: {model_path}")

        # Load the original model
        try:
            model = torch.load(model_path, map_location="cpu")
        except Exception as e:
            raise CompressionError(f"Failed to load model: {e}")

        # Get original size
        original_size = os.path.getsize(model_path)
        logger.info(f"Original model size: {original_size / 1024 / 1024:.2f} MB")

        # Convert float32 -> int8 using dynamic quantization
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},  # Quantize these layer types
                dtype=torch.qint8
            )
        except Exception as e:
            raise CompressionError(f"Quantization failed: {e}")

        # Save quantized model to buffer and measure size
        buffer = io.BytesIO()
        try:
            torch.save(quantized_model, buffer)
            compressed_size = buffer.tell()
        except Exception as e:
            raise CompressionError(f"Failed to serialize quantized model: {e}")

        # Calculate compression ratio
        compression_ratio = original_size / compressed_size

        logger.info(f"Compressed model size: {compressed_size / 1024 / 1024:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")

        # Store stats
        self.compression_stats = {
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": compression_ratio,
            "original_size_mb": original_size / 1024 / 1024,
            "compressed_size_mb": compressed_size / 1024 / 1024,
        }

        # Check if we meet the target compression ratio
        if compression_ratio < self.target_compression_ratio:
            raise CompressionError(
                f"Only achieved {compression_ratio:.2f}x compression, "
                f"need {self.target_compression_ratio}x"
            )

        buffer.seek(0)
        return buffer.getvalue()

    def quantize_model_from_object(self, model: Any) -> bytes:
        """Quantize a PyTorch model object directly.
        
        Args:
            model: PyTorch model object
            
        Returns:
            Compressed model as bytes
        """
        try:
            import torch
            import torch.quantization
        except ImportError:
            raise CompressionError("PyTorch not available - cannot quantize")

        # Get original size by serializing to buffer
        original_buffer = io.BytesIO()
        torch.save(model, original_buffer)
        original_size = original_buffer.tell()

        logger.info(f"Original model size: {original_size / 1024 / 1024:.2f} MB")

        # Quantize the model
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
        except Exception as e:
            raise CompressionError(f"Quantization failed: {e}")

        # Serialize quantized model
        buffer = io.BytesIO()
        torch.save(quantized_model, buffer)
        compressed_size = buffer.tell()

        # Calculate compression ratio
        compression_ratio = original_size / compressed_size

        logger.info(f"Compressed model size: {compressed_size / 1024 / 1024:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")

        # Store stats
        self.compression_stats = {
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": compression_ratio,
            "original_size_mb": original_size / 1024 / 1024,
            "compressed_size_mb": compressed_size / 1024 / 1024,
        }

        # Check compression ratio
        if compression_ratio < self.target_compression_ratio:
            raise CompressionError(
                f"Only achieved {compression_ratio:.2f}x compression, "
                f"need {self.target_compression_ratio}x"
            )

        buffer.seek(0)
        return buffer.getvalue()

    def get_compression_stats(self) -> dict[str, Any]:
        """Get detailed compression statistics."""
        return self.compression_stats.copy()

    def is_mobile_ready(self, max_size_mb: float = 50.0) -> bool:
        """Check if compressed model is suitable for mobile deployment.
        
        Args:
            max_size_mb: Maximum acceptable model size in MB
            
        Returns:
            True if model is mobile-ready
        """
        if not self.compression_stats:
            return False

        compressed_size_mb = self.compression_stats.get("compressed_size_mb", float("inf"))
        return compressed_size_mb <= max_size_mb

    @staticmethod
    def load_quantized_model(model_bytes: bytes):
        """Load a quantized model from bytes.
        
        Args:
            model_bytes: Compressed model bytes
            
        Returns:
            Loaded PyTorch model
        """
        try:
            import torch
        except ImportError:
            raise CompressionError("PyTorch not available")

        buffer = io.BytesIO(model_bytes)
        return torch.load(buffer, map_location="cpu")

    def save_quantized_model(self, model_bytes: bytes, output_path: str) -> None:
        """Save quantized model to file.
        
        Args:
            model_bytes: Compressed model bytes
            output_path: Path to save the model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(model_bytes)

        logger.info(f"Saved quantized model to {output_path}")

    def compress_for_mobile(self, model_path: str, output_dir: str = "mobile_models") -> str:
        """Complete mobile compression pipeline.
        
        Args:
            model_path: Input model path
            output_dir: Output directory for compressed model
            
        Returns:
            Path to compressed model
        """
        # Quantize model
        compressed_bytes = self.quantize_model(model_path)

        # Generate output filename
        input_path = Path(model_path)
        output_dir = Path(output_dir)
        output_path = output_dir / f"{input_path.stem}_mobile_quantized.pt"

        # Save compressed model
        self.save_quantized_model(compressed_bytes, output_path)

        # Log results
        stats = self.get_compression_stats()
        logger.info("Mobile compression complete:")
        logger.info(f"  Original: {stats['original_size_mb']:.2f} MB")
        logger.info(f"  Compressed: {stats['compressed_size_mb']:.2f} MB")
        logger.info(f"  Ratio: {stats['compression_ratio']:.2f}x")
        logger.info(f"  Mobile ready: {self.is_mobile_ready()}")

        return str(output_path)
