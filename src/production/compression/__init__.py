"""Production compression components organized by Sprint 2."""

# Export main compression classes
__all__ = []

# Primary unified compression interface
try:
    from .unified_compressor import (
        UnifiedCompressor,
        CompressionStrategy,
        CompressionResult,
        compress_simple,
        compress_mobile,
        compress_advanced
    )
    __all__.extend([
        "UnifiedCompressor", 
        "CompressionStrategy", 
        "CompressionResult",
        "compress_simple",
        "compress_mobile", 
        "compress_advanced"
    ])
except ImportError:
    # Handle missing dependencies gracefully
    UnifiedCompressor = None
    CompressionStrategy = None
    CompressionResult = None
    compress_simple = None
    compress_mobile = None
    compress_advanced = None

try:
    from .compression_pipeline import CompressionPipeline

    __all__.append("CompressionPipeline")
except ImportError:
    # Handle missing dependencies gracefully
    CompressionPipeline = None

try:
    from .model_compression.model_compression import ModelCompression

    __all__.append("ModelCompression")
except ImportError:
    # Handle missing dependencies gracefully
    ModelCompression = None

# For backwards compatibility, also try to export main classes
try:
    from .model_compression import HyperCompressor, ModelCompressionTask

    __all__.extend(["HyperCompressor", "ModelCompressionTask"])
except ImportError:
    HyperCompressor = None
    ModelCompressionTask = None
