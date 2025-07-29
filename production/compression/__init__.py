"""Production compression components organized by Sprint 2."""

# Export main compression classes
__all__ = []

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
