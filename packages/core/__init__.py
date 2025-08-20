"""AIVillage Core - Foundation components and infrastructure."""

__version__ = "1.2.0"
__author__ = "AI Village Team"

# Expose key components for easier importing
try:
    __all__ = ["SimpleQuantizer", "__version__"]
except ImportError:
    # Compression module not available
    __all__ = ["__version__"]

# Only import modules that actually exist and are needed
# Lazy imports to avoid circular dependencies
