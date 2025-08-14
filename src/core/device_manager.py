"""Device management and fallback utilities.

This module provides centralized device detection and management
with automatic fallbacks for GPU/CPU configurations.
"""

import builtins
import contextlib
import logging
import os

logger = logging.getLogger(__name__)

# Global device cache
_device_info_cache: dict | None = None


def get_available_device(preferred_device: str = "auto") -> str:
    """Get the best available device with fallback.

    Args:
        preferred_device: Preferred device ("auto", "cuda", "cpu", "mps")

    Returns:
        Device string that can be used with PyTorch/transformers
    """
    global _device_info_cache

    if _device_info_cache is None:
        _device_info_cache = _detect_device_capabilities()

    # Handle auto selection
    if preferred_device == "auto":
        if _device_info_cache["cuda_available"]:
            return "cuda"
        if _device_info_cache["mps_available"]:
            return "mps"
        return "cpu"

    # Handle specific device requests
    if preferred_device == "cuda":
        if _device_info_cache["cuda_available"]:
            return "cuda"
        logger.warning("CUDA requested but not available, falling back to CPU")
        return "cpu"

    if preferred_device == "mps":
        if _device_info_cache["mps_available"]:
            return "mps"
        logger.warning("MPS requested but not available, falling back to CPU")
        return "cpu"

    # Default to CPU for any other case
    return "cpu"


def _detect_device_capabilities() -> dict:
    """Detect available device capabilities."""
    capabilities = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_version": None,
        "mps_available": False,
        "torch_available": False,
        "device_memory": {},
        "recommended_device": "cpu",
    }

    # Try to import torch
    try:
        import torch

        capabilities["torch_available"] = True

        # Check CUDA
        if torch.cuda.is_available():
            capabilities["cuda_available"] = True
            capabilities["cuda_device_count"] = torch.cuda.device_count()
            with contextlib.suppress(builtins.BaseException):
                capabilities["cuda_version"] = torch.version.cuda

            # Get GPU memory info
            try:
                for i in range(capabilities["cuda_device_count"]):
                    memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    capabilities["device_memory"][f"cuda:{i}"] = {
                        "total_gb": round(memory_total, 2),
                        "allocated_gb": round(memory_allocated, 2),
                        "available_gb": round(memory_total - memory_allocated, 2),
                    }
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory info: {e}")

        # Check MPS (Apple Silicon)
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                capabilities["mps_available"] = True
        except Exception as e:
            logger.debug(f"MPS availability check failed: {e}")
            capabilities["mps_available"] = False

        # Determine recommended device
        if capabilities["cuda_available"]:
            capabilities["recommended_device"] = "cuda"
        elif capabilities["mps_available"]:
            capabilities["recommended_device"] = "mps"
        else:
            capabilities["recommended_device"] = "cpu"

    except ImportError:
        logger.warning("PyTorch not available, CPU mode only")

    return capabilities


def get_device_info() -> dict:
    """Get comprehensive device information."""
    global _device_info_cache

    if _device_info_cache is None:
        _device_info_cache = _detect_device_capabilities()

    return _device_info_cache.copy()


def configure_device_for_component(component_name: str, config_device: str | None = None) -> str:
    """Configure device for a specific component with appropriate fallbacks.

    Args:
        component_name: Name of the component (for logging)
        config_device: Device specified in configuration

    Returns:
        Actual device to use
    """
    # Get device from config or environment
    device = config_device or os.environ.get("RAG_DEVICE", "auto")

    # Get actual available device
    actual_device = get_available_device(device)

    if device not in (actual_device, "auto"):
        logger.info(f"{component_name}: Requested device '{device}' not available, using '{actual_device}'")
    else:
        logger.info(f"{component_name}: Using device '{actual_device}'")

    return actual_device


def get_optimal_batch_size(device: str, model_size: str = "small") -> int:
    """Get optimal batch size based on device and model size.

    Args:
        device: Device being used
        model_size: Size category ("small", "medium", "large")

    Returns:
        Recommended batch size
    """
    device_info = get_device_info()

    if device.startswith("cuda") and device_info["cuda_available"]:
        # Get GPU memory
        gpu_memory = 0
        device_idx = device.split(":")[-1] if ":" in device else "0"
        memory_key = f"cuda:{device_idx}"

        if memory_key in device_info["device_memory"]:
            gpu_memory = device_info["device_memory"][memory_key]["available_gb"]

        # Batch size recommendations based on GPU memory
        if model_size == "small":  # <1B parameters
            if gpu_memory >= 8:
                return 32
            if gpu_memory >= 4:
                return 16
            if gpu_memory >= 2:
                return 8
            return 4
        if model_size == "medium":  # 1-7B parameters
            if gpu_memory >= 16:
                return 16
            if gpu_memory >= 8:
                return 8
            if gpu_memory >= 4:
                return 4
            return 2
        if gpu_memory >= 24:
            return 8
        if gpu_memory >= 16:
            return 4
        if gpu_memory >= 8:
            return 2
        return 1

    # CPU fallback batch sizes
    if model_size == "small":
        return 16
    if model_size == "medium":
        return 8
    return 4


def log_device_status() -> None:
    """Log comprehensive device status for debugging."""
    info = get_device_info()

    logger.info("=== Device Status ===")
    logger.info(f"PyTorch available: {info['torch_available']}")
    logger.info(f"CUDA available: {info['cuda_available']}")

    if info["cuda_available"]:
        logger.info(f"CUDA device count: {info['cuda_device_count']}")
        logger.info(f"CUDA version: {info['cuda_version']}")

        for device, memory in info["device_memory"].items():
            logger.info(f"{device}: {memory['available_gb']:.1f}GB available / {memory['total_gb']:.1f}GB total")

    logger.info(f"MPS available: {info['mps_available']}")
    logger.info(f"Recommended device: {info['recommended_device']}")
    logger.info("==================")


def ensure_device_compatibility() -> bool | None:
    """Ensure device setup is working correctly and handle common issues."""
    warnings_to_suppress = []

    try:
        import torch

        # Suppress common warnings that don't affect functionality
        if not torch.cuda.is_available():
            warnings_to_suppress.extend(
                [
                    "CUDA is not available",
                    "torch.cuda.is_available() returned False",
                ]
            )

        # Test device operations
        get_device_info()
        device = get_available_device("auto")

        if device == "cpu":
            logger.info("Running in CPU mode - GPU acceleration not available")
        else:
            logger.info(f"GPU acceleration available using {device}")

        return True

    except Exception as e:
        logger.exception(f"Device compatibility check failed: {e}")
        logger.info("Falling back to CPU mode")
        return False


# Environment variable helpers
def set_device_env_vars() -> None:
    """Set appropriate environment variables for device configuration."""
    device_info = get_device_info()

    # Set RAG device if not already set
    if "RAG_DEVICE" not in os.environ:
        os.environ["RAG_DEVICE"] = device_info["recommended_device"]

    # Set CUDA visibility if needed
    if not device_info["cuda_available"] and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


if __name__ == "__main__":
    # Test device detection
    log_device_status()

    # Test device selection
    print(f"Auto device: {get_available_device('auto')}")
    print(f"CUDA device: {get_available_device('cuda')}")
    print(f"CPU device: {get_available_device('cpu')}")

    # Test batch size recommendations
    device = get_available_device("auto")
    print(f"Optimal batch size (small model): {get_optimal_batch_size(device, 'small')}")
    print(f"Optimal batch size (medium model): {get_optimal_batch_size(device, 'medium')}")
    print(f"Optimal batch size (large model): {get_optimal_batch_size(device, 'large')}")
