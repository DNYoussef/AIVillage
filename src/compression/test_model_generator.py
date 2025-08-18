"""Test model generator for consistent compression testing."""

import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_test_model(layers: int = 3, hidden_size: int = 256, size_mb: float = 10.0) -> Any:
    """Create a test PyTorch model for compression testing.

    Args:
        layers: Number of linear layers
        hidden_size: Size of hidden layers
        size_mb: Target approximate size in MB

    Returns:
        PyTorch model object
    """
    try:
        from torch import nn
    except ImportError:
        msg = "PyTorch not available - cannot create test model"
        raise ImportError(msg)

    # Calculate input size to approximately reach target size
    # Rough calculation: each float32 parameter is 4 bytes
    target_params = int((size_mb * 1024 * 1024) / 4)

    # Distribute parameters across layers
    # For linear layers: params = (input_size * hidden_size) + hidden_size (bias)
    params_per_layer = target_params // layers if layers > 0 else target_params

    # Estimate input size for first layer
    input_size = max(1, int((params_per_layer - hidden_size) / hidden_size))

    # Create model as Sequential to avoid nested class pickle issues
    layer_list = []

    # First layer
    layer_list.append(nn.Linear(input_size, hidden_size))
    layer_list.append(nn.ReLU())

    # Hidden layers
    for _ in range(max(0, layers - 2)):
        layer_list.append(nn.Linear(hidden_size, hidden_size))
        layer_list.append(nn.ReLU())

    # Output layer
    if layers > 1:
        layer_list.append(nn.Linear(hidden_size, 10))  # 10 class classification

    model = nn.Sequential(*layer_list)

    # Count actual parameters
    total_params = sum(p.numel() for p in model.parameters())
    actual_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32

    logger.info("Created test model:")
    logger.info(f"  Layers: {layers}")
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Input size: {input_size}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Actual size: {actual_size_mb:.2f} MB")

    return model


def create_test_model_file(
    layers: int = 3,
    hidden_size: int = 256,
    size_mb: float = 10.0,
    save_path: str | None = None,
) -> str:
    """Create and save a test PyTorch model to file.

    Args:
        layers: Number of linear layers
        hidden_size: Size of hidden layers
        size_mb: Target approximate size in MB
        save_path: Path to save model (if None, uses temp file)

    Returns:
        Path to saved model file
    """
    try:
        import torch
    except ImportError:
        msg = "PyTorch not available"
        raise ImportError(msg)

    # Create model
    model = create_test_model(layers, hidden_size, size_mb)

    # Determine save path
    if save_path is None:
        temp_dir = Path(tempfile.mkdtemp())
        save_path = temp_dir / f"test_model_{layers}layers_{size_mb:.0f}mb.pt"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model, save_path)

    logger.info(f"Saved test model to: {save_path}")
    logger.info(f"File size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")

    return str(save_path)


def create_simple_cnn_model(input_channels: int = 3, num_classes: int = 10) -> Any:
    """Create a simple CNN model for testing.

    Args:
        input_channels: Number of input channels (e.g., 3 for RGB)
        num_classes: Number of output classes

    Returns:
        PyTorch CNN model
    """
    try:
        from torch import nn
    except ImportError:
        msg = "PyTorch not available"
        raise ImportError(msg)

    # Create CNN as Sequential to avoid pickle issues
    model = nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = (total_params * 4) / (1024 * 1024)

    logger.info("Created CNN model:")
    logger.info(f"  Input channels: {input_channels}")
    logger.info(f"  Output classes: {num_classes}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Size: {size_mb:.2f} MB")

    return model


def create_mixed_model() -> Any:
    """Create a model with both Conv2d and Linear layers for comprehensive testing.

    Returns:
        PyTorch model with mixed layer types
    """
    try:
        from torch import nn
    except ImportError:
        msg = "PyTorch not available"
        raise ImportError(msg)

    # Create mixed model as Sequential to avoid pickle issues
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())

    # Count conv and linear params separately
    conv_params = 0
    linear_params = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Linear):
            linear_params += sum(p.numel() for p in module.parameters())

    size_mb = (total_params * 4) / (1024 * 1024)

    logger.info("Created mixed model:")
    logger.info(f"  Conv parameters: {conv_params:,}")
    logger.info(f"  Linear parameters: {linear_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Size: {size_mb:.2f} MB")

    return model
