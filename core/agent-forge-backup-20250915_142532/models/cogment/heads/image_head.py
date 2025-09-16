"""Image input heads for ARC tasks and visual reasoning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageHead(nn.Module):
    """Base image head for converting image data to model representations."""

    def __init__(self, d_model: int, input_channels: int = 3):
        super().__init__()
        self.d_model = d_model
        self.input_channels = input_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to d_model representations."""
        # Base implementation - should be overridden by subclasses
        # For now, return zero tensor with correct shape
        B = images.shape[0]
        return torch.zeros(B, 1, self.d_model)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ARCImageHead(ImageHead):
    """
    ARC-specific image head for 30x30 grids with 10 colors.

    Optimized for ARC reasoning tasks:
    - Input: Grid representations [B, H, W] or [B, C, H, W]
    - Output: Sequence of d_model vectors for RefinementCore
    - Lightweight: <100K parameters for budget compliance
    """

    def __init__(
        self,
        d_model: int,
        max_grid_size: int = 30,
        num_colors: int = 10,
        patch_size: int = 2,
        use_positional_encoding: bool = True,
    ):
        super().__init__(d_model, input_channels=1)

        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.patch_size = patch_size
        self.use_positional_encoding = use_positional_encoding

        # Color embedding: convert color indices to vectors
        self.color_embedding = nn.Embedding(num_colors + 1, d_model // 4)  # +1 for padding/unknown

        # Tiny CNN for spatial pattern extraction
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(d_model // 4, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model // 2, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((patch_size, patch_size)) if patch_size > 1 else nn.Identity(),
        )

        # Project to final d_model
        spatial_features = (d_model // 2) * (patch_size**2)
        self.projection = nn.Linear(spatial_features, d_model)

        # Positional encodings for grid positions
        if use_positional_encoding:
            self.register_buffer("position_encodings", self._create_positional_encodings(max_grid_size, d_model))

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.normal_(self.color_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _create_positional_encodings(self, max_size: int, d_model: int) -> torch.Tensor:
        """Create 2D positional encodings for grid positions."""
        position = torch.arange(max_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # Create position encodings for rows and columns
        pe_row = torch.zeros(max_size, d_model)
        pe_col = torch.zeros(max_size, d_model)

        pe_row[:, 0::2] = torch.sin(position * div_term)
        pe_row[:, 1::2] = torch.cos(position * div_term)
        pe_col[:, 0::2] = torch.sin(position * div_term)
        pe_col[:, 1::2] = torch.cos(position * div_term)

        # Combine row and column encodings
        pe_2d = torch.zeros(max_size, max_size, d_model)
        for i in range(max_size):
            for j in range(max_size):
                pe_2d[i, j] = (pe_row[i] + pe_col[j]) / 2

        return pe_2d

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Convert ARC grids to sequence representations.

        Args:
            grids: Input grids [B, H, W] with color indices 0-9

        Returns:
            sequences: Grid representations [B, seq_len, d_model]
        """
        B, H, W = grids.shape

        # Clamp grid values to valid color range
        grids = torch.clamp(grids.long(), 0, self.num_colors)

        # Color embedding: [B, H, W] → [B, H, W, d_model//4]
        color_embeds = self.color_embedding(grids)  # [B, H, W, d_model//4]

        # Reshape for CNN: [B, d_model//4, H, W]
        color_embeds = color_embeds.permute(0, 3, 1, 2)

        # Spatial encoding
        spatial_features = self.spatial_encoder(color_embeds)  # [B, d_model//2, patch_h, patch_w]

        # Flatten spatial dimensions
        B, C, patch_H, patch_W = spatial_features.shape
        spatial_flat = spatial_features.view(B, C * patch_H * patch_W)  # [B, spatial_features]

        # Project to d_model
        grid_representation = self.projection(spatial_flat)  # [B, d_model]

        # Add positional encoding if enabled
        if self.use_positional_encoding and hasattr(self, "position_encodings"):
            # Use center position encoding for grid-level representation
            center_pos = self.position_encodings[H // 2, W // 2]  # [d_model]
            grid_representation = grid_representation + center_pos.unsqueeze(0)

        # Normalize
        grid_representation = self.norm(grid_representation)

        # Return as sequence (single step for grid-level representation)
        return grid_representation.unsqueeze(1)  # [B, 1, d_model]

    def forward_sequence(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Convert ARC grids to patch-based sequences for detailed analysis.

        Args:
            grids: Input grids [B, H, W] with color indices 0-9

        Returns:
            sequences: Patch sequences [B, num_patches, d_model]
        """
        B, H, W = grids.shape

        # Ensure grid fits in patches
        patch_H = (H + self.patch_size - 1) // self.patch_size
        patch_W = (W + self.patch_size - 1) // self.patch_size

        # Pad if necessary
        pad_H = patch_H * self.patch_size - H
        pad_W = patch_W * self.patch_size - W
        if pad_H > 0 or pad_W > 0:
            grids = F.pad(grids, (0, pad_W, 0, pad_H), value=0)

        # Extract patches
        patches = grids.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, patch_H * patch_W, self.patch_size, self.patch_size)

        # Process each patch
        patch_representations = []
        for i in range(patches.shape[1]):
            patch = patches[:, i]  # [B, patch_size, patch_size]
            patch_repr = self.forward(patch)  # [B, 1, d_model]
            patch_representations.append(patch_repr)

        # Concatenate patch representations
        sequences = torch.cat(patch_representations, dim=1)  # [B, num_patches, d_model]

        return sequences


class TinyConvImageHead(ImageHead):
    """
    Tiny convolutional image head for general image inputs.

    Extremely lightweight for parameter budget compliance.
    - Input: Standard images [B, C, H, W]
    - Output: d_model representation
    - Target: <50K parameters
    """

    def __init__(
        self,
        d_model: int,
        input_channels: int = 3,
        output_seq_len: int = 1,
        hidden_dim: int = 128,
    ):
        super().__init__(d_model, input_channels)

        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim

        # Tiny CNN backbone
        self.backbone = nn.Sequential(
            # First conv: reduce spatial dimensions quickly
            nn.Conv2d(input_channels, hidden_dim // 4, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Second conv: extract features
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Third conv: final features
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )

        # Project to d_model
        self.projection = nn.Linear(hidden_dim, d_model * output_seq_len)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to d_model representations.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            representations: Image features [B, output_seq_len, d_model]
        """
        B = images.shape[0]

        # Extract features
        features = self.backbone(images)  # [B, hidden_dim, 1, 1]
        features = features.view(B, -1)  # [B, hidden_dim]

        # Project to output
        projected = self.projection(features)  # [B, d_model * output_seq_len]
        projected = projected.view(B, self.output_seq_len, self.d_model)  # [B, seq_len, d_model]

        # Normalize
        representations = self.norm(projected)

        return representations


class PatchEmbeddingHead(ImageHead):
    """
    Vision Transformer-style patch embedding head.

    Ultra-lightweight patch-based image processing.
    - Input: Images [B, C, H, W]
    - Output: Patch sequence [B, num_patches, d_model]
    - Target: <75K parameters
    """

    def __init__(
        self,
        d_model: int,
        input_channels: int = 3,
        patch_size: int = 16,
        max_image_size: int = 224,
    ):
        super().__init__(d_model, input_channels)

        self.patch_size = patch_size
        self.max_image_size = max_image_size
        self.num_patches = (max_image_size // patch_size) ** 2

        # Patch embedding projection
        self.patch_embed = nn.Conv2d(input_channels, d_model, kernel_size=patch_size, stride=patch_size)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Normalization
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch sequences.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            patch_sequences: Patch embeddings [B, num_patches, d_model]
        """
        B, C, H, W = images.shape

        # Resize if necessary
        if H != self.max_image_size or W != self.max_image_size:
            images = F.interpolate(
                images, size=(self.max_image_size, self.max_image_size), mode="bilinear", align_corners=False
            )

        # Patch embedding
        patches = self.patch_embed(images)  # [B, d_model, H//patch_size, W//patch_size]
        B, D, pH, pW = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]

        # Add positional embeddings
        patches = patches + self.pos_embed[:, : patches.shape[1], :]

        # Normalize
        patches = self.norm(patches)

        return patches


def create_image_head(head_type: str, d_model: int, **kwargs) -> ImageHead:
    """
    Factory function for creating image heads.

    Args:
        head_type: Type of head ("arc", "tiny_conv", "patch_embed")
        d_model: Model dimension
        **kwargs: Additional arguments for specific head types

    Returns:
        ImageHead: Configured image head
    """
    if head_type == "arc":
        return ARCImageHead(d_model, **kwargs)
    elif head_type == "tiny_conv":
        return TinyConvImageHead(d_model, **kwargs)
    elif head_type == "patch_embed":
        return PatchEmbeddingHead(d_model, **kwargs)
    else:
        raise ValueError(f"Unknown image head type: {head_type}")


if __name__ == "__main__":
    # Test ARC image head
    d_model = 320
    arc_head = ARCImageHead(d_model)

    # Test with synthetic ARC grid
    batch_size = 2
    grid_size = 10
    test_grid = torch.randint(0, 10, (batch_size, grid_size, grid_size))

    output = arc_head(test_grid)
    print(f"ARC Head - Input: {test_grid.shape}, Output: {output.shape}")
    print(f"ARC Head parameters: {arc_head.count_parameters():,}")

    # Test sequence output
    seq_output = arc_head.forward_sequence(test_grid)
    print(f"ARC Head sequence - Output: {seq_output.shape}")

    # Test tiny conv head
    tiny_head = TinyConvImageHead(d_model)
    test_image = torch.randn(batch_size, 3, 64, 64)

    output = tiny_head(test_image)
    print(f"Tiny Conv Head - Input: {test_image.shape}, Output: {output.shape}")
    print(f"Tiny Conv Head parameters: {tiny_head.count_parameters():,}")
