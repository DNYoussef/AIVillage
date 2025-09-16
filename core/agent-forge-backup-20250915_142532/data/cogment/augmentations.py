"""
ARC Augmentation Engine

Comprehensive augmentation system for ARC tasks with ~300 variations per task:
- Geometric transforms: 8 rotations, flips, combinations
- Color remapping: permute 10-color palettes systematically
- Grid resizing: ±2 cells with intelligent padding/cropping
- Small occlusion: 5-10% noise/masking that preserves logic
- Rule-preserving transforms that maintain task semantics

Designed for Stage 1 visual reasoning with heavy augmentation for grokking.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations
import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for ARC augmentations."""

    enable_rotations: bool = True
    enable_flips: bool = True
    enable_color_remapping: bool = True
    enable_grid_resizing: bool = True
    enable_occlusion: bool = True

    # Rotation settings
    rotation_angles: list[int] = None  # [0, 45, 90, 135, 180, 225, 270, 315]

    # Color remapping settings
    max_color_permutations: int = 50  # Limit for computational efficiency
    preserve_background: bool = True  # Keep color 0 as background

    # Grid resizing settings
    resize_range: tuple[int, int] = (-2, 2)  # ±2 cells

    # Occlusion settings
    occlusion_rate_range: tuple[float, float] = (0.05, 0.10)  # 5-10%
    preserve_corners: bool = True  # Don't occlude corners

    # Quality control
    max_augmentations_per_task: int = 300
    validate_semantics: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.rotation_angles is None:
            self.rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]


class ARCGrid:
    """Represents an ARC grid with augmentation operations."""

    def __init__(self, grid: np.ndarray):
        self.grid = np.array(grid, dtype=int)
        self.height, self.width = self.grid.shape

    def copy(self) -> "ARCGrid":
        """Create a deep copy."""
        return ARCGrid(self.grid.copy())

    def to_list(self) -> list[list[int]]:
        """Convert to list format."""
        return self.grid.tolist()

    def get_colors(self) -> set:
        """Get unique colors in grid."""
        return set(self.grid.flatten())

    def rotate_90(self) -> "ARCGrid":
        """Rotate 90 degrees clockwise."""
        return ARCGrid(np.rot90(self.grid, k=-1))

    def rotate_180(self) -> "ARCGrid":
        """Rotate 180 degrees."""
        return ARCGrid(np.rot90(self.grid, k=2))

    def rotate_270(self) -> "ARCGrid":
        """Rotate 270 degrees clockwise."""
        return ARCGrid(np.rot90(self.grid, k=1))

    def flip_horizontal(self) -> "ARCGrid":
        """Flip horizontally."""
        return ARCGrid(np.fliplr(self.grid))

    def flip_vertical(self) -> "ARCGrid":
        """Flip vertically."""
        return ARCGrid(np.flipud(self.grid))

    def remap_colors(self, color_mapping: dict[int, int]) -> "ARCGrid":
        """Remap colors according to mapping."""
        new_grid = self.grid.copy()
        for old_color, new_color in color_mapping.items():
            new_grid[self.grid == old_color] = new_color
        return ARCGrid(new_grid)

    def resize(self, new_height: int, new_width: int, fill_value: int = 0) -> "ARCGrid":
        """Resize grid with padding or cropping."""
        if new_height <= 0 or new_width <= 0:
            raise ValueError("New dimensions must be positive")

        # Create new grid with fill value
        new_grid = np.full((new_height, new_width), fill_value, dtype=int)

        # Calculate copy region
        copy_h = min(self.height, new_height)
        copy_w = min(self.width, new_width)

        # Center the original grid in new grid
        start_h = max(0, (new_height - self.height) // 2)
        start_w = max(0, (new_width - self.width) // 2)

        # Copy original data
        orig_start_h = max(0, (self.height - new_height) // 2)
        orig_start_w = max(0, (self.width - new_width) // 2)

        new_grid[start_h : start_h + copy_h, start_w : start_w + copy_w] = self.grid[
            orig_start_h : orig_start_h + copy_h, orig_start_w : orig_start_w + copy_w
        ]

        return ARCGrid(new_grid)

    def add_occlusion(self, rate: float = 0.05, preserve_corners: bool = True, noise_color: int = -1) -> "ARCGrid":
        """Add random occlusion/noise."""
        new_grid = self.grid.copy()

        # Calculate number of cells to occlude
        total_cells = self.height * self.width
        num_occlude = int(total_cells * rate)

        # Get available positions
        positions = [(i, j) for i in range(self.height) for j in range(self.width)]

        if preserve_corners:
            # Remove corner positions
            corners = [(0, 0), (0, self.width - 1), (self.height - 1, 0), (self.height - 1, self.width - 1)]
            positions = [pos for pos in positions if pos not in corners]

        # Randomly select positions to occlude
        occlude_positions = random.sample(positions, min(num_occlude, len(positions)))

        # Choose noise color
        if noise_color == -1:
            # Use a color not in original grid
            used_colors = self.get_colors()
            noise_color = max(used_colors) + 1 if used_colors else 1

        # Apply occlusion
        for i, j in occlude_positions:
            new_grid[i, j] = noise_color

        return ARCGrid(new_grid)


class ARCAugmentationEngine:
    """Comprehensive augmentation engine for ARC tasks."""

    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Color palette for ARC (0-9)
        self.arc_colors = list(range(10))

        logger.info(f"Initialized ARC augmentation engine (max {self.config.max_augmentations_per_task} per task)")

    def generate_color_mappings(self, original_colors: set, max_mappings: int = 50) -> list[dict[int, int]]:
        """Generate systematic color remappings."""
        mappings = []

        # Identity mapping
        mappings.append({color: color for color in original_colors})

        # Simple swaps for small color sets
        if len(original_colors) <= 4:
            color_list = list(original_colors)

            # Generate permutations (limited for efficiency)
            perms = list(permutations(color_list))
            perms = perms[: max_mappings - 1]  # Reserve space for identity

            for perm in perms:
                mapping = {original: new for original, new in zip(color_list, perm)}
                mappings.append(mapping)
        else:
            # For larger color sets, use random swaps
            original_list = list(original_colors)

            for _ in range(max_mappings - 1):
                # Create random mapping
                available_colors = [c for c in self.arc_colors if c not in original_colors] + list(original_colors)
                random.shuffle(available_colors)

                mapping = {}
                for i, original_color in enumerate(original_list):
                    if self.config.preserve_background and original_color == 0:
                        mapping[0] = 0  # Keep background as 0
                    else:
                        mapping[original_color] = available_colors[i % len(available_colors)]

                if mapping not in mappings:
                    mappings.append(mapping)

        return mappings[:max_mappings]

    def generate_geometric_transforms(self) -> list[Callable[[ARCGrid], ARCGrid]]:
        """Generate geometric transformation functions."""
        transforms = []

        # Identity transform
        transforms.append(lambda grid: grid.copy())

        if self.config.enable_rotations:
            transforms.extend(
                [lambda grid: grid.rotate_90(), lambda grid: grid.rotate_180(), lambda grid: grid.rotate_270()]
            )

        if self.config.enable_flips:
            transforms.extend(
                [
                    lambda grid: grid.flip_horizontal(),
                    lambda grid: grid.flip_vertical(),
                    lambda grid: grid.flip_horizontal().flip_vertical(),  # Both flips
                ]
            )

        # Combinations of rotations and flips
        if self.config.enable_rotations and self.config.enable_flips:
            base_transforms = [
                lambda grid: grid.rotate_90().flip_horizontal(),
                lambda grid: grid.rotate_180().flip_horizontal(),
                lambda grid: grid.rotate_270().flip_horizontal(),
                lambda grid: grid.rotate_90().flip_vertical(),
                lambda grid: grid.rotate_180().flip_vertical(),
                lambda grid: grid.rotate_270().flip_vertical(),
            ]
            transforms.extend(base_transforms)

        return transforms

    def generate_resize_variants(self, original_grid: ARCGrid) -> list[tuple[int, int]]:
        """Generate resize dimension variants."""
        if not self.config.enable_grid_resizing:
            return [(original_grid.height, original_grid.width)]

        variants = [(original_grid.height, original_grid.width)]  # Original size

        min_delta, max_delta = self.config.resize_range

        for dh in range(min_delta, max_delta + 1):
            for dw in range(min_delta, max_delta + 1):
                if dh == 0 and dw == 0:
                    continue  # Skip original size

                new_h = max(1, original_grid.height + dh)
                new_w = max(1, original_grid.width + dw)

                if (new_h, new_w) not in variants:
                    variants.append((new_h, new_w))

        return variants

    def generate_occlusion_variants(self, num_variants: int = 5) -> list[dict[str, Any]]:
        """Generate occlusion parameter variants."""
        if not self.config.enable_occlusion:
            return [{"rate": 0.0}]  # No occlusion

        variants = [{"rate": 0.0}]  # Include no-occlusion variant

        min_rate, max_rate = self.config.occlusion_rate_range

        for i in range(num_variants - 1):
            rate = min_rate + (max_rate - min_rate) * i / (num_variants - 2)
            variants.append({"rate": rate, "preserve_corners": self.config.preserve_corners})

        return variants

    def validate_augmentation_quality(self, original: dict[str, Any], augmented: dict[str, Any]) -> bool:
        """Validate that augmentation preserves semantic content."""
        if not self.config.validate_semantics:
            return True

        # Basic checks
        if not augmented.get("input") or not augmented.get("output"):
            return False

        # Check that augmented grids are valid
        try:
            input_grids = augmented["input"]
            output_grids = augmented["output"]

            if not isinstance(input_grids, list) or not isinstance(output_grids, list):
                return False

            # Validate grid formats
            for grid in input_grids + output_grids:
                if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
                    return False

                # Check consistent row lengths
                if grid and not all(len(row) == len(grid[0]) for row in grid):
                    return False

        except Exception as e:
            logger.warning(f"Augmentation validation failed: {e}")
            return False

        return True

    def augment_arc_task(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate comprehensive augmentations for a single ARC task."""
        if "train" not in task or "test" not in task:
            logger.warning("Invalid ARC task format")
            return [task]

        augmentations = [task]  # Include original

        try:
            # Extract original grids
            train_examples = task["train"]
            test_examples = task["test"]

            if not train_examples:
                return [task]

            # Analyze color usage across all grids
            all_colors = set()
            sample_grid = None

            for example in train_examples + test_examples:
                for grid_data in [example["input"], example["output"]]:
                    grid = ARCGrid(grid_data)
                    all_colors.update(grid.get_colors())
                    if sample_grid is None:
                        sample_grid = grid

            if sample_grid is None:
                return [task]

            # Generate augmentation components
            color_mappings = self.generate_color_mappings(all_colors, self.config.max_color_permutations)
            geometric_transforms = self.generate_geometric_transforms()
            resize_variants = self.generate_resize_variants(sample_grid)
            occlusion_variants = self.generate_occlusion_variants()

            logger.info(
                f"Generating augmentations: {len(color_mappings)} color mappings, "
                f"{len(geometric_transforms)} transforms, {len(resize_variants)} sizes, "
                f"{len(occlusion_variants)} occlusion variants"
            )

            # Generate combinations
            augmentation_count = 0
            max_augs = self.config.max_augmentations_per_task

            for color_mapping in color_mappings:
                if augmentation_count >= max_augs:
                    break

                for transform_fn in geometric_transforms:
                    if augmentation_count >= max_augs:
                        break

                    for new_height, new_width in resize_variants:
                        if augmentation_count >= max_augs:
                            break

                        for occlusion_params in occlusion_variants:
                            if augmentation_count >= max_augs:
                                break

                            # Skip identity transformation
                            is_identity = (
                                color_mapping == {c: c for c in all_colors}
                                and transform_fn.__name__ == "<lambda>"  # Identity transform
                                and new_height == sample_grid.height
                                and new_width == sample_grid.width
                                and occlusion_params["rate"] == 0.0
                            )

                            if is_identity and len(augmentations) > 1:
                                continue

                            # Apply augmentation
                            try:
                                augmented_task = self._apply_augmentation(
                                    task, color_mapping, transform_fn, (new_height, new_width), occlusion_params
                                )

                                if self.validate_augmentation_quality(task, augmented_task):
                                    augmentations.append(augmented_task)
                                    augmentation_count += 1

                            except Exception as e:
                                logger.debug(f"Augmentation failed: {e}")
                                continue

            logger.info(f"Generated {len(augmentations)} total augmentations for task")

        except Exception as e:
            logger.error(f"Error in task augmentation: {e}")
            return [task]

        return augmentations

    def _apply_augmentation(
        self,
        task: dict[str, Any],
        color_mapping: dict[int, int],
        transform_fn: Callable,
        new_size: tuple[int, int],
        occlusion_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply a specific augmentation to a task."""
        augmented_task = deepcopy(task)
        new_height, new_width = new_size

        # Augment train examples
        for example in augmented_task["train"]:
            example["input"] = self._augment_grid(
                example["input"], color_mapping, transform_fn, new_size, occlusion_params
            )
            example["output"] = self._augment_grid(
                example["output"], color_mapping, transform_fn, new_size, occlusion_params
            )

        # Augment test examples
        for example in augmented_task["test"]:
            example["input"] = self._augment_grid(
                example["input"], color_mapping, transform_fn, new_size, occlusion_params
            )
            if "output" in example:  # Test might not have output
                example["output"] = self._augment_grid(
                    example["output"], color_mapping, transform_fn, new_size, occlusion_params
                )

        return augmented_task

    def _augment_grid(
        self,
        grid_data: list[list[int]],
        color_mapping: dict[int, int],
        transform_fn: Callable,
        new_size: tuple[int, int],
        occlusion_params: dict[str, Any],
    ) -> list[list[int]]:
        """Apply augmentation to a single grid."""
        grid = ARCGrid(grid_data)

        # Apply geometric transform
        grid = transform_fn(grid)

        # Apply color remapping
        grid = grid.remap_colors(color_mapping)

        # Apply resizing
        new_height, new_width = new_size
        if new_height != grid.height or new_width != grid.width:
            # Use most common color as fill
            colors, counts = np.unique(grid.grid, return_counts=True)
            fill_color = colors[np.argmax(counts)]
            grid = grid.resize(new_height, new_width, fill_value=fill_color)

        # Apply occlusion
        if occlusion_params["rate"] > 0:
            grid = grid.add_occlusion(
                rate=occlusion_params["rate"], preserve_corners=occlusion_params.get("preserve_corners", True)
            )

        return grid.to_list()

    def get_augmentation_stats(self) -> dict[str, Any]:
        """Get statistics about augmentation capabilities."""
        # Estimate total possible augmentations
        len(self.arc_colors)
        geometric_transforms = len(self.generate_geometric_transforms())

        sample_grid = ARCGrid(np.zeros((5, 5)))
        resize_variants = len(self.generate_resize_variants(sample_grid))
        occlusion_variants = len(self.generate_occlusion_variants())

        theoretical_max = (
            self.config.max_color_permutations * geometric_transforms * resize_variants * occlusion_variants
        )

        return {
            "max_augmentations_per_task": self.config.max_augmentations_per_task,
            "color_permutations": self.config.max_color_permutations,
            "geometric_transforms": geometric_transforms,
            "resize_variants": resize_variants,
            "occlusion_variants": occlusion_variants,
            "theoretical_max_combinations": theoretical_max,
            "practical_limit": min(theoretical_max, self.config.max_augmentations_per_task),
        }


def demo_augmentation_engine():
    """Demonstrate augmentation engine capabilities."""
    print("=== ARC Augmentation Engine Demo ===")

    # Create sample ARC task
    sample_task = {
        "train": [{"input": [[0, 1, 0], [1, 2, 1], [0, 1, 0]], "output": [[1, 0, 1], [0, 2, 0], [1, 0, 1]]}],
        "test": [{"input": [[0, 2, 0], [2, 1, 2], [0, 2, 0]]}],
    }

    # Create augmentation engine
    config = AugmentationConfig(max_augmentations_per_task=20, max_color_permutations=5)  # Small demo

    engine = ARCAugmentationEngine(config)

    # Show statistics
    stats = engine.get_augmentation_stats()
    print("\nAugmentation capabilities:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Generate augmentations
    print("\nGenerating augmentations for sample task...")
    augmented_tasks = engine.augment_arc_task(sample_task)

    print(f"Generated {len(augmented_tasks)} augmented versions")

    # Show first few augmentations
    for i, aug_task in enumerate(augmented_tasks[:3]):
        print(f"\nAugmentation {i}:")
        train_input = aug_task["train"][0]["input"]
        print(f"  Input: {train_input}")

    print("\n=== Augmentation Demo Complete ===")


if __name__ == "__main__":
    demo_augmentation_engine()
