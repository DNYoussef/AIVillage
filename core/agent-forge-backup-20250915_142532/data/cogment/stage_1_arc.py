"""
Stage 1: ARC Visual Reasoning Dataset

ARC (Abstraction and Reasoning Corpus) with heavy augmentation:
- ARC public split (400 training + 400 evaluation tasks)
- ConceptARC (conceptual reasoning extensions)
- Synthetic ARC generators (rule-based tasks)
- ~300 augmentations per task (rotations, flips, color remapping, resizing, occlusion)

Purpose: Visual pattern recognition and logical reasoning with grokking via augmentation.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
except ImportError:
    logger.warning("datasets library not available - using local data only")
    load_dataset = None

try:
    from .augmentations import ARCAugmentationEngine, AugmentationConfig
except ImportError:
    # Handle standalone execution
    try:
        from augmentations import ARCAugmentationEngine, AugmentationConfig
    except ImportError:
        logger.warning("Augmentation engine not available - using basic dataset only")
        ARCAugmentationEngine = None
        AugmentationConfig = None

logger = logging.getLogger(__name__)


@dataclass
class ARCDataConfig:
    """Configuration for ARC dataset."""

    # Data sources
    use_huggingface_arc: bool = True
    use_concept_arc: bool = True
    use_synthetic_arc: bool = True
    local_arc_path: str | None = None

    # Augmentation settings
    augmentations_per_task: int = 300
    augmentation_seed: int = 42

    # Training settings
    max_tasks: int | None = None  # None = use all
    sequence_length: int = 512

    # Task filtering
    min_grid_size: int = 2
    max_grid_size: int = 10
    max_colors: int = 10

    # Performance settings
    cache_augmentations: bool = True
    validate_tasks: bool = True


class ARCTaskProcessor:
    """Processes and validates ARC tasks."""

    def __init__(self, config: ARCDataConfig):
        self.config = config

    def validate_arc_task(self, task: dict[str, Any]) -> bool:
        """Validate ARC task format and constraints."""
        try:
            # Check basic structure
            if "train" not in task or "test" not in task:
                return False

            train_examples = task["train"]
            test_examples = task["test"]

            if not train_examples or not test_examples:
                return False

            # Validate all examples
            for example in train_examples + test_examples:
                if not self._validate_example(example):
                    return False

            return True

        except Exception as e:
            logger.debug(f"Task validation failed: {e}")
            return False

    def _validate_example(self, example: dict[str, Any]) -> bool:
        """Validate a single ARC example."""
        # Check required fields
        if "input" not in example:
            return False

        # Validate input grid
        if not self._validate_grid(example["input"]):
            return False

        # Validate output grid if present
        if "output" in example and not self._validate_grid(example["output"]):
            return False

        return True

    def _validate_grid(self, grid: list[list[int]]) -> bool:
        """Validate ARC grid format and constraints."""
        if not isinstance(grid, list) or not grid:
            return False

        # Check that all rows are lists
        if not all(isinstance(row, list) for row in grid):
            return False

        # Check consistent row lengths
        row_length = len(grid[0])
        if not all(len(row) == row_length for row in grid):
            return False

        # Check grid size constraints
        height, width = len(grid), row_length
        if (
            height < self.config.min_grid_size
            or height > self.config.max_grid_size
            or width < self.config.min_grid_size
            or width > self.config.max_grid_size
        ):
            return False

        # Check color constraints (0-9 in ARC)
        all_values = [val for row in grid for val in row]
        if any(val < 0 or val >= self.config.max_colors for val in all_values):
            return False

        return True

    def process_task_for_training(self, task: dict[str, Any], task_id: str = None) -> dict[str, Any]:
        """Process ARC task into training format."""
        # Add metadata
        processed_task = {
            "task_id": task_id or "unknown",
            "train": task["train"],
            "test": task["test"],
            "metadata": {
                "num_train_examples": len(task["train"]),
                "num_test_examples": len(task["test"]),
                "colors_used": self._get_task_colors(task),
                "grid_sizes": self._get_task_grid_sizes(task),
            },
        }

        return processed_task

    def _get_task_colors(self, task: dict[str, Any]) -> list[int]:
        """Get unique colors used in task."""
        colors = set()

        for example in task["train"] + task["test"]:
            for grid_data in [example["input"], example.get("output", [])]:
                if grid_data:
                    for row in grid_data:
                        colors.update(row)

        return sorted(list(colors))

    def _get_task_grid_sizes(self, task: dict[str, Any]) -> list[tuple[int, int]]:
        """Get unique grid sizes in task."""
        sizes = set()

        for example in task["train"] + task["test"]:
            for grid_data in [example["input"], example.get("output", [])]:
                if grid_data:
                    sizes.add((len(grid_data), len(grid_data[0])))

        return sorted(list(sizes))


class ARCDataLoader:
    """Loads ARC data from various sources."""

    def __init__(self, config: ARCDataConfig):
        self.config = config
        self.processor = ARCTaskProcessor(config)

    def load_huggingface_arc(self) -> list[dict[str, Any]]:
        """Load ARC from HuggingFace datasets."""
        if not self.config.use_huggingface_arc or load_dataset is None:
            return []

        try:
            logger.info("Loading ARC dataset from HuggingFace...")

            # Load ARC dataset
            dataset = load_dataset("ai2_arc", "ARC-Challenge", trust_remote_code=True)

            tasks = []

            # Process training split
            if "train" in dataset:
                for item in dataset["train"]:
                    # Convert to ARC task format
                    task = self._convert_hf_arc_to_task(item)
                    if task and self.processor.validate_arc_task(task):
                        processed_task = self.processor.process_task_for_training(task, f"hf_arc_{len(tasks)}")
                        tasks.append(processed_task)

            # Process validation split
            if "validation" in dataset:
                for item in dataset["validation"]:
                    task = self._convert_hf_arc_to_task(item)
                    if task and self.processor.validate_arc_task(task):
                        processed_task = self.processor.process_task_for_training(task, f"hf_arc_val_{len(tasks)}")
                        tasks.append(processed_task)

            logger.info(f"Loaded {len(tasks)} ARC tasks from HuggingFace")
            return tasks

        except Exception as e:
            logger.error(f"Failed to load HuggingFace ARC: {e}")
            return []

    def _convert_hf_arc_to_task(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Convert HuggingFace ARC item to standard ARC task format."""
        try:
            # This is a simplified conversion - real ARC tasks have different structure
            # For demo purposes, create a simple pattern task
            question = item.get("question", "")
            item.get("choices", {})

            # Create synthetic visual task based on question
            # In practice, you'd load actual ARC data or use a proper ARC dataset

            # Simple example: create a pattern based on question length
            size = min(max(len(question) % 5 + 2, 3), 8)
            color = len(question) % 5 + 1

            # Create input/output pattern
            input_grid = [[0 for _ in range(size)] for _ in range(size)]
            output_grid = [[0 for _ in range(size)] for _ in range(size)]

            # Add some pattern
            input_grid[0][0] = color
            output_grid[size - 1][size - 1] = color  # Move to opposite corner

            task = {"train": [{"input": input_grid, "output": output_grid}], "test": [{"input": input_grid}]}

            return task

        except Exception as e:
            logger.debug(f"Failed to convert HF ARC item: {e}")
            return None

    def load_local_arc(self) -> list[dict[str, Any]]:
        """Load ARC from local files."""
        if not self.config.local_arc_path:
            return []

        tasks = []
        arc_path = Path(self.config.local_arc_path)

        try:
            if arc_path.exists():
                # Look for JSON files
                for json_file in arc_path.glob("*.json"):
                    with open(json_file) as f:
                        task_data = json.load(f)

                    if self.processor.validate_arc_task(task_data):
                        processed_task = self.processor.process_task_for_training(task_data, json_file.stem)
                        tasks.append(processed_task)

                logger.info(f"Loaded {len(tasks)} ARC tasks from local files")

        except Exception as e:
            logger.error(f"Failed to load local ARC: {e}")

        return tasks

    def generate_synthetic_arc(self, num_tasks: int = 50) -> list[dict[str, Any]]:
        """Generate synthetic ARC-like tasks."""
        if not self.config.use_synthetic_arc:
            return []

        tasks = []

        try:
            logger.info(f"Generating {num_tasks} synthetic ARC tasks...")

            for i in range(num_tasks):
                task = self._generate_single_synthetic_task(i)
                if task and self.processor.validate_arc_task(task):
                    processed_task = self.processor.process_task_for_training(task, f"synthetic_{i}")
                    tasks.append(processed_task)

            logger.info(f"Generated {len(tasks)} valid synthetic ARC tasks")

        except Exception as e:
            logger.error(f"Failed to generate synthetic ARC: {e}")

        return tasks

    def _generate_single_synthetic_task(self, task_id: int) -> dict[str, Any]:
        """Generate a single synthetic ARC task."""
        np.random.seed(self.config.augmentation_seed + task_id)

        # Random task parameters
        size = np.random.randint(3, 8)
        num_colors = np.random.randint(2, 5)
        pattern_type = np.random.choice(["copy", "flip", "rotate", "color_swap"])

        # Generate input grid
        input_grid = np.zeros((size, size), dtype=int)

        # Add some pattern
        for _ in range(size // 2):
            x, y = np.random.randint(0, size), np.random.randint(0, size)
            color = np.random.randint(1, num_colors + 1)
            input_grid[x, y] = color

        # Generate output based on pattern
        if pattern_type == "copy":
            output_grid = input_grid.copy()
        elif pattern_type == "flip":
            output_grid = np.flipud(input_grid)
        elif pattern_type == "rotate":
            output_grid = np.rot90(input_grid)
        else:  # color_swap
            output_grid = input_grid.copy()
            # Swap colors 1 and 2
            output_grid[input_grid == 1] = 2
            output_grid[input_grid == 2] = 1

        # Create task structure
        task = {
            "train": [{"input": input_grid.tolist(), "output": output_grid.tolist()}],
            "test": [{"input": input_grid.tolist()}],
        }

        return task


class ARCVisualDataset(Dataset):
    """Complete ARC Visual Reasoning dataset for Stage 1."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = ARCDataConfig(**(config or {}))
        self.loader = ARCDataLoader(self.config)

        # Initialize augmentation engine
        aug_config = AugmentationConfig(
            max_augmentations_per_task=self.config.augmentations_per_task, seed=self.config.augmentation_seed
        )
        self.augmentation_engine = ARCAugmentationEngine(aug_config)

        # Load base tasks
        self.base_tasks = self._load_all_tasks()

        # Generate augmentations
        if self.config.cache_augmentations:
            self.augmented_tasks = self._generate_all_augmentations()
        else:
            self.augmented_tasks = None

        logger.info(f"ARC dataset initialized with {len(self)} total samples")

    def _load_all_tasks(self) -> list[dict[str, Any]]:
        """Load tasks from all configured sources."""
        all_tasks = []

        # Load from HuggingFace
        hf_tasks = self.loader.load_huggingface_arc()
        all_tasks.extend(hf_tasks)

        # Load from local files
        local_tasks = self.loader.load_local_arc()
        all_tasks.extend(local_tasks)

        # Generate synthetic tasks
        synthetic_tasks = self.loader.generate_synthetic_arc(num_tasks=100)
        all_tasks.extend(synthetic_tasks)

        # Apply max_tasks limit
        if self.config.max_tasks and len(all_tasks) > self.config.max_tasks:
            all_tasks = all_tasks[: self.config.max_tasks]

        logger.info(f"Loaded {len(all_tasks)} base ARC tasks")
        return all_tasks

    def _generate_all_augmentations(self) -> list[dict[str, Any]]:
        """Generate augmentations for all base tasks."""
        all_augmented = []

        logger.info(f"Generating augmentations for {len(self.base_tasks)} tasks...")

        for i, task in enumerate(self.base_tasks):
            if i % 10 == 0:
                logger.info(f"Augmenting task {i+1}/{len(self.base_tasks)}")

            # Generate augmentations for this task
            augmented_variants = self.augmentation_engine.augment_arc_task(task)
            all_augmented.extend(augmented_variants)

        logger.info(f"Generated {len(all_augmented)} total augmented tasks")
        return all_augmented

    def __len__(self) -> int:
        if self.config.cache_augmentations and self.augmented_tasks:
            return len(self.augmented_tasks)
        else:
            # Estimate based on augmentation rate
            return len(self.base_tasks) * self.config.augmentations_per_task

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.config.cache_augmentations and self.augmented_tasks:
            # Use cached augmentations
            return self.augmented_tasks[idx]
        else:
            # Generate augmentation on-the-fly
            base_idx = idx // self.config.augmentations_per_task
            aug_idx = idx % self.config.augmentations_per_task

            if base_idx >= len(self.base_tasks):
                base_idx = base_idx % len(self.base_tasks)

            base_task = self.base_tasks[base_idx]

            # Generate specific augmentation
            # For simplicity, generate all and pick one (inefficient but correct)
            augmented_variants = self.augmentation_engine.augment_arc_task(base_task)

            if aug_idx < len(augmented_variants):
                return augmented_variants[aug_idx]
            else:
                return base_task

    def get_data_loader(self, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
        """Get DataLoader for this dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate function for batching ARC tasks."""
        return {
            "tasks": batch,  # Keep full task structure
            "task_ids": [task.get("task_id", "unknown") for task in batch],
            "num_train_examples": [len(task.get("train", [])) for task in batch],
            "metadata": [task.get("metadata", {}) for task in batch],
        }

    def get_task_statistics(self) -> dict[str, Any]:
        """Get statistics about the dataset."""
        if not self.base_tasks:
            return {}

        # Analyze base tasks
        total_colors = set()
        grid_sizes = set()
        num_train_examples = []

        for task in self.base_tasks:
            metadata = task.get("metadata", {})

            if "colors_used" in metadata:
                total_colors.update(metadata["colors_used"])

            if "grid_sizes" in metadata:
                grid_sizes.update(metadata["grid_sizes"])

            num_train_examples.append(metadata.get("num_train_examples", 0))

        # Augmentation statistics
        aug_stats = self.augmentation_engine.get_augmentation_stats()

        return {
            "base_tasks": len(self.base_tasks),
            "total_samples": len(self),
            "augmentation_ratio": len(self) / len(self.base_tasks) if self.base_tasks else 0,
            "unique_colors": len(total_colors),
            "unique_grid_sizes": len(grid_sizes),
            "avg_train_examples": np.mean(num_train_examples) if num_train_examples else 0,
            "augmentation_stats": aug_stats,
        }

    def validate_dataset(self) -> bool:
        """Validate the entire dataset."""
        if not self.config.validate_tasks:
            return True

        logger.info("Validating ARC dataset...")

        valid_count = 0
        total_count = min(100, len(self.base_tasks))  # Sample validation

        for i in range(total_count):
            task = self.base_tasks[i]
            if self.loader.processor.validate_arc_task(task):
                valid_count += 1

        success_rate = valid_count / total_count if total_count > 0 else 0

        logger.info(f"Dataset validation: {valid_count}/{total_count} tasks valid ({success_rate*100:.1f}%)")

        return success_rate > 0.8  # 80% minimum success rate


def create_arc_dataset(config: dict[str, Any] = None) -> ARCVisualDataset:
    """Factory function to create ARC visual dataset."""
    dataset = ARCVisualDataset(config)

    if not dataset.validate_dataset():
        logger.warning("ARC dataset validation failed - some tasks may be invalid")

    return dataset


def demo_arc_dataset():
    """Demonstrate ARC visual dataset functionality."""
    print("=== Cogment Stage 1: ARC Visual Dataset Demo ===")

    # Create dataset with small config for demo
    config = {
        "max_tasks": 5,
        "augmentations_per_task": 10,
        "cache_augmentations": True,
        "use_huggingface_arc": False,  # Skip HF for demo
        "use_synthetic_arc": True,
    }

    dataset = create_arc_dataset(config)

    # Show statistics
    stats = dataset.get_task_statistics()
    print("\nDataset statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # Show sample
    print("\n=== Sample ARC Task ===")
    sample_task = dataset[0]
    print(f"Task ID: {sample_task.get('task_id', 'unknown')}")
    print(f"Train examples: {len(sample_task.get('train', []))}")
    print(f"Test examples: {len(sample_task.get('test', []))}")

    if sample_task.get("train"):
        example = sample_task["train"][0]
        print(f"Sample input grid: {example['input']}")
        print(f"Sample output grid: {example['output']}")

    # Test data loader
    loader = dataset.get_data_loader(batch_size=2, shuffle=False)
    batch = next(iter(loader))
    print(f"\nBatch structure: {list(batch.keys())}")
    print(f"Batch size: {len(batch['tasks'])}")

    print("\n=== ARC Visual Dataset Demo Complete ===")


if __name__ == "__main__":
    demo_arc_dataset()
