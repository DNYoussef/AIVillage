"""
Stage 0: Sanity Check Dataset

Synthetic tasks for basic functionality validation:
- Linear function learning: y = Ax + b  
- Simple pattern recognition: sequence completion
- Toy mazes: shortest path finding
- Memory tests: recall sequences

Purpose: Verify ACT, GrokFast, loss functions work before complex training.
"""

from dataclasses import dataclass
import logging
import random
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class SanityTaskConfig:
    """Configuration for sanity check tasks."""

    task_type: str
    num_samples: int = 100
    sequence_length: int = 32
    difficulty_level: int = 1  # 1=easy, 2=medium, 3=hard
    seed: int = 42


class LinearMapTask:
    """Linear function learning: y = Ax + b"""

    def __init__(self, config: SanityTaskConfig):
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)

        # Generate random linear coefficients
        self.A = random.uniform(-5, 5)
        self.b = random.uniform(-10, 10)

        logger.info(f"Linear task: y = {self.A:.2f}x + {self.b:.2f}")

    def generate_samples(self) -> list[dict[str, Any]]:
        """Generate linear mapping samples."""
        samples = []

        for _ in range(self.config.num_samples):
            # Random input in reasonable range
            x = random.uniform(-10, 10)
            y = self.A * x + self.b

            # Format as simple calculation task
            input_text = f"Calculate y = {self.A:.2f} * {x:.2f} + {self.b:.2f}"
            target_text = f"y = {y:.3f}"

            samples.append(
                {
                    "input": input_text,
                    "target": target_text,
                    "task_type": "linear_map",
                    "metadata": {"A": self.A, "b": self.b, "x": x, "y": y},
                }
            )

        return samples


class SequenceCompletionTask:
    """Simple pattern recognition: sequence completion"""

    def __init__(self, config: SanityTaskConfig):
        self.config = config
        random.seed(config.seed)

    def generate_samples(self) -> list[dict[str, Any]]:
        """Generate sequence completion samples."""
        samples = []

        for _ in range(self.config.num_samples):
            pattern_type = random.choice(["arithmetic", "geometric", "fibonacci"])

            if pattern_type == "arithmetic":
                # Arithmetic progression: a, a+d, a+2d, ...
                start = random.randint(-10, 10)
                diff = random.randint(1, 5)
                sequence = [start + i * diff for i in range(5)]
                next_val = sequence[-1] + diff

            elif pattern_type == "geometric":
                # Geometric progression: a, ar, ar^2, ...
                start = random.randint(1, 5)
                ratio = random.choice([2, 3])
                sequence = [start * (ratio**i) for i in range(4)]
                next_val = sequence[-1] * ratio

            else:  # fibonacci
                # Fibonacci-like: a, b, a+b, a+2b, 2a+3b, ...
                a, b = random.randint(1, 3), random.randint(1, 3)
                sequence = [a, b]
                for _ in range(3):
                    sequence.append(sequence[-1] + sequence[-2])
                next_val = sequence[-1] + sequence[-2]

            input_text = f"Complete the sequence: {', '.join(map(str, sequence))}, ?"
            target_text = f"Next number: {next_val}"

            samples.append(
                {
                    "input": input_text,
                    "target": target_text,
                    "task_type": "sequence_completion",
                    "metadata": {"pattern": pattern_type, "sequence": sequence, "next": next_val},
                }
            )

        return samples


class ToyMazeTask:
    """Toy mazes: shortest path finding"""

    def __init__(self, config: SanityTaskConfig):
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)

        # Small maze sizes for sanity checks
        self.maze_sizes = [(3, 3), (4, 4), (5, 5)]

    def generate_samples(self) -> list[dict[str, Any]]:
        """Generate toy maze navigation samples."""
        samples = []

        for _ in range(self.config.num_samples):
            # Choose random maze size
            rows, cols = random.choice(self.maze_sizes)

            # Simple maze generation (ensure solvable)
            maze = np.zeros((rows, cols), dtype=int)

            # Add some walls (but keep it solvable)
            num_walls = max(1, (rows * cols) // 4)
            for _ in range(num_walls):
                r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
                if (r, c) != (0, 0) and (r, c) != (rows - 1, cols - 1):
                    maze[r, c] = 1  # Wall

            # Simple path finding (right-down bias for simplicity)
            (rows - 1, cols - 1)

            # Generate simple path description
            path_moves = []
            r, c = 0, 0
            while r < rows - 1 or c < cols - 1:
                if r < rows - 1 and (c >= cols - 1 or random.random() < 0.5):
                    path_moves.append("down")
                    r += 1
                else:
                    path_moves.append("right")
                    c += 1

            # Format maze as grid
            maze_str = "\n".join([" ".join(["." if cell == 0 else "#" for cell in row]) for row in maze])
            input_text = f"Find path from top-left to bottom-right:\n{maze_str}"
            target_text = f"Path: {' -> '.join(path_moves)}"

            samples.append(
                {
                    "input": input_text,
                    "target": target_text,
                    "task_type": "toy_maze",
                    "metadata": {"size": (rows, cols), "path": path_moves},
                }
            )

        return samples


class MemoryRecallTask:
    """Memory tests: recall sequences"""

    def __init__(self, config: SanityTaskConfig):
        self.config = config
        random.seed(config.seed)

    def generate_samples(self) -> list[dict[str, Any]]:
        """Generate memory recall samples."""
        samples = []

        for _ in range(self.config.num_samples):
            # Generate random sequence to remember
            seq_length = random.randint(3, 8)
            memory_type = random.choice(["numbers", "letters", "words"])

            if memory_type == "numbers":
                sequence = [random.randint(0, 9) for _ in range(seq_length)]
                seq_str = " ".join(map(str, sequence))
            elif memory_type == "letters":
                sequence = [chr(ord("A") + random.randint(0, 25)) for _ in range(seq_length)]
                seq_str = " ".join(sequence)
            else:  # words
                word_pool = ["cat", "dog", "car", "tree", "book", "sun", "moon", "star", "fish", "bird"]
                sequence = random.sample(word_pool, min(seq_length, len(word_pool)))
                seq_str = " ".join(sequence)

            # Add some distractor content
            distractor = "Now think about something else: The weather is nice today. What's 2+2? Four."

            input_text = f"Remember this sequence: {seq_str}\n{distractor}\nWhat was the sequence?"
            target_text = f"Sequence: {seq_str}"

            samples.append(
                {
                    "input": input_text,
                    "target": target_text,
                    "task_type": "memory_recall",
                    "metadata": {"type": memory_type, "length": seq_length, "sequence": sequence},
                }
            )

        return samples


class SanityCheckDataset(Dataset):
    """Complete sanity check dataset for Stage 0."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.samples_per_task = self.config.get("samples_per_task", 25)
        self.sequence_length = self.config.get("sequence_length", 128)
        self.seed = self.config.get("seed", 42)

        # Initialize task generators
        task_config = SanityTaskConfig(
            task_type="mixed", num_samples=self.samples_per_task, sequence_length=self.sequence_length, seed=self.seed
        )

        self.task_generators = {
            "linear_maps": LinearMapTask(task_config),
            "sequences": SequenceCompletionTask(task_config),
            "toy_mazes": ToyMazeTask(task_config),
            "memory": MemoryRecallTask(task_config),
        }

        # Generate all samples
        self.samples = []
        for task_name, generator in self.task_generators.items():
            task_samples = generator.generate_samples()
            self.samples.extend(task_samples)
            logger.info(f"Generated {len(task_samples)} {task_name} samples")

        # Shuffle for varied training
        random.seed(self.seed)
        random.shuffle(self.samples)

        logger.info(f"Total sanity check samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

    def get_data_loader(self, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
        """Get DataLoader for this dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate function for batching."""
        return {
            "inputs": [item["input"] for item in batch],
            "targets": [item["target"] for item in batch],
            "task_types": [item["task_type"] for item in batch],
            "metadata": [item["metadata"] for item in batch],
        }

    def get_task_distribution(self) -> dict[str, int]:
        """Get distribution of task types."""
        distribution = {}
        for sample in self.samples:
            task_type = sample["task_type"]
            distribution[task_type] = distribution.get(task_type, 0) + 1
        return distribution

    def validate_samples(self) -> bool:
        """Validate that all samples are properly formatted."""
        required_keys = {"input", "target", "task_type", "metadata"}

        for i, sample in enumerate(self.samples):
            if not all(key in sample for key in required_keys):
                logger.error(f"Sample {i} missing required keys: {required_keys - set(sample.keys())}")
                return False

            if not isinstance(sample["input"], str) or not isinstance(sample["target"], str):
                logger.error(f"Sample {i} input/target must be strings")
                return False

        logger.info("All sanity check samples validated successfully")
        return True


def create_sanity_dataset(config: dict[str, Any] = None) -> SanityCheckDataset:
    """Factory function to create sanity check dataset."""
    dataset = SanityCheckDataset(config)

    if not dataset.validate_samples():
        raise ValueError("Sanity check dataset validation failed")

    return dataset


def demo_sanity_checks():
    """Demonstrate sanity check dataset functionality."""
    print("=== Cogment Stage 0: Sanity Check Dataset Demo ===")

    # Create dataset
    config = {"samples_per_task": 5, "sequence_length": 64, "seed": 42}  # Small demo

    dataset = create_sanity_dataset(config)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Task distribution: {dataset.get_task_distribution()}")

    # Show sample examples
    print("\n=== Sample Examples ===")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1} ({sample['task_type']}):")
        print(f"Input: {sample['input'][:100]}...")
        print(f"Target: {sample['target']}")

    # Test data loader
    loader = dataset.get_data_loader(batch_size=2, shuffle=False)
    batch = next(iter(loader))
    print(f"\nBatch structure: {list(batch.keys())}")
    print(f"Batch size: {len(batch['inputs'])}")

    print("\n=== Sanity Check Demo Complete ===")


if __name__ == "__main__":
    demo_sanity_checks()
