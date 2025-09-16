"""
Stage 2: Algorithmic Puzzle Dataset

Structured algorithmic puzzles for logical reasoning:
- Sudoku: 9x9 puzzles with varying difficulty (easy, medium, hard)
- Mazes: Grid-based pathfinding (8x8, 16x16, 32x32)
- ListOps: Nested list operations from LRA benchmark
- Graph algorithms: Basic traversal and search problems

Purpose: Train structured reasoning and search discipline for ACT refinement.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import random
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class PuzzleType(Enum):
    SUDOKU = "sudoku"
    MAZE = "maze"
    LISTOPS = "listops"
    GRAPH = "graph"


class DifficultyLevel(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


@dataclass
class PuzzleConfig:
    """Configuration for puzzle generation."""

    puzzle_type: PuzzleType
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    num_samples: int = 100
    seed: int = 42

    # Sudoku-specific
    sudoku_givens_range: dict[DifficultyLevel, tuple[int, int]] = None

    # Maze-specific
    maze_sizes: dict[DifficultyLevel, list[tuple[int, int]]] = None

    # ListOps-specific
    listops_depth_range: dict[DifficultyLevel, tuple[int, int]] = None

    def __post_init__(self):
        if self.sudoku_givens_range is None:
            self.sudoku_givens_range = {
                DifficultyLevel.EASY: (40, 50),
                DifficultyLevel.MEDIUM: (30, 40),
                DifficultyLevel.HARD: (25, 30),
            }

        if self.maze_sizes is None:
            self.maze_sizes = {
                DifficultyLevel.EASY: [(8, 8), (10, 10)],
                DifficultyLevel.MEDIUM: [(16, 16), (20, 20)],
                DifficultyLevel.HARD: [(32, 32), (40, 40)],
            }

        if self.listops_depth_range is None:
            self.listops_depth_range = {
                DifficultyLevel.EASY: (2, 4),
                DifficultyLevel.MEDIUM: (4, 6),
                DifficultyLevel.HARD: (6, 8),
            }


class SudokuGenerator:
    """Generate Sudoku puzzles with varying difficulty."""

    def __init__(self, config: PuzzleConfig):
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)

    def generate_samples(self) -> list[dict[str, Any]]:
        """Generate Sudoku puzzle samples."""
        samples = []

        for i in range(self.config.num_samples):
            puzzle, solution = self._generate_sudoku_puzzle()

            if puzzle is not None and solution is not None:
                sample = {
                    "input": self._format_sudoku_input(puzzle),
                    "target": self._format_sudoku_solution(solution),
                    "task_type": "sudoku",
                    "difficulty": self.config.difficulty.name.lower(),
                    "metadata": {
                        "puzzle_id": i,
                        "givens": np.count_nonzero(puzzle),
                        "difficulty": self.config.difficulty.name,
                    },
                }
                samples.append(sample)

        logger.info(f"Generated {len(samples)} Sudoku puzzles ({self.config.difficulty.name})")
        return samples

    def _generate_sudoku_puzzle(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a single Sudoku puzzle and solution."""
        # Start with a complete valid solution
        solution = self._generate_complete_sudoku()

        if solution is None:
            return None, None

        # Create puzzle by removing numbers
        puzzle = solution.copy()

        # Determine number of givens based on difficulty
        min_givens, max_givens = self.config.sudoku_givens_range[self.config.difficulty]
        target_givens = random.randint(min_givens, max_givens)

        # Remove numbers while maintaining unique solution
        positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(positions)

        removed = 0
        max_to_remove = 81 - target_givens

        for i, j in positions:
            if removed >= max_to_remove:
                break

            # Temporarily remove the number
            original = puzzle[i, j]
            puzzle[i, j] = 0

            # Check if puzzle still has unique solution (simplified check)
            if self._count_solutions(puzzle) == 1:
                removed += 1
            else:
                # Restore the number
                puzzle[i, j] = original

        return puzzle, solution

    def _generate_complete_sudoku(self) -> np.ndarray | None:
        """Generate a complete valid Sudoku solution."""
        grid = np.zeros((9, 9), dtype=int)

        # Fill the grid using backtracking
        if self._solve_sudoku(grid):
            return grid

        return None

    def _solve_sudoku(self, grid: np.ndarray) -> bool:
        """Solve Sudoku using backtracking."""
        empty = self._find_empty_cell(grid)
        if empty is None:
            return True  # No empty cells, solved

        row, col = empty

        # Try numbers 1-9
        numbers = list(range(1, 10))
        random.shuffle(numbers)  # Randomize for variety

        for num in numbers:
            if self._is_valid_move(grid, row, col, num):
                grid[row, col] = num

                if self._solve_sudoku(grid):
                    return True

                # Backtrack
                grid[row, col] = 0

        return False

    def _find_empty_cell(self, grid: np.ndarray) -> tuple[int, int] | None:
        """Find an empty cell in the grid."""
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    return (i, j)
        return None

    def _is_valid_move(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid."""
        # Check row
        if num in grid[row, :]:
            return False

        # Check column
        if num in grid[:, col]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_row : box_row + 3, box_col : box_col + 3]:
            return False

        return True

    def _count_solutions(self, grid: np.ndarray) -> int:
        """Count number of solutions (simplified - just check if solvable)."""
        # For efficiency, just check if there's at least one solution
        test_grid = grid.copy()
        return 1 if self._solve_sudoku(test_grid) else 0

    def _format_sudoku_input(self, puzzle: np.ndarray) -> str:
        """Format Sudoku puzzle as input string."""
        lines = []
        lines.append("Solve this Sudoku puzzle (0 = empty):")

        for i in range(9):
            if i % 3 == 0 and i > 0:
                lines.append("------+-------+------")

            row_str = ""
            for j in range(9):
                if j % 3 == 0 and j > 0:
                    row_str += "| "
                row_str += str(puzzle[i, j]) + " "

            lines.append(row_str.rstrip())

        return "\n".join(lines)

    def _format_sudoku_solution(self, solution: np.ndarray) -> str:
        """Format Sudoku solution as target string."""
        lines = ["Solution:"]

        for i in range(9):
            if i % 3 == 0 and i > 0:
                lines.append("------+-------+------")

            row_str = ""
            for j in range(9):
                if j % 3 == 0 and j > 0:
                    row_str += "| "
                row_str += str(solution[i, j]) + " "

            lines.append(row_str.rstrip())

        return "\n".join(lines)


class MazeGenerator:
    """Generate maze puzzles for pathfinding."""

    def __init__(self, config: PuzzleConfig):
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)

    def generate_samples(self) -> list[dict[str, Any]]:
        """Generate maze samples."""
        samples = []
        available_sizes = self.config.maze_sizes[self.config.difficulty]

        for i in range(self.config.num_samples):
            # Choose random size for this difficulty
            height, width = random.choice(available_sizes)

            maze, path = self._generate_maze_with_solution(height, width)

            if maze is not None and path is not None:
                sample = {
                    "input": self._format_maze_input(maze),
                    "target": self._format_maze_solution(path),
                    "task_type": "maze",
                    "difficulty": self.config.difficulty.name.lower(),
                    "metadata": {
                        "maze_id": i,
                        "size": (height, width),
                        "path_length": len(path),
                        "difficulty": self.config.difficulty.name,
                    },
                }
                samples.append(sample)

        logger.info(f"Generated {len(samples)} maze puzzles ({self.config.difficulty.name})")
        return samples

    def _generate_maze_with_solution(self, height: int, width: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Generate maze and find optimal path."""
        # Generate maze using recursive backtracking
        maze = np.ones((height, width), dtype=int)  # 1 = wall, 0 = path

        # Create maze
        self._carve_maze(maze, 1, 1)

        # Ensure start and end are open
        maze[0, 0] = 0  # Start
        maze[height - 1, width - 1] = 0  # End

        # Find shortest path
        path = self._find_shortest_path(maze, (0, 0), (height - 1, width - 1))

        if path is None:
            # If no path found, create a simple path
            path = self._create_simple_path(maze, (0, 0), (height - 1, width - 1))

        return maze, path

    def _carve_maze(self, maze: np.ndarray, x: int, y: int):
        """Carve maze paths using recursive backtracking."""
        maze[x, y] = 0  # Mark as path

        # Randomize directions
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check bounds
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]:
                if maze[nx, ny] == 1:  # Unvisited
                    # Carve path between current and next
                    maze[x + dx // 2, y + dy // 2] = 0
                    self._carve_maze(maze, nx, ny)

    def _find_shortest_path(
        self, maze: np.ndarray, start: tuple[int, int], end: tuple[int, int]
    ) -> list[tuple[int, int]] | None:
        """Find shortest path using BFS."""
        from collections import deque

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()

            if (x, y) == end:
                return path

            # Check 4 directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if (
                    0 <= nx < maze.shape[0]
                    and 0 <= ny < maze.shape[1]
                    and (nx, ny) not in visited
                    and maze[nx, ny] == 0
                ):

                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

        return None

    def _create_simple_path(
        self, maze: np.ndarray, start: tuple[int, int], end: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Create a simple path when maze generation fails."""
        # Clear a simple path
        sx, sy = start
        ex, ey = end

        path = [start]
        x, y = sx, sy

        # Move right first
        while y < ey:
            y += 1
            maze[x, y] = 0
            path.append((x, y))

        # Then move down
        while x < ex:
            x += 1
            maze[x, y] = 0
            path.append((x, y))

        return path

    def _format_maze_input(self, maze: np.ndarray) -> str:
        """Format maze as input string."""
        lines = ["Find path from S (top-left) to E (bottom-right):"]
        lines.append("# = wall, . = path")
        lines.append("")

        for i, row in enumerate(maze):
            line = ""
            for j, cell in enumerate(row):
                if i == 0 and j == 0:
                    line += "S"
                elif i == maze.shape[0] - 1 and j == maze.shape[1] - 1:
                    line += "E"
                elif cell == 1:
                    line += "#"
                else:
                    line += "."
            lines.append(line)

        return "\n".join(lines)

    def _format_maze_solution(self, path: list[tuple[int, int]]) -> str:
        """Format maze solution as target string."""
        moves = []

        for i in range(1, len(path)):
            prev_x, prev_y = path[i - 1]
            curr_x, curr_y = path[i]

            if curr_x > prev_x:
                moves.append("down")
            elif curr_x < prev_x:
                moves.append("up")
            elif curr_y > prev_y:
                moves.append("right")
            else:
                moves.append("left")

        return f"Path: {' -> '.join(moves)} ({len(path)} steps)"


class ListOpsGenerator:
    """Generate ListOps tasks from LRA benchmark."""

    def __init__(self, config: PuzzleConfig):
        self.config = config
        random.seed(config.seed)

        self.operations = ["MAX", "MIN", "MEDIAN", "SUM_MOD", "COUNT"]

    def generate_samples(self) -> list[dict[str, Any]]:
        """Generate ListOps samples."""
        samples = []

        min_depth, max_depth = self.config.listops_depth_range[self.config.difficulty]

        for i in range(self.config.num_samples):
            depth = random.randint(min_depth, max_depth)
            expression, result = self._generate_listops_expression(depth)

            if expression and result is not None:
                sample = {
                    "input": f"Evaluate: {expression}",
                    "target": f"Result: {result}",
                    "task_type": "listops",
                    "difficulty": self.config.difficulty.name.lower(),
                    "metadata": {
                        "expr_id": i,
                        "depth": depth,
                        "operations_used": self._count_operations(expression),
                        "difficulty": self.config.difficulty.name,
                    },
                }
                samples.append(sample)

        logger.info(f"Generated {len(samples)} ListOps expressions ({self.config.difficulty.name})")
        return samples

    def _generate_listops_expression(self, max_depth: int) -> tuple[str, int | None]:
        """Generate a nested ListOps expression."""
        expression = self._build_expression(max_depth)

        try:
            result = self._evaluate_expression(expression)
            return expression, result
        except Exception as e:
            logger.debug(f"Failed to evaluate expression {expression}: {e}")
            return None, None

    def _build_expression(self, depth: int) -> str:
        """Build nested expression recursively."""
        if depth <= 0:
            # Base case: return a number
            return str(random.randint(0, 9))

        # Choose operation
        operation = random.choice(self.operations)

        # Generate arguments
        num_args = random.randint(2, 4)
        args = []

        for _ in range(num_args):
            if random.random() < 0.6 and depth > 1:
                # Recursive case
                args.append(self._build_expression(depth - 1))
            else:
                # Base case
                args.append(str(random.randint(0, 9)))

        return f"( {operation} {' '.join(args)} )"

    def _evaluate_expression(self, expr: str) -> int:
        """Evaluate ListOps expression."""
        # Simple recursive evaluator
        expr = expr.strip()

        if not expr.startswith("("):
            # Base case: number
            return int(expr)

        # Parse operation and arguments
        expr = expr[1:-1].strip()  # Remove outer parentheses
        parts = expr.split()

        operation = parts[0]
        args = []

        i = 1
        while i < len(parts):
            if parts[i] == "(":
                # Find matching closing parenthesis
                paren_count = 1
                j = i + 1
                while j < len(parts) and paren_count > 0:
                    if parts[j] == "(":
                        paren_count += 1
                    elif parts[j] == ")":
                        paren_count -= 1
                    j += 1

                # Reconstruct nested expression
                nested_expr = " ".join(parts[i:j])
                args.append(self._evaluate_expression(nested_expr))
                i = j
            else:
                # Simple number
                args.append(int(parts[i]))
                i += 1

        # Apply operation
        if operation == "MAX":
            return max(args)
        elif operation == "MIN":
            return min(args)
        elif operation == "MEDIAN":
            sorted_args = sorted(args)
            n = len(sorted_args)
            if n % 2 == 0:
                return (sorted_args[n // 2 - 1] + sorted_args[n // 2]) // 2
            else:
                return sorted_args[n // 2]
        elif operation == "SUM_MOD":
            return sum(args) % 10
        elif operation == "COUNT":
            return len(args)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _count_operations(self, expr: str) -> int:
        """Count number of operations in expression."""
        return sum(1 for op in self.operations if op in expr)


class AlgorithmicPuzzleDataset(Dataset):
    """Complete algorithmic puzzle dataset for Stage 2."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.samples_per_type = self.config.get("samples_per_type", 50)
        self.difficulties = self.config.get("difficulties", [DifficultyLevel.EASY, DifficultyLevel.MEDIUM])
        self.seed = self.config.get("seed", 42)

        # Generate all samples
        self.samples = []
        self._generate_all_samples()

        # Shuffle for variety
        random.seed(self.seed)
        random.shuffle(self.samples)

        logger.info(f"Algorithmic puzzle dataset initialized with {len(self.samples)} samples")

    def _generate_all_samples(self):
        """Generate samples for all puzzle types and difficulties."""

        for difficulty in self.difficulties:
            for puzzle_type in PuzzleType:
                config = PuzzleConfig(
                    puzzle_type=puzzle_type, difficulty=difficulty, num_samples=self.samples_per_type, seed=self.seed
                )

                try:
                    if puzzle_type == PuzzleType.SUDOKU:
                        generator = SudokuGenerator(config)
                    elif puzzle_type == PuzzleType.MAZE:
                        generator = MazeGenerator(config)
                    elif puzzle_type == PuzzleType.LISTOPS:
                        generator = ListOpsGenerator(config)
                    else:
                        continue  # Skip unsupported types

                    samples = generator.generate_samples()
                    self.samples.extend(samples)

                except Exception as e:
                    logger.error(f"Failed to generate {puzzle_type.value} samples: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

    def get_data_loader(self, batch_size: int = 6, shuffle: bool = True) -> DataLoader:
        """Get DataLoader for this dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate function for batching."""
        return {
            "inputs": [item["input"] for item in batch],
            "targets": [item["target"] for item in batch],
            "task_types": [item["task_type"] for item in batch],
            "difficulties": [item["difficulty"] for item in batch],
            "metadata": [item["metadata"] for item in batch],
        }

    def get_task_distribution(self) -> dict[str, dict[str, int]]:
        """Get distribution of tasks by type and difficulty."""
        distribution = {}

        for sample in self.samples:
            task_type = sample["task_type"]
            difficulty = sample["difficulty"]

            if task_type not in distribution:
                distribution[task_type] = {}

            if difficulty not in distribution[task_type]:
                distribution[task_type][difficulty] = 0

            distribution[task_type][difficulty] += 1

        return distribution

    def validate_samples(self) -> bool:
        """Validate dataset samples."""
        required_keys = {"input", "target", "task_type", "difficulty", "metadata"}

        for i, sample in enumerate(self.samples):
            if not all(key in sample for key in required_keys):
                logger.error(f"Sample {i} missing required keys")
                return False

        logger.info("All algorithmic puzzle samples validated successfully")
        return True


def create_puzzle_dataset(config: dict[str, Any] = None) -> AlgorithmicPuzzleDataset:
    """Factory function to create algorithmic puzzle dataset."""
    dataset = AlgorithmicPuzzleDataset(config)

    if not dataset.validate_samples():
        raise ValueError("Algorithmic puzzle dataset validation failed")

    return dataset


def demo_puzzle_dataset():
    """Demonstrate algorithmic puzzle dataset functionality."""
    print("=== Cogment Stage 2: Algorithmic Puzzle Dataset Demo ===")

    # Create dataset with small config for demo
    config = {"samples_per_type": 3, "difficulties": [DifficultyLevel.EASY], "seed": 42}  # Small demo

    dataset = create_puzzle_dataset(config)

    print(f"\nDataset size: {len(dataset)}")

    # Show task distribution
    distribution = dataset.get_task_distribution()
    print("\nTask distribution:")
    for task_type, diff_dist in distribution.items():
        print(f"  {task_type}:")
        for difficulty, count in diff_dist.items():
            print(f"    {difficulty}: {count}")

    # Show sample examples
    print("\n=== Sample Examples ===")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1} ({sample['task_type']} - {sample['difficulty']}):")
        print(f"Input: {sample['input'][:200]}...")
        print(f"Target: {sample['target'][:100]}...")

    # Test data loader
    loader = dataset.get_data_loader(batch_size=2, shuffle=False)
    batch = next(iter(loader))
    print(f"\nBatch structure: {list(batch.keys())}")
    print(f"Batch size: {len(batch['inputs'])}")

    print("\n=== Algorithmic Puzzle Dataset Demo Complete ===")


if __name__ == "__main__":
    demo_puzzle_dataset()
