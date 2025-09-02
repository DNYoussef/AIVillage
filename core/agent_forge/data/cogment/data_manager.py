"""
Cogment Data Manager

Central coordinator for the 4-stage Cogment curriculum data pipeline.
Replaces HRRM's limited synthetic approach with comprehensive real-world datasets.

Integrates with Agent 4's curriculum system to provide:
- Stage 0: Sanity checks (500 steps)
- Stage 1: ARC visual reasoning (~300 augmentations, 4K steps)
- Stage 2: Algorithmic puzzles (8K steps)
- Stage 3: Math & text reasoning (16K steps)
- Stage 4: Long-context (32K steps)

Designed for accelerated grokking through progressive curriculum training.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any

from torch.utils.data import DataLoader

# Import curriculum system from Agent 4
try:
    from ....models.cogment.training.curriculum import CurriculumStage, FourStageCurriculum, StageConfig
except ImportError:
    logger.warning("Could not import curriculum system - using standalone mode")
    CurriculumStage = None
    FourStageCurriculum = None
    StageConfig = None

# Import stage datasets
try:
    from .stage_0_sanity import create_sanity_dataset
    from .stage_1_arc import create_arc_dataset
    from .stage_2_puzzles import create_puzzle_dataset
    from .stage_3_reasoning import create_reasoning_dataset
    from .stage_4_longcontext import create_long_context_dataset
except ImportError:
    # Handle standalone execution
    try:
        from stage_0_sanity import create_sanity_dataset
        from stage_1_arc import create_arc_dataset
        from stage_2_puzzles import create_puzzle_dataset
        from stage_3_reasoning import create_reasoning_dataset
        from stage_4_longcontext import create_long_context_dataset
    except ImportError:
        logger.warning("Some dataset modules not available - limited functionality")

logger = logging.getLogger(__name__)


@dataclass
class StageDataConfig:
    """Configuration for a specific curriculum stage's data."""

    stage: int
    dataset_type: str
    config: dict[str, Any]
    batch_size: int
    sequence_length: int
    augmentation_enabled: bool = False
    cache_data: bool = True
    validate_quality: bool = True


class DataLoadingStrategy(Enum):
    """Strategies for loading curriculum data."""

    SEQUENTIAL = "sequential"  # Load stages one by one
    PRELOAD_ALL = "preload_all"  # Load all stages at initialization
    ON_DEMAND = "on_demand"  # Load only when requested
    HYBRID = "hybrid"  # Preload smaller stages, on-demand for larger


class CogmentDataManager:
    """
    Central data manager for Cogment 4-stage curriculum.

    Coordinates between Agent 4's curriculum system and comprehensive datasets,
    replacing HRRM's synthetic approach with real-world data for accelerated grokking.
    """

    def __init__(
        self,
        curriculum: FourStageCurriculum | None = None,
        loading_strategy: DataLoadingStrategy = DataLoadingStrategy.HYBRID,
        base_config: dict[str, Any] | None = None,
    ):

        self.curriculum = curriculum or self._create_default_curriculum()
        self.loading_strategy = loading_strategy
        self.base_config = base_config or {}

        # Stage dataset mapping
        self.stage_datasets: dict[int, Any] = {}
        self.stage_loaders: dict[int, DataLoader] = {}
        self.stage_configs: dict[int, StageDataConfig] = {}

        # Performance tracking
        self.load_times: dict[int, float] = {}
        self.dataset_stats: dict[int, dict[str, Any]] = {}

        # Initialize stage configurations
        self._initialize_stage_configs()

        # Load data based on strategy
        self._initialize_data_loading()

        logger.info(f"Cogment Data Manager initialized with {loading_strategy.value} strategy")

    def _create_default_curriculum(self) -> Any:
        """Create default curriculum if none provided."""
        if FourStageCurriculum is not None:
            return FourStageCurriculum()
        else:
            # Fallback standalone curriculum representation
            return {
                0: {"name": "Sanity Checks", "max_steps": 500, "batch_size": 16},
                1: {"name": "ARC Visual", "max_steps": 4000, "batch_size": 8},
                2: {"name": "Algorithmic", "max_steps": 8000, "batch_size": 6},
                3: {"name": "Math & Text", "max_steps": 16000, "batch_size": 4},
                4: {"name": "Long Context", "max_steps": 32000, "batch_size": 2},
            }

    def _initialize_stage_configs(self):
        """Initialize configurations for all curriculum stages."""

        # Stage 0: Sanity Checks
        self.stage_configs[0] = StageDataConfig(
            stage=0,
            dataset_type="sanity",
            config={"samples_per_task": 25, "sequence_length": 128, "seed": 42},
            batch_size=16,
            sequence_length=128,
            augmentation_enabled=False,
            cache_data=True,
            validate_quality=True,
        )

        # Stage 1: ARC Visual Reasoning
        self.stage_configs[1] = StageDataConfig(
            stage=1,
            dataset_type="arc",
            config={
                "max_tasks": 400,
                "augmentations_per_task": 300,
                "cache_augmentations": True,
                "use_synthetic_arc": True,
                "use_huggingface_arc": False,  # Use synthetic for stability
                "augmentation_seed": 42,
            },
            batch_size=8,
            sequence_length=256,
            augmentation_enabled=True,
            cache_data=True,
            validate_quality=True,
        )

        # Stage 2: Algorithmic Puzzles
        self.stage_configs[2] = StageDataConfig(
            stage=2,
            dataset_type="puzzles",
            config={"samples_per_type": 100, "difficulties": ["easy", "medium"], "seed": 42},
            batch_size=6,
            sequence_length=512,
            augmentation_enabled=True,
            cache_data=True,
            validate_quality=True,
        )

        # Stage 3: Math & Text Reasoning
        self.stage_configs[3] = StageDataConfig(
            stage=3,
            dataset_type="reasoning",
            config={
                "gsm8k_limit": 1000,
                "hotpotqa_limit": 500,
                "math_limit": 300,
                "chain_of_thought": True,
                "show_reasoning": True,
            },
            batch_size=4,
            sequence_length=1024,
            augmentation_enabled=False,
            cache_data=False,  # Large datasets - load on demand
            validate_quality=True,
        )

        # Stage 4: Long Context
        self.stage_configs[4] = StageDataConfig(
            stage=4,
            dataset_type="long_context",
            config={
                "longbench_limit": 200,
                "scrolls_limit": 150,
                "synthetic_limit": 100,
                "target_lengths": [1024, 2048, 4096],
                "min_context_length": 1024,
            },
            batch_size=2,
            sequence_length=2048,
            augmentation_enabled=False,
            cache_data=False,  # Very large - load on demand
            validate_quality=True,
        )

        logger.info(f"Initialized configurations for {len(self.stage_configs)} curriculum stages")

    def _initialize_data_loading(self):
        """Initialize data loading based on strategy."""
        if self.loading_strategy == DataLoadingStrategy.PRELOAD_ALL:
            self._preload_all_stages()
        elif self.loading_strategy == DataLoadingStrategy.HYBRID:
            # Preload smaller stages (0, 1), on-demand for larger (2, 3, 4)
            for stage in [0, 1]:
                self._load_stage_dataset(stage)
        # SEQUENTIAL and ON_DEMAND load as needed

    def _preload_all_stages(self):
        """Preload all stage datasets."""
        logger.info("Preloading all curriculum stage datasets...")

        for stage in range(5):
            self._load_stage_dataset(stage)

        logger.info("All stages preloaded successfully")

    def _load_stage_dataset(self, stage: int) -> Any:
        """Load dataset for a specific stage."""
        if stage in self.stage_datasets:
            return self.stage_datasets[stage]

        import time

        start_time = time.time()

        stage_config = self.stage_configs[stage]
        dataset_type = stage_config.dataset_type
        config = {**self.base_config, **stage_config.config}

        logger.info(f"Loading Stage {stage} dataset ({dataset_type})...")

        try:
            # Create appropriate dataset
            if dataset_type == "sanity":
                dataset = create_sanity_dataset(config)
            elif dataset_type == "arc":
                dataset = create_arc_dataset(config)
            elif dataset_type == "puzzles":
                dataset = create_puzzle_dataset(config)
            elif dataset_type == "reasoning":
                dataset = create_reasoning_dataset(config)
            elif dataset_type == "long_context":
                dataset = create_long_context_dataset(config)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            # Store dataset
            self.stage_datasets[stage] = dataset

            # Create data loader
            self.stage_loaders[stage] = dataset.get_data_loader(batch_size=stage_config.batch_size, shuffle=True)

            # Record performance
            load_time = time.time() - start_time
            self.load_times[stage] = load_time

            # Collect statistics
            self.dataset_stats[stage] = self._get_dataset_statistics(dataset, stage)

            logger.info(f"Stage {stage} loaded: {len(dataset)} samples in {load_time:.2f}s")

            return dataset

        except Exception as e:
            logger.error(f"Failed to load Stage {stage} dataset: {e}")
            raise

    def _get_dataset_statistics(self, dataset: Any, stage: int) -> dict[str, Any]:
        """Collect statistics about a dataset."""
        stats = {"stage": stage, "size": len(dataset), "dataset_type": self.stage_configs[stage].dataset_type}

        # Add dataset-specific statistics
        try:
            if hasattr(dataset, "get_task_distribution"):
                stats["task_distribution"] = dataset.get_task_distribution()

            if hasattr(dataset, "get_domain_distribution"):
                stats["domain_distribution"] = dataset.get_domain_distribution()

            if hasattr(dataset, "get_task_statistics"):
                stats["detailed_stats"] = dataset.get_task_statistics()

            if hasattr(dataset, "get_context_length_distribution"):
                stats["context_distribution"] = dataset.get_context_length_distribution()

        except Exception as e:
            logger.debug(f"Could not collect detailed stats for Stage {stage}: {e}")

        return stats

    def get_stage_loader(self, stage: int) -> DataLoader:
        """Get DataLoader for a specific curriculum stage."""
        # Ensure dataset is loaded
        if stage not in self.stage_datasets:
            self._load_stage_dataset(stage)

        return self.stage_loaders[stage]

    def get_stage_config(self, stage: int) -> StageDataConfig:
        """Get configuration for a specific stage."""
        if stage not in self.stage_configs:
            raise ValueError(f"Unknown stage: {stage}")
        return self.stage_configs[stage]

    def get_current_stage_loader(self) -> DataLoader:
        """Get DataLoader for the current curriculum stage."""
        if hasattr(self.curriculum, "current_stage"):
            # Get current stage from curriculum
            current_stage = self.curriculum.current_stage

            if hasattr(current_stage, "value"):
                stage_num = current_stage.value
            else:
                stage_num = int(current_stage)
        else:
            # Fallback to stage 0
            stage_num = 0

        return self.get_stage_loader(stage_num)

    def apply_augmentations(self, batch: dict[str, Any], stage: int) -> dict[str, Any]:
        """Apply stage-appropriate augmentations to a batch."""
        stage_config = self.stage_configs[stage]

        if not stage_config.augmentation_enabled:
            return batch

        # Currently only Stage 1 (ARC) has comprehensive augmentations
        if stage == 1:
            # ARC augmentations are handled within the dataset
            # This method could be extended for runtime augmentations
            return batch
        elif stage == 2:
            # Potential algorithmic puzzle augmentations
            return batch

        return batch

    def validate_data_quality(self, stage: int) -> bool:
        """Validate data quality for a specific stage."""
        if stage not in self.stage_datasets:
            self._load_stage_dataset(stage)

        dataset = self.stage_datasets[stage]
        stage_config = self.stage_configs[stage]

        if not stage_config.validate_quality:
            return True

        try:
            # Use dataset's validation method if available
            if hasattr(dataset, "validate_samples"):
                return dataset.validate_samples()
            elif hasattr(dataset, "validate_dataset"):
                return dataset.validate_dataset()
            elif hasattr(dataset, "validate_reasoning_quality"):
                return dataset.validate_reasoning_quality()
            elif hasattr(dataset, "validate_long_context_quality"):
                return dataset.validate_long_context_quality()
            else:
                # Basic validation
                return len(dataset) > 0

        except Exception as e:
            logger.error(f"Validation failed for Stage {stage}: {e}")
            return False

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about all loaded datasets."""
        stats = {
            "loading_strategy": self.loading_strategy.value,
            "stages_loaded": list(self.stage_datasets.keys()),
            "total_samples": sum(len(dataset) for dataset in self.stage_datasets.values()),
            "load_times": self.load_times.copy(),
            "stage_stats": self.dataset_stats.copy(),
        }

        # Memory usage estimation
        import sys

        total_memory = sum(sys.getsizeof(dataset) for dataset in self.stage_datasets.values())
        stats["estimated_memory_mb"] = total_memory / (1024 * 1024)

        return stats

    def optimize_for_stage(self, target_stage: int):
        """Optimize data loading for a specific target stage."""
        logger.info(f"Optimizing data manager for Stage {target_stage}")

        # Ensure target stage is loaded
        if target_stage not in self.stage_datasets:
            self._load_stage_dataset(target_stage)

        # If using on-demand strategy, may unload distant stages to save memory
        if self.loading_strategy == DataLoadingStrategy.ON_DEMAND:
            stages_to_keep = {target_stage}

            # Keep adjacent stages
            if target_stage > 0:
                stages_to_keep.add(target_stage - 1)
            if target_stage < 4:
                stages_to_keep.add(target_stage + 1)

            # Unload distant stages
            for stage in list(self.stage_datasets.keys()):
                if stage not in stages_to_keep:
                    self._unload_stage(stage)

    def _unload_stage(self, stage: int):
        """Unload a stage dataset to free memory."""
        if stage in self.stage_datasets:
            logger.info(f"Unloading Stage {stage} dataset to free memory")
            del self.stage_datasets[stage]
            if stage in self.stage_loaders:
                del self.stage_loaders[stage]

    def get_training_schedule_integration(self) -> dict[str, Any]:
        """Get integration info for Agent 4's training schedule."""
        return {
            "data_manager_ready": True,
            "stages_available": list(self.stage_configs.keys()),
            "loading_strategy": self.loading_strategy.value,
            "batch_sizes": {stage: config.batch_size for stage, config in self.stage_configs.items()},
            "sequence_lengths": {stage: config.sequence_length for stage, config in self.stage_configs.items()},
            "augmentation_stages": [
                stage for stage, config in self.stage_configs.items() if config.augmentation_enabled
            ],
            "validation_enabled": all(config.validate_quality for config in self.stage_configs.values()),
        }

    def demonstrate_full_pipeline(self):
        """Demonstrate the complete 4-stage data pipeline."""
        print("=== Cogment Data Manager: Full Pipeline Demonstration ===")

        for stage in range(5):
            print(f"\n--- Stage {stage}: {self.stage_configs[stage].dataset_type.title()} ---")

            try:
                # Load stage
                self._load_stage_dataset(stage)
                loader = self.get_stage_loader(stage)

                # Show statistics
                stats = self.dataset_stats[stage]
                print(f"Dataset size: {stats['size']}")
                print(f"Load time: {self.load_times[stage]:.2f}s")

                if "task_distribution" in stats:
                    print(f"Task distribution: {stats['task_distribution']}")

                # Validate quality
                is_valid = self.validate_data_quality(stage)
                print(f"Quality validation: {'✓ PASS' if is_valid else '✗ FAIL'}")

                # Show sample batch
                try:
                    batch = next(iter(loader))
                    print(f"Sample batch size: {len(batch.get('inputs', batch.get('tasks', [])))}")
                except Exception as e:
                    print(f"Could not load sample batch: {e}")

            except Exception as e:
                print(f"Failed to load Stage {stage}: {e}")

        # Show comprehensive stats
        print("\n=== Comprehensive Statistics ===")
        comp_stats = self.get_comprehensive_stats()

        print(f"Total samples across all stages: {comp_stats['total_samples']:,}")
        print(f"Estimated memory usage: {comp_stats['estimated_memory_mb']:.1f} MB")
        print(f"Stages loaded: {comp_stats['stages_loaded']}")
        print(f"Total load time: {sum(comp_stats['load_times'].values()):.2f}s")

        # Integration info
        print("\n=== Training System Integration ===")
        integration = self.get_training_schedule_integration()

        print(f"Data manager ready: {integration['data_manager_ready']}")
        print(f"Stages available: {integration['stages_available']}")
        print(f"Augmentation stages: {integration['augmentation_stages']}")
        print(f"Batch sizes: {integration['batch_sizes']}")

        print("\n=== Pipeline Demonstration Complete ===")
        print("🚀 Ready to replace HRRM with comprehensive Cogment curriculum! 🚀")


def create_cogment_data_manager(
    curriculum: Any | None = None,
    loading_strategy: DataLoadingStrategy = DataLoadingStrategy.HYBRID,
    base_config: dict[str, Any] | None = None,
) -> CogmentDataManager:
    """Factory function to create Cogment data manager."""

    return CogmentDataManager(curriculum=curriculum, loading_strategy=loading_strategy, base_config=base_config)


def demo_cogment_data_manager():
    """Demonstrate Cogment data manager functionality."""
    print("=== Cogment Data Manager Demo ===")
    print("Initializing comprehensive 4-stage curriculum data pipeline...")
    print("This replaces HRRM's limited synthetic approach with real-world datasets.")
    print()

    # Create data manager
    config = {"seed": 42, "demo_mode": True}  # Reduced dataset sizes for demo

    manager = create_cogment_data_manager(loading_strategy=DataLoadingStrategy.HYBRID, base_config=config)

    # Demonstrate full pipeline
    manager.demonstrate_full_pipeline()


if __name__ == "__main__":
    demo_cogment_data_manager()
