"""
Tests for Cogment Data Pipeline System (Agent 5)

Tests the 4-stage curriculum data pipeline including:
- Stage 0: Sanity checks (synthetic tasks, 500 steps)
- Stage 1: ARC visual reasoning (~300 augmentations, 4K steps)
- Stage 2: Algorithmic puzzles (Sudoku, Mazes, ListOps, 8K steps)
- Stage 3: Math & text reasoning (GSM8K, HotpotQA, 16K steps)
- Stage 4: Long-context (LongBench, SCROLLS, 32K steps)
- ARC augmentation engine with ~300 augmentations per task
- Data loading strategies and batch generation
"""

import pytest
import torch

# Import Cogment data components
try:
    from core.agent_forge.data.cogment.augmentations import ARCAugmentationEngine, AugmentationConfig
    from core.agent_forge.data.cogment.data_manager import CogmentDataManager, DataLoadingStrategy, StageDataConfig
    from core.agent_forge.data.cogment.stage_0_sanity import create_sanity_dataset
    from core.agent_forge.data.cogment.stage_1_arc import create_arc_dataset
    from core.agent_forge.data.cogment.stage_2_puzzles import create_puzzle_dataset
    from core.agent_forge.data.cogment.stage_3_reasoning import create_reasoning_dataset
    from core.agent_forge.data.cogment.stage_4_longcontext import create_long_context_dataset

    DATA_AVAILABLE = True
except ImportError as e:
    print(f"Cogment data components not available: {e}")
    DATA_AVAILABLE = False


class TestSanityCheckDataset:
    """Test Stage 0: Sanity Check Dataset."""

    @pytest.fixture
    def sanity_config(self):
        """Create sanity check configuration."""
        return {
            "samples_per_task": 10,  # Reduced for testing
            "sequence_length": 64,
            "seed": 42,
            "task_types": ["addition", "copying", "reversal"],
            "demo_mode": True,
        }

    @pytest.fixture
    def sanity_dataset(self, sanity_config):
        """Create SanityCheckDataset instance."""
        if not DATA_AVAILABLE:
            pytest.skip("Data components not available")
        return create_sanity_dataset(sanity_config)

    def test_sanity_dataset_creation(self, sanity_dataset, sanity_config):
        """Test sanity dataset instantiation."""
        assert len(sanity_dataset) > 0
        assert hasattr(sanity_dataset, "get_data_loader")
        assert hasattr(sanity_dataset, "validate_samples")

    def test_sanity_dataset_samples(self, sanity_dataset):
        """Test sanity dataset sample structure."""
        sample = sanity_dataset[0]

        # Verify sample structure
        assert "input_ids" in sample
        assert "target_ids" in sample
        assert "task_type" in sample

        # Verify tensor types and shapes
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert isinstance(sample["target_ids"], torch.Tensor)
        assert sample["input_ids"].ndim == 1  # 1D sequence
        assert sample["target_ids"].ndim == 1  # 1D sequence

    def test_sanity_dataset_task_distribution(self, sanity_dataset, sanity_config):
        """Test task type distribution in sanity dataset."""
        task_counts = {}

        for i in range(min(len(sanity_dataset), 30)):  # Sample subset
            sample = sanity_dataset[i]
            task_type = sample["task_type"]
            task_counts[task_type] = task_counts.get(task_type, 0) + 1

        # Should have multiple task types
        assert len(task_counts) >= 2, f"Should have multiple task types: {task_counts}"

        # All task types should be from expected set
        expected_tasks = set(sanity_config["task_types"])
        actual_tasks = set(task_counts.keys())
        assert actual_tasks.issubset(expected_tasks), f"Unexpected task types: {actual_tasks - expected_tasks}"

    def test_sanity_dataset_validation(self, sanity_dataset):
        """Test sanity dataset validation."""
        is_valid = sanity_dataset.validate_samples()
        assert is_valid, "Sanity dataset samples should be valid"

    def test_sanity_dataset_data_loader(self, sanity_dataset):
        """Test sanity dataset data loader."""
        loader = sanity_dataset.get_data_loader(batch_size=4, shuffle=True)

        batch = next(iter(loader))

        # Verify batch structure
        assert "input_ids" in batch
        assert "target_ids" in batch
        assert "task_type" in batch

        # Verify batch dimensions
        assert batch["input_ids"].shape[0] == 4  # Batch size
        assert batch["target_ids"].shape[0] == 4  # Batch size


class TestARCVisualDataset:
    """Test Stage 1: ARC Visual Reasoning Dataset."""

    @pytest.fixture
    def arc_config(self):
        """Create ARC configuration."""
        return {
            "max_tasks": 20,  # Reduced for testing
            "augmentations_per_task": 5,  # Reduced for testing
            "cache_augmentations": True,
            "use_synthetic_arc": True,
            "use_huggingface_arc": False,
            "augmentation_seed": 42,
            "demo_mode": True,
        }

    @pytest.fixture
    def arc_dataset(self, arc_config):
        """Create ARCVisualDataset instance."""
        if not DATA_AVAILABLE:
            pytest.skip("Data components not available")
        return create_arc_dataset(arc_config)

    def test_arc_dataset_creation(self, arc_dataset, arc_config):
        """Test ARC dataset instantiation."""
        assert len(arc_dataset) > 0
        assert hasattr(arc_dataset, "augmentation_engine")
        assert hasattr(arc_dataset, "get_task_distribution")

    def test_arc_dataset_samples(self, arc_dataset):
        """Test ARC dataset sample structure."""
        sample = arc_dataset[0]

        # Verify sample structure for visual reasoning
        assert "input_grid" in sample
        assert "output_grid" in sample
        assert "task_id" in sample

        # Verify tensor types and shapes
        assert isinstance(sample["input_grid"], torch.Tensor)
        assert isinstance(sample["output_grid"], torch.Tensor)

        # ARC grids should be 2D
        assert sample["input_grid"].ndim >= 2
        assert sample["output_grid"].ndim >= 2

    def test_arc_augmentation_engine(self, arc_dataset, arc_config):
        """Test ARC augmentation engine functionality."""
        if not hasattr(arc_dataset, "augmentation_engine"):
            pytest.skip("Augmentation engine not available")

        engine = arc_dataset.augmentation_engine

        # Test augmentation generation
        base_task = {"input_grid": torch.randint(0, 10, (5, 5)), "output_grid": torch.randint(0, 10, (5, 5))}

        augmentations = engine.generate_augmentations(base_task, num_augmentations=3)

        assert len(augmentations) == 3

        for aug in augmentations:
            assert "input_grid" in aug
            assert "output_grid" in aug
            assert "augmentation_type" in aug

    def test_arc_dataset_augmentation_count(self, arc_dataset, arc_config):
        """Test ARC dataset achieves target augmentation count."""
        # Count unique task IDs vs total samples
        task_ids = set()
        total_samples = min(len(arc_dataset), 50)  # Sample subset

        for i in range(total_samples):
            sample = arc_dataset[i]
            task_ids.add(sample["task_id"])

        unique_tasks = len(task_ids)
        augmentation_ratio = total_samples / unique_tasks if unique_tasks > 0 else 0

        # Should have multiple augmentations per task
        expected_ratio = arc_config["augmentations_per_task"]

        assert (
            augmentation_ratio >= expected_ratio * 0.8
        ), f"Augmentation ratio too low: {augmentation_ratio:.1f} (expected ~{expected_ratio})"

        print(f"✓ ARC augmentation ratio: {augmentation_ratio:.1f} augmentations per task")

    def test_arc_task_distribution(self, arc_dataset):
        """Test ARC task distribution."""
        if hasattr(arc_dataset, "get_task_distribution"):
            distribution = arc_dataset.get_task_distribution()

            assert isinstance(distribution, dict)
            assert len(distribution) > 0

            # Should have reasonable distribution
            total_tasks = sum(distribution.values())
            assert total_tasks > 0


class TestAlgorithmicPuzzleDataset:
    """Test Stage 2: Algorithmic Puzzle Dataset."""

    @pytest.fixture
    def puzzle_config(self):
        """Create puzzle configuration."""
        return {
            "samples_per_type": 10,  # Reduced for testing
            "puzzle_types": ["sudoku", "maze", "listops"],
            "difficulties": ["easy", "medium"],
            "seed": 42,
            "demo_mode": True,
        }

    @pytest.fixture
    def puzzle_dataset(self, puzzle_config):
        """Create AlgorithmicPuzzleDataset instance."""
        if not DATA_AVAILABLE:
            pytest.skip("Data components not available")
        return create_puzzle_dataset(puzzle_config)

    def test_puzzle_dataset_creation(self, puzzle_dataset, puzzle_config):
        """Test puzzle dataset instantiation."""
        assert len(puzzle_dataset) > 0
        assert hasattr(puzzle_dataset, "get_domain_distribution")

    def test_puzzle_dataset_samples(self, puzzle_dataset):
        """Test puzzle dataset sample structure."""
        sample = puzzle_dataset[0]

        # Verify sample structure
        assert "puzzle_input" in sample
        assert "puzzle_solution" in sample
        assert "puzzle_type" in sample
        assert "difficulty" in sample

        # Verify data types
        assert isinstance(sample["puzzle_input"], torch.Tensor | str)
        assert isinstance(sample["puzzle_solution"], torch.Tensor | str)

    def test_puzzle_type_distribution(self, puzzle_dataset, puzzle_config):
        """Test puzzle type distribution."""
        puzzle_counts = {}

        for i in range(min(len(puzzle_dataset), 30)):
            sample = puzzle_dataset[i]
            puzzle_type = sample["puzzle_type"]
            puzzle_counts[puzzle_type] = puzzle_counts.get(puzzle_type, 0) + 1

        # Should have multiple puzzle types
        assert len(puzzle_counts) >= 2, f"Should have multiple puzzle types: {puzzle_counts}"

        # All types should be from expected set
        expected_types = set(puzzle_config["puzzle_types"])
        actual_types = set(puzzle_counts.keys())
        assert actual_types.issubset(expected_types), f"Unexpected puzzle types: {actual_types - expected_types}"

    def test_puzzle_difficulty_distribution(self, puzzle_dataset, puzzle_config):
        """Test puzzle difficulty distribution."""
        difficulty_counts = {}

        for i in range(min(len(puzzle_dataset), 30)):
            sample = puzzle_dataset[i]
            difficulty = sample["difficulty"]
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

        # Should have multiple difficulties
        expected_difficulties = set(puzzle_config["difficulties"])
        actual_difficulties = set(difficulty_counts.keys())
        assert actual_difficulties.issubset(
            expected_difficulties
        ), f"Unexpected difficulties: {actual_difficulties - expected_difficulties}"


class TestMathTextReasoningDataset:
    """Test Stage 3: Math & Text Reasoning Dataset."""

    @pytest.fixture
    def reasoning_config(self):
        """Create reasoning configuration."""
        return {
            "gsm8k_limit": 20,  # Reduced for testing
            "hotpotqa_limit": 10,  # Reduced for testing
            "math_limit": 10,  # Reduced for testing
            "chain_of_thought": True,
            "show_reasoning": True,
            "demo_mode": True,
        }

    @pytest.fixture
    def reasoning_dataset(self, reasoning_config):
        """Create MathTextReasoningDataset instance."""
        if not DATA_AVAILABLE:
            pytest.skip("Data components not available")
        return create_reasoning_dataset(reasoning_config)

    def test_reasoning_dataset_creation(self, reasoning_dataset, reasoning_config):
        """Test reasoning dataset instantiation."""
        assert len(reasoning_dataset) > 0
        assert hasattr(reasoning_dataset, "validate_reasoning_quality")

    def test_reasoning_dataset_samples(self, reasoning_dataset):
        """Test reasoning dataset sample structure."""
        sample = reasoning_dataset[0]

        # Verify sample structure
        assert "question" in sample
        assert "answer" in sample
        assert "reasoning_type" in sample

        # Chain of thought samples should have reasoning
        if "reasoning" in sample:
            assert isinstance(sample["reasoning"], str)
            assert len(sample["reasoning"]) > 0

    def test_reasoning_type_distribution(self, reasoning_dataset):
        """Test reasoning type distribution."""
        reasoning_counts = {}

        for i in range(min(len(reasoning_dataset), 30)):
            sample = reasoning_dataset[i]
            reasoning_type = sample["reasoning_type"]
            reasoning_counts[reasoning_type] = reasoning_counts.get(reasoning_type, 0) + 1

        # Should have multiple reasoning types
        assert len(reasoning_counts) >= 2, f"Should have multiple reasoning types: {reasoning_counts}"

        expected_types = {"gsm8k", "hotpotqa", "math"}
        actual_types = set(reasoning_counts.keys())
        assert actual_types.issubset(expected_types), f"Unexpected reasoning types: {actual_types - expected_types}"

    def test_reasoning_quality_validation(self, reasoning_dataset):
        """Test reasoning quality validation."""
        is_valid = reasoning_dataset.validate_reasoning_quality()
        assert is_valid, "Reasoning dataset should pass quality validation"


class TestLongContextDataset:
    """Test Stage 4: Long Context Dataset."""

    @pytest.fixture
    def long_context_config(self):
        """Create long context configuration."""
        return {
            "longbench_limit": 10,  # Reduced for testing
            "scrolls_limit": 5,  # Reduced for testing
            "synthetic_limit": 5,  # Reduced for testing
            "target_lengths": [512, 1024],  # Reduced for testing
            "min_context_length": 512,
            "demo_mode": True,
        }

    @pytest.fixture
    def long_context_dataset(self, long_context_config):
        """Create LongContextDataset instance."""
        if not DATA_AVAILABLE:
            pytest.skip("Data components not available")
        return create_long_context_dataset(long_context_config)

    def test_long_context_dataset_creation(self, long_context_dataset, long_context_config):
        """Test long context dataset instantiation."""
        assert len(long_context_dataset) > 0
        assert hasattr(long_context_dataset, "validate_long_context_quality")
        assert hasattr(long_context_dataset, "get_context_length_distribution")

    def test_long_context_dataset_samples(self, long_context_dataset):
        """Test long context dataset sample structure."""
        sample = long_context_dataset[0]

        # Verify sample structure
        assert "context" in sample
        assert "question" in sample
        assert "answer" in sample
        assert "context_length" in sample

        # Verify context length
        context_length = sample["context_length"]
        assert isinstance(context_length, int | torch.Tensor)

        if isinstance(context_length, int):
            assert context_length >= 512  # Minimum length
        else:
            assert context_length.item() >= 512

    def test_long_context_length_distribution(self, long_context_dataset, long_context_config):
        """Test context length distribution."""
        if hasattr(long_context_dataset, "get_context_length_distribution"):
            distribution = long_context_dataset.get_context_length_distribution()

            assert isinstance(distribution, dict)
            assert len(distribution) > 0

            # Should have contexts meeting minimum length
            min_length = long_context_config["min_context_length"]
            long_contexts = sum(count for length, count in distribution.items() if length >= min_length)
            total_contexts = sum(distribution.values())

            long_ratio = long_contexts / total_contexts if total_contexts > 0 else 0
            assert long_ratio > 0.8, f"Most contexts should be long: {long_ratio:.2f}"

    def test_long_context_quality_validation(self, long_context_dataset):
        """Test long context quality validation."""
        is_valid = long_context_dataset.validate_long_context_quality()
        assert is_valid, "Long context dataset should pass quality validation"


class TestCogmentDataManager:
    """Test Cogment Data Manager integration."""

    @pytest.fixture
    def data_manager_config(self):
        """Create data manager configuration."""
        return {"seed": 42, "demo_mode": True}  # Reduced dataset sizes

    @pytest.fixture
    def data_manager(self, data_manager_config):
        """Create CogmentDataManager instance."""
        if not DATA_AVAILABLE:
            pytest.skip("Data components not available")
        return CogmentDataManager(loading_strategy=DataLoadingStrategy.HYBRID, base_config=data_manager_config)

    def test_data_manager_creation(self, data_manager, data_manager_config):
        """Test data manager instantiation."""
        assert data_manager.loading_strategy == DataLoadingStrategy.HYBRID
        assert data_manager.base_config == data_manager_config
        assert len(data_manager.stage_configs) == 5  # 5 stages (0-4)

    def test_data_manager_stage_configs(self, data_manager):
        """Test data manager stage configurations."""
        # Verify all stages are configured
        for stage in range(5):
            assert stage in data_manager.stage_configs
            config = data_manager.stage_configs[stage]

            assert isinstance(config, StageDataConfig)
            assert config.stage == stage
            assert config.batch_size > 0
            assert config.sequence_length > 0

    def test_data_manager_stage_loading(self, data_manager):
        """Test individual stage loading."""
        # Test loading stage 0 (should be fast)
        loader = data_manager.get_stage_loader(0)

        assert loader is not None
        assert hasattr(loader, "__iter__")

        # Test getting a batch
        batch = next(iter(loader))
        assert batch is not None
        assert len(batch) > 0  # Should have some data

    def test_data_manager_loading_strategies(self):
        """Test different loading strategies."""
        if not DATA_AVAILABLE:
            pytest.skip("Data components not available")

        strategies = [DataLoadingStrategy.HYBRID, DataLoadingStrategy.ON_DEMAND, DataLoadingStrategy.SEQUENTIAL]

        config = {"demo_mode": True, "seed": 42}

        for strategy in strategies:
            manager = CogmentDataManager(loading_strategy=strategy, base_config=config)

            assert manager.loading_strategy == strategy

            # Should be able to get stage 0 loader
            loader = manager.get_stage_loader(0)
            assert loader is not None

    def test_data_manager_comprehensive_stats(self, data_manager):
        """Test data manager comprehensive statistics."""
        # Load a few stages
        for stage in [0, 1]:
            data_manager.get_stage_loader(stage)

        stats = data_manager.get_comprehensive_stats()

        # Verify stats structure
        assert "loading_strategy" in stats
        assert "stages_loaded" in stats
        assert "total_samples" in stats
        assert "load_times" in stats
        assert "stage_stats" in stats

        # Verify stats content
        assert stats["loading_strategy"] == data_manager.loading_strategy.value
        assert len(stats["stages_loaded"]) >= 2  # At least stages 0 and 1
        assert stats["total_samples"] > 0

    def test_data_manager_validation(self, data_manager):
        """Test data manager validation."""
        # Test validation for stage 0
        is_valid = data_manager.validate_data_quality(0)
        assert is_valid, "Stage 0 should pass validation"

    def test_data_manager_training_integration(self, data_manager):
        """Test data manager training system integration."""
        integration_info = data_manager.get_training_schedule_integration()

        # Verify integration structure
        assert "data_manager_ready" in integration_info
        assert "stages_available" in integration_info
        assert "batch_sizes" in integration_info
        assert "sequence_lengths" in integration_info
        assert "augmentation_stages" in integration_info

        # Verify integration content
        assert integration_info["data_manager_ready"] is True
        assert len(integration_info["stages_available"]) == 5
        assert len(integration_info["batch_sizes"]) == 5
        assert len(integration_info["sequence_lengths"]) == 5


class TestARCAugmentationEngine:
    """Test ARC Augmentation Engine."""

    @pytest.fixture
    def augmentation_config(self):
        """Create augmentation configuration."""
        return AugmentationConfig(
            rotation_enabled=True,
            color_permutation_enabled=True,
            spatial_transforms_enabled=True,
            max_augmentations_per_task=5,  # Reduced for testing
            preserve_solution_structure=True,
            augmentation_seed=42,
        )

    @pytest.fixture
    def augmentation_engine(self, augmentation_config):
        """Create ARCAugmentationEngine instance."""
        if not DATA_AVAILABLE:
            pytest.skip("Data components not available")
        return ARCAugmentationEngine(augmentation_config)

    def test_augmentation_engine_creation(self, augmentation_engine, augmentation_config):
        """Test augmentation engine instantiation."""
        assert augmentation_engine.config == augmentation_config
        assert hasattr(augmentation_engine, "generate_augmentations")

    def test_augmentation_generation(self, augmentation_engine, augmentation_config):
        """Test augmentation generation."""
        # Create test ARC task
        base_task = {"input_grid": torch.randint(0, 10, (5, 5)), "output_grid": torch.randint(0, 10, (5, 5))}

        num_augmentations = 3
        augmentations = augmentation_engine.generate_augmentations(base_task, num_augmentations)

        assert len(augmentations) == num_augmentations

        for aug in augmentations:
            assert "input_grid" in aug
            assert "output_grid" in aug
            assert "augmentation_type" in aug

            # Verify grid shapes are preserved or reasonably transformed
            assert aug["input_grid"].shape[-2:] == base_task["input_grid"].shape[-2:]  # Same spatial dims
            assert aug["output_grid"].shape[-2:] == base_task["output_grid"].shape[-2:]

    def test_augmentation_variety(self, augmentation_engine):
        """Test augmentation variety and types."""
        base_task = {"input_grid": torch.randint(0, 10, (4, 4)), "output_grid": torch.randint(0, 10, (4, 4))}

        # Generate multiple augmentations
        augmentations = augmentation_engine.generate_augmentations(base_task, 10)

        # Collect augmentation types
        aug_types = [aug["augmentation_type"] for aug in augmentations]
        unique_types = set(aug_types)

        # Should have variety in augmentation types
        assert len(unique_types) >= 2, f"Should have variety in augmentations: {unique_types}"

    def test_augmentation_count_target(self, augmentation_engine):
        """Test achieving ~300 augmentations target."""
        base_task = {"input_grid": torch.randint(0, 10, (3, 3)), "output_grid": torch.randint(0, 10, (3, 3))}

        # Test with higher augmentation count (simulating production)
        target_count = 20  # Reduced from 300 for testing
        augmentations = augmentation_engine.generate_augmentations(base_task, target_count)

        assert len(augmentations) == target_count

        # Verify all augmentations are valid
        for aug in augmentations:
            assert aug["input_grid"].shape == base_task["input_grid"].shape
            assert aug["output_grid"].shape == base_task["output_grid"].shape
            assert not torch.isnan(aug["input_grid"]).any()
            assert not torch.isnan(aug["output_grid"]).any()


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline."""

    @pytest.mark.skipif(not DATA_AVAILABLE, reason="Data components not available")
    def test_full_pipeline_demonstration(self):
        """Test complete 4-stage data pipeline demonstration."""
        config = {"demo_mode": True, "seed": 42}

        manager = CogmentDataManager(loading_strategy=DataLoadingStrategy.HYBRID, base_config=config)

        # Test full pipeline demonstration
        try:
            manager.demonstrate_full_pipeline()
            success = True
        except Exception as e:
            print(f"Pipeline demonstration failed: {e}")
            success = False

        assert success, "Full pipeline demonstration should succeed"

    @pytest.mark.skipif(not DATA_AVAILABLE, reason="Data components not available")
    def test_all_stages_data_loading(self):
        """Test loading data from all 5 curriculum stages."""
        config = {"demo_mode": True, "seed": 42}

        manager = CogmentDataManager(loading_strategy=DataLoadingStrategy.ON_DEMAND, base_config=config)

        stage_results = {}

        # Test all stages
        for stage in range(5):
            try:
                loader = manager.get_stage_loader(stage)
                batch = next(iter(loader))

                stage_results[stage] = {
                    "success": True,
                    "batch_size": len(batch.get("inputs", batch.get("input_ids", []))),
                    "loader_size": len(loader.dataset) if hasattr(loader, "dataset") else "unknown",
                }

            except Exception as e:
                stage_results[stage] = {"success": False, "error": str(e)}

        # Verify results
        successful_stages = [s for s, r in stage_results.items() if r["success"]]
        assert len(successful_stages) >= 3, f"At least 3 stages should succeed: {stage_results}"

        print("✓ Stage loading results:")
        for stage, result in stage_results.items():
            if result["success"]:
                print(f"  Stage {stage}: ✓ {result['batch_size']} batch size, {result['loader_size']} samples")
            else:
                print(f"  Stage {stage}: ✗ {result['error']}")

    @pytest.mark.skipif(not DATA_AVAILABLE, reason="Data components not available")
    def test_pipeline_memory_efficiency(self):
        """Test data pipeline memory efficiency."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        config = {"demo_mode": True, "seed": 42}

        # Create manager and load stages
        manager = CogmentDataManager(loading_strategy=DataLoadingStrategy.HYBRID, base_config=config)

        # Load smaller stages
        for stage in [0, 1]:
            manager.get_stage_loader(stage)

        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - initial_memory

        # Should be reasonably efficient
        max_expected_memory = 200  # MB for demo mode

        assert (
            memory_used <= max_expected_memory
        ), f"Data pipeline memory usage too high: {memory_used:.1f}MB (expected ≤{max_expected_memory}MB)"

        print(f"✓ Data pipeline memory usage: {memory_used:.1f}MB")

    @pytest.mark.skipif(not DATA_AVAILABLE, reason="Data components not available")
    def test_curriculum_progression_integration(self):
        """Test data pipeline integration with curriculum progression."""
        config = {"demo_mode": True, "seed": 42}

        manager = CogmentDataManager(loading_strategy=DataLoadingStrategy.SEQUENTIAL, base_config=config)

        # Simulate curriculum progression
        progression_results = []

        for current_stage in range(3):  # Test first 3 stages
            try:
                # Optimize for current stage
                manager.optimize_for_stage(current_stage)

                # Get current stage loader
                loader = manager.get_stage_loader(current_stage)
                next(iter(loader))

                # Get stage config
                stage_config = manager.get_stage_config(current_stage)

                progression_results.append(
                    {
                        "stage": current_stage,
                        "success": True,
                        "batch_size": stage_config.batch_size,
                        "sequence_length": stage_config.sequence_length,
                        "dataset_type": stage_config.dataset_type,
                    }
                )

            except Exception as e:
                progression_results.append({"stage": current_stage, "success": False, "error": str(e)})

        # Verify progression
        successful_stages = [r for r in progression_results if r["success"]]
        assert len(successful_stages) >= 2, f"At least 2 stages should succeed in progression: {progression_results}"

        # Verify stage characteristics change
        if len(successful_stages) >= 2:
            stage0 = successful_stages[0]
            stage1 = successful_stages[1]

            # Later stages may have different characteristics
            configs_differ = (
                stage0["batch_size"] != stage1["batch_size"]
                or stage0["sequence_length"] != stage1["sequence_length"]
                or stage0["dataset_type"] != stage1["dataset_type"]
            )

            assert configs_differ, "Stage configurations should differ across curriculum"

        print("✓ Curriculum progression integration successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
