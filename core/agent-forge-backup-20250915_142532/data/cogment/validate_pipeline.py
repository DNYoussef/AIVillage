#!/usr/bin/env python3
"""
Cogment Data Pipeline Validation Script

Comprehensive validation of the 4-stage curriculum data pipeline
that replaces HRRM with real-world datasets for accelerated grokking.
"""

import sys
import time


def validate_stage_0():
    """Validate Stage 0: Sanity Checks"""
    print("🔍 Validating Stage 0: Sanity Checks...")

    try:
        import stage_0_sanity

        # Create small test dataset
        config = {"samples_per_task": 5, "sequence_length": 64, "seed": 42}

        dataset = stage_0_sanity.create_sanity_dataset(config)

        # Validate dataset
        assert len(dataset) > 0, "Empty dataset"
        assert dataset.validate_samples(), "Sample validation failed"

        # Test data loader
        loader = dataset.get_data_loader(batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert "inputs" in batch, "Missing inputs in batch"
        assert "targets" in batch, "Missing targets in batch"
        assert len(batch["inputs"]) == 2, "Incorrect batch size"

        print(f"   ✅ Stage 0: {len(dataset)} samples, validation passed")
        return True

    except Exception as e:
        print(f"   ❌ Stage 0 failed: {e}")
        return False


def validate_stage_1():
    """Validate Stage 1: ARC Visual Reasoning"""
    print("🔍 Validating Stage 1: ARC Visual Reasoning...")

    try:
        import stage_1_arc

        # Create test dataset
        config = {
            "max_tasks": 3,
            "augmentations_per_task": 5,
            "cache_augmentations": True,
            "use_synthetic_arc": True,
            "use_huggingface_arc": False,
        }

        dataset = stage_1_arc.create_arc_dataset(config)

        # Validate dataset
        assert len(dataset) > 0, "Empty dataset"

        # Test data loader
        loader = dataset.get_data_loader(batch_size=1, shuffle=False)
        batch = next(iter(loader))

        assert "tasks" in batch, "Missing tasks in batch"
        assert len(batch["tasks"]) == 1, "Incorrect batch size"

        print(f"   ✅ Stage 1: {len(dataset)} samples, ARC tasks validated")
        return True

    except Exception as e:
        print(f"   ❌ Stage 1 failed: {e}")
        return False


def validate_stage_2():
    """Validate Stage 2: Algorithmic Puzzles"""
    print("🔍 Validating Stage 2: Algorithmic Puzzles...")

    try:
        import stage_2_puzzles
        from stage_2_puzzles import DifficultyLevel

        # Create test dataset
        config = {"samples_per_type": 3, "difficulties": [DifficultyLevel.EASY], "seed": 42}

        dataset = stage_2_puzzles.create_puzzle_dataset(config)

        # Validate dataset
        assert len(dataset) > 0, "Empty dataset"
        assert dataset.validate_samples(), "Sample validation failed"

        # Test data loader
        loader = dataset.get_data_loader(batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert "inputs" in batch, "Missing inputs in batch"
        assert "targets" in batch, "Missing targets in batch"

        # Check task distribution
        distribution = dataset.get_task_distribution()
        assert len(distribution) > 0, "No task distribution"

        print(f"   ✅ Stage 2: {len(dataset)} samples, {len(distribution)} task types")
        return True

    except Exception as e:
        print(f"   ❌ Stage 2 failed: {e}")
        return False


def validate_stage_3():
    """Validate Stage 3: Math & Text Reasoning"""
    print("🔍 Validating Stage 3: Math & Text Reasoning...")

    try:
        import stage_3_reasoning

        # Create test dataset (synthetic only for speed)
        config = {
            "use_gsm8k": False,  # Use synthetic for testing
            "use_hotpotqa": False,
            "use_competition_math": False,
            "chain_of_thought": True,
            "show_reasoning": True,
        }

        dataset = stage_3_reasoning.create_reasoning_dataset(config)

        # Should still have synthetic data
        assert len(dataset) > 0, "Empty dataset"

        # Test data loader
        loader = dataset.get_data_loader(batch_size=2, shuffle=False)
        batch = next(iter(loader))

        assert "inputs" in batch, "Missing inputs in batch"
        assert "targets" in batch, "Missing targets in batch"

        print(f"   ✅ Stage 3: {len(dataset)} samples, reasoning validated")
        return True

    except Exception as e:
        print(f"   ❌ Stage 3 failed: {e}")
        return False


def validate_stage_4():
    """Validate Stage 4: Long Context"""
    print("🔍 Validating Stage 4: Long Context...")

    try:
        import stage_4_longcontext

        # Create test dataset
        config = {
            "longbench_limit": 5,
            "scrolls_limit": 5,
            "synthetic_limit": 10,
            "target_lengths": [512, 1024],  # Smaller for testing
            "min_context_length": 200,
        }

        dataset = stage_4_longcontext.create_long_context_dataset(config)

        # Validate dataset
        assert len(dataset) > 0, "Empty dataset"

        # Test data loader
        loader = dataset.get_data_loader(batch_size=1, shuffle=False)
        batch = next(iter(loader))

        assert "inputs" in batch, "Missing inputs in batch"
        assert "targets" in batch, "Missing targets in batch"

        # Check context length distribution
        context_dist = dataset.get_context_length_distribution()
        assert len(context_dist) > 0, "No context distribution"

        print(f"   ✅ Stage 4: {len(dataset)} samples, long context validated")
        return True

    except Exception as e:
        print(f"   ❌ Stage 4 failed: {e}")
        return False


def validate_augmentation_engine():
    """Validate ARC Augmentation Engine"""
    print("🔍 Validating Augmentation Engine...")

    try:
        import augmentations
        from augmentations import AugmentationConfig

        # Create test config
        config = AugmentationConfig(max_augmentations_per_task=10, max_color_permutations=3)

        engine = augmentations.ARCAugmentationEngine(config)

        # Test with sample ARC task
        sample_task = {
            "train": [{"input": [[0, 1, 0], [1, 2, 1], [0, 1, 0]], "output": [[1, 0, 1], [0, 2, 0], [1, 0, 1]]}],
            "test": [{"input": [[0, 2, 0], [2, 1, 2], [0, 2, 0]]}],
        }

        # Generate augmentations
        augmented_tasks = engine.augment_arc_task(sample_task)

        assert len(augmented_tasks) > 1, "No augmentations generated"
        assert len(augmented_tasks) <= config.max_augmentations_per_task + 1, "Too many augmentations"

        print(f"   ✅ Augmentation Engine: {len(augmented_tasks)} variants generated")
        return True

    except Exception as e:
        print(f"   ❌ Augmentation Engine failed: {e}")
        return False


def validate_data_manager():
    """Validate Central Data Manager"""
    print("🔍 Validating Central Data Manager...")

    try:
        import data_manager
        from data_manager import DataLoadingStrategy

        # Create test manager
        config = {"seed": 42, "demo_mode": True}

        manager = data_manager.create_cogment_data_manager(
            loading_strategy=DataLoadingStrategy.ON_DEMAND, base_config=config
        )

        # Test stage loading
        stage_0_loader = manager.get_stage_loader(0)
        assert stage_0_loader is not None, "Stage 0 loader not created"

        stage_2_loader = manager.get_stage_loader(2)
        assert stage_2_loader is not None, "Stage 2 loader not created"

        # Test configuration access
        stage_config = manager.get_stage_config(0)
        assert stage_config.stage == 0, "Incorrect stage config"

        # Test validation
        is_valid = manager.validate_data_quality(0)
        assert is_valid, "Stage 0 validation failed"

        # Test statistics
        stats = manager.get_comprehensive_stats()
        assert "stages_loaded" in stats, "Missing stats"
        assert len(stats["stages_loaded"]) > 0, "No stages loaded"

        print(f"   ✅ Data Manager: {len(stats['stages_loaded'])} stages loaded")
        return True

    except Exception as e:
        print(f"   ❌ Data Manager failed: {e}")
        return False


def main():
    """Run complete pipeline validation"""
    print("🚀 COGMENT 4-STAGE DATA PIPELINE VALIDATION")
    print("=" * 60)
    print("Replacing HRRM with comprehensive real-world datasets...")
    print()

    start_time = time.time()

    # Run all validations
    validators = [
        ("Stage 0 - Sanity Checks", validate_stage_0),
        ("Stage 1 - ARC Visual", validate_stage_1),
        ("Stage 2 - Algorithmic Puzzles", validate_stage_2),
        ("Stage 3 - Math & Text Reasoning", validate_stage_3),
        ("Stage 4 - Long Context", validate_stage_4),
        ("Augmentation Engine", validate_augmentation_engine),
        ("Central Data Manager", validate_data_manager),
    ]

    results = []

    for name, validator in validators:
        try:
            success = validator()
            results.append((name, success))
        except Exception as e:
            print(f"   ❌ {name} validation crashed: {e}")
            results.append((name, False))
        print()

    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, success in results if success)
    total = len(results)

    print("=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")

    print()
    print(f"🎯 Overall: {passed}/{total} components passed ({passed/total*100:.1f}%)")
    print(f"⏱️  Total validation time: {total_time:.2f}s")

    if passed == total:
        print()
        print("🎉 PIPELINE VALIDATION SUCCESSFUL!")
        print("🚀 Cogment 4-stage curriculum ready to replace HRRM!")
        print("🧠 Ready for accelerated grokking with comprehensive datasets!")
    else:
        print()
        print("⚠️  Some components failed validation")
        print("🔧 Check error messages above for debugging")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
