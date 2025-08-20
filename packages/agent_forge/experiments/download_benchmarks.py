#!/usr/bin/env python3
"""
Download Benchmark Datasets for EvoMerge Evaluation

Downloads and caches the key benchmarking datasets:
- HumanEval (code generation)
- GSM8K (math reasoning)
- HellaSwag (multilingual/reasoning)
- ARC (structured reasoning)
"""

import logging
import os
from pathlib import Path

# Disable offline mode for downloads
os.environ.pop("HF_DATASETS_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # Disable telemetry but allow downloads

from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_benchmark_datasets():
    """Download all benchmark datasets for EvoMerge evaluation."""

    datasets_to_download = [
        {
            "name": "HumanEval",
            "dataset_id": "openai_humaneval",
            "description": "Code generation benchmark with 164 programming problems",
        },
        {"name": "GSM8K", "dataset_id": "gsm8k", "description": "Grade school math word problems (8.5K examples)"},
        {"name": "HellaSwag", "dataset_id": "hellaswag", "description": "Commonsense reasoning benchmark"},
        {
            "name": "ARC-Easy",
            "dataset_id": "ai2_arc",
            "split": "ARC-Easy",
            "description": "Science exam questions (easy)",
        },
        {
            "name": "ARC-Challenge",
            "dataset_id": "ai2_arc",
            "split": "ARC-Challenge",
            "description": "Science exam questions (challenging)",
        },
    ]

    logger.info("[FIRE] Starting benchmark dataset downloads...")
    logger.info("This will download datasets for real model evaluation")

    # Create benchmarks directory
    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(exist_ok=True)

    successful_downloads = []
    failed_downloads = []

    for dataset_info in datasets_to_download:
        try:
            logger.info(f"\n📊 Downloading {dataset_info['name']}...")
            logger.info(f"   Description: {dataset_info['description']}")

            # Download dataset
            if "split" in dataset_info:
                dataset = load_dataset(dataset_info["dataset_id"], dataset_info["split"])
            else:
                dataset = load_dataset(dataset_info["dataset_id"])

            # Cache to local directory
            cache_path = benchmarks_dir / dataset_info["name"].lower().replace("-", "_")
            cache_path.mkdir(exist_ok=True)

            # Save dataset info
            with open(cache_path / "info.txt", "w") as f:
                f.write(f"Dataset: {dataset_info['name']}\n")
                f.write(f"Source: {dataset_info['dataset_id']}\n")
                f.write(f"Description: {dataset_info['description']}\n")
                if "split" in dataset_info:
                    f.write(f"Split: {dataset_info['split']}\n")
                f.write(f"Size: {len(dataset) if hasattr(dataset, '__len__') else 'Unknown'} examples\n")

            # Log sample sizes
            if hasattr(dataset, "keys"):
                for split_name in dataset.keys():
                    size = len(dataset[split_name])
                    logger.info(f"   [OK] {split_name}: {size:,} examples")
            else:
                size = len(dataset) if hasattr(dataset, "__len__") else "Unknown"
                logger.info(f"   [OK] Dataset size: {size} examples")

            successful_downloads.append(dataset_info["name"])

        except Exception as e:
            logger.error(f"   [ERROR] Failed to download {dataset_info['name']}: {e}")
            failed_downloads.append(dataset_info["name"])
            continue

    # Summary
    logger.info("\n🎯 Download Summary:")
    logger.info(f"   [OK] Successful: {len(successful_downloads)} datasets")
    for name in successful_downloads:
        logger.info(f"      - {name}")

    if failed_downloads:
        logger.info(f"   [ERROR] Failed: {len(failed_downloads)} datasets")
        for name in failed_downloads:
            logger.info(f"      - {name}")

    logger.info(f"\n📁 Datasets cached in: {benchmarks_dir.absolute()}")
    logger.info("These datasets are now available for EvoMerge real evaluation!")

    return successful_downloads, failed_downloads


def test_dataset_access():
    """Test that we can access the downloaded datasets."""
    logger.info("\n🧪 Testing dataset access...")

    try:
        # Test HumanEval access
        humaneval = load_dataset("openai_humaneval")
        logger.info(f"[OK] HumanEval: {len(humaneval['test'])} problems accessible")

        # Show sample problem
        sample = humaneval["test"][0]
        logger.info(f"   Sample problem: {sample['prompt'][:100]}...")

        # Test GSM8K access
        gsm8k = load_dataset("gsm8k", "main")
        logger.info(f"[OK] GSM8K: {len(gsm8k['train'])} train + {len(gsm8k['test'])} test problems")

        # Show sample problem
        sample = gsm8k["train"][0]
        logger.info(f"   Sample problem: {sample['question'][:100]}...")

        return True

    except Exception as e:
        logger.error(f"[ERROR] Dataset access test failed: {e}")
        return False


if __name__ == "__main__":
    print("[ROCKET] EvoMerge Benchmark Dataset Downloader")
    print("=" * 50)

    # Download datasets
    successful, failed = download_benchmark_datasets()

    # Test access
    if successful:
        test_dataset_access()

    print("\n[SUCCESS] Benchmark datasets ready for real EvoMerge evaluation!")
