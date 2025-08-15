#!/usr/bin/env python3
"""Benchmark Dataset Download Script for Agent Forge.

Downloads standard math benchmarking datasets for evaluating
evolutionary model performance:
- GSM8K (Grade School Math)
- MATH (Competition Mathematics)
- MathQA (Math Question Answering)

These datasets provide comprehensive evaluation across different
mathematical reasoning capabilities.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BENCHMARKS = {
    "gsm8k": {
        "dataset_id": "gsm8k",
        "config": "main",
        "splits": ["train", "test"],
        "description": "Grade School Math 8K - arithmetic word problems",
        "metrics": ["accuracy", "step_accuracy"],
        "difficulty": "elementary",
    },
    "math": {
        "dataset_id": "hendrycks/math",
        "config": None,
        "splits": ["train", "test"],
        "description": "MATH dataset - competition mathematics problems",
        "metrics": ["accuracy", "subject_accuracy"],
        "difficulty": "advanced",
    },
    "mathqa": {
        "dataset_id": "math_qa",
        "config": None,
        "splits": ["train", "validation", "test"],
        "description": "MathQA - multiple choice math questions",
        "metrics": ["accuracy", "category_accuracy"],
        "difficulty": "intermediate",
    },
}


def download_benchmark(benchmark_key: str, config: dict, base_path: Path) -> bool:
    """Download a single benchmark dataset."""
    dataset_id = config["dataset_id"]
    benchmark_path = base_path / benchmark_key

    logger.info(f"Downloading {benchmark_key}: {dataset_id}")
    logger.info(f"Description: {config['description']}")

    try:
        benchmark_path.mkdir(parents=True, exist_ok=True)

        # Download dataset
        if config["config"]:
            dataset = load_dataset(dataset_id, config["config"])
        else:
            dataset = load_dataset(dataset_id)

        # Save each split
        for split in config["splits"]:
            if split in dataset:
                split_path = benchmark_path / f"{split}.json"

                # Convert to list of dictionaries for easier processing
                split_data = []
                for example in dataset[split]:
                    split_data.append(dict(example))

                with open(split_path, "w") as f:
                    json.dump(split_data, f, indent=2)

                logger.info(f"Saved {split} split: {len(split_data)} examples")
            else:
                logger.warning(f"Split '{split}' not found in {dataset_id}")

        # Save metadata
        metadata = {
            "dataset_id": dataset_id,
            "config": config["config"],
            "description": config["description"],
            "metrics": config["metrics"],
            "difficulty": config["difficulty"],
            "splits_available": list(dataset.keys()),
            "total_examples": sum(len(dataset[split]) for split in dataset),
        }

        metadata_path = benchmark_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Successfully downloaded {benchmark_key}")
        return True

    except Exception as e:
        logger.exception(f"Failed to download {benchmark_key}: {e}")
        return False


def create_benchmark_manifest(
    base_path: Path, downloaded_benchmarks: list[str]
) -> None:
    """Create manifest file for benchmark datasets."""
    manifest_path = base_path / "benchmark_manifest.json"

    from datetime import datetime

    manifest = {
        "created_at": datetime.now().isoformat(),
        "download_location": str(base_path),
        "benchmarks": {},
    }

    for benchmark_key in downloaded_benchmarks:
        if benchmark_key in BENCHMARKS:
            config = BENCHMARKS[benchmark_key]
            benchmark_path = base_path / benchmark_key

            # Count total examples across splits
            total_examples = 0
            splits_info = {}

            for split in config["splits"]:
                split_file = benchmark_path / f"{split}.json"
                if split_file.exists():
                    try:
                        with open(split_file) as f:
                            split_data = json.load(f)
                            split_count = len(split_data)
                            splits_info[split] = split_count
                            total_examples += split_count
                    except BaseException:
                        splits_info[split] = 0

            manifest["benchmarks"][benchmark_key] = {
                "dataset_id": config["dataset_id"],
                "local_path": str(benchmark_path),
                "description": config["description"],
                "difficulty": config["difficulty"],
                "metrics": config["metrics"],
                "total_examples": total_examples,
                "splits": splits_info,
                "downloaded": benchmark_path.exists(),
            }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Created benchmark manifest: {manifest_path}")


def create_evaluation_script(base_path: Path) -> None:
    """Create evaluation script template."""
    eval_script = base_path / "evaluate_model.py"

    script_content = '''#!/usr/bin/env python3
"""
Model Evaluation Script for Agent Forge Benchmarks

This script evaluates models against the downloaded benchmark datasets
and provides standardized metrics for evolutionary comparison.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        return json.load(f)


def evaluate_on_gsm8k(predictions_path: str, dataset_path: Path) -> Dict[str, float]:
    """Evaluate predictions on GSM8K dataset."""
    data = _load_json(dataset_path / "test.json")
    preds = _load_json(Path(predictions_path))
    correct = 0
    for item, pred in zip(data, preds):
        if str(pred).strip() == str(item.get("answer", "")).strip():
            correct += 1
    accuracy = correct / len(data) if data else 0.0
    return {"accuracy": accuracy, "step_accuracy": accuracy}


def evaluate_on_math(predictions_path: str, dataset_path: Path) -> Dict[str, float]:
    """Evaluate predictions on MATH dataset."""
    data = _load_json(dataset_path / "test.json")
    preds = _load_json(Path(predictions_path))
    correct = 0
    subject_totals: Dict[str, int] = {}
    subject_correct: Dict[str, int] = {}
    for item, pred in zip(data, preds):
        subject = item.get("subject", "unknown")
        subject_totals[subject] = subject_totals.get(subject, 0) + 1
        if str(pred).strip() == str(item.get("solution", "")).strip():
            correct += 1
            subject_correct[subject] = subject_correct.get(subject, 0) + 1
    accuracy = correct / len(data) if data else 0.0
    subject_accs = [
        subject_correct.get(sub, 0) / total for sub, total in subject_totals.items()
    ]
    subject_accuracy = sum(subject_accs) / len(subject_accs) if subject_accs else 0.0
    return {"accuracy": accuracy, "subject_accuracy": subject_accuracy}


def evaluate_on_mathqa(predictions_path: str, dataset_path: Path) -> Dict[str, float]:
    """Evaluate predictions on MathQA dataset."""
    data = _load_json(dataset_path / "test.json")
    preds = _load_json(Path(predictions_path))
    correct = 0
    category_totals: Dict[str, int] = {}
    category_correct: Dict[str, int] = {}
    for item, pred in zip(data, preds):
        category = item.get("category", "unknown")
        category_totals[category] = category_totals.get(category, 0) + 1
        if str(pred).strip().lower() == str(item.get("correct", "")).strip().lower():
            correct += 1
            category_correct[category] = category_correct.get(category, 0) + 1
    accuracy = correct / len(data) if data else 0.0
    category_accs = [
        category_correct.get(cat, 0) / total for cat, total in category_totals.items()
    ]
    category_accuracy = sum(category_accs) / len(category_accs) if category_accs else 0.0
    return {"accuracy": accuracy, "category_accuracy": category_accuracy}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions on math benchmarks"
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to predictions file"
    )
    parser.add_argument(
        "--benchmarks-dir", default="./benchmarks", help="Benchmarks directory"
    )
    parser.add_argument(
        "--output", default="evaluation_results.json", help="Output file"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["gsm8k", "math", "mathqa", "all"],
        default=["all"],
        help="Benchmarks to evaluate",
    )

    args = parser.parse_args()

    benchmarks_dir = Path(args.benchmarks_dir)
    results = {}

    evaluators = {
        "gsm8k": evaluate_on_gsm8k,
        "math": evaluate_on_math,
        "mathqa": evaluate_on_mathqa,
    }

    if "all" in args.benchmarks:
        to_run = list(evaluators.keys())
    else:
        to_run = args.benchmarks

    for benchmark_name in to_run:
        evaluator = evaluators[benchmark_name]
        benchmark_path = benchmarks_dir / benchmark_name
        if benchmark_path.exists():
            print(f"Evaluating on {benchmark_name}...")
            results[benchmark_name] = evaluator(args.model_path, benchmark_path)
        else:
            print(f"Benchmark {benchmark_name} not found, skipping...")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation results saved to {args.output}")


if __name__ == "__main__":
    main()
'''

    with open(eval_script, "w") as f:
        f.write(script_content)

    logger.info(f"Created evaluation script template: {eval_script}")


def main() -> int:
    """Main download function."""
    parser = argparse.ArgumentParser(
        description="Download Agent Forge benchmark datasets"
    )
    parser.add_argument(
        "--benchmarks-dir",
        default="./benchmarks",
        help="Directory to store downloaded benchmarks",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=[*list(BENCHMARKS.keys()), "all"],
        default=["all"],
        help="Benchmarks to download",
    )

    args = parser.parse_args()

    # Setup paths
    base_path = Path(args.benchmarks_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Determine which benchmarks to download
    if "all" in args.benchmarks:
        benchmarks_to_download = list(BENCHMARKS.keys())
    else:
        benchmarks_to_download = args.benchmarks

    logger.info(
        f"Planning to download {len(benchmarks_to_download)} benchmark datasets"
    )

    # Download benchmarks
    downloaded_benchmarks = []
    failed_benchmarks = []

    for benchmark_key in benchmarks_to_download:
        if benchmark_key not in BENCHMARKS:
            logger.warning(f"Unknown benchmark: {benchmark_key}")
            continue

        success = download_benchmark(
            benchmark_key, BENCHMARKS[benchmark_key], base_path
        )
        if success:
            downloaded_benchmarks.append(benchmark_key)
        else:
            failed_benchmarks.append(benchmark_key)

    # Create manifest and evaluation script
    if downloaded_benchmarks:
        create_benchmark_manifest(base_path, downloaded_benchmarks)
        create_evaluation_script(base_path)

    # Report results
    logger.info("=" * 50)
    logger.info("BENCHMARK DOWNLOAD SUMMARY")
    logger.info(f"Successfully downloaded: {len(downloaded_benchmarks)}")
    for benchmark in downloaded_benchmarks:
        logger.info(f"  ✓ {benchmark}: {BENCHMARKS[benchmark]['dataset_id']}")

    if failed_benchmarks:
        logger.error(f"Failed downloads: {len(failed_benchmarks)}")
        for benchmark in failed_benchmarks:
            logger.error(f"  ✗ {benchmark}: {BENCHMARKS[benchmark]['dataset_id']}")

    logger.info(f"Benchmarks stored in: {base_path}")
    logger.info("=" * 50)

    return 0 if not failed_benchmarks else 1


if __name__ == "__main__":
    sys.exit(main())
