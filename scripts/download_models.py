#!/usr/bin/env python3
"""
Agent Forge Model Download Script

Downloads the optimal 3 models for evolution on RTX 2060 SUPER:
- Qwen2.5-Math-1.5B-Instruct (Math reasoning)
- Qwen2.5-Coder-1.5B-Instruct (Code generation)
- Qwen2.5-1.5B-Instruct (General instruction following)

These models are optimized for 8GB VRAM and provide diverse capabilities
for evolutionary merging.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login
import argparse
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations optimized for RTX 2060 SUPER
MODELS = {
    "math": {
        "repo_id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "description": "Mathematical reasoning and problem solving",
        "size_gb": 3.2,
        "domain": "mathematics"
    },
    "code": {
        "repo_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "description": "Code generation and programming",
        "size_gb": 3.2,
        "domain": "programming"
    },
    "general": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "General instruction following and reasoning",
        "size_gb": 3.2,
        "domain": "general"
    }
}

def check_disk_space(download_path: Path, required_gb: float) -> bool:
    """Check if there's enough disk space for downloads"""
    import shutil
    free_bytes = shutil.disk_usage(download_path).free
    free_gb = free_bytes / (1024**3)

    logger.info(f"Available space: {free_gb:.1f} GB")
    logger.info(f"Required space: {required_gb:.1f} GB")

    return free_gb >= required_gb

def download_model(model_key: str, model_config: Dict, base_path: Path) -> bool:
    """Download a single model"""
    repo_id = model_config["repo_id"]
    model_path = base_path / model_key

    logger.info(f"Downloading {model_key}: {repo_id}")
    logger.info(f"Description: {model_config['description']}")
    logger.info(f"Estimated size: {model_config['size_gb']} GB")

    try:
        # Create model directory
        model_path.mkdir(parents=True, exist_ok=True)

        # Download model
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            resume_download=True
        )

        logger.info(f"Successfully downloaded {model_key} to {model_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {model_key}: {e}")
        return False

def create_model_manifest(base_path: Path, downloaded_models: List[str]):
    """Create a manifest file listing downloaded models"""
    manifest_path = base_path / "model_manifest.json"

    import json
    from datetime import datetime

    manifest = {
        "created_at": datetime.now().isoformat(),
        "download_location": str(base_path),
        "models": {}
    }

    for model_key in downloaded_models:
        if model_key in MODELS:
            config = MODELS[model_key]
            model_path = base_path / model_key

            manifest["models"][model_key] = {
                "repo_id": config["repo_id"],
                "local_path": str(model_path),
                "description": config["description"],
                "domain": config["domain"],
                "size_gb": config["size_gb"],
                "downloaded": model_path.exists()
            }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Created model manifest: {manifest_path}")

def main():
    """Main download function"""
    parser = argparse.ArgumentParser(description="Download Agent Forge evolution models")
    parser.add_argument("--models-dir", default="D:/agent_forge_models",
                       help="Directory to store downloaded models")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()) + ["all"],
                       default=["all"], help="Models to download")
    parser.add_argument("--check-space", action="store_true",
                       help="Check disk space before downloading")

    args = parser.parse_args()

    # Setup paths
    base_path = Path(args.models_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Determine which models to download
    if "all" in args.models:
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = args.models

    # Calculate total size
    total_size_gb = sum(MODELS[m]["size_gb"] for m in models_to_download)

    logger.info(f"Planning to download {len(models_to_download)} models")
    logger.info(f"Total estimated size: {total_size_gb:.1f} GB")

    # Check disk space if requested
    if args.check_space or total_size_gb > 5:  # Auto-check for large downloads
        if not check_disk_space(base_path, total_size_gb * 1.2):  # 20% buffer
            logger.error("Insufficient disk space!")
            return 1

    # Download models
    downloaded_models = []
    failed_models = []

    for model_key in models_to_download:
        if model_key not in MODELS:
            logger.warning(f"Unknown model: {model_key}")
            continue

        success = download_model(model_key, MODELS[model_key], base_path)
        if success:
            downloaded_models.append(model_key)
        else:
            failed_models.append(model_key)

    # Create manifest
    if downloaded_models:
        create_model_manifest(base_path, downloaded_models)

    # Report results
    logger.info("=" * 50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"Successfully downloaded: {len(downloaded_models)}")
    for model in downloaded_models:
        logger.info(f"  ✓ {model}: {MODELS[model]['repo_id']}")

    if failed_models:
        logger.error(f"Failed downloads: {len(failed_models)}")
        for model in failed_models:
            logger.error(f"  ✗ {model}: {MODELS[model]['repo_id']}")

    logger.info(f"Models stored in: {base_path}")
    logger.info("=" * 50)

    return 0 if not failed_models else 1

if __name__ == "__main__":
    sys.exit(main())
