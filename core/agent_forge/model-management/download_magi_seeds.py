#!/usr/bin/env python3
"""Download and pin the selected Magi seed models"""

import json
import os
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

# Selected models based on analysis
SELECTED_MODELS = {
    "coding": "Qwen/Qwen2.5-Coder-1.5B",
    "math": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "tools": "Qwen/Qwen2.5-1.5B-Instruct",
}

# File patterns to download (avoid large unnecessary files)
ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "generation_config.json",
    "*.safetensors",
    "*.bin",
    "README.md",
    "LICENSE*",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
]


def download_and_pin_model(repo_id: str, category: str) -> dict:
    """Download model and get pinned revision info"""
    print(f"\n=== Downloading {category.upper()}: {repo_id} ===")

    try:
        # Get model info first to pin revision
        model = model_info(repo_id)
        revision = model.sha
        print(f"Pinning to revision: {revision[:8]}...")

        # Set download path
        local_path = Path(f"../models/seeds/magi/{category}/{repo_id}")
        local_path.mkdir(parents=True, exist_ok=True)

        # Download with pinned revision
        download_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_path),
            allow_patterns=ALLOW_PATTERNS,
            cache_dir=None,  # Download directly to target location
        )

        print(f"Downloaded to: {download_path}")

        # Collect download info
        downloaded_files = []
        total_size = 0

        for root, _dirs, files in os.walk(local_path):
            for file in files:
                filepath = Path(root) / file
                size = filepath.stat().st_size
                downloaded_files.append(
                    {
                        "name": file,
                        "size": size,
                        "path": str(filepath.relative_to(local_path)),
                    }
                )
                total_size += size

        return {
            "repo_id": repo_id,
            "category": category,
            "revision": revision,
            "download_path": str(local_path),
            "downloaded_files": downloaded_files,
            "total_size": total_size,
            "success": True,
        }

    except Exception as e:
        print(f"ERROR downloading {repo_id}: {e}")
        return {
            "repo_id": repo_id,
            "category": category,
            "error": str(e),
            "success": False,
        }


def main():
    """Main download function"""
    print("Downloading and pinning selected Magi seed models...")
    print(f"Selected models: {list(SELECTED_MODELS.values())}")

    results = {}
    total_downloaded = 0

    for category, repo_id in SELECTED_MODELS.items():
        result = download_and_pin_model(repo_id, category)
        results[category] = result

        if result["success"]:
            size_gb = result["total_size"] / (1024**3)
            print(f"✓ {category}: {size_gb:.2f}GB downloaded")
            total_downloaded += size_gb
        else:
            print(f"✗ {category}: Failed")

    print(f"\nTotal downloaded: {total_downloaded:.2f}GB")

    # Save download manifest
    with open("download_manifest.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Saved download manifest to download_manifest.json")

    # Print summary
    print("\n=== DOWNLOAD SUMMARY ===")
    success_count = sum(1 for r in results.values() if r["success"])
    print(f"Successfully downloaded: {success_count}/3 models")

    if success_count == 3:
        print("🎉 All Magi seed models downloaded and pinned!")
    else:
        print("⚠️  Some downloads failed. Check errors above.")

    return results


if __name__ == "__main__":
    main()
