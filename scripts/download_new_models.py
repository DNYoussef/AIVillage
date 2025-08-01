#!/usr/bin/env python3
"""Download the 3 new 1.5B parameter models for 50-generation evolution merge."""

import os
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def check_gpu_availability():
    """Check GPU availability and memory."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} - {gpu_memory:.1f}GB VRAM")
        return True
    print("CUDA not available - running on CPU")
    return False


def download_model_with_retry(model_name, cache_dir, max_retries=3):
    """Download a model with retry logic."""
    print(f"\nDownloading {model_name}...")

    for attempt in range(max_retries):
        try:
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only=False,
                revision="main",
                resume_download=True,  # Resume if interrupted
                local_dir_use_symlinks=False,  # Avoid symlink issues on Windows
            )
            print(f"✓ {model_name} downloaded successfully to {model_path}")
            return model_path

        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed for {model_name}: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"✗ Failed to download {model_name} after {max_retries} attempts")
                return None


def main():
    """Main function to download the new models."""
    print("=" * 80)
    print("DOWNLOADING NEW MODELS FOR 50-GENERATION EVOLUTION MERGE")
    print("=" * 80)

    # Check GPU
    check_gpu_availability()

    # Set up directories
    models_dir = Path("D:/AgentForge/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Set cache directory
    cache_dir = models_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")

    # New 1.5B parameter models for evolution
    target_models = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "nvidia/Nemotron-4-Reasoning-Qwen-1.5B",
        "Qwen/Qwen2-1.5B-Instruct",
    ]

    print("\nTarget models for 50-generation evolution:")
    for i, model in enumerate(target_models, 1):
        print(f"  {i}. {model}")

    print(f"\nStorage location: {models_dir}")
    print(f"Cache directory: {cache_dir}")

    # Check available disk space
    import shutil

    total, used, free = shutil.disk_usage(str(models_dir))
    free_gb = free / (1024**3)
    print(f"Available disk space: {free_gb:.1f} GB")

    if free_gb < 20:
        print(
            "⚠️  Warning: Less than 20GB free space. Models may not download completely."
        )
        proceed = input("Continue anyway? (y/N): ")
        if proceed.lower() != "y":
            return None

    # Download models
    downloaded_models = []
    start_time = time.time()

    for model_name in target_models:
        model_path = download_model_with_retry(model_name, cache_dir)
        if model_path:
            downloaded_models.append((model_name, model_path))
        else:
            print(f"⚠️  Skipping {model_name} due to download failure")

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Total time: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
    print(
        f"Successfully downloaded: {len(downloaded_models)}/{len(target_models)} models"
    )

    for name, path in downloaded_models:
        print(f"  ✓ {name}")
        print(f"    → {path}")

    # Save model list for evolution script
    model_list_file = models_dir / "downloaded_models_50gen.txt"
    with open(model_list_file, "w") as f:
        f.write("# Models for 50-generation evolution merge\n")
        f.write(f"# Downloaded on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total models: {len(downloaded_models)}\n\n")

        for name, path in downloaded_models:
            f.write(f"{name}\t{path}\n")

    print(f"\nModel list saved to: {model_list_file}")

    # Validate downloads
    print("\nValidating downloads...")
    for name, path in downloaded_models:
        model_path = Path(path)
        if model_path.exists():
            # Check for essential files
            config_file = model_path / "config.json"
            if config_file.exists():
                print(f"  ✓ {name}: Configuration file found")
            else:
                print(f"  ⚠️  {name}: Configuration file missing")
        else:
            print(f"  ✗ {name}: Path does not exist")

    if len(downloaded_models) >= 2:
        print("\n🎉 Ready for 50-generation evolution merge!")
        print(f"   Models available: {len(downloaded_models)}")
        print("   Minimum required: 2 (recommended: 3)")
    else:
        print(f"\n⚠️  Warning: Only {len(downloaded_models)} models downloaded")
        print("   Evolution merge may have limited diversity")

    return downloaded_models


if __name__ == "__main__":
    main()
