#!/usr/bin/env python3
"""Download 3 x 1.5B parameter models for Agent Forge evolution merging."""

import os
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
    print("CUDA not available")
    return False


def download_model(model_name, cache_dir):
    """Download a model to specified directory."""
    print(f"\nDownloading {model_name}...")
    try:
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            revision="main",
        )
        print(f"✓ {model_name} downloaded to {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        return None


def main():
    """Main function to download models."""
    # Check GPU
    check_gpu_availability()

    # Set up directories
    models_dir = Path("D:/AgentForge/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Set cache directory
    cache_dir = models_dir / ".cache"
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")

    # List of 1.5B parameter models optimized for RTX 2060
    models_to_download = [
        "microsoft/phi-1_5",  # 1.3B parameters - excellent for coding
        "stabilityai/stablelm-base-alpha-3b",  # 3B but efficient
        "microsoft/DialoGPT-medium",  # 355M but very capable for chat
    ]

    # Alternative smaller models that work well together
    efficient_models = [
        "microsoft/phi-1_5",  # 1.3B - coding specialist
        "Qwen/Qwen1.5-1.8B",  # 1.8B - general purpose
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B - chat optimized
    ]

    print("Downloading efficient models for RTX 2060...")
    downloaded_models = []

    for model_name in efficient_models:
        model_path = download_model(model_name, cache_dir)
        if model_path:
            downloaded_models.append((model_name, model_path))

    print(f"\n✓ Successfully downloaded {len(downloaded_models)} models:")
    for name, path in downloaded_models:
        print(f"  - {name}: {path}")

    # Save model list for evolution script
    model_list_file = models_dir / "downloaded_models.txt"
    with open(model_list_file, "w") as f:
        for name, path in downloaded_models:
            f.write(f"{name}\t{path}\n")

    print(f"\nModel list saved to: {model_list_file}")
    return downloaded_models


if __name__ == "__main__":
    main()
