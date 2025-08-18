"""Monitor Agent Forge model downloads and show progress."""

from datetime import datetime
from pathlib import Path


def monitor_downloads():
    """Monitor download progress."""

    base_dir = Path("D:/AIVillage/models")

    if not base_dir.exists():
        print("Download directory not found - downloads may not have started yet")
        return

    print("AGENT FORGE DOWNLOAD MONITOR")
    print("=" * 40)
    print(f"Monitoring: {base_dir}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print()

    # Check what's being downloaded
    model_dirs = list(base_dir.iterdir())

    if not model_dirs:
        print("No models found - downloads may be starting...")
        return

    print(f"Models in progress: {len(model_dirs)}")
    print()

    total_size = 0

    for model_dir in model_dirs:
        if model_dir.is_dir():
            # Calculate current size
            files = list(model_dir.rglob("*"))
            file_count = len([f for f in files if f.is_file()])

            if file_count > 0:
                size = sum(f.stat().st_size for f in files if f.is_file())
                size_mb = size / (1024**2)
                total_size += size_mb

                # Check for key model files
                has_model = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
                has_config = (model_dir / "config.json").exists()
                has_tokenizer = any(model_dir.glob("tokenizer*"))

                status = "COMPLETE" if (has_model and has_config) else "DOWNLOADING"

                print(f"  {model_dir.name}:")
                print(f"    Status: {status}")
                print(f"    Size: {size_mb:.1f} MB")
                print(f"    Files: {file_count}")
                print(f"    Model files: {'✓' if has_model else '✗'}")
                print(f"    Config: {'✓' if has_config else '✗'}")
                print(f"    Tokenizer: {'✓' if has_tokenizer else '✗'}")
                print()
            else:
                print(f"  {model_dir.name}: Starting...")

    print(f"Total downloaded so far: {total_size:.1f} MB ({total_size / 1024:.2f} GB)")

    # Estimate completion
    expected_models = [
        "Qwen_Qwen2.5-Coder-1.5B-Instruct",
        "Qwen_Qwen2-1.5B",
        "microsoft_phi-1_5",
    ]

    completed = sum(
        1 for name in expected_models if (base_dir / name).exists() and any((base_dir / name).glob("*.safetensors"))
    )

    print(f"\nProgress: {completed}/3 models completed")

    if completed == 3:
        print("🎉 ALL MODELS DOWNLOADED!")
        print("EvoMerge evolution should start automatically...")
    else:
        remaining = 3 - completed
        print(f"Remaining: {remaining} models")
        print("Estimated download size per model: ~3-4 GB")
        print(f"Total remaining: ~{remaining * 3.5:.1f} GB")


if __name__ == "__main__":
    monitor_downloads()
