#!/usr/bin/env python3
"""Helper CLI to display information about selected Magi seed models"""

import json
from pathlib import Path


def load_seed_manifest():
    """Load the seed manifest"""
    manifest_path = Path("../models/seeds/magi/seed_manifest.json")
    if not manifest_path.exists():
        print("ERROR: Seed manifest not found")
        return None

    with open(manifest_path) as f:
        return json.load(f)


def print_seed_info():
    """Print formatted seed information"""
    manifest = load_seed_manifest()
    if not manifest:
        return

    print("MAGI AGENT SEED MODELS")
    print("=" * 50)

    models = manifest["magi_seed_models"]["models"]

    for model in models:
        category = model["category"].upper()
        repo_id = model["repo_id"]
        status = model["download_status"]

        print(f"\n{category} MODEL:")
        print(f"  Repository: {repo_id}")
        print(f"  License: {model['license']}")
        print(f"  Parameters: {model['param_estimate_b']}B")
        print(f"  Context Length: {model['context_length']:,} tokens")
        print(f"  Download Status: {status}")
        print(f"  Path: {model['path']}")
        print(f"  Notes: {model['notes']}")

    print("\nSUMMARY:")
    print(f"Total Models: {len(models)}")
    ready_count = sum(1 for m in models if m["download_status"] == "complete")
    print(f"Ready for EvoMerge: {ready_count}/{len(models)}")

    if ready_count == len(models):
        print("MAGI_SEEDS_READY: All models available!")
    else:
        print("Download completion needed")


def print_download_commands():
    """Print commands to complete downloads"""
    print("\nDOWNLOAD COMMANDS:")
    print("To complete downloads, run:")
    print(
        "  huggingface-cli download Qwen/Qwen2.5-Coder-1.5B --local-dir models/seeds/magi/coding/Qwen/Qwen2.5-Coder-1.5B"
    )
    print(
        "  huggingface-cli download Qwen/Qwen2.5-Math-1.5B-Instruct --local-dir models/seeds/magi/math/Qwen/Qwen2.5-Math-1.5B-Instruct"
    )
    print(
        "  huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/seeds/magi/tools/Qwen/Qwen2.5-1.5B-Instruct"
    )


def print_evomerge_command():
    """Print EvoMerge initiation command"""
    print("\nEVOMERGE PIPELINE:")
    print("After validation, start evolution with:")
    print("  python -m agent_forge.evolution.evomerge --seeds magi --generations 50")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "download":
            print_download_commands()
        elif command == "evomerge":
            print_evomerge_command()
        else:
            print("Unknown command. Use: info, download, or evomerge")
    else:
        print_seed_info()
