#!/usr/bin/env python3
"""Targeted search for specific 1.5B models for Magi agent"""

import json

from huggingface_hub import HfApi, model_info
from huggingface_hub.utils import HfHubHTTPError


def get_known_candidates():
    """Return list of known good 1.5B parameter models by category"""
    return {
        "coding": [
            "Qwen/Qwen2.5-Coder-1.5B",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            "HuggingFaceTB/SmolLM-1.7B",
            "microsoft/DialoGPT-large",
        ],
        "math": [
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "Qwen/Qwen2-Math-1.5B-Instruct",
            "microsoft/DialoGPT-medium",
            "deepseek-ai/DeepSeek-Math-1.3B",
            "microsoft/DialoGPT-small",
        ],
        "tools": [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "microsoft/DialoGPT-large",
            "HuggingFaceTB/SmolLM-1.7B-Instruct",
        ],
    }


def analyze_specific_model(repo_id: str) -> dict:
    """Get detailed info for a specific model"""
    HfApi()

    try:
        model = model_info(repo_id, files_metadata=True)

        details = {
            "repo_id": repo_id,
            "exists": True,
            "downloads": getattr(model, "downloads", 0),
            "license": "unknown",
            "tags": getattr(model, "tags", []),
            "files": [],
            "has_safetensors": False,
            "has_tokenizer": False,
            "has_config": False,
            "size_gb": 0,
            "last_modified": str(getattr(model, "last_modified", "unknown")),
        }

        # Get license
        card_data = getattr(model, "card_data", {}) or {}
        details["license"] = card_data.get("license", "unknown")

        # Check files
        total_size = 0
        if hasattr(model, "siblings") and model.siblings:
            for file_info in model.siblings:
                filename = file_info.rfilename
                details["files"].append(filename)

                if hasattr(file_info, "size") and file_info.size:
                    total_size += file_info.size

                if filename.endswith((".safetensors", ".bin")):
                    details["has_safetensors"] = True
                elif "tokenizer" in filename.lower():
                    details["has_tokenizer"] = True
                elif filename == "config.json":
                    details["has_config"] = True

        details["size_gb"] = round(total_size / (1024**3), 2)

        return details

    except HfHubHTTPError as e:
        return {"repo_id": repo_id, "exists": False, "error": str(e)}
    except Exception as e:
        return {"repo_id": repo_id, "exists": False, "error": f"Unknown error: {e!s}"}


def main():
    """Main targeted search"""
    print("Performing targeted search for known 1.5B models...")

    candidates = get_known_candidates()
    results = {}

    for category, model_list in candidates.items():
        print(f"\n=== {category.upper()} MODELS ===")
        category_results = []

        for repo_id in model_list:
            print(f"Checking {repo_id}...")
            details = analyze_specific_model(repo_id)
            category_results.append(details)

            if details["exists"]:
                print(f"  OK {details['downloads']} downloads, {details['license']} license, {details['size_gb']}GB")
                print(
                    f"    Files: safetensors={details['has_safetensors']}, tokenizer={details['has_tokenizer']}, config={details['has_config']}"
                )
            else:
                print(f"  ERROR {details.get('error', 'Not found')}")

        results[category] = category_results

    # Save results
    with open("targeted_candidates.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nSaved analysis to targeted_candidates.json")

    # Print recommendations
    print("\n=== TOP RECOMMENDATIONS ===")
    for category, models in results.items():
        working_models = [m for m in models if m.get("exists", False) and m.get("license") in ["apache-2.0", "mit"]]
        if working_models:
            best = max(working_models, key=lambda x: x.get("downloads", 0))
            print(
                f"{category.upper()}: {best['repo_id']} ({best['downloads']} downloads, {best['license']}, {best['size_gb']}GB)"
            )


if __name__ == "__main__":
    main()
