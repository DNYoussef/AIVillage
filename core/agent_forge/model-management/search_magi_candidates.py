#!/usr/bin/env python3
"""Search for candidate 1.5B parameter models for Magi agent specializations"""

import json
import time

from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.utils import HfHubHTTPError


def estimate_params_from_config(config: dict) -> float | None:
    """Estimate parameters from model config"""
    try:
        if "num_parameters" in config:
            return float(config["num_parameters"]) / 1e9

        # Common estimation formulas by architecture
        if config.get("model_type") == "qwen2":
            hidden_size = config.get("hidden_size", 0)
            num_layers = config.get("num_hidden_layers", 0)
            vocab_size = config.get("vocab_size", 0)
            if hidden_size and num_layers and vocab_size:
                # Rough estimation for Qwen2
                params = (hidden_size * num_layers * 12 + vocab_size * hidden_size * 2) / 1e9
                return params

        elif config.get("model_type") in ["llama", "mistral"]:
            hidden_size = config.get("hidden_size", 0)
            num_layers = config.get("num_hidden_layers", 0)
            vocab_size = config.get("vocab_size", 0)
            intermediate_size = config.get("intermediate_size", hidden_size * 4)
            if hidden_size and num_layers and vocab_size:
                # Rough estimation for Llama/Mistral
                params = (hidden_size * intermediate_size * num_layers * 3 + vocab_size * hidden_size * 2) / 1e9
                return params

    except Exception as e:
        import logging

        logging.exception("Exception in model parameter estimation: %s", str(e))

    return None


def search_models_by_category():
    """Search for models by category using HuggingFace API"""
    api = HfApi()

    # Define search parameters for each category

    # Search for models
    try:
        # Get models with text-generation task and around 1.5B params
        models = api.list_models(
            task="text-generation",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=200,
        )

        candidates = []
        for model in models:
            try:
                # Check if it might be around 1.5B params based on name
                model_name_lower = model.id.lower()
                if any(size in model_name_lower for size in ["1.5b", "1.6b", "1.7b", "1.8b", "1.3b", "1.4b"]):
                    # Get model info
                    try:
                        model_info = api.model_info(model.id, files_metadata=True)

                        # Check license
                        license_info = getattr(model_info, "card_data", {})
                        if license_info:
                            license_name = license_info.get("license", "").lower()
                            if license_name in [
                                "apache-2.0",
                                "mit",
                                "bsd-3-clause",
                                "cc-by-4.0",
                            ]:
                                candidates.append(model_info)
                                print(f"Found candidate: {model.id}")

                    except HfHubHTTPError:
                        continue

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                print(f"Error processing {model.id}: {e}")
                continue

        print(f"Found {len(candidates)} total candidates")
        return candidates

    except Exception as e:
        print(f"API search failed: {e}")
        return []


def categorize_models(candidates: list[ModelInfo]) -> dict:
    """Categorize models by their likely specialization"""
    categorized = {"coding": [], "math": [], "tools": []}

    for model in candidates:
        model_name_lower = model.id.lower()
        tags = getattr(model, "tags", [])
        tags_str = " ".join(tags).lower()

        # Coding indicators
        if any(term in model_name_lower or term in tags_str for term in ["code", "coder", "coding", "python"]):
            categorized["coding"].append(model)

        # Math indicators
        elif any(term in model_name_lower or term in tags_str for term in ["math", "mathematics", "gsm", "reasoning"]):
            categorized["math"].append(model)

        # Tool/instruct indicators
        elif any(term in model_name_lower or term in tags_str for term in ["instruct", "function", "tool", "chat"]):
            categorized["tools"].append(model)

        # Default fallback - general instruct goes to tools
        else:
            categorized["tools"].append(model)

    return categorized


def analyze_model_details(model_info: ModelInfo) -> dict:
    """Analyze model details for candidate evaluation"""
    details = {
        "repo_id": model_info.id,
        "downloads": getattr(model_info, "downloads", 0),
        "license": "unknown",
        "tags": getattr(model_info, "tags", []),
        "files": [],
        "has_safetensors": False,
        "has_tokenizer": False,
        "has_config": False,
        "param_estimate": None,
        "context_length": "unknown",
        "last_modified": getattr(model_info, "last_modified", None),
    }

    # Get license info
    card_data = getattr(model_info, "card_data", {}) or {}
    details["license"] = card_data.get("license", "unknown")

    # Analyze files
    if hasattr(model_info, "siblings") and model_info.siblings:
        for file_info in model_info.siblings:
            filename = file_info.rfilename
            details["files"].append(filename)

            if filename.endswith(".safetensors"):
                details["has_safetensors"] = True
            elif "tokenizer" in filename:
                details["has_tokenizer"] = True
            elif filename == "config.json":
                details["has_config"] = True

    return details


def main():
    """Main search function"""
    print("Searching for Magi candidate models...")

    # Search for candidates
    candidates = search_models_by_category()

    if not candidates:
        print("No candidates found via API search")
        return

    # Categorize models
    categorized = categorize_models(candidates)

    # Analyze each category
    results = {}
    for category, models in categorized.items():
        print(f"\n=== {category.upper()} CATEGORY ===")
        category_results = []

        for model in models[:10]:  # Limit to top 10 per category
            details = analyze_model_details(model)
            category_results.append(details)
            print(f"  {details['repo_id']} - {details['downloads']} downloads - {details['license']}")

        results[category] = category_results

    # Save results
    with open("raw_candidates.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nSaved candidate analysis to tmp_models/raw_candidates.json")
    print(
        f"Found {len(results['coding'])} coding, {len(results['math'])} math, {len(results['tools'])} tool candidates"
    )


if __name__ == "__main__":
    main()
