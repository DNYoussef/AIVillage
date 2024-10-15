import argparse
import json
import logging
import os
import sys
import subprocess
from tqdm import tqdm
from .config import create_default_config, Configuration, ModelReference
from .merger import AdvancedModelMerger
from .utils import load_models, EvoMergeException, check_system_resources
from transformers import AutoModelForCausalLM, AutoTokenizer
from .logging_config import setup_logging

def is_local_path(path):
    path = os.path.expanduser(path)
    path = os.path.normpath(path)
    return os.path.isdir(path)

def download_model_with_cli(model_path):
    try:
        subprocess.run(["huggingface-cli", "download", model_path], check=True)
    except subprocess.CalledProcessError as e:
        raise EvoMergeException(f"Failed to download model {model_path} using Hugging Face CLI: {str(e)}")

def model_exists_in_cache(model_path):
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", f"models--{model_path.replace('/', '--')}")
    return os.path.exists(cache_dir)

def download_and_merge_models(model_paths, use_cli=False, verbose=False):
    logger = logging.getLogger(__name__)
    logger.info(f"Attempting to download and merge models: {model_paths}")

    config = create_default_config()
    config.models = [ModelReference(name=f"model{i+1}", path=path) for i, path in enumerate(model_paths)]

    if use_cli:
        for model_ref in config.models:
            if not model_exists_in_cache(model_ref.path):
                logger.info(f"Downloading model {model_ref.name} using Hugging Face CLI")
                download_model_with_cli(model_ref.path)
            else:
                logger.info(f"Model {model_ref.name} already exists in cache, skipping download")
            
            # Update the model path to point to the downloaded snapshot directory
            model_ref.path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", f"models--{model_ref.path.replace('/', '--')}", "snapshots")

    # Log model paths
    for model_ref in config.models:
        logger.info(f"Model {model_ref.name} path: {model_ref.path}")
        if os.path.exists(model_ref.path):
            logger.info(f"Path exists for {model_ref.name}")
            logger.info(f"Contents of {model_ref.path}:")
            for item in os.listdir(model_ref.path):
                logger.info(f"  {item}")
        else:
            logger.warning(f"Path does not exist for {model_ref.name}")

    # Check system resources before loading models
    check_system_resources(model_paths)

    logger.info("Loading models")
    try:
        models = load_models(config.models)
        logger.info(f"Loaded {len(models)} models successfully")
    except EvoMergeException as e:
        logger.error(f"Failed to load models: {str(e)}")
        return []

    merger = AdvancedModelMerger(config)
    
    merge_combinations = [
        ["linear", "ties", "frankenmerge"],
        ["linear", "ties", "dfs"],
        ["linear", "dare", "frankenmerge"],
        ["linear", "dare", "dfs"],
        ["slerp", "ties", "frankenmerge"],
        ["slerp", "ties", "dfs"],
        ["slerp", "dare", "frankenmerge"],
        ["slerp", "dare", "dfs"]
    ]

    merged_models = []
    for i, techniques in enumerate(tqdm(merge_combinations, desc="Creating merged models")):
        logger.info(f"Creating merged model {i+1} with techniques: {techniques}")
        config.merge_settings.ps_techniques = techniques[:2]
        config.merge_settings.dfs_techniques = [techniques[2]]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                merged_model_path = merger.merge()
                merged_models.append(merged_model_path)
                logger.info(f"Successfully created merged model: {merged_model_path}")
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed to create merged model with techniques {techniques}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to create merged model after {max_retries} attempts")

    logger.info(f"Finished downloading and merging models. Created {len(merged_models)} merged models.")
    return merged_models

def main():
    print("Starting main function")  # Debug print
    try:
        parser = argparse.ArgumentParser(description="EvoMerge: Evolutionary Model Merging System")
        parser.add_argument("--download-and-merge", action="store_true", help="Download models and create 8 merged models")
        parser.add_argument("--model1", type=str, required=True, help="Local path or Hugging Face model ID for the first model")
        parser.add_argument("--model2", type=str, required=True, help="Local path or Hugging Face model ID for the second model")
        parser.add_argument("--model3", type=str, help="Local path or Hugging Face model ID for the third model")
        parser.add_argument("--use-cli", action="store_true", help="Use Hugging Face CLI to download models")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        args = parser.parse_args()

        print("Setting up logging")  # Debug print
        logger = setup_logging(log_file='evomerge.log')
        logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

        logger.info("Starting EvoMerge CLI")

        if args.download_and_merge:
            model_paths = [args.model1, args.model2]
            if args.model3:
                model_paths.append(args.model3)
            
            logger.info(f"Model paths: {model_paths}")
            merged_models = download_and_merge_models(model_paths, args.use_cli, args.verbose)
            logger.info(f"Created {len(merged_models)} merged models:")
            for model_path in merged_models:
                logger.info(model_path)
        else:
            logger.info("No action specified. Use --download-and-merge to download models and create merged models.")
            parser.print_help()

        logger.info("EvoMerge CLI completed successfully")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

