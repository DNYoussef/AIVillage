import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
from .config import create_default_config, Configuration, ModelReference
from .merger import AdvancedModelMerger
from .utils import load_models
from transformers import AutoModelForCausalLM, AutoTokenizer

def is_local_path(path):
    path = os.path.expanduser(path)
    path = os.path.normpath(path)
    return os.path.isdir(path)

def download_and_merge_models(model_paths, verbose=False):
    logger = logging.getLogger(__name__)
    logger.info("Starting model download and merge process")

    config = create_default_config()
    config.models = [ModelReference(name=f"model{i+1}", path=path) for i, path in enumerate(model_paths)]

    logger.info("Loading models")
    models = load_models(config.models)
    logger.info(f"Loaded {len(models)} models successfully")

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

    logger.info(f"Created {len(merged_models)} merged models")
    return merged_models

def main():
    print("Starting main function")  # Debug print
    try:
        parser = argparse.ArgumentParser(description="EvoMerge: Evolutionary Model Merging System")
        parser.add_argument("--download-and-merge", action="store_true", help="Download models and create 8 merged models")
        parser.add_argument("--model1", type=str, required=True, help="Local path or Hugging Face model ID for the first model")
        parser.add_argument("--model2", type=str, required=True, help="Local path or Hugging Face model ID for the second model")
        parser.add_argument("--model3", type=str, help="Local path or Hugging Face model ID for the third model")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        args = parser.parse_args()

        print("Setting up logging")  # Debug print
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            stream=sys.stdout)  # Ensure logging to stdout
        logger = logging.getLogger(__name__)

        logger.info("Starting EvoMerge CLI")

        if args.download_and_merge:
            model_paths = [args.model1, args.model2]
            if args.model3:
                model_paths.append(args.model3)
            
            merged_models = download_and_merge_models(model_paths, args.verbose)
            logger.info(f"Created {len(merged_models)} merged models:")
            for model_path in merged_models:
                logger.info(model_path)
        else:
            logger.info("No action specified. Use --download-and-merge to download models and create merged models.")
            parser.print_help()

        logger.info("EvoMerge CLI completed successfully")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
