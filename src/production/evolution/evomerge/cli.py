import argparse
import logging

from .config import ModelReference, create_default_config
from .logging_config import setup_logging
from .merging.merger import AdvancedModelMerger
from .utils import EvoMergeException, check_system_resources, load_models


def download_and_merge_models(
    model_paths,
    merge_techniques,
    weight_mask_rate,
    use_weight_rescale,
    mask_strategy,
    use_disk_based_merge,
    chunk_size,
    use_cli=False,
    verbose=False,
):
    logger = logging.getLogger(__name__)
    logger.info(f"Attempting to download and merge models: {model_paths}")

    config = create_default_config()
    config.models = [ModelReference(name=f"model{i + 1}", path=path) for i, path in enumerate(model_paths)]

    # Update merge settings
    config.merge_settings.ps_techniques = merge_techniques[:2]
    config.merge_settings.dfs_techniques = [merge_techniques[2]]
    config.merge_settings.weight_mask_rate = weight_mask_rate
    config.merge_settings.use_weight_rescale = use_weight_rescale
    config.merge_settings.mask_strategy = mask_strategy
    config.merge_settings.use_disk_based_merge = use_disk_based_merge
    config.merge_settings.chunk_size = chunk_size

    # Check system resources before loading models
    check_system_resources([model_ref.path for model_ref in config.models])

    logger.info("Loading models")
    try:
        models = load_models(config.models)
        logger.info(f"Loaded {len(models)} models successfully")
    except EvoMergeException as e:
        logger.exception(f"Failed to load models: {e!s}")
        return []

    if not models:
        logger.error("No models were successfully loaded. Aborting merge process.")
        return []

    merger = AdvancedModelMerger(config)

    merged_models = []
    max_retries = 3
    for attempt in range(max_retries):
        try:
            merged_model_path = merger.merge()
            merged_models.append(merged_model_path)
            logger.info(f"Successfully created merged model: {merged_model_path}")
            break
        except Exception as e:
            logger.exception(f"Attempt {attempt + 1} failed to create merged model: {e!s}")
            if attempt == max_retries - 1:
                logger.exception(f"Failed to create merged model after {max_retries} attempts")

    logger.info(f"Finished downloading and merging models. Created {len(merged_models)} merged models.")
    return merged_models


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoMerge: Evolutionary Model Merging System")
    parser.add_argument(
        "--download-and-merge",
        action="store_true",
        help="Download models and create merged model",
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Hugging Face model ID for the first model",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Hugging Face model ID for the second model",
    )
    parser.add_argument("--model3", type=str, help="Hugging Face model ID for the third model")
    parser.add_argument(
        "--ps-technique1",
        type=str,
        default="linear",
        choices=["linear", "slerp", "ties", "dare"],
        help="First parameter space merging technique",
    )
    parser.add_argument(
        "--ps-technique2",
        type=str,
        default="ties",
        choices=["linear", "slerp", "ties", "dare"],
        help="Second parameter space merging technique",
    )
    parser.add_argument(
        "--dfs-technique",
        type=str,
        default="frankenmerge",
        choices=["frankenmerge", "dfs"],
        help="Deep fusion space merging technique",
    )
    parser.add_argument(
        "--weight-mask-rate",
        type=float,
        default=0.0,
        help="Weight mask rate (0.0 to 1.0)",
    )
    parser.add_argument("--use-weight-rescale", action="store_true", help="Use weight rescaling")
    parser.add_argument(
        "--mask-strategy",
        type=str,
        default="random",
        choices=["random", "magnitude"],
        help="Mask strategy",
    )
    parser.add_argument(
        "--use-disk-based-merge",
        action="store_true",
        help="Use disk-based merging to save RAM",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000000,
        help="Chunk size for disk-based operations",
    )
    parser.add_argument("--use-cli", action="store_true", help="Use Hugging Face CLI to download models")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    logger = setup_logging(log_file="evomerge.log")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    logger.info("Starting EvoMerge CLI")

    if args.download_and_merge:
        model_paths = [args.model1, args.model2]
        if args.model3:
            model_paths.append(args.model3)

        merge_techniques = [args.ps_technique1, args.ps_technique2, args.dfs_technique]

        logger.info(f"Model paths: {model_paths}")
        logger.info(f"Merge techniques: {merge_techniques}")
        logger.info(f"Weight mask rate: {args.weight_mask_rate}")
        logger.info(f"Use weight rescale: {args.use_weight_rescale}")
        logger.info(f"Mask strategy: {args.mask_strategy}")
        logger.info(f"Use disk-based merge: {args.use_disk_based_merge}")
        logger.info(f"Chunk size: {args.chunk_size}")

        merged_models = download_and_merge_models(
            model_paths,
            merge_techniques,
            args.weight_mask_rate,
            args.use_weight_rescale,
            args.mask_strategy,
            args.use_disk_based_merge,
            args.chunk_size,
            args.use_cli,
            args.verbose,
        )
        logger.info(f"Created {len(merged_models)} merged models:")
        for model_path in merged_models:
            logger.info(model_path)
    else:
        logger.info("No action specified. Use --download-and-merge to download models and create merged models.")
        parser.print_help()

    logger.info("EvoMerge CLI completed successfully")


if __name__ == "__main__":
    main()
