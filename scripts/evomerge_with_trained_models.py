#!/usr/bin/env python3
"""
EvoMerge with Trained Cognate Models
=====================================

This script properly uses the EvoMerge phase's built-in functionality to:
1. Load the 3 trained Cognate models
2. Save them in a format EvoMerge expects
3. Let EvoMerge create 8 merged models using different techniques
4. Run benchmarks on all models for 50 generations
"""

import os
import sys
import torch
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import shutil

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import EvoMerge components
from core.agent_forge.phases.evomerge import EvoMergePhase
from core.agent_forge.phases.phase2_evomerge.config import EvoMergeConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_models_for_evomerge():
    """
    Prepare the 3 trained Cognate models for EvoMerge.
    EvoMerge expects model files in a specific format with config.json
    """
    logger.info("Preparing trained models for EvoMerge...")

    # Create temp directory for EvoMerge input models
    temp_model_dir = project_root / "models" / "evomerge_input"
    temp_model_dir.mkdir(parents=True, exist_ok=True)

    model_paths = []

    for i in range(1, 4):
        source_path = project_root / f"models/cognate_real_data_model_{i}/model.pt"

        if not source_path.exists():
            raise FileNotFoundError(f"Model {i} not found at {source_path}")

        # Create directory for this model
        model_dir = temp_model_dir / f"cognate_model_{i}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Load the model checkpoint
        logger.info(f"Loading Model {i} from {source_path}")
        checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)

        # Save the model weights in format EvoMerge expects
        model_file = model_dir / "pytorch_model.bin"
        torch.save(checkpoint['model_state_dict'], model_file)

        # Create a config.json for the model
        config = checkpoint.get('config', None)

        # Handle TitansConfig object (has attributes, not dict)
        if config is not None and hasattr(config, 'd_model'):
            # Get actual dimensions from the config
            hidden_size = getattr(config, 'd_model', 216)  # Our models use 216
            num_heads = getattr(config, 'n_heads', 8)

            # Use llama model type since EvoMerge doesn't recognize cognate
            config_data = {
                "model_type": "llama",  # Changed from cognate to llama
                "hidden_size": hidden_size,  # Use actual hidden size (216)
                "num_hidden_layers": getattr(config, 'n_layers', 12),
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_heads,  # Llama needs this
                "vocab_size": getattr(config, 'vocab_size', 32000),
                "max_position_embeddings": getattr(config, 'max_seq_len', 1024),
                "intermediate_size": getattr(config, 'd_ff', 2048),
                "hidden_act": "silu",  # Llama uses SwiGLU
                "rms_norm_eps": 1e-6,  # Llama uses RMSNorm
                "initializer_range": 0.02,
                "use_cache": True,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "tie_word_embeddings": False,  # Llama doesn't tie embeddings
                "rope_theta": 10000.0,
                "rope_scaling": None,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "architectures": ["LlamaForCausalLM"]  # Standard Llama architecture
            }
        else:
            # Default config if none found - use standard tiny Llama config
            config_data = {
                "model_type": "llama",
                "hidden_size": 216,  # Match our actual model size
                "num_hidden_layers": 12,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "max_position_embeddings": 1024,
                "intermediate_size": 2048,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "initializer_range": 0.02,
                "use_cache": True,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "tie_word_embeddings": False,
                "rope_theta": 10000.0,
                "rope_scaling": None,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "architectures": ["LlamaForCausalLM"]
            }

        config_file = model_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        # Also save tokenizer config (minimal)
        max_length = getattr(config, 'max_seq_len', 1024) if config else 1024
        tokenizer_config = {
            "model_max_length": max_length,
            "tokenizer_class": "GPT2Tokenizer",
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>"
        }

        tokenizer_file = model_dir / "tokenizer_config.json"
        with open(tokenizer_file, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)

        model_paths.append(str(model_dir))
        logger.info(f"  Model {i} prepared at {model_dir}")

    return model_paths


async def run_evomerge_evolution(model_paths: List[str], generations: int = 50):
    """
    Run the EvoMerge evolution using the phase's built-in functionality.
    This will:
    1. Create 8 models from the 3 input models using merge techniques
    2. Evaluate each model on benchmarks every generation
    3. Evolve the population over 50 generations
    """

    logger.info(f"Starting {generations}-generation EvoMerge evolution")
    logger.info(f"Input models: {len(model_paths)}")
    logger.info("EvoMerge will create 8 models using different merge techniques:")
    logger.info("  - linear, slerp, ties, dare, frankenmerge, dfs")

    # Configure EvoMerge
    config = EvoMergeConfig(
        # Evolution parameters
        generations=generations,
        population_size=8,  # Will create 8 models from the 3 input models
        elite_size=2,
        tournament_size=3,

        # Genetic operation rates
        mutation_rate=0.1,
        crossover_rate=0.7,
        mutation_strength=0.05,

        # Merge techniques to use
        techniques=['linear', 'slerp', 'ties', 'dare', 'frankenmerge', 'dfs'],

        # Fitness evaluation - benchmarks to run
        fitness_weights={
            'perplexity': 0.3,
            'gsm8k': 0.3,      # Math problems
            'humaneval': 0.2,   # Code generation
            'hellaswag': 0.1,   # Common sense
            'arc': 0.1          # Reasoning
        },

        # Convergence and diversity
        convergence_threshold=0.001,
        convergence_patience=5,
        early_stopping=True,
        diversity_weight=0.3,
        min_diversity=0.2,

        # Performance settings
        enable_parallel=True,
        num_workers=4,
        enable_caching=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision=True,
        gradient_checkpointing=False,  # Not needed for inference

        # Checkpointing
        checkpoint_interval=10,
        checkpoint_dir=str(project_root / "checkpoints" / "evomerge"),
        keep_checkpoints=3,

        # Logging
        log_level="INFO",
        log_interval=1,
        wandb_project=None  # Disable wandb for now
    )

    # Add output directory
    config.output_dir = str(project_root / "outputs" / "evomerge")

    # Create EvoMerge phase
    evomerge = EvoMergePhase(config=config)

    logger.info("Running EvoMerge evolution...")
    logger.info("This will:")
    logger.info(f"  - Create 8 merged models using 6 different techniques")
    logger.info(f"  - Run 5 benchmarks on each model")
    logger.info(f"  - Total evaluations: {8 * generations} = {8 * generations} model evaluations")
    logger.info(f"  - Each evaluation runs: GSM8K, HumanEval, HellaSwag, ARC, Perplexity")

    # Run the evolution - EvoMerge handles everything internally
    try:
        best_model = await evomerge.run(model_paths)

        logger.info("\n=== EVOLUTION COMPLETE ===")
        logger.info(f"Best model saved to: {config.output_dir}")

        # Get final statistics
        if hasattr(evomerge, 'evolution_history'):
            history = evomerge.evolution_history
            logger.info(f"Final generation: {len(history)}")
            logger.info(f"Best fitness achieved: {history[-1]['best_fitness']:.4f}")
            logger.info(f"Final diversity: {history[-1]['diversity']:.4f}")

            # Save evolution history
            history_file = Path(config.output_dir) / "evolution_history.json"
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Evolution history saved to: {history_file}")

        return best_model

    except Exception as e:
        logger.error(f"Error during EvoMerge evolution: {e}")
        import traceback
        traceback.print_exc()
        raise


def cleanup_temp_models(model_paths: List[str]):
    """Clean up temporary model directories"""
    for path in model_paths:
        if Path(path).exists() and "evomerge_input" in path:
            try:
                shutil.rmtree(path)
                logger.info(f"Cleaned up: {path}")
            except Exception as e:
                logger.warning(f"Could not clean up {path}: {e}")


async def main():
    """Main function to run the complete EvoMerge pipeline"""

    logger.info("=== EVOMERGE WITH TRAINED COGNATE MODELS ===\n")

    # Step 1: Prepare models for EvoMerge
    logger.info("Step 1: Preparing trained models for EvoMerge...")
    model_paths = prepare_models_for_evomerge()
    logger.info(f"Prepared {len(model_paths)} models\n")

    # Step 2: Run EvoMerge evolution
    logger.info("Step 2: Running 50-generation EvoMerge evolution...")
    logger.info("EvoMerge will now:")
    logger.info("  1. Create 8 initial models using merge techniques")
    logger.info("  2. Evaluate each on benchmarks (GSM8K, HumanEval, etc.)")
    logger.info("  3. Evolve the population for 50 generations")
    logger.info("  4. Save the best model at the end\n")

    try:
        best_model = await run_evomerge_evolution(model_paths, generations=50)

        logger.info("\n=== SUCCESS ===")
        logger.info("EvoMerge has completed successfully!")
        logger.info("The best evolved model combines strengths from all 3 trained models")
        logger.info("Check outputs/evomerge/ for results and checkpoints/evomerge/ for saved models")

    finally:
        # Clean up temporary model directories
        logger.info("\nCleaning up temporary files...")
        cleanup_temp_models(model_paths)

    return best_model


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())