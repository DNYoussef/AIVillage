import os
import traceback

import click
import torch
import wandb
import yaml

from agent_forge.adas import ADASystem
from agent_forge.bakedquietiot.deepbaking import DeepSystemBaker
from agent_forge.compression import stream_compress_model
from agent_forge.evomerge.config import MergeConfig
from agent_forge.evomerge.merger import AdvancedModelMerger
from agent_forge.model_compression.bitlinearization import BitNetModel
from agent_forge.training.training import CognitiveTrainingPipeline, EnhancedQuietSTaR


@click.command()
@click.argument("config_file")
@click.argument("output_dir")
def main(config_file: str, output_dir: str):
    try:
        # Setup wandb
        wandb.login()
        wandb.init(
            project="agent_forge",
            name="agent_forge_run",
            config={
                "config_file": config_file,
            },
        )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading configuration from {config_file}")
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)

        enable_adas = config_dict.get("enable_adas", True)

        print("Configuration:")
        print(yaml.dump(config_dict, default_flow_style=False))

        # Step 1: Merge Models using evomerge
        print("Creating MergeConfig")
        config = MergeConfig(**config_dict)

        print("Initializing AdvancedModelMerger")
        merger = AdvancedModelMerger(config)

        print("Merging models")
        merged_model_path = merger.merge()
        print(f"Merged model saved to: {merged_model_path}")

        # Log merging step
        wandb.log({"step": "model_merging", "merged_model_path": merged_model_path})

        # Step 2: Apply enhancements using bakedquietiot
        print("Initializing DeepSystemBaker")
        baker = DeepSystemBaker(merged_model_path)
        print("Deep baking the model")
        baker.deep_bake_system()
        baked_model_path = "deep_baked_model"
        print(f"Deep baked model saved to: {baked_model_path}")

        # Log deep baking step
        wandb.log({"step": "deep_baking", "baked_model_path": baked_model_path})

        # Step 3: Compress the model using model_compression
        print("Loading the deep baked model for compression")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(baked_model_path)
        print("Converting model to BitNetModel")
        bitnet_model = BitNetModel(model)

        print("Compressing the model")
        compressed_model = stream_compress_model(bitnet_model.model)
        compressed_model_path = os.path.join(output_dir, "compressed_model.pt")
        torch.save(compressed_model, compressed_model_path)
        print(f"Compressed model saved to: {compressed_model_path}")

        # Log compression step
        compressed_model_size = os.path.getsize(compressed_model_path) / (1024 * 1024)
        wandb.log(
            {
                "step": "model_compression",
                "compressed_model_size_MB": compressed_model_size,
            }
        )

        # Step 4: Train the new smaller model using training
        print("Initializing EnhancedQuietSTaR for training")
        enhanced_model = EnhancedQuietSTaR(baked_model_path)

        print("Setting up training pipeline")
        # Replace the following placeholders with your actual data loaders
        train_data = []  # Your training data
        val_data = []  # Your validation data

        pipeline = CognitiveTrainingPipeline(
            enhanced_model, train_data, val_data, num_epochs=50
        )
        print("Training the model")
        trained_model = pipeline.train()

        print("Saving the trained model")
        trained_model_path = os.path.join(output_dir, "trained_model")
        enhanced_model.model.save_pretrained(trained_model_path)
        enhanced_model.tokenizer.save_pretrained(trained_model_path)
        print(f"Trained model saved to: {trained_model_path}")

        # Log training completion
        wandb.log({"step": "model_training", "trained_model_path": trained_model_path})

        # Optional ADAS optimization
        if enable_adas:
            print("Running ADAS optimization")
            adas_system = ADASystem(trained_model_path)
            optimized_model_path = adas_system.optimize_agent_architecture(output_dir)
            print(f"ADAS optimized model saved to: {optimized_model_path}")
            wandb.log(
                {"step": "adas_optimization", "adas_model_path": optimized_model_path}
            )
        else:
            optimized_model_path = trained_model_path

        # Finish wandb run
        wandb.finish()

        print("Process completed successfully")

    except Exception as e:
        print(f"An error occurred: {e!s}")
        print("Traceback:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
