import click
import yaml
import traceback
import logging
from agent_forge.mergekit.config import MergeKitConfig
from agent_forge.mergekit.merger import MergeKitMerger
from agent_forge.mergekit.utils import load_models, save_model, generate_text, validate_merge_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.command()
@click.argument("config_file")
@click.argument("out_path")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def main(config_file: str, out_path: str, verbose: bool):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        print(f"Loading configuration from {config_file}")
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        print("Configuration:")
        print(yaml.dump(config_dict, default_flow_style=False))
        
        print("Creating MergeKitConfig")
        config = MergeKitConfig(**config_dict)
        
        print("Validating merge configuration")
        validate_merge_config(config)
        
        print("Initializing MergeKitMerger")
        merger = MergeKitMerger(config)
        
        print("Loading models")
        models = load_models(config.models)
        
        print("Merging models")
        merged_model = merger.merge(models)
        
        print(f"Saving merged model to {out_path}")
        save_model(merged_model, out_path)
        
        print("Model merging completed successfully")
        
        if verbose:
            print("Generating sample text with merged model")
            sample_text = generate_text(merged_model['name'], "Hello, how are you?")
            print(f"Sample generated text: {sample_text}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if verbose:
            print("Traceback:")
            print(traceback.format_exc())
        else:
            print("Run with --verbose for full traceback.")

if __name__ == "__main__":
    main()