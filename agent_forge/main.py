import click
import yaml
import traceback
from agent_forge.mergekit.config import MergeKitConfig
from agent_forge.mergekit.merger import MergeKitMerger
from agent_forge.mergekit.utils import load_models, save_model, generate_text

@click.command()
@click.argument("config_file")
@click.argument("out_path")
def main(config_file: str, out_path: str):
    try:
        print(f"Loading configuration from {config_file}")
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        print("Configuration:")
        print(yaml.dump(config_dict, default_flow_style=False))
        
        print("Creating MergeKitConfig")
        config = MergeKitConfig(**config_dict)
        
        print("Initializing MergeKitMerger")
        merger = MergeKitMerger(config)
        
        print("Loading models")
        models = load_models(config.models)
        
        print("Merging models")
        merged_model = merger.merge(models)
        
        print(f"Saving merged model to {out_path}")
        save_model(merged_model, out_path)
        
        print("Model merging completed successfully")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()