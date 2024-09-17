import click
import yaml
from agent_forge.mergekit.config import MergeKitConfig
from agent_forge.mergekit.merger import MergeKitMerger
from agent_forge.mergekit.utils import load_models, save_model

@click.command()
@click.argument("config_file")
@click.argument("out_path")
def main(config_file: str, out_path: str):
    print(f"Loading configuration from {config_file}")
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
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

if __name__ == "__main__":
    main()
