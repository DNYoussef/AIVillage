import click
import yaml
import traceback
from agent_forge.evomerge.config import MergeKitConfig
from agent_forge.evomerge.merger import MergeKitMerger
from agent_forge.evomerge.utils import load_models, save_model

@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def main(config: str, output: str):
    try:
        click.echo(f"Loading configuration from {config}")
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)
        
        click.echo("Configuration:")
        click.echo(yaml.dump(config_dict, default_flow_style=False))
        
        click.echo("Creating MergeKitConfig")
        merge_config = MergeKitConfig(**config_dict)
        
        click.echo("Initializing MergeKitMerger")
        merger = MergeKitMerger(merge_config)
        
        click.echo("Loading models")
        models = load_models(merge_config.models)
        
        click.echo("Merging models")
        merged_model = merger.merge(models)
        
        click.echo(f"Saving merged model to {output}")
        save_result = save_model(merged_model, output)
        
        if save_result:
            click.echo(f"Model successfully merged and saved to {output}")
        else:
            click.echo("Failed to save the merged model")
    except Exception as e:
        click.echo(f"An error occurred: {str(e)}")
        click.echo("Traceback:")
        click.echo(traceback.format_exc())
if __name__ == "__main__":
    main()