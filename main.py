import os
import re
import json
import subprocess
from typing import List, Optional
from pydantic import BaseModel
import yaml
import click
from agent_forge.mergekit.config import MergeKitConfig
from agent_forge.mergekit.merger import MergeKitMerger
from agent_forge.mergekit.utils import load_models, save_model

def save_model(merged_model, path: str):
    print(f"Merged model metadata: {merged_model['name']}")
    print(f"Modelfile content:\n{merged_model['modelfile']}")
    
    # Ensure the path exists
    os.makedirs(path, exist_ok=True)
    
    # Sanitize the model name
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', merged_model['name'])
    
    # Create the Modelfile
    modelfile_path = os.path.join(path, f"{sanitized_name}.modelfile")
    with open(modelfile_path, "w") as f:
        f.write(merged_model['modelfile'])
    
    print(f"Modelfile saved as: {modelfile_path}")
    
    # Use Ollama CLI to create the model
    try:
        result = subprocess.run(["ollama", "create", sanitized_name, "-f", modelfile_path], 
                                capture_output=True, text=True, check=True)
        print(f"Model {sanitized_name} successfully created using Ollama CLI")
        print(f"CLI Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create model using Ollama CLI: {e}")
        print(f"CLI Error Output: {e.stderr}")
        return None

    print(f"Model {sanitized_name} successfully saved and created in Ollama")
    return True

@click.command()
@click.option('--config', required=True, help='Path to the configuration file')
@click.option('--output', required=True, help='Output path for the merged model')
def main(config: str, output: str):
    try:
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)
    
        merge_config = MergeKitConfig(**config_dict)
        merger = MergeKitMerger(merge_config)
    
        models = load_models(merge_config.models)
        merged_model = merger.merge(models)
        
        save_result = save_model(merged_model, output)
        if save_result:
            click.echo(f"Model successfully merged and saved to {output}")
        else:
            click.echo("Failed to save the merged model")
    except Exception as e:
        click.echo(f"An error occurred: {str(e)}")
        return
    
if __name__ == "__main__":
    main()