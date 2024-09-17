from typing import List, Optional
from pydantic import BaseModel
from agent_forge.mergekit.merger import MergeKitMerger
from agent_forge.mergekit.utils import load_models, save_model
import yaml

class ModelReference(BaseModel):
    # Assuming ModelReference is defined elsewhere or needs to be defined
    pass

class MergeKitConfig(BaseModel):
    merge_method: str
    models: List[ModelReference]
    parameters: Optional[dict] = None

def main(config_file: str, out_path: str):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = MergeKitConfig(**config_dict)
    
    merger = MergeKitMerger(config)
    
    models = load_models(config.models)
    
    merged_model = merger.merge(models)
    
    save_model(merged_model, out_path)

if __name__ == "__main__":
    main()