import os
import numpy as np
from typing import List, Dict, Tuple
import requests
from .config import MergeKitConfig
from .gguf_utils import GGUFReader, GGUFWriter

class MergeKitMerger:
    def __init__(self, config: MergeKitConfig):
        self.config = config
        self.ollama_dir = os.path.expanduser("~/.ollama/models")

    def merge(self, models: List[Dict]) -> Dict:
        print("Starting actual model merger")
        if self.config.merge_method == "linear":
            try:
                merged_model = self._linear_merge(models)
                print("Model merging completed successfully")
                return merged_model
            except Exception as e:
                        raise RuntimeError(f"Error during linear merge: {str(e)}")
        else:
                    raise NotImplementedError(f"Merge method {self.config.merge_method} not implemented")

    def _linear_merge(self, models: List[Dict]) -> Dict:
        weights = self.config.parameters.get("weights", [1/len(models)] * len(models))
        
        # Load and merge weights
        merged_weights, merged_metadata = self._load_and_merge_weights(models, weights)
        
        # Create new model
        merged_model_name = '_'.join(model['name'] for model in models)
        new_model_path = self._save_merged_model(merged_weights, merged_metadata, merged_model_name)
        
        # Create Ollama model
        self._create_ollama_model(new_model_path, merged_model_name)
        
        return {
            'name': merged_model_name,
            'path': new_model_path
        }

    def _load_and_merge_weights(self, models: List[Dict], weights: List[float]) -> Tuple[Dict[str, np.ndarray], Dict]:
        merged_weights = {}
        merged_metadata = {}
        for i, model in enumerate(models):
            model_path = os.path.join(self.ollama_dir, "blobs", f"{model['name']}.gguf")
            reader = GGUFReader(model_path)
            reader.read()
            
            if i == 0:
                merged_metadata = reader.metadata.copy()
                for name, tensor in reader.tensors.items():
                    merged_weights[name] = tensor * weights[i]
            else:
                for name, tensor in reader.tensors.items():
                    if name in merged_weights:
                        merged_weights[name] += tensor * weights[i]
                    else:
                        print(f"Warning: Tensor {name} not found in base model, skipping")
        
        return merged_weights, merged_metadata

    def _save_merged_model(self, weights: Dict[str, np.ndarray], metadata: Dict, model_name: str) -> str:
        new_model_path = os.path.join(self.ollama_dir, "blobs", f"{model_name}.gguf")
        writer = GGUFWriter(new_model_path)
        
        # Write metadata
        for key, value in metadata.items():
            writer.add_metadata(key, value)
        
        # Write merged weights
        for name, tensor in weights.items():
            writer.add_tensor(name, tensor)
        
        writer.write()
        return new_model_path

    def _create_ollama_modelfile(self, model_path: str, model_name: str) -> str:
        modelfile = f"FROM {model_path}\n\n"
        modelfile += f'SYSTEM "This is a merged model named {model_name}, combining multiple models using linear interpolation of weights."\n'
        return modelfile

    def _create_ollama_model(self, model_path: str, model_name: str) -> None:
        modelfile = self._create_ollama_modelfile(model_path, model_name)
        response = requests.post('http://localhost:11434/api/create', json={
            "name": model_name,
            "modelfile": modelfile
        })
        if response.status_code != 200:
            raise Exception(f"Failed to create Ollama model: {response.text}")