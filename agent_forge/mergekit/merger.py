from typing import List, Dict
from .config import MergeKitConfig

class MergeKitMerger:
    def __init__(self, config: MergeKitConfig):
        self.config = config

    def merge(self, models: List[Dict]) -> Dict:
        print("Debugging: Printing model information")
        for i, model in enumerate(models):
            print(f"Model {i + 1}:")
            self._print_model_info(model)
        
        if self.config.merge_method == "linear":
            return self._linear_merge(models)
        else:
            raise NotImplementedError(f"Merge method {self.config.merge_method} not implemented")

    def _linear_merge(self, models: List[Dict]) -> Dict:
        weights = self.config.parameters.get("weights", [1/len(models)] * len(models))
        merged_model = {}
        
        # Merge model metadata
        merged_model['name'] = '_'.join(model['name'] for model in models)
        
        # Merge modelfiles
        merged_modelfile = "FROM scratch\n\n"
        for model, weight in zip(models, weights):
            merged_modelfile += f"MERGE {model['name']} WEIGHT {weight}\n"
        
        # Add any parameters from the first model
        if 'parameters' in models[0] and isinstance(models[0]['parameters'], dict):
            for param, value in models[0]['parameters'].items():
                merged_modelfile += f"PARAMETER {param} {value}\n"
        elif 'parameters' in models[0]:
            # If 'parameters' is a string, just add it as is
            merged_modelfile += f"PARAMETER {models[0]['parameters']}\n"
        
        merged_model['modelfile'] = merged_modelfile
        
        return merged_model

    def _print_model_info(self, model):
        print(f"Model Information:")
        for key, value in model.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")