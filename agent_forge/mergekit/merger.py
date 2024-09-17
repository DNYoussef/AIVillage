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
            merged_model = self._linear_merge(models)
            print("\nGenerated Modelfile:")
            print(merged_model['modelfile'])
            if self._validate_modelfile(merged_model['modelfile']):
                return merged_model
            else:
                raise ValueError("Generated modelfile is invalid")
        else:
            raise NotImplementedError(f"Merge method {self.config.merge_method} not implemented")

    def _linear_merge(self, models: List[Dict]) -> Dict:
        weights = self.config.parameters.get("weights", [1/len(models)] * len(models))
        merged_model = {}
        
        # Merge model metadata
        merged_model['name'] = '_'.join(model['name'] for model in models)
        
        # Create a valid modelfile
        merged_modelfile = f"FROM {models[0]['name']}\n\n"
        
        # Add MERGE commands
        for model, weight in zip(models[1:], weights[1:]):
            merged_modelfile += f"MERGE {model['name']} WEIGHT {weight}\n"
        
        merged_modelfile += "\n"
        
        # Merge parameters
        merged_params = {}
        for model in models:
            if 'params' in model:
                if isinstance(model['params'], str):
                    model_params = eval(model['params'])
                else:
                    model_params = model['params']
                merged_params.update(model_params)
        
        # Add merged parameters to modelfile
        for key, value in merged_params.items():
            if isinstance(value, list):
                value = json.dumps(value)
            merged_modelfile += f"PARAMETER {key} {value}\n"
        
        merged_modelfile += "\n"
        
        # Combine templates (this might need manual adjustment)
        templates = [model.get('template', '') for model in models]
        combined_template = ' '.join(templates)
        merged_modelfile += f"TEMPLATE \"\"\"{combined_template}\"\"\"\n"
        
        merged_model['modelfile'] = merged_modelfile
        
        return merged_model

    def _validate_modelfile(self, modelfile: str) -> bool:
        valid_commands = {"from", "merge", "parameter", "template"}
        lines = modelfile.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                command = line.split()[0].lower()
                if command not in valid_commands:
                    print(f"Invalid command in modelfile: {line}")
                    return False
        return True

    def _print_model_info(self, model):
        print(f"Model Information:")
        for key, value in model.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")