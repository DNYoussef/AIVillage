from typing import List, Dict
from .config import MergeKitConfig

class MergeKitMerger:
    def __init__(self, config: MergeKitConfig):
        self.config = config

    def merge(self, models: List[dict]) -> dict:
        if self.config.merge_method == "linear":
            return self._linear_merge(models)
        else:
            raise NotImplementedError(f"Merge method {self.config.merge_method} not implemented")

    def _linear_merge(self, models: List[dict]) -> dict:
        merged_model = {}
        weights = self.config.parameters.get("weights", [1/len(models)] * len(models))
        
        for key in models[0].keys():
            if isinstance(models[0][key], (int, float)):
                merged_model[key] = sum(weight * model[key] for model, weight in zip(models, weights))
            else:
                merged_model[key] = models[0][key]

        return merged_model