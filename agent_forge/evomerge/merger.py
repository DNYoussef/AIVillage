import os
import torch
import logging
from typing import List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Configuration
from .utils import (
    load_models,
    save_model,
    evaluate_model,
    MERGE_TECHNIQUES
)

logger = logging.getLogger(__name__)

class AdvancedModelMerger:
    def __init__(self, config: Configuration):
        self.config = config

    def merge(self) -> str:
        logger.info("Starting model merging process")
        try:
            models = load_models(self.config.models)

            if self.config.merge_settings.merge_method == "ps":
                merged_model = self._ps_merge(models)
            elif self.config.merge_settings.merge_method == "dfs":
                merged_model = self._dfs_merge(models)
            elif self.config.merge_settings.merge_method == "ps_dfs":
                ps_model = self._ps_merge(models)
                merged_model = self._dfs_merge([ps_model] + models)
            else:
                raise NotImplementedError(f"Merge method {self.config.merge_settings.merge_method} not implemented")

            merged_model_path = self._save_merged_model(merged_model)

            logger.info(f"Model merging completed successfully. Saved to: {merged_model_path}")
            return merged_model_path
        except Exception as e:
            logger.error(f"Error during merge process: {str(e)}")
            raise

    def _ps_merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        weights = self._get_model_weights(models)
        
        for technique in self.config.merge_settings.ps_techniques:
            weights = MERGE_TECHNIQUES[technique](weights, **self.config.merge_settings.parameters.get(technique, {}))
        
        merged_model = AutoModelForCausalLM.from_pretrained(self.config.models[0].path)
        merged_model.load_state_dict(weights)
        return merged_model

    def _dfs_merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        weights = self._get_model_weights(models)
        
        for technique in self.config.merge_settings.dfs_techniques:
            weights = MERGE_TECHNIQUES[technique](weights, models=models, **self.config.merge_settings.parameters.get(technique, {}))
        
        merged_model = AutoModelForCausalLM.from_pretrained(self.config.models[0].path)
        merged_model.load_state_dict(weights)
        return merged_model

    def _get_model_weights(self, models: List[torch.nn.Module]) -> Dict[str, torch.Tensor]:
        weights = {}
        base_model = models[0]
        for key in base_model.state_dict().keys():
            tensors = []
            for i, model in enumerate(models):
                if key in model.state_dict():
                    tensor = model.state_dict()[key]
                    if tensor.shape != base_model.state_dict()[key].shape:
                        logger.warning(f"Shape mismatch for {key}: Model {i} has shape {tensor.shape}, base model has shape {base_model.state_dict()[key].shape}")
                        tensor = self._resize_tensor(tensor, base_model.state_dict()[key].shape)
                    tensors.append(tensor)
                else:
                    logger.warning(f"Key {key} not found in model {i}. Using zeros.")
                    tensors.append(torch.zeros_like(base_model.state_dict()[key]))
            weights[key] = torch.stack(tensors)
        return weights

    def _resize_tensor(self, tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        if len(tensor.shape) != len(target_shape):
            logger.error(f"Cannot resize tensor with shape {tensor.shape} to {target_shape}: Dimension mismatch")
            raise ValueError(f"Cannot resize tensor with shape {tensor.shape} to {target_shape}: Dimension mismatch")
        
        result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
        for dim in range(len(tensor.shape)):
            if tensor.shape[dim] > target_shape[dim]:
                indices = torch.randperm(tensor.shape[dim])[:target_shape[dim]]
                tensor = tensor.index_select(dim, indices)
            elif tensor.shape[dim] < target_shape[dim]:
                pad_size = target_shape[dim] - tensor.shape[dim]
                pad_dims = [0] * (2 * len(tensor.shape))
                pad_dims[2 * dim + 1] = pad_size
                tensor = torch.nn.functional.pad(tensor, pad_dims)
        result.copy_(tensor)
        return result

    def _save_merged_model(self, model: torch.nn.Module) -> str:
        merged_model_name = f"merged_{self.config.merge_settings.merge_method}_{'_'.join([m.name for m in self.config.models])}"
        merged_model_path = os.path.join(self.config.merge_settings.custom_dir, merged_model_name)
        save_model(model, merged_model_path)
        return merged_model_path

def main():
    # This main function is for testing purposes and can be removed in production
    from .config import ModelReference, create_default_config
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = create_default_config()

    merger = AdvancedModelMerger(config)
    merged_model_path = merger.merge()

    logger.info(f"Merged model saved at: {merged_model_path}")
    evaluation_result = evaluate_model(merged_model_path)
    logger.info(f"Evaluation result: {evaluation_result}")

if __name__ == "__main__":
    main()

