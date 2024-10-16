import os
import torch
import logging
from typing import List, Dict, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil

from .config import Configuration, ModelReference
from .utils import load_models, save_model, evaluate_model, check_system_resources, EvoMergeException, clean_up_models
from .merge_techniques import MERGE_TECHNIQUES
from .instruction_tuning import is_instruction_tuned_model, merge_instruction_tuned_models
from .cross_domain import get_model_domain, merge_cross_domain_models
from .model_tracker import model_tracker

logger = logging.getLogger(__name__)

class AdvancedModelMerger:
    def __init__(self, config: Configuration):
        self.config = config
        self.models = None
        self.tokenizers = None

    def merge(self) -> str:
        logger.info("Starting model merging process")
        try:
            self._prepare_merge_directory()
            self._check_resources()

            logger.info("Loading models and tokenizers")
            self.models, self.tokenizers = load_models(self.config.models)
            
            if all(is_instruction_tuned_model(model, tokenizer) for model, tokenizer in zip(self.models, self.tokenizers)):
                logger.info("All models are instruction-tuned. Using instruction-tuned merge method.")
                merged_model = merge_instruction_tuned_models(self.models, self.tokenizers, self.config.merge_settings)
            elif self._models_are_compatible(self.models):
                logger.info("Models are compatible. Using standard merge method.")
                merged_model = self._merge_compatible_models(self.models)
            else:
                logger.info("Models are not compatible. Using cross-domain merge method.")
                merged_model = merge_cross_domain_models(self.models, self.config.merge_settings)

            if merged_model is None:
                raise EvoMergeException("Merge process failed to produce a valid model")

            merged_model_path = self._save_merged_model(merged_model)

            self._track_merged_model(merged_model_path)

            # Clean up loaded models to free memory
            del self.models
            del self.tokenizers
            torch.cuda.empty_cache()

            logger.info(f"Model merging completed successfully. Saved to: {merged_model_path}")
            return merged_model_path
        except Exception as e:
            logger.error(f"Error during merge process: {str(e)}")
            logger.exception("Traceback:")
            raise

    def _prepare_merge_directory(self):
        if not os.path.exists(self.config.merge_settings.custom_dir):
            logger.warning(f"Custom directory does not exist: {self.config.merge_settings.custom_dir}")
            logger.info("Creating custom directory")
            os.makedirs(self.config.merge_settings.custom_dir, exist_ok=True)

    def _check_resources(self):
        if not check_system_resources([model.path for model in self.config.models]):
            raise EvoMergeException("Insufficient system resources to proceed with merging")

    def _models_are_compatible(self, models: List[torch.nn.Module]) -> bool:
        base_architecture = type(models[0])
        return all(isinstance(model, base_architecture) for model in models)

    def _merge_compatible_models(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        logger.info(f"Performing {self.config.merge_settings.merge_method} merge")
        if self.config.merge_settings.merge_method == "ps":
            return self._ps_merge(models)
        elif self.config.merge_settings.merge_method == "dfs":
            return self._dfs_merge(models)
        elif self.config.merge_settings.merge_method == "ps_dfs":
            ps_model = self._ps_merge(models)
            return self._dfs_merge([ps_model] + models)
        else:
            raise NotImplementedError(f"Merge method {self.config.merge_settings.merge_method} not implemented")

    def _ps_merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        logger.info("Performing parameter space merge")
        merged_state_dict = {}
        for technique in self.config.merge_settings.ps_techniques:
            if technique not in MERGE_TECHNIQUES:
                raise ValueError(f"Unknown merge technique: {technique}")
            merged_state_dict = MERGE_TECHNIQUES[technique](merged_state_dict, models, **self.config.merge_settings.parameters.get(technique, {}))
        
        merged_model = type(models[0])(**models[0].config.to_dict())
        merged_model.load_state_dict(merged_state_dict)
        return merged_model

    def _dfs_merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        logger.info("Performing deep fusion space merge")
        merged_model = type(models[0])(**models[0].config.to_dict())
        for technique in self.config.merge_settings.dfs_techniques:
            if technique not in MERGE_TECHNIQUES:
                raise ValueError(f"Unknown merge technique: {technique}")
            merged_model = MERGE_TECHNIQUES[technique](merged_model, models, **self.config.merge_settings.parameters.get(technique, {}))
        return merged_model

    def _save_merged_model(self, model: torch.nn.Module) -> str:
        merged_model_name = f"merged_{self.config.merge_settings.merge_method}_{'_'.join([m.name for m in self.config.models])}"
        merged_model_path = os.path.join(self.config.merge_settings.custom_dir, merged_model_name)
        save_model(model, merged_model_path)
        return merged_model_path

    def _track_merged_model(self, merged_model_path: str):
        parent_models = [model.path for model in self.config.models]
        merge_techniques = self.config.merge_settings.ps_techniques + self.config.merge_settings.dfs_techniques
        score = evaluate_model(merged_model_path)["overall_score"]
        model_tracker.add_model(
            model_path=merged_model_path,
            parent_models=parent_models,
            merge_techniques=merge_techniques,
            config=self.config.dict(),
            score=score
        )

    def _cleanup(self):
        clean_up_models([model.path for model in self.config.models if os.path.exists(model.path)])

if __name__ == "__main__":
    # For testing purposes
    from .config import create_default_config
    
    config = create_default_config()
    merger = AdvancedModelMerger(config)
    merged_model_path = merger.merge()
    print(f"Merged model saved at: {merged_model_path}")
