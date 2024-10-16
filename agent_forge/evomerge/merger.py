import os
import torch
import logging
from typing import List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil

from .config import Configuration, ModelReference
from .model_loading import load_models, save_model
from .evaluation import evaluate_model
from .merge_techniques import MERGE_TECHNIQUES
from .instruction_tuning import is_instruction_tuned_model
from .cross_domain import get_model_domain
from .utils import EvoMergeException, check_system_resources

logger = logging.getLogger(__name__)

class AdvancedModelMerger:
    def __init__(self, config: Configuration):
        self.config = config

    def merge(self) -> str:
        logger.info("Starting model merging process")
        try:
            # Check if model paths exist and log their contents
            for model in self.config.models:
                if not os.path.exists(model.path):
                    raise ValueError(f"Model path does not exist: {model.path}")
                logger.info(f"Model path exists: {model.path}")
                logger.info(f"Contents of {model.path}:")
                for item in os.listdir(model.path):
                    logger.info(f"  {item}")
                
                # Check for specific files that should be present
                expected_files = ['config.json', 'tokenizer.json']
                model_files = ['pytorch_model.bin', 'model.safetensors']
                missing_files = [file for file in expected_files if file not in os.listdir(model.path)]
                if not any(file in os.listdir(model.path) for file in model_files):
                    missing_files.extend(model_files)
                if missing_files:
                    logger.warning(f"Missing expected files in {model.path}: {', '.join(missing_files)}")
                    if all(file in missing_files for file in model_files):
                        raise ValueError(f"Missing required model file for {model.name}: pytorch_model.bin or model.safetensors")

            # Check and create custom directory if it doesn't exist
            if not os.path.exists(self.config.merge_settings.custom_dir):
                logger.warning(f"Custom directory does not exist: {self.config.merge_settings.custom_dir}")
                logger.info("Creating custom directory")
                os.makedirs(self.config.merge_settings.custom_dir, exist_ok=True)

            # Check available disk space
            check_system_resources([model.path for model in self.config.models])

            models = load_models(self.config.models)

            if all(is_instruction_tuned_model(model) for model in models):
                merged_model = self._merge_instruction_tuned_models(models)
            elif self._models_are_compatible(models):
                if self.config.merge_settings.merge_method == "ps":
                    merged_model = self._ps_merge(models)
                elif self.config.merge_settings.merge_method == "dfs":
                    merged_model = self._dfs_merge(models)
                elif self.config.merge_settings.merge_method == "ps_dfs":
                    ps_model = self._ps_merge(models)
                    merged_model = self._dfs_merge([ps_model] + models)
                else:
                    raise NotImplementedError(f"Merge method {self.config.merge_settings.merge_method} not implemented")
            else:
                merged_model = self._merge_cross_domain_models(models)

            merged_model_path = self._save_merged_model(merged_model)

            logger.info(f"Model merging completed successfully. Saved to: {merged_model_path}")
            return merged_model_path
        except Exception as e:
            logger.error(f"Error during merge process: {str(e)}")
            logger.exception("Traceback:")
            raise

    def _models_are_compatible(self, models: List[torch.nn.Module]) -> bool:
        base_architecture = type(models[0])
        return all(isinstance(model, base_architecture) for model in models)

    def _merge_instruction_tuned_models(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        # Implementation of instruction-tuned model merging
        ...

    def _ps_merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        # Implementation of parameter space merging
        ...

    def _dfs_merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        # Implementation of deep fusion space merging
        ...

    def _merge_cross_domain_models(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        # Implementation of cross-domain model merging
        ...

    def _save_merged_model(self, model: torch.nn.Module) -> str:
        merged_model_name = f"merged_{self.config.merge_settings.merge_method}_{'_'.join([m.name for m in self.config.models])}"
        merged_model_path = os.path.join(self.config.merge_settings.custom_dir, merged_model_name)
        save_model(model, merged_model_path)
        return merged_model_path

    def _merge_with_adapters(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        base_model = models[0]
        for i, model in enumerate(models[1:], 1):
            adapter = self._create_adapter(base_model, model)
            base_model.add_adapter(f"domain_{i}", adapter)
        return base_model

    def _create_adapter(self, base_model: torch.nn.Module, target_model: torch.nn.Module) -> torch.nn.Module:
        # Implement adapter creation logic
        # This is a placeholder and should be replaced with actual adapter creation code
        return torch.nn.Linear(base_model.config.hidden_size, target_model.config.hidden_size)

    def _merge_embeddings_only(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        base_model = models[0]
        all_embeddings = [model.get_input_embeddings().weight for model in models]
        merged_embeddings = torch.mean(torch.stack(all_embeddings), dim=0)
        base_model.get_input_embeddings().weight.data = merged_embeddings
        return base_model

    def _full_cross_domain_merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        # Implement full cross-domain merging logic
        # This is a complex task and might require advanced techniques like neural architecture search
        # For now, we'll use a simple averaging of weights where possible
        base_model = models[0]
        for name, param in base_model.named_parameters():
            if all(name in model.state_dict() for model in models):
                merged_param = torch.mean(torch.stack([model.state_dict()[name] for model in models]), dim=0)
                param.data = merged_param
        return base_model


