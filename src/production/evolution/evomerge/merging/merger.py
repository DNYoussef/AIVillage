import logging
import os

import torch
from transformers import AutoTokenizer

from ..config import Configuration
from ..cross_domain import (
    merge_cross_domain_models,
)
from ..instruction_tuning import (
    is_instruction_tuned_model,
    merge_instruction_tuned_models,
)
from ..model_tracker import model_tracker
from ..utils import (
    EvoMergeException,
    check_system_resources,
    clean_up_models,
    evaluate_model,
    load_models,
    mask_model_weights,
    save_model,
)
from .merge_techniques import MERGE_TECHNIQUES

logger = logging.getLogger(__name__)


class AdvancedModelMerger:
    def __init__(self, config: Configuration) -> None:
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

            if all(
                is_instruction_tuned_model(model, tokenizer)
                for model, tokenizer in zip(self.models, self.tokenizers, strict=False)
            ):
                logger.info(
                    "All models are instruction-tuned. Using instruction-tuned merge method."
                )
                merged_model, merged_tokenizer = merge_instruction_tuned_models(
                    self.models, self.tokenizers, self.config.merge_settings
                )
            elif self._models_are_compatible(self.models):
                logger.info("Models are compatible. Using standard merge method.")
                merged_model = self._merge_compatible_models(self.models)
                merged_tokenizer = self.tokenizers[
                    0
                ]  # Use the first tokenizer as the base
            else:
                logger.info(
                    "Models are not compatible. Using cross-domain merge method."
                )
                merged_model = merge_cross_domain_models(
                    self.models, self.config.merge_settings
                )
                merged_tokenizer = self.tokenizers[
                    0
                ]  # Use the first tokenizer as the base

            if merged_model is None:
                msg = "Merge process failed to produce a valid model"
                raise EvoMergeException(msg)

            merged_model_path = self._save_merged_model(merged_model, merged_tokenizer)

            self._track_merged_model(merged_model_path)

            # Clean up loaded models to free memory
            del self.models
            del self.tokenizers
            torch.cuda.empty_cache()

            logger.info(
                f"Model merging completed successfully. Saved to: {merged_model_path}"
            )
            return merged_model_path
        except Exception as e:
            logger.exception(f"Error during merge process: {e!s}")
            logger.exception("Traceback:")
            raise

    def _prepare_merge_directory(self) -> None:
        if not os.path.exists(self.config.merge_settings.custom_dir):
            logger.warning(
                f"Custom directory does not exist: {self.config.merge_settings.custom_dir}"
            )
            logger.info("Creating custom directory")
            os.makedirs(self.config.merge_settings.custom_dir, exist_ok=True)

    def _check_resources(self) -> None:
        if not check_system_resources([model.path for model in self.config.models]):
            msg = "Insufficient system resources to proceed with merging"
            raise EvoMergeException(msg)

    def _models_are_compatible(self, models: list[torch.nn.Module]) -> bool:
        base_architecture = type(models[0])
        return all(isinstance(model, base_architecture) for model in models)

    def _merge_compatible_models(
        self, models: list[torch.nn.Module]
    ) -> torch.nn.Module:
        logger.info(f"Performing {self.config.merge_settings.merge_method} merge")
        if self.config.merge_settings.merge_method == "ps":
            merged_model = self._ps_merge(models)
        elif self.config.merge_settings.merge_method == "dfs":
            merged_model = self._dfs_merge(models)
        elif self.config.merge_settings.merge_method == "ps_dfs":
            ps_model = self._ps_merge(models)
            merged_model = self._dfs_merge([ps_model, *models])
        else:
            msg = f"Merge method {self.config.merge_settings.merge_method} not implemented"
            raise EvoMergeException(msg)

        # Apply weight masking if configured
        if self.config.merge_settings.weight_mask_rate > 0:
            merged_model = self._apply_weight_masking(merged_model)

        return merged_model

    def _ps_merge(self, models: list[torch.nn.Module]) -> torch.nn.Module:
        logger.info("Performing parameter space merge")
        merged_state_dict = {}
        chunk_size = 1000000  # Adjust based on available memory

        for technique in self.config.merge_settings.ps_techniques:
            for name, _ in models[0].named_parameters():
                merged_param = None
                for i in range(0, models[0].state_dict()[name].numel(), chunk_size):
                    chunk_params = [
                        model.state_dict()[name].flatten()[i : i + chunk_size]
                        for model in models
                    ]
                    merged_chunk = MERGE_TECHNIQUES[technique](
                        chunk_params,
                        **self.config.merge_settings.parameters.get(technique, {}),
                    )
                    if merged_param is None:
                        merged_param = merged_chunk
                    else:
                        merged_param = torch.cat([merged_param, merged_chunk])

                merged_state_dict[name] = merged_param.reshape(
                    models[0].state_dict()[name].shape
                )

        merged_model = type(models[0])(**models[0].config.to_dict())
        merged_model.load_state_dict(merged_state_dict)
        return merged_model

    def _dfs_merge(self, models: list[torch.nn.Module]) -> torch.nn.Module:
        logger.info("Performing deep fusion space merge")
        merged_model = type(models[0])(**models[0].config.to_dict())
        for technique in self.config.merge_settings.dfs_techniques:
            if technique not in MERGE_TECHNIQUES:
                msg = f"Unknown merge technique: {technique}"
                raise ValueError(msg)
            merged_model = MERGE_TECHNIQUES[technique](
                merged_model,
                models,
                **self.config.merge_settings.parameters.get(technique, {}),
            )
        return merged_model

    def _apply_weight_masking(self, model: torch.nn.Module) -> torch.nn.Module:
        logger.info(
            f"Applying weight masking with rate {self.config.merge_settings.weight_mask_rate}"
        )
        masked_state_dict = mask_model_weights(
            finetuned_model=model,
            pretrained_model=self.models[
                0
            ],  # Use the first model as the pretrained model
            exclude_param_names_regex=[],  # No exclusions for now
            weight_format="finetuned_weight",
            weight_mask_rate=self.config.merge_settings.weight_mask_rate,
            use_weight_rescale=self.config.merge_settings.use_weight_rescale,
            mask_strategy=self.config.merge_settings.mask_strategy,
        )
        model.load_state_dict(masked_state_dict)
        return model

    def _save_merged_model(
        self, model: torch.nn.Module, tokenizer: AutoTokenizer
    ) -> str:
        merged_model_name = f"merged_{self.config.merge_settings.merge_method}_{'_'.join([m.name for m in self.config.models])}"
        merged_model_path = os.path.join(
            self.config.merge_settings.custom_dir, merged_model_name
        )
        save_model(model, merged_model_path)
        tokenizer.save_pretrained(merged_model_path)
        return merged_model_path

    def _track_merged_model(self, merged_model_path: str) -> None:
        parent_models = [model.path for model in self.config.models]
        merge_techniques = (
            self.config.merge_settings.ps_techniques
            + self.config.merge_settings.dfs_techniques
        )
        score = evaluate_model(merged_model_path)["overall_score"]
        model_tracker.add_model(
            model_path=merged_model_path,
            parent_models=parent_models,
            merge_techniques=merge_techniques,
            config=self.config.dict(),
            score=score,
        )

    def _cleanup(self) -> None:
        clean_up_models(
            [model.path for model in self.config.models if os.path.exists(model.path)]
        )


if __name__ == "__main__":
    # For testing purposes
    from ..config import create_default_config

    config = create_default_config()
    merger = AdvancedModelMerger(config)
    merged_model_path = merger.merge()
    print(f"Merged model saved at: {merged_model_path}")
