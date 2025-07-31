import importlib.util
import unittest

# Skip these heavy integration tests if PyTorch is unavailable.  They
# require loading pretrained models and running tensor operations.
try:
    torch_spec = importlib.util.find_spec("torch")
except ValueError:
    torch_spec = None
if torch_spec is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

import pytest
import torch

from .config import Configuration, EvolutionSettings, MergeSettings, ModelReference
from .evolutionary_tournament import EvolutionaryTournament, run_evolutionary_tournament
from .merging.merge_techniques import MERGE_TECHNIQUES
from .merging.merger import AdvancedModelMerger
from .model_loading import load_models
from .utils import (
    clean_up_models,
    mask_model_weights,
)


class TestEvoMerge(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Configuration(
            models=[
                ModelReference(name="model1", path="gpt2"),
                ModelReference(
                    name="model2", path="gpt2"
                ),  # Changed from gpt2-medium to gpt2
            ],
            merge_settings=MergeSettings(
                merge_method="ps_dfs",
                parameters={},
                custom_dir="./test_merged_models",
                ps_techniques=["linear"],
                dfs_techniques=["frankenmerge"],
                weight_mask_rate=0.1,
                use_weight_rescale=True,
                mask_strategy="random",
            ),
            evolution_settings=EvolutionSettings(),
        )

    def test_load_models(self) -> None:
        models, tokenizers = load_models(self.config.models)
        assert len(models) == 2
        assert isinstance(models[0], torch.nn.Module)
        assert isinstance(models[1], torch.nn.Module)

    def test_merge_settings_validation(self) -> None:
        # Test valid configuration
        valid_config = MergeSettings(
            merge_method="ps_dfs",
            parameters={},
            custom_dir="./test_merged_models",
            ps_techniques=["linear"],
            dfs_techniques=["frankenmerge"],
        )
        assert isinstance(valid_config, MergeSettings)

        # Test invalid configuration
        with pytest.raises(ValueError):
            MergeSettings(
                merge_method="invalid_method",
                parameters={},
                custom_dir="./test_merged_models",
                ps_techniques=["linear"],
                dfs_techniques=["frankenmerge"],
            )

    def test_merge_techniques(self) -> None:
        weights = {"layer1": torch.rand(2, 3, 4), "layer2": torch.rand(2, 4, 5)}

        for func in MERGE_TECHNIQUES.values():
            merged_weights = func(weights, [])
            assert len(merged_weights) == len(weights)
            for key in weights:
                assert merged_weights[key].shape == weights[key].shape

    def test_ties_merge(self) -> None:
        weights = {"layer1": torch.rand(2, 3, 4), "layer2": torch.rand(2, 4, 5)}
        merged_weights = MERGE_TECHNIQUES["ties"](weights, [], threshold=0.1)
        assert len(merged_weights) == len(weights)
        for key in weights:
            assert merged_weights[key].shape == weights[key].shape

    def test_dare_merge(self) -> None:
        weights = {"layer1": torch.rand(2, 3, 4), "layer2": torch.rand(2, 4, 5)}
        merged_weights = MERGE_TECHNIQUES["dare"](
            weights, [], threshold=0.1, amplification=2.0
        )
        assert len(merged_weights) == len(weights)
        for key in weights:
            assert merged_weights[key].shape == weights[key].shape

    def test_weight_masking(self) -> None:
        model = torch.nn.Linear(10, 10)
        masked_state_dict = mask_model_weights(
            finetuned_model=model,
            pretrained_model=model,
            exclude_param_names_regex=[],
            weight_format="finetuned_weight",
            weight_mask_rate=0.1,
            use_weight_rescale=True,
            mask_strategy="random",
        )
        assert len(masked_state_dict) == len(model.state_dict())
        for key in model.state_dict():
            assert masked_state_dict[key].shape == model.state_dict()[key].shape

    def test_advanced_model_merger(self) -> None:
        merger = AdvancedModelMerger(self.config)
        merged_model_path = merger.merge()
        assert merged_model_path.startswith(self.config.merge_settings.custom_dir)

    def test_evolutionary_tournament(self) -> None:
        evolutionary_tournament = EvolutionaryTournament(self.config)
        best_models = evolutionary_tournament.evolve()
        assert isinstance(best_models, list)
        assert all(isinstance(model, str) for model in best_models)
        assert all(model.startswith(self.config.merge_settings.custom_dir) for model in best_models)

    def test_run_evolutionary_tournament(self) -> None:
        best_models = run_evolutionary_tournament(self.config)
        assert isinstance(best_models, list)
        assert all(isinstance(model, str) for model in best_models)
        assert all(model.startswith(self.config.merge_settings.custom_dir) for model in best_models)

    def tearDown(self) -> None:
        clean_up_models([f"{self.config.merge_settings.custom_dir}/merged_*"])


if __name__ == "__main__":
    unittest.main()
