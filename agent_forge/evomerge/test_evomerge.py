import unittest
import torch
from .config import Configuration, ModelReference, MergeSettings, EvolutionSettings
from .utils import (
    EvoMergeException,
    validate_merge_config,
    generate_text,
    setup_gpu_if_available,
    clean_up_models,
    mask_model_weights
)
from .model_loading import load_models, save_model
from .evaluation import evaluate_model
from .merge_techniques import MERGE_TECHNIQUES
from .merger import AdvancedModelMerger
from .evolutionary_tournament import EvolutionaryTournament, run_evolutionary_tournament

class TestEvoMerge(unittest.TestCase):
    def setUp(self):
        self.config = Configuration(
            models=[
                ModelReference(name="model1", path="gpt2"),
                ModelReference(name="model2", path="gpt2-medium"),
            ],
            merge_settings=MergeSettings(
                merge_method="ps_dfs",
                parameters={},
                custom_dir="./test_merged_models",
                ps_techniques=["linear"],
                dfs_techniques=["frankenmerge"],
                weight_mask_rate=0.1,
                use_weight_rescale=True,
                mask_strategy="random"
            ),
            evolution_settings=EvolutionSettings()
        )

    def test_load_models(self):
        models, tokenizers = load_models(self.config.models)
        self.assertEqual(len(models), 2)
        self.assertIsInstance(models[0], torch.nn.Module)
        self.assertIsInstance(models[1], torch.nn.Module)

    def test_validate_merge_config(self):
        validate_merge_config(self.config.merge_settings)
        
        # Test invalid configuration
        invalid_config = MergeSettings(
            merge_method="invalid_method",
            parameters={},
            custom_dir="./test_merged_models",
            ps_techniques=["linear"],
            dfs_techniques=["frankenmerge"]
        )
        with self.assertRaises(EvoMergeException):
            validate_merge_config(invalid_config)

    def test_merge_techniques(self):
        weights = {
            "layer1": torch.rand(2, 3, 4),
            "layer2": torch.rand(2, 4, 5)
        }
        
        for technique, func in MERGE_TECHNIQUES.items():
            merged_weights = func(weights, [])
            self.assertEqual(len(merged_weights), len(weights))
            for key in weights:
                self.assertEqual(merged_weights[key].shape, weights[key].shape)

    def test_ties_merge(self):
        weights = {
            "layer1": torch.rand(2, 3, 4),
            "layer2": torch.rand(2, 4, 5)
        }
        merged_weights = MERGE_TECHNIQUES["ties"](weights, [], threshold=0.1)
        self.assertEqual(len(merged_weights), len(weights))
        for key in weights:
            self.assertEqual(merged_weights[key].shape, weights[key].shape)

    def test_dare_merge(self):
        weights = {
            "layer1": torch.rand(2, 3, 4),
            "layer2": torch.rand(2, 4, 5)
        }
        merged_weights = MERGE_TECHNIQUES["dare"](weights, [], threshold=0.1, amplification=2.0)
        self.assertEqual(len(merged_weights), len(weights))
        for key in weights:
            self.assertEqual(merged_weights[key].shape, weights[key].shape)

    def test_weight_masking(self):
        model = torch.nn.Linear(10, 10)
        masked_state_dict = mask_model_weights(
            finetuned_model=model,
            pretrained_model=model,
            exclude_param_names_regex=[],
            weight_format="finetuned_weight",
            weight_mask_rate=0.1,
            use_weight_rescale=True,
            mask_strategy="random"
        )
        self.assertEqual(len(masked_state_dict), len(model.state_dict()))
        for key in model.state_dict():
            self.assertEqual(masked_state_dict[key].shape, model.state_dict()[key].shape)

    def test_advanced_model_merger(self):
        merger = AdvancedModelMerger(self.config)
        merged_model_path = merger.merge()
        self.assertTrue(merged_model_path.startswith(self.config.merge_settings.custom_dir))

    def test_evolutionary_tournament(self):
        evolutionary_tournament = EvolutionaryTournament(self.config)
        best_models = evolutionary_tournament.evolve()
        self.assertIsInstance(best_models, list)
        self.assertTrue(all(isinstance(model, str) for model in best_models))
        self.assertTrue(all(model.startswith(self.config.merge_settings.custom_dir) for model in best_models))

    def test_run_evolutionary_tournament(self):
        best_models = run_evolutionary_tournament(self.config)
        self.assertIsInstance(best_models, list)
        self.assertTrue(all(isinstance(model, str) for model in best_models))
        self.assertTrue(all(model.startswith(self.config.merge_settings.custom_dir) for model in best_models))

    def tearDown(self):
        clean_up_models([f"{self.config.merge_settings.custom_dir}/merged_*"])

if __name__ == '__main__':
    unittest.main()
