import unittest
import torch
from .config import Configuration, ModelReference, MergeSettings, EvolutionSettings
from .utils import (
    EvoMergeException,
    load_models,
    save_model,
    validate_merge_config,
    generate_text,
    evaluate_model,
    setup_gpu_if_available,
    clean_up_models,
    MERGE_TECHNIQUES
)
from .merger import AdvancedModelMerger
from .evolutionary_tournament import EvolutionaryMerger, run_evolutionary_tournament

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
                dfs_techniques=["frankenmerge"]
            ),
            evolution_settings=EvolutionSettings()
        )

    def test_load_models(self):
        models = load_models(self.config.models)
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
            merged_weights = func(weights)
            self.assertEqual(len(merged_weights), len(weights))
            for key in weights:
                self.assertEqual(merged_weights[key].shape, weights[key][0].shape)

    def test_advanced_model_merger(self):
        merger = AdvancedModelMerger(self.config)
        merged_model_path = merger.merge()
        self.assertTrue(merged_model_path.startswith(self.config.merge_settings.custom_dir))

    def test_evolutionary_merger(self):
        evolutionary_merger = EvolutionaryMerger(self.config)
        best_model = evolutionary_merger.evolve()
        self.assertIsInstance(best_model, str)
        self.assertTrue(best_model.startswith(self.config.merge_settings.custom_dir))

    def test_run_evolutionary_tournament(self):
        best_model = run_evolutionary_tournament(self.config)
        self.assertIsInstance(best_model, str)
        self.assertTrue(best_model.startswith(self.config.merge_settings.custom_dir))

    def tearDown(self):
        clean_up_models([f"{self.config.merge_settings.custom_dir}/merged_*"])

if __name__ == '__main__':
    unittest.main()
