import time
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import create_default_config
from .evolutionary_tournament import run_evolutionary_tournament
from .utils import clean_up_models, generate_text
from agent_forge.evaluation.evaluator import evaluate_model


class TestEvoMergeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = create_default_config()
        cls.config.evolution_settings.num_generations = 2  # Reduce for faster testing
        cls.config.evolution_settings.population_size = 4

    def test_end_to_end_process(self) -> None:
        start_time = time.time()

        # Run the evolutionary tournament
        best_model_path = run_evolutionary_tournament(self.config)

        # Check if the best model was created
        assert best_model_path.startswith(self.config.merge_settings.custom_dir)

        # Load the best model and generate text
        model = AutoModelForCausalLM.from_pretrained(best_model_path)
        tokenizer = AutoTokenizer.from_pretrained(best_model_path)

        prompt = "The capital of France is"
        generated_text = generate_text(model, tokenizer, prompt)

        # Check if the generated text is non-empty and contains the prompt
        assert len(generated_text) > len(prompt)
        assert prompt in generated_text

        # Evaluate the model
        evaluation_result = evaluate_model(best_model_path)

        # Check if the evaluation result contains expected keys
        assert "overall_score" in evaluation_result
        assert "perplexity" in evaluation_result

        end_time = time.time()
        print(f"End-to-end test completed in {end_time - start_time:.2f} seconds")

    @classmethod
    def tearDownClass(cls) -> None:
        clean_up_models([f"{cls.config.merge_settings.custom_dir}/*"])


if __name__ == "__main__":
    unittest.main()
