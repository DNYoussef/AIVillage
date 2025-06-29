import os
import shutil
from utils.logging import get_logger
import random
import json

class ADASystem:
    """Simple wrapper for the Automatic Discovery of Agentic Space (ADAS) stage."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.logger = get_logger(__name__)

    def perturb_and_evaluate(self, config: dict) -> tuple[dict, float]:
        """Perturb hyperparameters and score the candidate.

        Parameters
        ----------
        config: dict
            Dictionary with ``num_layers`` and ``hidden_size`` keys.

        Returns
        -------
        Tuple containing the perturbed configuration and its score.
        """

        cand = dict(config)
        cand["num_layers"] = max(1, cand.get("num_layers", 1) + random.choice([-1, 0, 1]))
        cand["hidden_size"] = max(1, cand.get("hidden_size", 1) + random.randint(-16, 16))

        score = 1.0 / (cand["num_layers"] * cand["hidden_size"])
        return cand, score

    def optimize_agent_architecture(self, output_dir: str, iterations: int = 3) -> str:
        """Run a lightweight ADAS search and save the best candidate.

        Parameters
        ----------
        output_dir: str
            Directory in which to place the ``adas_optimized_model`` folder.
        iterations: int, optional
            Number of perturbation rounds to perform. Defaults to ``3``.
        """

        self.logger.info("Starting ADAS optimization")

        base_config = {"num_layers": 4, "hidden_size": 128}
        best_config, best_score = self.perturb_and_evaluate(base_config)

        for _ in range(iterations - 1):
            cand, score = self.perturb_and_evaluate(best_config)
            if score > best_score:
                best_config, best_score = cand, score

        self.logger.info(f"Best config after search: {best_config}")

        optimized_path = os.path.join(output_dir, "adas_optimized_model")
        os.makedirs(optimized_path, exist_ok=True)

        if os.path.isdir(self.model_path):
            for fname in os.listdir(self.model_path):
                src = os.path.join(self.model_path, fname)
                dst = os.path.join(optimized_path, fname)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        else:
            shutil.copy2(self.model_path, optimized_path)

        with open(os.path.join(optimized_path, "adas_config.json"), "w") as f:
            json.dump(best_config, f)

        self.logger.info(f"ADAS optimization complete. Saved to: {optimized_path}")
        return optimized_path
