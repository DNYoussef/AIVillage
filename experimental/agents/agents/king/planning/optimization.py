import logging
import random
from typing import Any

from torch import nn

from rag_system.utils.error_handling import AIVillageException, log_and_handle_errors

logger = logging.getLogger(__name__)


class Optimizer:
    def __init__(self) -> None:
        self.model = None
        self.hyperparameters = {}
        self.mcts = None  # Initialize MCTS if needed

    @log_and_handle_errors
    async def optimize_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        logger.info(f"Optimizing plan: {plan}")
        optimized_plan = await self.mcts.search(plan)
        return optimized_plan

    @log_and_handle_errors
    async def optimize_hyperparameters(
        self, hyperparameter_space: dict[str, Any], fitness_function
    ) -> dict[str, Any]:
        logger.info("Optimizing hyperparameters")
        best_hyperparameters = None
        best_fitness = float("-inf")

        for _ in range(50):  # Number of iterations
            try:
                hyperparameters = {
                    k: random.choice(v) for k, v in hyperparameter_space.items()
                }
                fitness = await fitness_function(hyperparameters)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_hyperparameters = hyperparameters

                logger.debug(
                    f"Evaluated hyperparameters: {hyperparameters}, fitness: {fitness}"
                )
            except Exception as e:
                logger.exception(f"Error evaluating hyperparameters: {e!s}")

        if best_hyperparameters is None:
            msg = "Failed to find valid hyperparameters"
            raise AIVillageException(msg)

        logger.info(
            f"Best hyperparameters found: {best_hyperparameters}, fitness: {best_fitness}"
        )
        return best_hyperparameters

    @log_and_handle_errors
    async def update_model(self, new_model: nn.Module) -> None:
        self.model = new_model
        logger.info("Model updated in Optimizer")

    @log_and_handle_errors
    async def update_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        self.hyperparameters.update(hyperparameters)
        logger.info("Hyperparameters updated in Optimizer")

    @log_and_handle_errors
    async def save_models(self, path: str) -> None:
        logger.info(f"Saving optimizer models to {path}")
        if self.mcts:
            self.mcts.save(path)

    @log_and_handle_errors
    async def load_models(self, path: str) -> None:
        logger.info(f"Loading optimizer models from {path}")
        if self.mcts:
            self.mcts.load(path)

    async def introspect(self) -> dict[str, Any]:
        return {
            "type": "Optimizer",
            "model_type": str(type(self.model)) if self.model else "None",
            "hyperparameters": self.hyperparameters,
        }
