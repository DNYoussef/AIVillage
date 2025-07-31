import logging
import random
from typing import Any

import torch
from torch import nn, optim

from rag_system.utils.error_handling import AIVillageException, log_and_handle_errors

# The optimization utilities live in the planning package.  The original
# import pointed to a deprecated `planning_and_task_management` package which
# no longer exists.
from .planning.optimization import Optimizer

logger = logging.getLogger(__name__)


class EvolutionManager:
    def __init__(
        self,
        population_size: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_individual = None

    @log_and_handle_errors
    async def evolve(
        self,
        architecture_space: dict[str, Any],
        fitness_function,
        generations: int = 50,
    ):
        self.initialize_population(architecture_space)

        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")

            await self.evaluate_population(fitness_function)

            new_population = []
            while len(new_population) < self.population_size:
                parents = self.select_parents()
                child = self.crossover(parents[0], parents[1])
                child = self.mutate(child, architecture_space)
                new_population.append(child)

            self.population = new_population

        await self.evaluate_population(fitness_function)
        logger.info(f"Best individual: {self.best_individual}")
        return self.best_individual

    def initialize_population(self, architecture_space: dict[str, Any]) -> None:
        self.population = []
        for _ in range(self.population_size):
            individual = {
                param: random.choice(values)
                for param, values in architecture_space.items()
            }
            self.population.append(individual)

    async def evaluate_population(self, fitness_function) -> None:
        for individual in self.population:
            individual["fitness"] = await fitness_function(individual)

        self.population.sort(key=lambda x: x["fitness"], reverse=True)
        self.best_individual = self.population[0]

    def select_parents(self):
        tournament_size = 3
        parents = []
        for _ in range(2):
            tournament = random.sample(self.population, tournament_size)
            parent = max(tournament, key=lambda x: x["fitness"])
            parents.append(parent)
        return parents

    def crossover(
        self, parent1: dict[str, Any], parent2: dict[str, Any]
    ) -> dict[str, Any]:
        if random.random() < self.crossover_rate:
            child = {}
            for key in parent1:
                if key != "fitness":
                    child[key] = random.choice([parent1[key], parent2[key]])
            return child
        return random.choice([parent1, parent2]).copy()

    def mutate(
        self, individual: dict[str, Any], architecture_space: dict[str, Any]
    ) -> dict[str, Any]:
        for key in individual:
            if key != "fitness" and random.random() < self.mutation_rate:
                individual[key] = random.choice(architecture_space[key])
        return individual


@log_and_handle_errors
async def run_evolution_and_optimization(king_agent) -> None:
    architecture_space = {
        "input_size": [64, 128, 256],
        "hidden_sizes": [[64, 64], [128, 64], [256, 128]],
        "output_size": [32, 64],
        "num_layers": [2, 3, 4],
        "activation": ["relu", "tanh", "sigmoid"],
        "dropout_rate": [0.1, 0.3, 0.5],
    }

    optimizer = Optimizer()

    async def fitness_function(architecture):
        try:
            model = king_agent.create_model_from_architecture(architecture)

            # Generate some dummy data for training and evaluation
            input_size = architecture["input_size"]
            output_size = architecture["output_size"]
            X_train = torch.randn(100, input_size)
            y_train = torch.randn(100, output_size)
            X_val = torch.randn(20, input_size)
            y_val = torch.randn(20, output_size)

            # Train the model for a few epochs
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters())

            for _epoch in range(5):  # Train for 5 epochs
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            # Evaluate the model
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)

            # Use negative loss as fitness (higher is better)
            fitness = -val_loss.item()

            logger.info(f"Evaluated architecture with fitness: {fitness}")
            return fitness
        except Exception as e:
            logger.exception(f"Error in fitness function: {e!s}")
            return float("-inf")  # Return worst possible fitness on error

    try:
        evolution_manager = EvolutionManager()
        best_architecture = await evolution_manager.evolve(
            architecture_space, fitness_function
        )

        king_agent.create_model_from_architecture(best_architecture)
        await king_agent.update_model_architecture(best_architecture)

        # Hyperparameter optimization
        hyperparameter_space = {
            "learning_rate": [0.001, 0.01, 0.1],
            "batch_size": [32, 64, 128],
            "optimizer": ["adam", "sgd", "rmsprop"],
        }

        best_hyperparameters = await optimizer.optimize_hyperparameters(
            hyperparameter_space, fitness_function
        )
        await king_agent.update_hyperparameters(best_hyperparameters)

        logger.info(
            f"Evolution and optimization completed. Best architecture: {best_architecture}"
        )
        logger.info(f"Best hyperparameters: {best_hyperparameters}")

    except Exception as e:
        logger.exception(f"Error in run_evolution_and_optimization: {e!s}")
        msg = f"Error in run_evolution_and_optimization: {e!s}"
        raise AIVillageException(msg)
