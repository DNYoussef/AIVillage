"""
Phase 8 Compression - Compression Optimizer

Advanced optimization techniques for compression algorithms including
hyperparameter tuning, multi-objective optimization, and adaptive compression.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, field
import json
import copy
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import optuna
from sklearn.metrics import pareto_front
import time


@dataclass
class OptimizationObjective:
    """Optimization objective definition."""
    name: str
    weight: float
    minimize: bool = True
    constraint: Optional[Tuple[float, float]] = None  # (min, max) bounds


@dataclass
class OptimizationConfig:
    """Configuration for compression optimization."""
    objectives: List[OptimizationObjective]
    search_space: Dict[str, Any]
    n_trials: int = 100
    n_jobs: int = 1
    timeout: Optional[int] = None
    sampler: str = 'TPE'  # Tree-structured Pareto Estimator
    pruner: str = 'median'
    study_name: Optional[str] = None


@dataclass
class OptimizationResult:
    """Result from optimization process."""
    best_params: Dict[str, Any]
    best_values: Dict[str, float]
    pareto_front: List[Dict[str, Any]]
    optimization_history: List[Dict[str, Any]]
    study_statistics: Dict[str, Any]
    execution_time: float


class ObjectiveEvaluator(ABC):
    """Abstract base class for objective evaluation."""

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        compressed_model: nn.Module,
        params: Dict[str, Any],
        validation_data: Any = None
    ) -> Dict[str, float]:
        """Evaluate objectives for given compression parameters."""
        pass


class CompressionObjectiveEvaluator(ObjectiveEvaluator):
    """Evaluates compression-specific objectives."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: nn.Module,
        compressed_model: nn.Module,
        params: Dict[str, Any],
        validation_data: Any = None
    ) -> Dict[str, float]:
        """Evaluate compression objectives."""

        # Model size metrics
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        compressed_size = sum(p.numel() * p.element_size() for p in compressed_model.parameters())

        # Parameter count metrics
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())

        # Basic metrics
        objectives = {
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else 1.0,
            'parameter_reduction': (original_params - compressed_params) / original_params,
            'model_size_mb': compressed_size / (1024 ** 2)
        }

        # Accuracy evaluation if validation data provided
        if validation_data is not None:
            original_accuracy = self._evaluate_accuracy(model, validation_data)
            compressed_accuracy = self._evaluate_accuracy(compressed_model, validation_data)

            objectives['accuracy_retention'] = compressed_accuracy / original_accuracy if original_accuracy > 0 else 0.0
            objectives['accuracy_drop'] = original_accuracy - compressed_accuracy
            objectives['compressed_accuracy'] = compressed_accuracy

        # Performance estimation
        objectives.update(self._estimate_performance_metrics(model, compressed_model))

        return objectives

    def _evaluate_accuracy(self, model: nn.Module, validation_data: Any) -> float:
        """Evaluate model accuracy on validation data."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in validation_data:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None

                if targets is None:
                    continue

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        return correct / total if total > 0 else 0.0

    def _estimate_performance_metrics(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module
    ) -> Dict[str, float]:
        """Estimate performance improvements."""

        # Simple latency estimation based on parameter count
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())

        param_ratio = compressed_params / original_params if original_params > 0 else 1.0

        # Rough estimates
        latency_improvement = 1.0 / param_ratio if param_ratio > 0 else 1.0
        memory_improvement = 1.0 / param_ratio if param_ratio > 0 else 1.0
        throughput_improvement = latency_improvement

        return {
            'estimated_latency_improvement': latency_improvement,
            'estimated_memory_improvement': memory_improvement,
            'estimated_throughput_improvement': throughput_improvement
        }


class HyperparameterOptimizer:
    """Optimizes compression hyperparameters using Optuna."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.study = None
        self.objective_evaluator = CompressionObjectiveEvaluator()

    def optimize(
        self,
        compression_function: Callable,
        model: nn.Module,
        validation_data: Any = None
    ) -> OptimizationResult:
        """
        Optimize compression hyperparameters.

        Args:
            compression_function: Function that takes model and params, returns compressed model
            model: Original model to compress
            validation_data: Validation dataset for accuracy evaluation

        Returns:
            OptimizationResult with best parameters and Pareto front
        """
        self.logger.info(f"Starting hyperparameter optimization with {self.config.n_trials} trials")

        start_time = time.time()

        # Create Optuna study
        study = self._create_study()
        self.study = study

        # Define objective function
        def objective(trial):
            # Sample hyperparameters
            params = self._sample_hyperparameters(trial)

            try:
                # Apply compression with sampled parameters
                model_copy = copy.deepcopy(model)
                compressed_model = compression_function(model_copy, **params)

                # Evaluate objectives
                objective_values = self.objective_evaluator.evaluate(
                    model, compressed_model, params, validation_data
                )

                # Calculate weighted objective for Optuna
                weighted_objective = 0.0
                for obj in self.config.objectives:
                    if obj.name in objective_values:
                        value = objective_values[obj.name]

                        # Apply constraints
                        if obj.constraint:
                            min_val, max_val = obj.constraint
                            if value < min_val or value > max_val:
                                return float('inf') if obj.minimize else float('-inf')

                        # Add to weighted objective
                        weight = obj.weight
                        if not obj.minimize:
                            value = -value  # Convert maximize to minimize

                        weighted_objective += weight * value

                # Store all objective values as user attributes
                for name, value in objective_values.items():
                    trial.set_user_attr(name, value)

                return weighted_objective

            except Exception as e:
                self.logger.error(f"Trial failed: {e}")
                return float('inf')

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout
        )

        execution_time = time.time() - start_time

        # Extract results
        best_params = study.best_params
        best_trial = study.best_trial
        best_values = {name: best_trial.user_attrs[name]
                      for name in best_trial.user_attrs}

        # Find Pareto front
        pareto_front = self._extract_pareto_front(study)

        # Optimization history
        optimization_history = self._extract_optimization_history(study)

        # Study statistics
        study_statistics = {
            'n_trials': len(study.trials),
            'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'n_failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'best_value': study.best_value
        }

        result = OptimizationResult(
            best_params=best_params,
            best_values=best_values,
            pareto_front=pareto_front,
            optimization_history=optimization_history,
            study_statistics=study_statistics,
            execution_time=execution_time
        )

        self.logger.info(f"Optimization completed in {execution_time:.2f}s")
        self.logger.info(f"Best objective value: {study.best_value:.4f}")

        return result

    def _create_study(self) -> optuna.Study:
        """Create Optuna study with specified configuration."""
        # Configure sampler
        sampler_map = {
            'TPE': optuna.samplers.TPESampler(),
            'CmaEs': optuna.samplers.CmaEsSampler(),
            'Random': optuna.samplers.RandomSampler()
        }
        sampler = sampler_map.get(self.config.sampler, optuna.samplers.TPESampler())

        # Configure pruner
        pruner_map = {
            'median': optuna.pruners.MedianPruner(),
            'percentile': optuna.pruners.PercentilePruner(25.0),
            'hyperband': optuna.pruners.HyperbandPruner()
        }
        pruner = pruner_map.get(self.config.pruner, optuna.pruners.MedianPruner())

        return optuna.create_study(
            study_name=self.config.study_name,
            direction='minimize',  # Always minimize (convert maximize objectives to minimize)
            sampler=sampler,
            pruner=pruner
        )

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}

        for param_name, param_config in self.config.search_space.items():
            if param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )

        return params

    def _extract_pareto_front(self, study: optuna.Study) -> List[Dict[str, Any]]:
        """Extract Pareto front from multi-objective optimization."""
        pareto_solutions = []

        if len(self.config.objectives) <= 1:
            # Single objective - just return best trial
            if study.best_trial:
                pareto_solutions.append({
                    'params': study.best_params,
                    'values': study.best_trial.user_attrs
                })
        else:
            # Multi-objective - find Pareto front
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

            if complete_trials:
                # Extract objective values
                objective_names = [obj.name for obj in self.config.objectives]
                objective_values = []
                trial_data = []

                for trial in complete_trials:
                    values = []
                    for obj in self.config.objectives:
                        if obj.name in trial.user_attrs:
                            value = trial.user_attrs[obj.name]
                            # Convert maximize to minimize for Pareto front calculation
                            if not obj.minimize:
                                value = -value
                            values.append(value)
                        else:
                            values = None
                            break

                    if values:
                        objective_values.append(values)
                        trial_data.append({
                            'params': trial.params,
                            'values': trial.user_attrs
                        })

                if objective_values:
                    # Find Pareto front indices
                    objective_array = np.array(objective_values)
                    pareto_indices = self._find_pareto_front(objective_array)

                    # Extract Pareto solutions
                    pareto_solutions = [trial_data[i] for i in pareto_indices]

        return pareto_solutions

    def _find_pareto_front(self, objective_values: np.ndarray) -> List[int]:
        """Find Pareto front indices."""
        n_points = objective_values.shape[0]
        pareto_front = []

        for i in range(n_points):
            is_pareto = True
            for j in range(n_points):
                if i != j:
                    # Check if j dominates i
                    if all(objective_values[j] <= objective_values[i]) and \
                       any(objective_values[j] < objective_values[i]):
                        is_pareto = False
                        break

            if is_pareto:
                pareto_front.append(i)

        return pareto_front

    def _extract_optimization_history(self, study: optuna.Study) -> List[Dict[str, Any]]:
        """Extract optimization history."""
        history = []

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'user_attrs': trial.user_attrs,
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                    'duration': trial.duration.total_seconds() if trial.duration else None
                })

        return sorted(history, key=lambda x: x['trial_number'])


class AdaptiveCompressionOptimizer:
    """Adaptive compression that adjusts based on performance feedback."""

    def __init__(self, target_metrics: Dict[str, float]):
        self.target_metrics = target_metrics
        self.adaptation_history = []
        self.logger = logging.getLogger(__name__)

    def adaptive_compress(
        self,
        model: nn.Module,
        compression_functions: Dict[str, Callable],
        validation_data: Any,
        max_iterations: int = 10
    ) -> Tuple[nn.Module, List[Dict[str, Any]]]:
        """
        Apply adaptive compression that iteratively adjusts parameters.

        Args:
            model: Original model to compress
            compression_functions: Dict of compression technique -> function
            validation_data: Validation data for feedback
            max_iterations: Maximum adaptation iterations

        Returns:
            Tuple of (best_model, adaptation_history)
        """
        self.logger.info("Starting adaptive compression optimization")

        current_model = copy.deepcopy(model)
        best_model = current_model
        best_score = float('-inf')

        evaluator = CompressionObjectiveEvaluator()

        for iteration in range(max_iterations):
            self.logger.info(f"Adaptation iteration {iteration + 1}/{max_iterations}")

            # Evaluate current model
            current_metrics = evaluator.evaluate(
                model, current_model, {}, validation_data
            )

            # Calculate adaptation score
            adaptation_score = self._calculate_adaptation_score(current_metrics)

            # Store adaptation history
            self.adaptation_history.append({
                'iteration': iteration,
                'metrics': current_metrics.copy(),
                'adaptation_score': adaptation_score
            })

            # Check if this is the best model so far
            if adaptation_score > best_score:
                best_score = adaptation_score
                best_model = copy.deepcopy(current_model)

            # Check convergence
            if self._check_convergence():
                self.logger.info(f"Converged at iteration {iteration + 1}")
                break

            # Adapt compression parameters
            next_params = self._adapt_compression_parameters(current_metrics)

            # Apply adapted compression
            next_model = self._apply_adaptive_compression(
                model, compression_functions, next_params
            )

            current_model = next_model

        return best_model, self.adaptation_history

    def _calculate_adaptation_score(self, metrics: Dict[str, float]) -> float:
        """Calculate adaptation score based on target metrics."""
        score = 0.0
        total_weight = 0.0

        for metric_name, target_value in self.target_metrics.items():
            if metric_name in metrics:
                current_value = metrics[metric_name]

                # Calculate normalized score (closer to target = higher score)
                if target_value > 0:
                    normalized_score = min(current_value / target_value, target_value / current_value)
                else:
                    normalized_score = 1.0 - abs(current_value - target_value)

                score += normalized_score
                total_weight += 1.0

        return score / total_weight if total_weight > 0 else 0.0

    def _check_convergence(self, window: int = 3, threshold: float = 0.01) -> bool:
        """Check if adaptation has converged."""
        if len(self.adaptation_history) < window:
            return False

        recent_scores = [h['adaptation_score'] for h in self.adaptation_history[-window:]]
        score_variance = np.var(recent_scores)

        return score_variance < threshold

    def _adapt_compression_parameters(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Adapt compression parameters based on current metrics."""
        adapted_params = {}

        # Adaptive sparsity based on accuracy
        if 'accuracy_retention' in current_metrics:
            accuracy_retention = current_metrics['accuracy_retention']
            target_accuracy = self.target_metrics.get('accuracy_retention', 0.95)

            if accuracy_retention < target_accuracy:
                # Reduce compression intensity
                adapted_params['sparsity_ratio'] = max(0.1,
                    adapted_params.get('sparsity_ratio', 0.5) * 0.8)
            else:
                # Increase compression intensity
                adapted_params['sparsity_ratio'] = min(0.9,
                    adapted_params.get('sparsity_ratio', 0.5) * 1.2)

        # Adaptive quantization bits based on accuracy drop
        if 'accuracy_drop' in current_metrics:
            accuracy_drop = current_metrics['accuracy_drop']
            target_drop = self.target_metrics.get('accuracy_drop', 0.05)

            if accuracy_drop > target_drop:
                # Use higher precision
                adapted_params['quantization_bits'] = min(16,
                    adapted_params.get('quantization_bits', 8) + 1)
            else:
                # Use lower precision
                adapted_params['quantization_bits'] = max(4,
                    adapted_params.get('quantization_bits', 8) - 1)

        return adapted_params

    def _apply_adaptive_compression(
        self,
        model: nn.Module,
        compression_functions: Dict[str, Callable],
        params: Dict[str, Any]
    ) -> nn.Module:
        """Apply compression with adapted parameters."""
        compressed_model = copy.deepcopy(model)

        # Apply compressions in sequence
        for compression_name, compression_func in compression_functions.items():
            try:
                # Extract relevant parameters for this compression
                relevant_params = {k: v for k, v in params.items()
                                 if compression_name in k or k in ['sparsity_ratio', 'quantization_bits']}

                compressed_model = compression_func(compressed_model, **relevant_params)

            except Exception as e:
                self.logger.error(f"Failed to apply {compression_name}: {e}")

        return compressed_model


class MultiObjectiveOptimizer:
    """Multi-objective optimization for compression trade-offs."""

    def __init__(self, objectives: List[OptimizationObjective]):
        self.objectives = objectives
        self.logger = logging.getLogger(__name__)

    def optimize_pareto_front(
        self,
        compression_function: Callable,
        model: nn.Module,
        search_space: Dict[str, Any],
        validation_data: Any,
        n_points: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Find Pareto-optimal solutions for multi-objective compression.

        Args:
            compression_function: Function to apply compression
            model: Original model
            search_space: Parameter search space
            validation_data: Validation data
            n_points: Number of Pareto points to find

        Returns:
            List of Pareto-optimal solutions
        """
        self.logger.info(f"Finding Pareto front with {n_points} points")

        evaluator = CompressionObjectiveEvaluator()
        solutions = []

        # Generate random parameter combinations
        for i in range(n_points * 2):  # Generate more than needed
            params = self._sample_random_params(search_space)

            try:
                # Apply compression
                model_copy = copy.deepcopy(model)
                compressed_model = compression_function(model_copy, **params)

                # Evaluate objectives
                objective_values = evaluator.evaluate(
                    model, compressed_model, params, validation_data
                )

                solutions.append({
                    'params': params,
                    'objectives': objective_values,
                    'model': compressed_model
                })

            except Exception as e:
                self.logger.error(f"Solution {i} failed: {e}")

        # Filter to Pareto front
        pareto_solutions = self._filter_to_pareto_front(solutions)

        # Limit to requested number of points
        if len(pareto_solutions) > n_points:
            pareto_solutions = self._select_diverse_solutions(pareto_solutions, n_points)

        self.logger.info(f"Found {len(pareto_solutions)} Pareto-optimal solutions")

        return pareto_solutions

    def _sample_random_params(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}

        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                low, high = param_config['low'], param_config['high']
                if param_config.get('log', False):
                    params[param_name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                else:
                    params[param_name] = np.random.uniform(low, high)

            elif param_config['type'] == 'int':
                params[param_name] = np.random.randint(
                    param_config['low'], param_config['high'] + 1
                )

            elif param_config['type'] == 'categorical':
                params[param_name] = np.random.choice(param_config['choices'])

        return params

    def _filter_to_pareto_front(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter solutions to Pareto front."""
        pareto_solutions = []

        for i, solution_i in enumerate(solutions):
            is_dominated = False

            for j, solution_j in enumerate(solutions):
                if i != j and self._dominates(solution_j, solution_i):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_solutions.append(solution_i)

        return pareto_solutions

    def _dominates(self, solution_a: Dict[str, Any], solution_b: Dict[str, Any]) -> bool:
        """Check if solution A dominates solution B."""
        better_or_equal_all = True
        strictly_better_at_least_one = False

        for objective in self.objectives:
            if objective.name not in solution_a['objectives'] or \
               objective.name not in solution_b['objectives']:
                continue

            value_a = solution_a['objectives'][objective.name]
            value_b = solution_b['objectives'][objective.name]

            if objective.minimize:
                if value_a > value_b:
                    better_or_equal_all = False
                    break
                elif value_a < value_b:
                    strictly_better_at_least_one = True
            else:
                if value_a < value_b:
                    better_or_equal_all = False
                    break
                elif value_a > value_b:
                    strictly_better_at_least_one = True

        return better_or_equal_all and strictly_better_at_least_one

    def _select_diverse_solutions(
        self,
        solutions: List[Dict[str, Any]],
        n_points: int
    ) -> List[Dict[str, Any]]:
        """Select diverse subset of solutions."""
        # Simple diversity selection - could be improved
        if len(solutions) <= n_points:
            return solutions

        # Sort by first objective and select evenly spaced solutions
        first_objective = self.objectives[0].name
        sorted_solutions = sorted(
            solutions,
            key=lambda x: x['objectives'].get(first_objective, 0)
        )

        indices = np.linspace(0, len(sorted_solutions) - 1, n_points, dtype=int)
        return [sorted_solutions[i] for i in indices]


if __name__ == "__main__":
    # Example usage
    import torch.nn as nn

    # Create test model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # Define optimization objectives
    objectives = [
        OptimizationObjective('compression_ratio', weight=0.4, minimize=False),
        OptimizationObjective('accuracy_retention', weight=0.6, minimize=False, constraint=(0.9, 1.0))
    ]

    # Define search space
    search_space = {
        'sparsity_ratio': {'type': 'float', 'low': 0.1, 'high': 0.9},
        'quantization_bits': {'type': 'int', 'low': 4, 'high': 16}
    }

    # Create optimization configuration
    config = OptimizationConfig(
        objectives=objectives,
        search_space=search_space,
        n_trials=50,
        sampler='TPE'
    )

    print("Compression optimization setup complete")
    print(f"Objectives: {len(objectives)}")
    print(f"Search space parameters: {list(search_space.keys())}")
    print(f"Number of trials: {config.n_trials}")