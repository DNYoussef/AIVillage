#!/usr/bin/env python3
"""
Cognate EvoMerge: Evolutionary Breeding of 25M Parameter Models

Takes the three trained 25M Cognate models (Alpha, Beta, Gamma) and evolves them
using sophisticated model merging techniques to create optimized offspring.

Breeding Strategy:
- Initial: 3 parent models → 8 first-generation candidates  
- Evolution: Top 2 → 6 children, Bottom 6 → 2 children (groups of 3)
- Techniques: Linear, SLERP, TIES, DARE, Frankenmerge, DFS
- Evaluation: Code, Math, Multilingual, Structured Data domains
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Local imports
from unified_refiner.refiner import UnifiedRefiner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CognateEvoConfig:
    """Configuration for Cognate EvoMerge process."""
    
    # Input models (our trained 25M models)
    alpha_model_path: str = "./alpha_25m_model"
    beta_model_path: str = "./beta_25m_model" 
    gamma_model_path: str = "./gamma_25m_model"
    
    # Output configuration
    output_dir: str = "./cognate_evomerge_output"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Evolution parameters  
    generations: int = 10
    population_size: int = 8
    
    # Evaluation domains
    evaluation_domains: list[str] = field(default_factory=lambda: ["code", "math", "reasoning", "language"])
    fitness_weights: dict[str, float] = field(default_factory=lambda: {
        "code": 0.25, "math": 0.25, "reasoning": 0.25, "language": 0.25
    })


@dataclass
class CognateCandidate:
    """Represents a merged Cognate model candidate."""
    
    model_path: str
    merge_recipe: dict[str, Any]
    fitness_scores: dict[str, float] = field(default_factory=dict)
    aggregated_fitness: float = 0.0
    generation: int = 0
    parents: list[str] | None = None
    
    def calculate_aggregated_fitness(self, weights: dict[str, float]) -> float:
        """Calculate weighted aggregate fitness score."""
        total_fitness = 0.0
        valid_weights_sum = 0.0
        
        for domain, weight in weights.items():
            score = self.fitness_scores.get(domain, 0)
            if not np.isnan(score):
                total_fitness += score * weight
                valid_weights_sum += weight
        
        if valid_weights_sum > 0:
            self.aggregated_fitness = total_fitness / valid_weights_sum
        else:
            self.aggregated_fitness = 0.0
            
        return self.aggregated_fitness


class CognateMergeOperators:
    """Specialized merge operators for Cognate models."""
    
    @staticmethod
    def linear_merge(models: list[UnifiedRefiner], weights: list[float] | None = None) -> UnifiedRefiner:
        """Linear weighted average of Cognate model parameters."""
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Use the first model's config as base
        merged_config = models[0].config
        merged = UnifiedRefiner(merged_config)
        merged_state = {}
        
        logger.info(f"Linear merging {len(models)} models with weights {weights}")
        
        for key in models[0].state_dict():
            # Skip if any model doesn't have this parameter
            if not all(key in model.state_dict() for model in models):
                logger.warning(f"Skipping parameter {key} - not present in all models")
                continue
                
            # Weighted average of parameters
            merged_param = torch.zeros_like(models[0].state_dict()[key])
            for weight, model in zip(weights, models):
                merged_param += weight * model.state_dict()[key]
            
            merged_state[key] = merged_param
        
        merged.load_state_dict(merged_state, strict=False)
        return merged
    
    @staticmethod
    def slerp_merge(model1: UnifiedRefiner, model2: UnifiedRefiner, t: float = 0.5) -> UnifiedRefiner:
        """Spherical linear interpolation between two Cognate models."""
        merged_config = model1.config
        merged = UnifiedRefiner(merged_config)
        merged_state = {}
        
        logger.info(f"SLERP merging with t={t}")
        
        for key in model1.state_dict():
            if key not in model2.state_dict():
                logger.warning(f"Skipping parameter {key} - not present in model2")
                continue
                
            w1 = model1.state_dict()[key].flatten()
            w2 = model2.state_dict()[key].flatten()
            
            # Compute angle between parameters
            dot_product = torch.clamp(torch.dot(w1, w2) / (w1.norm() * w2.norm()), -1, 1)
            omega = torch.arccos(dot_product)
            so = torch.sin(omega)
            
            if so.abs() < 1e-8:
                # Linear interpolation fallback
                merged_param = (1 - t) * w1 + t * w2
            else:
                # Spherical interpolation
                merged_param = torch.sin((1 - t) * omega) / so * w1 + torch.sin(t * omega) / so * w2
            
            merged_state[key] = merged_param.view(model1.state_dict()[key].shape)
        
        merged.load_state_dict(merged_state, strict=False)
        return merged
    
    @staticmethod
    def ties_merge(models: list[UnifiedRefiner], threshold: float = 0.1) -> UnifiedRefiner:
        """TIES merging: Trim, Interpolate, Elect, Sign."""
        merged_config = models[0].config
        merged = UnifiedRefiner(merged_config)
        merged_state = {}
        
        logger.info(f"TIES merging {len(models)} models with threshold {threshold}")
        
        for key in models[0].state_dict():
            if not all(key in model.state_dict() for model in models):
                continue
            
            params = [model.state_dict()[key] for model in models]
            
            # Trim: Remove small magnitude changes
            trimmed_params = []
            for param in params:
                mask = torch.abs(param) > threshold
                trimmed = param * mask
                trimmed_params.append(trimmed)
            
            # Sign election: Choose dominant sign
            signs = torch.sign(sum(trimmed_params))
            
            # Interpolate with sign correction
            merged_param = torch.zeros_like(params[0])
            for p in trimmed_params:
                merged_param += torch.abs(p)
            merged_param = merged_param / len(trimmed_params) * signs
            
            merged_state[key] = merged_param
        
        merged.load_state_dict(merged_state, strict=False)
        return merged
    
    @staticmethod
    def dare_merge(models: list[UnifiedRefiner], threshold: float = 0.2, amplification: float = 2.0) -> UnifiedRefiner:
        """DARE merging: Drop And REscale."""
        merged_config = models[0].config
        merged = UnifiedRefiner(merged_config)
        merged_state = {}
        
        logger.info(f"DARE merging {len(models)} models with threshold {threshold}, amplification {amplification}")
        
        for key in models[0].state_dict():
            if not all(key in model.state_dict() for model in models):
                continue
                
            params = [model.state_dict()[key] for model in models]
            
            # Random dropout mask
            mask = torch.rand_like(params[0]) > threshold
            
            # Merge with rescaling
            merged_param = torch.zeros_like(params[0])
            for p in params:
                merged_param += p * mask * amplification
            merged_param = merged_param / len(params)
            
            merged_state[key] = merged_param
        
        merged.load_state_dict(merged_state, strict=False)
        return merged
    
    @staticmethod
    def frankenmerge(models: list[UnifiedRefiner], layer_assignments: list[int] | None = None) -> UnifiedRefiner:
        """Frankenmerge: Mix layers from different models."""
        merged_config = models[0].config
        merged = UnifiedRefiner(merged_config)
        merged_state = {}
        
        # Random layer assignment if not provided
        if layer_assignments is None:
            num_layers = merged_config.n_layers
            layer_assignments = [random.randint(0, len(models) - 1) for _ in range(num_layers)]
        
        logger.info(f"Frankenmerge with layer assignments: {layer_assignments}")
        
        for key in models[0].state_dict():
            # Determine which model to use based on layer
            layer_idx = 0
            if "layers." in key:
                try:
                    layer_idx = int(key.split("layers.")[1].split(".")[0])
                except (IndexError, ValueError):
                    pass
            
            model_idx = layer_assignments[layer_idx % len(layer_assignments)]
            if key in models[model_idx].state_dict():
                merged_state[key] = models[model_idx].state_dict()[key].clone()
        
        merged.load_state_dict(merged_state, strict=False)
        return merged
    
    @staticmethod
    def dfs_merge(models: list[UnifiedRefiner], merge_ratio: float = 0.3) -> UnifiedRefiner:
        """DFS merge: Hierarchical merging."""
        if len(models) == 1:
            return models[0]
        
        # Recursively merge pairs
        mid = len(models) // 2
        left = CognateMergeOperators.dfs_merge(models[:mid], merge_ratio)
        right = CognateMergeOperators.dfs_merge(models[mid:], merge_ratio)
        
        # Merge the two halves
        return CognateMergeOperators.slerp_merge(left, right, merge_ratio)


class CognateEvaluator:
    """Evaluation system for merged Cognate models."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Simple evaluation domains using model perplexity
        self.test_prompts = {
            "code": [
                "def fibonacci(n):\n    if n <= 1: return n",
                "class Calculator:\n    def __init__(self):",
                "import numpy as np\narray = np.array([1, 2, 3])"
            ],
            "math": [
                "Solve for x: 2x + 5 = 13",
                "What is the derivative of x^2 + 3x?",
                "Calculate the area of a circle with radius 5"
            ],
            "reasoning": [
                "If all cats are animals, and Fluffy is a cat, then",
                "Given the premises: All birds can fly. Penguins are birds.",
                "Logic problem: If A implies B, and B implies C, then"
            ],
            "language": [
                "Complete this sentence: The weather today is",
                "Translate to Spanish: Hello, how are you?",
                "What is the meaning of the word 'serendipity'?"
            ]
        }
    
    def evaluate_model(self, model: UnifiedRefiner) -> dict[str, float]:
        """Evaluate a Cognate model across all domains."""
        model.eval()
        scores = {}
        
        with torch.no_grad():
            for domain, prompts in self.test_prompts.items():
                domain_scores = []
                
                for prompt in prompts:
                    try:
                        # Simple tokenization for evaluation
                        tokens = self._tokenize(prompt)
                        input_ids = torch.tensor([tokens], device=self.device)
                        attention_mask = torch.ones_like(input_ids)
                        
                        # Forward pass
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
                        
                        # Calculate perplexity-based score
                        logits = outputs['logits']
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        
                        loss = nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            reduction='mean'
                        )
                        
                        # Convert loss to score (lower loss = higher score)
                        score = torch.sigmoid(-loss + 5).item()
                        domain_scores.append(score)
                        
                    except Exception as e:
                        logger.warning(f"Evaluation failed for {domain} prompt: {e}")
                        domain_scores.append(0.0)
                
                scores[domain] = np.mean(domain_scores) if domain_scores else 0.0
        
        return scores
    
    def _tokenize(self, text: str) -> list[int]:
        """Simple tokenization for evaluation."""
        bytes_data = text.encode('utf-8')
        tokens = [min(b + 3, 31999) for b in bytes_data[:512]]
        return [1] + tokens + [2]  # BOS + content + EOS


class CognateEvoMerge:
    """Main Cognate EvoMerge evolutionary breeding system."""
    
    def __init__(self, config: CognateEvoConfig):
        self.config = config
        self.device = config.device
        self.merge_ops = CognateMergeOperators()
        self.evaluator = CognateEvaluator(self.device)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evolution tracking
        self.fitness_history = []
        self.convergence_patience = 10  # Stop if no improvement for 10 generations
        self.plateau_counter = 0
        
        logger.info(f"🧬 Cognate EvoMerge initialized on {self.device}")
        logger.info(f"🎯 Target generations: {config.generations}")
        logger.info(f"📊 Convergence patience: {self.convergence_patience}")
    
    def load_parent_models(self) -> list[UnifiedRefiner]:
        """Load the three ACTUAL trained 25M parameter models as parents."""
        logger.info("📂 Loading ACTUAL trained parent models...")
        
        models = []
        trained_models_dir = Path("./trained_25m_models")
        
        if not trained_models_dir.exists():
            logger.error("❌ Trained models directory not found!")
            logger.error("   Please run train_and_save_25m_models.py first to create the actual trained models")
            raise FileNotFoundError("Trained models not found. Run training script first.")
        
        for model_name in ["Alpha_25M", "Beta_25M", "Gamma_25M"]:
            model_path = trained_models_dir / model_name
            
            if not model_path.exists():
                logger.error(f"❌ {model_name} not found at {model_path}")
                continue
                
            try:
                # Load config
                config = torch.load(model_path / "config.pt", map_location='cpu')
                
                # Create model
                model = UnifiedRefiner(config).to(self.device)
                
                # Load trained weights
                state_dict = torch.load(model_path / "model.pt", map_location=self.device)
                model.load_state_dict(state_dict)
                
                # Load training metadata
                with open(model_path / "training_results.json") as f:
                    training_results = json.load(f)
                
                models.append(model)
                
                logger.info(f"✅ Loaded TRAINED {model_name}:")
                logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
                logger.info(f"   Final Loss: {training_results.get('final_loss', 0):.4f}")
                logger.info(f"   Loss Improvement: {training_results.get('loss_improvement', 0):.4f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")
                continue
        
        if len(models) != 3:
            logger.error(f"❌ Only loaded {len(models)}/3 trained models!")
            raise RuntimeError("Failed to load all trained parent models")
            
        logger.info("🎉 Successfully loaded all 3 TRAINED parent models for breeding!")
        return models
    
    def generate_initial_population(self, parent_models: list[UnifiedRefiner]) -> list[CognateCandidate]:
        """Generate 8 initial candidates using different merge techniques."""
        logger.info("🔬 Generating initial population of 8 candidates...")
        
        population = []
        
        # 8 different merge technique combinations
        merge_recipes = [
            {"primary": "linear", "weights": [0.4, 0.4, 0.2]},
            {"primary": "linear", "weights": [0.33, 0.33, 0.34]},
            {"primary": "slerp", "models": [0, 1], "t": 0.5},
            {"primary": "slerp", "models": [0, 2], "t": 0.6},
            {"primary": "ties", "threshold": 0.1},
            {"primary": "ties", "threshold": 0.2},
            {"primary": "dare", "threshold": 0.15, "amplification": 2.0},
            {"primary": "frankenmerge", "random_layers": True}
        ]
        
        for i, recipe in enumerate(merge_recipes):
            try:
                logger.info(f"Creating candidate {i+1}/8: {recipe}")
                
                # Apply merge technique based on recipe
                if recipe["primary"] == "linear":
                    merged = self.merge_ops.linear_merge(parent_models, recipe["weights"])
                elif recipe["primary"] == "slerp":
                    model_indices = recipe["models"]
                    merged = self.merge_ops.slerp_merge(
                        parent_models[model_indices[0]], 
                        parent_models[model_indices[1]], 
                        recipe["t"]
                    )
                elif recipe["primary"] == "ties":
                    merged = self.merge_ops.ties_merge(parent_models, recipe["threshold"])
                elif recipe["primary"] == "dare":
                    merged = self.merge_ops.dare_merge(
                        parent_models, 
                        recipe["threshold"], 
                        recipe["amplification"]
                    )
                elif recipe["primary"] == "frankenmerge":
                    merged = self.merge_ops.frankenmerge(parent_models)
                
                # Save merged model
                model_path = self.output_dir / f"gen0_candidate_{i+1}"
                model_path.mkdir(exist_ok=True)
                
                # Save model state dict
                torch.save(merged.state_dict(), model_path / "model.pt")
                torch.save(merged.config, model_path / "config.pt")
                
                candidate = CognateCandidate(
                    model_path=str(model_path),
                    merge_recipe=recipe,
                    generation=0,
                    parents=["Alpha_25M", "Beta_25M", "Gamma_25M"]
                )
                
                population.append(candidate)
                logger.info(f"✅ Created candidate {i+1}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create candidate {i+1}: {e}")
                continue
        
        logger.info(f"🧬 Initial population: {len(population)} candidates created")
        return population
    
    def evaluate_population(self, population: list[CognateCandidate]) -> None:
        """Evaluate all candidates in the population."""
        logger.info("📊 Evaluating population fitness...")
        
        for i, candidate in enumerate(population):
            if candidate.fitness_scores:
                logger.info(f"Candidate {i+1} already evaluated (fitness: {candidate.aggregated_fitness:.4f})")
                continue
                
            try:
                logger.info(f"Evaluating candidate {i+1}/{len(population)}...")
                
                # Load model
                model_path = Path(candidate.model_path)
                config = torch.load(model_path / "config.pt")
                model = UnifiedRefiner(config).to(self.device)
                model.load_state_dict(torch.load(model_path / "model.pt", map_location=self.device))
                
                # Evaluate across domains
                scores = self.evaluator.evaluate_model(model)
                candidate.fitness_scores = scores
                candidate.calculate_aggregated_fitness(self.config.fitness_weights)
                
                logger.info(f"✅ Candidate {i+1} fitness: {candidate.aggregated_fitness:.4f}")
                for domain, score in scores.items():
                    logger.info(f"   {domain}: {score:.4f}")
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate candidate {i+1}: {e}")
                candidate.fitness_scores = {d: 0.0 for d in self.config.evaluation_domains}
                candidate.aggregated_fitness = 0.0
    
    def create_next_generation(self, population: list[CognateCandidate], 
                             parent_models: list[UnifiedRefiner]) -> list[CognateCandidate]:
        """Create next generation using breeding algorithm."""
        logger.info("🧬 Creating next generation...")
        
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.aggregated_fitness, reverse=True)
        
        # Top 2 winners → 6 children (3 each)
        top_2 = sorted_pop[:2]
        logger.info(f"🏆 Top 2 winners: {top_2[0].aggregated_fitness:.4f}, {top_2[1].aggregated_fitness:.4f}")
        
        next_gen = []
        
        # Create 3 children from each winner
        for i, winner in enumerate(top_2):
            for child_idx in range(3):
                try:
                    # Load winner model
                    model_path = Path(winner.model_path)
                    config = torch.load(model_path / "config.pt")
                    winner_model = UnifiedRefiner(config).to(self.device)
                    winner_model.load_state_dict(torch.load(model_path / "model.pt", map_location=self.device))
                    
                    # Select random parent for crossover
                    parent_model = random.choice(parent_models)
                    
                    # Apply random merge technique
                    techniques = ["linear", "slerp", "ties", "dare"]
                    technique = random.choice(techniques)
                    
                    if technique == "linear":
                        weight = random.uniform(0.6, 0.8)  # Favor winner
                        child = self.merge_ops.linear_merge([winner_model, parent_model], [weight, 1-weight])
                    elif technique == "slerp":
                        t = random.uniform(0.3, 0.7)
                        child = self.merge_ops.slerp_merge(winner_model, parent_model, t)
                    elif technique == "ties":
                        threshold = random.uniform(0.05, 0.2)
                        child = self.merge_ops.ties_merge([winner_model, parent_model], threshold)
                    else:  # dare
                        child = self.merge_ops.dare_merge([winner_model, parent_model])
                    
                    # Save child
                    child_path = self.output_dir / f"gen1_winner{i+1}_child{child_idx+1}"
                    child_path.mkdir(exist_ok=True)
                    torch.save(child.state_dict(), child_path / "model.pt")
                    torch.save(child.config, child_path / "config.pt")
                    
                    candidate = CognateCandidate(
                        model_path=str(child_path),
                        merge_recipe={
                            "type": "winner_child",
                            "parent": winner.model_path,
                            "technique": technique,
                            "crossover_parent": f"parent_{parent_models.index(parent_model)}"
                        },
                        generation=1,
                        parents=[winner.model_path]
                    )
                    
                    next_gen.append(candidate)
                    logger.info(f"✅ Created winner {i+1} child {child_idx+1}")
                    
                    # Cleanup
                    del winner_model, child
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create winner child: {e}")
                    continue
        
        # Bottom 6 → 2 children (groups of 3 → 1 child each)  
        bottom_6 = sorted_pop[-6:] if len(sorted_pop) >= 6 else sorted_pop[2:]
        
        for group_idx in range(2):
            try:
                # Select 3 models for this group
                group_start = group_idx * 3
                group_models = []
                
                for j in range(3):
                    model_idx = (group_start + j) % len(bottom_6)
                    model_path = Path(bottom_6[model_idx].model_path)
                    config = torch.load(model_path / "config.pt")
                    model = UnifiedRefiner(config).to(self.device)
                    model.load_state_dict(torch.load(model_path / "model.pt", map_location=self.device))
                    group_models.append(model)
                
                # Merge all 3 with equal weights
                child = self.merge_ops.linear_merge(group_models, [1/3, 1/3, 1/3])
                
                # Save child
                child_path = self.output_dir / f"gen1_loser_group{group_idx+1}"
                child_path.mkdir(exist_ok=True)
                torch.save(child.state_dict(), child_path / "model.pt")
                torch.save(child.config, child_path / "config.pt")
                
                candidate = CognateCandidate(
                    model_path=str(child_path),
                    merge_recipe={
                        "type": "loser_child",
                        "parents": [bottom_6[(group_start + j) % len(bottom_6)].model_path for j in range(3)],
                        "technique": "linear_merge_3way"
                    },
                    generation=1,
                    parents=[bottom_6[(group_start + j) % len(bottom_6)].model_path for j in range(3)]
                )
                
                next_gen.append(candidate)
                logger.info(f"✅ Created loser group {group_idx+1} child")
                
                # Cleanup
                for model in group_models:
                    del model
                del child
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"❌ Failed to create loser child: {e}")
                continue
        
        logger.info(f"🧬 Next generation: {len(next_gen)} candidates created")
        return next_gen
    
    def save_checkpoint(self, generation: int, population: list[CognateCandidate]):
        """Save evolution checkpoint for resuming long runs."""
        checkpoint_data = {
            "generation": generation,
            "fitness_history": self.fitness_history,
            "plateau_counter": self.plateau_counter,
            "population": [
                {
                    "path": candidate.model_path,
                    "fitness": candidate.aggregated_fitness,
                    "scores": candidate.fitness_scores,
                    "recipe": candidate.merge_recipe,
                    "generation": candidate.generation
                } for candidate in population
            ]
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_gen{generation+1}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"📁 Checkpoint saved: {checkpoint_path}")
    
    def cleanup_old_models(self, current_generation: int):
        """
        Clean up models from n-2 generations to save disk space.
        Starting from generation 3, delete models from generation n-2.
        """
        cleanup_generation = current_generation - 2
        
        if cleanup_generation < 0:
            return  # No cleanup needed yet
            
        logger.info(f"🧹 Cleaning up generation {cleanup_generation} models (n-2 cleanup)")
        
        cleanup_patterns = [
            f"gen{cleanup_generation}_*",
        ]
        
        cleanup_count = 0
        total_size_freed = 0
        
        for pattern in cleanup_patterns:
            for model_dir in self.output_dir.glob(pattern):
                if model_dir.is_dir():
                    try:
                        # Calculate size before deletion
                        dir_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                        total_size_freed += dir_size
                        
                        # Delete the directory
                        import shutil
                        shutil.rmtree(model_dir)
                        cleanup_count += 1
                        logger.debug(f"   Deleted: {model_dir.name}")
                        
                    except Exception as e:
                        logger.warning(f"   Failed to delete {model_dir.name}: {e}")
        
        if cleanup_count > 0:
            size_mb = total_size_freed / (1024 * 1024)
            logger.info(f"🧹 Cleaned up {cleanup_count} model directories from generation {cleanup_generation}")
            logger.info(f"💾 Freed {size_mb:.1f}MB of disk space")
        else:
            logger.debug(f"🧹 No models to clean up from generation {cleanup_generation}")
    
    def run_evolution(self) -> dict[str, Any]:
        """Run the complete EvoMerge evolution process."""
        logger.info("🚀 Starting Cognate EvoMerge Evolution")
        logger.info("=" * 60)
        
        # Load parent models  
        parent_models = self.load_parent_models()
        
        # Generate initial population (8 candidates)
        population = self.generate_initial_population(parent_models)
        
        # Evaluate initial population
        self.evaluate_population(population)
        
        # Evolution loop
        for gen in range(self.config.generations):
            logger.info(f"\n{'='*50}")
            logger.info(f"🧬 Generation {gen + 1}/{self.config.generations}")
            logger.info(f"{'='*50}")
            
            # Sort by fitness and log stats
            population.sort(key=lambda x: x.aggregated_fitness, reverse=True)
            
            best_fitness = population[0].aggregated_fitness
            avg_fitness = np.mean([c.aggregated_fitness for c in population])
            
            logger.info(f"📊 Generation {gen + 1} Stats:")
            logger.info(f"   Best fitness: {best_fitness:.4f}")
            logger.info(f"   Avg fitness:  {avg_fitness:.4f}")
            
            # Track fitness improvement
            if self.fitness_history and best_fitness <= max(self.fitness_history):
                self.plateau_counter += 1
                logger.info(f"   Plateau counter: {self.plateau_counter}/{self.convergence_patience}")
            else:
                self.plateau_counter = 0
                if self.fitness_history:
                    improvement = ((best_fitness - max(self.fitness_history)) / max(self.fitness_history)) * 100
                    logger.info(f"   🚀 Improvement: +{improvement:.2f}%")
            
            self.fitness_history.append(best_fitness)
            
            # Check for convergence (early stopping)
            if self.plateau_counter >= self.convergence_patience:
                logger.info(f"🎯 Early convergence after {gen + 1} generations (no improvement for {self.convergence_patience} generations)")
                break
            
            # Save periodic checkpoints for long runs
            if (gen + 1) % 10 == 0:
                self.save_checkpoint(gen, population)
                logger.info(f"💾 Checkpoint saved at generation {gen + 1}")
            
            # Clean up old models (n-2 generations) to save disk space
            if gen >= 2:  # Starting from generation 3 (index 2)
                self.cleanup_old_models(gen + 1)
            
            # Create next generation  
            if gen < self.config.generations - 1:
                population = self.create_next_generation(population, parent_models)
                self.evaluate_population(population)
        
        # Final results
        population.sort(key=lambda x: x.aggregated_fitness, reverse=True)
        best_model = population[0]
        
        final_generation = len(self.fitness_history)
        peak_fitness = max(self.fitness_history) if self.fitness_history else 0
        total_improvement = ((peak_fitness - self.fitness_history[0]) / self.fitness_history[0] * 100) if self.fitness_history else 0
        
        logger.info("\n🏆 EVOLUTION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"🥇 Best Model: {best_model.model_path}")
        logger.info(f"🎯 Final Fitness: {best_model.aggregated_fitness:.4f}")
        logger.info(f"📈 Peak Fitness: {peak_fitness:.4f}")
        logger.info(f"🚀 Total Improvement: +{total_improvement:.2f}%")
        logger.info(f"🔄 Generations Completed: {final_generation}/{self.config.generations}")
        logger.info("📊 Domain Scores:")
        for domain, score in best_model.fitness_scores.items():
            logger.info(f"   {domain:12s}: {score:.4f}")
        
        # Save comprehensive results
        results = {
            "evolution_summary": {
                "total_generations": final_generation,
                "target_generations": self.config.generations,
                "peak_fitness": peak_fitness,
                "total_improvement_percent": total_improvement,
                "convergence_reason": "early_stopping" if self.plateau_counter >= self.convergence_patience else "completed",
                "total_models_created": final_generation * 8
            },
            "best_model": {
                "path": best_model.model_path,
                "fitness": best_model.aggregated_fitness,
                "scores": best_model.fitness_scores,
                "recipe": best_model.merge_recipe,
                "generation": best_model.generation
            },
            "fitness_history": self.fitness_history,
            "final_population": [
                {
                    "path": c.model_path,
                    "fitness": c.aggregated_fitness,
                    "scores": c.fitness_scores,
                    "recipe": c.merge_recipe,
                    "generation": c.generation
                } for c in population[:5]  # Top 5
            ]
        }
        
        results_path = self.output_dir / "evomerge_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {results_path}")
        
        return results


def main(generations: int = 50):
    """Run Cognate EvoMerge evolution.
    
    Args:
        generations: Number of generations to evolve (default: 50)
    """
    # Configuration for full 50-generation evolution
    config = CognateEvoConfig(
        output_dir="./cognate_evomerge_50gen_output",
        generations=generations,
        population_size=8,
        evaluation_domains=["code", "math", "reasoning", "language"],
        fitness_weights={"code": 0.25, "math": 0.25, "reasoning": 0.25, "language": 0.25}
    )
    
    logger.info("🧬 COGNATE EVOMERGE - 50 GENERATION EVOLUTIONARY BREEDING")
    logger.info("=" * 70)
    logger.info(f"🎯 Target Generations: {generations}")
    logger.info(f"🧬 Population Size: {config.population_size}")
    logger.info(f"📊 Evaluation Domains: {config.evaluation_domains}")
    logger.info(f"⚖️ Domain Weights: {config.fitness_weights}")
    logger.info(f"💾 Output Directory: {config.output_dir}")
    logger.info("=" * 70)
    
    evomerge = CognateEvoMerge(config)
    results = evomerge.run_evolution()
    
    return results

def main_quick(generations: int = 5):
    """Run quick evolution for testing (5 generations)."""
    return main(generations)


if __name__ == "__main__":
    results = main()