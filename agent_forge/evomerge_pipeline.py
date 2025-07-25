#!/usr/bin/env python3
"""
EvoMerge Pipeline - Production-Ready Evolutionary Model Merging

Implements a sophisticated evolutionary algorithm for merging language models:
- Generates 8 seed candidates from 3 base models using 2³ merge combinations
- Evaluates candidates across multiple domains (code, math, multilingual, structured data)
- Evolutionary selection with mutations and failure recovery
- Full W&B tracking and resume support
- Production error handling and configuration management

Usage:
    forge evo --gens 50 --base-models deepseek,nemotron,qwen2
    python evomerge_pipeline.py --config evolution_config.json
"""

import asyncio
import hashlib
import json
import logging
import random
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4
import click

import numpy as np
import torch
import wandb
from pydantic import BaseModel, Field, validator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evomerge_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Models
# ============================================================================

class BaseModelConfig(BaseModel):
    """Configuration for base models"""
    name: str
    path: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    domain_specialty: Optional[str] = None

class MergeOperatorConfig(BaseModel):
    """Configuration for merge operators"""
    linear_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    slerp_t: float = Field(default=0.5, ge=0.0, le=1.0)
    ties_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    dare_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    dare_amplification: float = Field(default=2.0, ge=1.0, le=5.0)
    frankenmerge_layers: List[int] = Field(default_factory=lambda: [0, 1, 2])
    dfs_merge_ratio: float = Field(default=0.3, ge=0.0, le=1.0)

class EvolutionConfig(BaseModel):
    """Main evolution configuration"""

    # Base models
    base_models: List[BaseModelConfig] = Field(
        default_factory=lambda: [
            BaseModelConfig(name="deepseek", path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", domain_specialty="reasoning"),
            BaseModelConfig(name="nemotron", path="nvidia/Nemotron-Research-Reasoning-Qwen-1.5B", domain_specialty="reasoning"),
            BaseModelConfig(name="qwen2", path="Qwen/Qwen2-1.5B-Instruct", domain_specialty="general")
        ]
    )

    # Evolution parameters
    max_generations: int = Field(default=50, ge=1, le=200)
    population_size: int = Field(default=8, ge=4, le=32)
    mutation_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    selection_pressure: float = Field(default=0.7, ge=0.1, le=1.0)
    plateau_patience: int = Field(default=5, ge=1, le=20)
    plateau_threshold: float = Field(default=0.01, ge=0.001, le=0.1)

    # Merge operators
    merge_operators: MergeOperatorConfig = Field(default_factory=MergeOperatorConfig)

    # Evaluation weights
    evaluation_weights: Dict[str, float] = Field(default_factory=lambda: {
        "code": 0.25,
        "math": 0.25,
        "multilingual": 0.25,
        "structured_data": 0.25
    })

    # System configuration
    device: str = Field(default="auto")
    max_memory_gb: float = Field(default=8.0, gt=0.0)
    output_dir: Path = Field(default=Path("./evomerge_output"))
    checkpoint_dir: Path = Field(default=Path("./evomerge_checkpoints"))
    models_cache_dir: Path = Field(default=Path("./model_cache"))

    # W&B configuration
    wandb_project: str = Field(default="agent-forge")
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = Field(default_factory=lambda: ["evomerge", "evolution"])

    # Resume configuration
    resume_from_checkpoint: Optional[str] = None
    save_intermediate_models: bool = Field(default=True)
    cleanup_failed_models: bool = Field(default=True)

    @validator('base_models')
    def validate_base_models(cls, v):
        if len(v) != 3:
            raise ValueError("Exactly 3 base models required")
        return v

    @validator('evaluation_weights')
    def validate_evaluation_weights(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Evaluation weights must sum to 1.0, got {total}")
        return v

    def __post_init__(self):
        """Create directories"""
        for dir_path in [self.output_dir, self.checkpoint_dir, self.models_cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Model Candidates and Evaluation
# ============================================================================

class ModelCandidate(BaseModel):
    """Represents a model candidate in the evolution"""

    model_config = {"protected_namespaces": ()}

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    generation: int
    parent_ids: List[str] = Field(default_factory=list)
    merge_recipe: Dict[str, Any]
    model_path: Optional[str] = None
    fitness_scores: Dict[str, float] = Field(default_factory=dict)
    overall_fitness: float = 0.0
    evaluation_time: Optional[float] = None
    creation_time: datetime = Field(default_factory=datetime.now)
    is_seed: bool = False
    mutation_applied: Optional[str] = None

    @property
    def short_id(self) -> str:
        return self.id[:6]

    def calculate_overall_fitness(self, weights: Dict[str, float]) -> float:
        """Calculate weighted overall fitness"""
        if not self.fitness_scores:
            return 0.0

        total_fitness = 0.0
        total_weight = 0.0

        for domain, weight in weights.items():
            if domain in self.fitness_scores:
                total_fitness += self.fitness_scores[domain] * weight
                total_weight += weight

        self.overall_fitness = total_fitness / total_weight if total_weight > 0 else 0.0
        return self.overall_fitness

class EvolutionState(BaseModel):
    """Tracks the current state of evolution"""

    current_generation: int = 0
    population: List[ModelCandidate] = Field(default_factory=list)
    best_candidate: Optional[ModelCandidate] = None
    fitness_history: List[Dict[str, float]] = Field(default_factory=list)
    plateau_count: int = 0
    start_time: datetime = Field(default_factory=datetime.now)
    wandb_run_id: Optional[str] = None

    def update_best_candidate(self):
        """Update the best candidate from current population"""
        if self.population:
            self.best_candidate = max(self.population, key=lambda x: x.overall_fitness)

    def check_plateau(self, threshold: float = 0.01) -> bool:
        """Check if evolution has plateaued"""
        if len(self.fitness_history) < 3:
            return False

        recent_scores = [h["best_fitness"] for h in self.fitness_history[-3:]]
        improvement = max(recent_scores) - min(recent_scores)
        return improvement < threshold

# ============================================================================
# Merge Operators
# ============================================================================

class MergeOperators:
    """Implementation of model merging operators"""

    @staticmethod
    def linear_interpolation(models: List[torch.nn.Module], weights: List[float]) -> torch.nn.Module:
        """Linear interpolation between models"""
        if len(models) != len(weights):
            raise ValueError("Number of models must match number of weights")

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        merged_model = models[0].__class__(models[0].config)
        merged_state_dict = {}

        # Get reference state dict
        ref_state_dict = models[0].state_dict()

        for key in ref_state_dict.keys():
            merged_param = torch.zeros_like(ref_state_dict[key])

            for model, weight in zip(models, weights):
                model_state = model.state_dict()
                if key in model_state:
                    merged_param += weight * model_state[key]

            merged_state_dict[key] = merged_param

        merged_model.load_state_dict(merged_state_dict)
        return merged_model

    @staticmethod
    def slerp_interpolation(model1: torch.nn.Module, model2: torch.nn.Module, t: float = 0.5) -> torch.nn.Module:
        """Spherical linear interpolation between two models"""
        merged_model = model1.__class__(model1.config)
        merged_state_dict = {}

        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()

        for key in state_dict1.keys():
            if key in state_dict2:
                param1 = state_dict1[key].flatten()
                param2 = state_dict2[key].flatten()

                # Normalize
                param1_norm = torch.nn.functional.normalize(param1, dim=0)
                param2_norm = torch.nn.functional.normalize(param2, dim=0)

                # Compute angle
                dot_product = torch.clamp(torch.dot(param1_norm, param2_norm), -1.0, 1.0)
                omega = torch.acos(torch.abs(dot_product))

                # SLERP interpolation
                if omega.item() < 1e-6:  # Vectors are nearly identical
                    interpolated = (1 - t) * param1 + t * param2
                else:
                    sin_omega = torch.sin(omega)
                    interpolated = (torch.sin((1 - t) * omega) * param1 + torch.sin(t * omega) * param2) / sin_omega

                merged_state_dict[key] = interpolated.reshape(state_dict1[key].shape)
            else:
                merged_state_dict[key] = state_dict1[key]

        merged_model.load_state_dict(merged_state_dict)
        return merged_model

    @staticmethod
    def ties_merge(models: List[torch.nn.Module], threshold: float = 0.1) -> torch.nn.Module:
        """TIES merging algorithm"""
        merged_model = models[0].__class__(models[0].config)
        merged_state_dict = {}

        ref_state_dict = models[0].state_dict()

        for key in ref_state_dict.keys():
            # Collect all parameter tensors for this key
            params = []
            for model in models:
                model_state = model.state_dict()
                if key in model_state:
                    params.append(model_state[key])

            if len(params) == 0:
                merged_state_dict[key] = ref_state_dict[key]
                continue

            # Stack parameters
            stacked_params = torch.stack(params)

            # Calculate magnitude threshold
            magnitude_threshold = threshold * torch.std(stacked_params)

            # Create mask for significant parameters
            mask = torch.abs(stacked_params) > magnitude_threshold

            # Average only significant parameters
            masked_params = stacked_params * mask.float()
            counts = torch.sum(mask.float(), dim=0)
            counts = torch.clamp(counts, min=1.0)  # Avoid division by zero

            merged_param = torch.sum(masked_params, dim=0) / counts
            merged_state_dict[key] = merged_param

        merged_model.load_state_dict(merged_state_dict)
        return merged_model

    @staticmethod
    def dare_merge(models: List[torch.nn.Module], threshold: float = 0.1, amplification: float = 2.0) -> torch.nn.Module:
        """DARE merging algorithm"""
        merged_model = models[0].__class__(models[0].config)
        merged_state_dict = {}

        ref_state_dict = models[0].state_dict()

        for key in ref_state_dict.keys():
            params = []
            for model in models:
                model_state = model.state_dict()
                if key in model_state:
                    params.append(model_state[key])

            if len(params) == 0:
                merged_state_dict[key] = ref_state_dict[key]
                continue

            stacked_params = torch.stack(params)

            # Random dropping with threshold
            drop_mask = torch.rand_like(stacked_params) > threshold
            amplified_params = stacked_params * drop_mask.float() * amplification

            # Average non-dropped parameters
            counts = torch.sum(drop_mask.float(), dim=0)
            counts = torch.clamp(counts, min=1.0)

            merged_param = torch.sum(amplified_params, dim=0) / counts
            merged_state_dict[key] = merged_param

        merged_model.load_state_dict(merged_state_dict)
        return merged_model

    @staticmethod
    def frankenmerge(models: List[torch.nn.Module], layer_assignment: List[int]) -> torch.nn.Module:
        """Frankenmerge - layer-wise model combination"""
        if len(models) != 3:
            raise ValueError("Frankenmerge requires exactly 3 models")

        merged_model = models[0].__class__(models[0].config)
        merged_state_dict = {}

        # Get layer information
        num_layers = models[0].config.num_hidden_layers

        for key in models[0].state_dict().keys():
            # Determine which model to use for this layer
            if "layers." in key:
                # Extract layer number
                layer_parts = key.split("layers.")[1].split(".")[0]
                try:
                    layer_idx = int(layer_parts)
                    model_idx = layer_assignment[layer_idx % len(layer_assignment)]
                    source_model = models[model_idx]
                except (ValueError, IndexError):
                    source_model = models[0]  # Fallback
            else:
                # Non-layer parameters (embeddings, etc.) - use first model
                source_model = models[0]

            merged_state_dict[key] = source_model.state_dict()[key]

        merged_model.load_state_dict(merged_state_dict)
        return merged_model

    @staticmethod
    def dfs_merge(models: List[torch.nn.Module], merge_ratio: float = 0.3) -> torch.nn.Module:
        """Depth-First Search merge strategy"""
        merged_model = models[0].__class__(models[0].config)
        merged_state_dict = {}

        ref_state_dict = models[0].state_dict()

        for key in ref_state_dict.keys():
            params = []
            for model in models:
                model_state = model.state_dict()
                if key in model_state:
                    params.append(model_state[key])

            if len(params) == 0:
                merged_state_dict[key] = ref_state_dict[key]
                continue

            # Use merge ratio to blend parameters
            base_param = params[0]
            for i, param in enumerate(params[1:], 1):
                weight = merge_ratio / i
                base_param = (1 - weight) * base_param + weight * param

            merged_state_dict[key] = base_param

        merged_model.load_state_dict(merged_state_dict)
        return merged_model

# ============================================================================
# Evaluators
# ============================================================================

class BaseEvaluator:
    """Base class for model evaluators"""

    def __init__(self, device: str = "auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    async def evaluate(self, model_path: str) -> float:
        """Evaluate model and return fitness score [0.0, 1.0]"""
        raise NotImplementedError

class CodeEvaluator(BaseEvaluator):
    """Evaluates code generation capabilities"""

    def __init__(self, device: str = "auto"):
        super().__init__(device)
        self.test_prompts = [
            "Write a Python function to calculate factorial:",
            "Implement binary search in Python:",
            "Create a class for a binary tree:",
            "Write a function to reverse a string:",
            "Implement quicksort algorithm:"
        ]

    async def evaluate(self, model_path: str) -> float:
        """Evaluate code generation"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )

            scores = []

            for prompt in self.test_prompts:
                try:
                    inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )

                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Simple heuristic scoring
                    code_indicators = ["def ", "class ", "import ", "return ", "if ", "for ", "while "]
                    score = sum(1 for indicator in code_indicators if indicator in generated_text.lower())
                    scores.append(min(score / len(code_indicators), 1.0))

                except Exception as e:
                    logger.warning(f"Code evaluation error for prompt '{prompt[:30]}...': {e}")
                    scores.append(0.0)

            return np.mean(scores) if scores else 0.0

        except Exception as e:
            logger.error(f"Code evaluator failed: {e}")
            return 0.0

class MathEvaluator(BaseEvaluator):
    """Evaluates mathematical reasoning capabilities"""

    def __init__(self, device: str = "auto"):
        super().__init__(device)
        self.test_problems = [
            "What is 15 * 23?",
            "Solve for x: 2x + 5 = 13",
            "What is the derivative of x^2 + 3x?",
            "Find the area of a circle with radius 5",
            "If a triangle has sides 3, 4, and 5, what is its area?"
        ]
        self.expected_answers = ["345", "4", "2x + 3", "25π", "6"]

    async def evaluate(self, model_path: str) -> float:
        """Evaluate mathematical reasoning"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )

            scores = []

            for problem, expected in zip(self.test_problems, self.expected_answers):
                try:
                    inputs = tokenizer(f"Problem: {problem}\nSolution:", return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )

                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Check if expected answer appears in response
                    score = 1.0 if expected.lower() in generated_text.lower() else 0.0
                    scores.append(score)

                except Exception as e:
                    logger.warning(f"Math evaluation error for problem '{problem[:30]}...': {e}")
                    scores.append(0.0)

            return np.mean(scores) if scores else 0.0

        except Exception as e:
            logger.error(f"Math evaluator failed: {e}")
            return 0.0

class MultilingualEvaluator(BaseEvaluator):
    """Evaluates multilingual capabilities"""

    def __init__(self, device: str = "auto"):
        super().__init__(device)
        self.test_prompts = [
            ("Translate to French: 'Hello, how are you?'", "french"),
            ("Translate to Spanish: 'Good morning'", "spanish"),
            ("Translate to German: 'Thank you very much'", "german"),
            ("What language is this: 'Bonjour'", "french"),
            ("Say hello in Japanese", "japanese")
        ]

    async def evaluate(self, model_path: str) -> float:
        """Evaluate multilingual capabilities"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )

            scores = []

            for prompt, language in self.test_prompts:
                try:
                    inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )

                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Simple heuristic: check if language name or common words appear
                    language_indicators = {
                        "french": ["français", "bonjour", "merci", "comment"],
                        "spanish": ["español", "hola", "gracias", "buenos"],
                        "german": ["deutsch", "guten", "danke", "sehr"],
                        "japanese": ["こんにちは", "ありがとう", "japanese", "nihongo"]
                    }

                    indicators = language_indicators.get(language, [language])
                    score = 1.0 if any(ind.lower() in generated_text.lower() for ind in indicators) else 0.0
                    scores.append(score)

                except Exception as e:
                    logger.warning(f"Multilingual evaluation error for prompt '{prompt[:30]}...': {e}")
                    scores.append(0.0)

            return np.mean(scores) if scores else 0.0

        except Exception as e:
            logger.error(f"Multilingual evaluator failed: {e}")
            return 0.0

class StructuredDataEvaluator(BaseEvaluator):
    """Evaluates structured data processing capabilities"""

    def __init__(self, device: str = "auto"):
        super().__init__(device)
        self.test_prompts = [
            "Convert this to JSON: Name: John, Age: 30, City: New York",
            "Parse this CSV line: 'apple,red,sweet,fruit'",
            "Create a table with columns: Name, Score, Grade",
            "Format this as YAML: database host localhost port 5432",
            "Extract the numbers from: 'Temperature is 25°C and humidity is 60%'"
        ]

    async def evaluate(self, model_path: str) -> float:
        """Evaluate structured data processing"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )

            scores = []

            for prompt in self.test_prompts:
                try:
                    inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )

                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Check for structured data indicators
                    structure_indicators = ["{", "}", "[", "]", ":", ",", "|", "-", "="]
                    structure_count = sum(1 for ind in structure_indicators if ind in generated_text)
                    score = min(structure_count / 5, 1.0)  # Normalize to [0, 1]
                    scores.append(score)

                except Exception as e:
                    logger.warning(f"Structured data evaluation error for prompt '{prompt[:30]}...': {e}")
                    scores.append(0.0)

            return np.mean(scores) if scores else 0.0

        except Exception as e:
            logger.error(f"Structured data evaluator failed: {e}")
            return 0.0

# ============================================================================
# Evolution Pipeline
# ============================================================================

class EvoMergePipeline:
    """Main evolutionary merging pipeline"""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.state = EvolutionState()
        self.merge_ops = MergeOperators()

        # Initialize evaluators
        self.evaluators = {
            "code": CodeEvaluator(config.device),
            "math": MathEvaluator(config.device),
            "multilingual": MultilingualEvaluator(config.device),
            "structured_data": StructuredDataEvaluator(config.device)
        }

        # Initialize W&B
        self.wandb_run = None

        logger.info(f"EvoMerge pipeline initialized with {len(self.config.base_models)} base models")

    def initialize_wandb(self):
        """Initialize Weights & Biases tracking"""
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                job_type="evomerge",
                tags=self.config.wandb_tags + [f"gen-{self.state.current_generation}"],
                config=self.config.dict(),
                resume="allow"
            )

            self.state.wandb_run_id = self.wandb_run.id
            logger.info(f"W&B initialized: {self.wandb_run.url}")

        except Exception as e:
            logger.error(f"W&B initialization failed: {e}")
            self.wandb_run = None

    def load_base_models(self) -> List[torch.nn.Module]:
        """Load the base models"""
        models = []

        for model_config in self.config.base_models:
            try:
                logger.info(f"Loading base model: {model_config.name}")

                model = AutoModelForCausalLM.from_pretrained(
                    model_config.path,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map=self.config.device
                )

                models.append(model)
                logger.info(f"Successfully loaded {model_config.name}")

            except Exception as e:
                logger.error(f"Failed to load {model_config.name}: {e}")
                raise

        return models

    def generate_seed_candidates(self, base_models: List[torch.nn.Module]) -> List[ModelCandidate]:
        """Generate 8 seed candidates using 2³ combinations"""
        candidates = []

        # Define the 2³ combinations
        merge_combinations = [
            # (continuous, ensemble, structured)
            ("linear", "ties", "frankenmerge"),
            ("linear", "ties", "dfs"),
            ("linear", "dare", "frankenmerge"),
            ("linear", "dare", "dfs"),
            ("slerp", "ties", "frankenmerge"),
            ("slerp", "ties", "dfs"),
            ("slerp", "dare", "frankenmerge"),
            ("slerp", "dare", "dfs"),
        ]

        for i, (continuous, ensemble, structured) in enumerate(merge_combinations):
            try:
                logger.info(f"Generating seed candidate {i+1}/8: {continuous}-{ensemble}-{structured}")

                # Apply merge operations in sequence
                merged_model = self.apply_merge_sequence(
                    base_models, continuous, ensemble, structured
                )

                # Save merged model
                model_path = self.config.output_dir / f"seed_{i+1}_{continuous}_{ensemble}_{structured}"
                model_path.mkdir(parents=True, exist_ok=True)

                merged_model.save_pretrained(model_path)

                # Create candidate
                candidate = ModelCandidate(
                    generation=0,
                    merge_recipe={
                        "continuous": continuous,
                        "ensemble": ensemble,
                        "structured": structured,
                        "base_models": [m.name for m in self.config.base_models]
                    },
                    model_path=str(model_path),
                    is_seed=True
                )

                candidates.append(candidate)
                logger.info(f"Seed candidate {candidate.short_id} created")

            except Exception as e:
                logger.error(f"Failed to create seed candidate {i+1}: {e}")
                continue

        logger.info(f"Generated {len(candidates)} seed candidates")
        return candidates

    def apply_merge_sequence(self, models: List[torch.nn.Module], continuous: str, ensemble: str, structured: str) -> torch.nn.Module:
        """Apply a sequence of merge operations"""

        # Step 1: Continuous interpolation
        if continuous == "linear":
            weights = [1.0/3, 1.0/3, 1.0/3]  # Equal weights
            merged = self.merge_ops.linear_interpolation(models, weights)
        elif continuous == "slerp":
            # SLERP between first two, then linear with third
            temp = self.merge_ops.slerp_interpolation(models[0], models[1], self.config.merge_operators.slerp_t)
            merged = self.merge_ops.linear_interpolation([temp, models[2]], [0.67, 0.33])
        else:
            raise ValueError(f"Unknown continuous method: {continuous}")

        # Step 2: Ensemble crossover
        if ensemble == "ties":
            merged = self.merge_ops.ties_merge([merged] + models, self.config.merge_operators.ties_threshold)
        elif ensemble == "dare":
            merged = self.merge_ops.dare_merge(
                [merged] + models,
                self.config.merge_operators.dare_threshold,
                self.config.merge_operators.dare_amplification
            )
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble}")

        # Step 3: Structured recombination
        if structured == "frankenmerge":
            merged = self.merge_ops.frankenmerge([merged] + models[:2], self.config.merge_operators.frankenmerge_layers)
        elif structured == "dfs":
            merged = self.merge_ops.dfs_merge([merged] + models, self.config.merge_operators.dfs_merge_ratio)
        else:
            raise ValueError(f"Unknown structured method: {structured}")

        return merged

    async def evaluate_candidates(self, candidates: List[ModelCandidate]) -> List[ModelCandidate]:
        """Evaluate all candidates"""
        logger.info(f"Evaluating {len(candidates)} candidates")

        for candidate in tqdm(candidates, desc="Evaluating candidates"):
            if not candidate.model_path or not Path(candidate.model_path).exists():
                logger.warning(f"Model path not found for candidate {candidate.short_id}")
                continue

            try:
                # Evaluate across all domains
                for domain, evaluator in self.evaluators.items():
                    score = await evaluator.evaluate(candidate.model_path)
                    candidate.fitness_scores[domain] = score
                    logger.debug(f"Candidate {candidate.short_id} {domain} score: {score:.3f}")

                # Calculate overall fitness
                candidate.calculate_overall_fitness(self.config.evaluation_weights)
                logger.info(f"Candidate {candidate.short_id} overall fitness: {candidate.overall_fitness:.3f}")

            except Exception as e:
                logger.error(f"Evaluation failed for candidate {candidate.short_id}: {e}")
                candidate.overall_fitness = 0.0

        # Sort by fitness
        candidates.sort(key=lambda x: x.overall_fitness, reverse=True)
        return candidates

    def select_and_mutate(self, candidates: List[ModelCandidate]) -> List[ModelCandidate]:
        """Selection and mutation for next generation"""

        # Select top 2 candidates
        top_candidates = candidates[:2]
        logger.info(f"Selected top candidates: {[c.short_id for c in top_candidates]}")

        # Generate 3 mutations for each top candidate
        mutants = []
        for parent in top_candidates:
            for i in range(3):
                mutant = self.create_mutant(parent, mutation_id=i)
                mutants.append(mutant)

        # Handle failure children (bottom 6)
        failure_candidates = candidates[-6:] if len(candidates) >= 6 else candidates[2:]
        failure_children = self.create_failure_children(failure_candidates)

        # Combine for next generation
        next_generation = mutants + failure_children
        logger.info(f"Next generation: {len(mutants)} mutants + {len(failure_children)} failure children")

        return next_generation

    def create_mutant(self, parent: ModelCandidate, mutation_id: int) -> ModelCandidate:
        """Create a mutant from a parent candidate"""

        # Load parent model
        parent_model = AutoModelForCausalLM.from_pretrained(parent.model_path)

        # Apply random mutation to merge recipe
        mutated_recipe = parent.merge_recipe.copy()

        # Randomly mutate one aspect
        mutation_type = random.choice(["continuous", "ensemble", "structured"])

        if mutation_type == "continuous":
            mutated_recipe["continuous"] = random.choice(["linear", "slerp"])
            mutation_applied = f"continuous->{mutated_recipe['continuous']}"
        elif mutation_type == "ensemble":
            mutated_recipe["ensemble"] = random.choice(["ties", "dare"])
            mutation_applied = f"ensemble->{mutated_recipe['ensemble']}"
        else:  # structured
            mutated_recipe["structured"] = random.choice(["frankenmerge", "dfs"])
            mutation_applied = f"structured->{mutated_recipe['structured']}"

        # Apply noise to model parameters
        mutated_model = self.apply_parameter_noise(parent_model, self.config.mutation_rate)

        # Save mutated model
        model_path = self.config.output_dir / f"gen_{self.state.current_generation+1}_mutant_{parent.short_id}_{mutation_id}"
        model_path.mkdir(parents=True, exist_ok=True)
        mutated_model.save_pretrained(model_path)

        # Create mutant candidate
        mutant = ModelCandidate(
            generation=self.state.current_generation + 1,
            parent_ids=[parent.id],
            merge_recipe=mutated_recipe,
            model_path=str(model_path),
            mutation_applied=mutation_applied
        )

        logger.debug(f"Created mutant {mutant.short_id} from {parent.short_id} with {mutation_applied}")
        return mutant

    def apply_parameter_noise(self, model: torch.nn.Module, noise_scale: float) -> torch.nn.Module:
        """Apply random noise to model parameters"""

        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * noise_scale * torch.std(param)
                    param.add_(noise)

        return model

    def create_failure_children(self, failure_candidates: List[ModelCandidate]) -> List[ModelCandidate]:
        """Create children from failure candidates by merging triples"""

        children = []

        # Group into triples and merge
        for i in range(0, len(failure_candidates), 3):
            triple = failure_candidates[i:i+3]

            if len(triple) >= 2:  # Need at least 2 models to merge
                try:
                    # Load models
                    models = []
                    for candidate in triple:
                        model = AutoModelForCausalLM.from_pretrained(candidate.model_path)
                        models.append(model)

                    # Merge using linear interpolation
                    weights = [1.0/len(models)] * len(models)
                    merged_model = self.merge_ops.linear_interpolation(models, weights)

                    # Save child model
                    child_path = self.config.output_dir / f"gen_{self.state.current_generation+1}_failure_child_{i//3}"
                    child_path.mkdir(parents=True, exist_ok=True)
                    merged_model.save_pretrained(child_path)

                    # Create child candidate
                    child = ModelCandidate(
                        generation=self.state.current_generation + 1,
                        parent_ids=[c.id for c in triple],
                        merge_recipe={
                            "type": "failure_recovery",
                            "method": "linear_interpolation",
                            "parents": [c.short_id for c in triple]
                        },
                        model_path=str(child_path)
                    )

                    children.append(child)
                    logger.debug(f"Created failure child {child.short_id} from {[c.short_id for c in triple]}")

                except Exception as e:
                    logger.error(f"Failed to create failure child from triple {i//3}: {e}")
                    continue

        return children

    def save_checkpoint(self):
        """Save evolution checkpoint"""
        checkpoint_path = self.config.checkpoint_dir / f"evolution_checkpoint_gen_{self.state.current_generation}.json"

        checkpoint_data = {
            "config": self.config.dict(),
            "state": self.state.dict(),
            "timestamp": datetime.now().isoformat()
        }

        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            logger.info(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load evolution checkpoint"""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)

            # Load state
            state_data = checkpoint_data["state"]
            self.state = EvolutionState(**state_data)

            logger.info(f"Checkpoint loaded from generation {self.state.current_generation}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def log_generation_metrics(self):
        """Log metrics for current generation"""
        if not self.state.population:
            return

        # Calculate generation metrics
        fitness_scores = [c.overall_fitness for c in self.state.population]
        generation_metrics = {
            "generation": self.state.current_generation,
            "best_fitness": max(fitness_scores),
            "avg_fitness": np.mean(fitness_scores),
            "std_fitness": np.std(fitness_scores),
            "population_size": len(self.state.population)
        }

        # Domain-specific metrics
        for domain in self.config.evaluation_weights.keys():
            domain_scores = [c.fitness_scores.get(domain, 0.0) for c in self.state.population]
            if domain_scores:
                generation_metrics[f"{domain}_avg"] = np.mean(domain_scores)
                generation_metrics[f"{domain}_best"] = max(domain_scores)

        # Update fitness history
        self.state.fitness_history.append(generation_metrics)

        # Log to W&B
        if self.wandb_run:
            self.wandb_run.log(generation_metrics)

            # Log best model as artifact
            if self.state.best_candidate and self.state.best_candidate.model_path:
                try:
                    artifact = wandb.Artifact(
                        f"model_gen_{self.state.current_generation}",
                        type="model",
                        description=f"Best model from generation {self.state.current_generation}"
                    )
                    artifact.add_dir(self.state.best_candidate.model_path)
                    self.wandb_run.log_artifact(artifact)

                except Exception as e:
                    logger.warning(f"Failed to log model artifact: {e}")

        logger.info(f"Generation {self.state.current_generation} metrics: {generation_metrics}")

    async def run_evolution(self) -> ModelCandidate:
        """Run the complete evolution process"""
        try:
            # Initialize W&B
            self.initialize_wandb()

            # Resume from checkpoint if specified
            if self.config.resume_from_checkpoint:
                if self.load_checkpoint(self.config.resume_from_checkpoint):
                    logger.info("Resumed from checkpoint")
                else:
                    logger.warning("Failed to load checkpoint, starting fresh")

            # Load base models
            base_models = self.load_base_models()

            # Generate seed candidates if starting fresh
            if self.state.current_generation == 0:
                logger.info("Generating seed candidates...")
                seed_candidates = self.generate_seed_candidates(base_models)
                self.state.population = seed_candidates

            # Evolution loop
            for generation in range(self.state.current_generation, self.config.max_generations):
                logger.info(f"=== Generation {generation + 1}/{self.config.max_generations} ===")
                self.state.current_generation = generation

                # Evaluate current population
                self.state.population = await self.evaluate_candidates(self.state.population)

                # Update best candidate
                self.state.update_best_candidate()

                # Log metrics
                self.log_generation_metrics()

                # Check for plateau
                if self.state.check_plateau(self.config.plateau_threshold):
                    self.state.plateau_count += 1
                    logger.info(f"Plateau detected ({self.state.plateau_count}/{self.config.plateau_patience})")

                    if self.state.plateau_count >= self.config.plateau_patience:
                        logger.info("Evolution stopped due to plateau")
                        break
                else:
                    self.state.plateau_count = 0

                # Save checkpoint
                self.save_checkpoint()

                # Generate next generation (unless this is the last)
                if generation < self.config.max_generations - 1:
                    self.state.population = self.select_and_mutate(self.state.population)
                    self.state.current_generation += 1

            # Final evaluation and cleanup
            logger.info("Evolution completed!")
            self.state.update_best_candidate()

            if self.state.best_candidate:
                logger.info(f"Best candidate: {self.state.best_candidate.short_id} "
                          f"(fitness: {self.state.best_candidate.overall_fitness:.3f})")

            return self.state.best_candidate

        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            logger.error(traceback.format_exc())
            raise

        finally:
            if self.wandb_run:
                self.wandb_run.finish()

# ============================================================================
# CLI Interface
# ============================================================================

@click.group()
def forge():
    """Agent Forge CLI"""
    pass

@forge.command()
@click.option('--gens', '--generations', default=50, help='Number of generations')
@click.option('--base-models', default='deepseek,nemotron,qwen2', help='Comma-separated base model names')
@click.option('--config', help='Configuration file path')
@click.option('--resume', help='Resume from checkpoint')
@click.option('--output-dir', default='./evomerge_output', help='Output directory')
@click.option('--device', default='auto', help='Device to use (auto, cuda, cpu)')
def evo(gens, base_models, config, resume, output_dir, device):
    """Run evolutionary model merging"""

    try:
        # Load configuration
        if config and Path(config).exists():
            with open(config, 'r') as f:
                config_data = json.load(f)
            evolution_config = EvolutionConfig(**config_data)
        else:
            # Create default configuration
            base_model_names = base_models.split(',')
            model_mapping = {
                'deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                'nemotron': 'nvidia/Nemotron-Research-Reasoning-Qwen-1.5B',
                'qwen2': 'Qwen/Qwen2-1.5B-Instruct'
            }

            base_model_configs = []
            for name in base_model_names:
                if name in model_mapping:
                    base_model_configs.append(BaseModelConfig(
                        name=name,
                        path=model_mapping[name]
                    ))

            evolution_config = EvolutionConfig(
                max_generations=gens,
                base_models=base_model_configs,
                output_dir=Path(output_dir),
                device=device,
                resume_from_checkpoint=resume
            )

        # Run evolution
        pipeline = EvoMergePipeline(evolution_config)

        logger.info("Starting evolutionary merging pipeline...")
        best_candidate = asyncio.run(pipeline.run_evolution())

        if best_candidate:
            logger.info(f"Evolution completed successfully!")
            logger.info(f"Best model: {best_candidate.model_path}")
            logger.info(f"Fitness: {best_candidate.overall_fitness:.3f}")
        else:
            logger.error("Evolution failed to produce a best candidate")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Evolution pipeline failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

# ============================================================================
# Orchestrator Integration
# ============================================================================

async def run_evomerge(config: Dict[str, Any]) -> 'PhaseResult':
    """
    Orchestrator entry point for EvoMerge phase.

    Args:
        config: Configuration dictionary with evomerge parameters

    Returns:
        PhaseResult with status, artifacts, and metrics
    """
    from agent_forge.forge_orchestrator import PhaseResult, PhaseStatus, PhaseType, PhaseArtifact

    start_time = time.time()

    try:
        logger.info("Starting EvoMerge phase via orchestrator")

        # Convert config to EvolutionConfig
        evo_config = EvolutionConfig(**config)

        # Create and run pipeline
        pipeline = EvoMergePipeline(evo_config)
        best_candidate = await pipeline.run_evolution()

        duration = time.time() - start_time

        if best_candidate and best_candidate.model_path:
            # Success - create artifacts
            artifacts = [
                PhaseArtifact(
                    phase_type=PhaseType.EVOMERGE,
                    artifact_type="best_model",
                    data={
                        "model_path": best_candidate.model_path,
                        "model_id": best_candidate.id,
                        "fitness_score": best_candidate.overall_fitness,
                        "generation": best_candidate.generation,
                        "merge_recipe": best_candidate.merge_recipe
                    },
                    metadata={
                        "evaluation_time": best_candidate.evaluation_time,
                        "creation_time": best_candidate.creation_time.isoformat()
                    }
                )
            ]

            # Create metrics summary
            metrics = {
                "best_fitness": best_candidate.overall_fitness,
                "final_generation": pipeline.state.current_generation,
                "total_candidates": len(pipeline.state.population),
                "execution_time": duration,
                "fitness_breakdown": best_candidate.fitness_scores,
                "plateau_count": pipeline.state.plateau_count
            }

            logger.info(f"EvoMerge completed successfully in {duration:.1f}s")

            return PhaseResult(
                phase_type=PhaseType.EVOMERGE,
                status=PhaseStatus.COMPLETED,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.now(),
                duration_seconds=duration,
                artifacts_produced=artifacts,
                metrics=metrics
            )
        else:
            # Failed to produce a valid candidate
            return PhaseResult(
                phase_type=PhaseType.EVOMERGE,
                status=PhaseStatus.FAILED,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.now(),
                duration_seconds=duration,
                error_message="Failed to produce a valid best candidate",
                metrics={"execution_time": duration}
            )

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"EvoMerge phase failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        return PhaseResult(
            phase_type=PhaseType.EVOMERGE,
            status=PhaseStatus.FAILED,
            start_time=datetime.fromtimestamp(start_time),
            end_time=datetime.now(),
            duration_seconds=duration,
            error_message=error_msg,
            metrics={"execution_time": duration}
        )

# Make the entry point discoverable
run = run_evomerge  # Alias for orchestrator discovery
execute = run_evomerge  # Alternative alias

if __name__ == "__main__":
    forge()
