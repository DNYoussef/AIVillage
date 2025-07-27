#!/usr/bin/env python3
"""
Agent Forge Compression Pipeline

Multi-stage compression pipeline that applies BitNet quantization (Phase 1) after Quiet-STaR baking.
This creates production-ready models with:
- Evolutionary optimization (EvoMerge)
- Reasoning enhancement (Quiet-STaR)
- Efficient deployment (BitNet compression)

Pipeline: EvoMerge → Quiet-STaR → BitNet → Deployment
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import click
import numpy as np
import torch
import torch.nn as nn
import wandb
from pydantic import BaseModel, Field, validator
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Import compression modules
from .compression.stage1_bitnet import (
    convert_to_bitnet,
    apply_hf_bitnet_finetune,
    GradualBitnetCallback,
    BitNetLinear
)
from .compression.vptq import VPTQQuantizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class CompressionConfig(BaseModel):
    """Configuration for the compression pipeline"""

    # Model paths
    input_model_path: str = Field(..., description="Path to Quiet-STaR baked model")
    output_model_path: str = Field(..., description="Path for compressed model output")

    # BitNet configuration
    bitnet_zero_threshold: float = Field(default=0.02, ge=0.0, le=0.1)
    bitnet_batch_size: int = Field(default=4, ge=1, le=32)
    bitnet_learning_rate: float = Field(default=1e-5, ge=1e-7, le=1e-3)
    bitnet_finetuning_epochs: int = Field(default=2, ge=1, le=10)
    bitnet_warmup_ratio: float = Field(default=0.4, ge=0.0, le=1.0)

    # Calibration dataset
    calibration_dataset: str = Field(default="wikitext", description="Dataset for calibration")
    calibration_samples: int = Field(default=1000, ge=100, le=10000)

    # Evaluation configuration
    eval_before_after: bool = Field(default=True, description="Evaluate before/after compression")
    eval_samples: int = Field(default=100, ge=10, le=500)
    eval_datasets: List[str] = Field(default_factory=lambda: ["gsm8k"])

    # System configuration
    device: str = Field(default="auto")
    mixed_precision: bool = Field(default=True)

    # W&B configuration
    wandb_project: str = Field(default="agent-forge")
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = Field(default_factory=lambda: ["compression", "bitnet"])

    @validator('device')
    def validate_device(cls, v):
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v

# ============================================================================
# Model Analysis
# ============================================================================

class ModelAnalyzer:
    """Analyzes model characteristics for optimal compression"""

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def analyze_model_structure(self) -> Dict[str, Any]:
        """Analyze model structure and parameters"""
        analysis = {
            "total_parameters": 0,
            "linear_parameters": 0,
            "attention_parameters": 0,
            "layer_count": 0,
            "hidden_size": 0,
            "vocab_size": 0,
            "compression_potential": 0.0
        }

        # Count parameters by type
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            analysis["total_parameters"] += param_count

            if "linear" in name.lower() or "dense" in name.lower():
                analysis["linear_parameters"] += param_count

            if "attention" in name.lower() or "attn" in name.lower():
                analysis["attention_parameters"] += param_count

        # Extract model config if available
        if hasattr(self.model, 'config'):
            config = self.model.config
            analysis["layer_count"] = getattr(config, 'num_hidden_layers', 0)
            analysis["hidden_size"] = getattr(config, 'hidden_size', 0)
            analysis["vocab_size"] = getattr(config, 'vocab_size', 0)

        # Estimate compression potential (linear layers compress best)
        if analysis["total_parameters"] > 0:
            linear_ratio = analysis["linear_parameters"] / analysis["total_parameters"]
            analysis["compression_potential"] = linear_ratio * 0.8  # ~80% reduction for linear layers

        logger.info(f"Model Analysis:")
        logger.info(f"  Total Parameters: {analysis['total_parameters']:,}")
        logger.info(f"  Linear Parameters: {analysis['linear_parameters']:,} ({analysis['linear_parameters']/analysis['total_parameters']*100:.1f}%)")
        logger.info(f"  Compression Potential: {analysis['compression_potential']*100:.1f}%")

        return analysis

    def estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage before/after compression"""
        total_params = sum(p.numel() for p in self.model.parameters())

        # Full precision (FP16/FP32)
        fp16_size_mb = (total_params * 2) / (1024**2)  # 2 bytes per param
        fp32_size_mb = (total_params * 4) / (1024**2)  # 4 bytes per param

        # BitNet estimation (ternary weights)
        # Most weights become ternary (2 bits), some remain FP16
        linear_params = 0
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                linear_params += module.weight.numel()
                if module.bias is not None:
                    linear_params += module.bias.numel()

        other_params = total_params - linear_params

        # Ternary weights: ~0.25 bytes per param + scaling factors
        bitnet_size_mb = (
            (linear_params * 0.25) +  # Ternary weights
            (linear_params * 2 / 8) +  # Scaling factors (FP16)
            (other_params * 2)  # Other params remain FP16
        ) / (1024**2)

        compression_ratio = fp16_size_mb / bitnet_size_mb if bitnet_size_mb > 0 else 1.0

        memory_usage = {
            "fp32_mb": fp32_size_mb,
            "fp16_mb": fp16_size_mb,
            "bitnet_mb": bitnet_size_mb,
            "compression_ratio": compression_ratio,
            "memory_savings_mb": fp16_size_mb - bitnet_size_mb
        }

        logger.info(f"Memory Estimation:")
        logger.info(f"  FP16: {fp16_size_mb:.1f} MB")
        logger.info(f"  BitNet: {bitnet_size_mb:.1f} MB")
        logger.info(f"  Compression Ratio: {compression_ratio:.1f}x")
        logger.info(f"  Memory Savings: {memory_usage['memory_savings_mb']:.1f} MB")

        return memory_usage

# ============================================================================
# Evaluation Suite
# ============================================================================

class CompressionEvaluator:
    """Evaluates model performance before/after compression"""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.device = torch.device(config.device)

    async def evaluate_model(self, model: nn.Module, tokenizer,
                           model_name: str = "model") -> Dict[str, float]:
        """Evaluate model on configured datasets"""
        model.eval()
        results = {}

        for dataset_name in self.config.eval_datasets:
            logger.info(f"Evaluating {model_name} on {dataset_name}")

            if dataset_name == "gsm8k":
                score = await self.evaluate_gsm8k(model, tokenizer)
            elif dataset_name == "math":
                score = await self.evaluate_math(model, tokenizer)
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                score = 0.0

            results[f"{dataset_name}_accuracy"] = score
            logger.info(f"{model_name} {dataset_name} accuracy: {score:.3f}")

        return results

    async def evaluate_gsm8k(self, model: nn.Module, tokenizer) -> float:
        """Evaluate on GSM8K dataset"""
        try:
            # Load dataset
            dataset = load_dataset("gsm8k", "main", split="test")
            samples = list(dataset)[:self.config.eval_samples]

            correct = 0
            total = len(samples)

            with torch.no_grad():
                for item in tqdm(samples, desc="GSM8K Evaluation"):
                    question = item["question"]
                    answer = item["answer"]

                    # Extract numerical answer
                    answer_parts = answer.split("####")
                    if len(answer_parts) >= 2:
                        target_answer = answer_parts[1].strip()
                    else:
                        continue

                    # Generate response
                    prompt = f"Question: {question}\nAnswer:"
                    inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

                    generated = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                    response = tokenizer.decode(generated[0][inputs.input_ids.shape[1]:],
                                             skip_special_tokens=True)

                    # Check if answer is correct
                    if target_answer in response:
                        correct += 1

            return correct / total if total > 0 else 0.0

        except Exception as e:
            logger.error(f"GSM8K evaluation failed: {e}")
            return 0.0

    async def evaluate_math(self, model: nn.Module, tokenizer) -> float:
        """Evaluate on MATH dataset"""
        try:
            dataset = load_dataset("hendrycks/math", split="test")
            samples = list(dataset)[:self.config.eval_samples]

            correct = 0
            total = len(samples)

            with torch.no_grad():
                for item in tqdm(samples, desc="MATH Evaluation"):
                    problem = item["problem"]
                    solution = item["solution"]

                    # Generate response
                    prompt = f"Problem: {problem}\nSolution:"
                    inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

                    generated = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                    response = tokenizer.decode(generated[0][inputs.input_ids.shape[1]:],
                                             skip_special_tokens=True)

                    # Simple correctness check (could be improved)
                    if self.extract_final_answer(solution) in response:
                        correct += 1

            return correct / total if total > 0 else 0.0

        except Exception as e:
            logger.error(f"MATH evaluation failed: {e}")
            return 0.0

    def extract_final_answer(self, solution: str) -> str:
        """Extract final answer from MATH solution"""
        import re
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, solution)

        if match:
            return match.group(1)

        # Fallback
        numbers = re.findall(r'-?\d+\.?\d*', solution)
        return numbers[-1] if numbers else ""

# ============================================================================
# Calibration Dataset
# ============================================================================

class CalibrationDataset:
    """Creates calibration dataset for compression"""

    def __init__(self, dataset_name: str, num_samples: int, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if dataset_name == "wikitext":
            self.examples = self.load_wikitext(num_samples)
        elif dataset_name == "openwebtext":
            self.examples = self.load_openwebtext(num_samples)
        else:
            raise ValueError(f"Unknown calibration dataset: {dataset_name}")

        logger.info(f"Loaded {len(self.examples)} calibration examples")

    def load_wikitext(self, num_samples: int) -> List[str]:
        """Load WikiText-2 for calibration"""
        try:
            dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")

            examples = []
            for item in dataset:
                text = item["text"].strip()
                if len(text) > 50:  # Filter out short texts
                    examples.append(text)

                if len(examples) >= num_samples:
                    break

            return examples

        except Exception as e:
            logger.error(f"Failed to load WikiText: {e}")
            # Fallback to synthetic data
            return [
                "The quick brown fox jumps over the lazy dog. This is a test sentence for model calibration.",
                "Machine learning models require careful calibration to achieve optimal performance in production environments.",
                "Natural language processing has revolutionized how we interact with artificial intelligence systems."
            ] * (num_samples // 3 + 1)

    def load_openwebtext(self, num_samples: int) -> List[str]:
        """Load OpenWebText for calibration"""
        try:
            dataset = load_dataset("openwebtext", split="train")

            examples = []
            for item in dataset:
                text = item["text"].strip()
                if len(text) > 100:
                    examples.append(text[:1000])  # Truncate long texts

                if len(examples) >= num_samples:
                    break

            return examples

        except Exception as e:
            logger.error(f"Failed to load OpenWebText: {e}")
            return self.load_wikitext(num_samples)

    def create_torch_dataset(self):
        """Create PyTorch dataset for training"""
        encoded_examples = []

        for text in self.examples:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            # For language modeling, labels = input_ids
            encoding["labels"] = encoding["input_ids"].clone()
            encoded_examples.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": encoding["labels"].squeeze()
            })

        return encoded_examples

# ============================================================================
# Main Compression Pipeline
# ============================================================================

class CompressionPipeline:
    """Main compression pipeline orchestrator"""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.wandb_run = None

        # Set device
        self.device = torch.device(config.device)

        logger.info(f"Compression pipeline initialized for {config.input_model_path}")

    def initialize_wandb(self):
        """Initialize W&B tracking"""
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                job_type="compression",
                tags=self.config.wandb_tags,
                config=self.config.dict()
            )

            logger.info(f"W&B initialized: {self.wandb_run.url}")

        except Exception as e:
            logger.error(f"W&B initialization failed: {e}")
            self.wandb_run = None

    async def run_compression_pipeline(self) -> Dict[str, Any]:
        """Run complete compression pipeline"""
        try:
            # Initialize W&B
            self.initialize_wandb()

            # Load model
            logger.info(f"Loading model from {self.config.input_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.input_model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.config.input_model_path,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
            ).to(self.device)

            # Analyze model structure
            analyzer = ModelAnalyzer(model, tokenizer)
            model_analysis = analyzer.analyze_model_structure()
            memory_analysis = analyzer.estimate_memory_usage()

            # Log analysis to W&B
            if self.wandb_run:
                self.wandb_run.log({
                    "model_parameters": model_analysis["total_parameters"],
                    "linear_parameters": model_analysis["linear_parameters"],
                    "compression_potential": model_analysis["compression_potential"],
                    "original_size_mb": memory_analysis["fp16_mb"],
                    "estimated_compressed_mb": memory_analysis["bitnet_mb"],
                    "estimated_compression_ratio": memory_analysis["compression_ratio"]
                })

            # Evaluate before compression (if requested)
            pre_compression_results = {}
            if self.config.eval_before_after:
                evaluator = CompressionEvaluator(self.config)
                pre_compression_results = await evaluator.evaluate_model(
                    model, tokenizer, "pre_compression"
                )

                if self.wandb_run:
                    for metric, value in pre_compression_results.items():
                        self.wandb_run.log({f"pre_{metric}": value})

            # Apply BitNet compression
            logger.info("Applying BitNet compression...")
            compressed_model = self.apply_bitnet_compression(model, tokenizer)

            # Evaluate after compression
            post_compression_results = {}
            if self.config.eval_before_after:
                post_compression_results = await evaluator.evaluate_model(
                    compressed_model, tokenizer, "post_compression"
                )

                if self.wandb_run:
                    for metric, value in post_compression_results.items():
                        self.wandb_run.log({f"post_{metric}": value})

            # Calculate actual compression metrics
            actual_memory = self.calculate_actual_compression(model, compressed_model)

            # Save compressed model
            logger.info(f"Saving compressed model to {self.config.output_model_path}")
            Path(self.config.output_model_path).mkdir(parents=True, exist_ok=True)

            compressed_model.save_pretrained(self.config.output_model_path)
            tokenizer.save_pretrained(self.config.output_model_path)

            # Save compression metadata
            compression_metadata = {
                "input_model": self.config.input_model_path,
                "output_model": self.config.output_model_path,
                "compression_method": "BitNet",
                "model_analysis": model_analysis,
                "memory_analysis": memory_analysis,
                "actual_compression": actual_memory,
                "pre_compression_eval": pre_compression_results,
                "post_compression_eval": post_compression_results,
                "timestamp": datetime.now().isoformat()
            }

            metadata_path = Path(self.config.output_model_path) / "compression_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(compression_metadata, f, indent=2, default=str)

            # Log final metrics to W&B
            if self.wandb_run:
                # Calculate performance retention
                if pre_compression_results and post_compression_results:
                    for dataset in self.config.eval_datasets:
                        metric_name = f"{dataset}_accuracy"
                        if metric_name in pre_compression_results and metric_name in post_compression_results:
                            pre_score = pre_compression_results[metric_name]
                            post_score = post_compression_results[metric_name]
                            retention = (post_score / pre_score * 100) if pre_score > 0 else 0
                            self.wandb_run.log({f"{dataset}_performance_retention": retention})

                self.wandb_run.log(actual_memory)

                # Save compressed model as artifact
                artifact = wandb.Artifact(
                    "compressed_model",
                    type="model",
                    description=f"BitNet compressed model with {actual_memory.get('compression_ratio', 1.0):.1f}x compression"
                )
                artifact.add_dir(self.config.output_model_path)
                self.wandb_run.log_artifact(artifact)

            # Create summary
            results = {
                "success": True,
                "compression_ratio": actual_memory.get("compression_ratio", 1.0),
                "memory_savings_mb": actual_memory.get("memory_savings_mb", 0),
                "model_path": self.config.output_model_path,
                "metadata": compression_metadata
            }

            logger.info("Compression pipeline completed successfully!")
            return results

        except Exception as e:
            logger.error(f"Compression pipeline failed: {e}")
            raise

        finally:
            if self.wandb_run:
                self.wandb_run.finish()

    def apply_bitnet_compression(self, model: nn.Module, tokenizer) -> nn.Module:
        """Apply BitNet compression to model"""
        logger.info("Converting model to BitNet...")

        # Convert model to BitNet
        compressed_model = convert_to_bitnet(model, threshold=self.config.bitnet_zero_threshold)

        # Prepare calibration dataset
        calibration_dataset = CalibrationDataset(
            self.config.calibration_dataset,
            self.config.calibration_samples,
            tokenizer
        )

        # Fine-tune with BitNet
        torch_dataset = calibration_dataset.create_torch_dataset()

        # Create mock config for fine-tuning
        class MockConfig:
            def __init__(self, config: CompressionConfig):
                self.bitnet_zero_threshold = config.bitnet_zero_threshold
                self.bitnet_batch_size = config.bitnet_batch_size
                self.bitnet_learning_rate = config.bitnet_learning_rate
                self.bitnet_finetuning_epochs = config.bitnet_finetuning_epochs

        mock_config = MockConfig(self.config)

        # Apply fine-tuning
        compressed_model = apply_hf_bitnet_finetune(compressed_model, torch_dataset, mock_config)

        logger.info("BitNet compression completed")
        return compressed_model

    def calculate_actual_compression(self, original_model: nn.Module,
                                   compressed_model: nn.Module) -> Dict[str, float]:
        """Calculate actual compression metrics"""

        # Count parameters
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())

        # Estimate memory usage
        original_memory_mb = (original_params * 2) / (1024**2)  # FP16

        # Count BitNet layers for accurate estimation
        bitnet_params = 0
        regular_params = 0

        for module in compressed_model.modules():
            if isinstance(module, BitNetLinear):
                bitnet_params += module.weight_fp.numel()
                if module.bias is not None:
                    bitnet_params += module.bias.numel()
            else:
                for param in module.parameters(recurse=False):
                    regular_params += param.numel()

        # BitNet memory: ternary weights + scaling factors + regular params
        compressed_memory_mb = (
            (bitnet_params * 0.25) +  # Ternary weights (~2 bits each)
            (bitnet_params * 2 / 8) +  # Scaling factors
            (regular_params * 2)  # Regular params in FP16
        ) / (1024**2)

        compression_ratio = original_memory_mb / compressed_memory_mb if compressed_memory_mb > 0 else 1.0
        memory_savings = original_memory_mb - compressed_memory_mb

        metrics = {
            "original_params": original_params,
            "compressed_params": compressed_params,
            "bitnet_params": bitnet_params,
            "regular_params": regular_params,
            "original_memory_mb": original_memory_mb,
            "compressed_memory_mb": compressed_memory_mb,
            "compression_ratio": compression_ratio,
            "memory_savings_mb": memory_savings,
            "compression_efficiency": (memory_savings / original_memory_mb * 100) if original_memory_mb > 0 else 0
        }

        logger.info(f"Actual Compression Metrics:")
        logger.info(f"  Original: {original_memory_mb:.1f} MB")
        logger.info(f"  Compressed: {compressed_memory_mb:.1f} MB")
        logger.info(f"  Ratio: {compression_ratio:.1f}x")
        logger.info(f"  Savings: {memory_savings:.1f} MB ({metrics['compression_efficiency']:.1f}%)")

        return metrics

# ============================================================================
# CLI Interface
# ============================================================================

@click.group()
def forge():
    """Agent Forge CLI"""
    pass

@forge.command()
@click.option('--input-model', required=True, help='Path to Quiet-STaR baked model')
@click.option('--output-model', required=True, help='Path for compressed model output')
@click.option('--calibration-dataset', default='wikitext', help='Calibration dataset (wikitext, openwebtext)')
@click.option('--calibration-samples', default=1000, help='Number of calibration samples')
@click.option('--eval-samples', default=100, help='Number of evaluation samples')
@click.option('--device', default='auto', help='Device to use (auto, cuda, cpu)')
@click.option('--config', help='Configuration JSON file')
def compress(input_model, output_model, calibration_dataset, calibration_samples,
             eval_samples, device, config):
    """Apply BitNet compression to Quiet-STaR baked model"""

    try:
        # Load configuration
        if config and Path(config).exists():
            with open(config, 'r') as f:
                config_data = json.load(f)
            compression_config = CompressionConfig(**config_data)
        else:
            # Create configuration from CLI args
            compression_config = CompressionConfig(
                input_model_path=input_model,
                output_model_path=output_model,
                calibration_dataset=calibration_dataset,
                calibration_samples=calibration_samples,
                eval_samples=eval_samples,
                device=device
            )

        # Run compression pipeline
        pipeline = CompressionPipeline(compression_config)

        logger.info("Starting BitNet compression pipeline...")
        results = asyncio.run(pipeline.run_compression_pipeline())

        # Print results
        print("\n" + "="*60)
        print("COMPRESSION PIPELINE RESULTS")
        print("="*60)
        print(f"Success: {results['success']}")
        print(f"Compression Ratio: {results['compression_ratio']:.1f}x")
        print(f"Memory Savings: {results['memory_savings_mb']:.1f} MB")
        print(f"Compressed Model: {results['model_path']}")
        print("="*60)

    except Exception as e:
        logger.error(f"Compression pipeline failed: {e}")
        raise click.ClickException(str(e))

# ============================================================================
# Orchestrator Integration
# ============================================================================

async def run_compression(config: Dict[str, Any]) -> 'PhaseResult':
    """
    Orchestrator entry point for Compression phase.

    Args:
        config: Configuration dictionary with compression parameters

    Returns:
        PhaseResult with status, artifacts, and metrics
    """
    try:
        from agent_forge.forge_orchestrator import PhaseResult, PhaseStatus, PhaseType, PhaseArtifact
    except ImportError:
        # Fallback classes for when orchestrator is not available
        from dataclasses import dataclass
        from enum import Enum

        class PhaseStatus(Enum):
            SUCCESS = "success"
            FAILURE = "failure"

        class PhaseType(Enum):
            COMPRESSION = "compression"

        @dataclass
        class PhaseArtifact:
            name: str
            path: str
            metadata: dict = None

        @dataclass
        class PhaseResult:
            status: PhaseStatus
            phase_type: PhaseType
            artifacts: list = None
            metrics: dict = None
    from datetime import datetime
    import time

    start_time = time.time()

    try:
        logger.info("Starting Compression phase via orchestrator")

        # Convert config to CompressionConfig
        compression_config = CompressionConfig(**config)

        # Create and run pipeline
        pipeline = CompressionPipeline(compression_config)
        results = await pipeline.run_compression_pipeline()

        duration = time.time() - start_time

        if results['success']:
            # Success - create artifacts
            artifacts = [
                PhaseArtifact(
                    phase_type=PhaseType.COMPRESSION,
                    artifact_type="compressed_model",
                    data={
                        "model_path": results['model_path'],
                        "compression_ratio": results['compression_ratio'],
                        "memory_savings_mb": results['memory_savings_mb'],
                        "original_size_mb": results.get('original_size_mb', 0),
                        "compressed_size_mb": results.get('compressed_size_mb', 0)
                    },
                    metadata={
                        "bitnet_config": compression_config.dict(),
                        "compression_method": "BitNet"
                    }
                )
            ]

            # Create metrics summary
            metrics = {
                "compression_ratio": results['compression_ratio'],
                "memory_savings_mb": results['memory_savings_mb'],
                "execution_time": duration,
                "success": True,
                "evaluation_metrics": results.get('evaluation_metrics', {}),
                "calibration_samples": compression_config.calibration_samples
            }

            logger.info(f"Compression completed successfully in {duration:.1f}s")

            return PhaseResult(
                phase_type=PhaseType.COMPRESSION,
                status=PhaseStatus.COMPLETED,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.now(),
                duration_seconds=duration,
                artifacts_produced=artifacts,
                metrics=metrics
            )
        else:
            # Failed compression
            return PhaseResult(
                phase_type=PhaseType.COMPRESSION,
                status=PhaseStatus.FAILED,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.now(),
                duration_seconds=duration,
                error_message=results.get('error', 'Compression failed with unknown error'),
                metrics={"execution_time": duration}
            )

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Compression phase failed: {str(e)}"
        logger.error(error_msg)

        return PhaseResult(
            phase_type=PhaseType.COMPRESSION,
            status=PhaseStatus.FAILED,
            start_time=datetime.fromtimestamp(start_time),
            end_time=datetime.now(),
            duration_seconds=duration,
            error_message=error_msg,
            metrics={"execution_time": duration}
        )

# Make the entry point discoverable
run = run_compression  # Alias for orchestrator discovery
execute = run_compression  # Alternative alias

if __name__ == "__main__":
    forge()
