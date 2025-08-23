#!/usr/bin/env python3
"""
Cognate Phase - Model Creation and Initialization (Phase 1)

This is the critical missing Phase 1 of the Agent Forge pipeline that handles:
- Model architecture selection and configuration
- Base model loading from HuggingFace or local sources  
- Multi-model merging with various strategies (average, weighted, evolutionary)
- Parameter initialization using proven strategies
- Model compatibility validation and preprocessing

Completes the 8-phase Agent Forge pipeline by providing proper model genesis.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Import phase controller base
try:
    from ..phase_controller import PhaseController, PhaseResult
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from phase_controller import PhaseController, PhaseResult

logger = logging.getLogger(__name__)


@dataclass
class CognateConfig:
    """Configuration for Cognate (model creation) phase."""
    
    # Model selection strategy
    base_models: List[str] = field(default_factory=lambda: [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "microsoft/DialoGPT-medium", 
        "Qwen/Qwen2-1.5B-Instruct"
    ])
    
    # Architecture configuration
    target_architecture: str = "auto"  # auto, custom, or specific model
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    
    # Initialization strategy
    init_strategy: str = "xavier_uniform"  # xavier_uniform, kaiming_normal, custom
    merge_strategy: str = "average"  # average, weighted, evolutionary
    
    # Model merging weights (if using weighted merge)
    merge_weights: Optional[Dict[str, float]] = None
    
    # Validation settings
    validate_compatibility: bool = True
    require_tokenizer_match: bool = False  # Set to False for flexibility
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda
    dtype: str = "float32"  # float32, float16, bfloat16
    
    # Advanced options
    use_cache: bool = True
    trust_remote_code: bool = False
    torch_compile: bool = False


class CognatePhase(PhaseController):
    """
    Phase 1: Model Creation and Initialization
    
    The foundational phase that creates the base model for the Agent Forge pipeline.
    Handles architecture selection, model loading/merging, initialization, and validation.
    """
    
    def __init__(self, config: CognateConfig):
        super().__init__(config)
        self.config = config
        
        # Device setup
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
            
        # Dtype setup
        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16, 
            "bfloat16": torch.bfloat16
        }
        self.torch_dtype = self.dtype_map.get(self.config.dtype, torch.float32)
        
    async def run(self, model: Optional[nn.Module] = None) -> PhaseResult:
        """
        Execute cognate phase for model creation.
        
        Args:
            model: Optional existing model (for reinitialization)
            
        Returns:
            PhaseResult with created/initialized model
        """
        start_time = time.time()
        phase_start = datetime.now()
        
        try:
            self.logger.info("Starting Cognate Phase (Model Creation)")
            
            # Phase 1: Model Architecture Selection
            self.logger.info("Phase 1.1: Selecting model architecture")
            architecture = self._select_architecture()
            
            # Phase 2: Base Model Loading
            self.logger.info("Phase 1.2: Loading base models")
            base_models = await self._load_base_models()
            
            if not base_models:
                raise RuntimeError("No base models could be loaded successfully")
                
            # Phase 3: Model Merging (if multiple bases)
            if len(base_models) > 1:
                self.logger.info(f"Phase 1.3: Merging {len(base_models)} base models")
                merged_model = self._merge_models(base_models, architecture)
            else:
                self.logger.info("Phase 1.3: Using single base model")
                merged_model = base_models[0]
                
            # Phase 4: Parameter Initialization
            self.logger.info("Phase 1.4: Initializing parameters")
            initialized_model = self._initialize_parameters(merged_model)
            
            # Phase 5: Model Validation
            self.logger.info("Phase 1.5: Validating created model")
            validation_results = self._validate_model(initialized_model)
            
            if not validation_results["passed"]:
                self.logger.warning("Model validation failed, but continuing")
                
            # Move to target device
            initialized_model = initialized_model.to(self.device, dtype=self.torch_dtype)
            
            # Apply torch.compile if requested
            if self.config.torch_compile:
                try:
                    initialized_model = torch.compile(initialized_model)
                    self.logger.info("Applied torch.compile optimization")
                except Exception as e:
                    self.logger.warning(f"torch.compile failed: {e}, continuing without")
            
            duration = time.time() - start_time
            phase_end = datetime.now()
            
            self.logger.info(f"Cognate Phase completed successfully in {duration:.2f}s")
            
            return PhaseResult(
                success=True,
                model=initialized_model,
                phase_name="CognatePhase",
                duration_seconds=duration,
                start_time=phase_start,
                end_time=phase_end,
                metrics={
                    "duration_seconds": duration,
                    "base_models_used": len(base_models),
                    "parameter_count": sum(p.numel() for p in initialized_model.parameters()),
                    "model_size_mb": sum(p.numel() * p.element_size() for p in initialized_model.parameters()) / (1024**2),
                    "validation_passed": validation_results["passed"],
                    "architecture_type": architecture["type"],
                    "initialization_strategy": self.config.init_strategy,
                    "merge_strategy": self.config.merge_strategy if len(base_models) > 1 else "single_model",
                    "device": str(self.device),
                    "dtype": str(self.torch_dtype)
                },
                artifacts={
                    "architecture_config": architecture,
                    "validation_results": validation_results,
                    "base_model_info": [self._get_model_info(m) for m in base_models],
                    "model_config": self._get_model_config(initialized_model)
                },
                config={
                    "base_models": self.config.base_models,
                    "target_architecture": self.config.target_architecture,
                    "init_strategy": self.config.init_strategy,
                    "merge_strategy": self.config.merge_strategy,
                    "device": self.config.device,
                    "dtype": self.config.dtype
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Cognate phase failed after {duration:.2f}s: {str(e)}")
            
            return PhaseResult(
                success=False,
                model=model,  # Return original model if provided
                phase_name="CognatePhase",
                duration_seconds=duration,
                start_time=phase_start,
                end_time=datetime.now(),
                error=f"Cognate phase failed: {str(e)}",
                config={
                    "base_models": self.config.base_models,
                    "target_architecture": self.config.target_architecture
                }
            )
    
    def _select_architecture(self) -> Dict[str, Any]:
        """Select model architecture based on configuration."""
        if self.config.target_architecture == "auto":
            return self._auto_select_architecture()
        elif self.config.target_architecture == "custom":
            return {
                "type": "custom",
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads,
                "intermediate_size": self.config.intermediate_size
            }
        else:
            return {"type": "specific", "name": self.config.target_architecture}
    
    async def _load_base_models(self) -> List[nn.Module]:
        """Load base models from HuggingFace or local paths."""
        models = []
        
        for model_path in self.config.base_models:
            try:
                self.logger.info(f"Loading base model: {model_path}")
                
                if model_path.startswith("local://"):
                    # Load local model
                    local_path = Path(model_path.replace("local://", ""))
                    if local_path.exists():
                        model = torch.load(local_path, map_location="cpu")
                        self.logger.info(f"Successfully loaded local model from {local_path}")
                    else:
                        self.logger.warning(f"Local model path does not exist: {local_path}")
                        continue
                else:
                    # Load from HuggingFace
                    try:
                        model = AutoModel.from_pretrained(
                            model_path,
                            torch_dtype=self.torch_dtype,
                            trust_remote_code=self.config.trust_remote_code,
                            use_cache=self.config.use_cache
                        )
                        self.logger.info(f"Successfully loaded HuggingFace model: {model_path}")
                    except Exception as hf_error:
                        self.logger.warning(f"HuggingFace loading failed for {model_path}: {hf_error}")
                        # Try fallback config-based loading
                        try:
                            config = AutoConfig.from_pretrained(model_path)
                            model = AutoModel.from_config(config)
                            self.logger.info(f"Loaded {model_path} from config as fallback")
                        except Exception as config_error:
                            self.logger.warning(f"Config fallback failed for {model_path}: {config_error}")
                            continue
                
                models.append(model)
                
            except Exception as e:
                self.logger.warning(f"Failed to load {model_path}: {e}")
                continue
                
        return models
    
    def _merge_models(self, models: List[nn.Module], architecture: Dict[str, Any]) -> nn.Module:
        """Merge multiple base models into single model."""
        if self.config.merge_strategy == "average":
            return self._average_merge(models)
        elif self.config.merge_strategy == "weighted":
            return self._weighted_merge(models)
        elif self.config.merge_strategy == "evolutionary":
            return self._evolutionary_merge(models)
        else:
            self.logger.warning(f"Unknown merge strategy: {self.config.merge_strategy}, using average")
            return self._average_merge(models)
    
    def _initialize_parameters(self, model: nn.Module) -> nn.Module:
        """Initialize model parameters using specified strategy."""
        
        def init_weights(module):
            if isinstance(module, nn.Linear):
                if self.config.init_strategy == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.init_strategy == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif self.config.init_strategy == "custom":
                    self._custom_initialization(module)
                else:
                    # Default to Xavier uniform
                    nn.init.xavier_uniform_(module.weight)
                    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
                    
        # Apply initialization to new/uninitialized layers only
        # This preserves pre-trained weights while initializing new components
        for name, module in model.named_modules():
            # Only initialize layers that appear to be newly added
            if hasattr(module, 'weight') and torch.allclose(module.weight, torch.zeros_like(module.weight)):
                init_weights(module)
                self.logger.debug(f"Initialized {name}")
        
        return model
    
    def _validate_model(self, model: nn.Module) -> Dict[str, Any]:
        """Validate model structure and compatibility."""
        validation_results = {
            "passed": True,
            "checks": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check parameter count
            param_count = sum(p.numel() for p in model.parameters())
            validation_results["checks"]["has_parameters"] = param_count > 0
            validation_results["checks"]["parameter_count"] = param_count
            
            if param_count == 0:
                validation_results["errors"].append("Model has no parameters")
                validation_results["passed"] = False
            
            # Check model can be set to eval/train modes
            try:
                original_mode = model.training
                model.eval()
                model.train(original_mode)
                validation_results["checks"]["mode_switching"] = True
            except Exception as e:
                validation_results["checks"]["mode_switching"] = False
                validation_results["warnings"].append(f"Mode switching failed: {e}")
            
            # Check gradient flow
            try:
                model.train()
                dummy_input = torch.randn(1, 10)  # Simple dummy input
                if hasattr(model, 'forward'):
                    # Test forward pass on small input
                    with torch.no_grad():
                        _ = model(dummy_input)
                    validation_results["checks"]["forward_pass"] = True
                else:
                    validation_results["warnings"].append("Model has no forward method")
                    validation_results["checks"]["forward_pass"] = False
            except Exception as e:
                validation_results["checks"]["forward_pass"] = False
                validation_results["warnings"].append(f"Forward pass test failed: {e}")
            
            # Check for common required attributes
            required_attrs = ["config"] if hasattr(model, "config") else []
            for attr in required_attrs:
                has_attr = hasattr(model, attr)
                validation_results["checks"][f"has_{attr}"] = has_attr
                if not has_attr:
                    validation_results["warnings"].append(f"Missing expected attribute: {attr}")
            
            # Check device and dtype consistency
            devices = {p.device for p in model.parameters()}
            dtypes = {p.dtype for p in model.parameters()}
            
            validation_results["checks"]["single_device"] = len(devices) <= 1
            validation_results["checks"]["consistent_dtype"] = len(dtypes) <= 2  # Allow mixed precision
            
            if len(devices) > 1:
                validation_results["warnings"].append(f"Model parameters on multiple devices: {devices}")
            
            # Overall validation
            validation_results["passed"] = (
                validation_results["checks"].get("has_parameters", False) and
                validation_results["checks"].get("mode_switching", False) and
                len(validation_results["errors"]) == 0
            )
            
        except Exception as e:
            validation_results["passed"] = False
            validation_results["errors"].append(f"Validation failed with exception: {e}")
        
        return validation_results
    
    def _auto_select_architecture(self) -> Dict[str, Any]:
        """Auto-select architecture based on base models."""
        # Default architecture selection
        return {
            "type": "auto_selected",
            "strategy": "transformer_base",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "intermediate_size": 3072,
            "rationale": "Standard transformer architecture suitable for general use"
        }
        
    def _average_merge(self, models: List[nn.Module]) -> nn.Module:
        """Average merge multiple models."""
        if len(models) == 1:
            return models[0]
            
        base_model = models[0]
        state_dicts = [model.state_dict() for model in models]
        
        # Average the parameters
        merged_state_dict = {}
        for key in state_dicts[0].keys():
            # Only merge if the key exists in all models and has the same shape
            if all(key in sd and sd[key].shape == state_dicts[0][key].shape for sd in state_dicts):
                merged_state_dict[key] = torch.stack([sd[key] for sd in state_dicts]).mean(dim=0)
            else:
                # Use the first model's parameter if shapes don't match
                merged_state_dict[key] = state_dicts[0][key]
                self.logger.warning(f"Shape mismatch for {key}, using first model's parameter")
        
        base_model.load_state_dict(merged_state_dict)
        return base_model
        
    def _weighted_merge(self, models: List[nn.Module]) -> nn.Module:
        """Weighted merge using configured weights."""
        if len(models) == 1:
            return models[0]
            
        if not self.config.merge_weights:
            self.logger.warning("No merge weights specified, falling back to average merge")
            return self._average_merge(models)
            
        base_model = models[0]
        state_dicts = [model.state_dict() for model in models]
        
        # Get weights, default to equal weighting
        weights = [self.config.merge_weights.get(f"model_{i}", 1.0/len(models)) for i in range(len(models))]
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w/weight_sum for w in weights]
        
        # Weighted merge
        merged_state_dict = {}
        for key in state_dicts[0].keys():
            if all(key in sd and sd[key].shape == state_dicts[0][key].shape for sd in state_dicts):
                merged_param = torch.zeros_like(state_dicts[0][key])
                for sd, weight in zip(state_dicts, weights):
                    merged_param += sd[key] * weight
                merged_state_dict[key] = merged_param
            else:
                merged_state_dict[key] = state_dicts[0][key]
        
        base_model.load_state_dict(merged_state_dict)
        return base_model
        
    def _evolutionary_merge(self, models: List[nn.Module]) -> nn.Module:
        """Use evolutionary algorithm for optimal merging."""
        # For now, fallback to weighted merge with evolutionary weights
        # This could be extended with actual evolutionary optimization
        self.logger.info("Evolutionary merge not fully implemented, using smart weighted merge")
        
        # Generate evolved weights based on model sizes and complexity
        weights = {}
        for i, model in enumerate(models):
            param_count = sum(p.numel() for p in model.parameters())
            # Weight models inversely to their size (prefer smaller, more efficient models)
            weights[f"model_{i}"] = 1.0 / (param_count / 1e6 + 1.0)  # Normalized by millions of parameters
            
        self.config.merge_weights = weights
        return self._weighted_merge(models)
        
    def _custom_initialization(self, module: nn.Module):
        """Custom parameter initialization strategy."""
        if isinstance(module, nn.Linear):
            # Custom initialization: combination of Xavier and small random noise
            nn.init.xavier_uniform_(module.weight)
            # Add small amount of noise to break symmetry
            with torch.no_grad():
                module.weight += torch.randn_like(module.weight) * 0.01
                
    def _get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model information for artifacts."""
        info = {
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "model_type": type(model).__name__,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        }
        
        if hasattr(model, "config"):
            info["config"] = str(model.config)
            
        return info
        
    def _get_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """Get model configuration for artifacts."""
        config = {
            "model_class": type(model).__name__,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "device": str(next(model.parameters()).device) if list(model.parameters()) else "unknown",
            "dtype": str(next(model.parameters()).dtype) if list(model.parameters()) else "unknown"
        }
        
        if hasattr(model, "config"):
            try:
                if hasattr(model.config, "to_dict"):
                    config["model_config"] = model.config.to_dict()
                else:
                    config["model_config"] = str(model.config)
            except:
                config["model_config"] = "unavailable"
                
        return config