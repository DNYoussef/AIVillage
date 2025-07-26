#!/usr/bin/env python3
"""
Memory Management Utilities for Agent Forge

Provides safe memory management for large model operations:
- GPU memory monitoring and cleanup
- Safe model loading with memory checks
- Gradient checkpointing utilities
- Memory-efficient tensor operations
"""

import gc
import logging
import psutil
import torch
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

class MemoryManager:
    """Comprehensive memory management for AI operations."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = torch.cuda.is_available()
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {
            'system_ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'system_ram_available_gb': psutil.virtual_memory().available / (1024**3),
            'system_ram_percent': psutil.virtual_memory().percent
        }
        
        if self.is_cuda:
            stats.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'gpu_memory_free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                                     torch.cuda.memory_reserved()) / (1024**3)
            })
        
        return stats
    
    def cleanup_memory(self, aggressive: bool = False):
        """Clean up memory safely."""
        gc.collect()
        
        if self.is_cuda:
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.ipc_collect()
        
        logger.debug(f"Memory cleanup completed. Stats: {self.get_memory_stats()}")
    
    def check_memory_available(self, required_gb: float) -> bool:
        """Check if enough memory is available for operation."""
        stats = self.get_memory_stats()
        
        if self.is_cuda:
            available = stats['gpu_memory_free_gb']
        else:
            available = stats['system_ram_available_gb']
        
        return available > required_gb
    
    @contextmanager
    def memory_guard(self, operation_name: str = "operation"):
        """Context manager for safe memory operations."""
        logger.info(f"Starting {operation_name} with memory guard")
        initial_stats = self.get_memory_stats()
        
        try:
            yield
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"Out of memory during {operation_name}")
                self.cleanup_memory(aggressive=True)
                raise MemoryError(f"Insufficient memory for {operation_name}: {e}")
            else:
                raise
        finally:
            self.cleanup_memory()
            final_stats = self.get_memory_stats()
            logger.info(f"Completed {operation_name}. Memory change: "
                       f"{final_stats.get('gpu_memory_allocated_gb', 0) - initial_stats.get('gpu_memory_allocated_gb', 0):.2f}GB")

def memory_efficient_operation(func: Callable) -> Callable:
    """Decorator for memory-efficient operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        memory_manager = MemoryManager()
        with memory_manager.memory_guard(func.__name__):
            return func(*args, **kwargs)
    return wrapper

class SafeModelLoader:
    """Safe model loading with memory management."""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
    
    def estimate_model_memory(self, model_name: str) -> float:
        """Estimate memory requirements for a model."""
        # Rough estimates based on parameter count
        size_estimates = {
            "1.5B": 3.0,  # GB
            "7B": 14.0,
            "13B": 26.0
        }
        
        # Extract parameter count from model name
        for size_key, memory_gb in size_estimates.items():
            if size_key.lower() in model_name.lower():
                return memory_gb
        
        # Default estimate for unknown models
        return 4.0
    
    def load_model_safely(self, model_path: str, **kwargs) -> tuple:
        """Load model with memory checks and error handling."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        estimated_memory = self.estimate_model_memory(model_path)
        
        if not self.memory_manager.check_memory_available(estimated_memory):
            raise MemoryError(f"Insufficient memory to load model {model_path}. "
                            f"Required: {estimated_memory}GB")
        
        logger.info(f"Loading model {model_path} (estimated {estimated_memory}GB)")
        
        try:
            with self.memory_manager.memory_guard(f"loading {model_path}"):
                # Load tokenizer first (smaller memory footprint)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model with appropriate precision based on available memory
                device_map = "auto" if self.memory_manager.is_cuda else None
                torch_dtype = torch.float16 if self.memory_manager.is_cuda else torch.float32
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
                
                return model, tokenizer
                
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            self.memory_manager.cleanup_memory(aggressive=True)
            raise

class GradientCheckpointing:
    """Utilities for gradient checkpointing to reduce memory usage."""
    
    @staticmethod
    def enable_for_model(model):
        """Enable gradient checkpointing for a model."""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model does not support gradient checkpointing")
    
    @staticmethod
    def optimize_model_for_memory(model):
        """Apply various memory optimizations to a model."""
        GradientCheckpointing.enable_for_model(model)
        
        # Enable attention slicing if available
        if hasattr(model, 'enable_attention_slicing'):
            model.enable_attention_slicing()
            logger.info("Attention slicing enabled")
        
        # Enable memory efficient attention if available
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
            logger.info("Disabled KV caching for memory efficiency")

# Global memory manager instance
memory_manager = MemoryManager()
safe_model_loader = SafeModelLoader()