#!/usr/bin/env python3
"""Test edge cases in compression"""
import torch
import torch.nn as nn
from core.compression.advanced_pipeline import AdvancedCompressionPipeline

def test_sparse_weights():
    """Test compression of sparse weights"""
    print("\n=== SPARSE WEIGHTS TEST ===")
    
    # Create very sparse tensor
    weights = torch.zeros(1000, 1000)
    weights[::10, ::10] = torch.randn(100, 100)  # Only 1% non-zero
    
    sparsity = (weights == 0).sum().item() / weights.numel()
    print(f"Sparsity: {sparsity*100:.1f}%")
    
    pipeline = AdvancedCompressionPipeline()
    # Test individual stages...


def test_repeated_patterns():
    """Test compression of repeated patterns"""
    print("\n=== REPEATED PATTERNS TEST ===")
    
    # Create weight matrix with repeated blocks
    block = torch.randn(10, 10)
    weights = block.repeat(100, 100)
    
    print(f"Weight tensor with repeated {block.shape} blocks")
    # Should compress extremely well...


def test_already_quantized():
    """Test compressing already quantized weights"""
    print("\n=== PRE-QUANTIZED WEIGHTS TEST ===")
    
    # Create int8 quantized weights
    weights = torch.randint(-128, 127, (1000, 1000), dtype=torch.float32)
    
    print("Testing compression of already quantized weights...")
    # Should reveal if compression assumes float32 input...


# Run tests
test_sparse_weights()
test_repeated_patterns() 
test_already_quantized()
