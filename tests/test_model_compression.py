"""Tests for model compression system."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List

from agent_forge.model_compression.model_compression import (
    CompressionConfig,
    TernaryQuantizer,
    VPTQLinear,
    BitLinear,
    CompressedModel,
    quantize_activations,
    compress_and_train
)

@pytest.fixture
def config():
    """Create test configuration."""
    return CompressionConfig(
        vector_size=8,
        codebook_size=256,
        group_size=128,
        batch_size=32,
        learning_rate=1e-4,
        epochs=1,
        device='cpu'  # Use CPU for testing
    )

@pytest.fixture
def mock_model():
    """Create mock PyTorch model."""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(5, 2)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    model = TestModel()
    model = model.cpu()
    return model

def test_compression_config():
    """Test compression configuration."""
    config = CompressionConfig()
    assert config.vector_size == 8
    assert config.codebook_size == 256
    assert config.group_size == 128
    assert config.lambda_warmup == 1000
    assert config.batch_size == 32
    assert config.learning_rate == 1e-4

def test_activation_quantization():
    """Test activation quantization."""
    x = torch.randn(10, 10)
    quantized = quantize_activations(x)
    
    # Verify quantization
    assert torch.all(torch.abs(quantized) <= 127)
    assert quantized.shape == x.shape

def test_bit_linear():
    """Test BitLinear layer."""
    layer = BitLinear(10, 5)
    layer = layer.cpu()
    input_tensor = torch.randn(32, 10)
    
    # Test forward pass
    output = layer(input_tensor)
    assert output.shape == (32, 5)
    
    # Test weight quantization
    layer.quantize_weight()
    assert layer.quantized_weight.dtype == torch.int8
    assert torch.all(torch.abs(layer.quantized_weight) <= 1)

@pytest.mark.asyncio
async def test_compression_pipeline(config, mock_model):
    """Test full compression pipeline."""
    # Create dummy data
    train_data = torch.randn(100, 10)
    train_labels = torch.randint(0, 2, (100,))
    train_loader = torch.utils.data.DataLoader(
        list(zip(train_data, train_labels)),
        batch_size=config.batch_size,
        shuffle=False  # Disable shuffling for reproducibility
    )
    val_loader = torch.utils.data.DataLoader(
        list(zip(train_data[:10], train_labels[:10])),
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Mock loss backward
    mock_loss = torch.tensor(0.5, requires_grad=True)
    mock_loss.backward = Mock()
    mock_criterion = Mock(return_value=mock_loss)
    mock_optimizer = Mock()
    mock_optimizer.step = Mock()
    mock_optimizer.zero_grad = Mock()
    
    # Mock VPTQLinear
    class MockVPTQLinear(nn.Linear):
        def __init__(self, in_features, out_features, bias=True, config=None):
            super().__init__(in_features, out_features, bias)
            self.config = config
            if config:
                self.centroids = torch.randn(config.codebook_size, config.vector_size)
                self.assignments = torch.randint(0, config.codebook_size, (in_features,))
                self.scales = torch.ones(in_features)
            else:
                self.centroids = None
                self.assignments = None
                self.scales = None
        
        def quantize(self):
            return self.weight.clone()
        
        def forward(self, x):
            return F.linear(x, self.weight, self.bias)
    
    # Run compression pipeline
    with patch('torch.optim.Adam', return_value=mock_optimizer), \
         patch('torch.nn.CrossEntropyLoss', return_value=mock_criterion), \
         patch('torch.Tensor.backward', Mock()), \
         patch('agent_forge.model_compression.model_compression.VPTQLinear', MockVPTQLinear):
        compressed_model, stats = await compress_and_train(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
    
    # Verify compression results
    assert isinstance(compressed_model, CompressedModel)
    assert "config" in stats
    assert "final_loss" in stats
    assert stats["final_loss"] >= 0
    
    # Verify model structure
    linear_count = 0
    for module in compressed_model.model.modules():
        if isinstance(module, (MockVPTQLinear, BitLinear)):
            linear_count += 1
    assert linear_count > 0  # Ensure we found quantized layers

def test_compressed_model(config, mock_model):
    """Test CompressedModel wrapper."""
    # Create mock VPTQLinear
    class MockVPTQLinear(nn.Linear):
        def __init__(self, in_features, out_features, bias=True, config=None):
            super().__init__(in_features, out_features, bias)
            self.config = config
            if config:
                self.centroids = torch.randn(config.codebook_size, config.vector_size)
                self.assignments = torch.randint(0, config.codebook_size, (in_features,))
                self.scales = torch.ones(in_features)
            else:
                self.centroids = None
                self.assignments = None
                self.scales = None
        
        def quantize(self):
            return self.weight.clone()
        
        def forward(self, x):
            return F.linear(x, self.weight, self.bias)
    
    # Mock BitLinear
    class MockBitLinear(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__(in_features, out_features, bias)
            self.weight_scale = torch.ones(1)
            self.quantized_weight = torch.zeros_like(self.weight, dtype=torch.int8)
        
        def quantize_weight(self):
            self.quantized_weight = torch.randint(-1, 2, self.weight.shape, dtype=torch.int8)
            self.weight_scale = torch.ones(1)
        
        def forward(self, x):
            x_norm = F.layer_norm(x, x.shape[-1:])
            x_quant = x_norm + (quantize_activations(x_norm) - x_norm).detach()
            w_quant = self.quantized_weight.float() * self.weight_scale
            return F.linear(x_quant, w_quant, self.bias)
    
    # Mock VPTQLinear creation
    with patch('agent_forge.model_compression.model_compression.VPTQLinear', MockVPTQLinear), \
         patch('agent_forge.model_compression.model_compression.BitLinear', MockBitLinear):
        # Create compressed model
        compressed = CompressedModel(mock_model, config)
        
        # Test VPTQ conversion
        linear_count = 0
        for module in compressed.model.modules():
            if isinstance(module, nn.Linear):
                linear_count += 1
                assert isinstance(module, MockVPTQLinear)
        assert linear_count > 0  # Ensure we found some linear layers
        
        # Test BitNet conversion
        compressed.convert_to_bitnet()
        linear_count = 0
        for module in compressed.model.modules():
            if isinstance(module, nn.Linear):
                linear_count += 1
                assert isinstance(module, MockBitLinear)
        assert linear_count > 0  # Ensure we found some linear layers
        
        # Test forward pass
        input_tensor = torch.randn(32, 10)
        output = compressed(input_tensor)
        assert output.shape == (32, 2)

if __name__ == "__main__":
    pytest.main([__file__])
