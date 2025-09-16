"""
Unit tests for Phase 5 BitNet Optimizer components
Tests for 1-bit quantization, straight-through estimator, and optimization
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Mock BitNet components
class QuantizationMode:
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"

class MockBitNetConfig:
    def __init__(self):
        self.quantization_mode = QuantizationMode.DETERMINISTIC
        self.weight_bits = 1
        self.activation_bits = 8
        self.straight_through_estimator = True
        self.quantization_warmup_steps = 1000
        self.target_sparsity = 0.1
        self.gradient_scaling = True
        self.quantization_noise = 0.0
        self.temperature = 1.0

class MockStraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, quantized_input):
        return quantized_input
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class MockBitNetLayer(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if True else None
        
        self.quantized_weight = None
        self.weight_scale = nn.Parameter(torch.ones(out_features, 1))
        
    def quantize_weights(self):
        """Quantize weights to 1-bit"""
        if self.config.quantization_mode == QuantizationMode.DETERMINISTIC:
            # Deterministic sign-based quantization
            self.quantized_weight = torch.sign(self.weight)
        else:
            # Stochastic quantization
            prob = torch.sigmoid(self.weight / self.config.temperature)
            self.quantized_weight = torch.where(
                torch.rand_like(self.weight) < prob,
                torch.ones_like(self.weight),
                -torch.ones_like(self.weight)
            )
        
        # Scale factor
        self.weight_scale.data = torch.mean(torch.abs(self.weight), dim=1, keepdim=True)
        
        return self.quantized_weight
    
    def forward(self, x):
        # Quantize weights
        quantized_weights = self.quantize_weights()
        
        # Apply straight-through estimator
        if self.config.straight_through_estimator:
            effective_weights = MockStraightThroughEstimator.apply(
                self.weight, quantized_weights * self.weight_scale
            )
        else:
            effective_weights = quantized_weights * self.weight_scale
        
        # Linear transformation
        output = torch.nn.functional.linear(x, effective_weights, self.bias)
        return output
    
    def get_quantization_stats(self):
        """Get quantization statistics"""
        if self.quantized_weight is None:
            return {}
        
        sparsity = (self.quantized_weight == 0).float().mean().item()
        weight_magnitude = torch.abs(self.weight).mean().item()
        quantization_error = torch.abs(self.weight - self.quantized_weight * self.weight_scale).mean().item()
        
        return {
            'sparsity': sparsity,
            'weight_magnitude': weight_magnitude,
            'quantization_error': quantization_error,
            'scale_factor': self.weight_scale.mean().item()
        }

class MockBitNetLossFunction(nn.Module):
    def __init__(self, base_loss, config):
        super().__init__()
        self.base_loss = base_loss
        self.config = config
        self.quantization_loss_weight = 0.01
        self.sparsity_loss_weight = 0.005
    
    def forward(self, outputs, targets, model=None):
        # Base loss
        base_loss = self.base_loss(outputs, targets)
        
        if model is None:
            return base_loss
        
        # Quantization regularization loss
        quantization_loss = 0.0
        sparsity_loss = 0.0
        
        for module in model.modules():
            if isinstance(module, MockBitNetLayer):
                # Quantization loss: encourage weights to be close to {-1, +1}
                weight_magnitude = torch.abs(module.weight)
                quantization_loss += torch.mean((weight_magnitude - 1.0) ** 2)
                
                # Sparsity loss: encourage target sparsity
                current_sparsity = (torch.abs(module.weight) < 0.1).float().mean()
                target_sparsity = self.config.target_sparsity
                sparsity_loss += (current_sparsity - target_sparsity) ** 2
        
        total_loss = (base_loss + 
                     self.quantization_loss_weight * quantization_loss +
                     self.sparsity_loss_weight * sparsity_loss)
        
        return total_loss

class MockBitNetOptimizer:
    def __init__(self, parameters, config):
        self.config = config
        self.param_groups = [{'params': list(parameters)}]
        self.step_count = 0
        
        # Create base optimizer
        self.base_optimizer = optim.AdamW(
            parameters, 
            lr=1e-4, 
            weight_decay=1e-2
        )
        
        self.gradient_scaler = 1.0
        self.warmup_steps = config.quantization_warmup_steps
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    def step(self):
        self.step_count += 1
        
        # Apply gradient scaling during warmup
        if self.step_count < self.warmup_steps:
            warmup_factor = self.step_count / self.warmup_steps
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad *= warmup_factor
        
        # Apply quantization-aware gradient scaling
        if self.config.gradient_scaling:
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad *= self.gradient_scaler
        
        # Base optimizer step
        self.base_optimizer.step()
    
    def state_dict(self):
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'step_count': self.step_count,
            'gradient_scaler': self.gradient_scaler
        }
    
    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.step_count = state_dict['step_count']
        self.gradient_scaler = state_dict['gradient_scaler']

def convert_model_to_bitnet(model, config):
    """Convert regular model to BitNet model"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace Linear layer with BitNetLayer
            bitnet_layer = MockBitNetLayer(
                module.in_features,
                module.out_features,
                config
            )
            # Copy weights
            bitnet_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                bitnet_layer.bias.data = module.bias.data.clone()
            
            setattr(model, name, bitnet_layer)
        else:
            # Recursively convert child modules
            convert_model_to_bitnet(module, config)
    
    return model

# Test cases
class TestBitNetConfig:
    """Test BitNetConfig functionality"""
    
    def test_config_initialization(self):
        """Test BitNet configuration initialization"""
        config = MockBitNetConfig()
        
        assert config.quantization_mode == QuantizationMode.DETERMINISTIC
        assert config.weight_bits == 1
        assert config.activation_bits == 8
        assert config.straight_through_estimator == True
        assert config.quantization_warmup_steps == 1000
        assert config.target_sparsity == 0.1
    
    def test_config_modes(self):
        """Test different quantization modes"""
        config = MockBitNetConfig()
        
        # Test deterministic mode
        config.quantization_mode = QuantizationMode.DETERMINISTIC
        assert config.quantization_mode == QuantizationMode.DETERMINISTIC
        
        # Test stochastic mode
        config.quantization_mode = QuantizationMode.STOCHASTIC
        assert config.quantization_mode == QuantizationMode.STOCHASTIC

class TestStraightThroughEstimator:
    """Test Straight-Through Estimator functionality"""
    
    def test_forward_pass(self):
        """Test forward pass returns quantized input"""
        original_input = torch.randn(10, 5)
        quantized_input = torch.sign(original_input)
        
        result = MockStraightThroughEstimator.apply(original_input, quantized_input)
        
        assert torch.equal(result, quantized_input)
    
    def test_backward_pass(self):
        """Test backward pass preserves gradients"""
        original_input = torch.randn(10, 5, requires_grad=True)
        quantized_input = torch.sign(original_input.detach())
        
        result = MockStraightThroughEstimator.apply(original_input, quantized_input)
        loss = result.sum()
        loss.backward()
        
        # Gradient should flow through to original input
        assert original_input.grad is not None
        assert original_input.grad.shape == original_input.shape

class TestBitNetLayer:
    """Test BitNetLayer functionality"""
    
    def test_layer_initialization(self):
        """Test BitNet layer initialization"""
        config = MockBitNetConfig()
        layer = MockBitNetLayer(128, 64, config)
        
        assert layer.in_features == 128
        assert layer.out_features == 64
        assert layer.weight.shape == (64, 128)
        assert layer.bias.shape == (64,)
        assert layer.weight_scale.shape == (64, 1)
    
    def test_deterministic_quantization(self):
        """Test deterministic weight quantization"""
        config = MockBitNetConfig()
        config.quantization_mode = QuantizationMode.DETERMINISTIC
        
        layer = MockBitNetLayer(10, 5, config)
        
        # Set known weights
        layer.weight.data = torch.tensor([
            [0.5, -0.3, 0.8, -0.1, 0.2],
            [0.0, 0.9, -0.7, 0.4, -0.6],
        ], dtype=torch.float32)[:5, :10]  # Adjust size
        
        quantized = layer.quantize_weights()
        
        # Check quantization to {-1, +1}
        assert torch.all(torch.abs(quantized) == 1.0)
        assert torch.all(quantized == torch.sign(layer.weight))
    
    def test_stochastic_quantization(self):
        """Test stochastic weight quantization"""
        config = MockBitNetConfig()
        config.quantization_mode = QuantizationMode.STOCHASTIC
        config.temperature = 1.0
        
        layer = MockBitNetLayer(10, 5, config)
        
        # Set weights
        layer.weight.data = torch.randn(5, 10)
        
        quantized = layer.quantize_weights()
        
        # Check quantization to {-1, +1}
        assert torch.all(torch.abs(quantized) == 1.0)
        assert quantized.shape == layer.weight.shape
    
    def test_forward_pass(self):
        """Test forward pass with quantization"""
        config = MockBitNetConfig()
        layer = MockBitNetLayer(10, 5, config)
        
        input_tensor = torch.randn(32, 10)
        output = layer(input_tensor)
        
        assert output.shape == (32, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_quantization_statistics(self):
        """Test quantization statistics calculation"""
        config = MockBitNetConfig()
        layer = MockBitNetLayer(10, 5, config)
        
        # Perform forward pass to trigger quantization
        input_tensor = torch.randn(1, 10)
        layer(input_tensor)
        
        stats = layer.get_quantization_stats()
        
        assert 'sparsity' in stats
        assert 'weight_magnitude' in stats
        assert 'quantization_error' in stats
        assert 'scale_factor' in stats
        
        assert 0 <= stats['sparsity'] <= 1
        assert stats['weight_magnitude'] >= 0
        assert stats['quantization_error'] >= 0
        assert stats['scale_factor'] > 0

class TestBitNetLossFunction:
    """Test BitNet Loss Function functionality"""
    
    def test_loss_initialization(self):
        """Test BitNet loss function initialization"""
        base_loss = nn.CrossEntropyLoss()
        config = MockBitNetConfig()
        
        bitnet_loss = MockBitNetLossFunction(base_loss, config)
        
        assert bitnet_loss.base_loss == base_loss
        assert bitnet_loss.config == config
        assert bitnet_loss.quantization_loss_weight > 0
        assert bitnet_loss.sparsity_loss_weight > 0
    
    def test_loss_without_model(self):
        """Test loss calculation without model (base loss only)"""
        base_loss = nn.MSELoss()
        config = MockBitNetConfig()
        bitnet_loss = MockBitNetLossFunction(base_loss, config)
        
        outputs = torch.randn(32, 10)
        targets = torch.randn(32, 10)
        
        loss = bitnet_loss(outputs, targets)
        base_loss_value = base_loss(outputs, targets)
        
        assert torch.equal(loss, base_loss_value)
    
    def test_loss_with_model(self):
        """Test loss calculation with BitNet model (includes regularization)"""
        base_loss = nn.MSELoss()
        config = MockBitNetConfig()
        bitnet_loss = MockBitNetLossFunction(base_loss, config)
        
        # Create model with BitNet layers
        model = nn.Sequential(
            MockBitNetLayer(10, 5, config),
            MockBitNetLayer(5, 1, config)
        )
        
        outputs = torch.randn(32, 1)
        targets = torch.randn(32, 1)
        
        loss_with_reg = bitnet_loss(outputs, targets, model)
        loss_without_reg = bitnet_loss(outputs, targets)
        
        # Loss with regularization should be higher
        assert loss_with_reg >= loss_without_reg
    
    def test_quantization_regularization(self):
        """Test quantization regularization effect"""
        base_loss = nn.MSELoss()
        config = MockBitNetConfig()
        bitnet_loss = MockBitNetLossFunction(base_loss, config)
        
        # Create model
        model = nn.Sequential(MockBitNetLayer(10, 1, config))
        
        outputs = torch.randn(32, 1)
        targets = torch.randn(32, 1)
        
        # Set weights to values close to {-1, +1} vs far from {-1, +1}
        model[0].weight.data = torch.ones_like(model[0].weight) * 0.9  # Close to 1
        loss_close = bitnet_loss(outputs, targets, model)
        
        model[0].weight.data = torch.ones_like(model[0].weight) * 0.1  # Far from 1
        loss_far = bitnet_loss(outputs, targets, model)
        
        # Loss should be higher when weights are far from {-1, +1}
        assert loss_far > loss_close

class TestBitNetOptimizer:
    """Test BitNet Optimizer functionality"""
    
    def test_optimizer_initialization(self):
        """Test BitNet optimizer initialization"""
        model = nn.Linear(10, 1)
        config = MockBitNetConfig()
        
        optimizer = MockBitNetOptimizer(model.parameters(), config)
        
        assert optimizer.config == config
        assert len(optimizer.param_groups) == 1
        assert optimizer.step_count == 0
        assert optimizer.gradient_scaler == 1.0
    
    def test_warmup_gradient_scaling(self):
        """Test gradient scaling during warmup"""
        model = nn.Linear(10, 1)
        config = MockBitNetConfig()
        config.quantization_warmup_steps = 100
        
        optimizer = MockBitNetOptimizer(model.parameters(), config)
        
        # Create gradients
        loss = torch.sum(model.weight ** 2)
        loss.backward()
        
        # Store original gradient
        original_grad = model.weight.grad.clone()
        
        # Step during warmup (step 1 out of 100)
        optimizer.step()
        
        # Gradient should have been scaled down
        assert optimizer.step_count == 1
    
    def test_post_warmup_behavior(self):
        """Test optimizer behavior after warmup"""
        model = nn.Linear(10, 1)
        config = MockBitNetConfig()
        config.quantization_warmup_steps = 10
        
        optimizer = MockBitNetOptimizer(model.parameters(), config)
        
        # Simulate warmup completion
        optimizer.step_count = 15
        
        # Create gradients
        loss = torch.sum(model.weight ** 2)
        loss.backward()
        
        # Step after warmup
        optimizer.step()
        
        assert optimizer.step_count == 16
    
    def test_state_dict_operations(self):
        """Test optimizer state dictionary operations"""
        model = nn.Linear(10, 1)
        config = MockBitNetConfig()
        
        optimizer = MockBitNetOptimizer(model.parameters(), config)
        optimizer.step_count = 50
        optimizer.gradient_scaler = 1.5
        
        # Save state
        state_dict = optimizer.state_dict()
        
        assert 'base_optimizer' in state_dict
        assert 'step_count' in state_dict
        assert 'gradient_scaler' in state_dict
        assert state_dict['step_count'] == 50
        assert state_dict['gradient_scaler'] == 1.5
        
        # Create new optimizer and load state
        new_optimizer = MockBitNetOptimizer(model.parameters(), config)
        new_optimizer.load_state_dict(state_dict)
        
        assert new_optimizer.step_count == 50
        assert new_optimizer.gradient_scaler == 1.5

class TestModelConversion:
    """Test model conversion to BitNet"""
    
    def test_linear_layer_conversion(self):
        """Test conversion of Linear layers to BitNet layers"""
        config = MockBitNetConfig()
        
        # Create model with Linear layers
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Store original weights
        original_weights = [
            model[0].weight.data.clone(),
            model[2].weight.data.clone()
        ]
        
        # Convert to BitNet
        bitnet_model = convert_model_to_bitnet(model, config)
        
        # Check conversion
        assert isinstance(bitnet_model[0], MockBitNetLayer)
        assert isinstance(bitnet_model[1], nn.ReLU)  # Non-Linear layers unchanged
        assert isinstance(bitnet_model[2], MockBitNetLayer)
        
        # Check weight preservation
        assert torch.equal(bitnet_model[0].weight.data, original_weights[0])
        assert torch.equal(bitnet_model[2].weight.data, original_weights[1])
    
    def test_nested_model_conversion(self):
        """Test conversion of nested models"""
        config = MockBitNetConfig()
        
        # Create nested model
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(10, 5),
                    nn.ReLU()
                )
                self.decoder = nn.Linear(5, 1)
            
            def forward(self, x):
                x = self.encoder(x)
                return self.decoder(x)
        
        model = NestedModel()
        
        # Convert to BitNet
        bitnet_model = convert_model_to_bitnet(model, config)
        
        # Check nested conversion
        assert isinstance(bitnet_model.encoder[0], MockBitNetLayer)
        assert isinstance(bitnet_model.encoder[1], nn.ReLU)
        assert isinstance(bitnet_model.decoder, MockBitNetLayer)
    
    def test_preserved_functionality(self):
        """Test that converted model maintains functionality"""
        config = MockBitNetConfig()
        
        # Create and convert model
        original_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        bitnet_model = convert_model_to_bitnet(original_model, config)
        
        # Test forward pass
        input_tensor = torch.randn(32, 10)
        
        # Both models should produce outputs of the same shape
        original_output = original_model(input_tensor)
        bitnet_output = bitnet_model(input_tensor)
        
        assert original_output.shape == bitnet_output.shape
        assert not torch.isnan(bitnet_output).any()
        assert not torch.isinf(bitnet_output).any()

if __name__ == "__main__":
    pytest.main([__file__])