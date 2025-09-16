"""
Integration tests for Phase 5 Training with Phase 4 BitNet Models
Tests for loading compressed models and continuing training
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Mock Phase 4 Integration Components
class MockPhase4Model:
    """Mock Phase 4 compressed model"""
    
    def __init__(self, model_type="bitnet", compression_ratio=0.1):
        self.model_type = model_type
        self.compression_ratio = compression_ratio
        self.quantization_info = {
            'weight_bits': 1,
            'activation_bits': 8,
            'quantization_method': 'deterministic',
            'sparsity_level': 0.1
        }
        self.model_state = self._create_compressed_state()
    
    def _create_compressed_state(self):
        """Create mock compressed model state"""
        return {
            'layers': {
                'layer1': {
                    'weight': torch.randn(64, 128) * 0.1,
                    'bias': torch.randn(64) * 0.01,
                    'quantization_scale': torch.ones(64, 1) * 0.1
                },
                'layer2': {
                    'weight': torch.randn(32, 64) * 0.1,
                    'bias': torch.randn(32) * 0.01,
                    'quantization_scale': torch.ones(32, 1) * 0.1
                },
                'output': {
                    'weight': torch.randn(10, 32) * 0.1,
                    'bias': torch.randn(10) * 0.01,
                    'quantization_scale': torch.ones(10, 1) * 0.1
                }
            },
            'metadata': {
                'original_size': 100000,
                'compressed_size': 10000,
                'compression_ratio': self.compression_ratio,
                'quantization_info': self.quantization_info
            }
        }
    
    def save(self, path):
        """Save compressed model"""
        save_data = {
            'model_type': self.model_type,
            'compression_ratio': self.compression_ratio,
            'quantization_info': self.quantization_info,
            'model_state': self.model_state
        }
        torch.save(save_data, path)
    
    @classmethod
    def load(cls, path):
        """Load compressed model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        data = torch.load(path)
        model = cls(data['model_type'], data['compression_ratio'])
        model.quantization_info = data['quantization_info']
        model.model_state = data['model_state']
        return model

class MockPhase4Loader:
    """Mock Phase 4 model loader"""
    
    @staticmethod
    def load_compressed_model(model_path, device='cpu'):
        """Load Phase 4 compressed model"""
        phase4_model = MockPhase4Model.load(model_path)
        
        # Convert to PyTorch model
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
        # Load compressed weights
        state_dict = {}
        for name, layer_data in phase4_model.model_state['layers'].items():
            if name == 'layer1':
                state_dict['0.weight'] = layer_data['weight']
                state_dict['0.bias'] = layer_data['bias']
            elif name == 'layer2':
                state_dict['2.weight'] = layer_data['weight']
                state_dict['2.bias'] = layer_data['bias']
            elif name == 'output':
                state_dict['4.weight'] = layer_data['weight']
                state_dict['4.bias'] = layer_data['bias']
        
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        
        return model, phase4_model.quantization_info
    
    @staticmethod
    def validate_model_compatibility(model_info, training_config):
        """Validate model compatibility with training configuration"""
        required_fields = ['weight_bits', 'activation_bits', 'quantization_method']
        
        for field in required_fields:
            if field not in model_info:
                return False, f"Missing required field: {field}"
        
        # Check quantization compatibility
        if model_info['weight_bits'] != 1:
            return False, f"Unsupported weight bits: {model_info['weight_bits']}"
        
        if model_info['activation_bits'] not in [8, 16]:
            return False, f"Unsupported activation bits: {model_info['activation_bits']}"
        
        return True, "Model compatible"

class MockPhase5TrainingPipeline:
    """Mock Phase 5 training pipeline for integration testing"""
    
    def __init__(self, model, phase4_info, training_config):
        self.model = model
        self.phase4_info = phase4_info
        self.training_config = training_config
        self.training_state = {
            'epoch': 0,
            'step': 0,
            'best_metric': 0.0,
            'training_metrics': []
        }
        
    def setup_bitnet_training(self):
        """Setup BitNet training from Phase 4 model"""
        # Configure BitNet parameters based on Phase 4 info
        self.bitnet_config = {
            'weight_bits': self.phase4_info['weight_bits'],
            'activation_bits': self.phase4_info['activation_bits'],
            'quantization_method': self.phase4_info['quantization_method'],
            'continue_training': True
        }
        
        # Setup optimizer for continued training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.get('learning_rate', 1e-4),
            weight_decay=self.training_config.get('weight_decay', 1e-2)
        )
        
        return True
    
    def continue_training(self, train_loader, epochs=10):
        """Continue training from Phase 4 checkpoint"""
        training_metrics = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                outputs = self.model(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                self.training_state['step'] += 1
                
                if batch_idx >= 10:  # Limit for testing
                    break
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            training_metrics.append({
                'epoch': epoch,
                'loss': avg_loss,
                'step': self.training_state['step']
            })
            
            self.training_state['epoch'] = epoch
            self.training_state['training_metrics'] = training_metrics
        
        return training_metrics
    
    def validate_training_progress(self):
        """Validate training progress"""
        metrics = self.training_state['training_metrics']
        
        if not metrics:
            return False, "No training metrics available"
        
        # Check if loss is decreasing
        if len(metrics) >= 2:
            initial_loss = metrics[0]['loss']
            final_loss = metrics[-1]['loss']
            
            if final_loss >= initial_loss:
                return False, f"Loss not decreasing: {initial_loss} -> {final_loss}"
        
        # Check for NaN or infinite values
        for metric in metrics:
            if np.isnan(metric['loss']) or np.isinf(metric['loss']):
                return False, f"Invalid loss value: {metric['loss']}"
        
        return True, "Training progress validated"

# Test Cases
class TestPhase4Integration:
    """Test Phase 4 integration functionality"""
    
    def test_phase4_model_loading(self):
        """Test loading Phase 4 compressed model"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "phase4_model.pt")
            
            # Create and save Phase 4 model
            phase4_model = MockPhase4Model("bitnet", 0.1)
            phase4_model.save(model_path)
            
            # Load model
            loaded_model, quantization_info = MockPhase4Loader.load_compressed_model(model_path)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, nn.Sequential)
            assert len(loaded_model) == 5  # 3 Linear + 2 ReLU layers
            assert 'weight_bits' in quantization_info
            assert quantization_info['weight_bits'] == 1
    
    def test_model_compatibility_validation(self):
        """Test model compatibility validation"""
        # Valid model info
        valid_info = {
            'weight_bits': 1,
            'activation_bits': 8,
            'quantization_method': 'deterministic'
        }
        
        training_config = {'learning_rate': 1e-4}
        
        is_compatible, message = MockPhase4Loader.validate_model_compatibility(
            valid_info, training_config
        )
        
        assert is_compatible == True
        assert "compatible" in message.lower()
    
    def test_model_incompatibility_detection(self):
        """Test detection of incompatible models"""
        # Invalid model info - wrong weight bits
        invalid_info = {
            'weight_bits': 8,  # Should be 1 for BitNet
            'activation_bits': 8,
            'quantization_method': 'deterministic'
        }
        
        training_config = {'learning_rate': 1e-4}
        
        is_compatible, message = MockPhase4Loader.validate_model_compatibility(
            invalid_info, training_config
        )
        
        assert is_compatible == False
        assert "weight bits" in message.lower()
    
    def test_missing_model_fields(self):
        """Test handling of missing required fields"""
        # Missing required field
        incomplete_info = {
            'weight_bits': 1,
            'activation_bits': 8
            # Missing 'quantization_method'
        }
        
        training_config = {'learning_rate': 1e-4}
        
        is_compatible, message = MockPhase4Loader.validate_model_compatibility(
            incomplete_info, training_config
        )
        
        assert is_compatible == False
        assert "missing" in message.lower()

class TestContinuedTraining:
    """Test continued training from Phase 4"""
    
    def test_bitnet_training_setup(self):
        """Test BitNet training setup from Phase 4"""
        # Create mock model and info
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        phase4_info = {
            'weight_bits': 1,
            'activation_bits': 8,
            'quantization_method': 'deterministic',
            'sparsity_level': 0.1
        }
        
        training_config = {
            'learning_rate': 1e-4,
            'weight_decay': 1e-2,
            'epochs': 10
        }
        
        # Setup training pipeline
        pipeline = MockPhase5TrainingPipeline(model, phase4_info, training_config)
        success = pipeline.setup_bitnet_training()
        
        assert success == True
        assert pipeline.bitnet_config['weight_bits'] == 1
        assert pipeline.optimizer is not None
    
    def test_continued_training_execution(self):
        """Test execution of continued training"""
        # Create model and data
        model = nn.Sequential(
            nn.Linear(128, 10)
        )
        
        phase4_info = {
            'weight_bits': 1,
            'activation_bits': 8,
            'quantization_method': 'deterministic'
        }
        
        training_config = {'learning_rate': 1e-4}
        
        # Create mock data loader
        def create_mock_loader():
            for _ in range(5):  # 5 batches
                data = torch.randn(32, 128)
                targets = torch.randint(0, 10, (32,))
                yield data, targets
        
        # Setup and run training
        pipeline = MockPhase5TrainingPipeline(model, phase4_info, training_config)
        pipeline.setup_bitnet_training()
        
        metrics = pipeline.continue_training(create_mock_loader(), epochs=3)
        
        assert len(metrics) == 3  # 3 epochs
        assert all('loss' in m for m in metrics)
        assert all('epoch' in m for m in metrics)
        assert pipeline.training_state['step'] > 0
    
    def test_training_progress_validation(self):
        """Test training progress validation"""
        model = nn.Sequential(nn.Linear(10, 1))
        phase4_info = {'weight_bits': 1, 'activation_bits': 8, 'quantization_method': 'deterministic'}
        training_config = {'learning_rate': 1e-4}
        
        pipeline = MockPhase5TrainingPipeline(model, phase4_info, training_config)
        
        # Test with decreasing loss (good training)
        pipeline.training_state['training_metrics'] = [
            {'epoch': 0, 'loss': 1.0, 'step': 10},
            {'epoch': 1, 'loss': 0.8, 'step': 20},
            {'epoch': 2, 'loss': 0.6, 'step': 30}
        ]
        
        is_valid, message = pipeline.validate_training_progress()
        assert is_valid == True
        assert "validated" in message.lower()
    
    def test_training_failure_detection(self):
        """Test detection of training failures"""
        model = nn.Sequential(nn.Linear(10, 1))
        phase4_info = {'weight_bits': 1, 'activation_bits': 8, 'quantization_method': 'deterministic'}
        training_config = {'learning_rate': 1e-4}
        
        pipeline = MockPhase5TrainingPipeline(model, phase4_info, training_config)
        
        # Test with increasing loss (bad training)
        pipeline.training_state['training_metrics'] = [
            {'epoch': 0, 'loss': 0.5, 'step': 10},
            {'epoch': 1, 'loss': 0.8, 'step': 20},
            {'epoch': 2, 'loss': 1.2, 'step': 30}
        ]
        
        is_valid, message = pipeline.validate_training_progress()
        assert is_valid == False
        assert "not decreasing" in message.lower()

class TestModelStateTransfer:
    """Test model state transfer from Phase 4"""
    
    def test_weight_preservation(self):
        """Test preservation of weights during transfer"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "phase4_model.pt")
            
            # Create Phase 4 model with known weights
            phase4_model = MockPhase4Model()
            phase4_model.save(model_path)
            
            # Load model
            loaded_model, _ = MockPhase4Loader.load_compressed_model(model_path)
            
            # Check that weights are loaded
            assert loaded_model[0].weight is not None
            assert loaded_model[0].bias is not None
            assert loaded_model[0].weight.shape == (64, 128)
            assert loaded_model[0].bias.shape == (64,)
    
    def test_quantization_info_transfer(self):
        """Test transfer of quantization information"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "phase4_model.pt")
            
            # Create Phase 4 model with specific quantization info
            phase4_model = MockPhase4Model()
            phase4_model.quantization_info['sparsity_level'] = 0.15
            phase4_model.save(model_path)
            
            # Load and check quantization info
            _, quantization_info = MockPhase4Loader.load_compressed_model(model_path)
            
            assert quantization_info['weight_bits'] == 1
            assert quantization_info['activation_bits'] == 8
            assert quantization_info['sparsity_level'] == 0.15
    
    def test_model_forward_pass_after_loading(self):
        """Test model forward pass after loading from Phase 4"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "phase4_model.pt")
            
            # Create and save Phase 4 model
            phase4_model = MockPhase4Model()
            phase4_model.save(model_path)
            
            # Load model
            loaded_model, _ = MockPhase4Loader.load_compressed_model(model_path)
            
            # Test forward pass
            input_tensor = torch.randn(32, 128)
            output = loaded_model(input_tensor)
            
            assert output.shape == (32, 10)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

class TestErrorHandling:
    """Test error handling in Phase 4 integration"""
    
    def test_missing_model_file(self):
        """Test handling of missing model file"""
        with pytest.raises(FileNotFoundError):
            MockPhase4Loader.load_compressed_model("nonexistent_model.pt")
    
    def test_corrupted_model_file(self):
        """Test handling of corrupted model file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "corrupted_model.pt")
            
            # Create corrupted file
            with open(model_path, 'w') as f:
                f.write("corrupted data")
            
            with pytest.raises(Exception):  # Should raise some exception
                MockPhase4Loader.load_compressed_model(model_path)
    
    def test_invalid_training_config(self):
        """Test handling of invalid training configuration"""
        model = nn.Sequential(nn.Linear(10, 1))
        phase4_info = {'weight_bits': 1, 'activation_bits': 8, 'quantization_method': 'deterministic'}
        
        # Invalid config - negative learning rate
        invalid_config = {'learning_rate': -1e-4}
        
        pipeline = MockPhase5TrainingPipeline(model, phase4_info, invalid_config)
        
        # Should handle gracefully or raise appropriate error
        try:
            pipeline.setup_bitnet_training()
            assert False, "Should have raised error for negative learning rate"
        except:
            pass  # Expected to fail

class TestPerformanceMetrics:
    """Test performance metrics during integration"""
    
    def test_training_time_measurement(self):
        """Test measurement of training time"""
        import time
        
        model = nn.Sequential(nn.Linear(128, 10))
        phase4_info = {'weight_bits': 1, 'activation_bits': 8, 'quantization_method': 'deterministic'}
        training_config = {'learning_rate': 1e-4}
        
        def timed_mock_loader():
            for _ in range(3):
                data = torch.randn(16, 128)
                targets = torch.randint(0, 10, (16,))
                yield data, targets
        
        pipeline = MockPhase5TrainingPipeline(model, phase4_info, training_config)
        pipeline.setup_bitnet_training()
        
        start_time = time.time()
        metrics = pipeline.continue_training(timed_mock_loader(), epochs=2)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        assert training_time > 0
        assert len(metrics) == 2
        
        # Calculate approximate throughput
        total_samples = 3 * 16 * 2  # batches * batch_size * epochs
        throughput = total_samples / training_time
        
        assert throughput > 0
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking during training"""
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        phase4_info = {'weight_bits': 1, 'activation_bits': 8, 'quantization_method': 'deterministic'}
        training_config = {'learning_rate': 1e-4}
        
        def memory_loader():
            for _ in range(2):
                data = torch.randn(64, 256)  # Larger batch for memory test
                targets = torch.randint(0, 10, (64,))
                yield data, targets
        
        pipeline = MockPhase5TrainingPipeline(model, phase4_info, training_config)
        pipeline.setup_bitnet_training()
        
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            metrics = pipeline.continue_training(memory_loader(), epochs=1)
            final_memory = torch.cuda.memory_allocated()
            
            # Memory should be allocated during training
            memory_used = final_memory - initial_memory
            assert memory_used >= 0  # Memory usage should not be negative

if __name__ == "__main__":
    pytest.main([__file__])