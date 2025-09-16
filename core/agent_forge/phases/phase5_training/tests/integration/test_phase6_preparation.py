"""
Integration tests for Phase 5 Training to Phase 6 Baking preparation
Tests for model export, state preparation, and handoff validation
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

# Mock Phase 6 Integration Components
class MockPhase6Requirements:
    """Mock Phase 6 baking requirements"""
    
    @staticmethod
    def get_required_model_format():
        """Get required model format for Phase 6"""
        return {
            'format': 'pytorch_state_dict',
            'precision': 'float32',
            'optimization_level': 'O2',
            'required_metadata': [
                'model_architecture',
                'training_metrics',
                'quantization_info',
                'performance_stats',
                'validation_results'
            ],
            'file_structure': {
                'model_state_dict': 'model.pt',
                'metadata': 'metadata.json',
                'training_history': 'training_history.json',
                'performance_benchmarks': 'benchmarks.json'
            }
        }
    
    @staticmethod
    def validate_model_readiness(model_package):
        """Validate if model is ready for Phase 6 baking"""
        requirements = MockPhase6Requirements.get_required_model_format()
        validation_results = {
            'is_ready': True,
            'missing_components': [],
            'warnings': [],
            'errors': []
        }
        
        # Check required files
        for component, filename in requirements['file_structure'].items():
            if component not in model_package:
                validation_results['missing_components'].append(component)
                validation_results['is_ready'] = False
        
        # Check required metadata
        if 'metadata' in model_package:
            metadata = model_package['metadata']
            for required_field in requirements['required_metadata']:
                if required_field not in metadata:
                    validation_results['missing_components'].append(f"metadata.{required_field}")
                    validation_results['is_ready'] = False
        
        # Check model state dict
        if 'model_state_dict' in model_package:
            state_dict = model_package['model_state_dict']
            if not isinstance(state_dict, dict):
                validation_results['errors'].append("Invalid model state dict format")
                validation_results['is_ready'] = False
            
            # Check for required layers
            layer_count = len([k for k in state_dict.keys() if 'weight' in k])
            if layer_count == 0:
                validation_results['errors'].append("No weight layers found in model")
                validation_results['is_ready'] = False
        
        return validation_results

class MockPhase5ExportManager:
    """Mock Phase 5 model export manager"""
    
    def __init__(self, model, training_metrics, config):
        self.model = model
        self.training_metrics = training_metrics
        self.config = config
        self.export_metadata = {}
    
    def prepare_for_phase6(self):
        """Prepare model for Phase 6 baking"""
        # Extract model state dict
        model_state_dict = self.model.state_dict()
        
        # Prepare metadata
        metadata = self._create_metadata()
        
        # Prepare training history
        training_history = self._create_training_history()
        
        # Prepare performance benchmarks
        benchmarks = self._create_performance_benchmarks()
        
        # Create model package
        model_package = {
            'model_state_dict': model_state_dict,
            'metadata': metadata,
            'training_history': training_history,
            'performance_benchmarks': benchmarks
        }
        
        return model_package
    
    def _create_metadata(self):
        """Create model metadata"""
        return {
            'model_architecture': self._get_model_architecture(),
            'training_metrics': self._get_final_metrics(),
            'quantization_info': self._get_quantization_info(),
            'performance_stats': self._get_performance_stats(),
            'validation_results': self._get_validation_results(),
            'export_timestamp': torch.cuda.Event(enable_timing=True).record() if torch.cuda.is_available() else 0,
            'model_size_mb': self._calculate_model_size(),
            'parameter_count': sum(p.numel() for p in self.model.parameters())
        }
    
    def _get_model_architecture(self):
        """Get model architecture description"""
        architecture = {
            'layers': [],
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = {
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
                
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    layer_info['input_dim'] = module.in_features
                    layer_info['output_dim'] = module.out_features
                
                architecture['layers'].append(layer_info)
        
        return architecture
    
    def _get_final_metrics(self):
        """Get final training metrics"""
        if not self.training_metrics:
            return {}
        
        final_metrics = self.training_metrics[-1] if self.training_metrics else {}
        
        # Calculate additional statistics
        if len(self.training_metrics) > 1:
            losses = [m.get('loss', 0) for m in self.training_metrics]
            final_metrics.update({
                'loss_reduction': losses[0] - losses[-1],
                'loss_reduction_pct': ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] > 0 else 0,
                'training_stability': np.std(losses[-10:]) if len(losses) >= 10 else 0
            })
        
        return final_metrics
    
    def _get_quantization_info(self):
        """Get quantization information"""
        quantization_info = {
            'weight_bits': getattr(self.config, 'weight_bits', 32),
            'activation_bits': getattr(self.config, 'activation_bits', 32),
            'quantization_method': getattr(self.config, 'quantization_method', 'none'),
            'sparsity_level': 0.0
        }
        
        # Calculate actual sparsity
        total_params = 0
        zero_params = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param.abs() < 1e-6).sum().item()
        
        if total_params > 0:
            quantization_info['sparsity_level'] = zero_params / total_params
        
        return quantization_info
    
    def _get_performance_stats(self):
        """Get performance statistics"""
        # Mock performance stats
        return {
            'inference_time_ms': np.random.uniform(1.0, 10.0),
            'throughput_samples_per_sec': np.random.uniform(100, 1000),
            'memory_usage_mb': self._calculate_model_size(),
            'gpu_utilization_pct': np.random.uniform(70, 95),
            'energy_efficiency_score': np.random.uniform(0.7, 0.95)
        }
    
    def _get_validation_results(self):
        """Get validation results"""
        return {
            'final_accuracy': np.random.uniform(0.85, 0.95),
            'validation_loss': np.random.uniform(0.1, 0.5),
            'f1_score': np.random.uniform(0.8, 0.9),
            'precision': np.random.uniform(0.8, 0.9),
            'recall': np.random.uniform(0.8, 0.9),
            'confusion_matrix': [[50, 5], [3, 42]]  # Mock confusion matrix
        }
    
    def _create_training_history(self):
        """Create detailed training history"""
        return {
            'metrics_per_epoch': self.training_metrics,
            'learning_rate_schedule': [m.get('learning_rate', 1e-4) for m in self.training_metrics],
            'gradient_norms': [np.random.uniform(0.1, 2.0) for _ in self.training_metrics],
            'training_time_per_epoch': [np.random.uniform(10, 60) for _ in self.training_metrics],
            'memory_usage_per_epoch': [np.random.uniform(100, 500) for _ in self.training_metrics]
        }
    
    def _create_performance_benchmarks(self):
        """Create performance benchmarks"""
        return {
            'training_speed': {
                'samples_per_second': np.random.uniform(1000, 5000),
                'batches_per_second': np.random.uniform(10, 50),
                'gpu_utilization': np.random.uniform(80, 95)
            },
            'memory_efficiency': {
                'peak_memory_mb': np.random.uniform(1000, 4000),
                'average_memory_mb': np.random.uniform(500, 2000),
                'memory_efficiency_score': np.random.uniform(0.7, 0.9)
            },
            'convergence_metrics': {
                'epochs_to_convergence': len(self.training_metrics),
                'final_loss': self.training_metrics[-1].get('loss', 0) if self.training_metrics else 0,
                'convergence_stability': np.random.uniform(0.8, 0.95)
            }
        }
    
    def _calculate_model_size(self):
        """Calculate model size in MB"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def save_package(self, output_dir):
        """Save model package to directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        package = self.prepare_for_phase6()
        
        # Save model state dict
        torch.save(package['model_state_dict'], output_path / 'model.pt')
        
        # Save metadata as JSON
        with open(output_path / 'metadata.json', 'w') as f:
            # Convert tensors to lists for JSON serialization
            metadata_json = self._convert_tensors_to_json(package['metadata'])
            json.dump(metadata_json, f, indent=2)
        
        # Save training history
        with open(output_path / 'training_history.json', 'w') as f:
            history_json = self._convert_tensors_to_json(package['training_history'])
            json.dump(history_json, f, indent=2)
        
        # Save benchmarks
        with open(output_path / 'benchmarks.json', 'w') as f:
            benchmarks_json = self._convert_tensors_to_json(package['performance_benchmarks'])
            json.dump(benchmarks_json, f, indent=2)
        
        return output_path
    
    def _convert_tensors_to_json(self, obj):
        """Convert tensors to JSON-serializable format"""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_tensors_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_to_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

# Test Cases
class TestPhase6Preparation:
    """Test Phase 6 preparation functionality"""
    
    def test_model_package_creation(self):
        """Test creation of model package for Phase 6"""
        # Create mock model and training data
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        training_metrics = [
            {'epoch': 0, 'loss': 1.0, 'accuracy': 0.3},
            {'epoch': 1, 'loss': 0.8, 'accuracy': 0.5},
            {'epoch': 2, 'loss': 0.6, 'accuracy': 0.7}
        ]
        
        config = type('Config', (), {
            'weight_bits': 1,
            'activation_bits': 8,
            'quantization_method': 'deterministic'
        })()
        
        # Create export manager
        export_manager = MockPhase5ExportManager(model, training_metrics, config)
        
        # Prepare package
        package = export_manager.prepare_for_phase6()
        
        # Validate package structure
        assert 'model_state_dict' in package
        assert 'metadata' in package
        assert 'training_history' in package
        assert 'performance_benchmarks' in package
        
        # Validate metadata content
        metadata = package['metadata']
        assert 'model_architecture' in metadata
        assert 'training_metrics' in metadata
        assert 'quantization_info' in metadata
        assert 'parameter_count' in metadata
    
    def test_metadata_completeness(self):
        """Test completeness of exported metadata"""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
        training_metrics = [{'epoch': 0, 'loss': 0.5}]
        config = type('Config', (), {})()
        
        export_manager = MockPhase5ExportManager(model, training_metrics, config)
        package = export_manager.prepare_for_phase6()
        
        metadata = package['metadata']
        
        # Check required fields
        required_fields = [
            'model_architecture',
            'training_metrics',
            'quantization_info',
            'performance_stats',
            'validation_results',
            'model_size_mb',
            'parameter_count'
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"
        
        # Validate architecture details
        architecture = metadata['model_architecture']
        assert 'layers' in architecture
        assert 'total_parameters' in architecture
        assert len(architecture['layers']) > 0
    
    def test_training_history_export(self):
        """Test export of training history"""
        model = nn.Linear(10, 1)
        training_metrics = [
            {'epoch': 0, 'loss': 1.0, 'learning_rate': 1e-3},
            {'epoch': 1, 'loss': 0.8, 'learning_rate': 8e-4},
            {'epoch': 2, 'loss': 0.6, 'learning_rate': 6e-4}
        ]
        config = type('Config', (), {})()
        
        export_manager = MockPhase5ExportManager(model, training_metrics, config)
        package = export_manager.prepare_for_phase6()
        
        history = package['training_history']
        
        assert 'metrics_per_epoch' in history
        assert 'learning_rate_schedule' in history
        assert 'gradient_norms' in history
        assert 'training_time_per_epoch' in history
        
        # Check that metrics match
        assert len(history['metrics_per_epoch']) == 3
        assert history['metrics_per_epoch'] == training_metrics
    
    def test_performance_benchmarks_export(self):
        """Test export of performance benchmarks"""
        model = nn.Linear(10, 1)
        training_metrics = [{'epoch': 0, 'loss': 0.5}]
        config = type('Config', (), {})()
        
        export_manager = MockPhase5ExportManager(model, training_metrics, config)
        package = export_manager.prepare_for_phase6()
        
        benchmarks = package['performance_benchmarks']
        
        assert 'training_speed' in benchmarks
        assert 'memory_efficiency' in benchmarks
        assert 'convergence_metrics' in benchmarks
        
        # Validate benchmark structure
        training_speed = benchmarks['training_speed']
        assert 'samples_per_second' in training_speed
        assert 'gpu_utilization' in training_speed
        
        memory_efficiency = benchmarks['memory_efficiency']
        assert 'peak_memory_mb' in memory_efficiency
        assert 'memory_efficiency_score' in memory_efficiency

class TestPhase6Compatibility:
    """Test compatibility with Phase 6 requirements"""
    
    def test_format_requirements_validation(self):
        """Test validation against Phase 6 format requirements"""
        requirements = MockPhase6Requirements.get_required_model_format()
        
        # Check that requirements are properly defined
        assert 'format' in requirements
        assert 'required_metadata' in requirements
        assert 'file_structure' in requirements
        
        # Validate required metadata fields
        required_metadata = requirements['required_metadata']
        expected_fields = [
            'model_architecture',
            'training_metrics',
            'quantization_info',
            'performance_stats',
            'validation_results'
        ]
        
        for field in expected_fields:
            assert field in required_metadata
    
    def test_model_readiness_validation_success(self):
        """Test successful model readiness validation"""
        # Create valid model package
        model_package = {
            'model_state_dict': {'layer.weight': torch.randn(10, 5), 'layer.bias': torch.randn(10)},
            'metadata': {
                'model_architecture': {'layers': []},
                'training_metrics': {'accuracy': 0.9},
                'quantization_info': {'weight_bits': 1},
                'performance_stats': {'inference_time_ms': 5.0},
                'validation_results': {'final_accuracy': 0.9}
            },
            'training_history': {'metrics_per_epoch': []},
            'performance_benchmarks': {'training_speed': {}}
        }
        
        validation_results = MockPhase6Requirements.validate_model_readiness(model_package)
        
        assert validation_results['is_ready'] == True
        assert len(validation_results['missing_components']) == 0
        assert len(validation_results['errors']) == 0
    
    def test_model_readiness_validation_failure(self):
        """Test model readiness validation failure"""
        # Create incomplete model package
        incomplete_package = {
            'model_state_dict': {'layer.weight': torch.randn(10, 5)},
            'metadata': {
                'model_architecture': {'layers': []},
                # Missing required metadata fields
            }
            # Missing training_history and performance_benchmarks
        }
        
        validation_results = MockPhase6Requirements.validate_model_readiness(incomplete_package)
        
        assert validation_results['is_ready'] == False
        assert len(validation_results['missing_components']) > 0
    
    def test_invalid_model_state_dict(self):
        """Test validation with invalid model state dict"""
        invalid_package = {
            'model_state_dict': "invalid_state_dict",  # Should be dict
            'metadata': {
                'model_architecture': {'layers': []},
                'training_metrics': {},
                'quantization_info': {},
                'performance_stats': {},
                'validation_results': {}
            },
            'training_history': {},
            'performance_benchmarks': {}
        }
        
        validation_results = MockPhase6Requirements.validate_model_readiness(invalid_package)
        
        assert validation_results['is_ready'] == False
        assert len(validation_results['errors']) > 0
        assert "Invalid model state dict format" in validation_results['errors']

class TestModelExportProcess:
    """Test model export process"""
    
    def test_model_save_and_load(self):
        """Test model save and load process"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create model and export manager
            model = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
            
            training_metrics = [
                {'epoch': 0, 'loss': 1.0, 'accuracy': 0.3},
                {'epoch': 1, 'loss': 0.7, 'accuracy': 0.6}
            ]
            
            config = type('Config', (), {
                'weight_bits': 1,
                'activation_bits': 8
            })()
            
            export_manager = MockPhase5ExportManager(model, training_metrics, config)
            
            # Save package
            output_path = export_manager.save_package(temp_dir)
            
            # Verify files exist
            assert (output_path / 'model.pt').exists()
            assert (output_path / 'metadata.json').exists()
            assert (output_path / 'training_history.json').exists()
            assert (output_path / 'benchmarks.json').exists()
            
            # Load and verify model
            loaded_state_dict = torch.load(output_path / 'model.pt')
            assert isinstance(loaded_state_dict, dict)
            assert len(loaded_state_dict) > 0
            
            # Load and verify metadata
            with open(output_path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            assert 'model_architecture' in metadata
            assert 'quantization_info' in metadata
    
    def test_json_serialization(self):
        """Test JSON serialization of tensors and numpy arrays"""
        model = nn.Linear(5, 1)
        training_metrics = [{'epoch': 0, 'loss': 0.5}]
        config = type('Config', (), {})()
        
        export_manager = MockPhase5ExportManager(model, training_metrics, config)
        
        # Test tensor conversion
        test_obj = {
            'tensor': torch.tensor([1.0, 2.0, 3.0]),
            'numpy_array': np.array([4.0, 5.0, 6.0]),
            'nested': {
                'tensor': torch.tensor([[1, 2], [3, 4]]),
                'list': [torch.tensor(7.0), torch.tensor(8.0)]
            }
        }
        
        converted = export_manager._convert_tensors_to_json(test_obj)
        
        # Verify conversion
        assert isinstance(converted['tensor'], list)
        assert isinstance(converted['numpy_array'], list)
        assert isinstance(converted['nested']['tensor'], list)
        assert isinstance(converted['nested']['list'][0], (int, float))
    
    def test_model_size_calculation(self):
        """Test model size calculation"""
        # Create models of different sizes
        small_model = nn.Linear(10, 5)
        large_model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.Linear(500, 100),
            nn.Linear(100, 10)
        )
        
        config = type('Config', (), {})()
        
        small_export = MockPhase5ExportManager(small_model, [], config)
        large_export = MockPhase5ExportManager(large_model, [], config)
        
        small_size = small_export._calculate_model_size()
        large_size = large_export._calculate_model_size()
        
        # Large model should be significantly bigger
        assert large_size > small_size
        assert small_size > 0
        assert large_size > 0

class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""
    
    def test_empty_training_metrics(self):
        """Test handling of empty training metrics"""
        model = nn.Linear(10, 1)
        training_metrics = []  # Empty metrics
        config = type('Config', (), {})()
        
        export_manager = MockPhase5ExportManager(model, training_metrics, config)
        package = export_manager.prepare_for_phase6()
        
        # Should handle gracefully
        assert 'metadata' in package
        assert 'training_history' in package
        
        # Metadata should have default values
        metadata = package['metadata']
        assert 'training_metrics' in metadata
    
    def test_model_without_parameters(self):
        """Test handling of model without parameters"""
        # Create model with no parameters
        model = nn.Sequential()  # Empty model
        training_metrics = [{'epoch': 0, 'loss': 0.5}]
        config = type('Config', (), {})()
        
        export_manager = MockPhase5ExportManager(model, training_metrics, config)
        package = export_manager.prepare_for_phase6()
        
        # Should handle gracefully
        metadata = package['metadata']
        assert metadata['parameter_count'] == 0
        assert metadata['model_size_mb'] == 0
    
    def test_large_model_handling(self):
        """Test handling of very large models"""
        # Create large model
        large_model = nn.Sequential(*[
            nn.Linear(1000, 1000) for _ in range(10)
        ])
        
        training_metrics = [{'epoch': 0, 'loss': 0.5}]
        config = type('Config', (), {})()
        
        export_manager = MockPhase5ExportManager(large_model, training_metrics, config)
        
        # Should handle without errors
        package = export_manager.prepare_for_phase6()
        
        metadata = package['metadata']
        assert metadata['parameter_count'] > 0
        assert metadata['model_size_mb'] > 0
        
        # Architecture should be properly captured
        architecture = metadata['model_architecture']
        assert len(architecture['layers']) > 0

if __name__ == "__main__":
    pytest.main([__file__])