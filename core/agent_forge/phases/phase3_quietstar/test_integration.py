"""
Test suite for Quiet-STaR Integration Layer
"""

import pytest
import torch
import torch.nn as nn
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import json

from .integration import (
    QuietSTaRIntegration,
    IntegrationContract,
    CheckpointData,
    ValidationError,
    IntegrationError
)
from .quietstar import ThoughtConfig


class MockModel(nn.Module):
    """Mock model for testing"""

    def __init__(self, param_count=25_000_000):
        super().__init__()

        # Create layers to reach target parameter count
        hidden_size = int((param_count / 3) ** 0.5)  # Rough approximation
        self.embedding = nn.Embedding(1000, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1000)

        # Mock config
        self.config = Mock()
        self.config.hidden_size = hidden_size

    def forward(self, x):
        return self.output(self.linear2(self.linear1(self.embedding(x))))


@pytest.fixture
def integration_config():
    """Test configuration"""
    return ThoughtConfig(
        num_thoughts=2,
        thought_length=16,
        coherence_threshold=0.5,
        temperature=0.7
    )


@pytest.fixture
def temp_checkpoint_dir():
    """Temporary directory for checkpoints"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_evomerge_output():
    """Mock output from EvoMerge phase"""
    model = MockModel(25_000_000)

    return {
        'model': model,
        'phase_2_metrics': {
            'fitness': 0.85,
            'perplexity': 10.5,
            'generation': 50
        },
        'evolution_history': {
            'generations': 50,
            'fitness': 0.85,
            'technique': 'evolutionary'
        },
        'model_stats': {
            'parameters': 25_000_000,
            'layers': 10,
            'device': 'cpu'
        }
    }


@pytest.fixture
def integration_instance(integration_config, temp_checkpoint_dir):
    """Integration instance for testing"""
    return QuietSTaRIntegration(
        config=integration_config,
        checkpoint_dir=temp_checkpoint_dir,
        websocket_port=8766  # Different port to avoid conflicts
    )


class TestIntegrationContract:
    """Test integration contracts"""

    def test_contract_initialization(self):
        """Test contract initialization"""
        contract = IntegrationContract()

        # Check input requirements
        assert 'model' in contract.input_requirements
        assert 'phase_2_metrics' in contract.input_requirements
        assert 'evolution_history' in contract.input_requirements
        assert 'model_stats' in contract.input_requirements

        # Check output requirements
        assert 'enhanced_model' in contract.output_requirements
        assert 'thought_metrics' in contract.output_requirements
        assert 'performance_data' in contract.output_requirements
        assert 'integration_status' in contract.output_requirements


class TestQuietSTaRIntegration:
    """Test main integration class"""

    def test_initialization(self, integration_config, temp_checkpoint_dir):
        """Test integration initialization"""
        integration = QuietSTaRIntegration(
            config=integration_config,
            checkpoint_dir=temp_checkpoint_dir
        )

        assert integration.config == integration_config
        assert integration.checkpoint_dir == Path(temp_checkpoint_dir)
        assert integration.current_phase == "initialization"
        assert integration.progress == 0.0
        assert integration.error_recovery_attempts == 0

    def test_validate_contract_field_success(self, integration_instance):
        """Test successful field validation"""
        # Test required field present
        assert integration_instance._validate_contract_field(
            "test_value",
            "test_field",
            {'required': True, 'type': str}
        )

        # Test optional field missing
        assert integration_instance._validate_contract_field(
            None,
            "optional_field",
            {'required': False, 'type': str}
        )

        # Test dict with required keys
        data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        assert integration_instance._validate_contract_field(
            data,
            "dict_field",
            {'required': True, 'type': dict, 'required_keys': ['key1', 'key2']}
        )

    def test_validate_contract_field_failure(self, integration_instance):
        """Test field validation failures"""
        # Test required field missing
        assert not integration_instance._validate_contract_field(
            None,
            "required_field",
            {'required': True, 'type': str}
        )

        # Test wrong type
        assert not integration_instance._validate_contract_field(
            123,
            "string_field",
            {'required': True, 'type': str}
        )

        # Test missing required keys
        data = {'key1': 'value1'}
        assert not integration_instance._validate_contract_field(
            data,
            "dict_field",
            {'required': True, 'type': dict, 'required_keys': ['key1', 'key2']}
        )

    def test_validate_input_from_evomerge_success(self, integration_instance, mock_evomerge_output):
        """Test successful input validation"""
        assert integration_instance.validate_input_from_evomerge(mock_evomerge_output)

    def test_validate_input_from_evomerge_failure(self, integration_instance):
        """Test input validation failures"""
        # Missing required field
        invalid_output = {
            'phase_2_metrics': {'fitness': 0.8},
            'evolution_history': {'generations': 50, 'fitness': 0.8, 'technique': 'test'},
            'model_stats': {'parameters': 25000000, 'layers': 10, 'device': 'cpu'}
        }

        with pytest.raises(ValidationError):
            integration_instance.validate_input_from_evomerge(invalid_output)

    def test_parameter_increase_calculation(self, integration_instance):
        """Test parameter increase calculation"""
        # Create mock enhanced model
        enhanced_model = MockModel(30_000_000)
        enhanced_model.thought_generator = nn.Linear(100, 100)  # Add some params
        enhanced_model.attention_mixer = nn.Linear(50, 50)    # Add some params

        increase = integration_instance._calculate_parameter_increase(enhanced_model)
        assert isinstance(increase, float)
        assert increase >= 0.0

    def test_memory_overhead_estimation(self, integration_instance):
        """Test memory overhead estimation"""
        model = MockModel(25_000_000)
        overhead = integration_instance._estimate_memory_overhead(model)

        assert isinstance(overhead, dict)
        assert 'model_size_mb' in overhead
        assert 'thought_overhead_mb' in overhead
        assert 'attention_overhead_mb' in overhead
        assert 'total_overhead_mb' in overhead

    def test_quantization_compatibility_check(self, integration_instance):
        """Test quantization compatibility assessment"""
        model = MockModel(25_000_000)
        compatibility = integration_instance._check_quantization_compatibility(model)

        assert isinstance(compatibility, dict)
        assert 'int8_compatible' in compatibility
        assert 'int4_compatible' in compatibility
        assert 'bitnet_compatible' in compatibility
        assert 'dynamic_quantization' in compatibility

    def test_critical_layer_identification(self, integration_instance):
        """Test critical layer identification"""
        model = MockModel(25_000_000)

        # Add some thought-related layers
        model.thought_generator = nn.Linear(100, 100)
        model.attention_mixer = nn.MultiheadAttention(100, 8)
        model.lm_head = nn.Linear(100, 1000)

        critical_layers = integration_instance._identify_critical_layers(model)
        assert isinstance(critical_layers, list)

        # Should identify thought-related and output layers
        layer_names = '\n'.join(critical_layers)
        assert any('thought' in name or 'lm_head' in name for name in critical_layers)

    def test_compression_sensitivity_assessment(self, integration_instance):
        """Test compression sensitivity assessment"""
        model = MockModel(25_000_000)
        sensitivity = integration_instance._assess_compression_sensitivity(model)

        assert isinstance(sensitivity, dict)
        assert 'overall_sensitivity' in sensitivity
        assert 'thought_layer_sensitivity' in sensitivity
        assert 'attention_sensitivity' in sensitivity

    def test_compression_ratio_recommendation(self, integration_instance):
        """Test compression ratio recommendation"""
        # High quality metrics
        high_quality_metrics = {
            'coherence_score': 0.9,
            'reasoning_quality': 0.85
        }

        ratio = integration_instance._recommend_compression_ratio(high_quality_metrics)
        assert isinstance(ratio, dict)
        assert 'recommended_ratio' in ratio
        assert 'conservative_ratio' in ratio
        assert 'aggressive_ratio' in ratio

        # Should recommend higher compression for high quality
        assert ratio['recommended_ratio'] <= 0.5

        # Low quality metrics
        low_quality_metrics = {
            'coherence_score': 0.3,
            'reasoning_quality': 0.4
        }

        ratio_low = integration_instance._recommend_compression_ratio(low_quality_metrics)
        # Should recommend lower compression for low quality
        assert ratio_low['recommended_ratio'] >= 0.5

    @pytest.mark.asyncio
    async def test_checkpoint_save_load(self, integration_instance):
        """Test checkpoint saving and loading"""
        model = MockModel(25_000_000)
        metrics = {'accuracy': 0.85, 'loss': 0.15}
        config = {'test': 'value'}
        validation_results = {'input_validation': True}

        # Save checkpoint
        checkpoint_file = await integration_instance.save_checkpoint(
            phase="test_phase",
            model=model,
            metrics=metrics,
            config=config,
            validation_results=validation_results
        )

        assert Path(checkpoint_file).exists()

        # Load checkpoint
        loaded_data = await integration_instance.load_checkpoint(checkpoint_file)

        assert isinstance(loaded_data, CheckpointData)
        assert loaded_data.phase == "test_phase"
        assert loaded_data.metrics == metrics
        assert loaded_data.config == config
        assert loaded_data.validation_results == validation_results

    @pytest.mark.asyncio
    async def test_error_recovery(self, integration_instance):
        """Test error recovery mechanism"""
        # Create a mock checkpoint first
        model = MockModel(25_000_000)
        await integration_instance.save_checkpoint(
            phase="test_recovery",
            model=model,
            metrics={},
            config={},
            validation_results={}
        )

        # Test recovery
        test_error = Exception("Test error")
        context = {'phase': 'test_phase', 'progress': 0.5}

        recovery_result = await integration_instance.recover_from_error(test_error, context)
        assert recovery_result is True
        assert integration_instance.error_recovery_attempts == 1

    def test_prepare_output_for_bitnet(self, integration_instance):
        """Test output preparation for BitNet phase"""
        enhanced_model = MockModel(30_000_000)
        enhanced_model.thought_generator = nn.Linear(100, 100)
        enhanced_model.attention_mixer = nn.MultiheadAttention(100, 8)
        enhanced_model.integrator = Mock()

        thought_metrics = {
            'coherence_score': 0.8,
            'thought_diversity': 0.7,
            'reasoning_quality': 0.75,
            'generation_speed': 1.5,
            'memory_efficiency': 0.85
        }

        performance_data = {
            'baseline_perplexity': 10.0,
            'enhanced_perplexity': 8.5,
            'improvement_ratio': 1.18,
            'inference_time': 0.15,
            'memory_usage': 2.1
        }

        # Mock the architectural validation
        with patch('core.agent_forge.phases.phase3_quietstar.integration.ArchitecturalContract') as mock_contract:
            mock_contract.validate_integrator.return_value = True

            output = integration_instance.prepare_output_for_bitnet(
                enhanced_model=enhanced_model,
                thought_metrics=thought_metrics,
                performance_data=performance_data
            )

        # Verify output structure
        assert 'enhanced_model' in output
        assert 'thought_metrics' in output
        assert 'performance_data' in output
        assert 'integration_status' in output
        assert 'enhancement_verification' in output
        assert 'compression_readiness' in output

        # Verify integration status
        assert output['integration_status']['validation_passed']
        assert output['integration_status']['ready_for_compression']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])