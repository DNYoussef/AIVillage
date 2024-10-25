"""Tests for core reasoning techniques."""

import pytest
from typing import Dict, Any
from agents.core.techniques.base import (
    BaseTechnique, TechniqueResult, TechniqueMetrics,
    ValidationError, TechniqueError
)
from agents.core.techniques.registry import TechniqueRegistry, TechniqueRegistryError
from datetime import datetime

# Mock classes
class MockTechnique(BaseTechnique):
    async def initialize(self):
        pass
    
    async def execute(self, input_data):
        return TechniqueResult(
            output=input_data,
            metrics=TechniqueMetrics(
                execution_time=0.1,
                success=True,
                confidence=0.8,
                uncertainty=0.2,
                timestamp=datetime.now(),
                additional_metrics=None
            ),
            intermediate_steps=[{"step": 1}],
            reasoning_trace=["Step 1"]
        )
    
    async def validate_input(self, input_data):
        if input_data is None:
            raise ValidationError("Input data cannot be None")
        return isinstance(input_data, dict)
    
    async def validate_output(self, output_data):
        return True

@pytest.fixture
def registry():
    """Create and initialize a technique registry."""
    registry = TechniqueRegistry()
    
    # Register mock techniques
    techniques = [
        ('multi_path_exploration', 'MultiPathExploration'),
        ('scale_aware_solving', 'ScaleAwareSolving'),
        ('perspective_shifting', 'PerspectiveShifting'),
        ('progressive_refinement', 'ProgressiveRefinement'),
        ('pattern_integration', 'PatternIntegration'),
        ('controlled_disruption', 'ControlledDisruption'),
        ('solution_unit_manipulation', 'SolutionUnitManipulation')
    ]
    
    for name, class_name in techniques:
        # Create a subclass of MockTechnique for each technique
        technique_class = type(class_name, (MockTechnique,), {})
        registry.register(technique_class, name=name)
    
    return registry

@pytest.mark.asyncio
async def test_technique_registration(registry):
    """Test that all core techniques are properly registered."""
    expected_techniques = {
        'multi_path_exploration',
        'scale_aware_solving',
        'perspective_shifting',
        'progressive_refinement',
        'pattern_integration',
        'controlled_disruption',
        'solution_unit_manipulation'
    }
    
    registered = {t.name for t in registry.list_techniques()}
    assert registered == expected_techniques

@pytest.mark.asyncio
async def test_technique_execution(registry):
    """Test that each technique can be executed."""
    test_input = {'test_input': 'data'}
    
    for technique_info in registry.list_techniques():
        technique = await registry.get_instance(technique_info.name)
        result = await technique.execute(test_input)
        
        assert result.metrics.success is True
        assert len(result.intermediate_steps) > 0
        assert len(result.reasoning_trace) > 0
        assert result.metrics.confidence >= 0
        assert result.metrics.uncertainty >= 0
        assert result.metrics.execution_time >= 0

@pytest.mark.asyncio
async def test_technique_composition(registry):
    """Test that techniques can be composed together."""
    test_input = {'test_input': 'data'}
    
    # Get two techniques
    mpe = await registry.get_instance('multi_path_exploration')
    sas = await registry.get_instance('scale_aware_solving')
    
    # Test sequential composition
    result1 = await mpe.execute(test_input)
    result2 = await sas.execute(result1.output)
    
    assert result1.metrics.success is True
    assert result2.metrics.success is True

@pytest.mark.asyncio
async def test_error_handling(registry):
    """Test that techniques handle errors gracefully."""
    for technique_info in registry.list_techniques():
        technique = await registry.get_instance(technique_info.name)
        
        # Test with invalid input
        with pytest.raises(ValidationError):
            await technique(None)
        
        # Test with empty input
        result = await technique({})
        assert result.metrics.success is not None

@pytest.mark.asyncio
async def test_registry_error_handling(registry):
    """Test registry error handling."""
    # Test registering duplicate technique
    with pytest.raises(TechniqueRegistryError):
        registry.register(MockTechnique, name='multi_path_exploration')
    
    # Test getting non-existent technique
    with pytest.raises(TechniqueRegistryError):
        await registry.get_instance('non_existent_technique')
    
    # Test unregistering non-existent technique
    with pytest.raises(TechniqueRegistryError):
        registry.unregister('non_existent_technique')

@pytest.fixture(autouse=True)
async def cleanup(registry):
    """Clean up after each test."""
    yield
    await registry.cleanup()
