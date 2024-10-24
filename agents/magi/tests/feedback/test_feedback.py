"""Unit tests for MAGI feedback system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from ....magi.feedback.analysis import (
    FeedbackAnalyzer,
    PerformanceMetrics,
    TechniqueMetrics,
    SystemMetrics
)
from ....magi.feedback.improvement import (
    ImprovementManager,
    ImprovementPlan,
    ImprovementResult
)
from ....magi.core.exceptions import ToolError

@pytest.fixture
def analyzer():
    """Create feedback analyzer instance."""
    return FeedbackAnalyzer()

@pytest.fixture
def improvement_manager(analyzer):
    """Create improvement manager instance."""
    return ImprovementManager(analyzer)

@pytest.fixture
def sample_technique_history():
    """Create sample technique execution history."""
    return [
        {
            'technique': 'chain_of_thought',
            'success': True,
            'execution_time': 1.0,
            'confidence': 0.8,
            'timestamp': datetime.now()
        },
        {
            'technique': 'tree_of_thoughts',
            'success': True,
            'execution_time': 2.0,
            'confidence': 0.9,
            'timestamp': datetime.now()
        }
    ]

@pytest.fixture
def sample_system_metrics():
    """Create sample system metrics."""
    return SystemMetrics(
        total_tasks=100,
        successful_tasks=85,
        failed_tasks=15,
        average_response_time=1.5,
        peak_memory_usage=1024,
        active_techniques=['chain_of_thought', 'tree_of_thoughts'],
        active_tools=['tool1', 'tool2']
    )

# Analysis Tests
@pytest.mark.asyncio
async def test_technique_analysis(analyzer, sample_technique_history):
    """Test analysis of technique performance."""
    # Record technique executions
    for execution in sample_technique_history:
        analyzer.technique_history[execution['technique']].append(execution)
    
    # Analyze specific technique
    metrics = analyzer.analyze_technique_performance('chain_of_thought')
    
    assert metrics.success_rate > 0
    assert metrics.average_execution_time > 0
    assert metrics.average_confidence > 0
    assert len(metrics.common_patterns) >= 0

@pytest.mark.asyncio
async def test_system_performance_analysis(analyzer, sample_system_metrics):
    """Test analysis of system performance."""
    analyzer.system_metrics.append(sample_system_metrics)
    
    metrics = analyzer.analyze_system_performance()
    
    assert metrics.total_tasks == 100
    assert metrics.successful_tasks == 85
    assert metrics.failed_tasks == 15
    assert metrics.average_response_time == 1.5
    assert len(metrics.active_techniques) == 2
    assert len(metrics.active_tools) == 2

@pytest.mark.asyncio
async def test_error_pattern_analysis(analyzer):
    """Test analysis of error patterns."""
    # Record some errors
    analyzer.error_patterns['validation_error'].extend([
        {
            'error': 'Invalid input',
            'context': {'input': 'test'},
            'severity': 'medium',
            'timestamp': datetime.now()
        },
        {
            'error': 'Invalid input',
            'context': {'input': 'test2'},
            'severity': 'medium',
            'timestamp': datetime.now()
        }
    ])
    
    analysis = analyzer.analyze_error_patterns()
    
    assert analysis['total_errors'] == 2
    assert 'validation_error' in analysis['error_categories']
    assert len(analysis['common_causes']) > 0
    assert len(analysis['recommendations']) > 0

@pytest.mark.asyncio
async def test_performance_trend_analysis(analyzer, sample_technique_history):
    """Test analysis of performance trends."""
    # Record executions over time
    for execution in sample_technique_history:
        analyzer.technique_history[execution['technique']].append(execution)
    
    # Add some older executions
    old_execution = sample_technique_history[0].copy()
    old_execution['timestamp'] = datetime.now() - timedelta(days=7)
    analyzer.technique_history[old_execution['technique']].append(old_execution)
    
    # Analyze trends
    trends = analyzer._analyze_performance_trends()
    
    assert 'short_term' in trends
    assert 'long_term' in trends
    assert trends['short_term']['confidence'] > 0

# Improvement Tests
@pytest.mark.asyncio
async def test_improvement_plan_generation(improvement_manager):
    """Test generation of improvement plans."""
    plans = await improvement_manager.generate_improvement_plans()
    
    assert len(plans) > 0
    for plan in plans:
        assert isinstance(plan, ImprovementPlan)
        assert plan.target in ['technique', 'tool', 'system']
        assert len(plan.improvements) > 0
        assert 1 <= plan.priority <= 5
        assert 0 <= plan.estimated_impact <= 1

@pytest.mark.asyncio
async def test_improvement_implementation(improvement_manager):
    """Test implementation of improvements."""
    plan = ImprovementPlan(
        target='technique',
        target_name='chain_of_thought',
        improvements=[
            {'type': 'parameter_tuning', 'details': {'temperature': 0.8}},
            {'type': 'prompt_enhancement', 'details': {'template': 'enhanced'}}
        ],
        priority=1,
        estimated_impact=0.8,
        dependencies=[],
        implementation_steps=[
            'Update parameters',
            'Enhance prompt template'
        ]
    )
    
    results = await improvement_manager.implement_improvements([plan])
    
    assert len(results) == 1
    assert isinstance(results[0], ImprovementResult)
    assert results[0].success
    assert len(results[0].changes_made) > 0

@pytest.mark.asyncio
async def test_improvement_validation(improvement_manager):
    """Test validation of improvements."""
    # Invalid plan (missing steps)
    invalid_plan = ImprovementPlan(
        target='technique',
        target_name='test',
        improvements=[],
        priority=1,
        estimated_impact=0.5,
        dependencies=[],
        implementation_steps=[]
    )
    
    with pytest.raises(ToolError):
        await improvement_manager.implement_improvements([invalid_plan])

@pytest.mark.asyncio
async def test_improvement_history(improvement_manager):
    """Test improvement history tracking."""
    # Implement some improvements
    plan = ImprovementPlan(
        target='system',
        target_name=None,
        improvements=[{'type': 'optimization', 'details': {}}],
        priority=1,
        estimated_impact=0.7,
        dependencies=[],
        implementation_steps=['Optimize system']
    )
    
    await improvement_manager.implement_improvements([plan])
    
    # Get history
    history = improvement_manager.get_improvement_history()
    assert len(history) > 0
    assert all(isinstance(result, ImprovementResult) for result in history)

@pytest.mark.asyncio
async def test_dependency_resolution(improvement_manager):
    """Test improvement dependency resolution."""
    # Create plans with dependencies
    plan1 = ImprovementPlan(
        target='tool',
        target_name='tool1',
        improvements=[{'type': 'basic', 'details': {}}],
        priority=1,
        estimated_impact=0.5,
        dependencies=[],
        implementation_steps=['Step 1']
    )
    
    plan2 = ImprovementPlan(
        target='tool',
        target_name='tool2',
        improvements=[{'type': 'advanced', 'details': {}}],
        priority=2,
        estimated_impact=0.7,
        dependencies=['tool1'],
        implementation_steps=['Step 2']
    )
    
    # Implement plans
    results = await improvement_manager.implement_improvements([plan2, plan1])
    
    # Verify order of implementation
    assert results[0].plan.target_name == 'tool1'
    assert results[1].plan.target_name == 'tool2'

@pytest.mark.asyncio
async def test_improvement_metrics(improvement_manager):
    """Test improvement metrics tracking."""
    plan = ImprovementPlan(
        target='technique',
        target_name='test_technique',
        improvements=[{'type': 'enhancement', 'details': {}}],
        priority=1,
        estimated_impact=0.6,
        dependencies=[],
        implementation_steps=['Enhance technique']
    )
    
    # Record initial metrics
    initial_metrics = {'performance': 0.7, 'reliability': 0.8}
    
    # Implement improvement
    results = await improvement_manager.implement_improvements(
        [plan],
        initial_metrics=initial_metrics
    )
    
    # Verify metrics tracking
    result = results[0]
    assert result.metrics_before == initial_metrics
    assert result.metrics_after != initial_metrics
    assert all(
        result.metrics_after[key] >= value
        for key, value in initial_metrics.items()
    )

@pytest.mark.asyncio
async def test_concurrent_improvements(improvement_manager):
    """Test concurrent improvement implementation."""
    # Create independent plans
    plans = [
        ImprovementPlan(
            target='tool',
            target_name=f'tool{i}',
            improvements=[{'type': 'basic', 'details': {}}],
            priority=1,
            estimated_impact=0.5,
            dependencies=[],
            implementation_steps=['Step 1']
        )
        for i in range(3)
    ]
    
    # Implement concurrently
    results = await improvement_manager.implement_improvements(plans)
    
    assert len(results) == 3
    assert all(result.success for result in results)
    assert len(set(r.plan.target_name for r in results)) == 3
