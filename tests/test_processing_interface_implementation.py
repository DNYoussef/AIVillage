"""Comprehensive test suite for processing interface implementations.

This test suite validates the enhanced processing interface implementations,
including async processing workflows, error handling, progress tracking,
and integration with the federated AIVillage infrastructure.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta

# Import the implementations to test
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from processing_interface_implementation import (
    EnhancedProcessingInterface,
    TextProcessingInterface,
    ProcessingContext,
    ProgressTracker,
    ProcessorHealth,
    CodeQualityAnalyzer,
)

from agents.base import ProcessConfig, ProcessResult, ProcessStatus


class MockProcessingInterface(EnhancedProcessingInterface[str, str]):
    """Mock implementation for testing."""

    def __init__(self, processor_id: str = "test_processor", fail_initialization: bool = False):
        super().__init__(processor_id, ProcessConfig())
        self.fail_initialization = fail_initialization
        self.processing_calls = []
        self.validation_calls = []

    async def _initialize_processor(self) -> None:
        if self.fail_initialization:
            raise Exception("Initialization failed")
        await asyncio.sleep(0.01)  # Simulate initialization time

    async def _shutdown_processor(self) -> None:
        await asyncio.sleep(0.01)  # Simulate cleanup time

    async def _validate_input_implementation(self, input_data: str) -> bool:
        self.validation_calls.append(input_data)
        return isinstance(input_data, str) and len(input_data) > 0

    async def _process_implementation(
        self, input_data: str, context: ProcessingContext, progress_tracker: ProgressTracker, **kwargs
    ) -> str:
        self.processing_calls.append((input_data, context.request_id))

        # Simulate processing with progress updates
        progress_tracker.update_progress(25, "Processing step 1")
        await asyncio.sleep(0.01)

        progress_tracker.update_progress(50, "Processing step 2")
        await asyncio.sleep(0.01)

        progress_tracker.update_progress(75, "Processing step 3")
        await asyncio.sleep(0.01)

        progress_tracker.update_progress(90, "Finalizing")

        # Check for cancellation
        if progress_tracker.cancellation_token.is_set():
            raise asyncio.CancelledError("Processing cancelled")

        return f"Processed: {input_data}"


@pytest.fixture
def processor():
    """Create a mock processor for testing."""
    return MockProcessingInterface()


@pytest.fixture
def processing_context():
    """Create a processing context for testing."""
    return ProcessingContext(request_id="test_request_001")


@pytest.fixture
async def initialized_processor():
    """Create and initialize a processor."""
    proc = MockProcessingInterface()
    await proc.initialize()
    yield proc
    if proc.is_initialized:
        await proc.shutdown()


class TestEnhancedProcessingInterface:
    """Test the enhanced processing interface base class."""

    async def test_initialization_success(self, processor):
        """Test successful processor initialization."""
        assert not processor.is_initialized
        assert processor.health_status == ProcessorHealth.OFFLINE

        result = await processor.initialize()

        assert result is True
        assert processor.is_initialized
        assert processor.health_status == ProcessorHealth.HEALTHY
        assert not processor.circuit_open

    async def test_initialization_failure(self):
        """Test processor initialization failure."""
        processor = MockProcessingInterface(fail_initialization=True)

        result = await processor.initialize()

        assert result is False
        assert not processor.is_initialized
        assert processor.health_status == ProcessorHealth.CRITICAL

    async def test_shutdown_success(self, initialized_processor):
        """Test successful processor shutdown."""
        assert initialized_processor.is_initialized

        result = await initialized_processor.shutdown()

        assert result is True
        assert not initialized_processor.is_initialized
        assert initialized_processor.health_status == ProcessorHealth.OFFLINE
        assert initialized_processor.is_shutting_down

    async def test_processing_workflow(self, initialized_processor, processing_context):
        """Test complete processing workflow."""
        input_data = "test input data"

        result = await initialized_processor.process(input_data, processing_context)

        assert isinstance(result, ProcessResult)
        assert result.status == ProcessStatus.SUCCESS
        assert result.data == f"Processed: {input_data}"
        assert "processing_time" in result.metadata
        assert result.metadata["request_id"] == processing_context.request_id

        # Verify processing was tracked
        assert len(initialized_processor.processing_calls) == 1
        assert initialized_processor.processing_calls[0][0] == input_data
        assert initialized_processor.processing_calls[0][1] == processing_context.request_id

    async def test_processing_with_invalid_input(self, initialized_processor):
        """Test processing with invalid input."""
        # Empty string should fail validation
        context = ProcessingContext(request_id="invalid_test")

        result = await initialized_processor.process("", context)

        assert result.status == ProcessStatus.FAILED
        assert "Input validation failed" in result.error

    async def test_processing_without_initialization(self, processor):
        """Test processing fails when processor not initialized."""
        context = ProcessingContext(request_id="uninitialized_test")

        with pytest.raises(Exception) as exc_info:
            await processor.process("test", context)

        assert "not initialized" in str(exc_info.value)

    async def test_circuit_breaker_functionality(self, initialized_processor):
        """Test circuit breaker pattern."""
        # Force multiple failures to trigger circuit breaker
        for i in range(12):  # Exceed failure threshold
            initialized_processor._handle_failure()

        assert initialized_processor.circuit_open
        assert initialized_processor.health_status == ProcessorHealth.DEGRADED

        # Try processing with circuit open
        context = ProcessingContext(request_id="circuit_test")
        with pytest.raises(Exception) as exc_info:
            await initialized_processor.process("test", context)

        assert "Circuit breaker is open" in str(exc_info.value)

    async def test_progress_tracking(self, initialized_processor):
        """Test progress tracking functionality."""
        context = ProcessingContext(request_id="progress_test")

        # Start processing in background
        task = asyncio.create_task(initialized_processor.process("test progress", context))

        # Give it time to start
        await asyncio.sleep(0.005)

        # Check progress
        progress = await initialized_processor.get_progress("progress_test")
        assert progress is not None

        # Wait for completion
        result = await task
        assert result.status == ProcessStatus.SUCCESS

    async def test_task_cancellation(self, initialized_processor):
        """Test task cancellation functionality."""
        context = ProcessingContext(request_id="cancel_test")

        # Start processing
        task = asyncio.create_task(initialized_processor.process("test cancel", context))

        # Give it time to start
        await asyncio.sleep(0.005)

        # Cancel the task
        success = await initialized_processor.cancel_task("cancel_test")
        assert success

        # Wait for task to complete (should be cancelled)
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_health_check(self, initialized_processor):
        """Test health check functionality."""
        health = await initialized_processor.health_check()

        assert isinstance(health, dict)
        assert health["processor_id"] == initialized_processor.processor_id
        assert health["health_status"] == ProcessorHealth.HEALTHY.value
        assert health["is_initialized"] is True
        assert "active_tasks" in health
        assert "total_processed" in health
        assert "success_rate" in health

    async def test_metrics_tracking(self, initialized_processor):
        """Test metrics collection and tracking."""
        context1 = ProcessingContext(request_id="metrics_1")
        context2 = ProcessingContext(request_id="metrics_2")

        # Process some tasks
        await initialized_processor.process("test 1", context1)
        await initialized_processor.process("test 2", context2)

        # Check metrics
        assert initialized_processor.total_processed == 2
        assert initialized_processor.successful_processes == 2
        assert initialized_processor.failed_processes == 0
        assert initialized_processor.average_processing_time > 0
        assert initialized_processor.last_activity is not None

    async def test_concurrent_processing(self, initialized_processor):
        """Test concurrent processing capabilities."""
        contexts = [ProcessingContext(request_id=f"concurrent_{i}") for i in range(5)]

        # Process multiple tasks concurrently
        tasks = [initialized_processor.process(f"test {i}", contexts[i]) for i in range(5)]

        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert len(results) == 5
        assert all(r.status == ProcessStatus.SUCCESS for r in results)
        assert initialized_processor.total_processed == 5


class TestTextProcessingInterface:
    """Test the text processing interface implementation."""

    async def test_text_processing_initialization(self):
        """Test text processor initialization."""
        processor = TextProcessingInterface("text_proc", ProcessConfig())

        result = await processor.initialize()
        assert result is True

        await processor.shutdown()

    async def test_text_processing_workflow(self):
        """Test text processing workflow."""
        processor = TextProcessingInterface("text_proc", ProcessConfig())
        await processor.initialize()

        try:
            context = ProcessingContext(request_id="text_test")
            result = await processor.process("hello world", context)

            assert result.status == ProcessStatus.SUCCESS
            assert "HELLO WORLD" in result.data

        finally:
            await processor.shutdown()

    async def test_text_processing_estimation(self):
        """Test processing time estimation."""
        processor = TextProcessingInterface("text_proc", ProcessConfig())
        await processor.initialize()

        try:
            # Short text
            short_estimate = await processor.estimate_processing_time("short")

            # Long text
            long_text = "long " * 100
            long_estimate = await processor.estimate_processing_time(long_text)

            assert long_estimate > short_estimate

        finally:
            await processor.shutdown()


class TestProgressTracker:
    """Test progress tracking functionality."""

    def test_progress_tracker_initialization(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker("test_task", total_steps=100)

        assert tracker.task_id == "test_task"
        assert tracker.total_steps == 100
        assert tracker.current_step == 0
        assert tracker.progress_percentage == 0.0
        assert not tracker.cancellation_token.is_set()

    def test_progress_updates(self):
        """Test progress update functionality."""
        tracker = ProgressTracker("test_task", total_steps=100)

        tracker.update_progress(25, "Step 1 complete")
        assert tracker.current_step == 25
        assert tracker.progress_percentage == 25.0
        assert tracker.status_message == "Step 1 complete"

        tracker.update_progress(50, "Step 2 complete")
        assert tracker.current_step == 50
        assert tracker.progress_percentage == 50.0

    def test_progress_estimation(self):
        """Test progress estimation functionality."""
        tracker = ProgressTracker("test_task", total_steps=100)

        # Start tracking
        start_time = tracker.start_time

        # Simulate progress
        time.sleep(0.01)
        tracker.update_progress(25)

        assert tracker.elapsed_time > 0
        assert tracker.estimated_completion is not None
        assert tracker.estimated_completion > start_time


class TestCodeQualityAnalyzer:
    """Test code quality analysis functionality."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = CodeQualityAnalyzer()

        assert hasattr(analyzer, "logger")
        assert hasattr(analyzer, "analysis_results")

    def test_codebase_analysis(self):
        """Test comprehensive codebase analysis."""
        analyzer = CodeQualityAnalyzer()

        analysis = analyzer.analyze_codebase()

        assert isinstance(analysis, dict)
        assert "timestamp" in analysis
        assert "summary" in analysis
        assert "critical_issues" in analysis
        assert "code_smells" in analysis
        assert "refactoring_opportunities" in analysis
        assert "positive_findings" in analysis
        assert "recommendations" in analysis
        assert "technical_debt_estimate" in analysis

    def test_summary_generation(self):
        """Test analysis summary generation."""
        analyzer = CodeQualityAnalyzer()

        summary = analyzer._generate_summary()

        assert "overall_quality_score" in summary
        assert "files_analyzed" in summary
        assert "issues_found" in summary
        assert "technical_debt_hours" in summary
        assert isinstance(summary["overall_quality_score"], (int, float))
        assert summary["overall_quality_score"] >= 0

    def test_critical_issues_identification(self):
        """Test critical issues identification."""
        analyzer = CodeQualityAnalyzer()

        issues = analyzer._identify_critical_issues()

        assert isinstance(issues, list)
        for issue in issues:
            assert "type" in issue
            assert "description" in issue
            assert "severity" in issue
            assert "recommendation" in issue

    def test_code_smells_detection(self):
        """Test code smells detection."""
        analyzer = CodeQualityAnalyzer()

        smells = analyzer._detect_code_smells()

        assert isinstance(smells, list)
        for smell in smells:
            assert "type" in smell
            assert "description" in smell
            assert "severity" in smell

    def test_refactoring_opportunities(self):
        """Test refactoring opportunities identification."""
        analyzer = CodeQualityAnalyzer()

        opportunities = analyzer._identify_refactoring_opportunities()

        assert isinstance(opportunities, list)
        for opp in opportunities:
            assert "type" in opp
            assert "description" in opp
            assert "benefit" in opp
            assert "effort" in opp

    def test_technical_debt_estimation(self):
        """Test technical debt estimation."""
        analyzer = CodeQualityAnalyzer()

        debt = analyzer._estimate_technical_debt()

        assert isinstance(debt, dict)
        assert "total_hours" in debt
        assert "categories" in debt
        assert "priority_distribution" in debt
        assert isinstance(debt["total_hours"], (int, float))
        assert debt["total_hours"] >= 0


class TestProcessingContext:
    """Test processing context functionality."""

    def test_context_creation(self):
        """Test processing context creation."""
        context = ProcessingContext(request_id="test_123", user_id="user_456", priority=5, timeout_seconds=300.0)

        assert context.request_id == "test_123"
        assert context.user_id == "user_456"
        assert context.priority == 5
        assert context.timeout_seconds == 300.0
        assert context.retry_count == 0
        assert context.max_retries == 3
        assert isinstance(context.created_at, datetime)

    def test_context_with_metadata(self):
        """Test context with metadata and tags."""
        metadata = {"source": "test", "version": "1.0"}
        tags = {"urgent", "testing"}

        context = ProcessingContext(request_id="meta_test", metadata=metadata, tags=tags)

        assert context.metadata == metadata
        assert context.tags == tags

    def test_context_deadline(self):
        """Test context deadline functionality."""
        deadline = datetime.now() + timedelta(hours=1)
        context = ProcessingContext(request_id="deadline_test", deadline=deadline)

        assert context.deadline == deadline


@pytest.mark.asyncio
async def test_processing_interface_integration():
    """Integration test for processing interface with async workflows."""
    processor = TextProcessingInterface("integration_test", ProcessConfig())

    try:
        # Initialize
        init_success = await processor.initialize()
        assert init_success

        # Health check
        health = await processor.health_check()
        assert health["health_status"] == ProcessorHealth.HEALTHY.value

        # Process multiple items
        contexts = [ProcessingContext(request_id=f"integration_{i}") for i in range(3)]

        tasks = [processor.process(f"Integration test {i}", contexts[i]) for i in range(3)]

        results = await asyncio.gather(*tasks)

        # Verify results
        assert len(results) == 3
        assert all(r.status == ProcessStatus.SUCCESS for r in results)

        # Check final health
        final_health = await processor.health_check()
        assert final_health["total_processed"] == 3
        assert final_health["success_rate"] == 1.0

    finally:
        await processor.shutdown()


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_processing_interface_integration())
    print("All tests passed!")
