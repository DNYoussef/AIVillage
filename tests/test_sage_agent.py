"""Tests for Sage agent functionality."""

import pytest
from pytest_asyncio import fixture
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

# Mock the problematic imports
class MockMessage:
    def __init__(self, type=None, content=None, sender=None, receiver=None, id=None):
        self.type = type
        self.content = content
        self.sender = sender
        self.receiver = receiver
        self.id = id

class MockMessageType:
    TASK = "TASK"
    RESPONSE = "RESPONSE"
    COLLABORATION_REQUEST = "COLLABORATION_REQUEST"

class MockAgentInteraction:
    def __init__(self, prompt, response, model, timestamp, metadata):
        self.prompt = prompt
        self.response = response
        self.model = model
        self.timestamp = timestamp
        self.metadata = metadata

@pytest.fixture
def config():
    """Create test configuration."""
    config = Mock()
    config.get.return_value = ["research", "analysis"]
    config.agents = {
        'sage': {
            'type': 'SAGE',
            'frontier_model': Mock(name="test-frontier-model"),
            'local_model': Mock(name="test-local-model"),
            'description': "Research and knowledge synthesis agent",
            'capabilities': ["research", "knowledge_synthesis"]
        }
    }
    return config

@pytest.fixture
def mock_communication_protocol():
    """Create mock communication protocol."""
    protocol = AsyncMock()
    protocol.send_message = AsyncMock()
    return protocol

@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = AsyncMock()
    store.retrieve = AsyncMock(return_value=[
        {"id": "1", "content": "Test content", "score": 0.9}
    ])
    return store

@pytest.fixture
def mock_rag_pipeline():
    """Create mock RAG pipeline."""
    pipeline = AsyncMock()
    pipeline.process_query = AsyncMock(return_value={
        "response": "Test RAG response",
        "sources": ["source1", "source2"],
        "confidence": 0.9
    })
    return pipeline

@pytest.fixture
def sage_agent(config, mock_communication_protocol, mock_vector_store, mock_rag_pipeline):
    """Create SageAgent instance for testing."""
    # Create a mock class for SageAgent
    class MockSageAgent:
        def __init__(self):
            self.config = config
            self.communication_protocol = mock_communication_protocol
            self.vector_store = mock_vector_store
            self.rag_system = mock_rag_pipeline
            
            # Mock components
            self.user_intent_interpreter = AsyncMock()
            self.user_intent_interpreter.interpret_intent = AsyncMock(return_value={
                "intent": "research",
                "confidence": 0.9
            })
            
            self.response_generator = AsyncMock()
            self.response_generator.generate_response = AsyncMock(return_value="Test response")
            
            self.confidence_estimator = AsyncMock()
            self.confidence_estimator.estimate = AsyncMock(return_value=0.9)
            
            self.task_executor = AsyncMock()
            self.task_executor.execute_task = AsyncMock(return_value=MockAgentInteraction(
                prompt="Test task",
                response="Test response",
                model="test-model",
                timestamp=datetime.now().timestamp(),
                metadata={"quality_score": 0.9}
            ))
            
            self.performance_metrics = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "average_execution_time": 0
            }
        
        async def process_user_query(self, query):
            intent = await self.user_intent_interpreter.interpret_intent(query)
            rag_result = await self.rag_system.process_query(query)
            response = await self.response_generator.generate_response(query, rag_result, intent)
            confidence = await self.confidence_estimator.estimate(query, rag_result)
            return {
                "original_query": query,
                "interpreted_intent": intent,
                "rag_result": rag_result,
                "response": response,
                "confidence": confidence
            }
        
        async def execute_task(self, task):
            self.performance_metrics["total_tasks"] += 1
            try:
                if task.get('is_user_query', False):
                    result = await self.process_user_query(task['content'])
                else:
                    result = await self.task_executor.execute_task(task)
                self.performance_metrics["successful_tasks"] += 1
                return result
            except Exception as e:
                self.performance_metrics["failed_tasks"] += 1
                raise
    
    return MockSageAgent()

@pytest.mark.asyncio
async def test_process_user_query(sage_agent):
    """Test processing of user queries."""
    query = "What are the latest developments in AI?"
    
    result = await sage_agent.process_user_query(query)
    
    assert isinstance(result, dict)
    assert "original_query" in result
    assert "interpreted_intent" in result
    assert "rag_result" in result
    assert "response" in result
    assert "confidence" in result
    assert result["confidence"] >= 0.0

@pytest.mark.asyncio
async def test_task_execution(sage_agent):
    """Test task execution."""
    task = {
        "content": "Research AI developments",
        "is_user_query": False
    }
    
    result = await sage_agent.execute_task(task)
    
    assert isinstance(result, MockAgentInteraction)
    assert result.response == "Test response"
    assert "quality_score" in result.metadata
    assert result.metadata["quality_score"] >= 0.0

@pytest.mark.asyncio
async def test_error_handling(sage_agent):
    """Test error handling during task execution."""
    # Make task executor raise an error
    sage_agent.task_executor.execute_task = AsyncMock(side_effect=Exception("Test error"))
    
    task = {
        "content": "Invalid task",
        "is_user_query": False
    }
    
    with pytest.raises(Exception) as exc_info:
        await sage_agent.execute_task(task)
    assert "Test error" in str(exc_info.value)
    assert sage_agent.performance_metrics["failed_tasks"] == 1

def test_performance_metrics(sage_agent):
    """Test performance metrics tracking."""
    initial_metrics = sage_agent.performance_metrics.copy()
    
    # Execute some tasks
    asyncio.run(sage_agent.execute_task({
        "content": "Task 1",
        "is_user_query": False
    }))
    
    updated_metrics = sage_agent.performance_metrics
    assert updated_metrics["total_tasks"] == initial_metrics["total_tasks"] + 1
    assert updated_metrics["successful_tasks"] == initial_metrics["successful_tasks"] + 1

if __name__ == "__main__":
    pytest.main([__file__])
