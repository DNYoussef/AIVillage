"""Tests for tool baking system."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, patch, AsyncMock, MagicMock, create_autospec
import os
import shutil
import gc
from typing import Dict, Any, List
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock Task class that matches langroid.Task's interface
class MockTask:
    def __init__(self, agent):
        self.agent = agent
        self.init_state = agent.init_state if hasattr(agent, 'init_state') else {}
        self.state = self.init_state.copy()
        self.tools = []
        self.system_prompt = ""
        self.messages = []
        self.memory = {}

    async def run(self, *args, **kwargs):
        return "Task completed"

class MockChatAgent:
    def __init__(self, config=None):
        self.config = config
        self._init_state = {
            'system_prompt': '',
            'messages': [],
            'memory': {},
            'tools': []
        }
        self.state = self._init_state.copy()
        self.message_history = []
        self.interactive = True
        self.default_human_response = None
        self.name = "test_agent"

    def init_state(self):
        """Initialize agent state."""
        self.state = self._init_state.copy()
        return self.state

    def entity_responders(self):
        """Return list of entity responders."""
        return []

    def entity_responders_async(self):
        """Return list of async entity responders."""
        return []

    def set_system_message(self, message):
        """Set system message."""
        self.state['system_prompt'] = message

    def set_user_message(self, message):
        """Set user message."""
        if 'messages' not in self.state:
            self.state['messages'] = []
        self.state['messages'].append({"role": "user", "content": message})

# Mock PyTorch model for testing
class MockPyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.generate = create_autospec(self.generate)
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = torch.tensor([[1, 2, 3]])
        return Mock(
            logits=torch.randn(input_ids.shape[0], input_ids.shape[1], 100),
            loss=torch.tensor(0.5, requires_grad=True)
        )
        
    def generate(self, **kwargs):
        batch_size = kwargs.get('input_ids', torch.tensor([[1]])).shape[0]
        return torch.tensor([[1, 2, 3]] * batch_size)
        
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        
    def to(self, device):
        return self
        
    def resize_token_embeddings(self, new_size):
        return self

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    # Clean up PyTorch resources
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Clean up files
    if os.path.exists("deep_baked_model"):
        shutil.rmtree("deep_baked_model")

@pytest.fixture
def mock_agent():
    """Create mock ChatAgent."""
    return MockChatAgent(config=Mock())

@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.special_tokens = []
    
    # Create actual tensors for the tokenizer output
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    
    # Create a dictionary-like object that returns actual tensors
    class TokenizerOutput(dict):
        def __init__(self, input_ids, attention_mask):
            super().__init__()
            self['input_ids'] = input_ids
            self['attention_mask'] = attention_mask
            
        def __getitem__(self, key):
            return super().__getitem__(key)
    
    def mock_call(text, return_tensors=None, **kwargs):
        return TokenizerOutput(input_ids, attention_mask)
    
    tokenizer.__call__ = mock_call
    
    tokenizer.decode = Mock(return_value="<start of thought>Test response<end of thought>")
    tokenizer.add_special_tokens = Mock()
    tokenizer.save_pretrained = Mock()
    
    # Mock len() to return a reasonable vocabulary size
    tokenizer.__len__ = Mock(return_value=50000)
    
    return tokenizer

@pytest.fixture
def deep_baker(mock_agent, mock_tokenizer):
    """Create DeepSystemBakerTask instance."""
    from agent_forge.bakedquietiot.deepbaking import DeepSystemBakerTask
    
    class TestDeepSystemBakerTask(DeepSystemBakerTask):
        def __init__(self, agent, model_name, device):
            # Skip the parent class __init__ and set up our own mocked version
            super(DeepSystemBakerTask, self).__init__(agent, "", device)  # Pass empty string as model_name
            self.tokenizer = mock_tokenizer
            self.model = MockPyTorchModel()
            
        async def bake(self, prompt):
            tokenizer_output = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**tokenizer_output)
                # Create a dummy target tensor of the same shape as input_ids
                target = torch.zeros_like(tokenizer_output['input_ids'], dtype=torch.long)
                loss = F.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)), 
                    target.view(-1)
                )
            return loss
    
    with patch('langroid.Task', MockTask):
        baker = TestDeepSystemBakerTask(agent=mock_agent, model_name="", device="cpu")
        return baker

@pytest.mark.asyncio
async def test_special_tokens(deep_baker):
    """Test special tokens initialization."""
    expected_tokens = [
        "<start of thought>", "<end of thought>",
        "<initial thought>", "</initial thought>",
        "<refined thought>", "</refined thought>",
        "<alternative perspective>", "</alternative perspective>",
        "<key insight>", "</key insight>",
        "<memory recall>", "</memory recall>",
        "<hypothesis>", "</hypothesis>",
        "<evidence>", "</evidence>",
        "<confidence score>", "<continue thinking>",
        "<ready to answer>",
        "<analyze>", "</analyze>",
        "<plan>", "</plan>",
        "<execute>", "</execute>",
        "<evaluate>", "</evaluate>",
        "<revise>", "</revise>",
        "<systems_thinking>", "</systems_thinking>",
        "<first_principles>", "</first_principles>",
        "<cross_domain>", "</cross_domain>",
        "<probabilistic_thinking>", "</probabilistic_thinking>",
        "<rapid_iteration>", "</rapid_iteration>",
        "<paradox_resolution>", "</paradox_resolution>"
    ]
    
    assert all(token in deep_baker.special_tokens for token in expected_tokens)
    deep_baker.tokenizer.add_special_tokens.assert_called_once()

@pytest.mark.asyncio
async def test_deep_baking(deep_baker):
    """Test deep baking process."""
    # Mock evaluate_consistency to return increasing scores
    deep_baker.evaluate_consistency = AsyncMock(side_effect=[0.7, 0.85])
    
    # Mock optimizer
    mock_optimizer = Mock()
    mock_optimizer.step = Mock()
    mock_optimizer.zero_grad = Mock()
    
    with patch('torch.optim.AdamW', return_value=mock_optimizer):
        await deep_baker.deep_bake_system(max_iterations=2, consistency_threshold=0.8)
    
    # Verify model was saved
    assert os.path.exists("deep_baked_model")
    deep_baker.model.save_pretrained.assert_called_once_with("deep_baked_model")
    deep_baker.tokenizer.save_pretrained.assert_called_once_with("deep_baked_model")

@pytest.mark.asyncio
async def test_baking_step(deep_baker):
    """Test individual baking step."""
    test_prompt = "Test prompt"
    
    # Mock optimizer
    mock_optimizer = Mock()
    mock_optimizer.step = Mock()
    mock_optimizer.zero_grad = Mock()
    
    with patch('torch.optim.AdamW', return_value=mock_optimizer):
        await deep_baker.bake(test_prompt)
    
    # Verify optimizer steps
    mock_optimizer.step.assert_called_once()
    mock_optimizer.zero_grad.assert_called_once()

@pytest.mark.asyncio
async def test_consistency_evaluation(deep_baker):
    """Test consistency evaluation."""
    # Mock generate_response to return a response with special tokens
    deep_baker.generate_response = AsyncMock(return_value="""
    <start of thought>
    <initial thought>Test thought</initial thought>
    <analyze>Test analysis</analyze>
    <end of thought>
    """)
    
    consistency = await deep_baker.evaluate_consistency(num_samples=2)
    
    # Verify consistency score
    assert 0 <= consistency <= 1
    # Verify generate_response was called correct number of times
    assert deep_baker.generate_response.call_count == 2

@pytest.mark.asyncio
async def test_response_generation(deep_baker):
    """Test response generation."""
    # Mock model.generate to return a tensor
    deep_baker.model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
    
    response = await deep_baker.generate_response("Test prompt")
    
    # Verify response generation
    assert isinstance(response, str)
    assert "<start of thought>" in response
    assert "<end of thought>" in response
    
    # Verify model.generate was called
    deep_baker.model.generate.assert_called_once()

@pytest.mark.asyncio
async def test_response_scoring(deep_baker):
    """Test response scoring."""
    test_response = """
    <start of thought>
    <initial thought>Test thought</initial thought>
    <analyze>Test analysis</analyze>
    <plan>Test plan</plan>
    <execute>Test execution</execute>
    <evaluate>Test evaluation</evaluate>
    <end of thought>
    """
    
    score = await deep_baker.score_response(test_response)
    
    # Verify scoring
    assert 0 <= score <= 1
    # Score should be higher since tokens appear in correct order
    assert score > 0.1  # At least some tokens should be found in order

@pytest.mark.asyncio
async def test_task_run(deep_baker):
    """Test full task run."""
    # Mock evaluate_consistency to return increasing scores
    deep_baker.evaluate_consistency = AsyncMock(side_effect=[0.7, 0.85, 0.9])
    
    # Mock model.generate to return a tensor
    deep_baker.model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
    
    result = await deep_baker.run(max_iterations=3, consistency_threshold=0.8)
    
    # Verify result
    assert result == "Deep baking completed successfully"
    # Verify evaluate_consistency was called correct number of times
    assert deep_baker.evaluate_consistency.call_count == 2  # Should stop after reaching threshold at 0.85

if __name__ == "__main__":
    pytest.main([__file__])
