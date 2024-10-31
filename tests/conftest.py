"""Global test configuration and fixtures."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock classes for tool_baking tests
class MockChatAgentConfig:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'test_agent')
        self.llm = kwargs.get('llm', None)

class MockTask:
    def __init__(self, agent):
        self.agent = agent
        self.init_state = agent.init_state
        self.state = self.init_state.copy()
        self.tools = []
        self.system_prompt = ""
        self.messages = []
        self.memory = {}

    async def run(self, *args, **kwargs):
        return "Task completed"

class MockChatAgent:
    def __init__(self, config=None):
        self.config = config or MockChatAgentConfig()
        self.init_state = {
            'system_prompt': '',
            'messages': [],
            'memory': {},
            'tools': []
        }
        self.state = self.init_state.copy()

class MockTensor:
    def __init__(self, data):
        self.data = data
        self.requires_grad = True
        self.grad_fn = Mock()

    def to(self, device):
        return self

    def view(self, *args):
        result = MockTensor(self.data)
        result.requires_grad = True
        result.grad_fn = Mock()
        return result

    def size(self, *args):
        return 100

    def backward(self, *args, **kwargs):
        pass

class MockLoss(MockTensor):
    def backward(self, *args, **kwargs):
        pass

class MockTokenizerOutput(dict):
    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self['input_ids'] = MockTensor(input_ids)
        self['attention_mask'] = MockTensor(attention_mask)

    def items(self):
        return super().items()

class MockTokenizer:
    def __init__(self):
        self.special_tokens = []
        self.vocab_size = 1000

    def __call__(self, text, **kwargs):
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        return MockTokenizerOutput(input_ids, attention_mask)

    def add_special_tokens(self, special_tokens_dict):
        self.special_tokens.extend(special_tokens_dict['additional_special_tokens'])

    def save_pretrained(self, path):
        pass

    def decode(self, token_ids, **kwargs):
        return "<start of thought>Test response<end of thought>"

    def __len__(self):
        return self.vocab_size

class MockAutoModel:
    @classmethod
    def from_pretrained(cls, model_name):
        model = Mock()
        model.to = Mock(return_value=model)
        model.config = Mock(hidden_size=768)
        
        def mock_generate(**kwargs):
            if kwargs.get('return_dict_in_generate', False):
                return Mock(
                    sequences=torch.tensor([[1, 2, 3]]),
                    hidden_states=[torch.randn(1, 1, 768) for _ in range(5)]
                )
            return torch.tensor([[1, 2, 3]])
        model.generate = Mock(side_effect=mock_generate)
        
        model.save_pretrained = Mock()
        model.resize_token_embeddings = Mock(return_value=model)
        
        def mock_call(**kwargs):
            return Mock(
                logits=MockTensor(torch.randn(1, 10, 100)),
                loss=MockLoss(torch.tensor(0.5)),
                hidden_states=[MockTensor(torch.randn(1, 1, 768)) for _ in range(5)],
                return_dict=True
            )
        model.__call__ = Mock(side_effect=mock_call)
        return model

class MockAutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_name):
        return MockTokenizer()

class MockOptimizer:
    def __init__(self, *args, **kwargs):
        self.step = Mock()
        self.zero_grad = Mock()

@pytest.fixture(scope="function")
def mock_tool_baking_dependencies():
    """Mock dependencies for tool_baking tests."""
    patches = [
        patch('transformers.AutoModelForCausalLM', MockAutoModel),
        patch('transformers.AutoTokenizer', MockAutoTokenizer),
        patch('langroid.Task', MockTask),
        patch('langroid.ChatAgent', MockChatAgent),
        patch('langroid.ChatAgentConfig', MockChatAgentConfig),
        patch('torch.optim.AdamW', MockOptimizer),
        patch('torch.nn.functional.cross_entropy', return_value=MockLoss(torch.tensor(0.5)))
    ]
    
    for p in patches:
        p.start()
    
    yield
    
    for p in patches:
        try:
            p.stop()
        except RuntimeError:
            pass  # Ignore if patch was already stopped

@pytest.fixture(scope="function")
def deep_baker(mock_tool_baking_dependencies):
    """Create DeepSystemBakerTask instance."""
    try:
        from agent_forge.bakedquietiot.deepbaking import DeepSystemBakerTask
        config = MockChatAgentConfig(name="test_agent")
        agent = MockChatAgent(config=config)
        baker = DeepSystemBakerTask(agent=agent, model_name="test-model", device="cpu")
        return baker
    except Exception as e:
        logger.error(f"Failed to create DeepSystemBakerTask: {str(e)}")
        pytest.fail(f"Failed to create DeepSystemBakerTask: {str(e)}")
