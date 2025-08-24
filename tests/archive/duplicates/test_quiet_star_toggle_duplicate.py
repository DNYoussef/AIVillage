import importlib.util

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_forge.training.quiet_star import QuietSTaRModel

spec = importlib.util.find_spec("torch")
if spec is None:
    pytest.skip("PyTorch not installed", allow_module_level=True)


def test_quiet_star_toggle_changes_output():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    qs_model = QuietSTaRModel(base_model)

    tokens = tokenizer.encode("Hello", return_tensors="pt")

    base_logits, _ = qs_model(tokens, generate_thoughts=False)
    qs_logits, _ = qs_model(tokens, generate_thoughts=True)

    assert not torch.allclose(base_logits, qs_logits)
