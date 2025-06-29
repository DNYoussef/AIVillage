import unittest
import importlib.util

if importlib.util.find_spec('torch') is None or importlib.util.find_spec('transformers') is None:
    raise unittest.SkipTest('Required dependency not installed')

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import importlib.machinery, importlib.util, pathlib, types, sys
lt_stub = types.SimpleNamespace(check=lambda txt: [])
language_tool_stub = types.ModuleType('language_tool_python')
language_tool_stub.LanguageTool = lambda *a, **k: lt_stub
sys.modules['language_tool_python'] = language_tool_stub
evaluator_spec = importlib.util.spec_from_file_location('evaluator', pathlib.Path('agent_forge/evaluation/evaluator.py'))
evaluator = importlib.util.module_from_spec(evaluator_spec)
evaluator_spec.loader.exec_module(evaluator)
evaluator.evaluate_thought_quality = lambda *a, **k: {'avg_coherence':0.0,'avg_relevance':0.0}

class TestSelfModelingEval(unittest.TestCase):
    def test_perplexity_computation(self):
        model_name = 'hf-internal-testing/tiny-random-BertModel'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        texts = ['hello world', 'testing evaluation']
        val_loader = [{'txt': tokenizer.encode(t, return_tensors='pt')} for t in texts]

        eval_data = []
        for batch in val_loader:
            txt = batch['txt']
            attn = torch.ones_like(txt)
            eval_data.append((txt, attn, txt))

        class Wrapper(torch.nn.Module):
            def __init__(self, m, tok):
                super().__init__()
                self.m = m
                self.tok = tok
            def forward(self, input_ids, attention_mask=None, labels=None):
                return self.m(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            def generate_thoughts(self, inputs, attention_mask):
                with torch.no_grad():
                    out = self.m.generate(inputs, attention_mask=attention_mask, max_length=inputs.size(1)+1)
                return self.tok.decode(out[0], skip_special_tokens=True)

        wrapped = Wrapper(model, tokenizer)
        metrics = evaluator.evaluate_model(wrapped, eval_data)
        self.assertIsInstance(metrics['perplexity'], float)
        self.assertGreater(metrics['perplexity'], 0.0)

if __name__ == '__main__':
    unittest.main()
