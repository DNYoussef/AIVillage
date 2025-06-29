"""Collection of lightweight ADAS techniques."""

from __future__ import annotations

from typing import Dict, Any
import random
import math
import torch
import transformers

from .adas import AgentTechnique

WIKI_SNIPPETS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming many industries.",
    "Large language models learn from vast amounts of text data.",
    "Neural networks can approximate complex functions.",
    "Open-source software accelerates research and collaboration.",
]


def prune_heads(model_path: str, work_dir, params: Dict[str, Any]) -> float:
    """Very small pruning heuristic used for tests."""

    threshold = params.get("norm_threshold", 0.10)
    lm = transformers.AutoModelForCausalLM.from_pretrained(model_path).eval().half()

    pruned = 0
    for module in lm.modules():
        if hasattr(module, "attn_drop_heads"):
            weight = module.c_attn.weight
            norms = weight.norm(dim=1)
            mask = norms < threshold * norms.mean()
            if mask.any():
                weight[mask] = 0
                pruned += int(mask.sum())

    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokens = tok.encode(
        " ".join(random.sample(WIKI_SNIPPETS, 5)), return_tensors="pt"
    )
    with torch.inference_mode():
        loss = lm(tokens, labels=tokens).loss
    ppl = math.exp(loss.item())
    return 1.0 / (1.0 + ppl)


TECH_POOL = [AgentTechnique("prune_heads", prune_heads)]

