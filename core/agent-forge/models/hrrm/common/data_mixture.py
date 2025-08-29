# packages/hrrm/common/data_mixture.py
from __future__ import annotations

import torch

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def text_stream(batch_size=8, seq_len=256, limit_steps=1000):
    if load_dataset is None:
        # synthetic tokens fallback
        for _ in range(limit_steps):
            x = torch.randint(0, 32000, (batch_size, seq_len))
            y = x.clone()
            yield {"x_ids": x, "labels": y}
    else:
        ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)  # may be replaced by local shards
        iter(ds)
        for _ in range(limit_steps):
            # naive text to tokens stub (Claude: replace with tokenizer when built)
            x = torch.randint(0, 32000, (batch_size, seq_len))
            y = x.clone()
            yield {"x_ids": x, "labels": y}
