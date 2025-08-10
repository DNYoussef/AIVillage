from __future__ import annotations

import datetime as dt
import json
import logging
import pathlib
from datetime import timezone
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import AutoTokenizer

log = logging.getLogger("PromptBake")

ANCHOR_NS = "morality_v1"  # change when you update wording


def bake(
    model,
    tokenizer: AutoTokenizer,
    prompt_path: str | pathlib.Path,
    prefix_len: int = 32,
    lr: float = 2e-5,
    steps: int = 200,
):
    """Bake prompt into a prefix embedding matrix."""
    text = pathlib.Path(prompt_path).read_text()
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    class PrefixEncoder(torch.nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.embed = torch.nn.Parameter(torch.randn(prefix_len, hidden) * 0.02)

        def forward(self, bsz):
            return self.embed.expand(bsz, *self.embed.shape)

    pe = PrefixEncoder(model.config.hidden_size).to(model.device)
    optim = torch.optim.AdamW(pe.parameters(), lr=lr)

    model.eval()
    for step in range(steps):
        prefix = pe(1)
        out = model(inputs_embeds=prefix, labels=ids)
        loss = out.loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        if loss.item() < 0.05:
            break

    log.info("Prompt baked in %d steps, final loss %.4f", step, loss.item())

    pe.eval().requires_grad_(False)
    model.prompt_bank.register_buffer(f"{ANCHOR_NS}_emb", pe.embed)

    meta = {
        "ns": ANCHOR_NS,
        "txt": text[:160] + "…" if len(text) > 160 else text,
        "dt": dt.datetime.now(timezone.utc).isoformat(),
        "loss": float(loss.item()),
        "steps": step,
    }
    pathlib.Path("prompt_baking").mkdir(exist_ok=True)
    (pathlib.Path("prompt_baking") / f"{ANCHOR_NS}.json").write_text(
        json.dumps(meta, indent=2)
    )
    return meta
