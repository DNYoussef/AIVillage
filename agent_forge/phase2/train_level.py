from __future__ import annotations

from collections.abc import Sequence

import torch

from ..geometry import snapshot
from ..optim import Adam
from ..svf.svf_ops import apply_svf
from .pid import EdgePID


def run_level(model: torch.nn.Module, dataset: Sequence, config, state: dict) -> None:
    opt = Adam(model.parameters(), lr=config.lr)
    pid = EdgePID()
    geo2z = state.get("geo2z")
    for task in dataset:
        logits, H = model(task.prompt, return_h=True)
        G = snapshot(H)
        slow = opt.slow_power() if hasattr(opt, "slow_power") else 0.0
        z_geo = geo2z(torch.tensor([G[k] for k in ("ID_nl", "ratio", "entropy")], device=H.device))
        apply_svf(model, {"transformer.h.11.mlp": z_geo})
        if slow > config.tau and state.get("id_drop", 0) > config.delta and abs(task.score - 0.5) < config.eps:
            state["level_grok"] = True
            break
