# packages/hrrm/common/training_utils.py
from __future__ import annotations

import random

from accelerate import Accelerator
import numpy as np
import torch


def set_seed(s=1337):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def simple_train_loop(model, loader, max_steps, lr=3e-4, wd=0.1, grad_accum=1):
    acc = Accelerator()
    (model,) = acc.prepare(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    step = 0
    loss_ema = None
    while step < max_steps:
        for batch in loader:
            with acc.accumulate(model):
                out = model(**batch)
                loss = out[0] if isinstance(out, tuple) else out
                acc.backward(loss)
                opt.step()
                opt.zero_grad()
                step += 1
                loss_ema = loss.item() if loss_ema is None else 0.9 * loss_ema + 0.1 * loss.item()
                if acc.is_local_main_process and (step % 50 == 0):
                    print(f"step {step} loss {loss_ema:.3f}")
                if step >= max_steps:
                    break
    if acc.is_local_main_process:
        print("done")
