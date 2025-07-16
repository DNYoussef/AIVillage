"""self_modeling_gate.py
---------------------
Self-model cycle; promotion is blocked until the grok signature
(slow_grad ↑ & ID_nl ↓) re-appears internally.
"""

from collections.abc import Sequence
import logging
import random

from geometry.snapshot import snapshot
import torch
from torch.nn import functional as F

logger = logging.getLogger("AF-SelfGrokk")


def self_model_cycle(model, tokenizer, tasks: Sequence[str], opt, thresholds, state):
    """Runs until internal grok OR max_iter.

    thresholds : dict = {slow, id_drop, chaos}
    state      : carries rule_id, prev_geom, etc.
    """
    τ, δ, ε = thresholds.values()
    hidden_pred = state["hidden_pred"]

    for step in range(thresholds.get("max_iter", 8_000)):
        text = random.choice(tasks)
        temp = random.uniform(0.0, 1.0)
        prompt = f"<geom id={state['G']['ID_nl']:.2f} t={temp:.2f}/>{text}"
        ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids

        # ----- forward  ------------------------------------------------------
        out = model(ids, output_hidden_states=True)
        H = out.hidden_states[-1]
        L_mask = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)), ids[:, 1:].reshape(-1)
        )
        L_pred = F.mse_loss(hidden_pred(H.detach()), H)
        loss = L_mask + 0.1 * L_pred
        opt.zero_grad()
        loss.backward()
        opt.step(filter=True)

        # ----- sense geometry  ----------------------------------------------
        G = snapshot(H)
        slow = opt.slow_power()
        drop = max(0, state["G"]["ID_nl"] - G["ID_nl"])
        chaos = state["complexity"](G, state["rule_id"])
        state["G"] = G

        if slow > τ and drop > δ and abs(chaos - 0.5) < ε:
            logger.info(
                ">> INTERNAL GROK at step %d  (slow=%.3f  drop=%.3f)", step, slow, drop
            )
            state["self_grok"] = True
            break


if __name__ == "__main__":

    class DummyPred(torch.nn.Module):
        def forward(self, x):
            return x

    model = torch.nn.Linear(5, 5)
    tok = lambda s, **k: type(
        "T", (), {"to": lambda self, d: self, "input_ids": torch.randint(0, 10, (1, 3))}
    )()
    opt = torch.optim.Adam(model.parameters())
    state = {
        "hidden_pred": DummyPred(),
        "G": {"ID_nl": 0.0},
        "complexity": lambda g, r: 0.5,
        "rule_id": 0,
    }
    self_model_cycle(
        model,
        tok,
        ["a"],
        opt,
        {"slow": 0.1, "id_drop": 0.1, "chaos": 0.1, "max_iter": 1},
        state,
    )
