# skeletal wrapper; copy full models from arXiv 2409.01633 repo ➜ Dreaming-is-All-You-Need
import torch, torch.nn as nn
from geometry.snapshot import geom_snapshot


class SleepDream:
    def __init__(self, model, sleep_net: nn.Module, dream_net: nn.Module):
        self.model, self.sleep, self.dream = model, sleep_net, dream_net

    @torch.no_grad()
    def run(self, hidden_seq):   # hidden_seq: T×B×D
        z = self.sleep(hidden_seq)
        delta = self.dream(z)
        id_before = geom_snapshot(hidden_seq.reshape(-1, hidden_seq.size(-1)))['ID_nl']
        # apply scaled delta
        for (p, d) in zip(self.model.parameters(), delta):
            p.add_(d * .2)
        id_after = geom_snapshot(hidden_seq.reshape(-1, hidden_seq.size(-1)))['ID_nl']
        return id_after < id_before   # accept flag
