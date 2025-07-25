"""Grokfast variant of Adam exposing ``slow_power``."""

import torch
from torch.optim import Adam


class GrokfastAdam(Adam):
    def __init__(self, params, **kw):
        self.slow_freq = kw.pop("slow_freq", 0.08)
        self.boost = kw.pop("boost", 1.5)
        super().__init__(params, **kw)
        self._slow_cache = {}
        self._step_count = 0

    def slow_power(self) -> float:
        tot, n = 0.0, 0
        for g in self._slow_cache.values():
            tot += (g**2).mean().item()
            n += 1
        return (tot / n) ** 0.5 if n else 0.0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if p in self._slow_cache:
                    self._slow_cache[p] = self._slow_cache[p] * (1 - self.slow_freq) + grad * self.slow_freq
                else:
                    self._slow_cache[p] = grad.clone()

        return super().step(closure)


if __name__ == "__main__":
    lin = torch.nn.Linear(10, 1)
    opt = GrokfastAdam(lin.parameters(), lr=1e-2)
    for _ in range(3):
        out = lin(torch.randn(4, 10)).sum()
        out.backward()
        opt.step()
        opt.zero_grad()
        print("slow", opt.slow_power())
