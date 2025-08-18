# distilled from lucidrains/grokfast-pytorch MIT impl  ➜  https://github.com/lucidrains/grokfast-pytorch

import torch
from torch.optim import Adam


class AugmentedAdam(Adam):
    def __init__(self, params, boost=1.5, beta_cut=0.08, **kw) -> None:
        super().__init__(params, **kw)
        self.boost, self.beta_cut = boost, beta_cut
        self._grad_window = None

    def step(self, closure=None, amplify=False):
        if amplify:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad.data
                    # window buffer
                    if self._grad_window is None:
                        self._grad_window = torch.zeros_like(g)
                    self._grad_window.mul_(1 - self.beta_cut).add_(self.beta_cut, g)
                    g_slow = self._grad_window
                    p.grad.data = g + self.boost * g_slow
        return super().step(closure)
