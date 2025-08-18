import torch
from torch.optim import Adam


class AugmentedAdam(Adam):
    """Adam variant with slow-gradient amplification."""

    def __init__(
        self,
        params,
        slow_freq: float = 0.1,
        boost_start: float = 1.0,
        boost_cap: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self.slow_freq = slow_freq
        self.boost_start = boost_start
        self.boost_cap = boost_cap
        self._grad_history = []

    def record_grad(self) -> None:
        grads = torch.cat(
            [
                p.grad.detach().flatten()
                for p in self.param_groups[0]["params"]
                if p.grad is not None
            ],
            dim=0,
        )
        self._grad_history.append(grads)
        if len(self._grad_history) > 1000:
            self._grad_history.pop(0)

    def step(self, filter_slow: bool = False, **kwargs) -> None:
        if filter_slow and len(self._grad_history) >= int(1 / self.slow_freq):
            window = self._grad_history[-int(1 / self.slow_freq) :]
            g_slow = torch.stack(window, dim=0).mean(0)
            lam = min(
                self.boost_cap, self.boost_start * (len(self._grad_history) / 1000)
            )
            offset = 0
            for p in self.param_groups[0]["params"]:
                if p.grad is None:
                    continue
                numel = p.grad.numel()
                p.grad.data.add_(
                    lam
                    * g_slow[offset : offset + numel].view_as(p.grad).to(p.grad.device)
                )
                offset += numel

        super().step(**kwargs)
