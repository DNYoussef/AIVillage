import torch
from collections import deque
from typing import Dict, Optional, Literal

class GrokFast:
    def __init__(self, model: torch.nn.Module, method: Literal['MA', 'EMA'] = 'EMA', 
                 window_size: int = 100, lamb: float = 2.0, alpha: float = 0.98):
        self.model = model
        self.method = method
        self.window_size = window_size
        self.lamb = lamb
        self.alpha = alpha
        self.grads = None

    def filter_gradients(self):
        if self.method == 'MA':
            return self._filter_ma()
        else:
            return self._filter_ema()

    def _filter_ma(self):
        if self.grads is None:
            self.grads = {n: deque(maxlen=self.window_size) 
                          for n, p in self.model.named_parameters() if p.requires_grad}

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.grads[n].append(p.grad.data.detach())

                if len(self.grads[n]) == self.window_size:
                    avg = sum(self.grads[n]) / self.window_size
                    p.grad.data = p.grad.data + avg * self.lamb

    def _filter_ema(self):
        if self.grads is None:
            self.grads = {n: p.grad.data.detach() 
                          for n, p in self.model.named_parameters() if p.requires_grad}

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.grads[n] = self.grads[n] * self.alpha + p.grad.data.detach() * (1 - self.alpha)
                p.grad.data = p.grad.data + self.grads[n] * self.lamb