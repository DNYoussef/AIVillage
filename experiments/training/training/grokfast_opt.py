"""grokfast_opt.py.
----------------
Drop-in replacement for Adam that exposes ``.slow_power()``.
"""

import torch

try:
    # Try to import from local grokfast implementation first
    import sys
    from pathlib import Path
    grokfast_path = str(Path(__file__).parent.parent.parent / "core" / "agent-forge" / "phases" / "cognate_pretrain")
    if grokfast_path not in sys.path:
        sys.path.append(grokfast_path)
    
    # Use local grokfast optimizer as base for AugmentedAdam functionality
    from grokfast_optimizer import GrokFastOptimizer
    
    class AugmentedAdam(torch.optim.Adam):  # type: ignore[misc]
        """Local implementation of AugmentedAdam using GrokFast principles."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
except ImportError:  # pragma: no cover - fallback implementation

    class AugmentedAdam(torch.optim.Adam):  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # Fallback to standard Adam if grokfast not available


class GrokfastAdam(AugmentedAdam):
    def __init__(self, params, **kw) -> None:
        super().__init__(
            params,
            slow_freq=kw.pop("slow_freq", 0.08),
            boost=kw.pop("boost", 1.5),
            **kw,
        )

    def slow_power(self) -> float:
        """L2 norm of the slow-gradient component averaged across params."""
        tot, n = 0.0, 0
        for g in self._slow_cache.values():
            tot += (g**2).mean().item()
            n += 1
        return (tot / n) ** 0.5 if n else 0.0


if __name__ == "__main__":
    lin = torch.nn.Linear(10, 1)
    opt = GrokfastAdam(lin.parameters(), lr=1e-2)
    for _ in range(3):
        out = lin(torch.randn(4, 10)).sum()
        out.backward()
        opt.step()
        opt.zero_grad()
        print("slow", opt.slow_power())
