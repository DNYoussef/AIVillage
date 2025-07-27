from collections.abc import Iterable

from ..geometry.snapshot import snapshot
from ..meta.geo2z_policy import Geo2Z
from ..optim.augmented_adam import AugmentedAdam
import torch

from .pid_edgechaos import EdgePID
from .svf_ops import apply_svf

Batch = tuple[torch.Tensor, torch.Tensor, str]


def train_geometry_model(
    model: torch.nn.Module, dataset: Iterable[Batch], *, epochs: int = 1
) -> None:
    """Simple geometry-aware training loop for demonstration.

    Parameters
    ----------
    model : nn.Module
        Model with Linear layers to update via SVF.
    dataset : iterable of (input, target, tag)
        Dataset yielding mini-batches of tensors.
    epochs : int
        Number of passes over ``dataset``.
    """
    opt = AugmentedAdam(model.parameters(), lr=1e-3)
    pid = EdgePID()
    geo2z = Geo2Z()
    state = {"G_prev": {"ID_nl": 0.0}, "pre_grok": False}

    for _ in range(epochs):
        for x, y, _ in dataset:
            opt.zero_grad()
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, y)
            loss.backward()
            G = snapshot(out.detach())
            geom_vec = torch.tensor(
                [G[k] for k in ["ID_nl", "ID_lin", "ratio", "entropy"]]
            )
            z = geo2z(geom_vec)
            for m in model.modules():
                if isinstance(m, torch.nn.Linear):
                    apply_svf(m, z)
            lr_gain = pid.update(G["ratio"])
            for group in opt.param_groups:
                group["lr"] *= 1 + lr_gain
            opt.step(amplify=state["pre_grok"])
            state["pre_grok"] = G["ratio"] < 0.1
            state["G_prev"] = G
