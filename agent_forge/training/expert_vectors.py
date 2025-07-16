from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MoralArchetype:
    """Representation of a moral archetype used for expert vectors."""
    name: str
    focus: str
    question_template: str


@dataclass
class ExpertVector:
    """Stores singular value deltas for a model."""
    name: str
    singular_values: dict[str, torch.Tensor]


class ExpertVectorSystem:
    """Create and apply expert vectors using simple SVF."""

    def __init__(self, model: nn.Module):
        self.model = model

    def train_expert_vector_svf(self, name: str, scale: float = 0.05) -> ExpertVector:
        """Compute a basic expert vector by amplifying singular values."""
        deltas: dict[str, torch.Tensor] = {}
        for pname, param in self.model.named_parameters():
            if param.ndim < 2:
                continue
            u, s, v = torch.linalg.svd(param.data)
            delta = scale * s
            deltas[pname] = delta
        return ExpertVector(name=name, singular_values=deltas)

    def train_expert_vector_from_texts(
        self,
        name: str,
        texts: Iterable[str],
        *,
        epochs: int = 1,
        lr: float = 1e-3,
    ) -> ExpertVector:
        """Train the model on ``texts`` and capture the Σ-deltas."""
        # Snapshot singular values before training
        before: dict[str, torch.Tensor] = {}
        for pname, param in self.model.named_parameters():
            if param.ndim < 2:
                continue
            _, s, _ = torch.linalg.svd(param.data)
            before[pname] = s.clone()

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return ExpertVector(name=name, singular_values={})

        opt = torch.optim.SGD(params, lr=lr)

        # Determine input/output dims from first trainable param
        in_dim = params[0].shape[-1]
        out_dim = params[0].shape[0]

        def text_to_tensor(txt: str, dim: int) -> torch.Tensor:
            g = torch.Generator(device=params[0].device)
            g.manual_seed(abs(hash(txt)) % (2**32))
            return torch.randn(dim, generator=g, device=params[0].device)

        for _ in range(max(1, epochs)):
            for t in texts:
                x = text_to_tensor(t, in_dim).unsqueeze(0)
                target = text_to_tensor(t + "_tgt", out_dim).unsqueeze(0)
                opt.zero_grad()
                out = self.model(x)
                loss = F.mse_loss(out, target)
                loss.backward()
                opt.step()

        deltas: dict[str, torch.Tensor] = {}
        for pname, param in self.model.named_parameters():
            if param.ndim < 2 or pname not in before:
                continue
            _, s, _ = torch.linalg.svd(param.data)
            deltas[pname] = s - before[pname]

        return ExpertVector(name=name, singular_values=deltas)

    def apply_expert_vector(self, vector: ExpertVector, scaling: float = 1.0) -> None:
        """Apply an expert vector to the current model in-place."""
        for pname, param in self.model.named_parameters():
            if param.ndim < 2 or pname not in vector.singular_values:
                continue
            u, s, v = torch.linalg.svd(param.data)
            s = s + scaling * vector.singular_values[pname]
            param.data = (u @ torch.diag(s) @ v).to(param.device)

    def create_moral_experts(self, archetypes: list[MoralArchetype]) -> dict[str, ExpertVector]:
        """Create a simple expert vector for each archetype."""
        experts = {}
        for arch in archetypes:
            experts[arch.name] = self.train_expert_vector_svf(arch.name)
        return experts
