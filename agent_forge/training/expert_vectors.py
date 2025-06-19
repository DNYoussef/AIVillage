from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn


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
    singular_values: Dict[str, torch.Tensor]


class ExpertVectorSystem:
    """Create and apply expert vectors using simple SVF."""

    def __init__(self, model: nn.Module):
        self.model = model

    def train_expert_vector_svf(self, name: str, scale: float = 0.05) -> ExpertVector:
        """Compute a basic expert vector by amplifying singular values."""
        deltas: Dict[str, torch.Tensor] = {}
        for pname, param in self.model.named_parameters():
            if param.ndim < 2:
                continue
            u, s, v = torch.linalg.svd(param.data)
            delta = scale * s
            deltas[pname] = delta
        return ExpertVector(name=name, singular_values=deltas)

    def apply_expert_vector(self, vector: ExpertVector, scaling: float = 1.0) -> None:
        """Apply an expert vector to the current model in-place."""
        for pname, param in self.model.named_parameters():
            if param.ndim < 2 or pname not in vector.singular_values:
                continue
            u, s, v = torch.linalg.svd(param.data)
            s = s + scaling * vector.singular_values[pname]
            param.data = (u @ torch.diag(s) @ v).to(param.device)

    def create_moral_experts(self, archetypes: List[MoralArchetype]) -> Dict[str, ExpertVector]:
        """Create a simple expert vector for each archetype."""
        experts = {}
        for arch in archetypes:
            experts[arch.name] = self.train_expert_vector_svf(arch.name)
        return experts
