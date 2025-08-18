import torch
from agent_forge.phase3 import self_model_cycle

from .expert_vectors import ExpertVectorSystem
from .training_loop import run_level


def train_level(dataset, self_model_tasks, model, state, *, expert_vector_path: str | None = None) -> None:
    """Run one curriculum level with self-model grok gate."""
    run_level(dataset)
    self_model_cycle(model, self_model_tasks, state)
    if expert_vector_path:
        system = ExpertVectorSystem(model)
        vector = system.train_expert_vector_svf("level_vector")
        torch.save({"name": vector.name, "values": vector.singular_values}, expert_vector_path)
