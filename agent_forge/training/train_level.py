from .training_loop import run_level
from agent_forge.phase3 import self_model_cycle


def train_level(dataset, self_model_tasks, model, state):
    """Run one curriculum level with self-model grok gate."""
    run_level(dataset)
    self_model_cycle(model, self_model_tasks, state)
