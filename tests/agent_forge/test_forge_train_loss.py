# ruff: noqa: S101  # Use of assert detected - Expected in test files
from types import SimpleNamespace

import torch

from src.agent_forge.training.forge_train import ForgeTrainConfig, ForgeTrainer


def test_regression_loss():
    config = ForgeTrainConfig(task_type="regression")
    trainer = SimpleNamespace(config=config)
    outputs = torch.tensor([[0.5]])
    batch = {"labels": torch.tensor([[1.0]])}
    loss = ForgeTrainer.compute_task_loss(trainer, outputs, batch)
    assert torch.allclose(loss, torch.tensor(0.25))


def test_custom_loss_fn():
    def l1_loss(outputs, batch):
        return torch.nn.functional.l1_loss(outputs, batch["labels"])

    config = ForgeTrainConfig(task_type="regression", custom_loss_fn=l1_loss)
    trainer = SimpleNamespace(config=config)
    outputs = torch.tensor([[0.5]])
    batch = {"labels": torch.tensor([[1.0]])}
    loss = ForgeTrainer.compute_task_loss(trainer, outputs, batch)
    assert torch.allclose(loss, torch.tensor(0.5))
