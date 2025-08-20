import torch
from torch import nn


class TaskVector:
    def __init__(
        self,
        pretrained_model: nn.Module = None,
        finetuned_model: nn.Module = None,
        exclude_param_names_regex: list[str] | None = None,
        task_vector_param_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Initialize a TaskVector object.

        :param pretrained_model: The pretrained model
        :param finetuned_model: The finetuned model
        :param exclude_param_names_regex: List of regex patterns for parameter names to exclude
        :param task_vector_param_dict: A dictionary of task vector parameters
        """
        self.task_vector_param_dict = {}

        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        elif pretrained_model is not None and finetuned_model is not None:
            self._compute_task_vector(pretrained_model, finetuned_model, exclude_param_names_regex)

    def _compute_task_vector(
        self,
        pretrained_model: nn.Module,
        finetuned_model: nn.Module,
        exclude_param_names_regex: list[str],
    ) -> None:
        """Compute the task vector by subtracting pretrained model parameters from finetuned model parameters.

        :param pretrained_model: The pretrained model
        :param finetuned_model: The finetuned model
        :param exclude_param_names_regex: List of regex patterns for parameter names to exclude
        """
        import re

        for name, param in finetuned_model.named_parameters():
            if not any(re.match(regex, name) for regex in (exclude_param_names_regex or [])):
                self.task_vector_param_dict[name] = param.data - pretrained_model.state_dict()[name].data

    def combine_with_pretrained_model(
        self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0
    ) -> dict[str, torch.Tensor]:
        """Combine the task vector with a pretrained model.

        :param pretrained_model: The pretrained model to combine with
        :param scaling_coefficient: The scaling factor for the task vector
        :return: A dictionary of combined model parameters
        """
        combined_params = {}
        for name, param in pretrained_model.named_parameters():
            if name in self.task_vector_param_dict:
                combined_params[name] = param.data + scaling_coefficient * self.task_vector_param_dict[name]
            else:
                combined_params[name] = param.data
        return combined_params

    def __add__(self, other):
        """Add two TaskVector objects.

        :param other: Another TaskVector object
        :return: A new TaskVector object representing the sum
        """
        if not isinstance(other, TaskVector):
            msg = "Can only add TaskVector objects"
            raise TypeError(msg)

        new_task_vector_param_dict = {}
        for name in self.task_vector_param_dict:
            if name in other.task_vector_param_dict:
                new_task_vector_param_dict[name] = (
                    self.task_vector_param_dict[name] + other.task_vector_param_dict[name]
                )
            else:
                new_task_vector_param_dict[name] = self.task_vector_param_dict[name]

        for name in other.task_vector_param_dict:
            if name not in self.task_vector_param_dict:
                new_task_vector_param_dict[name] = other.task_vector_param_dict[name]

        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)
