import torch
from typing import Dict, Callable
from torch.nn.functional import cosine_similarity

def linear_merge(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    merged_weights = {}
    for key in weights:
        if 'weights' in kwargs:
            w = torch.tensor(kwargs['weights'])
        else:
            w = torch.ones(len(weights[key])) / len(weights[key])
        merged_weights[key] = torch.sum(weights[key] * w.unsqueeze(-1).unsqueeze(-1), dim=0)
    return merged_weights

def slerp_merge(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    merged_weights = {}
    for key in weights:
        t = kwargs.get("t", 0.5)
        w1, w2 = weights[key][0], weights[key][1]
        omega = torch.arccos(torch.clamp(cosine_similarity(w1.flatten(), w2.flatten(), dim=0), -1, 1))
        so = torch.sin(omega)
        merged_weights[key] = (torch.sin((1.0-t)*omega) / so).unsqueeze(-1).unsqueeze(-1) * w1 + \
                              (torch.sin(t*omega) / so).unsqueeze(-1).unsqueeze(-1) * w2
    return merged_weights

def ties_merge(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    threshold = kwargs.get("threshold", 0.1)
    merged_weights = {}
    for key in weights:
        tensor = weights[key]
        abs_tensor = torch.abs(tensor)
        mask = abs_tensor > threshold
        merged_weights[key] = torch.where(mask, tensor, torch.zeros_like(tensor))
    return merged_weights

def dare_merge(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    threshold = kwargs.get("threshold", 0.1)
    amplification = kwargs.get("amplification", 2.0)
    merged_weights = {}
    for key in weights:
        tensor = weights[key]
        abs_diff = torch.abs(tensor)
        mask = abs_diff > threshold
        merged_weights[key] = torch.where(mask, tensor * amplification, torch.zeros_like(tensor))
    return merged_weights

def task_arithmetic_merge(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    base_weights = kwargs.get("base_weights", {})
    task_weights = kwargs.get("task_weights", [])
    merged_weights = {}
    for key in base_weights:
        task_vectors = [task_weight[key] - base_weights[key] for task_weight in task_weights]
        combined_task_vector = sum(task_vectors)
        merged_weights[key] = base_weights[key] + combined_task_vector
    return merged_weights

def frankenmerge(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    models = kwargs.get("models", [])
    merged_weights = {}
    for i, (name, tensor) in enumerate(weights.items()):
        layer_num = int(name.split('.')[1]) if '.' in name else -1
        if layer_num == -1 or layer_num % len(models) == i:
            merged_weights[name] = tensor[i]
    return merged_weights

def dfs_merge(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    models = kwargs.get("models", [])
    I = kwargs.get("I", torch.ones(len(weights)))
    W = kwargs.get("W", torch.eye(len(weights), len(models)))
    merged_weights = {}
    layer_index = 0
    for name, tensor in weights.items():
        if any(layer_type in name for layer_type in ['layer', 'block', 'transformer']):
            if I[layer_index] > 0:
                merged_weights[name] = torch.sum(tensor * W[layer_index].unsqueeze(1).unsqueeze(2), dim=0)
            layer_index += 1
        else:
            merged_weights[name] = tensor[0]
    return merged_weights

def attention_flow_merge(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    models = kwargs.get("models", [])
    merged_weights = {}
    for name, tensor in weights.items():
        if 'attention' in name:
            # Combine attention weights using a weighted sum
            merged_weights[name] = torch.sum(tensor * torch.softmax(torch.randn(len(models)), dim=0).unsqueeze(-1).unsqueeze(-1), dim=0)
        else:
            # For non-attention weights, use a simple average
            merged_weights[name] = torch.mean(tensor, dim=0)
    return merged_weights

def layer_mixing_merge(weights: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
    models = kwargs.get("models", [])
    merged_weights = {}
    num_layers = max([int(name.split('.')[1]) for name in weights.keys() if '.' in name]) + 1
    layer_assignments = torch.randint(0, len(models), (num_layers,))
    for name, tensor in weights.items():
        if '.' in name:
            layer_num = int(name.split('.')[1])
            model_index = layer_assignments[layer_num]
            merged_weights[name] = tensor[model_index]
        else:
            # For global parameters, use an average
            merged_weights[name] = torch.mean(tensor, dim=0)
    return merged_weights

# Merge technique mapping
MERGE_TECHNIQUES: Dict[str, Callable] = {
    "linear": linear_merge,
    "slerp": slerp_merge,
    "ties": ties_merge,
    "dare": dare_merge,
    "task_arithmetic": task_arithmetic_merge,
    "frankenmerge": frankenmerge,
    "dfs": dfs_merge,
    "attention_flow": attention_flow_merge,
    "layer_mixing": layer_mixing_merge
}
