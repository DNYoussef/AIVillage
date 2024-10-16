import torch
from typing import Dict, List

def linear_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    weights = kwargs.get('weights', torch.ones(len(models)) / len(models))
    device = next(models[0].parameters()).device
    for key in models[0].state_dict().keys():
        tensors = []
        for model in models:
            tensor = model.state_dict()[key]
            if tensor.device.type == 'meta':
                tensor = torch.empty_like(tensor, device='cpu')
            tensors.append(tensor.to(device))
        merged_state_dict[key] = sum(w * t for w, t in zip(weights, tensors))
    return merged_state_dict

def slerp_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    t = kwargs.get("t", 0.5)
    device = next(models[0].parameters()).device
    for key in models[0].state_dict().keys():
        w1, w2 = models[0].state_dict()[key], models[1].state_dict()[key]
        if w1.device.type == 'meta':
            w1 = torch.empty_like(w1, device='cpu')
        if w2.device.type == 'meta':
            w2 = torch.empty_like(w2, device='cpu')
        w1, w2 = w1.to(device), w2.to(device)
        omega = torch.arccos(torch.clamp(torch.dot(w1.flatten(), w2.flatten()) / (w1.norm() * w2.norm()), -1, 1))
        so = torch.sin(omega)
        merged_state_dict[key] = (torch.sin((1.0-t)*omega) / so) * w1 + (torch.sin(t*omega) / so) * w2
    return merged_state_dict

def ties_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    threshold = kwargs.get("threshold", 0.1)
    device = next(models[0].parameters()).device
    for key in models[0].state_dict().keys():
        tensors = []
        for model in models:
            tensor = model.state_dict()[key]
            if tensor.device.type == 'meta':
                tensor = torch.empty_like(tensor, device='cpu')
            tensors.append(tensor.to(device))
        tensor = torch.stack(tensors)
        abs_tensor = torch.abs(tensor)
        mask = abs_tensor > threshold
        merged_state_dict[key] = torch.where(mask, tensor.mean(dim=0), torch.zeros_like(tensor[0]))
    return merged_state_dict

def dare_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    threshold = kwargs.get("threshold", 0.1)
    amplification = kwargs.get("amplification", 2.0)
    device = next(models[0].parameters()).device
    for key in models[0].state_dict().keys():
        tensors = []
        for model in models:
            tensor = model.state_dict()[key]
            if tensor.device.type == 'meta':
                tensor = torch.empty_like(tensor, device='cpu')
            tensors.append(tensor.to(device))
        tensor = torch.stack(tensors)
        abs_diff = torch.abs(tensor - tensor.mean(dim=0, keepdim=True))
        mask = abs_diff > threshold
        merged_state_dict[key] = torch.where(mask, tensor.mean(dim=0) * amplification, tensor.mean(dim=0))
    return merged_state_dict

def task_arithmetic_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    base_weights = kwargs.get("base_weights", {})
    task_weights = kwargs.get("task_weights", [])
    device = next(models[0].parameters()).device
    for key in base_weights:
        base_tensor = base_weights[key]
        if base_tensor.device.type == 'meta':
            base_tensor = torch.empty_like(base_tensor, device='cpu')
        base_tensor = base_tensor.to(device)
        task_vectors = []
        for task_weight in task_weights:
            task_tensor = task_weight[key]
            if task_tensor.device.type == 'meta':
                task_tensor = torch.empty_like(task_tensor, device='cpu')
            task_vectors.append(task_tensor.to(device) - base_tensor)
        combined_task_vector = sum(task_vectors)
        merged_state_dict[key] = base_tensor + combined_task_vector
    return merged_state_dict

def frankenmerge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    device = next(models[0].parameters()).device
    for i, (name, tensor) in enumerate(models[0].state_dict().items()):
        layer_num = int(name.split('.')[1]) if '.' in name else -1
        if layer_num == -1 or layer_num % len(models) == i:
            if tensor.device.type == 'meta':
                tensor = torch.empty_like(tensor, device='cpu')
            merged_state_dict[name] = tensor.to(device)
    return merged_state_dict

def dfs_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    I = kwargs.get("I", torch.ones(len(merged_state_dict)))
    W = kwargs.get("W", torch.eye(len(merged_state_dict), len(models)))
    device = next(models[0].parameters()).device
    layer_index = 0
    for name, tensor in models[0].state_dict().items():
        if any(layer_type in name for layer_type in ['layer', 'block', 'transformer']):
            if I[layer_index] > 0:
                tensors = []
                for model in models:
                    t = model.state_dict()[name]
                    if t.device.type == 'meta':
                        t = torch.empty_like(t, device='cpu')
                    tensors.append(t.to(device))
                merged_state_dict[name] = torch.sum(torch.stack(tensors) * W[layer_index].unsqueeze(1).unsqueeze(2), dim=0)
            layer_index += 1
        else:
            if tensor.device.type == 'meta':
                tensor = torch.empty_like(tensor, device='cpu')
            merged_state_dict[name] = tensor.to(device)
    return merged_state_dict

MERGE_TECHNIQUES = {
    "linear": linear_merge,
    "slerp": slerp_merge,
    "ties": ties_merge,
    "dare": dare_merge,
    "task_arithmetic": task_arithmetic_merge,
    "frankenmerge": frankenmerge,
    "dfs": dfs_merge
}
