import torch
from typing import Dict, List
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

def linear_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    weights = kwargs.get('weights', torch.ones(len(models)) / len(models))
    device = next(models[0].parameters()).device
    chunk_size = kwargs.get('chunk_size', 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for key in models[0].state_dict().keys():
            merged_param = None

            # Check for meta tensors before chunking
            meta_tensors = [model.state_dict()[key] for model in models if model.state_dict()[key].device.type == 'meta']
            if not meta_tensors:
                # If no meta tensors, proceed with chunked merging
                for i in range(0, models[0].state_dict()[key].numel(), chunk_size):
                    chunk_params = []
                    for model in models:
                        chunk_param = model.state_dict()[key].flatten()[i:i+chunk_size].to(device)
                        if chunk_param.device.type == 'meta':
                            chunk_param = torch.empty_like(chunk_param, device='cpu')
                        chunk_params.append(chunk_param)

                    merged_chunk = sum(w * t for w, t in zip(weights, chunk_params))
                    merged_chunk_path = os.path.join(tmpdirname, f"{key}_{i}.pt")
                    torch.save(merged_chunk.cpu(), merged_chunk_path)
                    del merged_chunk
                    del chunk_params

                merged_chunks = []
                for i in range(0, models[0].state_dict()[key].numel(), chunk_size):
                    merged_chunk_path = os.path.join(tmpdirname, f"{key}_{i}.pt")
                    merged_chunk = torch.load(merged_chunk_path, map_location=device)
                    merged_chunks.append(merged_chunk)
                merged_param = torch.cat(merged_chunks)
                del merged_chunks
            else:
                # If any meta tensors are found, create a full-sized empty tensor on CPU
                merged_param = torch.empty_like(meta_tensors[0], device='cpu')

            merged_state_dict[key] = merged_param.reshape(models[0].state_dict()[key].shape)

    return merged_state_dict

def slerp_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    t = kwargs.get("t", 0.5)
    device = next(models[0].parameters()).device
    chunk_size = kwargs.get('chunk_size', 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for key in models[0].state_dict().keys():
            merged_param = None

            # Check for meta tensors before chunking
            meta_tensors = [model.state_dict()[key] for model in models if model.state_dict()[key].device.type == 'meta']
            if not meta_tensors:
                # If no meta tensors, proceed with chunked merging
                for i in range(0, models[0].state_dict()[key].numel(), chunk_size):
                    w1 = models[0].state_dict()[key].flatten()[i:i+chunk_size].to(device)
                    w2 = models[1].state_dict()[key].flatten()[i:i+chunk_size].to(device)
                    if w1.device.type == 'meta':
                        w1 = torch.empty_like(w1, device='cpu')
                    if w2.device.type == 'meta':
                        w2 = torch.empty_like(w2, device='cpu')

                    omega = torch.arccos(torch.clamp(torch.dot(w1, w2) / (w1.norm() * w2.norm()), -1, 1))
                    so = torch.sin(omega)
                    merged_chunk = (torch.sin((1.0 - t) * omega) / so) * w1 + (torch.sin(t * omega) / so) * w2
                    merged_chunk_path = os.path.join(tmpdirname, f"{key}_{i}.pt")
                    torch.save(merged_chunk.cpu(), merged_chunk_path)
                    del merged_chunk
                    del w1
                    del w2

                merged_chunks = []
                for i in range(0, models[0].state_dict()[key].numel(), chunk_size):
                    merged_chunk_path = os.path.join(tmpdirname, f"{key}_{i}.pt")
                    merged_chunk = torch.load(merged_chunk_path, map_location=device)
                    merged_chunks.append(merged_chunk)
                merged_param = torch.cat(merged_chunks)
                del merged_chunks
            else:
                # If any meta tensors are found, create a full-sized empty tensor on CPU
                merged_param = torch.empty_like(meta_tensors[0], device='cpu')

            merged_state_dict[key] = merged_param.reshape(models[0].state_dict()[key].shape)

    return merged_state_dict

def disk_based_ties_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    threshold = kwargs.get("threshold", 0.1)
    device = torch.device('cpu')  # Use CPU to handle larger tensors without GPU memory issues
    chunk_size = kwargs.get('chunk_size', 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for key in models[0].state_dict().keys():
            merged_param = None

            # Check for meta tensors before chunking
            meta_tensors = [model.state_dict()[key] for model in models if model.state_dict()[key].is_meta]
            if not meta_tensors:
                numel = models[0].state_dict()[key].numel()
                shape = models[0].state_dict()[key].shape
                for i in range(0, numel, chunk_size):
                    chunks = []
                    for model in models:
                        chunk = model.state_dict()[key].flatten()[i:i+chunk_size].to(device)
                        chunks.append(chunk)

                    stacked_chunks = torch.stack(chunks)
                    abs_chunks = stacked_chunks.abs()
                    mean_abs = abs_chunks.mean(dim=0)
                    mask = mean_abs > threshold

                    # Apply mask element-wise without changing shape
                    masked_chunks = stacked_chunks * mask

                    # Compute the average of the masked chunks
                    sum_mask = mask.sum()
                    if sum_mask.item() == 0:
                        merged_chunk = torch.zeros_like(masked_chunks[0])
                    else:
                        merged_chunk = masked_chunks.sum(dim=0) / sum_mask

                    merged_chunk_path = os.path.join(tmpdirname, f"{key}_{i}.pt")
                    torch.save(merged_chunk.cpu(), merged_chunk_path)
                    del merged_chunk
                    del chunks
                    del stacked_chunks
                    del abs_chunks
                    del mean_abs
                    del mask
                    del masked_chunks

                merged_chunks = []
                for i in range(0, numel, chunk_size):
                    merged_chunk_path = os.path.join(tmpdirname, f"{key}_{i}.pt")
                    merged_chunk = torch.load(merged_chunk_path, map_location=device)
                    merged_chunks.append(merged_chunk)
                merged_param = torch.cat(merged_chunks)
                del merged_chunks
            else:
                # If any meta tensors are found, create a full-sized empty tensor on CPU
                merged_param = torch.empty_like(meta_tensors[0], device='cpu')

            merged_state_dict[key] = merged_param.view(shape)

    return merged_state_dict

def dare_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    threshold = kwargs.get("threshold", 0.1)
    amplification = kwargs.get("amplification", 2.0)
    device = next(models[0].parameters()).device
    chunk_size = kwargs.get('chunk_size', 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for key in models[0].state_dict().keys():
            merged_param = None

            # Check for meta tensors before chunking
            meta_tensors = [model.state_dict()[key] for model in models if model.state_dict()[key].device.type == 'meta']
            if not meta_tensors:
                # If no meta tensors, proceed with chunked merging
                numel = models[0].state_dict()[key].numel()
                shape = models[0].state_dict()[key].shape
                for i in range(0, numel, chunk_size):
                    chunks = []
                    for model in models:
                        chunk = model.state_dict()[key].flatten()[i:i+chunk_size].to(device)
                        if chunk.device.type == 'meta':
                            chunk = torch.empty_like(chunk, device='cpu')
                        chunks.append(chunk)

                    chunk_tensor = torch.stack(chunks)
                    mean_chunk = chunk_tensor.mean(dim=0)
                    abs_diff = torch.abs(chunk_tensor - mean_chunk)
                    mask = abs_diff > threshold
                    mask = mask.to(device)

                    # Apply mask element-wise without changing shape
                    amplified_diff = (chunk_tensor - mean_chunk) * amplification * mask
                    merged_chunk = mean_chunk + amplified_diff.mean(dim=0)
                    merged_chunk_path = os.path.join(tmpdirname, f"{key}_{i}.pt")
                    torch.save(merged_chunk.cpu(), merged_chunk_path)
                    del merged_chunk
                    del chunks
                    del chunk_tensor
                    del mean_chunk
                    del abs_diff
                    del mask
                    del amplified_diff

                merged_chunks = []
                for i in range(0, numel, chunk_size):
                    merged_chunk_path = os.path.join(tmpdirname, f"{key}_{i}.pt")
                    merged_chunk = torch.load(merged_chunk_path, map_location=device)
                    merged_chunks.append(merged_chunk)
                merged_param = torch.cat(merged_chunks)
                del merged_chunks
            else:
                # If any meta tensors are found, create a full-sized empty tensor on CPU
                merged_param = torch.empty_like(meta_tensors[0], device='cpu')

            merged_state_dict[key] = merged_param.reshape(shape)

    return merged_state_dict

def task_arithmetic_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    base_weights = kwargs.get("base_weights", {})
    task_weights = kwargs.get("task_weights", [])
    device = next(models[0].parameters()).device
    chunk_size = kwargs.get('chunk_size', 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for key in base_weights:
            merged_param = None

            # Check for meta tensors before chunking
            meta_tensors = [base_weights[key]] + [task_weight[key] for task_weight in task_weights if task_weight[key].device.type == 'meta']
            if not meta_tensors:
                # If no meta tensors, proceed with chunked merging
                numel = base_weights[key].numel()
                shape = base_weights[key].shape
                for i in range(0, numel, chunk_size):
                    base_chunk = base_weights[key].flatten()[i:i+chunk_size].to(device)
                    if base_chunk.device.type == 'meta':
                        base_chunk = torch.empty_like(base_chunk, device='cpu')

                    task_chunks = []
                    for task_weight in task_weights:
                        task_chunk = task_weight[key].flatten()[i:i+chunk_size].to(device)
                        if task_chunk.device.type == 'meta':
                            task_chunk = torch.empty_like(task_chunk, device='cpu')
                        task_chunks.append(task_chunk - base_chunk)

                    combined_task_chunk = sum(task_chunks)
                    merged_chunk = base_chunk + combined_task_chunk
                    merged_chunk_path = os.path.join(tmpdirname, f"{key}_{i}.pt")
                    torch.save(merged_chunk.cpu(), merged_chunk_path)
                    del merged_chunk
                    del base_chunk
                    del task_chunks
                    del combined_task_chunk

                merged_chunks = []
                for i in range(0, numel, chunk_size):
                    merged_chunk_path = os.path.join(tmpdirname, f"{key}_{i}.pt")
                    merged_chunk = torch.load(merged_chunk_path, map_location=device)
                    merged_chunks.append(merged_chunk)
                merged_param = torch.cat(merged_chunks)
                del merged_chunks
            else:
                # If any meta tensors are found, create a full-sized empty tensor on CPU
                merged_param = torch.empty_like(meta_tensors[0], device='cpu')

            merged_state_dict[key] = merged_param.reshape(shape)

    return merged_state_dict

def dfs_merge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    I = kwargs.get("I", torch.ones(len(merged_state_dict)))
    W = kwargs.get("W", torch.eye(len(merged_state_dict), len(models)))
    device = next(models[0].parameters()).device
    chunk_size = kwargs.get('chunk_size', 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        layer_index = 0
        for name, tensor in models[0].state_dict().items():
            if any(layer_type in name for layer_type in ['layer', 'block', 'transformer']):
                if I[layer_index] > 0:
                    merged_param = None

                    # Check for meta tensors before chunking
                    meta_tensors = [model.state_dict()[name] for model in models if model.state_dict()[name].device.type == 'meta']
                    if not meta_tensors:
                        # If no meta tensors, proceed with chunked merging
                        numel = tensor.numel()
                        shape = tensor.shape
                        for i in range(0, numel, chunk_size):
                            chunks = []
                            for model in models:
                                chunk = model.state_dict()[name].flatten()[i:i+chunk_size].to(device)
                                if chunk.device.type == 'meta':
                                    chunk = torch.empty_like(chunk, device='cpu')
                                chunks.append(chunk)

                            chunk_tensor = torch.stack(chunks)
                            merged_chunk = torch.sum(chunk_tensor * W[layer_index].unsqueeze(1).unsqueeze(2), dim=0)
                            merged_chunk_path = os.path.join(tmpdirname, f"{name}_{i}.pt")
                            torch.save(merged_chunk.cpu(), merged_chunk_path)
                            del merged_chunk
                            del chunks
                            del chunk_tensor

                        merged_chunks = []
                        for i in range(0, numel, chunk_size):
                            merged_chunk_path = os.path.join(tmpdirname, f"{name}_{i}.pt")
                            merged_chunk = torch.load(merged_chunk_path, map_location=device)
                            merged_chunks.append(merged_chunk)
                        merged_param = torch.cat(merged_chunks)
                        del merged_chunks
                        merged_state_dict[name] = merged_param.reshape(shape)
                    layer_index += 1
            else:
                if tensor.device.type == 'meta':
                    tensor = torch.empty_like(tensor, device='cpu')
                merged_state_dict[name] = tensor.to(device)

    return merged_state_dict

def frankenmerge(merged_state_dict: Dict[str, torch.Tensor], models: List[torch.nn.Module], **kwargs) -> Dict[str, torch.Tensor]:
    device = next(models[0].parameters()).device
    chunk_size = kwargs.get('chunk_size', 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for i, (name, tensor) in enumerate(models[0].state_dict().items()):
            layer_num = int(name.split('.')[1]) if '.' in name and name.split('.')[1].isdigit() else -1
            if layer_num == -1 or layer_num % len(models) == i:
                merged_param = None
                numel = tensor.numel()
                shape = tensor.shape
                for j in range(0, numel, chunk_size):
                    chunk = tensor.flatten()[j:j+chunk_size].to(device)
                    if chunk.device.type == 'meta':
                        chunk = torch.empty_like(chunk, device='cpu')

                    merged_chunk_path = os.path.join(tmpdirname, f"{name}_{j}.pt")
                    torch.save(chunk.cpu(), merged_chunk_path)
                    del chunk

                merged_chunks = []
                for j in range(0, numel, chunk_size):
                    merged_chunk_path = os.path.join(tmpdirname, f"{name}_{j}.pt")
                    merged_chunk = torch.load(merged_chunk_path, map_location=device)
                    merged_chunks.append(merged_chunk)

                merged_param = torch.cat(merged_chunks)
                del merged_chunks
                merged_state_dict[name] = merged_param.reshape(shape)

    return merged_state_dict
MERGE_TECHNIQUES = {
    "linear": linear_merge,
    "slerp": slerp_merge,
    "ties": disk_based_ties_merge,
    "dare": dare_merge,
    "task_arithmetic": task_arithmetic_merge,
    "frankenmerge": frankenmerge,
    "dfs": dfs_merge,
}
