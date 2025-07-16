import logging
import tempfile

import torch

logger = logging.getLogger(__name__)

def linear_merge(merged_state_dict: dict[str, torch.Tensor], models: list[torch.nn.Module], **kwargs) -> dict[str, torch.Tensor]:
    weights = kwargs.get("weights", torch.ones(len(models)) / len(models))
    device = next(models[0].parameters()).device
    chunk_size = kwargs.get("chunk_size", 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for key in models[0].state_dict().keys():
            # Initialize merged_param as an empty list to collect chunks
            merged_chunks = []

            # Check for meta tensors
            meta_tensors = [model.state_dict()[key] for model in models if model.state_dict()[key].is_meta]
            if meta_tensors:
                # If meta tensors are found, create an empty tensor of the correct shape
                param_shape = models[0].state_dict()[key].shape
                merged_param = torch.empty(param_shape, device="cpu")
            else:
                numel = models[0].state_dict()[key].numel()
                param_shape = models[0].state_dict()[key].shape
                for i in range(0, numel, chunk_size):
                    chunk_params = []
                    for model in models:
                        chunk_param = model.state_dict()[key].flatten()[i:i+chunk_size].to(device)
                        chunk_params.append(chunk_param)

                    merged_chunk = sum(w * t for w, t in zip(weights, chunk_params, strict=False))
                    merged_chunks.append(merged_chunk.cpu())

                    del merged_chunk, chunk_params

                merged_param = torch.cat(merged_chunks)
                del merged_chunks

            # Reshape merged_param to the original parameter shape
            merged_state_dict[key] = merged_param.view(param_shape)

    return merged_state_dict

def slerp_merge(merged_state_dict: dict[str, torch.Tensor], models: list[torch.nn.Module], **kwargs) -> dict[str, torch.Tensor]:
    t = kwargs.get("t", 0.5)
    device = next(models[0].parameters()).device
    chunk_size = kwargs.get("chunk_size", 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for key in models[0].state_dict().keys():
            # Initialize merged_param as an empty list to collect chunks
            merged_chunks = []

            # Check for meta tensors before chunking
            meta_tensors = [model.state_dict()[key] for model in models if model.state_dict()[key].is_meta]
            if not meta_tensors:
                numel = models[0].state_dict()[key].numel()
                param_shape = models[0].state_dict()[key].shape  # Correctly assign param_shape
                for i in range(0, numel, chunk_size):
                    w1 = models[0].state_dict()[key].flatten()[i:i+chunk_size].to(device)
                    w2 = models[1].state_dict()[key].flatten()[i:i+chunk_size].to(device)

                    omega = torch.arccos(torch.clamp(torch.dot(w1, w2) / (w1.norm() * w2.norm()), -1, 1))
                    so = torch.sin(omega)
                    merged_chunk = (torch.sin((1.0 - t) * omega) / so) * w1 + (torch.sin(t * omega) / so) * w2
                    merged_chunks.append(merged_chunk.cpu())

                    del merged_chunk, w1, w2

                merged_param = torch.cat(merged_chunks)
                del merged_chunks
            else:
                param_shape = models[0].state_dict()[key].shape
                merged_param = torch.empty(param_shape, device="cpu")

            merged_state_dict[key] = merged_param.view(param_shape)

    return merged_state_dict

def disk_based_ties_merge(merged_state_dict: dict[str, torch.Tensor], models: list[torch.nn.Module], **kwargs) -> dict[str, torch.Tensor]:
    threshold = kwargs.get("threshold", 0.1)
    device = torch.device("cpu")  # Use CPU to handle larger tensors without GPU memory issues
    chunk_size = kwargs.get("chunk_size", 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for key in models[0].state_dict().keys():
            # Initialize merged_param as an empty list to collect chunks
            merged_chunks = []

            # Get the original shape once
            param_shape = models[0].state_dict()[key].shape
            numel = models[0].state_dict()[key].numel()

            # Check for meta tensors before chunking
            meta_tensors = [model.state_dict()[key] for model in models if model.state_dict()[key].is_meta]
            if not meta_tensors:
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
                    mask_sum = mask.sum(dim=0)
                    mask_sum[mask_sum == 0] = 1.0  # Prevent division by zero

                    merged_chunk = masked_chunks.sum(dim=0) / mask_sum
                    merged_chunks.append(merged_chunk.cpu())

                    del merged_chunk, chunks, stacked_chunks, abs_chunks, mean_abs, mask, masked_chunks, mask_sum

                merged_param = torch.cat(merged_chunks)
                del merged_chunks
            else:
                merged_param = torch.empty(param_shape, device="cpu")

            # Reshape merged_param to the original parameter shape
            merged_state_dict[key] = merged_param.view(param_shape)

    return merged_state_dict

def dare_merge(merged_state_dict: dict[str, torch.Tensor], models: list[torch.nn.Module], **kwargs) -> dict[str, torch.Tensor]:
    threshold = kwargs.get("threshold", 0.1)
    amplification = kwargs.get("amplification", 2.0)
    device = torch.device("cpu")
    chunk_size = kwargs.get("chunk_size", 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for key in models[0].state_dict().keys():
            merged_chunks = []

            param_shape = models[0].state_dict()[key].shape
            numel = models[0].state_dict()[key].numel()

            meta_tensors = [model.state_dict()[key] for model in models if model.state_dict()[key].is_meta]
            if not meta_tensors:
                for i in range(0, numel, chunk_size):
                    chunks = []
                    for model in models:
                        chunk = model.state_dict()[key].flatten()[i:i+chunk_size].to(device)
                        chunks.append(chunk)

                    chunk_tensor = torch.stack(chunks)
                    mean_chunk = chunk_tensor.mean(dim=0)
                    abs_diff = torch.abs(chunk_tensor - mean_chunk)
                    mask = abs_diff > threshold

                    # Apply mask element-wise without changing shape
                    amplified_diff = (chunk_tensor - mean_chunk) * amplification * mask
                    merged_chunk = mean_chunk + amplified_diff.mean(dim=0)
                    merged_chunks.append(merged_chunk.cpu())

                    del merged_chunk, chunks, chunk_tensor, mean_chunk, abs_diff, mask, amplified_diff

                merged_param = torch.cat(merged_chunks)
                del merged_chunks
            else:
                merged_param = torch.empty(param_shape, device="cpu")

            merged_state_dict[key] = merged_param.view(param_shape)

    return merged_state_dict

def task_arithmetic_merge(merged_state_dict: dict[str, torch.Tensor], models: list[torch.nn.Module], **kwargs) -> dict[str, torch.Tensor]:
    base_weights = kwargs.get("base_weights", {})
    task_weights = kwargs.get("task_weights", [])
    device = torch.device("cpu")
    chunk_size = kwargs.get("chunk_size", 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        for key in base_weights:
            merged_chunks = []

            param_shape = base_weights[key].shape
            numel = base_weights[key].numel()

            meta_tensors = [base_weights[key]] + [task_weight[key] for task_weight in task_weights if task_weight[key].is_meta]
            if not meta_tensors:
                for i in range(0, numel, chunk_size):
                    base_chunk = base_weights[key].flatten()[i:i+chunk_size].to(device)

                    task_chunks = []
                    for task_weight in task_weights:
                        task_chunk = task_weight[key].flatten()[i:i+chunk_size].to(device)
                        task_chunks.append(task_chunk - base_chunk)

                    combined_task_chunk = sum(task_chunks)
                    merged_chunk = base_chunk + combined_task_chunk
                    merged_chunks.append(merged_chunk.cpu())

                    del merged_chunk, base_chunk, task_chunks, combined_task_chunk

                merged_param = torch.cat(merged_chunks)
                del merged_chunks
            else:
                merged_param = torch.empty(param_shape, device="cpu")

            merged_state_dict[key] = merged_param.view(param_shape)

    return merged_state_dict

def dfs_merge(merged_state_dict: dict[str, torch.Tensor], models: list[torch.nn.Module], **kwargs) -> dict[str, torch.Tensor]:
    I = kwargs.get("I", torch.ones(len(merged_state_dict)))
    W = kwargs.get("W", torch.eye(len(merged_state_dict), len(models)))
    device = torch.device("cpu")
    chunk_size = kwargs.get("chunk_size", 1000000)

    with tempfile.TemporaryDirectory() as tmpdirname:
        layer_index = 0
        for name, tensor in models[0].state_dict().items():
            if any(layer_type in name for layer_type in ["layer", "block", "transformer"]):
                if I[layer_index] > 0:
                    merged_chunks = []

                    param_shape = tensor.shape
                    numel = tensor.numel()

                    meta_tensors = [model.state_dict()[name] for model in models if model.state_dict()[name].is_meta]
                    if not meta_tensors:
                        for i in range(0, numel, chunk_size):
                            chunks = []
                            for model in models:
                                chunk = model.state_dict()[name].flatten()[i:i+chunk_size].to(device)
                                chunks.append(chunk)

                            chunk_tensor = torch.stack(chunks)
                            merged_chunk = torch.sum(chunk_tensor * W[layer_index].unsqueeze(1), dim=0)
                            merged_chunks.append(merged_chunk.cpu())

                            del merged_chunk, chunks, chunk_tensor

                        merged_param = torch.cat(merged_chunks)
                        del merged_chunks
                        merged_state_dict[name] = merged_param.view(param_shape)
                    layer_index += 1
            else:
                param_shape = tensor.shape
                merged_state_dict[name] = tensor.detach().cpu().view(param_shape)

    return merged_state_dict

def frankenmerge(merged_model, models, **kwargs):
    merged_state_dict = {}
    for name, param in merged_model.named_parameters():
        param_shape = param.shape
        merged_param = torch.zeros_like(param)
        for i, model in enumerate(models):
            if name in model.state_dict():
                merged_param += model.state_dict()[name]
        merged_param /= len(models)
        merged_state_dict[name] = merged_param.view(param_shape)

    # Create a new state dict with only the compatible parameters
    compatible_state_dict = {k: v for k, v in merged_state_dict.items() if k in merged_model.state_dict()}

    # Load the compatible state dict into the model
    merged_model.load_state_dict(compatible_state_dict, strict=False)

    return merged_model

MERGE_TECHNIQUES = {
    "linear": linear_merge,
    "slerp": slerp_merge,
    "ties": disk_based_ties_merge,
    "dare": dare_merge,
    "task_arithmetic": task_arithmetic_merge,
    "frankenmerge": frankenmerge,
    "dfs": dfs_merge,
}



