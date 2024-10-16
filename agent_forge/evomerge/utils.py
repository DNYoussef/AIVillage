import logging
import os
import torch
import numpy as np
import psutil
import shutil
from typing import List, Dict, Union, Callable
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import traceback
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EvoMergeException(Exception):
    """Custom exception class for EvoMerge errors."""
    pass

class ModelReference(BaseModel):
    name: str
    path: str

class MergeConfig(BaseModel):
    merge_method: str
    models: List[ModelReference]
    parameters: Dict[str, Union[float, List[float], Dict[str, float]]]
    custom_dir: str
    ps_techniques: List[str]
    dfs_techniques: List[str]

def check_system_resources(model_paths: List[str]) -> bool:
    total_model_size = 0
    for path in model_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                total_model_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
    
    if total_model_size > 0:
        free_disk_space = shutil.disk_usage(os.path.dirname(next(path for path in model_paths if os.path.exists(path)))).free
    else:
        free_disk_space = shutil.disk_usage(".").free
    
    available_ram = psutil.virtual_memory().available

    logger.info(f"Total model size: {total_model_size / (1024**3):.2f} GB")
    logger.info(f"Free disk space: {free_disk_space / (1024**3):.2f} GB")
    logger.info(f"Available RAM: {available_ram / (1024**3):.2f} GB")

    if total_model_size > free_disk_space:
        logger.error("Not enough disk space to store merged models!")
        return False
    if total_model_size > available_ram:
        logger.warning("Available RAM might not be sufficient to load all models simultaneously!")
        # We'll continue with a warning, but you might want to implement a more sophisticated
        # memory management strategy if this is a common issue
    return True

def setup_gpu_if_available():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU available. Using device: {device}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available. Using CPU.")
    return device

def clean_up_models(model_paths: List[str]):
    for path in model_paths:
        try:
            if os.path.exists(path):
                logger.info(f"Removing model: {path}")
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to remove model {path}: {str(e)}")
            logger.warning("This is not a critical error, but you may want to manually remove the file or directory.")



def load_models(model_references: List[ModelReference]) -> List[torch.nn.Module]:
    logger.info("Starting to load models")
    models = []
    for model_ref in tqdm(model_references, desc="Loading models"):
        try:
            logger.info(f"Loading model: {model_ref.name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_ref.path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            models.append(model)
            logger.info(f"Successfully loaded model: {model_ref.name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_ref.name}: {str(e)}")
    return models

def save_model(model: torch.nn.Module, path: str) -> None:
    logger.info(f"Saving merged model to: {path}")
    try:
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        tokenizer.save_pretrained(path)
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise EvoMergeException(f"Error saving model: {str(e)}")

def generate_text(model: torch.nn.Module, tokenizer: AutoTokenizer, prompt: str, max_length: int = 100) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        raise EvoMergeException(f"Error generating text: {str(e)}")

def evaluate_model(model_path: str) -> Dict[str, Union[float, str]]:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Implement your evaluation logic here
        # This is a placeholder implementation
        return {
            "perplexity": 100.0,
            "accuracy": 0.8,
            "overall_score": 0.9
        }

    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise EvoMergeException(f"Error evaluating model: {str(e)}")

def parallel_evaluate_models(model_paths: List[str], max_workers: int = None) -> List[Dict[str, Union[float, str]]]:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(evaluate_model, model_paths))

# Merge Techniques
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

MERGE_TECHNIQUES = {
    "linear": linear_merge,
    "slerp": slerp_merge,
    "ties": ties_merge,
    "dare": dare_merge,
    "task_arithmetic": task_arithmetic_merge,
    "frankenmerge": frankenmerge,
    "dfs": dfs_merge
}
