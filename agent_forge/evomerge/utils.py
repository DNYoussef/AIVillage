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

def check_system_resources(model_paths: List[str]):
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
        logger.warning("Not enough disk space to store merged models!")
    if total_model_size > available_ram:
        logger.warning("Available RAM might not be sufficient to load all models simultaneously!")

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
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to remove model {path}: {str(e)}")

# Add any other utility functions that don't depend on evaluation.py here
