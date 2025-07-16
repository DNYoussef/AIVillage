import logging
import os
import shutil

import psutil
from pydantic import BaseModel
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class ModelReference(BaseModel):
    name: str
    path: str

class EvoMergeException(Exception):
    """Custom exception class for EvoMerge errors."""

def check_system_resources(model_paths: list[str]):
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

def load_model_with_mmap(model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)

    for name, param in model.named_parameters():
        param_path = os.path.join(model_path, f"{name}.mmap")
        if os.path.exists(param_path):
            param.data = torch.nn.Parameter(torch.from_file(param_path, shared=True))
        else:
            # If mmap file doesn't exist, create it
            param.data.to_file(param_path)
            param.data = torch.nn.Parameter(torch.from_file(param_path, shared=True))

    return model

def load_models(model_references):
    models = []
    tokenizers = []
    for model_ref in model_references:
        model = load_model_with_mmap(model_ref.path)
        tokenizer = AutoTokenizer.from_pretrained(model_ref.path)
        models.append(model)
        tokenizers.append(tokenizer)
    return models, tokenizers

def save_model(model: torch.nn.Module, path: str) -> None:
    logger.info(f"Saving merged model to: {path}")
    try:
        os.makedirs(path, exist_ok=True)

        # Save model in shards
        model.save_pretrained(path, max_shard_size="500MB")

        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        tokenizer.save_pretrained(path)
    except Exception as e:
        logger.error(f"Failed to save model: {e!s}")
        raise EvoMergeException(f"Error saving model: {e!s}")
