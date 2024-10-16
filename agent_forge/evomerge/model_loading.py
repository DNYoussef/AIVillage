import logging
import os
import torch
import psutil
import shutil
from typing import List
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import multiprocessing
import traceback
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelReference(BaseModel):
    name: str
    path: str

class EvoMergeException(Exception):
    """Custom exception class for EvoMerge errors."""
    pass

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

def load_single_model(model_ref: ModelReference, use_8bit: bool = False, use_4bit: bool = False) -> torch.nn.Module:
    logger.info(f"Starting to load model: {model_ref.name}")
    try:
        # Get the latest snapshot directory
        snapshot_dirs = [d for d in os.listdir(model_ref.path) if os.path.isdir(os.path.join(model_ref.path, d))]
        if snapshot_dirs:
            latest_snapshot_dir = max(snapshot_dirs, key=lambda d: os.path.getctime(os.path.join(model_ref.path, d)))
            model_path = os.path.join(model_ref.path, latest_snapshot_dir)
        else:
            model_path = model_ref.path
        
        logger.info(f"Loading model {model_ref.name} from {model_path}")
        logger.info("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer initialized successfully")
        
        logger.info("Initializing model...")
        load_in_8bit = use_8bit
        load_in_4bit = use_4bit
        if use_8bit and use_4bit:
            logger.warning("Both 8-bit and 4-bit quantization specified. Defaulting to 8-bit.")
            load_in_4bit = False

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            low_cpu_mem_usage=True
        )
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        logger.info(f"Successfully loaded model: {model_ref.name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_ref.name}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def load_model_process(model_ref, queue, use_8bit, use_4bit):
    try:
        model = load_single_model(model_ref, use_8bit, use_4bit)
        queue.put(("success", model))
    except Exception as e:
        queue.put(("error", str(e), traceback.format_exc()))

def load_model_with_timeout(model_ref: ModelReference, timeout: int = 1800, max_retries: int = 3, use_8bit: bool = False, use_4bit: bool = False) -> torch.nn.Module:
    for attempt in range(max_retries):
        logger.info(f"Attempt {attempt + 1} to load model: {model_ref.name}")
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=load_model_process, args=(model_ref, queue, use_8bit, use_4bit))
        process.start()
        logger.info(f"Started loading process for model: {model_ref.name}")
        for _ in range(timeout):
            if not process.is_alive():
                break
            if not queue.empty():
                result = queue.get()
                if result[0] == "success":
                    logger.info(f"Successfully loaded model {model_ref.name}")
                    return result[1]
                else:
                    logger.error(f"Error loading model {model_ref.name}: {result[1]}")
                    logger.error(f"Traceback: {result[2]}")
                    break
            process.join(1)
            logger.info(f"Still loading model {model_ref.name}...")
        if process.is_alive():
            logger.error(f"Timeout occurred while loading model {model_ref.name}")
            process.terminate()
            process.join()
        if attempt == max_retries - 1:
            raise EvoMergeException(f"Failed to load model {model_ref.name} after {max_retries} attempts")
        logger.info(f"Retrying to load model: {model_ref.name}")
    raise EvoMergeException(f"Failed to load model {model_ref.name} after {max_retries} attempts")

def load_models(model_references: List[ModelReference], use_8bit: bool = False, use_4bit: bool = False) -> List[torch.nn.Module]:
    logger.info("Starting to load models")
    model_paths = [ref.path for ref in model_references]
    check_system_resources(model_paths)
    
    models = []
    for model_ref in tqdm(model_references, desc="Loading models"):
        try:
            logger.info(f"Attempting to load model: {model_ref.name}")
            logger.info(f"Model path: {model_ref.path}")
            if not os.path.exists(model_ref.path):
                raise EvoMergeException(f"Model path does not exist: {model_ref.path}")
            model = load_model_with_timeout(model_ref, use_8bit=use_8bit, use_4bit=use_4bit)
            models.append(model)
            logger.info(f"Successfully loaded model: {model_ref.name}")
        except EvoMergeException as e:
            logger.error(f"Failed to load model {model_ref.name}: {str(e)}")
            logger.error("Continuing to load other models...")
    logger.info(f"Finished loading models. Successfully loaded {len(models)} out of {len(model_references)} models.")
    return models

def save_model(model: torch.nn.Module, path: str) -> None:
    logger.info(f"Saving merged model to: {path}")
    try:
        os.makedirs(path, exist_ok=True)
        
        # Save model in shards
        model.save_pretrained(path, max_shard_size="500MB")
        
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        tokenizer.save_pretrained(path)
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise EvoMergeException(f"Error saving model: {str(e)}")
