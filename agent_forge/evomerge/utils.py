import logging
import os
import torch
from typing import List, Dict, Union
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

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

def load_models(model_references: List[ModelReference]) -> List[torch.nn.Module]:
    models = []
    for model_ref in model_references:
        logger.info(f"Loading model: {model_ref.name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_ref.path)
            models.append(model)
        except Exception as e:
            logger.error(f"Failed to load model {model_ref.name}: {str(e)}")
            raise
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
        raise

def validate_merge_config(config: MergeConfig) -> None:
    logger.info("Validating merge configuration")
    if config.merge_method not in ["ps", "dfs", "ps_dfs"]:
        logger.error(f"Invalid merge method: {config.merge_method}")
        raise ValueError(f"Invalid merge method. Choose from: ps, dfs, ps_dfs")
    if len(config.models) < 2:
        logger.error("At least two models are required for merging")
        raise ValueError("At least two models are required for merging")
    valid_techniques = ["linear", "slerp", "ties", "dare", "task_arithmetic", "frankenmerge", "dfs"]
    for technique in config.ps_techniques + config.dfs_techniques:
        if technique not in valid_techniques:
            logger.error(f"Invalid technique: {technique}")
            raise ValueError(f"Invalid technique: {technique}. Choose from: {', '.join(valid_techniques)}")
    logger.info("Merge configuration is valid")

def generate_text(model: torch.nn.Module, tokenizer: AutoTokenizer, prompt: str, max_length: int = 100) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        return ""

def evaluate_model(model_path: str) -> Dict[str, Union[float, str]]:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        prompts = [
            "The capital of France is",
            "The theory of relativity was proposed by",
            "The largest planet in our solar system is"
        ]

        total_perplexity = 0
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss)
                total_perplexity += perplexity.item()

        avg_perplexity = total_perplexity / len(prompts)
        coherence_score = 1 / avg_perplexity  # Lower perplexity means higher coherence
        return {"overall_score": coherence_score, "perplexity": avg_perplexity}

    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        return {"overall_score": float('-inf'), "error": str(e)}

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
