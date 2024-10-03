import logging
import os
import re
import torch
from typing import List, Optional
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class ModelReference(BaseModel):
    name: str
    path: Optional[str] = None

class MergeConfig(BaseModel):
    merge_method: str
    models: List[ModelReference]
def load_models(model_references: List[ModelReference]) -> List[torch.nn.Module]:
    models = []
    for model_ref in model_references:
        logger.info(f"Loading model: {model_ref.name}")
        model = AutoModelForCausalLM.from_pretrained(model_ref.path)
        models.append(model)
    return models

def save_model(model: torch.nn.Module, path: str) -> None:
    logger.info(f"Saving merged model to: {path}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    tokenizer.save_pretrained(path)
    
def validate_merge_config(config: MergeConfig) -> None:
    logger.info("Validating merge configuration")
    if config.merge_method not in ["ps", "dfs", "ps_dfs"]:
        logger.error(f"Invalid merge method: {config.merge_method}")
        raise ValueError(f"Invalid merge method. Choose from: ps, dfs, ps_dfs")
    if len(config.models) < 2:
        logger.error("At least two models are required for merging")
        raise ValueError("At least two models are required for merging")
    logger.info("Merge configuration is valid")

def generate_text(model: torch.nn.Module, tokenizer: AutoTokenizer, prompt: str, max_length: int = 100) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model_path: str) -> float:
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
        return coherence_score

    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        return float('-inf')  # Return worst possible score if evaluation fails