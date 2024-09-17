import ollama
from typing import List
from .config import ModelReference

def load_models(model_references: List[ModelReference]):
    models = []
    for ref in model_references:
        model = ollama.Client().get_model(ref.name)
        models.append(model)
    return models

def save_model(merged_model, path: str):
    print(f"Merged model structure: {merged_model}")
    print(f"Model would be saved to: {path}")