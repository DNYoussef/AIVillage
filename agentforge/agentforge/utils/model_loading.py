import torch
from transformers import AutoModelForCausalLM
from mergekit.io.lazy_tensor_loader import LazyTensorLoader

def load_models(model_references):
    models = []
    for ref in model_references:
        loader = LazyTensorLoader.from_disk(ref.path)
        model = AutoModelForCausalLM.from_pretrained(ref.path, state_dict=loader.get_tensor)
        models.append(model)
    return models

def save_model(model, path):
    model.save_pretrained(path)