import requests
import json
from typing import List, Optional
from pydantic import BaseModel
import os
import re
import subprocess
import logging

BASE_URL = "http://localhost:11434/api"
logger = logging.getLogger(__name__)

class ModelReference(BaseModel):
    name: str
    path: Optional[str] = None

def load_models(model_references: List[ModelReference]):
    models = []
    for ref in model_references:
        if ref.name.startswith("ollama:"):
            model_name = ref.name.split(":", 1)[1]
            model_info = _get_model_info(model_name)
            model_info['name'] = model_name
            models.append(model_info)
        else:
            raise NotImplementedError(f"Loading of non-Ollama models is not implemented yet: {ref.name}")
    return models

def save_model(merged_model, path: str):
    logger.info(f"Saving merged model: {merged_model['name']}")
    logger.debug(f"Modelfile content:\n{merged_model['modelfile']}")
    
    os.makedirs(path, exist_ok=True)
    
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', merged_model['name'])
    if sanitized_name != merged_model['name']:
        logger.warning(f"Model name sanitized from '{merged_model['name']}' to '{sanitized_name}'")
    
    modelfile_path = os.path.join(path, f"{sanitized_name}.modelfile")
    with open(modelfile_path, "w") as f:
        f.write(merged_model['modelfile'])
    
    logger.info(f"Modelfile saved as: {modelfile_path}")
    
    try:
        result = subprocess.run(["ollama", "create", sanitized_name, "-f", modelfile_path], 
                                capture_output=True, text=True, check=True)
        logger.info(f"Model {sanitized_name} successfully created using CLI")
        logger.debug(f"CLI Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create model using CLI: {e}")
        logger.error(f"CLI Error Output: {e.stderr}")
        logger.debug("Contents of the modelfile:")
        with open(modelfile_path, 'r') as f:
            logger.debug(f.read())
        return None
    
    logger.info(f"Model {sanitized_name} successfully saved to {path}")
    return True

def _get_model_info(model_name):
    logger.info(f"Fetching info for model: {model_name}")
    response = requests.post(f"{BASE_URL}/show", json={"name": model_name})
    if response.status_code == 200:
        logger.debug(f"Raw API response for {model_name}:")
        logger.debug(json.dumps(response.json(), indent=2))
        return response.json()
    else:
        logger.error(f"Failed to get model info: {response.text}")
        raise Exception(f"Failed to get model info: {response.text}")

def _create_model(name, modelfile):
    logger.info(f"Creating model: {name}")
    response = requests.post(f"{BASE_URL}/create", json={"name": name, "modelfile": modelfile})
    if response.status_code == 200:
        logger.info(f"Model {name} created successfully")
        return response.json()
    else:
        error_message = response.text
        try:
            error_json = response.json()
            if 'error' in error_json:
                error_message = error_json['error']
        except:
            pass
        logger.error(f"Failed to create model: {error_message}")
        logger.error(f"Response status code: {response.status_code}")
        logger.error(f"Full response: {response.text}")
        return None

def generate_text(model_name, prompt):
    logger.info(f"Generating text with model: {model_name}")
    response = requests.post(f"{BASE_URL}/generate", json={"model": model_name, "prompt": prompt})
    if response.status_code == 200:
        return response.json()['response']
    else:
        logger.error(f"Failed to generate text: {response.text}")
        raise Exception(f"Failed to generate text: {response.text}")

def validate_merge_config(config):
    logger.info("Validating merge configuration")
    if config.merge_method not in config.merge_methods:
        logger.error(f"Invalid merge method: {config.merge_method}")
        raise ValueError(f"Invalid merge method. Choose from: {', '.join(config.merge_methods)}")
    if len(config.models) < 2:
        logger.error("At least two models are required for merging")
        raise ValueError("At least two models are required for merging")
    logger.info("Merge configuration is valid")