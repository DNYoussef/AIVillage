import requests
import json
from typing import List, Optional
from pydantic import BaseModel
import os
import re
import subprocess

BASE_URL = "http://localhost:11434/api"

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
    print(f"Merged model metadata: {merged_model['name']}")
    print(f"Modelfile content:\n{merged_model['modelfile']}")
    print(f"Saving model to: {path}")
    
    os.makedirs(path, exist_ok=True)
    
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', merged_model['name'])
    if sanitized_name != merged_model['name']:
        print(f"Warning: Model name sanitized from '{merged_model['name']}' to '{sanitized_name}'")
    
    modelfile_path = os.path.join(path, f"{sanitized_name}.modelfile")
    with open(modelfile_path, "w") as f:
        f.write(merged_model['modelfile'])
    
    print(f"Modelfile saved as: {modelfile_path}")
    
    try:
        result = subprocess.run(["ollama", "create", sanitized_name, "-f", modelfile_path], 
                                capture_output=True, text=True, check=True)
        print(f"Model {sanitized_name} successfully created using CLI")
        print(f"CLI Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create model using CLI: {e}")
        print(f"CLI Error Output: {e.stderr}")
        print("Contents of the modelfile:")
        with open(modelfile_path, 'r') as f:
            print(f.read())
        return None
    
    print(f"Model {sanitized_name} successfully saved to {path}")
    return True

def _get_model_info(model_name):
    response = requests.post(f"{BASE_URL}/show", json={"name": model_name})
    if response.status_code == 200:
        print(f"Raw API response for {model_name}:")
        print(json.dumps(response.json(), indent=2))
        return response.json()
    else:
        raise Exception(f"Failed to get model info: {response.text}")

def _create_model(name, modelfile):
    response = requests.post(f"{BASE_URL}/create", json={"name": name, "modelfile": modelfile})
    if response.status_code == 200:
        print(f"Model {name} created successfully")
        return response.json()
    else:
        error_message = response.text
        try:
            error_json = response.json()
            if 'error' in error_json:
                error_message = error_json['error']
        except:
            pass
        print(f"Failed to create model: {error_message}")
        print(f"Response status code: {response.status_code}")
        print(f"Full response: {response.text}")
        return None

def generate_text(model_name, prompt):
    response = requests.post(f"{BASE_URL}/generate", json={"model": model_name, "prompt": prompt})
    if response.status_code == 200:
        return response.json()['response']
    else:
        raise Exception(f"Failed to generate text: {response.text}")