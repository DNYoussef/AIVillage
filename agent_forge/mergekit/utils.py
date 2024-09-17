import requests
import json
from typing import List
from .config import ModelReference
import os
import re

BASE_URL = "http://localhost:11434/api"

def load_models(model_references: List[ModelReference]):
    models = []
    for ref in model_references:
        if ref.name.startswith("ollama:"):
            model_name = ref.name.split(":", 1)[1]
            model_info = _get_model_info(model_name)
            # Add the name to the model_info dictionary
            model_info['name'] = model_name
            models.append(model_info)
        else:
            raise NotImplementedError(f"Loading of non-Ollama models is not implemented yet: {ref.name}")
    return models

def save_model(merged_model, path: str):
    print(f"Merged model metadata: {merged_model['name']}, {merged_model['modelfile']}")
    print(f"Saving model to: {path}")
    
    # Ensure the path exists
    os.makedirs(path, exist_ok=True)
    
    # Sanitize the model name
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', merged_model['name'])
    if sanitized_name != merged_model['name']:
        print(f"Warning: Model name sanitized from '{merged_model['name']}' to '{sanitized_name}'")
    
    # Print sanitized name for debugging
    print(f"Attempting to create model with name: '{sanitized_name}'")
    
    try:
        # Create the new model using the Ollama API
        create_result = _create_model(sanitized_name, merged_model['modelfile'])

        # Save the modelfile content to a file
        modelfile_path = os.path.join(path, f"{sanitized_name}.modelfile")
        with open(modelfile_path, "w") as f:
            f.write(merged_model['modelfile'])
        
        print(f"Model {sanitized_name} successfully saved to {path}")
        print(f"Modelfile saved as: {modelfile_path}")
        return create_result
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        print("Modelfile content:")
        print(merged_model['modelfile'])
        return None

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