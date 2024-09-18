import os
import numpy as np
from typing import List, Dict, Tuple
import requests
import logging
from .config import MergeKitConfig
from .gguf_utils import GGUFReader, GGUFWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MergeKitMerger:
    def __init__(self, config: MergeKitConfig):
        self.config = config
        self.ollama_dir = os.path.expanduser("~/.ollama/models")
        self.custom_dir = config.custom_dir if hasattr(config, 'custom_dir') else os.path.join(os.getcwd(), 'custom_models')

    def merge(self, models: List[Dict]) -> Dict:
        logger.info("Starting model merger")
        if self.config.merge_method == "linear":
            try:
                merged_model = self._linear_merge(models)
                logger.info("Model merging completed successfully")
                return merged_model
            except Exception as e:
                logger.error(f"Error during linear merge: {str(e)}")
                raise RuntimeError(f"Error during linear merge: {str(e)}")
        elif self.config.merge_method == "slerp":
            try:
                merged_model = self._slerp_merge(models)
                logger.info("Model merging completed successfully")
                return merged_model
            except Exception as e:
                logger.error(f"Error during SLERP merge: {str(e)}")
                raise RuntimeError(f"Error during SLERP merge: {str(e)}")
        else:
            raise NotImplementedError(f"Merge method {self.config.merge_method} not implemented")

    def _linear_merge(self, models: List[Dict]) -> Dict:
        weights = self._normalize_weights(self.config.parameters.get("weights", [1/len(models)] * len(models)))
        merged_weights, merged_metadata = self._load_and_merge_weights(models, weights, self._linear_interpolation)
        return self._save_and_create_model(merged_weights, merged_metadata, models)

    def _slerp_merge(self, models: List[Dict]) -> Dict:
        weights = self._normalize_weights(self.config.parameters.get("weights", [1/len(models)] * len(models)))
        merged_weights, merged_metadata = self._load_and_merge_weights(models, weights, self._slerp_interpolation)
        return self._save_and_create_model(merged_weights, merged_metadata, models)

    def _load_and_merge_weights(self, models: List[Dict], weights: List[float], merge_func) -> Tuple[Dict[str, np.ndarray], Dict]:
        merged_weights = {}
        merged_metadata = {}
        for i, model in enumerate(models):
            model_path = self._find_model_file(model['name'])
            if not model_path:
                raise FileNotFoundError(f"Model file not found for: {model['name']}")

            reader = GGUFReader(model_path)
            reader.read()

            if i == 0:
                merged_metadata = reader.metadata.copy()
                for name, tensor in reader.tensors.items():
                    merged_weights[name] = tensor * weights[i]
            else:
                for name, tensor in reader.tensors.items():
                    if name in merged_weights:
                        merged_weights[name] = merge_func(merged_weights[name], tensor, weights[i-1], weights[i])
                    else:
                        logger.warning(f"Tensor {name} not found in base model, skipping")

        return merged_weights, merged_metadata

    def _find_model_file(self, model_name: str) -> str:
        possible_locations = [
            os.path.join(self.ollama_dir, "blobs", f"{model_name}.gguf"),
            os.path.join(self.custom_dir, f"{model_name}.gguf"),
            os.path.join(self.custom_dir, model_name, f"{model_name}.gguf")
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                logger.info(f"Found model file at: {location}")
                return location
        
        logger.error(f"Model file not found for: {model_name}")
        logger.error("Searched locations:")
        for location in possible_locations:
            logger.error(f"  {location}")
        return ""

    def _save_merged_model(self, weights: Dict[str, np.ndarray], metadata: Dict, model_name: str) -> str:
        new_model_path = os.path.join(self.custom_dir, f"{model_name}.gguf")
        writer = GGUFWriter(new_model_path)
        for key, value in metadata.items():
            writer.add_metadata(key, value)
        for name, tensor in weights.items():
            writer.add_tensor(name, tensor)
        writer.write()
        return new_model_path

    def _linear_interpolation(self, tensor1, tensor2, weight1, weight2):
        return tensor1 * weight1 + tensor2 * weight2

    def _slerp_interpolation(self, tensor1, tensor2, weight1, weight2):
        omega = np.arccos(np.clip(np.sum(tensor1*tensor2) / (np.linalg.norm(tensor1) * np.linalg.norm(tensor2)), -1, 1))
        so = np.sin(omega)
        return np.sin((1.0-weight2)*omega) / so * tensor1 + np.sin(weight2*omega) / so * tensor2

    def _save_and_create_model(self, weights: Dict[str, np.ndarray], metadata: Dict, models: List[Dict]) -> Dict:
        merged_model_name = '_'.join(model['name'] for model in models)
        new_model_path = self._save_merged_model(weights, metadata, merged_model_name)
        self._create_ollama_model(new_model_path, merged_model_name)
        return {'name': merged_model_name, 'path': new_model_path}

    def _save_merged_model(self, weights: Dict[str, np.ndarray], metadata: Dict, model_name: str) -> str:
        new_model_path = os.path.join(self.ollama_dir, "blobs", f"{model_name}.gguf")
        writer = GGUFWriter(new_model_path)
        for key, value in metadata.items():
            writer.add_metadata(key, value)
        for name, tensor in weights.items():
            writer.add_tensor(name, tensor)
        writer.write()
        return new_model_path

    def _create_ollama_modelfile(self, model_path: str, model_name: str) -> str:
        modelfile = f"FROM {model_path}\n\n"
        modelfile += f'SYSTEM "This is a merged model named {model_name}, combining multiple models using {self.config.merge_method} interpolation of weights."\n\n'

        # Add the template from the first model
        if 'template' in self.config.models[0].dict():
            template = self.config.models[0].dict()['template']
            modelfile += f'TEMPLATE """\n{template}\n"""\n'

        return modelfile

    def _create_ollama_model(self, model_path: str, model_name: str) -> None:
        modelfile = self._create_ollama_modelfile(model_path, model_name)
        if self._validate_modelfile(modelfile):
            response = requests.post('http://localhost:11434/api/create', json={
                "name": model_name,
                "modelfile": modelfile
            })
            if response.status_code != 200:
                raise Exception(f"Failed to create Ollama model: {response.text}")
        else:
            raise ValueError("Invalid modelfile generated")

    def _validate_modelfile(self, modelfile: str) -> bool:
        valid_commands = {"from", "merge", "parameter", "template", "system"}
        lines = modelfile.split('\n')
        state = 'normal'

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if state == 'normal':
                if line.lower().startswith('template'):
                    if not line.lower().startswith('template """'):
                        logger.error(f"Invalid TEMPLATE syntax: {line}")
                        return False
                    state = 'in_template'
                    continue

                parts = line.split(maxsplit=1)
                if parts and parts[0].lower() not in valid_commands:
                    logger.error(f"Invalid command in modelfile: {line}")
                    return False

            elif state == 'in_template':
                if line.strip() == '"""':
                    state = 'normal'

        if state == 'in_template':
            logger.error("TEMPLATE section not properly closed")
            return False

        return True

    def _normalize_weights(self, weights: List[float]) -> List[float]:
        total = sum(weights)
        return [w / total for w in weights]

    def validate_merged_model(self, model_path: str) -> bool:
        logger.info(f"Validating merged model at {model_path}")

        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist at {model_path}")
            return False

        try:
            reader = GGUFReader(model_path)
            reader.read()

            # Check if the model has the expected metadata
            required_metadata = ['general.name', 'general.architecture', 'general.file_type']
            for key in required_metadata:
                if key not in reader.metadata:
                    logger.error(f"Missing required metadata: {key}")
                    return False

            # Check if the model has expected tensors
            expected_tensors = ['token_embd.weight', 'output.weight']  # Add more as needed
            for tensor_name in expected_tensors:
                if tensor_name not in reader.tensors:
                    logger.error(f"Missing expected tensor: {tensor_name}")
                    return False

            # Check if tensor shapes are consistent
            vocab_size = reader.metadata.get('general.vocab_size')
            if vocab_size is not None:
                vocab_size = int(vocab_size)
                if reader.tensors['token_embd.weight'].shape[0] != vocab_size:
                    logger.error(f"Inconsistent vocab size in token_embd.weight")
                    return False
                if reader.tensors['output.weight'].shape[0] != vocab_size:
                    logger.error(f"Inconsistent vocab size in output.weight")
                    return False

            # Add more specific checks based on your model architecture

            logger.info("Model validation passed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during model validation: {str(e)}")
            return False


