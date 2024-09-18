import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import requests
import json
import logging
from pydantic import BaseModel
from .gguf_utils import GGUFReader, GGUFWriter
from cmaes import CMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelReference(BaseModel):
    name: str
    template: Optional[str] = None

class MergeKitConfig(BaseModel):
    merge_method: str
    models: List[ModelReference]
    parameters: Optional[dict] = None
    custom_dir: Optional[str] = None
    evolutionary_params: Optional[dict] = None

class MergeKitMerger:
    def __init__(self, config: MergeKitConfig):
        self.config = config
        self.ollama_dir = os.path.expanduser("~/.ollama/models")
        self.custom_dir = config.custom_dir if hasattr(config, 'custom_dir') else "C:\\Users\\17175\\Desktop\\AI_Models"
        self.ollama_api_url = "http://localhost:11434/api"

    def merge(self, models: List[Dict]) -> Dict:
        logger.info("Starting model merger")
        try:
            if self.config.merge_method == "linear":
                merged_model = self._linear_merge(models)
            elif self.config.merge_method == "slerp":
                merged_model = self._slerp_merge(models)
            elif self.config.merge_method == "ps":
                merged_model = self._ps_merge(models)
            elif self.config.merge_method == "dfs":
                merged_model = self._dfs_merge(models)
            elif self.config.merge_method == "ps_dfs":
                ps_model = self._ps_merge(models)
                merged_model = self._dfs_merge([ps_model] + models)
            else:
                raise NotImplementedError(f"Merge method {self.config.merge_method} not implemented")
            
            logger.info("Model merging completed successfully")
            return merged_model
        except Exception as e:
            logger.error(f"Error during {self.config.merge_method} merge: {str(e)}")
            raise RuntimeError(f"Error during {self.config.merge_method} merge: {str(e)}")

    def _linear_merge(self, models: List[Dict]) -> Dict:
        weights = self._normalize_weights(self.config.parameters.get("weights", [1/len(models)] * len(models)))
        merged_weights, merged_metadata = self._load_and_merge_weights(models, weights, self._linear_interpolation)
        return self._save_and_create_model(merged_weights, merged_metadata, models)

    def _slerp_merge(self, models: List[Dict]) -> Dict:
        weights = self._normalize_weights(self.config.parameters.get("weights", [1/len(models)] * len(models)))
        merged_weights, merged_metadata = self._load_and_merge_weights(models, weights, self._slerp_interpolation)
        return self._save_and_create_model(merged_weights, merged_metadata, models)

    def _ps_merge(self, models: List[Dict]) -> Dict:
        evolutionary_params = self.config.evolutionary_params or {}
        generations = evolutionary_params.get("generations", 100)
        population_size = evolutionary_params.get("population_size", 20)

        optimizer = CMA(mean=np.zeros(len(models)), sigma=1.0, population_size=population_size)
        
        best_weights = None
        best_score = float('-inf')
        
        for _ in range(generations):
            solutions = []
            for _ in range(optimizer.population_size):
                weights = self._normalize_weights(optimizer.ask())
                merged_weights, merged_metadata = self._load_and_merge_weights(models, weights, self._linear_interpolation)
                score = self._evaluate_model(merged_weights, merged_metadata)
                solutions.append((weights, -score))  # Negative because CMA-ES minimizes
                
                if score > best_score:
                    best_score = score
                    best_weights = weights
            
            optimizer.tell(solutions)
        
        merged_weights, merged_metadata = self._load_and_merge_weights(models, best_weights, self._linear_interpolation)
        return self._save_and_create_model(merged_weights, merged_metadata, models)

    def _dfs_merge(self, models: List[Dict]) -> Dict:
        evolutionary_params = self.config.evolutionary_params or {}
        generations = evolutionary_params.get("generations", 100)
        population_size = evolutionary_params.get("population_size", 20)

        M = sum(len(model['model'].state_dict()) for model in models)
        I = np.ones(M * 3)  # 3 repetitions as mentioned in the paper
        W = np.eye(M)
        
        optimizer = CMA(mean=np.concatenate([I, W.flatten()]), sigma=0.1, population_size=population_size)
        
        best_solution = None
        best_score = float('-inf')
        
        for _ in range(generations):
            solutions = []
            for _ in range(optimizer.population_size):
                solution = optimizer.ask()
                I, W = solution[:M*3], solution[M*3:].reshape(M, M)
                merged_model = self._create_dfs_model(models, I, W)
                score = self._evaluate_model(merged_model)
                solutions.append((solution, -score))  # Negative because CMA-ES minimizes
                
                if score > best_score:
                    best_score = score
                    best_solution = solution
            
            optimizer.tell(solutions)
        
        I, W = best_solution[:M*3], best_solution[M*3:].reshape(M, M)
        return self._create_dfs_model(models, I, W)

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

    def _get_ollama_model_path(self, model_name: str) -> str:
        try:
            response = requests.post(f"{self.ollama_api_url}/show", json={"name": model_name})
            if response.status_code == 200:
                model_info = response.json()
                if isinstance(model_info, dict):
                    return model_info.get('modelfile', {}).get('from', '')
                else:
                    logger.warning(f"Unexpected response format from Ollama API for {model_name}")
            else:
                logger.warning(f"Failed to get model info from Ollama API for {model_name}. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error querying Ollama API: {str(e)}")
        return ""

    def _find_model_file(self, model_name: str) -> str:
        ollama_path = self._get_ollama_model_path(model_name)
        if ollama_path and os.path.exists(ollama_path):
            logger.info(f"Found model file via Ollama API at: {ollama_path}")
            return ollama_path

        possible_locations = [
            self.ollama_dir,
            os.path.join(self.ollama_dir, "blobs"),
            os.path.join(self.ollama_dir, model_name),
            self.custom_dir,
            os.path.join(self.custom_dir, model_name)
        ]

        extensions = ['.gguf', '.bin', '.model', '']
        name_formats = [model_name, model_name.replace(':', '_'), model_name.split(':')[0]]

        for location in possible_locations:
            for name_format in name_formats:
                for ext in extensions:
                    full_path = os.path.join(location, f"{name_format}{ext}")
                    if os.path.exists(full_path):
                        logger.info(f"Found model file at: {full_path}")
                        return full_path

        logger.error(f"Model file not found for: {model_name}")
        logger.error("Searched locations:")
        for location in possible_locations:
            for name_format in name_formats:
                for ext in extensions:
                    logger.error(f"  {os.path.join(location, f'{name_format}{ext}')}")
        return ""

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
        modelfile += f'PARAMETER temperature {self.config.parameters.get("temperature", 0.7)}\n'
        modelfile += f'PARAMETER num_ctx {self.config.parameters.get("num_ctx", 2048)}\n'
        modelfile += f'SYSTEM "This is a merged model named {model_name}, combining multiple models using {self.config.merge_method} interpolation of weights."\n\n'

        if 'template' in self.config.models[0].dict():
            template = self.config.models[0].dict()['template']
            modelfile += f'TEMPLATE """\n{template}\n"""\n'

        return modelfile

    def _create_ollama_model(self, model_path: str, model_name: str) -> None:
        modelfile_content = self._create_ollama_modelfile(model_path, model_name)
        modelfile_path = os.path.join(self.ollama_dir, f"{model_name}.modelfile")
        
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        os.system(f"ollama create {model_name} -f {modelfile_path}")

    def _evaluate_model(self, model: Dict) -> float:
        model_name = model['name']
        try:
            prompt = "Hello, are you working?"
            response = requests.post(
                f"{self.ollama_api_url}/generate",
                json={"model": model_name, "prompt": prompt}
            )
            
            if response.status_code != 200:
                logger.error(f"Error in API call: {response.text}")
                return float('-inf')

            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    full_response += data.get('response', '')
                    if data.get('done', False):
                        break

            coherence_score = self._check_coherence(full_response)
            return coherence_score

        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            return float('-inf')  # Return worst possible score if evaluation fails

    def _check_coherence(self, response: str) -> float:
        words = response.split()
        if len(words) < 2:
            return 0.0
        
        if "hello are you working" in response.lower():
            unique_words = set(words) - {"hello", "are", "you", "working"}
            if len(unique_words) < 2:
                return 0.0

        uniqueness = len(set(words)) / len(words)
        length_score = min(len(words) / 10, 1.0)
        
        return uniqueness * length_score

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

            required_metadata = ['general.name', 'general.architecture', 'general.file_type']
            for key in required_metadata:
                if key not in reader.metadata:
                    logger.error(f"Missing required metadata: {key}")
                    return False

            expected_tensors = ['token_embd.weight', 'output.weight']
            for tensor_name in expected_tensors:
                if tensor_name not in reader.tensors:
                    logger.error(f"Missing expected tensor: {tensor_name}")
                    return False

            vocab_size = reader.metadata.get('general.vocab_size')
            if vocab_size is not None:
                vocab_size = int(vocab_size)
                if reader.tensors['token_embd.weight'].shape[0] != vocab_size:
                    logger.error(f"Inconsistent vocab size in token_embd.weight")
                    return False
                if reader.tensors['output.weight'].shape[0] != vocab_size:
                    logger.error(f"Inconsistent vocab size in output.weight")
                    return False

            logger.info("Model validation passed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during model validation: {str(e)}")
            return False

    def _create_dfs_model(self, models: List[Dict], I: np.ndarray, W: np.ndarray) -> Dict:
        merged_weights = {}
        merged_metadata = {}
        layer_index = 0

        for i, model in enumerate(models):
            reader = GGUFReader(self._find_model_file(model['name']))
            reader.read()

            if i == 0:
                merged_metadata = reader.metadata.copy()

            for name, tensor in reader.tensors.items():
                if 'layer' in name:
                    if I[layer_index] > 0:
                        scaled_tensor = tensor * W[layer_index, i]
                        if name in merged_weights:
                            merged_weights[name] += scaled_tensor
                        else:
                            merged_weights[name] = scaled_tensor
                    layer_index += 1
                else:
                    # For non-layer tensors (like embeddings), use the first model's values
                    if i == 0:
                        merged_weights[name] = tensor

        return self._save_and_create_model(merged_weights, merged_metadata, models)



    def _apply_dare(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Implement DARE (Difference-Aware Residual Enhancement) method
        threshold = self.config.parameters.get("dare_threshold", 0.1)
        amplification = self.config.parameters.get("dare_amplification", 2.0)

        for name, tensor in weights.items():
            if name.endswith('.weight'):
                abs_diff = np.abs(tensor)
                mask = abs_diff > threshold
                weights[name] = np.where(mask, tensor * amplification, 0)

        return weights

    def _apply_ties(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Implement TIES (Token Importance-based Editing and Selection) method
        threshold = self.config.parameters.get("ties_threshold", 0.1)

        for name, tensor in weights.items():
            if name.endswith('.weight'):
                abs_tensor = np.abs(tensor)
                mask = abs_tensor > threshold
                weights[name] = np.where(mask, tensor, 0)

        return weights

    def _task_arithmetic(self, base_weights: Dict[str, np.ndarray], task_weights: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        # Implement Task Arithmetic
        task_vectors = []
        for task_weight in task_weights:
            task_vector = {name: task_weight[name] - base_weights[name] for name in base_weights}
            task_vectors.append(task_vector)

        combined_task_vector = {name: sum(task_vector[name] for task_vector in task_vectors) for name in base_weights}
        return {name: base_weights[name] + combined_task_vector[name] for name in base_weights}

    def _frankenmerge(self, models: List[Dict]) -> Dict:
        # Implement Frankenmerge (stacking different layers from multiple models)
        merged_weights = {}
        merged_metadata = {}

        for i, model in enumerate(models):
            reader = GGUFReader(self._find_model_file(model['name']))
            reader.read()

            if i == 0:
                merged_metadata = reader.metadata.copy()

            for name, tensor in reader.tensors.items():
                layer_num = int(name.split('.')[1]) if '.' in name else -1
                if layer_num == -1 or layer_num % len(models) == i:
                    merged_weights[name] = tensor

        return self._save_and_create_model(merged_weights, merged_metadata, models)
