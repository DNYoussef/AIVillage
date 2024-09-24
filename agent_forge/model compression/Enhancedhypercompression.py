import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import math
import io
from matplotlib import transforms
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, DataCollatorWithPadding, Trainer, TrainingArguments
import time
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import psutil
import GPUtil
import nltk
from sklearn.decomposition import DictionaryLearning
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from paraphrase_identification.source_code_in_theano.data import load_data
from paraphrase_identification.source_code_in_theano.test_lstm import generate_feature_vector, build_classifier_and_test
from sklearn.neural_network import MLPClassifier

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class TernaryDictionaryLearner:
    def __init__(self, n_components=256, patch_size=16):
        self.n_components = n_components
        self.patch_size = patch_size
        self.dictionary = None
    
    def extract_patches(self, weights):
        # Extract overlapping patches from the weight matrix
        patches = []
        for i in range(0, weights.shape[0] - self.patch_size + 1, self.patch_size // 2):
            for j in range(0, weights.shape[1] - self.patch_size + 1, self.patch_size // 2):
                patch = weights[i:i+self.patch_size, j:j+self.patch_size].flatten()
                patches.append(patch)
        return np.array(patches)
    
    def fit(self, weights):
        patches = self.extract_patches(weights)
        dl = DictionaryLearning(n_components=self.n_components, fit_algorithm='cd', transform_algorithm='lasso_lars')
        dl.fit(patches)
        self.dictionary = self.ternarize_dictionary(dl.components_)
    
    def ternarize_dictionary(self, dictionary):
        # Convert dictionary elements to ternary values
        return np.sign(dictionary) * (np.abs(dictionary) > 0.5)
    
    def encode(self, weights):
        patches = self.extract_patches(weights)
        codes = []
        for patch in patches:
            code = np.zeros(self.n_components)
            residual = patch
            for _ in range(3):  # Limit to 3 non-zero coefficients for sparsity
                correlations = np.abs(np.dot(self.dictionary, residual))
                best_atom = np.argmax(correlations)
                coef = np.dot(self.dictionary[best_atom], residual)
                code[best_atom] += np.sign(coef)
                residual -= np.sign(coef) * self.dictionary[best_atom]
            codes.append(code)
        return csr_matrix(np.array(codes))
    
    def decode(self, codes, original_shape):
        reconstructed = np.zeros(original_shape)
        count = np.zeros(original_shape)
        idx = 0
        for i in range(0, original_shape[0] - self.patch_size + 1, self.patch_size // 2):
            for j in range(0, original_shape[1] - self.patch_size + 1, self.patch_size // 2):
                patch = np.dot(codes[idx].toarray(), self.dictionary).reshape(self.patch_size, self.patch_size)
                reconstructed[i:i+self.patch_size, j:j+self.patch_size] += patch
                count[i:i+self.patch_size, j:j+self.patch_size] += 1
                idx += 1
        return np.sign(reconstructed / np.maximum(count, 1))

def detect_hardware() -> Dict[str, any]:
    """Detect and return relevant hardware information."""
    hardware_info = {}

    # CPU Information
    cpu_info = psutil.cpu_freq()
    hardware_info['cpu_cores'] = psutil.cpu_count(logical=False)
    hardware_info['cpu_threads'] = psutil.cpu_count(logical=True)
    hardware_info['cpu_freq'] = cpu_info.current if cpu_info else None

    # Memory Information
    memory = psutil.virtual_memory()
    hardware_info['total_memory'] = memory.total
    hardware_info['available_memory'] = memory.available

    # Cache Information (simplified approximation)
    hardware_info['l1_cache'] = 32 * 1024  # Assume 32KB L1 cache
    hardware_info['l2_cache'] = 256 * 1024  # Assume 256KB L2 cache
    hardware_info['l3_cache'] = 8 * 1024 * 1024  # Assume 8MB L3 cache

    # GPU Information
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assume using the first GPU
        hardware_info['gpu_name'] = gpu.name
        hardware_info['gpu_memory'] = gpu.memoryTotal
        hardware_info['gpu_free_memory'] = gpu.memoryFree

    return hardware_info

def get_gpu_info() -> Dict[str, int]:
    """Get relevant GPU information."""
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assume using the first GPU
        return {
            'max_threads_per_block': 1024,  # Default value, may not be accurate
            'max_shared_memory_per_block': 49152,  # Default value in bytes, may not be accurate
            'warp_size': 32  # Default warp size for most GPUs
        }
    else:
        return {
            'max_threads_per_block': 256,  # Fallback CPU value
            'max_shared_memory_per_block': 32768,  # Fallback CPU value
            'warp_size': 1  # Not applicable for CPU
        }

def encode_chunk(chunk: List[int]) -> Tuple[bytes, int]:
    """Encode a chunk of ternary values (-1, 0, 1) into byte-aligned format."""
    encoded = []
    bit_buffer = []

    def flush_bits():
        while len(bit_buffer) >= 8:
            byte = int(''.join(map(str, bit_buffer[:8])), 2)
            encoded.append(byte)
            del bit_buffer[:8]

    for value in chunk:
        if value == 1:
            bit_buffer.extend([0, 1])
        elif value == -1:
            bit_buffer.extend([1, 0])
        else:  # value == 0
            bit_buffer.extend([0, 0])
        flush_bits()

    # Pad the last byte if necessary
    if bit_buffer:
        while len(bit_buffer) < 8:
            bit_buffer.append(0)
        flush_bits()

    return bytes(encoded), len(chunk)

def decode_chunk(encoded_chunk: bytes, chunk_length: int) -> List[int]:
    """Decode a byte-aligned chunk back into ternary values."""
    decoded = []
    for byte in encoded_chunk:
        for i in range(3, -1, -1):  # Process 4 values per byte
            two_bits = (byte >> (i * 2)) & 0b11
            if two_bits == 0b01:
                decoded.append(1)
            elif two_bits == 0b10:
                decoded.append(-1)
            else:
                decoded.append(0)
            if len(decoded) == chunk_length:
                break
        if len(decoded) == chunk_length:
            break
    return decoded

def encode_matrix(matrix: np.ndarray, chunk_size: int) -> List[Tuple[bytes, int]]:
    """Encode an entire matrix, chunking it for parallel processing."""
    flat = matrix.ravel()
    encoded_chunks = []
    for i in range(0, len(flat), chunk_size):
        chunk = flat[i:i+chunk_size].tolist()
        encoded_chunk, chunk_length = encode_chunk(chunk)
        encoded_chunks.append((encoded_chunk, chunk_length))
    return encoded_chunks

def decode_matrix(encoded_chunks: List[Tuple[bytes, int]], original_shape: Tuple[int, ...], chunk_size: int) -> np.ndarray:
    """Decode the entire matrix from its encoded chunks."""
    decoded_flat = []
    for encoded_chunk, chunk_length in encoded_chunks:
        decoded_flat.extend(decode_chunk(encoded_chunk, chunk_length))
    return np.array(decoded_flat).reshape(original_shape)

def benchmark_chunk_size_single(args: Tuple[np.ndarray, int, int]) -> Tuple[int, float]:
    """Benchmark a single chunk size."""
    matrix, chunk_size, num_trials = args
    total_time = 0
    for _ in range(num_trials):
        start_time = time.time()
        encoded = encode_matrix(matrix, chunk_size)
        decoded = decode_matrix(encoded, matrix.shape, chunk_size)
        end_time = time.time()
        total_time += end_time - start_time
    avg_time = total_time / num_trials
    return chunk_size, avg_time

def parallel_benchmark_chunk_size(matrix: np.ndarray, chunk_sizes: List[int], num_trials: int = 3) -> int:
    """Benchmark different chunk sizes in parallel and return the best performing one."""
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(benchmark_chunk_size_single,
                           [(matrix, size, num_trials) for size in chunk_sizes])
    return min(results, key=lambda x: x[1])[0]

def get_optimal_chunk_size(matrix: np.ndarray, hardware_info: dict) -> int:
    """Determine the optimal chunk size based on matrix size, hardware info, and GPU considerations."""
    matrix_size = matrix.size
    l3_cache = hardware_info.get('l3_cache', 8 * 1024 * 1024)  # Use L3 cache size for CPU

    # Define a range of chunk sizes to test
    base_chunk_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    # Filter chunk sizes based on matrix size and cache size
    valid_chunk_sizes = [size for size in base_chunk_sizes if size <= matrix_size and size * 2 <= l3_cache]

    # GPU considerations
    gpu_info = get_gpu_info()
    max_threads = gpu_info['max_threads_per_block']
    shared_memory = gpu_info['max_shared_memory_per_block']
    warp_size = gpu_info['warp_size']

    # Adjust valid chunk sizes based on GPU constraints
    valid_chunk_sizes = [size for size in valid_chunk_sizes if size <= max_threads and size % warp_size == 0]

    # Ensure chunk sizes don't exceed shared memory capacity (assuming 4 bytes per value)
    valid_chunk_sizes = [size for size in valid_chunk_sizes if size * 4 <= shared_memory]

    if not valid_chunk_sizes:
        return warp_size  # Default to warp size if no valid sizes

    return parallel_benchmark_chunk_size(matrix, valid_chunk_sizes)

class AdaptiveChunkManager:
    def __init__(self, initial_hardware_info: Dict[str, any]):
        self.hardware_info = initial_hardware_info
        self.chunk_size_cache = {}
        self.update_interval = 60  # Update hardware info every 60 seconds
        self.last_update_time = time.time()
        self.gpu_info = get_gpu_info()

    def update_hardware_info(self):
        """Update hardware info if the update interval has passed."""
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            self.hardware_info = detect_hardware()
            self.gpu_info = get_gpu_info()
            self.last_update_time = current_time

    def get_chunk_size(self, matrix: np.ndarray) -> int:
        """Get the optimal chunk size, adapting to current system state and GPU considerations."""
        self.update_hardware_info()

        matrix_shape = matrix.shape
        if matrix_shape in self.chunk_size_cache:
            return self.chunk_size_cache[matrix_shape]

        optimal_size = get_optimal_chunk_size(matrix, self.hardware_info)
        self.chunk_size_cache[matrix_shape] = optimal_size
        return optimal_size

    def encode_matrix_adaptive(self, matrix: np.ndarray) -> List[Tuple[bytes, int]]:
        """Encode matrix with adaptive chunk size."""
        chunk_size = self.get_chunk_size(matrix)
        return encode_matrix(matrix, chunk_size)

    def decode_matrix_adaptive(self, encoded_chunks: List[Tuple[bytes, int]], original_shape: Tuple[int, ...]) -> np.ndarray:
        """Decode matrix with adaptive chunk size."""
        chunk_size = self.chunk_size_cache.get(original_shape, 64)  # Default to 64 if not found
        return decode_matrix(encoded_chunks, original_shape, chunk_size)

def extract_weights(model):
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.detach().cpu().numpy())
    return weights

def enhanced_hypercompress(weights):
    compressed_weights = []
    chunk_manager = AdaptiveChunkManager(detect_hardware())
    learner = TernaryDictionaryLearner()
    
    for weight_matrix in weights:
        # Convert to ternary values: -1, 0, 1
        ternary_weights = np.sign(weight_matrix)
        ternary_weights[np.abs(weight_matrix) < 0.5] = 0
        
        # Learn dictionary and encode
        learner.fit(ternary_weights)
        codes = learner.encode(ternary_weights)
        
        # Compress codes using adaptive chunk manager
        compressed_codes = chunk_manager.encode_matrix_adaptive(codes.toarray())
        compressed_weights.append((compressed_codes, ternary_weights.shape))
    
    return compressed_weights, learner.dictionary

def compress_bitnet(sequence):
    # Compress the ternary sequence into a bit representation
    # where 01 represents 1, 10 represents -1, and 00 represents 0
    compressed = []
    for value in sequence:
        if value == 1:
            compressed.extend([0, 1])
        elif value == -1:
            compressed.extend([1, 0])
        else:  # value == 0
            compressed.extend([0, 0])
    
    # Pack bits into bytes
    byte_array = bytearray()
    for i in range(0, len(compressed), 8):
        byte = 0
        for j in range(8):
            if i + j < len(compressed):
                byte |= compressed[i + j] << (7 - j)
        byte_array.append(byte)
    
    return byte_array

def decompress_bitnet(compressed_data, dictionary):
    chunk_manager = AdaptiveChunkManager(detect_hardware())
    learner = TernaryDictionaryLearner(n_components=dictionary.shape[0])
    learner.dictionary = dictionary
    
    decompressed_weights = []
    for compressed_codes, original_shape in compressed_data:
        # Decompress codes using adaptive chunk manager
        decompressed_codes = chunk_manager.decode_matrix_adaptive(compressed_codes, (original_shape[0], learner.n_components))
        
        # Reconstruct ternary weights using the learned dictionary
        reconstructed = learner.decode(csr_matrix(decompressed_codes), original_shape)
        decompressed_weights.append(reconstructed)
    
    return decompressed_weights

def calculate_compression_ratio(original_weights, compressed_weights, dictionary, accuracy_change):
    original_size = sum(w.size * w.itemsize for w in original_weights)
    compressed_size = sum(c[0].size * c[0].itemsize for c in compressed_weights) + dictionary.size * dictionary.itemsize
    compression_ratio = original_size / compressed_size
    return compression_ratio, accuracy_change


def plot_weight_distribution(weights, title):
    plt.figure(figsize=(10, 5))
    plt.hist(weights.flatten(), bins=100)
    plt.title(title)
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.show()

def decompress_weights(compressed_weights, original_shapes):
    decompressed = []
    for compressed, shape in zip(compressed_weights, original_shapes):
        decompressed_flat = np.array([value for value, count in compressed for _ in range(count)])
        decompressed.append(decompressed_flat.reshape(shape))
    return decompressed

def load_paraphrase_data():
    sent1_train, sent2_train, sent1_test, sent2_test, label_train, label_test = load_data()
    return sent1_train, sent2_train, sent1_test, sent2_test, label_train, label_test

def evaluate_model(model, tokenizer, dataset):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataset:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(model.device)
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item()
    return total_loss / len(dataset)

def evaluate_paraphrase_identification(model, tokenizer, sent1, sent2, labels):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for s1, s2, label in zip(sent1, sent2, labels):
            inputs = tokenizer(s1, s2, return_tensors='pt', padding=True, truncation=True).to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted = torch.argmax(logits, dim=1).item()
            correct += (predicted == label)
            total += 1
    accuracy = correct / total
    return accuracy

    # Generate embeddings for train and test data
    first_train_sentences = generate_sentence_embeddings(sent1_train_indices)
    second_train_sentences = generate_sentence_embeddings(sent2_train_indices)
    first_test_sentences = generate_sentence_embeddings(sent1_test_indices)
    second_test_sentences = generate_sentence_embeddings(sent2_test_indices)

    # Generate feature vectors
    feature_vector_train = generate_feature_vector(first_train_sentences, second_train_sentences)
    feature_vector_test = generate_feature_vector(first_test_sentences, second_test_sentences)

    # Build classifier and evaluate
    classifier = MLPClassifier()
    build_classifier_and_test(feature_vector_train, label_train, feature_vector_test, label_test, classifier)

def main():
    # Load paraphrase data
    sent1_train, sent2_train, sent1_test, sent2_test, label_train, label_test = load_paraphrase_data()

    # Evaluate original model
    print("Evaluating original model on paraphrase identification task...")
    original_accuracy = evaluate_paraphrase_identification(model, tokenizer, sent1_test, sent2_test, label_test)
    print(f"Original model accuracy: {original_accuracy:.4f}")
    
    model_name = "HF1BitLLM/Llama3-8B-1.58-100B-tokens"
    tokenizer_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    print("Loading model and tokenizer...")
    config = AutoConfig.from_pretrained(model_name)
    
    if hasattr(config, 'quantization_config'):
        del config.quantization_config

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("\nModel Architecture:")
    print(model)
    print("\nParameter Data Types:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")

    print("\nExtracting model weights...")
    model_weights = extract_weights(model)

    print("Applying enhanced hypercompression...")
    compressed_weights, dictionary = enhanced_hypercompress(model_weights)

    compression_ratio, accuracy_change = calculate_compression_ratio(model_weights, compressed_weights, dictionary, accuracy_change)
    print(f"\nCompression Ratio: {compression_ratio:.2f}x")
    print(f"Accuracy Change: {accuracy_change:.4f}")

    print("\nDecompressing weights...")
    decompressed_weights = decompress_bitnet(compressed_weights, dictionary)

    print("Updating model with decompressed weights...")
    with torch.no_grad():
        for (name, param), decompressed in zip(model.named_parameters(), decompressed_weights):
            if 'weight' in name:
                param.data = torch.from_numpy(decompressed).to(param.device).to(param.dtype)

    plot_weight_distribution(model_weights[0], "Original Weight Distribution")
    plot_weight_distribution(decompressed_weights[0], "Decompressed Weight Distribution")

    # Evaluate compressed model
    print("Evaluating compressed model on paraphrase identification task...")
    compressed_accuracy = evaluate_paraphrase_identification(model, tokenizer, sent1_test, sent2_test, label_test)
    print(f"Compressed model accuracy: {compressed_accuracy:.4f}")

    # Calculate accuracy change
    accuracy_change = compressed_accuracy - original_accuracy
    print(f"Accuracy change: {accuracy_change:.4f}")