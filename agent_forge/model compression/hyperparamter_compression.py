import torch
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

class HyperCompressor:
    def __init__(self, K=256, U=1000000):
        self.K = K  # Dimension of hypercube
        self.U = U  # Upper bound for theta

    def compress(self, ternary_weights):
        """Compress ternary weights using hyper-compression."""
        W = ternary_weights.flatten().numpy()
        num_groups = len(W) // self.K
        thetas = np.zeros(num_groups, dtype=np.float32)
        
        for i in range(num_groups):
            group = W[i*self.K : (i+1)*self.K]
            theta = self.optimize_theta(group)
            thetas[i] = theta
        
        return {
            'thetas': thetas,
            'original_shape': ternary_weights.shape
        }

    def optimize_theta(self, group):
        """Find optimal theta for a group of weights."""
        def loss(theta):
            reconstructed = self.reconstruct_group(theta)
            return np.sum(np.abs(reconstructed - group))
        
        result = minimize_scalar(loss, bounds=(0, self.U), method='bounded')
        return result.x

    def reconstruct_group(self, theta):
        """Reconstruct a group of weights from theta."""
        n = np.arange(1, self.K+1)
        values = theta / (np.pi + n)
        values = values - np.floor(values)  # fractional part
        return np.round(values * 2 - 1).astype(np.int8)  # map to {-1, 0, 1}

    def decompress(self, compressed_data):
        """Decompress the hyper-compressed representation back to ternary values."""
        thetas = compressed_data['thetas']
        original_shape = compressed_data['original_shape']
        
        reconstructed = np.zeros(len(thetas) * self.K, dtype=np.int8)
        for i, theta in enumerate(thetas):
            group = self.reconstruct_group(theta)
            reconstructed[i*self.K : (i+1)*self.K] = group
        
        return torch.tensor(reconstructed, dtype=torch.int8).reshape(original_shape)

def efficient_compress(compressor, ternary_weights, batch_size=1000000):
    """Compress weights efficiently using batching and KD-trees."""
    W = ternary_weights.flatten().numpy()
    num_groups = len(W) // compressor.K
    thetas = np.zeros(num_groups, dtype=np.float32)
    
    for start in range(0, num_groups, batch_size):
        end = min(start + batch_size, num_groups)
        batch = W[start*compressor.K : end*compressor.K].reshape(-1, compressor.K)
        
        # Generate points in the ergodic sequence
        points = np.array([compressor.reconstruct_group(theta) for theta in np.linspace(0, compressor.U, 10000)])
        tree = cKDTree(points)
        
        # Find nearest neighbors
        _, indices = tree.query(batch, k=1)
        thetas[start:end] = np.linspace(0, compressor.U, 10000)[indices]
    
    return {
        'thetas': thetas,
        'original_shape': ternary_weights.shape
    }

# Example usage
def main():
    # Simulate a 1.58-bit quantized model's weights
    ternary_weights = torch.randint(-1, 2, (10000000,), dtype=torch.int8)
    
    compressor = HyperCompressor(K=256, U=1000000)
    
    # Compress the ternary weights
    compressed_data = efficient_compress(compressor, ternary_weights)
    
    # Calculate compression ratio
    original_size = ternary_weights.numel() * ternary_weights.element_size()
    compressed_size = compressed_data['thetas'].size * compressed_data['thetas'].itemsize
    compression_ratio = original_size / compressed_size
    
    print(f"Original size: {original_size / 1024:.2f} KB")
    print(f"Compressed size: {compressed_size / 1024:.2f} KB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Decompress the weights
    decompressed_weights = compressor.decompress(compressed_data)
    
    # Check accuracy
    accuracy = torch.sum(ternary_weights == decompressed_weights).item() / ternary_weights.numel()
    print(f"Reconstruction accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()