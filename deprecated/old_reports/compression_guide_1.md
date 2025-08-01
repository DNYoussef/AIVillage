# SeedLM Compression Guide

## Overview

SeedLM is an advanced neural network compression technique that uses Linear Feedback Shift Registers (LFSR) to generate pseudo-random basis matrices for efficient weight representation. This guide covers the enhanced progressive SeedLM implementation in AI Village.

## Key Features

### 🚀 Progressive Multi-Resolution Encoding
- **Base Layer**: Fundamental compression with configurable quality levels
- **Enhancement Layers**: Progressive quality improvements through residual encoding
- **Bandwidth Adaptive**: Streams optimal quality within bandwidth constraints
- **Quality Tiers**: 5 compression levels from fast (0.1) to research quality (0.9)

### 🎯 Adaptive Block Sizing
- **Variance Analysis**: Automatically selects optimal block sizes based on weight variance
- **High Variance → Small Blocks**: Better adaptation to complex patterns
- **Low Variance → Large Blocks**: Higher compression efficiency
- **Memory Efficient**: Configurable memory limits and streaming compression

### 🔧 Multi-Scale LFSR Generation
- **Deterministic Seeding**: Reproducible compression with seed management
- **Orthogonal Bases**: Gram-Schmidt orthogonalization for better reconstruction
- **Multiple Tap Configurations**: Various LFSR patterns for diverse weight structures
- **Hardware Friendly**: Optimized for future CUDA kernel implementation

## Architecture

### Core Components

```
ProgressiveSeedLMEncoder
├── SeedLMConfig (configuration management)
├── AdaptiveBlockAnalyzer (optimal block sizing)
├── MultiScaleLFSRGenerator (basis generation)
└── Error Handling (comprehensive validation)
```

### Compression Pipeline

```
Input Weight Matrix
    ↓
Adaptive Block Analysis → Block Size Selection
    ↓
Progressive Encoding → Multiple Quality Levels
    ↓
LFSR Basis Generation → Pseudo-random Matrices
    ↓
Least Squares Fitting → Coefficient Calculation
    ↓
Quantization → 8-bit Coefficient Storage
    ↓
Metadata Packaging → Integrity & Reconstruction Info
```

## Usage

### Basic Compression

```python
from agent_forge.compression.seedlm import ProgressiveSeedLMEncoder, SeedLMConfig

# Create configuration
config = SeedLMConfig(
    compression_levels=[0.1, 0.3, 0.5, 0.7, 0.9],
    block_sizes=[4, 8, 16, 32],
    latent_dims=[2, 4, 8, 16],
    error_threshold=0.001
)

# Initialize encoder
encoder = ProgressiveSeedLMEncoder(config)

# Compress weight matrix
weight = torch.randn(512, 768)
compressed = encoder.encode(weight, compression_level=0.5)

# Decompress
reconstructed = encoder.decode(compressed)
```

### Progressive Compression

```python
# Enable progressive encoding with enhancement layers
progressive_data = encoder.encode_progressive(
    weight,
    base_quality=0.3,
    enhancement_layers=3,
    quality_increments=[0.2, 0.3, 0.2]
)

# Decode with different quality levels
base_only = encoder.decode_progressive(progressive_data, num_layers=1)
high_quality = encoder.decode_progressive(progressive_data, num_layers=4)
```

### Bandwidth-Adaptive Streaming

```python
# Get compressed data within bandwidth limit
streaming_data = encoder.get_streaming_data(
    progressive_data,
    max_bytes=100_000
)

# Still decodable with included layers
partial_reconstruction = encoder.decode_progressive(streaming_data)
```

### Integration with Existing Pipeline

```python
from agent_forge.compression import CompressionConfig, TwoStageCompressor

# Enable progressive SeedLM in pipeline
config = CompressionConfig(
    use_progressive_seedlm=True,
    seedlm_compression_level=0.7,
    seedlm_preset="balanced"
)

compressor = TwoStageCompressor(config)
compressed_layer = compressor.compress_layer(weight)
reconstructed_layer = compressor.decompress_layer(compressed_layer)
```

## Configuration

### Layer-Specific Settings

Configure different compression strategies for various layer types:

```yaml
# config/compression.yaml
layer_specific:
  "*.attention.*":
    compression_level: 0.4  # Preserve attention quality
    block_size: 16
    latent_dim: 8

  "*.mlp.*":
    compression_level: 0.7  # Higher compression for MLP
    block_size: 8
    latent_dim: 4
```

### Compression Presets

- **Fast**: 8x compression, minimal processing time
- **Balanced**: 15x compression, good quality/speed trade-off
- **Quality**: 25x compression, optimized for accuracy
- **Research**: 30x compression, maximum ratio with experimental features

## Advanced Features

### Error Handling & Verification

```python
# Enable integrity verification
compressed = encoder.encode(
    weight,
    enable_verification=True,
    compression_level=0.6
)

# Verify during decompression
try:
    reconstructed = encoder.decode(compressed, verify=True)
except SeedLMVerificationError:
    print("Data corruption detected!")
```

### Memory Management

```python
# Stream large models in chunks
large_weight = torch.randn(4096, 8192)
compressed = encoder.encode(
    large_weight,
    streaming=True,  # Enable memory-efficient processing
    compression_level=0.5
)
```

### Performance Monitoring

```python
# Access compression statistics
stats = encoder.compression_stats
print(f"Average compression time: {stats['total_time'] / stats['total_compressions']:.3f}s")
print(f"Average compression ratio: {stats['average_ratio']:.2f}x")
```

## Performance Characteristics

### Compression Ratios by Layer Type

| Layer Type | Typical Ratio | Quality Setting |
|------------|---------------|----------------|
| Embeddings | 5-8x | High (0.3-0.4) |
| Attention | 8-12x | Medium (0.4-0.6) |
| MLP/Linear | 12-20x | Variable (0.5-0.8) |
| Convolution | 10-15x | Medium (0.4-0.7) |

### Speed Benchmarks

- **Small layers** (< 1M params): ~0.1s compression
- **Medium layers** (1-10M params): ~1-5s compression
- **Large layers** (10M+ params): ~10-30s compression
- **Decompression**: ~5-10x faster than compression

## Limitations & Trade-offs

### Current Limitations
- **Compression Speed**: Slower than BitNet/VPTQ for large layers
- **Memory Overhead**: Basis generation requires temporary storage
- **Quality Loss**: Aggressive compression (0.7+) can impact accuracy
- **Hardware Acceleration**: CUDA kernels not yet implemented

### Quality vs Compression Trade-offs
- **Low Compression (0.1-0.3)**: < 1% accuracy loss, 3-8x compression
- **Medium Compression (0.4-0.6)**: 1-3% accuracy loss, 8-15x compression
- **High Compression (0.7-0.9)**: 3-8% accuracy loss, 15-25x compression

## Integration Examples

### With BitNet Pipeline

```python
# BitNet + SeedLM combination
config = CompressionConfig(
    bitnet_finetune=True,
    use_progressive_seedlm=True,
    seedlm_compression_level=0.6
)

# Compression order: BitNet → SeedLM → VPTQ
```

### Model-Specific Optimization

```python
# Transformer-specific settings
transformer_config = SeedLMConfig()
transformer_config.block_sizes = [8, 16]  # Optimize for attention patterns
transformer_config.latent_dims = [4, 8]   # Balanced representation

# CNN-specific settings
cnn_config = SeedLMConfig()
cnn_config.block_sizes = [4, 8, 16]  # Handle diverse kernel sizes
cnn_config.compression_levels = [0.4, 0.6, 0.8]  # Spatial redundancy
```

## Troubleshooting

### Common Issues

**High Reconstruction Error**
- Reduce compression level (0.7 → 0.5)
- Increase latent dimensions
- Use smaller block sizes for complex weights

**Slow Compression**
- Reduce number of candidate seeds
- Use "fast" preset configuration
- Enable streaming for large weights

**Memory Issues**
- Set memory limits in config
- Enable streaming compression
- Reduce batch sizes

**Integration Errors**
- Verify PyTorch version compatibility
- Check for missing dependencies (numpy, scipy)
- Ensure proper tensor shapes and types

### Performance Optimization

```python
# Fast compression settings
config = SeedLMConfig(
    compression_levels=[0.3, 0.5, 0.7],  # Fewer levels
    block_sizes=[8, 16],                  # Fewer block sizes
    latent_dims=[2, 4],                   # Smaller latent space
)

# Reduce candidate seeds for speed
compressor = SeedLMCompressor(num_seeds=32)  # vs default 256
```

## Future Developments

### Planned Enhancements
- **CUDA Acceleration**: Custom kernels for 10x speed improvements
- **Learned Bases**: Adaptive basis optimization during training
- **Model-Aware Compression**: Architecture-specific optimization strategies
- **Real-time Streaming**: Dynamic quality adjustment based on bandwidth

### Research Directions
- **Hybrid Quantization**: Combination with other compression methods
- **Attention-Aware Compression**: Specialized handling for transformer layers
- **Differentiable Compression**: End-to-end training with compression loss
- **Hardware Co-design**: Custom silicon for LFSR operations

## References

- [SeedLM Paper](https://arxiv.org/abs/example) - Original algorithm description
- [LFSR Theory](https://en.wikipedia.org/wiki/Linear-feedback_shift_register) - Mathematical background
- [AI Village Documentation](../README.md) - Project overview
- [Compression Benchmarks](benchmark_results.md) - Detailed performance analysis
