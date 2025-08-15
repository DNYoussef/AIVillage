# Compression Algorithm Notes

## Overview

This document outlines the compression algorithms integrated with the EvoMerge system for model optimization.

## Supported Algorithms

### 1. BitNet Quantization
- **Target**: 1-bit weights with 8-bit activations
- **Compression Ratio**: ~8x model size reduction
- **Accuracy Retention**: 85-95% of original performance
- **Implementation**: `src/agent_forge/compression/bitnet_enhanced.py`

### 2. VPTQ (Vector Post-Training Quantization)
- **Target**: 4-bit quantization with vector clustering
- **Compression Ratio**: ~4x model size reduction
- **Accuracy Retention**: 90-98% of original performance
- **Implementation**: `src/production/compression/vptq/`

### 3. SeedLM Compression
- **Target**: Structured pruning + quantization
- **Compression Ratio**: ~6x model size reduction
- **Accuracy Retention**: 88-96% of original performance
- **Implementation**: `src/production/compression/seedlm/`

## Integration with EvoMerge

Compression can be applied at multiple stages:

1. **Pre-merge**: Compress parent models before merging
2. **Post-merge**: Compress merged children before benchmarking
3. **Final**: Compress final Pareto-optimal models

### Configuration Example

```yaml
compression:
  enabled: true
  algorithm: "bitnet"  # or "vptq", "seedlm"
  apply_at: ["post_merge", "final"]
  target_ratio: 8.0
  accuracy_threshold: 0.85
```

## Performance Targets

| Algorithm | Compression | Latency | Memory | Accuracy |
|-----------|-------------|---------|---------|----------|
| BitNet    | 8x          | 2-3x    | 8x      | 85-95%   |
| VPTQ      | 4x          | 1.5-2x  | 4x      | 90-98%   |
| SeedLM    | 6x          | 2-2.5x  | 6x      | 88-96%   |

## Mobile Optimization

For mobile deployments, compression parameters are automatically adjusted:

- **Low Memory**: BitNet with aggressive quantization
- **Battery Saving**: SeedLM with structured pruning
- **Balanced**: VPTQ with adaptive bit allocation

## Quality Gates

Before accepting compressed models:

1. **Functional Test**: Basic inference validation
2. **Accuracy Test**: Benchmark score vs. threshold
3. **Size Test**: Compression ratio vs. target
4. **Latency Test**: Inference speed improvement

## Future Enhancements

- **Adaptive Compression**: Dynamic algorithm selection based on model characteristics
- **Progressive Compression**: Gradual compression throughout evolution
- **Hardware-Aware**: Optimization for specific target hardware
- **Knowledge Distillation**: Teacher-student compression during evolution
