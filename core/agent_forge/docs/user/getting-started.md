# Getting Started with BitNet Phase 4

## Overview

Welcome to BitNet Phase 4! This guide will help you get started with 1-bit neural network optimization, achieving 8x memory reduction and 2-4x speedup while maintaining <10% accuracy degradation with NASA POT10 compliance.

## Quick Start (5 minutes)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/agentforge/bitnet-phase4.git
cd bitnet-phase4

# Install dependencies
pip install -r requirements.txt

# Install BitNet Phase 4
pip install -e .
```

### 2. Basic Usage

```python
from src.ml.bitnet import create_bitnet_model, optimize_bitnet_model, validate_bitnet_performance

# Create a BitNet model
model = create_bitnet_model({
    'model_size': 'base',
    'optimization_profile': 'production'
})

# Optimize the model
optimized_model, stats = optimize_bitnet_model(model, optimization_level="production")

# Validate performance
results = validate_bitnet_performance(optimized_model)
print(f"Production Ready: {results['final_report']['executive_summary']['production_ready']}")
```

### 3. Check Results

```python
# View optimization results
print(f"Memory Reduction: {stats['memory_reduction_achieved']:.1f}x")
print(f"Model Size: {stats['model_memory_mb']:.1f} MB")
print(f"Compression Ratio: {stats['compression_ratio']:.1f}x")
```

## Detailed Setup

### Prerequisites

- **Python**: 3.8 or higher
- **PyTorch**: 2.0+ (recommended for torch.compile)
- **CUDA**: 11.8+ (for GPU acceleration)
- **Memory**: 16GB+ RAM (for comprehensive benchmarking)

### Environment Setup

```bash
# Create conda environment
conda create -n bitnet-phase4 python=3.10
conda activate bitnet-phase4

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install additional dependencies
pip install numpy scipy matplotlib seaborn
pip install transformers datasets tokenizers
pip install pytest coverage black flake8

# Install development dependencies
pip install jupyter notebook ipywidgets
pip install tensorboard wandb  # For experiment tracking
```

### Development Installation

```bash
# Clone with development setup
git clone https://github.com/agentforge/bitnet-phase4.git
cd bitnet-phase4

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify installation
pytest tests/ -v

# Check code quality
black --check src/
flake8 src/
```

## Your First BitNet Model

### Step 1: Create Configuration

```python
from src.ml.bitnet import BitNetConfig, ModelSize, OptimizationProfile

# Create custom configuration
config = BitNetConfig(
    model_size=ModelSize.BASE,
    optimization_profile=OptimizationProfile.PRODUCTION
)

# Customize architecture
config.architecture.hidden_size = 768
config.architecture.num_hidden_layers = 12
config.architecture.use_1bit_quantization = True

# Set training parameters
config.training.learning_rate = 1e-4
config.training.batch_size = 32
config.training.gradient_checkpointing = True

print(f"Model will have {config.get_model_parameters_count() / 1e6:.1f}M parameters")
```

### Step 2: Initialize Model

```python
from src.ml.bitnet import BitNetModel

# Create model from configuration
model = BitNetModel(config)

# Display model statistics
stats = model.get_model_stats()
memory_info = model.get_memory_footprint()

print(f"Total Parameters: {stats['total_parameters_millions']:.1f}M")
print(f"Quantized Parameters: {stats['quantized_parameters_millions']:.1f}M")
print(f"Memory Footprint: {memory_info['model_memory_mb']:.1f} MB")
print(f"Compression Ratio: {memory_info['compression_ratio']:.1f}x")
```

### Step 3: Apply Optimizations

```python
from src.ml.bitnet.optimization import create_memory_optimizer, create_inference_optimizer

# Set up device
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create optimizers
memory_optimizer = create_memory_optimizer(device, "production")
inference_optimizer = create_inference_optimizer(device, "production")

# Apply memory optimization
with memory_optimizer.memory_optimization_context():
    model = memory_optimizer.optimize_model(model)

# Apply inference optimization
example_input = (torch.randint(0, 50000, (1, 128)).to(device),)
model = inference_optimizer.optimize_model_for_inference(model, example_input)

print("Optimizations applied successfully!")
```

### Step 4: Test Performance

```python
# Create test data
batch_size = 16
seq_len = 512
test_inputs = [
    (torch.randint(0, 50000, (batch_size, seq_len)).to(device),)
    for _ in range(10)
]

# Run performance validation
from src.ml.bitnet.validate_performance_targets import validate_bitnet_performance

results = validate_bitnet_performance(model, test_inputs, create_baseline=True)

# Display results
validation = results['final_report']
print(f"\nPerformance Validation Results:")
print(f"Overall Status: {validation['executive_summary']['overall_status']}")
print(f"Production Ready: {validation['executive_summary']['production_ready']}")
print(f"Memory Target: {'✓' if validation['executive_summary']['memory_target_achieved'] else '✗'}")
print(f"Speed Target: {'✓' if validation['executive_summary']['speed_target_achieved'] else '✗'}")
print(f"Accuracy Target: {'✓' if validation['executive_summary']['accuracy_target_achieved'] else '✗'}")
```

## Common Use Cases

### Use Case 1: Memory-Constrained Deployment

```python
# Configure for maximum memory efficiency
config = BitNetConfig(
    model_size=ModelSize.LARGE,  # Large model with aggressive compression
    optimization_profile=OptimizationProfile.PRODUCTION
)

# Override for extreme memory optimization
config.training.gradient_checkpointing = True
config.inference.enable_cpu_offload = True
config.inference.memory_fraction = 0.6

model = BitNetModel(config)
```

### Use Case 2: High-Speed Inference

```python
# Configure for maximum speed
config = BitNetConfig(
    model_size=ModelSize.BASE,
    optimization_profile=OptimizationProfile.INFERENCE
)

# Speed-specific settings
config.inference.use_torch_compile = True
config.inference.dynamic_batching = True
config.inference.use_kv_cache = True

model = BitNetModel(config)
```

### Use Case 3: NASA POT10 Compliance

```python
# Configure for defense-grade compliance
from src.ml.bitnet import ComplianceLevel

config = BitNetConfig(
    model_size=ModelSize.BASE,
    optimization_profile=OptimizationProfile.PRODUCTION
)

# Enhanced compliance settings
config.nasa_compliance.compliance_level = ComplianceLevel.DEFENSE_GRADE
config.nasa_compliance.enable_audit_trail = True
config.nasa_compliance.security_validation = True
config.nasa_compliance.performance_monitoring = True

model = BitNetModel(config)
```

## Working with Different Model Sizes

### Tiny Model (Fast Prototyping)

```python
# 6M parameters, perfect for experimentation
config = BitNetConfig(model_size=ModelSize.TINY)
model = BitNetModel(config)

print(f"Parameters: {config.get_model_parameters_count() / 1e6:.1f}M")
print(f"Memory: {config.get_memory_estimate()['model_memory_mb']:.1f} MB")
```

### Base Model (Production Ready)

```python
# 25M parameters, balanced performance
config = BitNetConfig(model_size=ModelSize.BASE)
model = BitNetModel(config)

# Apply full optimization suite
model, stats = optimize_bitnet_model(model, optimization_level="production")
```

### Large Model (Maximum Performance)

```python
# 66M parameters, high capability
config = BitNetConfig(model_size=ModelSize.LARGE)
model = BitNetModel(config)

# Validate against all targets
results = validate_bitnet_performance(model, test_inputs, create_baseline=True)
```

## Training Your First Model

### Basic Training Setup

```python
from src.ml.bitnet import BitNetTrainer

# Create trainer
trainer = BitNetTrainer(model, config)

# Prepare training data (example)
def create_dummy_batch():
    return {
        'input_ids': torch.randint(0, 50000, (config.training.batch_size, 128)),
        'attention_mask': torch.ones(config.training.batch_size, 128),
        'labels': torch.randint(0, 50000, (config.training.batch_size, 128))
    }

# Training loop
for step in range(100):
    batch = create_dummy_batch()
    stats = trainer.train_step(batch)

    if step % 10 == 0:
        print(f"Step {step}: Loss={stats['loss']:.4f}, "
              f"LR={stats['learning_rate']:.6f}")

# Get training summary
summary = trainer.get_training_summary()
print(f"Training completed: {summary['training_steps']} steps")
print(f"NASA Compliance: {summary['nasa_compliance']['overall_status']}")
```

### Advanced Training with Validation

```python
# Training with periodic validation
for epoch in range(5):
    trainer.training_stats['epoch'] = epoch

    # Training phase
    for step in range(100):
        batch = create_dummy_batch()
        stats = trainer.train_step(batch)

    # Validation phase
    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            validation_results = validate_bitnet_performance(model, test_inputs)
            production_ready = validation_results['final_report']['executive_summary']['production_ready']
            print(f"Epoch {epoch}: Production Ready = {production_ready}")
        model.train()

print("Training completed with validation!")
```

## Monitoring and Profiling

### Memory Profiling

```python
from src.ml.bitnet.profiling import create_memory_profiler

# Create profiler
profiler = create_memory_profiler(device, "comprehensive")

# Profile model inference
with profiler.profile_memory("model_inference"):
    with torch.no_grad():
        for inputs in test_inputs[:5]:
            output = model(*inputs)

# Analyze results
analysis = profiler.analyze_memory_usage()
print(f"Peak Memory: {analysis['memory_usage_summary']['peak_memory_usage_mb']:.1f} MB")
print(f"Memory Efficiency: {analysis['memory_usage_summary']['memory_efficiency']:.2f}")
print(f"8x Target Achieved: {analysis['memory_reduction_validation']['target_achieved']}")
```

### Speed Profiling

```python
from src.ml.bitnet.profiling import create_speed_profiler

# Create profiler
profiler = create_speed_profiler(device, "comprehensive")

# Input generator for testing
def input_generator(batch_size=1):
    return (torch.randint(0, 50000, (batch_size, 128)).to(device),)

# Comprehensive speed analysis
results = profiler.comprehensive_speed_analysis(model, input_generator, "bitnet_base")

# Check speed targets
speed_validation = results["speed_validation"]
print(f"Speedup: {speed_validation['speedup_ratio']:.1f}x")
print(f"2x Target: {'PASS' if speed_validation['min_target_achieved'] else 'FAIL'}")
print(f"4x Target: {'PASS' if speed_validation['optimal_target_achieved'] else 'FAIL'}")
```

## Integration with Agent Forge

### Phase 2 Integration (EvoMerge)

```python
# Load Phase 2 optimized model
phase2_checkpoint = "phase2_outputs/evomerge_model.pt"

# Integrate with BitNet
if config.phase_integration.evomerge_integration:
    # Load EvoMerge optimizations
    evomerge_state = torch.load(phase2_checkpoint, map_location=device)

    # Apply to BitNet model while preserving quantization
    for name, param in model.named_parameters():
        if name in evomerge_state and 'weight' in name:
            param.data = evomerge_state[name].data.clone()

    print("Phase 2 EvoMerge integration completed")
```

### Phase 3 Integration (Quiet-STaR)

```python
# Enable Quiet-STaR reasoning integration
if config.phase_integration.quiet_star_integration:
    # Create thought vectors (example)
    thought_vectors = torch.randn(batch_size, seq_len, config.architecture.hidden_size).to(device)

    # Forward pass with thought integration
    outputs = model(
        input_ids=test_inputs[0][0],
        thought_vectors=thought_vectors
    )

    print(f"Phase 3 Quiet-STaR integration: Output shape {outputs['logits'].shape}")
```

## Troubleshooting

### Common Issues

#### Issue 1: Out of Memory

```python
# Solution: Reduce batch size or enable CPU offload
config.training.batch_size = 8  # Reduce from 32
config.inference.enable_cpu_offload = True
config.inference.memory_fraction = 0.6
```

#### Issue 2: Slow Training

```python
# Solution: Enable optimizations
config.training.mixed_precision = True
config.training.gradient_checkpointing = True
config.architecture.use_flash_attention = True
```

#### Issue 3: Poor Accuracy

```python
# Solution: Adjust quantization settings
config.architecture.weight_quantization_bits = 2  # Instead of 1
config.training.quantization_aware_training = True
config.training.straight_through_estimator = True
```

### Debugging Tools

```python
# Check model configuration
validation_results = config.validate()
for component, issues in validation_results.items():
    if issues:
        print(f"{component}: {issues}")

# Monitor training stability
trainer_summary = trainer.get_training_summary()
nasa_compliance = trainer_summary['nasa_compliance']
if nasa_compliance['overall_status'] != 'COMPLIANT':
    print(f"Issues: {nasa_compliance['issues']}")

# Profile performance bottlenecks
bottlenecks = profiling_results['bottleneck_analysis']
for bottleneck in bottlenecks:
    print(f"Bottleneck: {bottleneck['operation']} - {bottleneck['time_percentage']:.1f}%")
```

## Next Steps

### 1. Explore Advanced Features

- **Custom Quantization**: Experiment with different bit widths
- **Hardware Optimization**: Test on different GPU architectures
- **Phase Integration**: Connect with other Agent Forge phases

### 2. Production Deployment

- **Docker Containerization**: Package for deployment
- **API Integration**: Use REST API for model serving
- **Monitoring Setup**: Implement production monitoring

### 3. Contribute

- **Bug Reports**: Submit issues on GitHub
- **Feature Requests**: Propose new optimizations
- **Code Contributions**: Submit pull requests

## Resources

- **Documentation**: Full API reference in `/docs/`
- **Examples**: Sample notebooks in `/examples/`
- **Tests**: Unit and integration tests in `/tests/`
- **Support**: support@agentforge.dev

Congratulations! You're now ready to use BitNet Phase 4 for 1-bit neural network optimization. Start with the basic examples and gradually explore advanced features as you become more familiar with the system.