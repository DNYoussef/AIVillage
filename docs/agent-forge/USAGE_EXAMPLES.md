# Agent Forge Pipeline Usage Examples

## Quick Start Guide

### Basic Pipeline Usage

```python
import asyncio
from pathlib import Path
from unified_pipeline import UnifiedPipeline, UnifiedConfig

# Create basic configuration
config = UnifiedConfig(
    base_models=["microsoft/DialoGPT-small"],
    output_dir=Path("./my_agent_output"),
    device="cuda" if torch.cuda.is_available() else "cpu",

    # Enable desired phases
    enable_evomerge=True,
    enable_quietstar=True,
    enable_training=True,
    enable_dspy_optimization=True,

    # Fast settings for development
    evomerge_generations=10,
    evomerge_population_size=6,
    training_steps=1000,

    wandb_project=None,  # Set to "my-project" to enable tracking
)

# Create and run pipeline
async def main():
    pipeline = UnifiedPipeline(config)
    result = await pipeline.run_pipeline()

    if result.success:
        print(f"✓ Pipeline completed successfully!")
        print(f"  Phases completed: {result.metrics['phases_completed']}")
        print(f"  Total duration: {result.duration_seconds:.1f}s")

        # Access the trained model
        final_model = result.model

        # Model is ready for inference or further fine-tuning
        return final_model
    else:
        print(f"✗ Pipeline failed: {result.error}")
        return None

# Run the pipeline
model = asyncio.run(main())
```

Enable DSPy prompt optimization globally by setting `enable_dspy_optimization=True` in
`UnifiedConfig`. This applies DSPy-optimized prompts during phases like Quiet-STaR and
tool/persona baking. Set the flag to `False` to disable the optimization.

### Production Configuration

```python
# High-performance configuration for production
production_config = UnifiedConfig(
    base_models=[
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
        "Qwen/Qwen2-1.5B-Instruct",
    ],
    output_dir=Path("./production_output"),
    checkpoint_dir=Path("./production_checkpoints"),
    device="cuda",

    # Enable full 8-phase pipeline
    enable_cognate=False,  # Skip if import issues
    enable_evomerge=True,
    enable_quietstar=True,
    enable_initial_compression=True,
    enable_training=True,
    enable_tool_baking=True,
    enable_adas=True,
    enable_final_compression=True,

    # Production settings
    evomerge_generations=50,
    evomerge_population_size=8,
    training_steps=100000,
    batch_size=32,
    learning_rate=1e-4,

    # Enable advanced features
    grokfast_enabled=True,
    edge_control_enabled=True,
    self_model_enabled=True,
    dream_enabled=True,
    enable_dspy_optimization=True,

    # Monitoring
    wandb_project="agent-forge-production",
    log_interval=50,
    checkpoint_interval=5000,
)
```

### Custom Phase Configuration

```python
# Configure specific phases only
custom_config = UnifiedConfig(
    base_models=["microsoft/DialoGPT-small"],
    output_dir=Path("./custom_experiment"),

    # Only run evolutionary optimization and training
    enable_cognate=False,
    enable_evomerge=True,
    enable_quietstar=False,
    enable_initial_compression=False,
    enable_training=True,
    enable_tool_baking=False,
    enable_adas=False,
    enable_final_compression=False,

    # Custom EvoMerge settings
    evomerge_generations=25,
    evomerge_population_size=12,
    evomerge_techniques=["linear", "slerp", "ties", "dare"],

    # Custom training settings
    training_steps=50000,
    batch_size=16,
    grokfast_enabled=True,
    grokfast_ema_alpha=0.99,

    # Edge-of-chaos training
    edge_control_enabled=True,
    target_success_range=(0.6, 0.8),
)
```

## Performance Benchmarking

### Running Benchmarks

```python
from tests.agent_forge_benchmark import AgentForgeBenchmark
import asyncio

async def benchmark_pipeline():
    # Create benchmark system
    benchmark = AgentForgeBenchmark(
        output_dir=Path("./benchmark_results")
    )

    # Define test configurations
    test_configs = [
        # Fast test configuration
        UnifiedConfig(
            enable_evomerge=True,
            evomerge_generations=5,
            evomerge_population_size=4,
            wandb_project=None,
        ),

        # Medium configuration
        UnifiedConfig(
            enable_evomerge=True,
            enable_quietstar=True,
            enable_training=True,
            evomerge_generations=10,
            training_steps=1000,
            wandb_project=None,
        ),
    ]

    # Run comprehensive benchmark
    results = await benchmark.run_comprehensive_benchmark(
        configs=test_configs,
        include_swe_bench=False,  # Use mock SWE-Bench for now
        include_performance=True,
        include_stress_test=True
    )

    # Analyze results
    for i, result in enumerate(results):
        print(f"\nConfiguration {i+1} Results:")
        print(f"  SWE-Bench Rate: {result.swe_bench_solve_rate:.1%}")
        print(f"  Token Reduction: {result.token_reduction_percent:.1%}")
        print(f"  Speed Multiplier: {result.speed_multiplier:.1f}x")
        print(f"  Pipeline Success: {result.pipeline_success_rate:.1%}")

        # Check if targets met
        targets_met = [
            result.swe_bench_target_met,
            result.token_reduction_target_met,
            result.speed_target_met
        ]
        print(f"  Targets Met: {sum(targets_met)}/3")

    return results

# Run benchmarks
benchmark_results = asyncio.run(benchmark_pipeline())
```

### Performance Monitoring

```python
import wandb

# Enable Weights & Biases tracking
config = UnifiedConfig(
    # ... other settings ...
    wandb_project="agent-forge-experiment",
    wandb_entity="your-wandb-team",
    log_interval=10,
)

# The pipeline will automatically log:
# - Phase completion metrics
# - Model performance metrics
# - Resource usage
# - Training curves
# - Compression ratios
```

## Advanced Usage Patterns

### Resume from Checkpoint

```python
async def resume_pipeline():
    config = UnifiedConfig(
        # ... configuration ...
        checkpoint_dir=Path("./existing_checkpoints")
    )

    pipeline = UnifiedPipeline(config)

    # Resume from specific phase
    result = await pipeline.run_pipeline(resume_from="ForgeTrainingPhase")

    return result
```

### Custom Phase Integration

```python
from core.phase_controller import PhaseController, PhaseResult

class CustomPhase(PhaseController):
    """Example custom phase implementation."""

    def __init__(self, config):
        super().__init__(config)
        self.custom_setting = config.get('custom_setting', 'default')

    async def run(self, model):
        # Custom processing logic
        start_time = time.time()

        # Validate input
        if not self.validate_input_model(model):
            return self.create_failure_result(
                model, "Input validation failed"
            )

        # Process model
        try:
            # Your custom logic here
            processed_model = self.custom_processing(model)

            duration = time.time() - start_time

            return self.create_success_result(
                processed_model,
                metrics={
                    "processing_time": duration,
                    "custom_metric": self.calculate_custom_metric(),
                },
                duration=duration
            )

        except Exception as e:
            return self.create_failure_result(
                model, str(e), time.time() - start_time
            )

    def custom_processing(self, model):
        """Implement your custom processing logic."""
        # Example: add custom layers, modify architecture, etc.
        return model

    def calculate_custom_metric(self):
        """Calculate custom performance metrics."""
        return 42.0

# Integrate custom phase
class CustomUnifiedPipeline(UnifiedPipeline):
    def _initialize_phases_safe(self):
        phases = super()._initialize_phases_safe()

        # Add custom phase
        custom_config = {'custom_setting': 'value'}
        phases.append(("CustomPhase", CustomPhase(custom_config)))

        return phases
```

### Model Export and Deployment

```python
async def train_and_export():
    # Train model with pipeline
    config = UnifiedConfig(
        base_models=["microsoft/DialoGPT-small"],
        enable_evomerge=True,
        enable_training=True,
        enable_final_compression=True,
    )

    pipeline = UnifiedPipeline(config)
    result = await pipeline.run_pipeline()

    if result.success:
        final_model = result.model

        # Export for deployment
        export_path = Path("./exported_model")
        export_path.mkdir(exist_ok=True)

        # Save PyTorch model
        torch.save(final_model.state_dict(), export_path / "model.pt")

        # Save configuration
        with open(export_path / "config.json", 'w') as f:
            json.dump(result.artifacts['config'], f, indent=2, default=str)

        # Save metrics
        with open(export_path / "metrics.json", 'w') as f:
            json.dump(result.metrics, f, indent=2, default=str)

        print(f"✓ Model exported to {export_path}")

        # Convert to Hugging Face format (if applicable)
        try:
            from transformers import AutoModel, AutoTokenizer
            # Implementation depends on your specific model architecture
            print("Consider converting to Hugging Face format for easier deployment")
        except ImportError:
            pass

        return export_path

    return None
```

## Performance Optimization Tips

### 1. Memory Optimization

```python
# Optimize for memory usage
memory_optimized_config = UnifiedConfig(
    batch_size=8,  # Smaller batch size
    enable_final_compression=True,  # Enable compression
    device="cuda",

    # Use gradient checkpointing
    grokfast_enabled=True,

    # Optimize phase selection
    enable_initial_compression=True,  # Early compression
)
```

### 2. Speed Optimization

```python
# Optimize for speed
speed_optimized_config = UnifiedConfig(
    # Reduce phase complexity
    evomerge_generations=20,  # Fewer generations
    evomerge_population_size=6,  # Smaller population
    training_steps=50000,  # Fewer training steps

    # Use efficient techniques
    evomerge_techniques=["linear", "slerp"],  # Fast techniques

    # Parallel processing
    device="cuda",
    batch_size=32,  # Larger batch size if memory allows
)
```

### 3. Quality Optimization

```python
# Optimize for model quality
quality_optimized_config = UnifiedConfig(
    # Enable all phases
    enable_evomerge=True,
    enable_quietstar=True,
    enable_training=True,
    enable_tool_baking=True,
    enable_adas=True,

    # High-quality settings
    evomerge_generations=100,  # More generations
    evomerge_population_size=12,  # Larger population
    training_steps=200000,  # Longer training

    # Advanced techniques
    grokfast_enabled=True,
    edge_control_enabled=True,
    self_model_enabled=True,
    dream_enabled=True,

    # Quality monitoring
    wandb_project="high-quality-experiment",
)
```

## Error Handling and Debugging

### Common Issues and Solutions

```python
async def robust_pipeline_execution():
    config = UnifiedConfig(
        # ... your config ...
    )

    try:
        pipeline = UnifiedPipeline(config)

        # Check if phases are available
        if not pipeline.phases:
            print("⚠ No phases available - check imports")
            return None

        print(f"Pipeline created with {len(pipeline.phases)} phases:")
        for phase_name, _ in pipeline.phases:
            print(f"  ✓ {phase_name}")

        # Run pipeline with error handling
        result = await pipeline.run_pipeline()

        if result.success:
            print("✓ Pipeline completed successfully!")
            return result.model
        else:
            print(f"✗ Pipeline failed: {result.error}")

            # Check partial results
            if result.artifacts and 'partial_phase_results' in result.artifacts:
                print("Partial results available:")
                for phase_result in result.artifacts['partial_phase_results']:
                    status = "✓" if phase_result['success'] else "✗"
                    print(f"  {status} {phase_result['phase_name']}")
                    if not phase_result['success'] and phase_result.get('error'):
                        print(f"    Error: {phase_result['error']}")

            return None

    except ImportError as e:
        print(f"Import error: {e}")
        print("Check that all required packages are installed")
        return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Usage
model = asyncio.run(robust_pipeline_execution())
```

### Debugging Tips

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check phase imports individually**:
   ```python
   try:
       from phases.evomerge import EvoMergePhase
       print("✓ EvoMerge available")
   except ImportError as e:
       print(f"✗ EvoMerge not available: {e}")
   ```

3. **Use minimal configurations for testing**:
   ```python
   debug_config = UnifiedConfig(
       base_models=["microsoft/DialoGPT-small"],
       enable_evomerge=True,  # Only enable one phase
       evomerge_generations=2,  # Minimal settings
       evomerge_population_size=4,
       wandb_project=None,
   )
   ```

## Integration Examples

### With Hugging Face Models

```python
from transformers import AutoModel, AutoTokenizer

async def integrate_with_hf():
    # Load base model from Hugging Face
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    # Configure pipeline to work with HF model
    config = UnifiedConfig(
        base_models=[model_name],
        # ... other settings ...
    )

    # Run pipeline
    pipeline = UnifiedPipeline(config)
    result = await pipeline.run_pipeline()

    if result.success:
        # Enhanced model ready for use
        enhanced_model = result.model

        # Use with tokenizer for inference
        inputs = tokenizer("Hello, how are you?", return_tensors="pt")
        # outputs = enhanced_model(**inputs)  # Depending on compatibility

        return enhanced_model, tokenizer

    return None, None
```

### With FastAPI Deployment

```python
from fastapi import FastAPI
import torch

app = FastAPI()

# Global model storage
global_model = None
global_tokenizer = None

@app.on_event("startup")
async def load_model():
    global global_model, global_tokenizer

    # Train model with Agent Forge
    config = UnifiedConfig(
        base_models=["microsoft/DialoGPT-small"],
        enable_evomerge=True,
        enable_training=True,
    )

    pipeline = UnifiedPipeline(config)
    result = await pipeline.run_pipeline()

    if result.success:
        global_model = result.model
        # global_tokenizer = ... # Load appropriate tokenizer
        print("✓ Model loaded successfully")
    else:
        print("✗ Model loading failed")

@app.post("/generate")
async def generate_text(prompt: str):
    if global_model is None:
        return {"error": "Model not loaded"}

    # Implement generation logic
    # This depends on your specific model architecture
    return {"response": "Generated response based on enhanced model"}

# Run with: uvicorn your_app:app --reload
```

This comprehensive usage guide covers the major patterns for using the Agent Forge pipeline effectively. The fixed import structure should resolve the primary issues identified in testing.
