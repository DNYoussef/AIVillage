# Training Service Integration Guide

## Overview

The `TrainingService` has been extracted from `unified_agent_forge_backend.py` into a focused service at `infrastructure/gateway/services/training_service.py` (~400 lines) with clean separation of concerns.

## Key Components Extracted

### Core Classes

1. **TrainingService** - Main service class with dependency injection
2. **TrainingConfig** - Configuration dataclass for all training parameters
3. **TrainingProgress** - Progress tracking dataclass
4. **ModelArtifacts** - Model metadata and artifact management

### Abstract Interfaces

1. **ProgressEmitter** - Interface for emitting training events
2. **DatasetLoader** - Interface for dataset operations  
3. **ModelTrainer** - Interface for actual model training

### Mock Implementations

1. **MockProgressEmitter** - Testing implementation
2. **MockDatasetLoader** - Testing implementation
3. **MockModelTrainer** - Testing implementation

## Features Extracted

### 1. Model Creation and Initialization
- 25M parameter Cognate models with configurable architecture
- Support for ACT (Adaptive Computation Time) 
- Support for LTM (Long-Term Memory) cross-attention
- Parameter count calculation based on configuration

### 2. Training Loop Implementation
- **GrokFast Optimization** with configurable alpha (0.98) and lambda (2.0) parameters
- Realistic training simulation with loss decay
- Learning rate scheduling
- Progress callbacks with step-by-step updates

### 3. Dataset Loading and Processing
- Multi-source dataset support (GSM8K, SVAMP, HotpotQA)
- Mixed training dataset creation
- Fallback to synthetic data on failure
- Configurable dataset sources and limits

### 4. Training Progress Tracking
- Real-time progress updates with percentage completion
- Step-by-step training metrics (loss, learning rate)
- Phase-based progress tracking (Dataset Prep, Model Training, etc.)
- Event emission for WebSocket updates (via injected emitter)

### 5. Model Checkpointing
- Comprehensive model artifact storage
- Training statistics persistence
- File path management for model weights, configs, logs
- Capability tracking and metadata

## Dependency Injection Pattern

The service uses clean dependency injection:

```python
service = TrainingService(
    progress_emitter=WebSocketProgressEmitter(),  # Your WebSocket implementation
    dataset_loader=RealDatasetLoader(),           # Your dataset implementation
    model_trainer=RealModelTrainer(),            # Your training implementation
    config=TrainingConfig(...)                   # Configuration
)
```

## Integration Points

### For WebSocket Service Integration
```python
class WebSocketProgressEmitter(ProgressEmitter):
    async def emit_progress(self, progress: TrainingProgress):
        # Send to WebSocket clients
        await websocket_manager.broadcast({
            "type": "training_update",
            "progress": progress.progress,
            "message": progress.message,
            "phase": progress.phase_name
        })
```

### For Real Training Integration  
```python
class RealModelTrainer(ModelTrainer):
    async def train_model(self, model_name, config, progress_callback):
        # Use actual PyTorch training loop
        # Call progress_callback(step, total, loss, lr) during training
        # Return real training statistics
```

## API Usage

### Start Training Session
```python
session_info = await training_service.start_training_session(
    task_id="training_001",
    training_parameters={"use_real_training": True},
    model_names=["model_1", "model_2", "model_3"]
)
```

### Execute Training Pipeline
```python
trained_models = await training_service.execute_training_pipeline("training_001")
```

### Monitor Progress
```python
status = await training_service.get_training_status("training_001")
models = await training_service.list_trained_models()
```

## Configuration Options

### GrokFast Optimization
```python
config = TrainingConfig(
    grokfast_enabled=True,
    grokfast_alpha=0.98,    # Momentum factor
    grokfast_lamb=2.0       # Scaling factor
)
```

### Model Architecture
```python
config = TrainingConfig(
    d_model=216,           # Model dimension
    n_layers=11,           # Number of layers  
    n_heads=4,             # Attention heads
    vocab_size=32000,      # Vocabulary size
    max_seq_len=4096       # Sequence length
)
```

### Training Parameters
```python
config = TrainingConfig(
    max_steps=2000,
    batch_size=2,
    learning_rate=2e-4,
    gradient_accumulation_steps=4
)
```

## Memory Coordination Key

Store and retrieve coordination information at:
```
swarm/phase2/services/training
```

## File Locations

- **Service**: `infrastructure/gateway/services/training_service.py`
- **Integration Guide**: `docs/services/training_service_integration.md` 
- **Original Source**: `infrastructure/gateway/unified_agent_forge_backend.py` (lines 300-700+)

## Next Steps for Integration

1. **WebSocket Integration**: Implement `WebSocketProgressEmitter` using existing `websocket_service.py`
2. **Real Training**: Implement `RealModelTrainer` with actual PyTorch training loops
3. **Dataset Integration**: Implement `RealDatasetLoader` with actual HuggingFace dataset loading
4. **API Integration**: Integrate with existing FastAPI endpoints in backend
5. **Testing**: Create comprehensive tests using the mock implementations

The service is designed to be a drop-in replacement for the training functionality in the unified backend while providing better testability, maintainability, and integration capabilities.