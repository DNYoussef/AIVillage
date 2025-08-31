# Agent Training Service Separation

## Overview

The training services have been completely separated into two distinct systems:

1. **Cognate Pretraining Service** (`training_service.py`) - Specialized for Cognate model pretraining
2. **Agent Forge Training Service** (`agent_forge_training_service.py`) - Specialized for agent behavior training

## Service Separation Details

### Cognate Pretraining Service (Original)
**Location**: `infrastructure/gateway/services/training_service.py`

**Responsibilities**:
- 25M parameter Cognate model creation and training
- GrokFast optimization for Cognate architectures  
- GSM8K/HotpotQA dataset handling for pretraining
- ACT (Adaptive Computation Time) training
- LTM (Long-Term Memory) cross-attention training
- Large-scale model pretraining workflows

**Key Features**:
- Model architectures: `d_model=216, n_layers=11, n_heads=4`
- Parameter count: ~25M parameters
- Datasets: GSM8K, SVAMP, HotpotQA for reasoning tasks
- Advanced optimizations: GrokFast, ACT, LTM
- Focus: Foundation model pretraining

### Agent Forge Training Service (New)
**Location**: `infrastructure/gateway/services/agent_forge_training_service.py`

**Responsibilities**:
- General agent behavior training
- Task-specific fine-tuning workflows
- Non-Cognate model architectures
- Agent coordination and communication training
- Skill acquisition and adaptation
- Multi-agent collaboration training

**Key Features**:
- Model architectures: `hidden_size=128, num_layers=4, num_heads=8`
- Parameter count: ~10K-100K parameters (agent-sized)
- Training modes: Behavior adaptation, task specialization, coordination
- Agent architectures: Hierarchical, planning, reactive, hybrid agents
- Focus: Agent behavior and coordination

## Configuration Differences

### Cognate Service Configuration
```python
@dataclass
class TrainingConfig:
    max_steps: int = 2000
    d_model: int = 216
    n_layers: int = 11
    n_heads: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 4096
    
    # GrokFast optimization
    grokfast_alpha: float = 0.98
    grokfast_lamb: float = 2.0
    
    # ACT/LTM parameters
    act_threshold: float = 0.99
    d_mem: int = 216
    mem_capacity: int = 4096
    
    # Pretraining datasets
    dataset_sources: List[str] = ["GSM8K", "SVAMP", "HotpotQA"]
```

### Agent Service Configuration
```python
@dataclass  
class AgentTrainingConfig:
    max_episodes: int = 1000
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 8
    max_context_length: int = 512
    
    # Agent-specific parameters
    training_mode: AgentTrainingMode
    agent_architecture: AgentArchitecture
    exploration_rate: float = 0.1
    adaptation_threshold: float = 0.8
    
    # Multi-agent settings
    max_agents: int = 4
    coordination_strategy: str = "hierarchical"
    
    # Environment configuration
    environment_types: List[str] = ["simulation", "interactive"]
    communication_protocols: List[str] = ["direct", "broadcast"]
```

## Interface Differences

### Cognate Service API
- `start_training_session()` - Creates Cognate pretraining session
- `execute_training_pipeline()` - Runs complete pretraining pipeline  
- `_prepare_datasets()` - Downloads GSM8K/HotpotQA datasets
- `_train_models()` - Trains 25M parameter Cognate models
- `_create_model_artifacts()` - Creates pretraining artifacts

### Agent Service API
- `start_agent_training_session()` - Creates agent training session
- `execute_agent_training_pipeline()` - Runs agent training pipeline
- `_prepare_training_environments()` - Sets up agent environments
- `_train_agents()` - Trains agent behaviors
- `_train_multi_agent_coordination()` - Trains coordination protocols
- `_create_agent_artifacts()` - Creates agent training artifacts

## Progress Tracking Differences

### Cognate Service Progress
```python
@dataclass
class TrainingProgress:
    progress: float
    message: str
    step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
```

### Agent Service Progress
```python
@dataclass
class AgentTrainingProgress:
    progress: float
    message: str
    episode: Optional[int] = None
    total_episodes: Optional[int] = None
    reward: Optional[float] = None
    success_rate: Optional[float] = None
    coordination_score: Optional[float] = None
```

## Artifact Differences

### Cognate Model Artifacts
- Model focus: Reasoning, memory integration, adaptive computation
- Training stats: Loss metrics, validation accuracy, convergence
- Capabilities: GrokFast acceleration, ACT computation, LTM memory
- Files: `pytorch_model.bin`, `config.json`, training logs

### Agent Training Artifacts  
- Agent focus: Coordination, task execution, communication
- Training stats: Reward metrics, success rates, coordination scores
- Capabilities: Multi-agent communication, task delegation, adaptation
- Files: `agent_weights.pt`, `agent_config.json`, `behavior_policy.json`

## Complete Separation Achieved

✅ **No Shared Code**: Both services have completely separate implementations  
✅ **Different Interfaces**: Distinct APIs and configuration systems  
✅ **Different Progress Tracking**: Separate progress metrics and reporting  
✅ **Different Resource Requirements**: Cognate needs 25M params, Agent needs <100K params  
✅ **No Shared Training Logic**: Completely separate training algorithms and utilities  

## Usage Examples

### Cognate Pretraining
```python
# Large-scale foundation model pretraining
cognate_service = TrainingService(...)
await cognate_service.start_training_session(
    "cognate_pretrain_001",
    {"focus": "reasoning", "datasets": ["GSM8K", "HotpotQA"]}
)
```

### Agent Behavior Training  
```python
# Agent coordination and behavior training
agent_service = AgentForgeTrainingService(...)
await agent_service.start_agent_training_session(
    "agent_train_001",
    {"focus": "coordination", "mode": "multi_agent_collaboration"}
)
```

The services are now completely independent and specialized for their respective use cases.