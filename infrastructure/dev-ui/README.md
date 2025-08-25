# Agent Forge Developer UI

A comprehensive developer backend UI for monitoring and controlling the 8-phase Agent Forge system with real-time progress tracking and model testing capabilities.

## 🚀 Quick Start

### Windows
```bash
infrastructure\dev-ui\start-agent-forge-ui.bat
```

### Linux/macOS
```bash
./infrastructure/dev-ui/start-agent-forge-ui.sh
```

### Manual Start
```bash
cd infrastructure/dev-ui
python integration-setup.py
```

## 🌐 Access Points

Once started, access the system at:

- **🎛️ Main Developer UI**: http://localhost:8080
- **🤖 Agent Forge Controller**: http://localhost:8083
- **💬 Model Chat Interface**: http://localhost:8084
- **📡 WebSocket Manager**: http://localhost:8085

## ✨ Features

### 🎯 **Phase Control**
- **8 Phase Buttons**: Start each Agent Forge phase with real execution
  - **START COGNATE** → Creates 3 actual Cognate models (25M parameters each)
  - **START EVOMERGE** → Evolutionary model merging with fitness tracking
  - **START QUIET-STAR** → Reasoning enhancement integration
  - **START BITNET** → 1.58-bit quantization compression
  - **START FORGE** → Advanced training with Grokfast optimization
  - **START TOOL BAKING** → Tool integration and persona specialization
  - **START ADAS** → Architecture discovery and search
  - **START FINAL COMPRESSION** → Hypercompression stack

### ⚡ **Real-Time Monitoring**
- **Live Progress Bars**: Visual progress for each phase execution
- **System Metrics**: CPU, GPU, Memory usage with real-time updates
- **WebSocket Integration**: Instant updates without page refresh
- **Status Indicators**: Ready → Running → Completed → Error states

### 💬 **Model Chat Interface**
- **Immediate Testing**: Chat with models right after each phase completes
- **Model Comparison**: Side-by-side testing of different training phases
- **Real-Time Chat**: WebSocket-based instant responses
- **Session Management**: Persistent conversation history

### 📊 **Developer Dashboard**
- **Mission Control Design**: Professional AI training control center
- **Resource Monitoring**: Live system performance metrics
- **Training Metrics**: Loss curves, accuracy, performance tracking
- **Error Handling**: Comprehensive logging and debugging support

## 🏗️ Architecture

### **Service Stack**
```
Developer UI (8080) ←→ Agent Forge Controller (8083)
        ↕                           ↕
WebSocket Manager (8085) ←→ Model Chat Interface (8084)
```

### **Key Components**
- **`agent_forge_controller.py`**: Real phase execution and control
- **`model_chat.py`**: Model testing and interaction interface
- **`websocket_manager.py`**: Real-time updates and monitoring
- **`integration-setup.py`**: Service coordination and management

### **Frontend Components**
- **`agent-forge-control.html`**: Main developer interface
- **`phase-monitor.js`**: Phase execution and progress tracking
- **`model-chat.js`**: Model interaction and comparison
- **`agent-forge-ui.css`**: Professional mission control styling

## 🎯 Developer Workflow

### 1. **Start Training Phase**
Click **[START COGNATE]** → Triggers real Cognate model creation

### 2. **Monitor Progress**
Watch real-time progress bars and system metrics during training

### 3. **Test Model**
Click **[CHAT WITH COGNATE MODEL 1]** → Immediately test trained model

### 4. **Compare Results**
Progress through phases and compare model responses to see training effects

### 5. **Full Pipeline**
Execute all 8 phases with continuous testing and monitoring

## 🔧 Technical Details

### **API Endpoints**
- `POST /phases/cognate/start` - Start Cognate phase (creates 3 models)
- `POST /phases/evomerge/start` - Start EvoMerge phase
- `GET /phases/status` - Get all phase statuses
- `GET /models` - List available trained models
- `POST /chat` - Chat with trained models
- `GET /system/metrics` - System resource monitoring

### **WebSocket Channels**
- `agent_forge_phases` - Phase progress updates
- `system_metrics` - CPU/GPU/Memory monitoring
- `training_metrics` - Training loss and accuracy
- `model_updates` - Model creation and loading events

### **Real Agent Forge Integration**
- **Cognate Creation**: Calls `create_three_cognate_models()` from Agent Forge
- **EvoMerge Execution**: Uses `EvoMergePhase` and `EvoMergeConfig`
- **Progress Tracking**: Integrates with `PhaseController` and `PhaseResult`
- **Model Management**: Handles trained model artifacts and storage

## 🚨 Requirements

### **Python Dependencies**
```bash
pip install fastapi uvicorn websockets torch transformers
```

### **Agent Forge System**
- Working Agent Forge installation in `core/agent-forge/`
- Cognate pretrain system at `core/agent-forge/phases/cognate_pretrain/`
- EvoMerge system at `core/agent-forge/phases/evomerge.py`

## 🎨 Customization

### **Adding New Phases**
1. Add phase button in `agent-forge-control.html`
2. Add API endpoint in `agent_forge_controller.py`
3. Add JavaScript handler in `phase-monitor.js`

### **Styling Changes**
- Edit `agent-forge-ui.css` for visual customization
- Modify color schemes in CSS custom properties
- Adjust responsive breakpoints for different screen sizes

## 🐛 Debugging

### **Check Service Status**
```bash
python integration-setup.py status
```

### **Run Integration Tests**
```bash
python integration-setup.py test
```

### **View Logs**
- Check console output for real-time logging
- WebSocket connection status shown in UI
- API error responses displayed in developer console

## 📈 Performance

### **System Requirements**
- **CPU**: Multi-core recommended for parallel phase execution
- **Memory**: 8GB+ for model loading and training
- **GPU**: CUDA-compatible GPU recommended for training acceleration
- **Storage**: 10GB+ for model artifacts and checkpoints

### **Optimization**
- Models are loaded on-demand for memory efficiency
- WebSocket connections use efficient binary encoding
- Background tasks handle long-running training processes
- Automatic cleanup and resource management

## 🤝 Contributing

This developer UI is designed to be extended with additional Agent Forge phases and capabilities. The modular architecture supports easy integration of new training methods and monitoring features.

---

**Built with the Agent Forge ecosystem for AI Village developers**
