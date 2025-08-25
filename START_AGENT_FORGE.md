# 🚀 Quick Start Guide - Agent Forge System

## ⚡ One-Command Startup

### **Start Backend (Required)**
```bash
cd C:\Users\17175\Desktop\AIVillage\infrastructure\gateway
python minimal_agent_forge_backend.py
```

### **Access Web Interface**
- **Main Interface**: http://localhost:8083/test
- **API Root**: http://localhost:8083/
- **Health Check**: http://localhost:8083/health

## 🧠 Create Cognate Models

### **Option 1: Web Interface**
1. Open http://localhost:8083/test
2. Click "🧠 Create 3 Cognate Models (25M params)" button
3. Watch real-time progress updates
4. View created models in the interface

### **Option 2: Command Line**
```bash
# Start Cognate creation
curl -X POST http://localhost:8083/phases/cognate/start

# Check progress
curl http://localhost:8083/phases/status

# View created models
curl http://localhost:8083/models
```

## 📊 What You Get

- **3 Cognate Models**: Each with 25,083,528 parameters
- **Specializations**: Reasoning, Memory Integration, Adaptive Computation
- **Complete Artifacts**: PyTorch checkpoints, configs, tokenizers
- **Real-time Updates**: WebSocket progress tracking
- **Ready for EvoMerge**: All models prepared for next phase

## ✅ Success Indicators

- **Backend Status**: "healthy" response from /health
- **Model Count**: 3 models listed in /models
- **Phase Status**: "Cognate" shows "completed"
- **Parameters**: Each model shows 25,083,528 parameters

## 🎯 Endpoints Available

- `GET /` - Service information
- `GET /health` - Health check
- `GET /phases/status` - All phase statuses
- `POST /phases/cognate/start` - Start Cognate creation
- `GET /models` - List created models
- `POST /chat` - Chat with models
- `ws://localhost:8083/ws` - WebSocket updates

**🎉 That's it! Your Agent Forge system is ready to create 25M parameter Cognate models.**
