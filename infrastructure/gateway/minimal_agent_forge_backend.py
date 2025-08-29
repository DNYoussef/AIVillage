#!/usr/bin/env python3
"""
Minimal Agent Forge Backend - Working Implementation

Provides a minimal but complete backend for Agent Forge without complex imports.
Includes:
- REST API endpoints for phase management
- WebSocket support for real-time updates
- Simulated Cognate model creation (3x 25M parameter models)
- Simple chat interface

Ports:
- 8080: Main gateway server
- 8083: API server
- 8085: WebSocket server
"""

import asyncio
from datetime import datetime
import json
import logging
import time
from typing import Any
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Agent Forge Backend - Minimal",
    description="Complete working backend for Agent Forge system",
    version="1.0.0",
)

# CORS middleware
# SECURITY: Secure CORS configuration - NO WILDCARDS
import os
cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000")
secure_origins = [origin.strip() for origin in cors_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=secure_origins,  # SECURITY: Environment-controlled, no wildcards
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "Authorization", "X-Requested-With"],
)

# Global state
active_phases: dict[str, Any] = {}
created_models: list[dict[str, Any]] = []
is_running: dict[str, bool] = {}
websocket_connections: set[WebSocket] = set()


# Data models
class ChatRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    message: str


class PhaseStartRequest(BaseModel):
    phase_name: str = "Cognate"
    force_restart: bool = False


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self.subscriptions: dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[client_id] = {"websocket": websocket, "channels": {"global", "phases", "system"}}
        logger.info(f"Client {client_id} connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, client_id: str):
        self.active_connections.discard(websocket)
        self.subscriptions.pop(client_id, None)
        logger.info(f"Client {client_id} disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict, channel: str = "global"):
        disconnected = set()
        for client_id, sub_info in self.subscriptions.items():
            if channel in sub_info["channels"]:
                try:
                    await sub_info["websocket"].send_json(message)
                except:
                    disconnected.add((sub_info["websocket"], client_id))

        # Clean up disconnected clients
        for ws, client_id in disconnected:
            self.disconnect(ws, client_id)


manager = ConnectionManager()


# Simulation functions
async def simulate_cognate_training():
    """Simulate realistic Cognate model training with progress updates."""
    phase_name = "Cognate"
    is_running[phase_name] = True

    try:
        # Initialize phase status
        active_phases[phase_name] = {
            "phase_name": phase_name,
            "status": "initializing",
            "progress": 0.0,
            "message": "Setting up consolidated cognate implementation...",
            "start_time": datetime.now().isoformat(),
            "models_completed": 0,
            "total_models": 3,
            "current_step": "Loading model factory",
            "estimated_time_remaining": 90,
        }

        # Broadcast initial status
        await manager.broadcast(
            {"type": "phase_update", "phase_name": phase_name, **active_phases[phase_name]}, "phases"
        )

        # Training simulation steps with realistic progress
        steps = [
            (0.05, "Loading consolidated cognate implementation...", "Importing core modules"),
            (0.10, "Validating 25,083,528 parameter architecture...", "Architecture validation"),
            (0.15, "Setting up training configurations...", "Configuration setup"),
            (0.25, "Creating cognate_foundation_1 (reasoning focus)...", "Model 1 creation"),
            (0.35, "Training cognate_foundation_1 layers...", "Model 1 training"),
            (0.45, "Creating cognate_foundation_2 (memory integration)...", "Model 2 creation"),
            (0.55, "Training cognate_foundation_2 layers...", "Model 2 training"),
            (0.70, "Creating cognate_foundation_3 (adaptive computation)...", "Model 3 creation"),
            (0.85, "Training cognate_foundation_3 layers...", "Model 3 training"),
            (0.95, "Finalizing model artifacts and metadata...", "Artifact generation"),
            (1.0, "Successfully created 3 x 25M parameter models!", "Completed"),
        ]

        # Update phase to running
        active_phases[phase_name]["status"] = "running"

        for i, (progress, message, step) in enumerate(steps):
            if not is_running.get(phase_name, False):
                break

            # Update progress
            active_phases[phase_name]["progress"] = progress
            active_phases[phase_name]["message"] = message
            active_phases[phase_name]["current_step"] = step
            active_phases[phase_name]["estimated_time_remaining"] = max(0, (len(steps) - i - 1) * 8)

            # Update models completed
            if progress >= 0.35:
                completed = min(3, int((progress - 0.2) * 4))
                active_phases[phase_name]["models_completed"] = completed

            # Broadcast update
            await manager.broadcast(
                {"type": "phase_update", "phase_name": phase_name, **active_phases[phase_name]}, "phases"
            )

            logger.info(f"[{progress*100:5.1f}%] {message}")
            await asyncio.sleep(3 if progress < 0.9 else 1)  # Realistic timing

        # Create model entries on completion
        if is_running.get(phase_name, False):
            focuses = ["reasoning", "memory_integration", "adaptive_computation"]
            new_models = []

            for i, focus in enumerate(focuses, 1):
                model = {
                    "model_id": f"cognate_foundation_{i}",
                    "model_name": f"Cognate Foundation Model {i}",
                    "phase_name": "Cognate",
                    "parameter_count": 25083528,
                    "created_at": datetime.now().isoformat(),
                    "training_status": "completed",
                    "focus": focus,
                    "architecture": {"layers": 24, "hidden_size": 1024, "attention_heads": 16, "vocab_size": 32000},
                    "artifacts": {
                        "config_path": f"core/agent-forge/phases/cognate_pretrain/cognate_25m_models/cognate_foundation_{i}/config.json",
                        "weights_path": f"core/agent-forge/phases/cognate_pretrain/cognate_25m_models/cognate_foundation_{i}/pytorch_model.bin",
                        "tokenizer_path": f"core/agent-forge/phases/cognate_pretrain/cognate_25m_models/cognate_foundation_{i}/tokenizer.json",
                        "metadata_path": f"core/agent-forge/phases/cognate_pretrain/cognate_25m_models/cognate_foundation_{i}/metadata.json",
                    },
                    "performance_metrics": {
                        "training_loss": round(2.45 - (i * 0.12), 2),
                        "validation_perplexity": round(12.3 - (i * 0.8), 1),
                        "training_time_minutes": 28 + (i * 2),
                    },
                }
                new_models.append(model)
                created_models.append(model)

            # Final status update
            active_phases[phase_name]["status"] = "completed"
            active_phases[phase_name]["models_completed"] = 3
            active_phases[phase_name]["estimated_time_remaining"] = 0
            active_phases[phase_name]["artifacts"] = {
                "models_created": len(new_models),
                "total_parameters": sum(m["parameter_count"] for m in new_models),
                "output_directory": "core/agent-forge/phases/cognate_pretrain/cognate_25m_models",
                "ready_for_evomerge": True,
                "model_ids": [m["model_id"] for m in new_models],
            }

            await manager.broadcast(
                {
                    "type": "phase_completed",
                    "phase_name": phase_name,
                    "models_created": new_models,
                    **active_phases[phase_name],
                },
                "phases",
            )

            logger.info(f"Cognate phase completed successfully with {len(new_models)} models")

    except Exception as e:
        logger.exception(f"Cognate phase failed: {e}")
        active_phases[phase_name]["status"] = "error"
        active_phases[phase_name]["message"] = f"Error: {str(e)}"
        active_phases[phase_name]["current_step"] = "Error occurred"

        await manager.broadcast(
            {"type": "phase_error", "phase_name": phase_name, **active_phases[phase_name]}, "phases"
        )

    finally:
        is_running[phase_name] = False


# API Endpoints


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Agent Forge Backend - Minimal",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "phases": "/phases/status",
            "cognate_start": "/phases/cognate/start",
            "models": "/models",
            "chat": "/chat",
            "websocket": "ws://localhost:8083/ws",
        },
        "models_created": len(created_models),
        "websocket_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "agent_forge_minimal_backend",
        "uptime_seconds": time.time(),
        "active_phases": len(active_phases),
        "stored_models": len(created_models),
        "websocket_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/phases/status")
async def get_phase_status():
    """Get status of all Agent Forge phases."""
    phases_list = []

    # Add Cognate phase if not exists
    if "Cognate" not in active_phases:
        active_phases["Cognate"] = {
            "phase_name": "Cognate",
            "status": "ready",
            "progress": 0.0,
            "message": "Ready to create 25M parameter foundation models",
            "models_completed": 0,
            "total_models": 3,
            "current_step": "Awaiting start command",
        }

    # Add other phases in ready state
    other_phases = ["EvoMerge", "Quiet-STaR", "BitNet", "Forge Training", "Tool Baking", "ADAS", "Final Compression"]

    for phase_name in other_phases:
        if phase_name not in active_phases:
            active_phases[phase_name] = {
                "phase_name": phase_name,
                "status": "ready",
                "progress": 0.0,
                "message": f"Ready to start {phase_name} phase",
                "models_completed": 0,
                "total_models": 1 if phase_name != "EvoMerge" else 3,
                "current_step": "Waiting for prerequisites",
            }

    for phase_name, phase_data in active_phases.items():
        phases_list.append({"phase_name": phase_name, **phase_data})

    return {
        "phases": phases_list,
        "total_phases": len(phases_list),
        "active_phases": len([p for p in phases_list if p["status"] in ["running", "initializing"]]),
        "completed_phases": len([p for p in phases_list if p["status"] == "completed"]),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/phases/cognate/start")
async def start_cognate_phase(background_tasks: BackgroundTasks, request: PhaseStartRequest = None):
    """Start the Cognate model creation phase."""
    phase_name = "Cognate"

    # Check if already running
    if is_running.get(phase_name, False) and not (request and request.force_restart):
        raise HTTPException(
            status_code=400, detail=f"{phase_name} phase is already running. Use force_restart=true to restart."
        )

    # Force stop if restart requested
    if request and request.force_restart:
        is_running[phase_name] = False
        await asyncio.sleep(1)  # Give time for cleanup

    # Start the phase
    background_tasks.add_task(simulate_cognate_training)

    return {
        "success": True,
        "message": "Cognate phase started - Creating 3x 25M parameter foundation models",
        "phase_name": phase_name,
        "total_models": 3,
        "target_params_per_model": 25083528,
        "estimated_duration_minutes": 3,
        "started_at": datetime.now().isoformat(),
    }


@app.get("/phases/cognate/stop")
async def stop_cognate_phase():
    """Stop the running Cognate phase."""
    phase_name = "Cognate"

    if not is_running.get(phase_name, False):
        raise HTTPException(status_code=400, detail=f"{phase_name} phase is not running")

    is_running[phase_name] = False

    if phase_name in active_phases:
        active_phases[phase_name]["status"] = "stopped"
        active_phases[phase_name]["message"] = "Phase stopped by user request"
        active_phases[phase_name]["current_step"] = "Stopped"

        await manager.broadcast(
            {"type": "phase_stopped", "phase_name": phase_name, **active_phases[phase_name]}, "phases"
        )

    return {
        "success": True,
        "message": f"{phase_name} phase stopped successfully",
        "phase_name": phase_name,
        "stopped_at": datetime.now().isoformat(),
    }


@app.get("/models")
async def get_models():
    """Get all created models."""
    total_params = sum(m.get("parameter_count", 0) for m in created_models)

    return {
        "models": created_models,
        "total_models": len(created_models),
        "total_parameters": total_params,
        "models_by_phase": {"Cognate": len([m for m in created_models if m.get("phase_name") == "Cognate"])},
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/chat")
async def chat_with_model(request: ChatRequest):
    """Chat interface with created models."""
    # Find the requested model
    model = None
    for m in created_models:
        if m["model_id"] == request.model_id:
            model = m
            break

    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_id}' not found. Available models: {[m['model_id'] for m in created_models]}",
        )

    if model["training_status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_id}' is not ready for inference (status: {model['training_status']})",
        )

    # Simulate model response based on focus area
    focus_responses = {
        "reasoning": "I excel at logical reasoning and problem-solving. Let me analyze your question step by step.",
        "memory_integration": "I specialize in memory integration and context retention. I can help maintain conversation context.",
        "adaptive_computation": "I focus on adaptive computation and dynamic processing. I can adjust my responses based on complexity.",
    }

    focus = model.get("focus", "general")
    focus_intro = focus_responses.get(focus, "I'm a general-purpose foundation model.")

    # Generate response
    response = (
        f"Hello! I'm {model['model_name']} ({model['parameter_count']:,} parameters). {focus_intro}\n\n"
        f"Regarding your message: '{request.message}' - I'd be happy to help! "
        f"As a {focus} specialist, I can provide detailed assistance in my area of expertise."
    )

    return {
        "success": True,
        "model_id": request.model_id,
        "model_name": model["model_name"],
        "user_message": request.message,
        "model_response": response,
        "response_metadata": {
            "focus_area": focus,
            "parameter_count": model["parameter_count"],
            "response_time_ms": 127,
            "confidence_score": 0.94,
        },
        "timestamp": datetime.now().isoformat(),
    }


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    client_id = str(uuid.uuid4())

    try:
        await manager.connect(websocket, client_id)

        # Send welcome message
        await websocket.send_json(
            {
                "type": "connection",
                "status": "connected",
                "client_id": client_id,
                "message": "Connected to Agent Forge Backend WebSocket",
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Send initial status
        await websocket.send_json(
            {
                "type": "initial_status",
                "active_phases": list(active_phases.keys()),
                "created_models": len(created_models),
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                elif message.get("type") == "get_status":
                    # Send current status
                    phases_status = await get_phase_status()
                    models_status = await get_models()

                    await websocket.send_json(
                        {
                            "type": "status_update",
                            "phases": phases_status["phases"],
                            "models": models_status["models"],
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON format"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.exception(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(websocket, client_id)


# Test HTML interface
TEST_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Forge Backend Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { background: white; margin: 10px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 5px 0; border-radius: 4px; }
        .status.ready { background: #e7f3ff; border-left: 4px solid #0066cc; }
        .status.running { background: #fff3e0; border-left: 4px solid #ff9800; }
        .status.completed { background: #e8f5e8; border-left: 4px solid #4caf50; }
        .status.error { background: #ffebee; border-left: 4px solid #f44336; }
        .progress-bar { background: #f0f0f0; height: 20px; border-radius: 10px; overflow: hidden; margin: 5px 0; }
        .progress-fill { background: #4caf50; height: 100%; transition: width 0.3s ease; }
        button { background: #0066cc; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #0052a3; }
        button:disabled { background: #cccccc; cursor: not-allowed; }
        .log { background: #1e1e1e; color: #00ff00; font-family: monospace; padding: 10px; border-radius: 4px; max-height: 300px; overflow-y: auto; }
        .model-card { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 4px; }
        .model-card h4 { margin-top: 0; color: #0066cc; }
        #chatArea { border: 1px solid #ddd; height: 200px; overflow-y: auto; padding: 10px; background: white; }
        input, select, textarea { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Agent Forge Backend Test Interface</h1>

        <div class="section">
            <h2>System Status</h2>
            <div id="systemStatus">Loading...</div>
        </div>

        <div class="section">
            <h2>Phase Management</h2>
            <button onclick="startCognatePhase()">Start Cognate Phase</button>
            <button onclick="stopCognatePhase()">Stop Cognate Phase</button>
            <button onclick="refreshStatus()">Refresh Status</button>
            <div id="phaseStatus"></div>
        </div>

        <div class="section">
            <h2>Created Models</h2>
            <button onclick="refreshModels()">Refresh Models</button>
            <div id="modelsStatus"></div>
        </div>

        <div class="section">
            <h2>Model Chat Interface</h2>
            <select id="modelSelect">
                <option value="">Select a model...</option>
            </select>
            <br>
            <textarea id="chatInput" placeholder="Enter your message..." rows="3" style="width: 100%;"></textarea>
            <br>
            <button onclick="sendChatMessage()">Send Message</button>
            <div id="chatArea"></div>
        </div>

        <div class="section">
            <h2>Real-time WebSocket Log</h2>
            <button onclick="connectWebSocket()">Connect WebSocket</button>
            <button onclick="disconnectWebSocket()">Disconnect</button>
            <div class="log" id="websocketLog"></div>
        </div>
    </div>

    <script>
        let ws = null;

        function log(message) {
            const logDiv = document.getElementById('websocketLog');
            const timestamp = new Date().toLocaleTimeString();
            logDiv.innerHTML += `[${timestamp}] ${message}\n`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }

        function connectWebSocket() {
            if (ws) return;

            ws = new WebSocket('ws://localhost:8083/ws');

            ws.onopen = () => {
                log('🔗 WebSocket connected');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                log(`📨 ${data.type}: ${JSON.stringify(data, null, 2)}`);

                if (data.type === 'phase_update') {
                    refreshStatus();
                } else if (data.type === 'phase_completed') {
                    refreshStatus();
                    refreshModels();
                }
            };

            ws.onclose = () => {
                log('❌ WebSocket disconnected');
                ws = null;
            };

            ws.onerror = (error) => {
                log(`⚠️ WebSocket error: ${error}`);
            };
        }

        function disconnectWebSocket() {
            if (ws) {
                ws.close();
                ws = null;
                log('🔌 WebSocket disconnected by user');
            }
        }

        async function startCognatePhase() {
            try {
                const response = await fetch('/phases/cognate/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const data = await response.json();

                if (response.ok) {
                    log(`✅ Started Cognate phase: ${data.message}`);
                    setTimeout(refreshStatus, 1000);
                } else {
                    log(`❌ Failed to start: ${data.detail}`);
                }
            } catch (error) {
                log(`⚠️ Error: ${error.message}`);
            }
        }

        async function stopCognatePhase() {
            try {
                const response = await fetch('/phases/cognate/stop');
                const data = await response.json();

                if (response.ok) {
                    log(`⏹️ Stopped Cognate phase: ${data.message}`);
                    refreshStatus();
                } else {
                    log(`❌ Failed to stop: ${data.detail}`);
                }
            } catch (error) {
                log(`⚠️ Error: ${error.message}`);
            }
        }

        async function refreshStatus() {
            try {
                const [systemResponse, phaseResponse] = await Promise.all([
                    fetch('/health'),
                    fetch('/phases/status')
                ]);

                const systemData = await systemResponse.json();
                const phaseData = await phaseResponse.json();

                // Update system status
                document.getElementById('systemStatus').innerHTML = `
                    <div class="status ready">
                        <strong>Service:</strong> ${systemData.service}<br>
                        <strong>Status:</strong> ${systemData.status}<br>
                        <strong>Models Created:</strong> ${systemData.stored_models}<br>
                        <strong>WebSocket Connections:</strong> ${systemData.websocket_connections}
                    </div>
                `;

                // Update phase status
                let phaseHtml = '';
                phaseData.phases.forEach(phase => {
                    const progressPercent = (phase.progress * 100).toFixed(1);
                    phaseHtml += `
                        <div class="status ${phase.status}">
                            <strong>${phase.phase_name}</strong> - ${phase.status}<br>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${progressPercent}%"></div>
                            </div>
                            Progress: ${progressPercent}% | ${phase.message}<br>
                            Models: ${phase.models_completed}/${phase.total_models}
                            ${phase.current_step ? `| Step: ${phase.current_step}` : ''}
                        </div>
                    `;
                });

                document.getElementById('phaseStatus').innerHTML = phaseHtml;

            } catch (error) {
                log(`⚠️ Error refreshing status: ${error.message}`);
            }
        }

        async function refreshModels() {
            try {
                const response = await fetch('/models');
                const data = await response.json();

                let modelsHtml = `<p><strong>Total Models:</strong> ${data.total_models} | <strong>Total Parameters:</strong> ${data.total_parameters.toLocaleString()}</p>`;

                if (data.models.length === 0) {
                    modelsHtml += '<p>No models created yet. Start the Cognate phase to create models.</p>';
                } else {
                    data.models.forEach(model => {
                        modelsHtml += `
                            <div class="model-card">
                                <h4>${model.model_name} (${model.model_id})</h4>
                                <p><strong>Focus:</strong> ${model.focus}</p>
                                <p><strong>Parameters:</strong> ${model.parameter_count.toLocaleString()}</p>
                                <p><strong>Status:</strong> ${model.training_status}</p>
                                <p><strong>Created:</strong> ${new Date(model.created_at).toLocaleString()}</p>
                                ${model.performance_metrics ? `
                                    <p><strong>Performance:</strong> Loss: ${model.performance_metrics.training_loss}, Perplexity: ${model.performance_metrics.validation_perplexity}</p>
                                ` : ''}
                            </div>
                        `;
                    });
                }

                document.getElementById('modelsStatus').innerHTML = modelsHtml;

                // Update model selector
                const select = document.getElementById('modelSelect');
                select.innerHTML = '<option value="">Select a model...</option>';
                data.models.forEach(model => {
                    if (model.training_status === 'completed') {
                        select.innerHTML += `<option value="${model.model_id}">${model.model_name}</option>`;
                    }
                });

            } catch (error) {
                log(`⚠️ Error refreshing models: ${error.message}`);
            }
        }

        async function sendChatMessage() {
            const modelId = document.getElementById('modelSelect').value;
            const message = document.getElementById('chatInput').value;

            if (!modelId || !message) {
                alert('Please select a model and enter a message.');
                return;
            }

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_id: modelId, message: message })
                });

                const data = await response.json();

                if (response.ok) {
                    const chatArea = document.getElementById('chatArea');
                    chatArea.innerHTML += `
                        <div style="margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 4px;">
                            <strong>You:</strong> ${message}
                        </div>
                        <div style="margin: 10px 0; padding: 10px; background: #e7f3ff; border-radius: 4px;">
                            <strong>${data.model_name}:</strong><br>${data.model_response}
                        </div>
                    `;
                    chatArea.scrollTop = chatArea.scrollHeight;
                    document.getElementById('chatInput').value = '';
                } else {
                    alert(`Chat error: ${data.detail}`);
                }

            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }

        // Initialize
        window.onload = () => {
            refreshStatus();
            refreshModels();
            connectWebSocket();
        };

        // Refresh every 5 seconds
        setInterval(() => {
            refreshStatus();
        }, 5000);
    </script>
</body>
</html>
"""


@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Test HTML interface."""
    return TEST_HTML


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("AGENT FORGE MINIMAL BACKEND STARTING")
    print("=" * 80)
    print("API Server: http://localhost:8083")
    print("WebSocket: ws://localhost:8083/ws")
    print("Test Interface: http://localhost:8083/test")
    print("")
    print("Key Endpoints:")
    print("  POST /phases/cognate/start - Start Cognate training")
    print("  GET  /phases/status        - Get phase status")
    print("  GET  /models              - Get created models")
    print("  POST /chat                - Chat with models")
    print("  GET  /health              - Health check")
    print("=" * 80)

    try:
        uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info", access_log=True)
    except KeyboardInterrupt:
        print("\nBackend server stopped by user")
    except Exception as e:
        print(f"Backend server error: {e}")
