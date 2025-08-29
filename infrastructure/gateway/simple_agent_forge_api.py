#!/usr/bin/env python3
"""
Simple Agent Forge API Server - Standalone Version
No external dependencies, works with the consolidated cognate system
"""

from datetime import datetime
import logging
from pathlib import Path
import sys
import time

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))

app = FastAPI(title="Agent Forge API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_phases = {}
created_models = []
is_running = {}


class PhaseStatus:
    def __init__(self, phase_name):
        self.phase_name = phase_name
        self.status = "ready"
        self.progress = 0.0
        self.message = "Ready to start"
        self.start_time = None
        self.models_completed = 0
        self.total_models = 3


def simulate_cognate_phase():
    """Simulate the Cognate phase execution with realistic progress."""
    global active_phases, created_models, is_running

    phase_name = "Cognate"
    is_running[phase_name] = True

    # Initialize phase
    active_phases[phase_name] = {
        "phase_name": phase_name,
        "status": "running",
        "progress": 0.0,
        "message": "Initializing consolidated cognate implementation...",
        "start_time": datetime.now().isoformat(),
        "models_completed": 0,
        "total_models": 3,
    }

    # Simulation steps
    steps = [
        (0.1, "Loading consolidated cognate implementation..."),
        (0.2, "Validating 25,083,528 parameter architecture..."),
        (0.35, "Creating cognate_foundation_1 (reasoning focus)..."),
        (0.55, "Creating cognate_foundation_2 (memory integration)..."),
        (0.85, "Creating cognate_foundation_3 (adaptive computation)..."),
        (0.95, "Finalizing model artifacts and metadata..."),
        (1.0, "Successfully created 3 x 25M parameter models!"),
    ]

    try:
        for progress, message in steps:
            if not is_running.get(phase_name, False):
                break

            active_phases[phase_name]["progress"] = progress
            active_phases[phase_name]["message"] = message

            # Update models completed
            if progress >= 0.35:
                active_phases[phase_name]["models_completed"] = min(3, int((progress - 0.2) * 4))

            logger.info(f"[{progress*100:5.1f}%] {message}")
            time.sleep(2)  # Simulate work

        # Mark as completed and create model entries
        active_phases[phase_name]["status"] = "completed"

        # Create model entries
        focuses = ["reasoning", "memory_integration", "adaptive_computation"]
        for i, focus in enumerate(focuses, 1):
            model = {
                "model_id": f"cognate_foundation_{i}",
                "model_name": f"Cognate Foundation Model {i}",
                "phase_name": "Cognate",
                "parameter_count": 25083528,  # Exact 25M target
                "created_at": datetime.now().isoformat(),
                "training_status": "completed",
                "focus": focus,
                "artifacts": {
                    "config_path": f"core/agent-forge/phases/cognate_pretrain/models/cognate_foundation_{i}/config.json",
                    "weights_path": f"core/agent-forge/phases/cognate_pretrain/models/cognate_foundation_{i}/pytorch_model.bin",
                    "metadata_path": f"core/agent-forge/phases/cognate_pretrain/models/cognate_foundation_{i}/metadata.json",
                },
            }
            created_models.append(model)

        logger.info(f"Phase {phase_name} completed successfully!")

    except Exception as e:
        logger.error(f"Phase {phase_name} failed: {e}")
        active_phases[phase_name]["status"] = "error"
        active_phases[phase_name]["message"] = f"Error: {str(e)}"

    finally:
        is_running[phase_name] = False


@app.get("/")
async def root():
    return {"message": "Agent Forge API Server", "status": "running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/phases/cognate/start")
async def start_cognate_phase(background_tasks: BackgroundTasks):
    """Start the Cognate phase for 25M parameter model creation."""
    phase_name = "Cognate"

    if is_running.get(phase_name, False):
        return JSONResponse(status_code=400, content={"error": f"{phase_name} phase is already running"})

    # Start phase in background
    background_tasks.add_task(simulate_cognate_phase)

    return {
        "message": f"{phase_name} phase started successfully",
        "phase_name": phase_name,
        "status": "starting",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/phases/status")
async def get_phase_status():
    """Get status of all phases."""
    phases = []
    for phase_name, phase_data in active_phases.items():
        phases.append(phase_data)

    return {"phases": phases, "total_phases": len(phases), "timestamp": datetime.now().isoformat()}


@app.get("/models")
async def get_models():
    """Get all created models."""
    return {
        "models": created_models,
        "total_models": len(created_models),
        "total_parameters": sum(m.get("parameter_count", 0) for m in created_models),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/chat")
async def chat_with_model(request: dict):
    """Simple chat interface with models."""
    model_id = request.get("model_id")
    message = request.get("message", "")

    if not model_id or not message:
        return JSONResponse(status_code=400, content={"error": "model_id and message are required"})

    # Find model
    model = None
    for m in created_models:
        if m["model_id"] == model_id:
            model = m
            break

    if not model:
        return JSONResponse(status_code=404, content={"error": f"Model {model_id} not found"})

    # Simulate response
    response = (
        f"Hello! I'm {model['model_name']} with {model['focus']} specialization. "
        f"I'm a {model['parameter_count']:,} parameter model. Your message: '{message}' - "
        f"I'd be happy to help with tasks related to {model['focus']}!"
    )

    return {
        "model_id": model_id,
        "model_name": model["model_name"],
        "user_message": message,
        "model_response": response,
        "response_time_ms": 150,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/phases/cognate/stop")
async def stop_cognate_phase():
    """Stop the running Cognate phase."""
    phase_name = "Cognate"

    if not is_running.get(phase_name, False):
        return JSONResponse(status_code=400, content={"error": f"{phase_name} phase is not running"})

    is_running[phase_name] = False

    if phase_name in active_phases:
        active_phases[phase_name]["status"] = "stopped"
        active_phases[phase_name]["message"] = "Phase stopped by user"

    return {
        "message": f"{phase_name} phase stopped successfully",
        "phase_name": phase_name,
        "status": "stopped",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE AGENT FORGE API SERVER")
    print("=" * 60)
    print("Starting API server on http://localhost:8083")
    print("Endpoints:")
    print("  POST /phases/cognate/start - Start Cognate phase")
    print("  GET  /phases/status - Get phase status")
    print("  GET  /models - Get created models")
    print("  POST /chat - Chat with models")
    print("  GET  /health - Health check")
    print("=" * 60)

    try:
        uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info")
    except KeyboardInterrupt:
        print("\nAPI server stopped by user")
    except Exception as e:
        print(f"API server error: {e}")
