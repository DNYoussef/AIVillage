#!/usr/bin/env python3
"""
Model Chat Interface API

Provides chat interface for testing trained models after each Agent Forge phase:
- Load and interact with trained models
- Chat session management
- Performance monitoring during inference
- Model comparison capabilities
"""

import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

# Add core to path for Agent Forge imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Chat message structure."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()
    model_id: str | None = None
    response_time_ms: float | None = None


class ChatRequest(BaseModel):
    """Chat request structure."""

    model_id: str
    message: str
    session_id: str | None = None
    max_tokens: int | None = 256
    temperature: float | None = 0.7


class ChatResponse(BaseModel):
    """Chat response structure."""

    session_id: str
    model_id: str
    model_name: str
    response: str
    response_time_ms: float
    token_count: int
    conversation_history: list[ChatMessage]


# Global state for chat sessions and loaded models
CHAT_SESSIONS: dict[str, dict[str, Any]] = {}
LOADED_MODELS: dict[str, dict[str, Any]] = {}
MODEL_REGISTRY: dict[str, dict[str, Any]] = {}

app = FastAPI(title="Model Chat Interface API")

# SECURITY: Add secure CORS middleware - NO WILDCARDS
try:
    from src.security.cors_config import SECURE_CORS_CONFIG
    app.add_middleware(CORSMiddleware, **SECURE_CORS_CONFIG)
except ImportError:
    # Fallback secure configuration
    import os
    env = os.getenv("AIVILLAGE_ENV", "development")
    cors_origins = ["http://localhost:3000", "http://localhost:8080"] if env != "production" else ["https://aivillage.app"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "Authorization"]
    )


def register_model(model_id: str, model_info: dict[str, Any]):
    """Register a trained model for chat interface."""
    MODEL_REGISTRY[model_id] = {**model_info, "registered_at": datetime.now()}
    logger.info(f"Registered model for chat: {model_id}")


def get_mock_response(model_id: str, message: str) -> str:
    """Generate mock responses for different model types during development."""
    model_info = MODEL_REGISTRY.get(model_id, {})
    phase_name = model_info.get("phase_name", "Unknown")

    # Different response styles based on training phase
    if phase_name == "Cognate":
        return f"[Cognate Model] I'm a newly created Cognate model with {model_info.get('parameter_count', 0):,} parameters. You asked: '{message}'. I'm still learning basic reasoning patterns."

    elif phase_name == "EvoMerge":
        fitness = model_info.get("artifacts", {}).get("fitness_score", 0.5)
        return f"[EvoMerge Champion] As an evolved model with fitness {fitness:.3f}, I can provide better responses. Regarding '{message}': I've been optimized through evolutionary merging and show improved capabilities."

    elif phase_name == "Quiet-STaR":
        return f"[Quiet-STaR Enhanced] *thinking step by step* For your question '{message}', let me reason through this... I now have enhanced reasoning capabilities from thought integration."

    elif phase_name == "BitNet":
        return f"[BitNet Compressed] I'm now running with 1.58-bit quantization! Despite compression, I can still handle: '{message}'. My responses are faster but maintain quality."

    elif phase_name == "Forge Training":
        return f"[Forge Trained] After intensive forge training with Grokfast optimization, I provide: '{message}' requires deep understanding which I've developed through advanced training."

    elif phase_name == "Tool Baking":
        return f"[Tool-Enhanced] I now have baked tool capabilities! For '{message}', I can access specialized functions and persona-adapted responses."

    elif phase_name == "ADAS":
        return f"[Architecture Optimized] My architecture has been optimized through ADAS. Regarding '{message}': I now have improved efficiency and specialized reasoning pathways."

    elif phase_name == "Final Compression":
        return f"[Hypercompressed] Final compression complete! I'm now ultra-efficient while maintaining capabilities. Your query '{message}' demonstrates my compressed yet powerful inference."

    else:
        return f"[Unknown Phase] I'm a model from phase '{phase_name}'. You said: '{message}'. I can provide responses but my exact training phase capabilities are unclear."


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "model_chat",
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "loaded_models": len(LOADED_MODELS),
        "active_sessions": len(CHAT_SESSIONS),
    }


@app.get("/models")
async def get_available_models():
    """Get list of models available for chat."""
    available_models = []
    for model_id, model_info in MODEL_REGISTRY.items():
        model_status = "loaded" if model_id in LOADED_MODELS else "available"
        available_models.append({**model_info, "status": model_status})

    return {"models": available_models, "total": len(available_models), "loaded": len(LOADED_MODELS)}


@app.post("/models/{model_id}/register")
async def register_model_endpoint(model_id: str, model_info: dict[str, Any]):
    """Register a new trained model for chat interface."""
    register_model(model_id, model_info)
    return {"message": f"Model {model_id} registered successfully"}


@app.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """Load a model for chat interface."""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Model not found in registry")

    if model_id in LOADED_MODELS:
        return {"message": f"Model {model_id} already loaded"}

    try:
        model_info = MODEL_REGISTRY[model_id]

        # In development, we'll use mock loading
        # In production, this would load the actual model
        if TRANSFORMERS_AVAILABLE and os.path.exists(model_info.get("model_path", "")):
            # Actual model loading (when models exist)
            tokenizer = AutoTokenizer.from_pretrained(model_info["model_path"])
            model = AutoModelForCausalLM.from_pretrained(model_info["model_path"])

            LOADED_MODELS[model_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "model_info": model_info,
                "loaded_at": datetime.now(),
                "inference_count": 0,
            }
        else:
            # Mock loading for development
            LOADED_MODELS[model_id] = {
                "model": None,  # Mock model
                "tokenizer": None,  # Mock tokenizer
                "model_info": model_info,
                "loaded_at": datetime.now(),
                "inference_count": 0,
                "mock_mode": True,
            }

        logger.info(f"Loaded model: {model_id}")
        return {
            "message": f"Model {model_id} loaded successfully",
            "model_name": model_info.get("model_name", model_id),
            "phase_name": model_info.get("phase_name", "Unknown"),
        }

    except Exception as e:
        logger.exception(f"Failed to load model {model_id}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.delete("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a model to free memory."""
    if model_id not in LOADED_MODELS:
        raise HTTPException(status_code=404, detail="Model not loaded")

    del LOADED_MODELS[model_id]

    # Clear GPU cache if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"message": f"Model {model_id} unloaded successfully"}


@app.post("/chat", response_model=ChatResponse)
async def chat_with_model(request: ChatRequest):
    """Chat with a trained model."""
    if request.model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Model not found")

    # Load model if not already loaded
    if request.model_id not in LOADED_MODELS:
        await load_model(request.model_id)

    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in CHAT_SESSIONS:
        CHAT_SESSIONS[session_id] = {
            "session_id": session_id,
            "model_id": request.model_id,
            "created_at": datetime.now(),
            "messages": [],
        }

    session = CHAT_SESSIONS[session_id]
    model_info = LOADED_MODELS[request.model_id]

    # Record user message
    user_message = ChatMessage(role="user", content=request.message)
    session["messages"].append(user_message)

    # Generate response
    start_time = time.time()

    try:
        if model_info.get("mock_mode", False):
            # Use mock response during development
            response_text = get_mock_response(request.model_id, request.message)
            await asyncio.sleep(0.5)  # Simulate inference time
        else:
            # Actual model inference
            tokenizer = model_info["tokenizer"]
            model = model_info["model"]

            inputs = tokenizer.encode(request.message, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        logger.exception("Model inference failed")
        response_text = f"Error during inference: {str(e)}"

    response_time_ms = (time.time() - start_time) * 1000

    # Record assistant response
    assistant_message = ChatMessage(
        role="assistant", content=response_text, model_id=request.model_id, response_time_ms=response_time_ms
    )
    session["messages"].append(assistant_message)

    # Update inference count
    model_info["inference_count"] += 1

    # Get model name
    model_registry_info = MODEL_REGISTRY[request.model_id]
    model_name = model_registry_info.get("model_name", request.model_id)

    return ChatResponse(
        session_id=session_id,
        model_id=request.model_id,
        model_name=model_name,
        response=response_text,
        response_time_ms=response_time_ms,
        token_count=len(response_text.split()),  # Rough token estimate
        conversation_history=session["messages"][-10:],  # Last 10 messages
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get chat session history."""
    if session_id not in CHAT_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    return CHAT_SESSIONS[session_id]


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id not in CHAT_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    del CHAT_SESSIONS[session_id]
    return {"message": f"Session {session_id} deleted"}


@app.get("/sessions")
async def get_all_sessions():
    """Get all active chat sessions."""
    sessions = []
    for session_id, session_info in CHAT_SESSIONS.items():
        sessions.append(
            {
                "session_id": session_id,
                "model_id": session_info["model_id"],
                "model_name": MODEL_REGISTRY.get(session_info["model_id"], {}).get("model_name", "Unknown"),
                "created_at": session_info["created_at"],
                "message_count": len(session_info["messages"]),
            }
        )

    return {"sessions": sessions, "total": len(sessions)}


@app.post("/compare")
async def compare_models(request: dict[str, Any]):
    """Compare responses from multiple models to the same prompt."""
    message = request.get("message")
    model_ids = request.get("model_ids", [])

    if not message or not model_ids:
        raise HTTPException(status_code=400, detail="Message and model_ids required")

    comparisons = []

    for model_id in model_ids:
        if model_id not in MODEL_REGISTRY:
            comparisons.append({"model_id": model_id, "error": "Model not found"})
            continue

        try:
            # Create chat request for each model
            chat_request = ChatRequest(model_id=model_id, message=message)
            response = await chat_with_model(chat_request)

            comparisons.append(
                {
                    "model_id": model_id,
                    "model_name": response.model_name,
                    "response": response.response,
                    "response_time_ms": response.response_time_ms,
                    "token_count": response.token_count,
                }
            )

        except Exception as e:
            comparisons.append({"model_id": model_id, "error": str(e)})

    return {"prompt": message, "comparisons": comparisons, "timestamp": datetime.now()}


# WebSocket endpoint for real-time chat
@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat interface."""
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            model_id = data.get("model_id")
            message = data.get("message")

            if not model_id or not message:
                await websocket.send_json({"error": "model_id and message required"})
                continue

            # Process chat request
            try:
                chat_request = ChatRequest(model_id=model_id, message=message, session_id=session_id)

                response = await chat_with_model(chat_request)

                # Send response back
                await websocket.send_json({"type": "chat_response", "data": response.dict()})

            except Exception as e:
                await websocket.send_json({"type": "error", "error": str(e)})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8084)
