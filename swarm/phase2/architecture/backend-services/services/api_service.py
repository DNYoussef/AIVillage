"""
API Service - REST endpoint handlers and business logic orchestration

This service is responsible for:
- REST API endpoint implementation
- Request validation and routing
- Business logic coordination between services
- Error handling and response formatting
- Authentication and authorization

Size Target: <400 lines
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from interfaces.service_contracts import (
    IAPIService, APIResponse, PhaseStartRequest, ChatRequest,
    TrainingJob, TrainingConfig, ModelPhase, TaskStatus
)

logger = logging.getLogger(__name__)


class APIService(IAPIService):
    """Implementation of the API Service."""
    
    def __init__(self, 
                 training_service=None,
                 model_service=None,
                 websocket_service=None,
                 monitoring_service=None):
        self.training_service = training_service
        self.model_service = model_service
        self.websocket_service = websocket_service
        self.monitoring_service = monitoring_service
        
        # Service state
        self.phase_status: Dict[str, Dict] = {}
        self.active_pipeline = None
        
        # Setup FastAPI app
        self.app = FastAPI(title="Agent Forge API Service", version="2.0.0")
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {"message": "Agent Forge API Service", "version": "2.0.0"}
        
        @self.app.get("/health")
        async def health():
            return await self.health_check()
        
        @self.app.post("/phases/{phase_name}/start")
        async def start_phase(phase_name: str, request: PhaseStartRequest, background_tasks: BackgroundTasks):
            request.phase_name = phase_name
            return await self.start_phase(request, background_tasks)
        
        @self.app.get("/phases/{phase_id}/status")
        async def get_phase_status(phase_id: str):
            return await self.get_phase_status(phase_id)
        
        @self.app.get("/phases/status")
        async def get_all_phases_status():
            return await self.get_all_phases_status()
        
        @self.app.post("/models/export")
        async def export_models(model_ids: list[str]):
            return await self.export_models(model_ids)
        
        @self.app.get("/models")
        async def list_models(phase: Optional[str] = None):
            return await self.list_models(phase)
        
        @self.app.post("/chat")
        async def chat_with_model(request: ChatRequest):
            return await self.chat_with_model(request)
        
        @self.app.post("/pipeline/run")
        async def run_pipeline(request: PhaseStartRequest, background_tasks: BackgroundTasks):
            return await self.run_complete_pipeline(request, background_tasks)
        
        @self.app.get("/pipeline/status")
        async def get_pipeline_status():
            return await self.get_pipeline_status()
        
        @self.app.post("/pipeline/reset")
        async def reset_pipeline():
            return await self.reset_pipeline()
    
    async def start_phase(self, request: PhaseStartRequest, background_tasks: BackgroundTasks = None) -> APIResponse:
        """Start a training phase."""
        try:
            # Validate phase name
            try:
                phase = ModelPhase(request.phase_name.lower().replace('-', '_'))
            except ValueError:
                return APIResponse(
                    success=False,
                    error=f"Invalid phase name: {request.phase_name}"
                )
            
            # Check if phase is already running
            if request.phase_name in self.phase_status:
                current_status = self.phase_status[request.phase_name]
                if current_status.get("status") == "running":
                    return APIResponse(
                        success=False,
                        error=f"Phase {request.phase_name} is already running"
                    )
            
            # Create training configuration
            config = TrainingConfig(
                max_steps=request.parameters.get("max_steps", 2000),
                batch_size=request.parameters.get("batch_size", 2),
                learning_rate=request.parameters.get("learning_rate", 2e-4),
                output_dir=f"./models/{phase.value}",
                max_train_samples=request.parameters.get("max_train_samples", 5000),
                max_eval_samples=request.parameters.get("max_eval_samples", 500),
                use_grokfast=request.parameters.get("use_grokfast", True)
            )
            
            # Create training job
            training_job = TrainingJob(
                phase=phase,
                config=config,
                parameters=request.parameters
            )
            
            # Start training job
            if self.training_service:
                job_id = await self.training_service.start_training_job(training_job)
                
                # Update phase status
                self.phase_status[request.phase_name] = {
                    "phase_name": request.phase_name,
                    "status": "running",
                    "job_id": job_id,
                    "started_at": datetime.now().isoformat(),
                    "progress": 0.0,
                    "message": f"Started {request.phase_name} training"
                }
                
                return APIResponse(
                    success=True,
                    data={
                        "job_id": job_id,
                        "phase": request.phase_name,
                        "message": f"Started {request.phase_name} training"
                    }
                )
            else:
                # Fallback for when training service is not available
                task_id = str(uuid.uuid4())
                self.phase_status[request.phase_name] = {
                    "phase_name": request.phase_name,
                    "status": "running",
                    "task_id": task_id,
                    "started_at": datetime.now().isoformat(),
                    "progress": 0.0,
                    "message": f"Started {request.phase_name} simulation"
                }
                
                # Start simulation in background
                if background_tasks:
                    background_tasks.add_task(self._simulate_phase_execution, request.phase_name, task_id)
                
                return APIResponse(
                    success=True,
                    data={
                        "task_id": task_id,
                        "phase": request.phase_name,
                        "message": f"Started {request.phase_name} simulation"
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to start phase {request.phase_name}: {e}")
            return APIResponse(
                success=False,
                error=f"Failed to start phase: {str(e)}"
            )
    
    async def get_phase_status(self, phase_id: str) -> APIResponse:
        """Get status of a training phase."""
        try:
            if phase_id not in self.phase_status:
                return APIResponse(
                    success=False,
                    error=f"Phase {phase_id} not found"
                )
            
            status = self.phase_status[phase_id]
            
            # If we have a training service, get real status
            if self.training_service and "job_id" in status:
                try:
                    job = await self.training_service.get_job_status(status["job_id"])
                    status.update({
                        "status": job.status.value,
                        "progress": job.progress,
                        "message": job.message,
                        "current_step": job.current_step,
                        "error": job.error_message
                    })
                except Exception as e:
                    logger.error(f"Failed to get job status: {e}")
            
            return APIResponse(
                success=True,
                data=status
            )
            
        except Exception as e:
            logger.error(f"Failed to get phase status for {phase_id}: {e}")
            return APIResponse(
                success=False,
                error=f"Failed to get phase status: {str(e)}"
            )
    
    async def get_all_phases_status(self) -> APIResponse:
        """Get status of all phases."""
        try:
            return APIResponse(
                success=True,
                data={"phases": self.phase_status}
            )
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"Failed to get phases status: {str(e)}"
            )
    
    async def chat_with_model(self, request: ChatRequest) -> APIResponse:
        """Chat with a trained model."""
        try:
            # Get model from model service
            if self.model_service:
                model = await self.model_service.get_model(request.model_id)
                if not model:
                    return APIResponse(
                        success=False,
                        error=f"Model {request.model_id} not found"
                    )
            
            # Simulate chat response (in real implementation, this would load and run the model)
            response = {
                "model_id": request.model_id,
                "response": f"Hello! I'm a trained AI model. You said: '{request.message}'. I'm a simulated response for now.",
                "timestamp": datetime.now().isoformat()
            }
            
            return APIResponse(
                success=True,
                data=response
            )
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return APIResponse(
                success=False,
                error=f"Chat failed: {str(e)}"
            )
    
    async def export_models(self, model_ids: list[str]) -> APIResponse:
        """Export models."""
        try:
            if not self.model_service:
                return APIResponse(
                    success=False,
                    error="Model service not available"
                )
            
            from interfaces.service_contracts import ModelExportRequest
            export_request = ModelExportRequest(model_ids=model_ids)
            
            export_path = await self.model_service.export_models(export_request)
            
            return APIResponse(
                success=True,
                data={
                    "export_path": export_path,
                    "exported_models": len(model_ids)
                }
            )
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return APIResponse(
                success=False,
                error=f"Export failed: {str(e)}"
            )
    
    async def list_models(self, phase: Optional[str] = None) -> APIResponse:
        """List models."""
        try:
            if not self.model_service:
                return APIResponse(
                    success=False,
                    error="Model service not available"
                )
            
            phase_enum = None
            if phase:
                try:
                    phase_enum = ModelPhase(phase.lower().replace('-', '_'))
                except ValueError:
                    return APIResponse(
                        success=False,
                        error=f"Invalid phase: {phase}"
                    )
            
            models = await self.model_service.list_models(phase_enum)
            
            return APIResponse(
                success=True,
                data={
                    "models": [model.dict() for model in models],
                    "total": len(models)
                }
            )
            
        except Exception as e:
            logger.error(f"List models failed: {e}")
            return APIResponse(
                success=False,
                error=f"Failed to list models: {str(e)}"
            )
    
    async def health_check(self) -> APIResponse:
        """Health check endpoint."""
        try:
            # Check service dependencies
            health_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "training_service": self.training_service is not None,
                    "model_service": self.model_service is not None,
                    "websocket_service": self.websocket_service is not None,
                    "monitoring_service": self.monitoring_service is not None
                },
                "active_phases": len(self.phase_status),
                "version": "2.0.0"
            }
            
            return APIResponse(
                success=True,
                data=health_data
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"Health check failed: {str(e)}"
            )
    
    async def run_complete_pipeline(self, request: PhaseStartRequest, background_tasks: BackgroundTasks) -> APIResponse:
        """Run complete training pipeline."""
        try:
            pipeline_id = str(uuid.uuid4())
            self.active_pipeline = {
                "pipeline_id": pipeline_id,
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "phases": [],
                "current_phase": "cognate"
            }
            
            # Start pipeline execution in background
            background_tasks.add_task(self._execute_complete_pipeline, pipeline_id, request.parameters)
            
            return APIResponse(
                success=True,
                data={
                    "pipeline_id": pipeline_id,
                    "message": "Started complete training pipeline"
                }
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"Failed to start pipeline: {str(e)}"
            )
    
    async def get_pipeline_status(self) -> APIResponse:
        """Get pipeline status."""
        if not self.active_pipeline:
            return APIResponse(
                success=True,
                data={"status": "no_active_pipeline"}
            )
        
        return APIResponse(
            success=True,
            data=self.active_pipeline
        )
    
    async def reset_pipeline(self) -> APIResponse:
        """Reset pipeline state."""
        self.active_pipeline = None
        self.phase_status.clear()
        
        return APIResponse(
            success=True,
            data={"message": "Pipeline reset successfully"}
        )
    
    async def _simulate_phase_execution(self, phase_name: str, task_id: str):
        """Simulate phase execution for demonstration."""
        try:
            total_steps = 100
            for step in range(total_steps + 1):
                progress = step / total_steps
                
                self.phase_status[phase_name].update({
                    "progress": progress,
                    "message": f"Step {step}/{total_steps}",
                    "current_step": step
                })
                
                # Broadcast progress if websocket service available
                if self.websocket_service:
                    from interfaces.service_contracts import WebSocketMessage
                    message = WebSocketMessage(
                        type="training_progress",
                        source_service="api_service",
                        data=self.phase_status[phase_name]
                    )
                    await self.websocket_service.broadcast_to_topic("training_updates", message)
                
                await asyncio.sleep(0.1)
            
            # Mark as completed
            self.phase_status[phase_name].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "message": f"{phase_name} simulation completed"
            })
            
        except Exception as e:
            logger.error(f"Simulation failed for {phase_name}: {e}")
            self.phase_status[phase_name].update({
                "status": "failed",
                "error": str(e)
            })
    
    async def _execute_complete_pipeline(self, pipeline_id: str, parameters: Dict[str, Any]):
        """Execute complete training pipeline."""
        phases = ["cognate", "evomerge", "quietstar", "bitnet", "forge-training", 
                 "tool-persona", "adas", "final-compression"]
        
        try:
            for phase in phases:
                self.active_pipeline["current_phase"] = phase
                
                # Start phase
                request = PhaseStartRequest(phase_name=phase, parameters=parameters)
                await self.start_phase(request)
                
                # Wait for completion (simplified)
                await asyncio.sleep(10)
                
                self.active_pipeline["phases"].append({
                    "phase": phase,
                    "status": "completed",
                    "completed_at": datetime.now().isoformat()
                })
            
            # Mark pipeline as completed
            self.active_pipeline.update({
                "status": "completed",
                "completed_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.active_pipeline.update({
                "status": "failed",
                "error": str(e)
            })


# Service factory
def create_api_service(training_service=None, model_service=None, 
                      websocket_service=None, monitoring_service=None) -> APIService:
    """Create and configure the API Service."""
    return APIService(
        training_service=training_service,
        model_service=model_service,
        websocket_service=websocket_service,
        monitoring_service=monitoring_service
    )