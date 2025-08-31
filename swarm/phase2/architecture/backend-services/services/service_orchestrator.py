"""
Service Orchestrator - Example implementation showing how all services work together

This demonstrates:
- Service initialization and dependency injection
- Event-driven communication setup
- Complete workflow examples
- Service coordination patterns

This is a reference implementation showing the microservices in action.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from interfaces.service_contracts import (
    TrainingJob, TrainingConfig, ModelPhase, PhaseStartRequest, ChatRequest
)
from integration.service_communication import CommunicationLayer
from services.training_service import create_training_service
from services.model_service import create_model_service
from services.websocket_service import create_websocket_service
from services.api_service import create_api_service
from services.monitoring_service import create_monitoring_service

logger = logging.getLogger(__name__)


class ServiceOrchestrator:
    """Orchestrates all microservices and their interactions."""
    
    def __init__(self, 
                 communication_type: str = "memory",
                 redis_url: str = "redis://localhost:6379",
                 storage_path: str = "./microservice_storage"):
        
        # Initialize communication layer
        self.communication = CommunicationLayer(communication_type, redis_url)
        
        # Service instances
        self.training_service = None
        self.model_service = None
        self.websocket_service = None
        self.api_service = None
        self.monitoring_service = None
        
        self.storage_path = storage_path
        self.is_running = False
        
    async def initialize(self):
        """Initialize all services with proper dependencies."""
        try:
            logger.info("Initializing Service Orchestrator...")
            
            # Initialize communication layer first
            await self.communication.initialize()
            event_bus = self.communication.get_event_bus()
            service_discovery = self.communication.get_service_discovery()
            
            # Create services in dependency order
            logger.info("Creating Model Service...")
            self.model_service = create_model_service(
                storage_path=f"{self.storage_path}/models",
                event_publisher=event_bus
            )
            
            logger.info("Creating Training Service...")
            self.training_service = create_training_service(
                event_publisher=event_bus,
                model_service=self.model_service,
                storage_path=f"{self.storage_path}/training"
            )
            
            logger.info("Creating WebSocket Service...")
            self.websocket_service = create_websocket_service(max_connections=500)
            
            logger.info("Creating Monitoring Service...")
            self.monitoring_service = create_monitoring_service(
                service_endpoints={
                    "training_service": "http://localhost:8001",
                    "model_service": "http://localhost:8002", 
                    "websocket_service": "http://localhost:8003",
                    "api_service": "http://localhost:8000"
                },
                event_publisher=event_bus
            )
            
            logger.info("Creating API Service...")
            self.api_service = create_api_service(
                training_service=self.training_service,
                model_service=self.model_service,
                websocket_service=self.websocket_service,
                monitoring_service=self.monitoring_service
            )
            
            # Register services with discovery
            service_discovery.register_service("training_service", "http://localhost:8001")
            service_discovery.register_service("model_service", "http://localhost:8002")
            service_discovery.register_service("websocket_service", "http://localhost:8003")
            service_discovery.register_service("api_service", "http://localhost:8000")
            service_discovery.register_service("monitoring_service", "http://localhost:8004")
            
            # Setup event subscriptions
            await self._setup_event_handlers()
            
            self.is_running = True
            logger.info("✅ All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    async def _setup_event_handlers(self):
        """Setup event handling between services."""
        event_bus = self.communication.get_event_bus()
        
        # Training progress events → WebSocket broadcasts
        async def handle_training_progress(event):
            """Forward training progress to WebSocket clients."""
            from interfaces.service_contracts import WebSocketMessage
            message = WebSocketMessage(
                type="training_progress",
                source_service="orchestrator",
                data=event.data
            )
            await self.websocket_service.broadcast_to_topic("training_updates", message)
        
        # Model saved events → WebSocket notifications  
        async def handle_model_saved(event):
            """Notify clients when models are saved."""
            from interfaces.service_contracts import WebSocketMessage
            message = WebSocketMessage(
                type="model_saved",
                source_service="orchestrator", 
                data=event.data
            )
            await self.websocket_service.broadcast_to_topic("model_updates", message)
        
        # Health change events → Admin notifications
        async def handle_health_change(event):
            """Notify admins of health changes."""
            from interfaces.service_contracts import WebSocketMessage
            message = WebSocketMessage(
                type="service_health_changed",
                source_service="orchestrator",
                data=event.data
            )
            await self.websocket_service.broadcast_to_topic("admin_alerts", message)
        
        # Subscribe to events
        await event_bus.subscribe("training_progress", handle_training_progress)
        await event_bus.subscribe("training_completed", handle_training_progress)
        await event_bus.subscribe("model_saved", handle_model_saved)
        await event_bus.subscribe("service_health_changed", handle_health_change)
        
        logger.info("Event handlers configured")
    
    async def run_example_workflow(self):
        """Run example workflow showing service coordination."""
        if not self.is_running:
            raise RuntimeError("Services not initialized")
        
        logger.info("🚀 Starting example workflow...")
        
        try:
            # Example 1: Start Cognate training
            logger.info("📚 Starting Cognate training...")
            request = PhaseStartRequest(
                phase_name="cognate",
                parameters={
                    "max_steps": 100,
                    "batch_size": 2,
                    "learning_rate": 2e-4
                }
            )
            
            response = await self.api_service.start_phase(request)
            if response.success:
                job_id = response.data["job_id"]
                logger.info(f"✅ Training started with job ID: {job_id}")
                
                # Monitor progress
                for i in range(10):
                    await asyncio.sleep(2)
                    status_response = await self.api_service.get_phase_status("cognate")
                    if status_response.success:
                        progress = status_response.data.get("progress", 0)
                        logger.info(f"📊 Training progress: {progress:.1%}")
                        
                        if progress >= 1.0:
                            break
            
            # Example 2: List models
            logger.info("📋 Listing available models...")
            models_response = await self.api_service.list_models()
            if models_response.success:
                model_count = models_response.data["total"]
                logger.info(f"📦 Found {model_count} models")
            
            # Example 3: Check system health
            logger.info("🏥 Checking system health...")
            health_response = await self.api_service.health_check()
            if health_response.success:
                active_phases = health_response.data["active_phases"]
                logger.info(f"⚡ System healthy with {active_phases} active phases")
            
            # Example 4: WebSocket statistics
            if self.websocket_service:
                stats = await self.websocket_service.get_statistics()
                logger.info(f"🌐 WebSocket: {stats['active_connections']} connections, {stats['total_messages_sent']} messages sent")
            
            # Example 5: Monitoring overview
            if self.monitoring_service:
                overview = await self.monitoring_service.get_system_overview()
                healthy_services = overview.get('services', {}).get('healthy', 0)
                total_services = overview.get('services', {}).get('total', 0)
                logger.info(f"🔍 Monitoring: {healthy_services}/{total_services} services healthy")
            
            logger.info("✅ Example workflow completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Example workflow failed: {e}")
            raise
    
    async def simulate_training_pipeline(self):
        """Simulate complete training pipeline through all phases."""
        phases = [
            "cognate", "evomerge", "quietstar", "bitnet", 
            "forge-training", "tool-persona", "adas", "final-compression"
        ]
        
        logger.info("🏭 Starting complete training pipeline simulation...")
        
        for phase in phases:
            logger.info(f"🔄 Starting phase: {phase}")
            
            request = PhaseStartRequest(
                phase_name=phase,
                parameters={
                    "max_steps": 50,  # Reduced for demo
                    "batch_size": 2
                }
            )
            
            # Start phase
            response = await self.api_service.start_phase(request)
            if not response.success:
                logger.error(f"❌ Failed to start {phase}: {response.error}")
                continue
            
            # Wait for completion
            while True:
                await asyncio.sleep(1)
                status_response = await self.api_service.get_phase_status(phase)
                if status_response.success:
                    status = status_response.data.get("status")
                    progress = status_response.data.get("progress", 0)
                    
                    if status == "completed":
                        logger.info(f"✅ Phase {phase} completed")
                        break
                    elif status == "failed":
                        logger.error(f"❌ Phase {phase} failed")
                        break
                    else:
                        logger.debug(f"📊 {phase} progress: {progress:.1%}")
                else:
                    logger.warning(f"⚠️ Could not get status for {phase}")
                    break
        
        logger.info("🎉 Complete training pipeline finished!")
    
    async def demonstrate_chat_functionality(self):
        """Demonstrate chat with trained models."""
        logger.info("💬 Demonstrating chat functionality...")
        
        # Get available models
        models_response = await self.api_service.list_models()
        if not models_response.success or not models_response.data["models"]:
            logger.warning("No models available for chat")
            return
        
        # Use first available model
        model = models_response.data["models"][0]
        model_id = model["model_id"]
        
        # Test chat
        chat_request = ChatRequest(
            model_id=model_id,
            message="Hello! Can you tell me about your capabilities?"
        )
        
        chat_response = await self.api_service.chat_with_model(chat_request)
        if chat_response.success:
            response_text = chat_response.data["response"]
            logger.info(f"🤖 Model response: {response_text}")
        else:
            logger.error(f"❌ Chat failed: {chat_response.error}")
    
    async def shutdown(self):
        """Shutdown all services gracefully."""
        logger.info("🔄 Shutting down services...")
        
        try:
            # Shutdown in reverse dependency order
            if self.api_service:
                logger.info("Shutting down API service...")
                # API service shutdown would happen here
            
            if self.websocket_service:
                logger.info("Shutting down WebSocket service...")
                await self.websocket_service.shutdown()
            
            if self.monitoring_service:
                logger.info("Shutting down Monitoring service...")
                await self.monitoring_service.shutdown()
            
            if self.training_service:
                logger.info("Shutting down Training service...")
                # Training service shutdown would happen here
            
            if self.model_service:
                logger.info("Shutting down Model service...")
                # Model service shutdown would happen here
            
            # Shutdown communication layer last
            await self.communication.shutdown()
            
            self.is_running = False
            logger.info("✅ All services shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_api_app(self):
        """Get FastAPI app for running with uvicorn."""
        if not self.api_service:
            raise RuntimeError("API service not initialized")
        return self.api_service.app


async def main():
    """Main function demonstrating the microservice architecture."""
    logging.basicConfig(level=logging.INFO)
    
    # Create orchestrator
    orchestrator = ServiceOrchestrator()
    
    try:
        # Initialize all services
        await orchestrator.initialize()
        
        # Run example workflows
        await orchestrator.run_example_workflow()
        
        # Simulate training pipeline
        await orchestrator.simulate_training_pipeline()
        
        # Demonstrate chat
        await orchestrator.demonstrate_chat_functionality()
        
        logger.info("🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        raise
        
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())