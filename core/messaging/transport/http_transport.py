"""
HTTP Transport Implementation

HTTP/REST transport implementation consolidating gateway server functionality.
Provides compatibility with existing FastAPI-based endpoints.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import json

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    import httpx
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("FastAPI not available, HTTP transport will be limited")

from .base_transport import BaseTransport, TransportState
from ..message_format import UnifiedMessage, MessageType, TransportType

logger = logging.getLogger(__name__)


class HttpTransport(BaseTransport):
    """HTTP/REST transport implementation"""
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for HTTP transport")
        
        self.app = FastAPI(title=f"MessageBus HTTP Transport - {node_id}")
        self.server = None
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Configuration
        self.port = config.get("port", 8000)
        self.host = config.get("host", "0.0.0.0")
        self.enable_cors = config.get("enable_cors", True)
        
        # Node registry for target resolution
        self.node_registry: Dict[str, str] = config.get("node_registry", {})
        
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_middleware(self) -> None:
        """Setup FastAPI middleware"""
        if self.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    
    def _setup_routes(self) -> None:
        """Setup HTTP endpoints"""
        
        @self.app.post("/message")
        async def receive_message(request: Request):
            """Receive incoming message"""
            try:
                message_data = await request.json()
                message = UnifiedMessage.from_dict(message_data)
                
                # Handle message
                await self.handle_incoming_message(message)
                
                return {"status": "delivered", "message_id": message.message_id}
                
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return await self.health_check()
        
        @self.app.get("/metrics")
        async def metrics():
            """Metrics endpoint"""
            return self.get_metrics()
        
        @self.app.post("/register_node")
        async def register_node(node_data: dict):
            """Register node endpoint for discovery"""
            node_id = node_data.get("node_id")
            endpoint = node_data.get("endpoint")
            
            if node_id and endpoint:
                self.node_registry[node_id] = endpoint
                logger.info(f"Node registered: {node_id} -> {endpoint}")
                return {"status": "registered"}
            else:
                raise HTTPException(status_code=400, detail="node_id and endpoint required")
        
        @self.app.get("/nodes")
        async def list_nodes():
            """List registered nodes"""
            return {"nodes": self.node_registry}
        
        # Legacy compatibility endpoints
        
        @self.app.post("/api/v1/chat")
        async def legacy_chat(request: Request):
            """Legacy chat endpoint for edge compatibility"""
            try:
                chat_data = await request.json()
                
                # Convert to unified message format
                message = UnifiedMessage(
                    message_type=MessageType.EDGE_CHAT,
                    transport=TransportType.HTTP,
                    source_id="edge_client",
                    target_id="chat_engine",
                    payload=chat_data
                )
                
                await self.handle_incoming_message(message)
                
                return {"status": "processed", "message_id": message.message_id}
                
            except Exception as e:
                logger.error(f"Error in legacy chat endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/agents/message")
        async def legacy_agent_message(request: Request):
            """Legacy agent message endpoint for gateway compatibility"""
            try:
                agent_data = await request.json()
                
                # Convert to unified message format
                message = UnifiedMessage(
                    message_type=MessageType.AGENT_REQUEST,
                    transport=TransportType.HTTP,
                    source_id=agent_data.get("source", "gateway_client"),
                    target_id=agent_data.get("target"),
                    payload=agent_data.get("payload", {})
                )
                
                await self.handle_incoming_message(message)
                
                return {"delivered": True, "message_id": message.message_id}
                
            except Exception as e:
                logger.error(f"Error in legacy agent message endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def start(self) -> None:
        """Start HTTP server"""
        if self.running:
            logger.warning("HTTP transport already running")
            return
        
        logger.info(f"Starting HTTP transport on {self.host}:{self.port}")
        self.state = TransportState.STARTING
        
        try:
            # Start uvicorn server
            config = uvicorn.Config(
                self.app, 
                host=self.host, 
                port=self.port, 
                log_level="warning"  # Reduce uvicorn logging noise
            )
            self.server = uvicorn.Server(config)
            
            # Start server in background
            asyncio.create_task(self.server.serve())
            
            # Wait a moment for server to start
            await asyncio.sleep(0.5)
            
            self.running = True
            self.state = TransportState.RUNNING
            
            logger.info(f"HTTP transport started successfully on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start HTTP transport: {e}")
            self.state = TransportState.ERROR
            raise
    
    async def stop(self) -> None:
        """Stop HTTP server"""
        if not self.running:
            logger.warning("HTTP transport not running")
            return
        
        logger.info("Stopping HTTP transport")
        self.state = TransportState.STOPPING
        
        try:
            # Close HTTP client
            await self.client.aclose()
            
            # Stop server
            if self.server:
                self.server.should_exit = True
                await asyncio.sleep(0.5)  # Give server time to shutdown
            
            self.running = False
            self.state = TransportState.STOPPED
            
            logger.info("HTTP transport stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping HTTP transport: {e}")
            self.state = TransportState.ERROR
            raise
    
    async def send(self, message: UnifiedMessage, target: str) -> bool:
        """Send HTTP message to target endpoint"""
        if not self.running:
            logger.error("Cannot send: HTTP transport not running")
            return False
        
        try:
            # Resolve target URL
            target_url = self._resolve_target_url(target)
            if not target_url:
                logger.error(f"Cannot resolve target: {target}")
                self._record_send_error()
                return False
            
            # Send message
            response = await self.client.post(
                f"{target_url}/message",
                json=message.to_dict(),
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                self._record_send_success()
                logger.debug(f"HTTP message sent successfully to {target}")
                return True
            else:
                logger.warning(f"HTTP send failed with status {response.status_code}: {target}")
                self._record_send_error()
                return False
                
        except Exception as e:
            logger.error(f"Error sending HTTP message to {target}: {e}")
            self._record_send_error()
            return False
    
    async def broadcast(self, message: UnifiedMessage) -> Dict[str, bool]:
        """Broadcast message to all registered nodes"""
        if not self.running:
            logger.error("Cannot broadcast: HTTP transport not running")
            return {}
        
        results = {}
        
        for node_id, endpoint in self.node_registry.items():
            if node_id != self.node_id:  # Don't send to self
                success = await self.send(message, node_id)
                results[node_id] = success
        
        logger.info(f"HTTP broadcast completed: {sum(results.values())}/{len(results)} successful")
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check HTTP transport health"""
        return {
            "status": "healthy" if self.running else "stopped",
            "state": self.state,
            "host": self.host,
            "port": self.port,
            "registered_nodes": len(self.node_registry),
            "metrics": self.get_metrics()
        }
    
    def register_node(self, node_id: str, endpoint: str) -> None:
        """Register node for message routing"""
        self.node_registry[node_id] = endpoint
        logger.info(f"Node registered: {node_id} -> {endpoint}")
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister node"""
        if node_id in self.node_registry:
            del self.node_registry[node_id]
            logger.info(f"Node unregistered: {node_id}")
            return True
        return False
    
    def _resolve_target_url(self, target: str) -> Optional[str]:
        """Resolve target node ID to URL"""
        # Check if target is already a URL
        if target.startswith(("http://", "https://")):
            return target
        
        # Look up in node registry
        if target in self.node_registry:
            return self.node_registry[target]
        
        # Try default local resolution
        if target == "localhost" or target.startswith("127."):
            return f"http://{target}:8000"
        
        # Default assumption for development
        if ":" not in target:
            return f"http://localhost:8000"  # Default fallback
        
        return None
    
    def get_app(self) -> Optional[Any]:
        """Get FastAPI app instance for external use"""
        return self.app
