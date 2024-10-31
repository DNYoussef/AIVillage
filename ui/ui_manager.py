"""UI Manager implementation for AI Village system."""

import logging
import json
from typing import Dict, Any, List, Optional, Set
from aiohttp import web
import aiohttp_cors
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        self.connections: Set[web.WebSocketResponse] = set()
        self.message_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    async def add_connection(self, ws: web.WebSocketResponse):
        """Add a new WebSocket connection."""
        self.connections.add(ws)
        
        # Send message history to new connection
        for message in self.message_history:
            await ws.send_json(message)
    
    async def remove_connection(self, ws: web.WebSocketResponse):
        """Remove a WebSocket connection."""
        self.connections.remove(ws)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        # Add message to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
        
        # Broadcast to all connections
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {str(e)}")
                await self.remove_connection(ws)

class UIManager:
    """Manages the web-based user interface for AI Village."""
    
    def __init__(self):
        """Initialize UI Manager."""
        self.app = web.Application()
        self.websocket_manager = WebSocketManager()
        self.metrics: Dict[str, Any] = {}
        self.last_update = datetime.now()
    
    async def initialize(self):
        """Initialize the UI system."""
        logger.info("Initializing UI Manager")
        
        try:
            # Set up routes
            self.app.router.add_get('/', self.index_handler)
            self.app.router.add_get('/ws', self.websocket_handler)
            self.app.router.add_get('/metrics', self.metrics_handler)
            self.app.router.add_get('/status', self.status_handler)
            self.app.router.add_post('/task', self.task_handler)
            
            # Set up static files
            self.app.router.add_static('/static/', path='ui/static', name='static')
            
            # Set up CORS
            cors = aiohttp_cors.setup(self.app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
            })
            
            # Apply CORS to all routes
            for route in list(self.app.router.routes()):
                cors.add(route)
            
            logger.info("UI Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing UI Manager: {str(e)}")
            raise
    
    async def start(self, host: str = '0.0.0.0', port: int = 8080):
        """Start the web server."""
        logger.info(f"Starting UI server on {host}:{port}")
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
    
    async def shutdown(self):
        """Shutdown the UI system."""
        logger.info("Shutting down UI Manager")
        
        # Close all WebSocket connections
        for ws in self.websocket_manager.connections.copy():
            await ws.close()
        
        # Cleanup
        if hasattr(self.app, 'cleanup'):
            await self.app.cleanup()
    
    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics."""
        self.metrics = metrics
        self.last_update = datetime.now()
        
        # Broadcast metrics update via WebSocket
        await self.websocket_manager.broadcast({
            "type": "metrics_update",
            "data": metrics,
            "timestamp": self.last_update.isoformat()
        })
    
    # Route handlers
    
    async def index_handler(self, request: web.Request) -> web.Response:
        """Handle root path request."""
        with open('ui/index.html') as f:
            return web.Response(
                text=f.read(),
                content_type='text/html'
            )
    
    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        await self.websocket_manager.add_connection(ws)
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        response = await self._handle_ws_message(data)
                        await ws.send_json(response)
                    except json.JSONDecodeError:
                        await ws.send_json({
                            "error": "Invalid JSON format"
                        })
                    except Exception as e:
                        await ws.send_json({
                            "error": str(e)
                        })
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f'WebSocket connection closed with exception {ws.exception()}')
        
        finally:
            await self.websocket_manager.remove_connection(ws)
        
        return ws
    
    async def metrics_handler(self, request: web.Request) -> web.Response:
        """Handle metrics request."""
        return web.json_response({
            "metrics": self.metrics,
            "last_update": self.last_update.isoformat()
        })
    
    async def status_handler(self, request: web.Request) -> web.Response:
        """Handle status request."""
        return web.json_response({
            "status": "running",
            "connections": len(self.websocket_manager.connections),
            "last_update": self.last_update.isoformat()
        })
    
    async def task_handler(self, request: web.Request) -> web.Response:
        """Handle task submission."""
        try:
            data = await request.json()
            
            # Validate task data
            if not self._validate_task_data(data):
                raise ValueError("Invalid task format")
            
            # Broadcast task via WebSocket
            await self.websocket_manager.broadcast({
                "type": "new_task",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            
            return web.json_response({
                "status": "success",
                "message": "Task submitted successfully"
            })
            
        except json.JSONDecodeError:
            return web.json_response({
                "error": "Invalid JSON format"
            }, status=400)
        except Exception as e:
            return web.json_response({
                "error": str(e)
            }, status=400)
    
    # Helper methods
    
    async def _handle_ws_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming WebSocket messages."""
        message_type = data.get('type')
        
        handlers = {
            'get_metrics': lambda: {
                "type": "metrics",
                "data": self.metrics,
                "timestamp": self.last_update.isoformat()
            },
            'get_status': lambda: {
                "type": "status",
                "data": {
                    "status": "running",
                    "connections": len(self.websocket_manager.connections)
                }
            }
        }
        
        handler = handlers.get(message_type)
        if handler:
            return handler()
        else:
            return {
                "error": f"Unknown message type: {message_type}"
            }
    
    def _validate_task_data(self, data: Dict[str, Any]) -> bool:
        """Validate task submission data."""
        required_fields = ['type', 'content']
        return all(field in data for field in required_fields)
