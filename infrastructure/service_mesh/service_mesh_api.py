#!/usr/bin/env python3
"""
Service Mesh API Gateway

Provides centralized API for service mesh management, monitoring, and control.
Implements production-ready service mesh architecture with comprehensive
monitoring and management capabilities.

Key features:
- Service mesh coordination
- Real-time monitoring dashboard
- Service discovery API
- Health check aggregation
- Load balancing control
- Circuit breaker management
- Deployment orchestration
"""

import asyncio
from datetime import datetime
import json
import logging
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .production_service_manager import ProductionServiceManager
from .service_registry import LoadBalancingStrategy

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AIVillage Service Mesh API", description="Production Service Mesh Management and Monitoring", version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
service_manager: ProductionServiceManager | None = None
websocket_connections: list[WebSocket] = []


# Request/Response models
class ServiceStartRequest(BaseModel):
    service_name: str
    force_restart: bool = False


class ServiceStopRequest(BaseModel):
    service_name: str


class LoadBalancingRequest(BaseModel):
    service_name: str
    strategy: str = "health_based"


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    services: dict[str, Any]


class ServiceDiscoveryResponse(BaseModel):
    service_name: str
    instances: list[dict[str, Any]]
    load_balancing_strategy: str


# WebSocket connection manager
class WebSocketManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.connections)}")

    async def broadcast(self, message: dict):
        if not self.connections:
            return

        disconnected = []
        for connection in self.connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)

        # Remove disconnected connections
        for conn in disconnected:
            if conn in self.connections:
                self.connections.remove(conn)


websocket_manager = WebSocketManager()


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Service mesh dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AIVillage Service Mesh Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }
            .services { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
            .service-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .status-healthy { color: #27ae60; }
            .status-unhealthy { color: #e74c3c; }
            .status-degraded { color: #f39c12; }
            .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }
            .metric { background: white; padding: 15px; border-radius: 8px; text-align: center; }
            .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
            .controls { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
            .btn:hover { background: #2980b9; }
            .btn-danger { background: #e74c3c; }
            .btn-success { background: #27ae60; }
            .log { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 8px; font-family: monospace; height: 200px; overflow-y: scroll; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌐 AIVillage Service Mesh</h1>
                <p>Production Service Integration & Monitoring Dashboard</p>
            </div>

            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="total-services">-</div>
                    <div>Total Services</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="healthy-services">-</div>
                    <div>Healthy Services</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="codex-services">-</div>
                    <div>CODEX Production</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="uptime">-</div>
                    <div>System Uptime</div>
                </div>
            </div>

            <div class="controls">
                <h3>Service Controls</h3>
                <button class="btn" onclick="refreshServices()">🔄 Refresh</button>
                <button class="btn btn-success" onclick="startAllServices()">▶️ Start All</button>
                <button class="btn btn-danger" onclick="stopAllServices()">⏹️ Stop All</button>
                <button class="btn" onclick="runIntegrationTests()">🧪 Run Tests</button>
            </div>

            <div id="services-container" class="services">
                <!-- Services will be populated here -->
            </div>

            <div class="controls">
                <h3>System Logs</h3>
                <div id="logs" class="log">Connecting to service mesh...</div>
            </div>
        </div>

        <script>
            const ws = new WebSocket('ws://localhost:8090/ws');
            let startTime = Date.now();

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
                addLog(JSON.stringify(data, null, 2));
            };

            function updateDashboard(data) {
                if (data.type === 'service_status') {
                    updateMetrics(data.data);
                    updateServices(data.data);
                }
            }

            function updateMetrics(status) {
                document.getElementById('total-services').textContent = status.services_configured || 0;
                document.getElementById('healthy-services').textContent = status.services_running || 0;
                document.getElementById('codex-services').textContent = Object.keys(status.codex_status || {}).length;

                const uptime = Math.floor((Date.now() - startTime) / 1000 / 60);
                document.getElementById('uptime').textContent = uptime + 'm';
            }

            function updateServices(status) {
                const container = document.getElementById('services-container');
                container.innerHTML = '';

                if (status.registry) {
                    Object.entries(status.registry).forEach(([serviceId, serviceData]) => {
                        const service = serviceData.service;
                        const health = serviceData.health;
                        const metrics = serviceData.metrics;

                        const card = document.createElement('div');
                        card.className = 'service-card';

                        const statusClass = health ? `status-${health.status}` : 'status-unknown';

                        card.innerHTML = `
                            <h3>${service.name}</h3>
                            <p><strong>Status:</strong> <span class="${statusClass}">${health ? health.status : 'unknown'}</span></p>
                            <p><strong>Endpoint:</strong> ${service.base_url || `${service.host}:${service.port}`}</p>
                            <p><strong>Response Time:</strong> ${health ? health.response_time_ms.toFixed(1) : '-'}ms</p>
                            <p><strong>Tags:</strong> ${service.tags ? Array.from(service.tags).join(', ') : 'none'}</p>
                            <button class="btn" onclick="restartService('${service.name}')">🔄 Restart</button>
                            <button class="btn btn-danger" onclick="stopService('${service.name}')">⏹️ Stop</button>
                        `;

                        container.appendChild(card);
                    });
                }
            }

            function addLog(message) {
                const logs = document.getElementById('logs');
                const timestamp = new Date().toLocaleTimeString();
                logs.innerHTML += `[${timestamp}] ${message}\n`;
                logs.scrollTop = logs.scrollHeight;
            }

            async function refreshServices() {
                const response = await fetch('/api/services/status');
                const data = await response.json();
                updateDashboard({type: 'service_status', data: data});
                addLog('Services refreshed');
            }

            async function startAllServices() {
                const response = await fetch('/api/services/start-all', {method: 'POST'});
                const result = await response.json();
                addLog(`Start all services: ${result.status}`);
            }

            async function stopAllServices() {
                const response = await fetch('/api/services/stop-all', {method: 'POST'});
                const result = await response.json();
                addLog(`Stop all services: ${result.status}`);
            }

            async function restartService(serviceName) {
                const response = await fetch(`/api/services/${serviceName}/restart`, {method: 'POST'});
                const result = await response.json();
                addLog(`Restart ${serviceName}: ${result.status}`);
            }

            async function stopService(serviceName) {
                const response = await fetch(`/api/services/${serviceName}/stop`, {method: 'POST'});
                const result = await response.json();
                addLog(`Stop ${serviceName}: ${result.status}`);
            }

            async function runIntegrationTests() {
                addLog('Starting integration tests...');
                const response = await fetch('/api/tests/integration', {method: 'POST'});
                const result = await response.json();
                addLog(`Integration tests: ${result.status}`);
            }

            // Initial load
            refreshServices();

            // Periodic refresh
            setInterval(refreshServices, 10000);
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check for service mesh API."""
    global service_manager

    status = "healthy"
    services_info = {}

    if service_manager:
        try:
            manager_status = service_manager.get_service_status()
            services_info = manager_status

            if not manager_status.get("running", False):
                status = "degraded"

        except Exception as e:
            status = "unhealthy"
            services_info = {"error": str(e)}
    else:
        status = "starting"
        services_info = {"message": "Service manager not initialized"}

    return HealthCheckResponse(status=status, timestamp=datetime.now().isoformat(), services=services_info)


@app.get("/api/services/status")
async def get_services_status():
    """Get status of all services."""
    global service_manager

    if not service_manager:
        raise HTTPException(status_code=503, detail="Service manager not initialized")

    return service_manager.get_service_status()


@app.get("/api/services/discovery/{service_name}")
async def discover_service(service_name: str, strategy: str = "health_based") -> ServiceDiscoveryResponse:
    """Discover service instances."""
    global service_manager

    if not service_manager or not service_manager.registry:
        raise HTTPException(status_code=503, detail="Service registry not available")

    # Convert string strategy to enum
    try:
        lb_strategy = LoadBalancingStrategy(strategy)
    except ValueError:
        lb_strategy = LoadBalancingStrategy.HEALTH_BASED

    service = service_manager.registry.get_service(service_name, lb_strategy)

    if not service:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")

    # Get all instances of the service
    all_services = service_manager.registry.get_all_services()
    instances = []

    for service_id, service_data in all_services.items():
        if service_data["service"]["name"] == service_name:
            instances.append(
                {
                    "service_id": service_id,
                    "endpoint": service_data["service"]["base_url"],
                    "health": service_data["health"],
                    "metrics": service_data["metrics"],
                }
            )

    return ServiceDiscoveryResponse(service_name=service_name, instances=instances, load_balancing_strategy=strategy)


@app.post("/api/services/start-all")
async def start_all_services(background_tasks: BackgroundTasks):
    """Start all services."""
    global service_manager

    if not service_manager:
        service_manager = ProductionServiceManager()

    background_tasks.add_task(service_manager.start_all_services)

    return {"status": "started", "message": "Starting all services"}


@app.post("/api/services/stop-all")
async def stop_all_services(background_tasks: BackgroundTasks):
    """Stop all services."""
    global service_manager

    if not service_manager:
        return {"status": "success", "message": "No services to stop"}

    background_tasks.add_task(service_manager.stop_all_services)

    return {"status": "stopping", "message": "Stopping all services"}


@app.post("/api/services/{service_name}/start")
async def start_service(service_name: str, background_tasks: BackgroundTasks):
    """Start a specific service."""
    global service_manager

    if not service_manager:
        raise HTTPException(status_code=503, detail="Service manager not initialized")

    background_tasks.add_task(service_manager.start_service, service_name)

    return {"status": "starting", "service": service_name}


@app.post("/api/services/{service_name}/stop")
async def stop_service(service_name: str, background_tasks: BackgroundTasks):
    """Stop a specific service."""
    global service_manager

    if not service_manager:
        raise HTTPException(status_code=404, detail="Service manager not initialized")

    background_tasks.add_task(service_manager.stop_service, service_name)

    return {"status": "stopping", "service": service_name}


@app.post("/api/services/{service_name}/restart")
async def restart_service(service_name: str, background_tasks: BackgroundTasks):
    """Restart a specific service."""
    global service_manager

    if not service_manager:
        raise HTTPException(status_code=503, detail="Service manager not initialized")

    background_tasks.add_task(service_manager.restart_service, service_name)

    return {"status": "restarting", "service": service_name}


@app.post("/api/tests/integration")
async def run_integration_tests(background_tasks: BackgroundTasks):
    """Run integration tests."""

    async def run_tests():
        try:
            import subprocess

            # Run pytest on the integration tests
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/integration/test_production_integration.py",
                    "-v",
                    "--tb=short",
                    "--asyncio-mode=auto",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Broadcast test results
            await websocket_manager.broadcast(
                {
                    "type": "integration_tests",
                    "status": "completed" if result.returncode == 0 else "failed",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                }
            )

        except Exception as e:
            await websocket_manager.broadcast({"type": "integration_tests", "status": "error", "error": str(e)})

    background_tasks.add_task(run_tests)

    return {"status": "started", "message": "Integration tests started"}


@app.get("/api/codex/status")
async def get_codex_status():
    """Get CODEX integration status."""
    global service_manager

    if not service_manager:
        raise HTTPException(status_code=503, detail="Service manager not initialized")

    status = service_manager.get_service_status()
    codex_status = status.get("codex_status", {})

    # Calculate conversion metrics
    total_services = len(codex_status)
    production_services = sum(1 for s in codex_status.values() if s.get("status") == "production")

    return {
        "total_codex_services": total_services,
        "production_services": production_services,
        "simulation_services": total_services - production_services,
        "conversion_rate": production_services / max(total_services, 1),
        "services": codex_status,
        "status": "production_ready" if production_services == total_services else "partial_conversion",
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket)

    try:
        await websocket.send_json(
            {
                "type": "connection_established",
                "message": "Connected to Service Mesh API",
                "timestamp": datetime.now().isoformat(),
            }
        )

        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})

                elif message.get("type") == "get_status":
                    if service_manager:
                        status = service_manager.get_service_status()
                        await websocket.send_json(
                            {"type": "service_status", "data": status, "timestamp": datetime.now().isoformat()}
                        )

            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON", "timestamp": datetime.now().isoformat()}
                )

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


async def broadcast_status_updates():
    """Background task to broadcast status updates."""
    while True:
        try:
            if service_manager:
                status = service_manager.get_service_status()
                await websocket_manager.broadcast(
                    {"type": "service_status", "data": status, "timestamp": datetime.now().isoformat()}
                )

            await asyncio.sleep(10)  # Broadcast every 10 seconds

        except Exception as e:
            logger.error(f"Error broadcasting status: {e}")
            await asyncio.sleep(10)


@app.on_event("startup")
async def startup_event():
    """Initialize service mesh on startup."""
    global service_manager

    logger.info("Starting Service Mesh API...")

    # Initialize service manager
    service_manager = ProductionServiceManager()

    # Start background status broadcasting
    asyncio.create_task(broadcast_status_updates())

    logger.info("✅ Service Mesh API ready on port 8090")
    logger.info("🌐 Dashboard: http://localhost:8090/")
    logger.info("📊 API Docs: http://localhost:8090/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global service_manager

    logger.info("Shutting down Service Mesh API...")

    if service_manager:
        await service_manager.stop_all_services()

    logger.info("✅ Service Mesh API shutdown complete")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Service Mesh API on port 8090...")
    uvicorn.run(app, host="0.0.0.0", port=8090)
