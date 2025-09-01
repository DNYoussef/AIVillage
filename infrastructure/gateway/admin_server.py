#!/usr/bin/env python3
"""
Admin Dashboard Server for AIVillage Backend Monitoring
Serves the admin dashboard on port 3006 with health check endpoints
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import psutil
import uvicorn

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AdminDashboardServer:
    """Admin dashboard server for backend monitoring"""

    def __init__(self, port: int = 3006):
        self.port = port
        self.app = FastAPI(
            title="AIVillage Admin Dashboard", description="Backend monitoring and health dashboard", version="1.0.0"
        )
        self.start_time = time.time()
        self.setup_middleware()
        self.setup_routes()

    def setup_middleware(self):
        """Configure CORS and other middleware"""
        # SECURITY: Admin interface requires MOST restrictive CORS - NO WILDCARDS
        try:
            from src.security.cors_config import ADMIN_CORS_CONFIG

            self.app.add_middleware(CORSMiddleware, **ADMIN_CORS_CONFIG)
        except ImportError:
            # Fallback admin configuration (very restrictive)
            import os

            env = os.getenv("AIVILLAGE_ENV", "development")
            if env == "production":
                admin_origins = ["https://admin.aivillage.app"]
            else:
                admin_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]  # Admin localhost only

            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=admin_origins,
                allow_credentials=True,
                allow_methods=["GET", "POST"],  # Limited methods for admin
                allow_headers=["Content-Type", "Authorization"],
            )

    def setup_routes(self):
        """Setup API routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the admin dashboard"""
            dashboard_path = project_root / "packages" / "ui" / "admin-dashboard.html"
            try:
                with open(dashboard_path, encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
            except FileNotFoundError:
                return HTMLResponse(
                    content="<h1>Admin Dashboard Not Found</h1><p>Dashboard file not found at expected location.</p>",
                    status_code=404,
                )

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return JSONResponse(
                {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "service": "admin-dashboard",
                    "port": self.port,
                    "uptime_seconds": int(time.time() - self.start_time),
                }
            )

        @self.app.get("/api/system-metrics")
        async def get_system_metrics():
            """Get system health metrics"""
            try:
                # Get system metrics using psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")
                boot_time = psutil.boot_time()

                # Network connections
                connections = len(psutil.net_connections())

                return JSONResponse(
                    {
                        "cpu_usage": round(cpu_percent, 1),
                        "memory_usage": round(memory.percent, 1),
                        "memory_total": memory.total,
                        "memory_used": memory.used,
                        "disk_usage": round(disk.percent, 1),
                        "disk_total": disk.total,
                        "disk_used": disk.used,
                        "active_connections": connections,
                        "uptime_seconds": int(time.time() - boot_time),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"Error getting system metrics: {e}")
                return JSONResponse({"error": "Failed to get system metrics", "details": str(e)}, status_code=500)

        @self.app.get("/api/service-status")
        async def get_service_status():
            """Check status of all AIVillage services"""
            services = {
                "8080": "Main API Server",
                "8081": "Auth Service",
                "8082": "Data Service",
                "3006": "Admin Dashboard",
            }

            service_status = {}

            for port, name in services.items():
                try:
                    # Check if port is in use (simple check)
                    connections = psutil.net_connections()
                    port_in_use = any(conn.laddr.port == int(port) for conn in connections if conn.laddr)

                    service_status[port] = {
                        "name": name,
                        "status": "online" if port_in_use else "offline",
                        "port": int(port),
                        "checked_at": datetime.now().isoformat(),
                    }
                except Exception as e:
                    service_status[port] = {
                        "name": name,
                        "status": "error",
                        "port": int(port),
                        "error": str(e),
                        "checked_at": datetime.now().isoformat(),
                    }

            return JSONResponse(service_status)

        @self.app.get("/api/agent-metrics")
        async def get_agent_metrics():
            """Get agent activity metrics (simulated for now)"""
            # In a real implementation, this would connect to the agent system
            import random

            return JSONResponse(
                {
                    "active_agents": random.randint(2, 10),
                    "tasks_processed": random.randint(100, 600),
                    "avg_response_time_ms": random.randint(50, 250),
                    "success_rate": round(random.uniform(85, 100), 1),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        @self.app.get("/api/network-status")
        async def get_network_status():
            """Get network connectivity status"""
            # Simulate network status checks
            import random

            services = ["database", "redis", "external_apis", "p2p_network"]
            status = {}

            for service in services:
                # Simulate 80% uptime
                is_online = random.random() > 0.2
                status[service] = {
                    "status": "online" if is_online else "offline",
                    "response_time_ms": random.randint(10, 100) if is_online else None,
                    "last_check": datetime.now().isoformat(),
                }

            return JSONResponse(status)

        @self.app.get("/api/logs")
        async def get_recent_logs(limit: int = 50):
            """Get recent system logs"""
            # In a real implementation, this would read from actual log files
            logs = [
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": "Admin dashboard server running",
                    "service": "admin-dashboard",
                },
                {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": "Health checks initialized",
                    "service": "admin-dashboard",
                },
            ]

            return JSONResponse(logs[-limit:])

    async def start_server(self):
        """Start the admin dashboard server"""
        logger.info(f"Starting Admin Dashboard Server on port {self.port}")

        config = uvicorn.Config(app=self.app, host="0.0.0.0", port=self.port, log_level="info", reload=False)

        server = uvicorn.Server(config)
        await server.serve()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="AIVillage Admin Dashboard Server")
    parser.add_argument("--port", type=int, default=3006, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and start server
    server = AdminDashboardServer(port=args.port)

    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
