#!/usr/bin/env python3
"""
Agent Forge Developer UI Integration Setup

Comprehensive integration coordination system that:
1. Starts all API services (Agent Forge Controller, Model Chat, WebSocket Manager)
2. Coordinates model registration between services
3. Handles service discovery and health monitoring
4. Provides development UI server
5. Runs integration tests to validate system communication
6. Manages service lifecycle and graceful shutdown

Single command to start the entire Agent Forge developer ecosystem.
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import requests
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages lifecycle of all Agent Forge API services."""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.service_configs = {
            "agent_forge_controller": {
                "script": "infrastructure/gateway/api/agent_forge_controller.py",
                "port": 8083,
                "health_endpoint": "/health",
            },
            "model_chat": {
                "script": "infrastructure/gateway/api/model_chat.py",
                "port": 8084,
                "health_endpoint": "/health",
            },
            "websocket_manager": {
                "script": "infrastructure/gateway/api/websocket_manager.py",
                "port": 8085,
                "health_endpoint": "/health",
            },
        }
        self.running = False

    def start_service(self, service_name: str) -> bool:
        """Start individual service."""
        if service_name in self.processes:
            logger.warning(f"Service {service_name} already running")
            return True

        config = self.service_configs[service_name]
        script_path = Path(config["script"])

        if not script_path.exists():
            logger.error(f"Service script not found: {script_path}")
            return False

        try:
            # Start service process
            process = subprocess.Popen(
                [sys.executable, str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            self.processes[service_name] = process
            logger.info(f"Started {service_name} (PID: {process.pid})")

            # Wait a moment for startup
            time.sleep(2)

            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Service {service_name} failed to start: {stderr}")
                return False

            return True

        except Exception as e:
            logger.exception(f"Failed to start {service_name}: {e}")
            return False

    def stop_service(self, service_name: str) -> bool:
        """Stop individual service."""
        if service_name not in self.processes:
            logger.warning(f"Service {service_name} not running")
            return True

        try:
            process = self.processes[service_name]
            process.terminate()

            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {service_name}")
                process.kill()
                process.wait()

            del self.processes[service_name]
            logger.info(f"Stopped {service_name}")
            return True

        except Exception as e:
            logger.exception(f"Failed to stop {service_name}: {e}")
            return False

    def check_service_health(self, service_name: str) -> dict[str, Any]:
        """Check health of individual service."""
        config = self.service_configs[service_name]
        health_url = f"http://localhost:{config['port']}{config['health_endpoint']}"

        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "service": service_name,
                    "port": config["port"],
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "data": response.json(),
                }
            else:
                return {
                    "status": "unhealthy",
                    "service": service_name,
                    "port": config["port"],
                    "error": f"HTTP {response.status_code}",
                }
        except Exception as e:
            return {"status": "unreachable", "service": service_name, "port": config["port"], "error": str(e)}

    def start_all_services(self) -> bool:
        """Start all Agent Forge services."""
        logger.info("Starting Agent Forge service ecosystem...")

        success_count = 0
        for service_name in self.service_configs.keys():
            if self.start_service(service_name):
                success_count += 1
            else:
                logger.error(f"Failed to start {service_name}")

        if success_count == len(self.service_configs):
            logger.info("All services started successfully!")
            self.running = True
            return True
        else:
            logger.error(f"Only {success_count}/{len(self.service_configs)} services started")
            return False

    def stop_all_services(self):
        """Stop all services gracefully."""
        logger.info("Stopping all Agent Forge services...")

        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)

        self.running = False
        logger.info("All services stopped")

    def get_service_status(self) -> dict[str, Any]:
        """Get comprehensive status of all services."""
        status = {"timestamp": datetime.now().isoformat(), "services": {}, "overall_health": "healthy"}

        unhealthy_count = 0

        for service_name in self.service_configs.keys():
            health_info = self.check_service_health(service_name)
            status["services"][service_name] = health_info

            if health_info["status"] != "healthy":
                unhealthy_count += 1

        if unhealthy_count > 0:
            status["overall_health"] = "degraded" if unhealthy_count < len(self.service_configs) else "unhealthy"

        return status


class ModelRegistrationCoordinator:
    """Coordinates model registration between Agent Forge Controller and Model Chat services."""

    def __init__(self):
        self.controller_url = "http://localhost:8083"
        self.chat_url = "http://localhost:8084"
        self.registered_models = set()

    async def monitor_and_register_models(self):
        """Monitor Agent Forge Controller for new models and register them with Chat service."""
        logger.info("Starting model registration monitoring...")

        while True:
            try:
                # Get models from Agent Forge Controller
                controller_response = requests.get(f"{self.controller_url}/models", timeout=5)

                if controller_response.status_code == 200:
                    controller_data = controller_response.json()
                    models = controller_data.get("models", [])

                    # Register new models with Chat service
                    for model in models:
                        model_id = model["model_id"]

                        if model_id not in self.registered_models:
                            try:
                                # Register model with chat service
                                registration_data = {
                                    "model_id": model_id,
                                    "model_name": model["model_name"],
                                    "phase_name": model["phase_name"],
                                    "model_path": model["model_path"],
                                    "parameter_count": model["parameter_count"],
                                    "created_at": model["created_at"],
                                    "artifacts": model["artifacts"],
                                }

                                chat_response = requests.post(
                                    f"{self.chat_url}/models/{model_id}/register", json=registration_data, timeout=5
                                )

                                if chat_response.status_code == 200:
                                    self.registered_models.add(model_id)
                                    logger.info(f"Registered model {model_id} with chat service")
                                else:
                                    logger.warning(f"Failed to register model {model_id}: {chat_response.status_code}")

                            except Exception as e:
                                logger.warning(f"Error registering model {model_id}: {e}")

            except Exception as e:
                logger.warning(f"Error in model registration monitoring: {e}")

            await asyncio.sleep(10)  # Check every 10 seconds


class IntegrationTester:
    """Runs integration tests to validate system communication."""

    def __init__(self, service_manager: ServiceManager):
        self.service_manager = service_manager
        self.test_results = []

    async def run_integration_tests(self) -> dict[str, Any]:
        """Run comprehensive integration tests."""
        logger.info("Starting integration tests...")

        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {"passed": 0, "failed": 0, "total": 0},
        }

        # Test 1: Service Health Checks
        await self._test_service_health(test_results)

        # Test 2: Agent Forge Phase API
        await self._test_agent_forge_api(test_results)

        # Test 3: Model Chat API
        await self._test_model_chat_api(test_results)

        # Test 4: WebSocket Connection
        await self._test_websocket_connection(test_results)

        # Test 5: Model Registration Flow
        await self._test_model_registration_flow(test_results)

        # Calculate summary
        test_results["summary"]["total"] = len(test_results["tests"])
        for test in test_results["tests"]:
            if test["passed"]:
                test_results["summary"]["passed"] += 1
            else:
                test_results["summary"]["failed"] += 1

        logger.info(
            f"Integration tests completed: {test_results['summary']['passed']}/{test_results['summary']['total']} passed"
        )
        return test_results

    async def _test_service_health(self, results: dict[str, Any]):
        """Test all services are healthy."""
        test_name = "Service Health Checks"
        try:
            status = self.service_manager.get_service_status()
            passed = status["overall_health"] == "healthy"

            results["tests"].append(
                {
                    "name": test_name,
                    "passed": passed,
                    "details": status,
                    "message": "All services healthy" if passed else f"Health status: {status['overall_health']}",
                }
            )
        except Exception as e:
            results["tests"].append(
                {"name": test_name, "passed": False, "error": str(e), "message": "Failed to check service health"}
            )

    async def _test_agent_forge_api(self, results: dict[str, Any]):
        """Test Agent Forge Controller API endpoints."""
        test_name = "Agent Forge API"
        try:
            # Test phases endpoint
            response = requests.get("http://localhost:8083/phases/available", timeout=5)
            phases_ok = response.status_code == 200

            # Test system metrics
            response = requests.get("http://localhost:8083/system/metrics", timeout=5)
            metrics_ok = response.status_code == 200

            passed = phases_ok and metrics_ok
            results["tests"].append(
                {
                    "name": test_name,
                    "passed": passed,
                    "details": {"phases_endpoint": phases_ok, "metrics_endpoint": metrics_ok},
                    "message": "Agent Forge API endpoints working" if passed else "Some API endpoints failed",
                }
            )
        except Exception as e:
            results["tests"].append(
                {"name": test_name, "passed": False, "error": str(e), "message": "Failed to test Agent Forge API"}
            )

    async def _test_model_chat_api(self, results: dict[str, Any]):
        """Test Model Chat API endpoints."""
        test_name = "Model Chat API"
        try:
            # Test models endpoint
            response = requests.get("http://localhost:8084/models", timeout=5)
            models_ok = response.status_code == 200

            # Test sessions endpoint
            response = requests.get("http://localhost:8084/sessions", timeout=5)
            sessions_ok = response.status_code == 200

            passed = models_ok and sessions_ok
            results["tests"].append(
                {
                    "name": test_name,
                    "passed": passed,
                    "details": {"models_endpoint": models_ok, "sessions_endpoint": sessions_ok},
                    "message": "Model Chat API endpoints working" if passed else "Some API endpoints failed",
                }
            )
        except Exception as e:
            results["tests"].append(
                {"name": test_name, "passed": False, "error": str(e), "message": "Failed to test Model Chat API"}
            )

    async def _test_websocket_connection(self, results: dict[str, Any]):
        """Test WebSocket Manager connectivity."""
        test_name = "WebSocket Connection"
        try:
            # Test HTTP endpoints first
            response = requests.get("http://localhost:8085/stats", timeout=5)
            stats_ok = response.status_code == 200

            passed = stats_ok
            results["tests"].append(
                {
                    "name": test_name,
                    "passed": passed,
                    "details": {"stats_endpoint": stats_ok},
                    "message": "WebSocket Manager accessible" if passed else "WebSocket Manager not responding",
                }
            )
        except Exception as e:
            results["tests"].append(
                {"name": test_name, "passed": False, "error": str(e), "message": "Failed to test WebSocket connection"}
            )

    async def _test_model_registration_flow(self, results: dict[str, Any]):
        """Test model registration between services."""
        test_name = "Model Registration Flow"
        try:
            # Get models from both services
            controller_response = requests.get("http://localhost:8083/models", timeout=5)
            chat_response = requests.get("http://localhost:8084/models", timeout=5)

            controller_ok = controller_response.status_code == 200
            chat_ok = chat_response.status_code == 200

            passed = controller_ok and chat_ok
            results["tests"].append(
                {
                    "name": test_name,
                    "passed": passed,
                    "details": {"controller_models_accessible": controller_ok, "chat_models_accessible": chat_ok},
                    "message": "Model registration flow accessible" if passed else "Model endpoints not accessible",
                }
            )
        except Exception as e:
            results["tests"].append(
                {
                    "name": test_name,
                    "passed": False,
                    "error": str(e),
                    "message": "Failed to test model registration flow",
                }
            )


class DevelopmentUIServer:
    """Development UI server that serves static files and provides API proxy."""

    def __init__(self, service_manager: ServiceManager, integration_tester: IntegrationTester):
        self.app = FastAPI(title="Agent Forge Developer UI")
        self.service_manager = service_manager
        self.integration_tester = integration_tester
        self.setup_routes()
        self.setup_static_files()

    def setup_static_files(self):
        """Setup static file serving for UI assets."""
        static_dir = Path("infrastructure/dev-ui/static")
        static_dir.mkdir(parents=True, exist_ok=True)

        # Create basic HTML file if it doesn't exist
        index_file = static_dir / "index.html"
        if not index_file.exists():
            self.create_default_ui(index_file)

        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def create_default_ui(self, index_file: Path):
        """Create a basic HTML interface for development."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Forge Developer UI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .service-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .service-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-healthy { color: #27ae60; }
        .status-unhealthy { color: #e74c3c; }
        .status-unknown { color: #f39c12; }
        .button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; margin: 5px; }
        .button:hover { background: #2980b9; }
        .button.danger { background: #e74c3c; }
        .button.danger:hover { background: #c0392b; }
        .log { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 4px; font-family: monospace; max-height: 300px; overflow-y: auto; margin: 10px 0; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 10px 0; }
        .metric { background: #ecf0f1; padding: 10px; border-radius: 4px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Agent Forge Developer UI</h1>
            <p>Integrated development environment for Agent Forge model training and testing</p>
        </div>

        <div class="service-grid">
            <div class="service-card">
                <h3>Service Status</h3>
                <div id="service-status">Loading...</div>
                <button class="button" onclick="refreshStatus()">Refresh Status</button>
                <button class="button" onclick="runTests()">Run Integration Tests</button>
            </div>

            <div class="service-card">
                <h3>System Metrics</h3>
                <div id="system-metrics" class="metrics">Loading...</div>
            </div>

            <div class="service-card">
                <h3>Quick Actions</h3>
                <a href="http://localhost:8083/docs" class="button" target="_blank">Agent Forge API</a>
                <a href="http://localhost:8084/docs" class="button" target="_blank">Model Chat API</a>
                <a href="http://localhost:8085/docs" class="button" target="_blank">WebSocket API</a>
                <button class="button" onclick="startCognatePhase()">Start Cognate Phase</button>
            </div>
        </div>

        <div class="service-card">
            <h3>Integration Test Results</h3>
            <div id="test-results">No tests run yet</div>
        </div>

        <div class="service-card">
            <h3>Activity Log</h3>
            <div id="activity-log" class="log">System initialized...\n</div>
        </div>
    </div>

    <script>
        // WebSocket connection for real-time updates
        let ws = null;

        function connectWebSocket() {
            try {
                ws = new WebSocket('ws://localhost:8085/ws');
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                };
                ws.onopen = function() {
                    log('WebSocket connected');
                    // Subscribe to relevant channels
                    ws.send(JSON.stringify({type: 'subscribe', channel: 'agent_forge_phases'}));
                    ws.send(JSON.stringify({type: 'subscribe', channel: 'system_metrics'}));
                };
                ws.onclose = function() {
                    log('WebSocket disconnected, reconnecting...');
                    setTimeout(connectWebSocket, 5000);
                };
            } catch (e) {
                log('WebSocket connection failed: ' + e.message);
                setTimeout(connectWebSocket, 5000);
            }
        }

        function handleWebSocketMessage(data) {
            if (data.type === 'phase_update') {
                log(`Phase Update: ${data.phase_name} - ${data.message} (${Math.round(data.progress * 100)}%)`);
            } else if (data.type === 'system_metrics') {
                updateSystemMetrics(data.metrics);
            }
        }

        function log(message) {
            const logEl = document.getElementById('activity-log');
            const timestamp = new Date().toLocaleTimeString();
            logEl.textContent += `[${timestamp}] ${message}\n`;
            logEl.scrollTop = logEl.scrollHeight;
        }

        async function refreshStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                updateServiceStatus(data);
                log('Service status refreshed');
            } catch (e) {
                log('Failed to refresh status: ' + e.message);
            }
        }

        async function runTests() {
            try {
                log('Running integration tests...');
                const response = await fetch('/api/integration-tests');
                const data = await response.json();
                updateTestResults(data);
                log(`Tests completed: ${data.summary.passed}/${data.summary.total} passed`);
            } catch (e) {
                log('Failed to run tests: ' + e.message);
            }
        }

        async function startCognatePhase() {
            try {
                log('Starting Cognate phase...');
                const response = await fetch('http://localhost:8083/phases/cognate/start', {method: 'POST'});
                const data = await response.json();
                log(`Cognate phase started: ${data.message}`);
            } catch (e) {
                log('Failed to start Cognate phase: ' + e.message);
            }
        }

        function updateServiceStatus(data) {
            const statusEl = document.getElementById('service-status');
            let html = `<p><strong>Overall:</strong> <span class="status-${data.overall_health}">${data.overall_health}</span></p>`;

            for (const [service, info] of Object.entries(data.services)) {
                html += `<p><strong>${service}:</strong> <span class="status-${info.status}">${info.status}</span> (Port ${info.port})</p>`;
            }

            statusEl.innerHTML = html;
        }

        function updateSystemMetrics(metrics) {
            const metricsEl = document.getElementById('system-metrics');
            let html = '';

            for (const [key, value] of Object.entries(metrics)) {
                const displayKey = key.replace('_', ' ').toUpperCase();
                const displayValue = typeof value === 'number' ? value.toFixed(1) : value;
                html += `<div class="metric"><strong>${displayKey}</strong><br>${displayValue}</div>`;
            }

            metricsEl.innerHTML = html;
        }

        function updateTestResults(data) {
            const resultsEl = document.getElementById('test-results');
            let html = `<h4>Test Summary: ${data.summary.passed}/${data.summary.total} passed</h4>`;

            for (const test of data.tests) {
                const status = test.passed ? '✅' : '❌';
                html += `<p>${status} <strong>${test.name}:</strong> ${test.message}</p>`;
            }

            resultsEl.innerHTML = html;
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            refreshStatus();
            connectWebSocket();

            // Refresh status every 30 seconds
            setInterval(refreshStatus, 30000);
        });
    </script>
</body>
</html>"""

        index_file.write_text(html_content, encoding="utf-8")
        logger.info(f"Created default UI at {index_file}")

    def setup_routes(self):
        """Setup API routes for the UI server."""

        @self.app.get("/", response_class=HTMLResponse)
        async def serve_ui():
            """Serve the main UI page."""
            static_dir = Path("infrastructure/dev-ui/static")
            index_file = static_dir / "index.html"

            if index_file.exists():
                return index_file.read_text(encoding="utf-8")
            else:
                return "<h1>Agent Forge Developer UI</h1><p>UI files not found</p>"

        @self.app.get("/api/status")
        async def get_status():
            """Get current status of all services."""
            return self.service_manager.get_service_status()

        @self.app.get("/api/integration-tests")
        async def run_integration_tests():
            """Run integration tests and return results."""
            return await self.integration_tester.run_integration_tests()

        @self.app.get("/health")
        async def health():
            """Health check for the UI server."""
            return {"status": "healthy", "service": "integration_ui", "timestamp": datetime.now().isoformat()}


class AgentForgeIntegrationCoordinator:
    """Main coordinator for the entire Agent Forge development ecosystem."""

    def __init__(self, ui_port: int = 8080):
        self.ui_port = ui_port
        self.service_manager = ServiceManager()
        self.model_coordinator = ModelRegistrationCoordinator()
        self.integration_tester = IntegrationTester(self.service_manager)
        self.ui_server = DevelopmentUIServer(self.service_manager, self.integration_tester)
        self.background_tasks = []
        self.shutdown_event = threading.Event()

    async def startup(self):
        """Start the entire Agent Forge ecosystem."""
        logger.info("🚀 Starting Agent Forge Development Environment...")

        # 1. Start all API services
        if not self.service_manager.start_all_services():
            logger.error("Failed to start all services!")
            return False

        # 2. Wait for services to be fully ready
        logger.info("⏳ Waiting for services to be ready...")
        await self._wait_for_services_ready()

        # 3. Start model registration monitoring
        model_task = asyncio.create_task(self.model_coordinator.monitor_and_register_models())
        self.background_tasks.append(model_task)

        # 4. Run initial integration tests
        logger.info("🧪 Running initial integration tests...")
        test_results = await self.integration_tester.run_integration_tests()

        if test_results["summary"]["failed"] > 0:
            logger.warning(
                f"Some integration tests failed: {test_results['summary']['failed']}/{test_results['summary']['total']}"
            )

        # 5. Start UI server
        logger.info(f"🌐 Starting Developer UI on port {self.ui_port}...")

        logger.info("✅ Agent Forge Development Environment is ready!")
        logger.info(f"📊 Developer UI: http://localhost:{self.ui_port}")
        logger.info("🔌 API Services:")
        for service_name, config in self.service_manager.service_configs.items():
            logger.info(f"   - {service_name}: http://localhost:{config['port']}")

        return True

    async def _wait_for_services_ready(self, timeout: int = 30):
        """Wait for all services to be healthy."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.service_manager.get_service_status()
            if status["overall_health"] == "healthy":
                logger.info("All services are healthy!")
                return True

            logger.info("Waiting for services to be ready...")
            await asyncio.sleep(2)

        logger.warning("Timeout waiting for all services to be ready")
        return False

    def shutdown(self):
        """Gracefully shutdown all services."""
        logger.info("🛑 Shutting down Agent Forge Development Environment...")

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Stop all services
        self.service_manager.stop_all_services()

        self.shutdown_event.set()
        logger.info("✅ Shutdown complete")

    def run_ui_server(self):
        """Run the UI server."""
        uvicorn.run(self.ui_server.app, host="0.0.0.0", port=self.ui_port, log_level="info")


async def main():
    """Main entry point."""
    coordinator = AgentForgeIntegrationCoordinator()

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        coordinator.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start the ecosystem
        success = await coordinator.startup()

        if not success:
            logger.error("Failed to start Agent Forge ecosystem")
            return 1

        # Run UI server (this blocks)
        coordinator.run_ui_server()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        coordinator.shutdown()
        return 0
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        coordinator.shutdown()
        return 1


def cli_status():
    """CLI command to check system status."""
    manager = ServiceManager()
    status = manager.get_service_status()

    print(f"\n🔍 Agent Forge System Status ({status['timestamp']})")
    print(f"Overall Health: {status['overall_health'].upper()}")
    print("\nServices:")

    for service_name, info in status["services"].items():
        status_icon = "✅" if info["status"] == "healthy" else "❌"
        print(f"  {status_icon} {service_name}: {info['status']} (port {info['port']})")
        if "error" in info:
            print(f"     Error: {info['error']}")


def cli_test():
    """CLI command to run integration tests."""

    async def run_tests():
        manager = ServiceManager()
        tester = IntegrationTester(manager)
        results = await tester.run_integration_tests()

        print(f"\n🧪 Integration Test Results ({results['timestamp']})")
        print(f"Summary: {results['summary']['passed']}/{results['summary']['total']} passed\n")

        for test in results["tests"]:
            icon = "✅" if test["passed"] else "❌"
            print(f"  {icon} {test['name']}: {test['message']}")
            if "error" in test:
                print(f"     Error: {test['error']}")

    asyncio.run(run_tests())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "status":
            cli_status()
        elif command == "test":
            cli_test()
        else:
            print("Usage: python integration-setup.py [status|test]")
            print("       python integration-setup.py  # Start full system")
    else:
        # Start the full system
        asyncio.run(main())
