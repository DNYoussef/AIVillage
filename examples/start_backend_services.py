#!/usr/bin/env python3
"""
Start All Agent Forge Backend Services

This script starts all the backend services in the correct order:
1. Unified Agent Forge Backend (port 8083) - Main API with UnifiedCognateRefiner
2. Model Chat API (port 8084) - Chat interface for trained models
3. WebSocket Manager (port 8085) - Real-time updates and progress
"""

import asyncio
import logging
from pathlib import Path
import subprocess
import sys
import time

import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_service_in_background(service_name, script_path, port):
    """Start a service in the background."""
    try:
        logger.info(f"Starting {service_name} on port {port}...")

        # Start the service
        process = subprocess.Popen([sys.executable, str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Give it a moment to start
        time.sleep(3)

        # Check if it's responding
        try:
            response = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/health"], capture_output=True, text=True, timeout=5
            )

            if response.returncode == 0:
                logger.info(f"[OK] {service_name} is responding on port {port}")
                return process
            else:
                logger.warning(f"[WARN] {service_name} may not be fully started yet")
                return process

        except Exception as e:
            logger.warning(f"[WARN] Could not verify {service_name} status: {e}")
            return process

    except Exception as e:
        logger.error(f"[ERROR] Failed to start {service_name}: {e}")
        return None


async def check_service_health(port, service_name):
    """Check if a service is healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"[OK] {service_name} (:{port}) - {data.get('status', 'unknown')}")
                return True
            else:
                logger.error(f"[ERROR] {service_name} (:{port}) - HTTP {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"[ERROR] {service_name} (:{port}) - {e}")
        return False


async def verify_integration():
    """Verify that all services are integrated correctly."""
    logger.info("Verifying service integration...")

    # Check each service
    services = [(8083, "Unified Agent Forge Backend"), (8084, "Model Chat API"), (8085, "WebSocket Manager")]

    all_healthy = True
    for port, name in services:
        healthy = await check_service_health(port, name)
        all_healthy = all_healthy and healthy

    if all_healthy:
        logger.info("[SUCCESS] All services are healthy and integrated!")

        # Test the main API
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8083/")
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"[OK] Main API: {data['service']} v{data['version']}")
                    logger.info(f"[OK] Architecture: {data['architecture']['model_type']}")
                    logger.info(f"[OK] Parameters: {data['architecture']['exact_parameters']:,}")
                    logger.info(f"[OK] Real training: {data['real_training_available']}")

                    # Check UnifiedCognateRefiner features
                    unified_features = [f for f in data["features"] if "UnifiedCognateRefiner" in f]
                    if unified_features:
                        logger.info("[OK] UnifiedCognateRefiner integration confirmed:")
                        for feature in unified_features:
                            logger.info(f"   - {feature}")

        except Exception as e:
            logger.error(f"[ERROR] Could not verify main API: {e}")
            all_healthy = False

    return all_healthy


def main():
    """Main function to start all services."""
    logger.info("Starting Agent Forge Backend Services...")
    logger.info("=" * 60)

    # Define services
    services = [
        {
            "name": "Unified Agent Forge Backend",
            "script": "infrastructure/gateway/unified_agent_forge_backend.py",
            "port": 8083,
        },
        {"name": "Model Chat API", "script": "infrastructure/gateway/api/model_chat.py", "port": 8084},
        {"name": "WebSocket Manager", "script": "infrastructure/gateway/api/websocket_manager.py", "port": 8085},
    ]

    processes = []

    # Start each service
    for service in services:
        script_path = Path(service["script"])
        if script_path.exists():
            process = start_service_in_background(service["name"], script_path, service["port"])
            if process:
                processes.append((service, process))
        else:
            logger.error(f"[ERROR] Script not found: {script_path}")

    if not processes:
        logger.error("[ERROR] No services could be started")
        return False

    logger.info(f"\nStarted {len(processes)} services. Waiting for initialization...")
    time.sleep(5)

    # Verify integration
    logger.info("\nVerifying service integration...")
    integration_success = asyncio.run(verify_integration())

    if integration_success:
        logger.info("\n" + "=" * 60)
        logger.info("[SUCCESS] ALL BACKEND SERVICES STARTED SUCCESSFULLY!")
        logger.info("")
        logger.info("Available services:")
        logger.info("- Main API: http://localhost:8083")
        logger.info("- Model Chat: http://localhost:8084")
        logger.info("- WebSocket: ws://localhost:8085/ws")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Open admin UI in browser: http://localhost:8083")
        logger.info("2. Click 'Start Cognate Training' to create UnifiedCognateRefiner models")
        logger.info("3. Monitor real-time progress via WebSocket")
        logger.info("4. Test trained models via chat interface")
        logger.info("")
        logger.info("Press Ctrl+C to stop all services")

        try:
            # Keep the main process running
            while True:
                time.sleep(10)

        except KeyboardInterrupt:
            logger.info("\nShutting down services...")
            for service, process in processes:
                logger.info(f"Stopping {service['name']}...")
                process.terminate()
                process.wait()
            logger.info("All services stopped.")

    else:
        logger.error("[ERROR] Service integration failed")
        # Clean up
        for service, process in processes:
            process.terminate()
            process.wait()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
