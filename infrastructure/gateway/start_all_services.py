#!/usr/bin/env python3
"""
Start All AIVillage Services for Complete Integration

Runs all three services required by the Admin UI:
- Agent Forge Controller API (port 8083)
- Model Chat API (port 8084)
- WebSocket Manager (port 8085)
"""

import asyncio
import logging
import multiprocessing
import os
import sys
from pathlib import Path
import signal
import time

# Add API directory to path
api_dir = Path(__file__).parent / "api"
sys.path.insert(0, str(api_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_agent_forge_api():
    """Run Agent Forge Controller API on port 8083."""
    try:
        import agent_forge_controller
        import uvicorn

        logger.info("Starting Agent Forge Controller API on port 8083...")
        uvicorn.run(agent_forge_controller.app, host="0.0.0.0", port=8083, log_level="info")
    except Exception as e:
        logger.exception(f"Failed to start Agent Forge API: {e}")


def run_model_chat_api():
    """Run Model Chat API on port 8084."""
    try:
        import model_chat
        import uvicorn

        logger.info("Starting Model Chat API on port 8084...")
        uvicorn.run(model_chat.app, host="0.0.0.0", port=8084, log_level="info")
    except Exception as e:
        logger.exception(f"Failed to start Model Chat API: {e}")


def run_websocket_manager():
    """Run WebSocket Manager on port 8085."""
    try:
        import websocket_manager
        import uvicorn

        logger.info("Starting WebSocket Manager on port 8085...")
        uvicorn.run(websocket_manager.app, host="0.0.0.0", port=8085, log_level="info")
    except Exception as e:
        logger.exception(f"Failed to start WebSocket Manager: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal, stopping all services...")
    sys.exit(0)


def main():
    """Start all services using multiprocessing."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("🚀 Starting AIVillage Service Architecture")
    logger.info("   Agent Forge Controller API: http://localhost:8083")
    logger.info("   Model Chat API: http://localhost:8084")
    logger.info("   WebSocket Manager: ws://localhost:8085/ws")

    # Start all services as separate processes
    processes = []

    try:
        # Agent Forge Controller API
        p1 = multiprocessing.Process(target=run_agent_forge_api)
        p1.start()
        processes.append(p1)

        # Model Chat API
        p2 = multiprocessing.Process(target=run_model_chat_api)
        p2.start()
        processes.append(p2)

        # WebSocket Manager
        p3 = multiprocessing.Process(target=run_websocket_manager)
        p3.start()
        processes.append(p3)

        logger.info("✅ All services started successfully!")
        logger.info("   Access Admin UI at: http://localhost:3000 (if React app is running)")
        logger.info("   Or use the simple HTML interface in gateway/admin_interface.html")

        # Wait for all processes
        for process in processes:
            process.join()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.exception(f"Error in main process: {e}")
    finally:
        # Clean shutdown
        logger.info("Stopping all services...")
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()

        logger.info("🛑 All services stopped")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
