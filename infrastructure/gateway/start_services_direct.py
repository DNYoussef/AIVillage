#!/usr/bin/env python3
"""
Direct Service Startup - Bypasses problematic imports

Starts all three services directly by importing individual modules.
"""

import logging
import multiprocessing
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def start_model_chat_service():
    """Start Model Chat API on port 8084."""
    try:
        # Add API directory to path
        api_dir = Path(__file__).parent / "api"
        sys.path.insert(0, str(api_dir))

        # Import directly
        import model_chat
        import uvicorn

        logger.info("Starting Model Chat API on port 8084...")
        uvicorn.run(model_chat.app, host="0.0.0.0", port=8084, log_level="info")

    except Exception as e:
        logger.exception(f"Failed to start Model Chat API: {e}")


def start_websocket_service():
    """Start WebSocket Manager on port 8085."""
    try:
        # Add API directory to path
        api_dir = Path(__file__).parent / "api"
        sys.path.insert(0, str(api_dir))

        # Import directly
        import uvicorn
        import websocket_manager

        logger.info("Starting WebSocket Manager on port 8085...")
        uvicorn.run(websocket_manager.app, host="0.0.0.0", port=8085, log_level="info")

    except Exception as e:
        logger.exception(f"Failed to start WebSocket Manager: {e}")


def start_agent_forge_service():
    """Start Agent Forge Controller API on port 8083."""
    try:
        # Add API directory to path
        api_dir = Path(__file__).parent / "api"
        sys.path.insert(0, str(api_dir))

        # Import directly
        import agent_forge_controller
        import uvicorn

        logger.info("Starting Agent Forge Controller API on port 8083...")
        uvicorn.run(agent_forge_controller.app, host="0.0.0.0", port=8083, log_level="info")

    except Exception as e:
        logger.exception(f"Failed to start Agent Forge Controller API: {e}")


def main():
    """Start all services using multiprocessing."""
    if len(sys.argv) > 1:
        service = sys.argv[1].lower()
        if service == "chat":
            start_model_chat_service()
        elif service == "websocket":
            start_websocket_service()
        elif service == "agent_forge":
            start_agent_forge_service()
        else:
            print(f"Unknown service: {service}")
            print("Available services: chat, websocket, agent_forge")
        return

    # Start all services
    logger.info("Starting AIVillage Service Architecture...")
    logger.info("  Agent Forge Controller API: http://localhost:8083")
    logger.info("  Model Chat API: http://localhost:8084")
    logger.info("  WebSocket Manager: ws://localhost:8085/ws")

    processes = []

    try:
        # Start Model Chat API
        p1 = multiprocessing.Process(target=start_model_chat_service)
        p1.start()
        processes.append(("Model Chat", p1))

        # Start WebSocket Manager
        p2 = multiprocessing.Process(target=start_websocket_service)
        p2.start()
        processes.append(("WebSocket Manager", p2))

        # Start Agent Forge Controller API
        p3 = multiprocessing.Process(target=start_agent_forge_service)
        p3.start()
        processes.append(("Agent Forge Controller", p3))

        logger.info("All services started successfully!")
        logger.info("Press Ctrl+C to stop all services")

        # Wait for all processes
        for name, process in processes:
            process.join()

    except KeyboardInterrupt:
        logger.info("Shutting down all services...")
    except Exception as e:
        logger.exception(f"Error in main: {e}")
    finally:
        # Clean shutdown
        for name, process in processes:
            if process.is_alive():
                logger.info(f"Stopping {name}...")
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()

        logger.info("All services stopped")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
