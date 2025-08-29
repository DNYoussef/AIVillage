#!/usr/bin/env python3
"""
Service Mesh Startup Script

Quick start script for the AIVillage service mesh in production mode.
Launches all services with proper service discovery and health monitoring.

Usage:
    python scripts/start_service_mesh.py
    python scripts/start_service_mesh.py --dev    # Development mode
    python scripts/start_service_mesh.py --service gateway  # Single service
"""

import asyncio
import logging
from pathlib import Path
import sys

import click

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.service_mesh.production_service_manager import ProductionServiceManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--dev", is_flag=True, help="Run in development mode")
@click.option("--service", help="Start only a specific service")
@click.option("--port", default=8090, help="Service mesh API port")
@click.option("--config", help="Custom configuration file")
async def start_service_mesh(dev: bool, service: str, port: int, config: str):
    """Start the AIVillage service mesh."""

    logger.info("🌐 Starting AIVillage Service Mesh...")

    if dev:
        logger.info("Running in DEVELOPMENT mode")
    else:
        logger.info("Running in PRODUCTION mode")

    try:
        # Initialize service manager
        service_manager = ProductionServiceManager(config)

        if service:
            # Start single service
            logger.info(f"Starting single service: {service}")
            success = await service_manager.start_service(service)

            if success:
                logger.info(f"✅ Service {service} started successfully")

                # Keep running
                while True:
                    await asyncio.sleep(10)
                    status = service_manager.get_service_status()
                    if not status.get("running", False):
                        break
            else:
                logger.error(f"❌ Failed to start service {service}")
                sys.exit(1)

        else:
            # Start all services
            await service_manager.start_all_services()

            # Start service mesh API
            logger.info(f"🚀 Starting Service Mesh API on port {port}")

            # Import uvicorn here to avoid import issues
            import uvicorn

            # Run the service mesh API
            uvicorn.run(
                "infrastructure.service_mesh.service_mesh_api:app",
                host="0.0.0.0",
                port=port,
                log_level="info" if not dev else "debug",
                reload=dev,
            )

    except KeyboardInterrupt:
        logger.info("🛑 Shutting down service mesh...")

        if "service_manager" in locals():
            await service_manager.stop_all_services()

        logger.info("✅ Service mesh stopped")

    except Exception as e:
        logger.error(f"❌ Service mesh startup failed: {e}")
        sys.exit(1)


def main():
    """Main entry point with async support."""
    import inspect

    # Get the click command
    command = start_service_mesh

    if inspect.iscoroutinefunction(command.callback):
        # Wrap async command
        original_callback = command.callback

        def sync_wrapper(*args, **kwargs):
            return asyncio.run(original_callback(*args, **kwargs))

        command.callback = sync_wrapper

    # Run the command
    command()


if __name__ == "__main__":
    main()
