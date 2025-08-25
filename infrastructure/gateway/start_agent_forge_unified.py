#!/usr/bin/env python3
"""
Unified Agent Forge Backend Launcher - Production Ready
Starts all services needed for Agent Forge with proper error handling.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent paths to sys.path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Service ports configuration
SERVICES = {
    "main_gateway": {"port": 8080, "module": "simple_server", "description": "Main Gateway Server", "optional": False},
    "agent_forge_api": {
        "port": 8083,
        "module": "simple_agent_forge_api",  # Use simpler version that works
        "description": "Agent Forge API",
        "optional": False,
    },
    "websocket": {"port": 8085, "module": "websocket_server", "description": "WebSocket Server", "optional": False},
}


class ServiceManager:
    def __init__(self):
        self.processes = {}
        self.running = False

    def start_service(self, name: str, config: dict) -> Optional[subprocess.Popen]:
        """Start a single service."""
        try:
            logger.info(f"🚀 Starting {config['description']} on port {config['port']}...")

            # Construct the command
            module_path = Path(__file__).parent / f"{config['module']}.py"

            if not module_path.exists():
                logger.error(f"❌ Module not found: {module_path}")
                if not config.get("optional", False):
                    raise FileNotFoundError(f"Required module not found: {module_path}")
                return None

            # Start the process
            process = subprocess.Popen(
                [sys.executable, str(module_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(current_dir)
            )

            # Give it a moment to start
            time.sleep(1)

            # Check if it's still running
            if process.poll() is not None:
                logger.error(f"❌ {config['description']} failed to start")
                if not config.get("optional", False):
                    raise RuntimeError(f"Required service failed: {name}")
                return None

            logger.info(f"✅ {config['description']} started (PID: {process.pid})")
            return process

        except Exception as e:
            logger.error(f"❌ Failed to start {name}: {e}")
            if not config.get("optional", False):
                raise
            return None

    def start_all(self):
        """Start all services."""
        logger.info("=" * 60)
        logger.info("🎯 Starting Agent Forge Unified Backend")
        logger.info("=" * 60)

        self.running = True

        for name, config in SERVICES.items():
            process = self.start_service(name, config)
            if process:
                self.processes[name] = process
            time.sleep(1)  # Stagger startup

        logger.info("\n" + "=" * 60)
        logger.info("🎉 Agent Forge Backend Started!")
        logger.info("=" * 60)
        logger.info("📊 Service Endpoints:")
        logger.info("   • Main Gateway:     http://localhost:8080")
        logger.info("   • Agent Forge API:  http://localhost:8083")
        logger.info("   • WebSocket:        ws://localhost:8085/ws")
        logger.info("\n🌐 UI Access:")
        logger.info("   • Admin Interface:  http://localhost:8080/admin_interface.html")
        logger.info("   • React Dev:        http://localhost:5173 (run: npm run dev)")
        logger.info("\n⚡ Ready for Agent Forge operations!")

    def monitor(self):
        """Monitor running services."""
        try:
            while self.running:
                time.sleep(5)

                # Check service health
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        logger.warning(f"⚠️ {SERVICES[name]['description']} stopped (exit code: {process.returncode})")

                        # Try to restart if not optional
                        if not SERVICES[name].get("optional", False):
                            logger.info(f"🔄 Attempting to restart {name}...")
                            new_process = self.start_service(name, SERVICES[name])
                            if new_process:
                                self.processes[name] = new_process
                            else:
                                logger.error(f"❌ Failed to restart {name}")

        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping Agent Forge Backend...")
            self.stop_all()

    def stop_all(self):
        """Stop all services."""
        self.running = False

        for name, process in self.processes.items():
            if process.poll() is None:  # Still running
                logger.info(f"   Stopping {SERVICES[name]['description']}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"   Force killing {name}...")
                    process.kill()

        logger.info("✅ All services stopped")


def main():
    """Main entry point."""
    manager = ServiceManager()

    try:
        # Start all services
        manager.start_all()

        # Monitor services
        manager.monitor()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        manager.stop_all()
        sys.exit(1)


if __name__ == "__main__":
    main()
