#!/usr/bin/env python3
"""MCP Server Management Script

This script manages the lifecycle of MCP servers for the AIVillage project.
It can start, stop, and monitor the configured MCP servers.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCPServerManager:
    """Manages MCP servers for the project."""

    def __init__(self, config_path: str = "mcp_config.json"):
        self.config_path = config_path
        self.config = {}
        self.processes = {}
        self.shutdown_event = asyncio.Event()

    def load_config(self) -> None:
        """Load MCP server configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)

    def setup_environment(self) -> None:
        """Set up environment variables and directories."""
        # Create necessary directories
        os.makedirs(".mcp", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("config", exist_ok=True)

        # Set up Python path
        if "PYTHONPATH" not in os.environ:
            os.environ["PYTHONPATH"] = "."

        logger.info("Environment setup complete")

    async def start_server(self, name: str, config: Dict) -> Optional[subprocess.Popen]:
        """Start a single MCP server."""
        if config.get("disabled", False):
            logger.info(f"Skipping disabled server: {name}")
            return None

        try:
            # Build command
            command = [config["command"]] + config.get("args", [])

            # Set up environment
            env = os.environ.copy()
            env.update(config.get("env", {}))

            # Start process
            logger.info(f"Starting MCP server: {name}")
            logger.debug(f"Command: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Wait a moment to check if it started successfully
            await asyncio.sleep(1)

            if process.poll() is None:
                logger.info(f"Successfully started MCP server: {name} (PID: {process.pid})")
                return process
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Failed to start MCP server {name}: {stderr}")
                return None

        except Exception as e:
            logger.error(f"Error starting MCP server {name}: {e}")
            return None

    async def stop_server(self, name: str, process: subprocess.Popen) -> None:
        """Stop a single MCP server."""
        try:
            logger.info(f"Stopping MCP server: {name} (PID: {process.pid})")

            # Send SIGTERM
            process.terminate()

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process(process)),
                    timeout=10
                )
                logger.info(f"MCP server {name} stopped gracefully")
            except asyncio.TimeoutError:
                # Force kill if graceful shutdown failed
                logger.warning(f"Force killing MCP server: {name}")
                process.kill()
                await asyncio.create_task(self._wait_for_process(process))

        except Exception as e:
            logger.error(f"Error stopping MCP server {name}: {e}")

    async def _wait_for_process(self, process: subprocess.Popen) -> None:
        """Wait for a process to terminate."""
        while process.poll() is None:
            await asyncio.sleep(0.1)

    async def start_all_servers(self) -> None:
        """Start all configured MCP servers."""
        logger.info("Starting all MCP servers...")

        servers = self.config.get("mcpServers", {})

        for name, config in servers.items():
            process = await self.start_server(name, config)
            if process:
                self.processes[name] = process

        logger.info(f"Started {len(self.processes)} MCP servers")

    async def stop_all_servers(self) -> None:
        """Stop all running MCP servers."""
        logger.info("Stopping all MCP servers...")

        # Stop servers concurrently
        tasks = []
        for name, process in self.processes.items():
            tasks.append(self.stop_server(name, process))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.processes.clear()
        logger.info("All MCP servers stopped")

    async def monitor_servers(self) -> None:
        """Monitor server health and restart if needed."""
        while not self.shutdown_event.is_set():
            # Check each server
            dead_servers = []
            for name, process in self.processes.items():
                if process.poll() is not None:
                    logger.warning(f"MCP server {name} has died (exit code: {process.returncode})")
                    dead_servers.append(name)

            # Restart dead servers
            for name in dead_servers:
                del self.processes[name]
                config = self.config["mcpServers"][name]
                process = await self.start_server(name, config)
                if process:
                    self.processes[name] = process

            # Wait before next check
            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=30)
                break  # Shutdown requested
            except asyncio.TimeoutError:
                continue  # Continue monitoring

    async def run(self) -> None:
        """Main run loop."""
        # Set up signal handlers
        def signal_handler():
            logger.info("Received shutdown signal")
            self.shutdown_event.set()

        if sys.platform != "win32":
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop = asyncio.get_event_loop()
                loop.add_signal_handler(sig, signal_handler)

        try:
            # Load configuration and setup
            self.load_config()
            self.setup_environment()

            # Start servers
            await self.start_all_servers()

            if not self.processes:
                logger.error("No servers started successfully")
                return

            # Monitor servers
            logger.info("MCP servers are running. Press Ctrl+C to stop.")
            await self.monitor_servers()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            # Clean shutdown
            await self.stop_all_servers()

    def status(self) -> None:
        """Print status of all configured servers."""
        servers = self.config.get("mcpServers", {})

        print("MCP Server Status:")
        print("=" * 50)

        for name, config in servers.items():
            status = "DISABLED" if config.get("disabled", False) else "CONFIGURED"
            description = config.get("description", "No description")

            # Check if server is running
            if name in self.processes:
                process = self.processes[name]
                if process.poll() is None:
                    status = f"RUNNING (PID: {process.pid})"
                else:
                    status = f"DEAD (Exit: {process.returncode})"

            print(f"{name:20} | {status:20} | {description}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Server Manager")
    parser.add_argument(
        "action",
        choices=["start", "stop", "status", "restart"],
        help="Action to perform"
    )
    parser.add_argument(
        "--config",
        default="mcp_config.json",
        help="Configuration file path"
    )

    args = parser.parse_args()

    manager = MCPServerManager(args.config)

    if args.action == "start":
        await manager.run()
    elif args.action == "status":
        manager.load_config()
        manager.status()
    elif args.action == "stop":
        # TODO: Implement proper stop via IPC
        print("Stop functionality requires running instance communication")
        sys.exit(1)
    elif args.action == "restart":
        # TODO: Implement restart
        print("Restart functionality not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
