#!/usr/bin/env python3
"""HypeRAG MCP Server Startup Script.

Simple script to start the HypeRAG MCP server with proper configuration.
"""

import asyncio
import logging
from pathlib import Path
import sys

from mcp_servers.hyperag.server import HypeRAGMCPServer

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def main() -> None:
    """Start the HypeRAG MCP Server."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting HypeRAG MCP Server...")

    # Configuration file path
    config_path = project_root / "config" / "hyperag_mcp.yaml"

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create the configuration file or copy from the template")
        return

    # Create and start server
    server = HypeRAGMCPServer(str(config_path))

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.exception(f"Server error: {e!s}")
        raise
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e!s}")
        sys.exit(1)
