#!/usr/bin/env python3
"""HypeRAG MCP Server - Standard MCP Implementation.

Standard MCP server implementation for HypeRAG using stdio transport.
Compatible with Claude Desktop and other MCP clients.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mcp_servers.hyperag.auth import AuthContext, PermissionManager
from mcp_servers.hyperag.models import ModelRegistry
from mcp_servers.hyperag.protocol import MCPProtocolHandler
from software.hyper_rag.hyper_rag_pipeline import HyperRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HypeRAGMCPServer:
    """Standard MCP server for HypeRAG."""

    def __init__(self) -> None:
        self.permission_manager = None
        self.model_registry = None
        self.protocol_handler = None
        self.initialized = False
        # Local HyperRAG pipeline used for all queries and memory operations
        self.pipeline = HyperRAGPipeline()

    async def initialize(self) -> None:
        """Initialize server components."""
        if self.initialized:
            return

        try:
            # Initialize permission manager with secure config
            import os

            jwt_secret = os.getenv("MCP_SERVER_SECRET")
            if not jwt_secret or len(jwt_secret) < 32:
                logger.warning("Using default MCP secret - CHANGE IN PRODUCTION!")
                jwt_secret = "INSECURE_DEFAULT_MCP_SECRET_CHANGE_IMMEDIATELY_IN_PRODUCTION"

            self.permission_manager = PermissionManager(jwt_secret=jwt_secret, enable_audit=False)

            # Initialize model registry
            self.model_registry = ModelRegistry()

            # Initialize protocol handler
            self.protocol_handler = MCPProtocolHandler(
                permission_manager=self.permission_manager,
                model_registry=self.model_registry,
                storage_backend=None,
            )

            self.initialized = True
            logger.info("HypeRAG MCP server initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize server: {e}")
            sys.exit(1)

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP request."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            # Create auth context for local MCP usage
            from datetime import datetime, timedelta

            auth_context = AuthContext(
                user_id="mcp-user",
                agent_id="mcp-agent",
                session_id="mcp-session",
                role="admin",
                permissions={"hyperag:*"},
                expires_at=datetime.now() + timedelta(hours=24),
                ip_address="127.0.0.1",
            )

            # Handle standard MCP methods
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": True,
                            "resources": True,
                            "logging": True,
                        },
                    },
                }

            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "hyperag_query",
                                "description": "Query the HypeRAG knowledge graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "Natural language query",
                                        },
                                        "context": {
                                            "type": "string",
                                            "description": "Additional context",
                                        },
                                    },
                                    "required": ["query"],
                                },
                            },
                            {
                                "name": "hyperag_memory",
                                "description": "Store or retrieve memories",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "action": {
                                            "type": "string",
                                            "enum": ["store", "retrieve", "search"],
                                            "description": "Memory operation",
                                        },
                                        "content": {
                                            "type": "string",
                                            "description": "Content to store or search for",
                                        },
                                        "tags": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Optional tags",
                                        },
                                    },
                                    "required": ["action"],
                                },
                            },
                        ]
                    },
                }

            if method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if tool_name == "hyperag_query":
                    result = await self._handle_query(arguments, auth_context)
                elif tool_name == "hyperag_memory":
                    result = await self._handle_memory(arguments, auth_context)
                else:
                    msg = f"Unknown tool: {tool_name}"
                    raise ValueError(msg)

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
                }

            if method == "resources/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "resources": [
                            {
                                "uri": "hyperag://knowledge-graph",
                                "name": "HypeRAG Knowledge Graph",
                                "description": "Access to the hypergraph knowledge base",
                                "mimeType": "application/json",
                            },
                            {
                                "uri": "hyperag://memory-index",
                                "name": "HypeRAG Memory Index",
                                "description": "Episodic memory storage and retrieval",
                                "mimeType": "application/json",
                            },
                        ]
                    },
                }

            # Unknown method
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        except Exception as e:
            logger.exception(f"Error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32000, "message": f"Server error: {e!s}"},
            }

    async def _handle_query(self, arguments: dict[str, Any], context: AuthContext) -> dict[str, Any]:
        """Handle hyperag_query tool call."""
        query = arguments.get("query", "")

        retrieval = await self.pipeline.search(query)
        answer = retrieval.items[0].content if retrieval.items else ""
        sources = [{"id": item.item_id, "content": item.content} for item in retrieval.items]

        return {"answer": answer, "sources": sources, "metrics": retrieval.metrics}

    async def _handle_memory(self, arguments: dict[str, Any], context: AuthContext) -> dict[str, Any]:
        """Handle hyperag_memory tool call."""
        action = arguments.get("action")

        if action == "store":
            content = arguments.get("content", "")
            belief = float(arguments.get("belief", 0.0))
            item_id = await self.pipeline.ingest_knowledge(content, source_confidence=belief)
            return {"status": "stored", "item_id": item_id}

        if action == "search":
            query = arguments.get("content", "")
            retrieval = await self.pipeline.search(query)
            return {"results": [{"id": i.item_id, "content": i.content} for i in retrieval.items]}

        return {"status": "unknown action"}

    async def run(self) -> None:
        """Run the MCP server using stdio transport."""
        await self.initialize()

        logger.info("HypeRAG MCP server starting...")

        # Read from stdin, write to stdout
        while True:
            try:
                # Read JSON-RPC message from stdin
                line = sys.stdin.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse JSON-RPC request
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.exception(f"Invalid JSON: {e}")
                    continue

                # Handle request
                response = await self.handle_request(request)

                # Write JSON-RPC response to stdout
                print(json.dumps(response), flush=True)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.exception(f"Error in main loop: {e}")
                break


async def main() -> None:
    """Main entry point."""
    server = HypeRAGMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
