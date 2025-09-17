#!/usr/bin/env python3
"""
BetaNet Bridge Service - Python service for TypeScript integration

This service provides a JSON-RPC interface for TypeScript components to interact
with the existing Python BetaNet infrastructure. It replaces mock implementations
with actual network communication and constitutional validation.
"""

import argparse
import asyncio
import json
import logging
import socket
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, 'C:/Users/17175/Desktop/AIVillage')

# Import existing BetaNet infrastructure
from infrastructure.p2p.betanet.constitutional_transport import (
    ConstitutionalBetaNetTransport,
    ConstitutionalBetaNetService,
    ConstitutionalTransportConfig,
    ConstitutionalMessage,
    ConstitutionalTransportMode,
)
from infrastructure.p2p.betanet.constitutional_frames import (
    ConstitutionalFrameProcessor,
    ConstitutionalFrame,
    ConstitutionalFrameType,
    ConstitutionalTier,
)
from infrastructure.p2p.betanet.constitutional_mixnodes import (
    create_constitutional_routing_request,
)
from infrastructure.fog.bridges.betanet_integration import (
    create_betanet_transport,
    get_betanet_capabilities,
)
from core.agents.bridges.fog_tools import CreateSandboxTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for the BetaNet bridge service."""
    host: str = 'localhost'
    port: int = 9876
    constitutional_tier: str = 'Silver'
    privacy_mode: str = 'standard'
    enable_mixnode_routing: bool = True
    enable_zero_knowledge_proofs: bool = True
    enable_fog_integration: bool = True


class JsonRpcServer:
    """JSON-RPC server for handling TypeScript requests."""

    def __init__(self, bridge: 'BetaNetBridge'):
        self.bridge = bridge
        self.server: Optional[asyncio.Server] = None
        self.clients: List[asyncio.StreamWriter] = []

    async def start(self, host: str, port: int):
        """Start the JSON-RPC server."""
        self.server = await asyncio.start_server(
            self.handle_client,
            host,
            port
        )

        logger.info(f"JSON-RPC server listening on {host}:{port}")
        print(f"BetaNet Bridge ready on {host}:{port}", flush=True)

        async with self.server:
            await self.server.serve_forever()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a connected client."""
        addr = writer.get_extra_info('peername')
        logger.info(f"Client connected from {addr}")
        self.clients.append(writer)

        try:
            while True:
                # Read JSON-RPC request (newline delimited)
                data = await reader.readline()
                if not data:
                    break

                try:
                    request = json.loads(data.decode('utf-8'))
                    response = await self.handle_request(request)

                    # Send response
                    response_data = json.dumps(response) + '\n'
                    writer.write(response_data.encode('utf-8'))
                    await writer.drain()

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = self.create_error_response(
                        None, -32700, "Parse error"
                    )
                    writer.write(json.dumps(error_response).encode('utf-8') + b'\n')
                    await writer.drain()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            logger.info(f"Client disconnected from {addr}")
            self.clients.remove(writer)
            writer.close()
            await writer.wait_closed()

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC request."""
        request_id = request.get('id')
        method = request.get('method')
        params = request.get('params', {})

        logger.debug(f"Handling request: {method}")

        try:
            # Route to appropriate handler
            handlers = {
                'translate_to_betanet': self.bridge.translate_to_betanet,
                'translate_from_betanet': self.bridge.translate_from_betanet,
                'send_betanet_message': self.bridge.send_betanet_message,
                'receive_betanet_messages': self.bridge.receive_betanet_messages,
                'configure_betanet': self.bridge.configure_betanet,
                'get_health_status': self.bridge.get_health_status,
                'shutdown': self.bridge.shutdown,
            }

            handler = handlers.get(method)
            if not handler:
                return self.create_error_response(
                    request_id, -32601, f"Method not found: {method}"
                )

            # Execute handler
            result = await handler(**params)

            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'result': result
            }

        except Exception as e:
            logger.error(f"Request handler error: {e}\n{traceback.format_exc()}")
            return self.create_error_response(
                request_id, -32603, str(e)
            )

    def create_error_response(self, request_id: Any, code: int, message: str) -> Dict[str, Any]:
        """Create a JSON-RPC error response."""
        return {
            'jsonrpc': '2.0',
            'id': request_id,
            'error': {
                'code': code,
                'message': message
            }
        }


class BetaNetBridge:
    """Bridge between TypeScript and Python BetaNet infrastructure."""

    def __init__(self, config: BridgeConfig):
        self.config = config
        self.transport: Optional[ConstitutionalBetaNetTransport] = None
        self.service: Optional[ConstitutionalBetaNetService] = None
        self.frame_processor: Optional[ConstitutionalFrameProcessor] = None
        self.fog_sandbox: Optional[CreateSandboxTool] = None
        self.start_time = time.time()
        self.metrics = {
            'requests_processed': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'average_latency_ms': 0,
            'p95_latency_ms': 0,
        }
        self.latencies: List[float] = []

    async def initialize(self):
        """Initialize BetaNet components."""
        logger.info("Initializing BetaNet infrastructure...")

        # Create transport configuration
        transport_config = ConstitutionalTransportConfig(
            constitutional_tier=getattr(ConstitutionalTier, self.config.constitutional_tier.upper()),
            enable_privacy_preservation=self.config.privacy_mode != 'standard',
            enable_zero_knowledge_proofs=self.config.enable_zero_knowledge_proofs,
            enable_mixnode_routing=self.config.enable_mixnode_routing,
            max_latency_ms=75,  # Target <75ms p95
        )

        # Initialize transport
        self.transport = ConstitutionalBetaNetTransport(transport_config)
        await self.transport.initialize()

        # Initialize service
        self.service = ConstitutionalBetaNetService(self.transport)
        await self.service.start()

        # Initialize frame processor
        self.frame_processor = ConstitutionalFrameProcessor(
            constitutional_tier=transport_config.constitutional_tier
        )

        # Initialize fog integration if enabled
        if self.config.enable_fog_integration:
            self.fog_sandbox = CreateSandboxTool()

        logger.info("BetaNet infrastructure initialized successfully")

    async def translate_to_betanet(
        self,
        request: Dict[str, Any],
        constitutional_tier: str,
        privacy_mode: str,
        enable_mixnode: bool,
        enable_zk_proofs: bool
    ) -> Dict[str, Any]:
        """Translate AIVillage request to BetaNet message."""
        start_time = time.perf_counter()

        try:
            # Create constitutional message
            constitutional_msg = ConstitutionalMessage(
                content=json.dumps(request),
                sender_id=request.get('sender_id', 'aivillage'),
                recipient_id=request.get('recipient_id', 'betanet'),
                constitutional_tier=getattr(ConstitutionalTier, constitutional_tier.upper()),
                privacy_mode=ConstitutionalTransportMode.PRIVACY_PRESERVING if privacy_mode != 'standard' else ConstitutionalTransportMode.STANDARD,
                timestamp=datetime.now(UTC),
            )

            # Apply constitutional frame processing
            frame = await self.frame_processor.create_frame(
                constitutional_msg,
                ConstitutionalFrameType.DATA
            )

            # Apply mixnode routing if requested
            if enable_mixnode:
                routing_request = await create_constitutional_routing_request(
                    frame,
                    num_hops=3
                )
                frame = routing_request.get('frame', frame)

            # Convert to BetaNet message format
            betanet_msg = {
                'id': frame.frame_id,
                'type': frame.frame_type.value,
                'payload': frame.payload,
                'constitutional_metadata': {
                    'tier': constitutional_tier,
                    'privacy_mode': privacy_mode,
                    'mixnode_routing': enable_mixnode,
                    'zk_proofs': enable_zk_proofs,
                },
                'timestamp': frame.timestamp.isoformat(),
            }

            self.metrics['requests_processed'] += 1
            self._update_latency(time.perf_counter() - start_time)

            return betanet_msg

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Translation error: {e}")
            raise

    async def translate_from_betanet(
        self,
        message: Dict[str, Any],
        validate_constitutional: bool,
        apply_privacy_filters: bool
    ) -> Dict[str, Any]:
        """Translate BetaNet message to AIVillage response."""
        start_time = time.perf_counter()

        try:
            # Parse BetaNet message
            frame = ConstitutionalFrame(
                frame_id=message['id'],
                frame_type=ConstitutionalFrameType(message['type']),
                payload=message['payload'],
                timestamp=datetime.fromisoformat(message['timestamp'])
            )

            # Validate constitutional compliance if requested
            if validate_constitutional:
                validation_result = await self.frame_processor.validate_frame(frame)
                if not validation_result['valid']:
                    raise ValueError(f"Constitutional validation failed: {validation_result['reason']}")

            # Extract content
            content = json.loads(frame.payload) if isinstance(frame.payload, str) else frame.payload

            # Apply privacy filters if requested
            if apply_privacy_filters:
                content = self._apply_privacy_filters(content)

            # Create AIVillage response
            response = {
                'success': True,
                'data': content,
                'metadata': {
                    'betanet_id': frame.frame_id,
                    'constitutional_tier': message.get('constitutional_metadata', {}).get('tier'),
                    'validated': validate_constitutional,
                },
                'timestamp': int(time.time() * 1000),
            }

            self.metrics['requests_processed'] += 1
            self._update_latency(time.perf_counter() - start_time)

            return response

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Translation error: {e}")
            raise

    async def send_betanet_message(
        self,
        message: Dict[str, Any],
        use_fog_relay: bool,
        priority: str
    ) -> Dict[str, Any]:
        """Send message through BetaNet network."""
        start_time = time.perf_counter()

        try:
            # Send via transport
            result = await self.transport.send_message(
                recipient_id=message.get('recipient_id', 'broadcast'),
                content=json.dumps(message['payload']),
                priority=priority
            )

            # Use fog relay if requested
            if use_fog_relay and self.fog_sandbox:
                fog_result = await self.fog_sandbox.execute({
                    'namespace': 'betanet',
                    'runtime': 'wasi',
                    'resources': {'cpu_cores': 1, 'memory_gb': 0.5},
                    'name': f"betanet-relay-{message['id']}"
                })
                logger.info(f"Fog relay result: {fog_result}")

            self.metrics['messages_sent'] += 1
            self._update_latency(time.perf_counter() - start_time)

            return {
                'success': True,
                'message_id': result.get('message_id'),
                'timestamp': int(time.time() * 1000)
            }

        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Send message error: {e}")
            raise

    async def receive_betanet_messages(
        self,
        timeout: int,
        max_messages: int
    ) -> List[Dict[str, Any]]:
        """Receive messages from BetaNet network."""
        try:
            messages = await self.transport.receive_messages(
                timeout_ms=timeout,
                max_messages=max_messages
            )

            # Convert to dictionary format
            result = []
            for msg in messages:
                result.append({
                    'id': msg.message_id,
                    'sender_id': msg.sender_id,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat()
                })

            self.metrics['messages_received'] += len(result)

            return result

        except Exception as e:
            logger.error(f"Receive messages error: {e}")
            return []

    async def configure_betanet(self, **kwargs) -> Dict[str, Any]:
        """Configure BetaNet settings."""
        # Update configuration
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Reinitialize if needed
        if self.transport:
            await self.transport.update_config(kwargs)

        return {'success': True, 'config': asdict(self.config)}

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the bridge."""
        health = {
            'healthy': True,
            'uptime': int(time.time() - self.start_time),
            'bridge_status': 'operational',
            'betanet_status': 'unknown',
            'fog_status': 'unknown',
            'constitutional_status': 'unknown',
            'active_connections': len(getattr(self.transport, 'connections', [])) if self.transport else 0,
            'metrics': self.metrics,
        }

        # Check BetaNet status
        if self.transport:
            try:
                transport_health = await self.transport.get_health()
                health['betanet_status'] = 'healthy' if transport_health.get('healthy') else 'unhealthy'
            except:
                health['betanet_status'] = 'error'
                health['healthy'] = False

        # Check fog status
        if self.fog_sandbox:
            health['fog_status'] = 'enabled'
        else:
            health['fog_status'] = 'disabled'

        # Check constitutional validation
        if self.frame_processor:
            health['constitutional_status'] = 'operational'
        else:
            health['constitutional_status'] = 'disabled'

        return health

    async def shutdown(self) -> Dict[str, Any]:
        """Shutdown the bridge."""
        logger.info("Shutting down BetaNet bridge...")

        if self.service:
            await self.service.stop()

        if self.transport:
            await self.transport.close()

        return {'success': True}

    def _apply_privacy_filters(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy filters to content."""
        # Remove sensitive fields
        sensitive_fields = ['private_key', 'secret', 'password', 'token']

        filtered = {}
        for key, value in content.items():
            if any(field in key.lower() for field in sensitive_fields):
                filtered[key] = '***REDACTED***'
            else:
                filtered[key] = value

        return filtered

    def _update_latency(self, latency_seconds: float):
        """Update latency metrics."""
        latency_ms = latency_seconds * 1000
        self.latencies.append(latency_ms)

        # Keep only last 1000 measurements
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]

        # Calculate metrics
        self.metrics['average_latency_ms'] = sum(self.latencies) / len(self.latencies)

        # Calculate P95
        sorted_latencies = sorted(self.latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        self.metrics['p95_latency_ms'] = sorted_latencies[p95_index] if sorted_latencies else 0

        # Log warning if P95 > 75ms
        if self.metrics['p95_latency_ms'] > 75:
            logger.warning(f"P95 latency {self.metrics['p95_latency_ms']:.2f}ms exceeds 75ms target")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='BetaNet Bridge Service')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=9876, help='Port to bind to')
    parser.add_argument('--constitutional-tier', default='Silver',
                       choices=['Bronze', 'Silver', 'Gold', 'Platinum'],
                       help='Constitutional tier')
    parser.add_argument('--privacy-mode', default='standard',
                       choices=['standard', 'enhanced', 'maximum'],
                       help='Privacy mode')

    args = parser.parse_args()

    # Create configuration
    config = BridgeConfig(
        host=args.host,
        port=args.port,
        constitutional_tier=args.constitutional_tier,
        privacy_mode=args.privacy_mode
    )

    # Create and initialize bridge
    bridge = BetaNetBridge(config)
    await bridge.initialize()

    # Create and start JSON-RPC server
    server = JsonRpcServer(bridge)

    try:
        await server.start(config.host, config.port)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await bridge.shutdown()


if __name__ == '__main__':
    asyncio.run(main())