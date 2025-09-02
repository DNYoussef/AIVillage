"""
MCP CI/CD Session Manager - Robust Session Lifecycle Management for CI/CD Pipelines

This module provides comprehensive session management for MCP servers within CI/CD environments,
addressing the specific challenges of GitHub Actions and similar automated environments.

Key Features:
- Robust session lifecycle management with automatic cleanup
- Fallback mechanisms for MCP server unavailability
- CI/CD-specific optimizations for limited execution time
- Health monitoring and failure detection
- Authentication management across different environments
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MCPSessionState(Enum):
    """States for MCP session lifecycle"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"  # Some servers unavailable but core functionality works
    FAILING = "failing"    # Multiple failures but still operational
    FAILED = "failed"      # Session cannot continue
    CLEANUP = "cleanup"
    TERMINATED = "terminated"


class CICDEnvironment(Enum):
    """CI/CD environment types"""
    GITHUB_ACTIONS = "github_actions"
    AZURE_DEVOPS = "azure_devops"
    JENKINS = "jenkins"
    GITLAB_CI = "gitlab_ci"
    BITBUCKET = "bitbucket"
    LOCAL = "local"
    UNKNOWN = "unknown"


@dataclass
class MCPServerHealth:
    """Health status of an MCP server"""
    server_name: str
    is_available: bool = False
    is_responsive: bool = False
    last_check: datetime = field(default_factory=datetime.now)
    response_time_ms: float = 0.0
    error_count: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    
    @property
    def is_healthy(self) -> bool:
        """Check if server is healthy based on multiple criteria"""
        return (
            self.is_available and 
            self.is_responsive and 
            self.consecutive_failures < 3 and
            self.response_time_ms < 5000  # 5 second timeout
        )


@dataclass
class SessionConfiguration:
    """Configuration for MCP session in CI/CD environment"""
    # Environment detection
    environment: CICDEnvironment = CICDEnvironment.UNKNOWN
    max_session_duration_minutes: int = 45  # GitHub Actions default timeout
    
    # Required vs Optional servers
    required_servers: Set[str] = field(default_factory=lambda: {"memory", "sequential-thinking"})
    optional_servers: Set[str] = field(default_factory=lambda: {"github", "hyperag"})
    
    # Timeout configurations
    server_startup_timeout_seconds: int = 30
    health_check_interval_seconds: int = 15
    cleanup_timeout_seconds: int = 60
    
    # Retry configurations
    max_connection_retries: int = 3
    retry_delay_seconds: int = 2
    backoff_multiplier: float = 1.5
    
    # Fallback configurations
    enable_offline_mode: bool = True
    enable_degraded_mode: bool = True
    
    # Authentication
    use_environment_auth: bool = True
    auth_timeout_seconds: int = 10


class MCPCICDSessionManager:
    """
    Robust MCP Session Manager for CI/CD Pipelines
    
    Provides:
    - Automatic environment detection and optimization
    - Robust session lifecycle with cleanup
    - Graceful degradation when servers are unavailable
    - Health monitoring and failure recovery
    - CI/CD-specific timeout and resource management
    """
    
    def __init__(self, config: Optional[SessionConfiguration] = None):
        self.config = config or SessionConfiguration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Session state
        self.session_id = f"mcp_session_{int(time.time())}"
        self.state = MCPSessionState.INITIALIZING
        self.start_time = datetime.now()
        
        # Server management
        self.server_health: Dict[str, MCPServerHealth] = {}
        self.active_connections: Dict[str, Any] = {}
        self.failed_servers: Set[str] = set()
        
        # Monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.cleanup_handlers: List[callable] = []
        
        # Statistics
        self.stats = {
            "servers_attempted": 0,
            "servers_connected": 0,
            "servers_failed": 0,
            "health_checks_performed": 0,
            "cleanup_operations": 0,
            "fallback_activations": 0
        }
        
        # Detect CI/CD environment
        self._detect_environment()
        
        self.logger.info(f"MCP Session Manager initialized for {self.config.environment.value} environment")
    
    def _detect_environment(self):
        """Detect the CI/CD environment and adjust configuration"""
        if os.getenv('GITHUB_ACTIONS'):
            self.config.environment = CICDEnvironment.GITHUB_ACTIONS
            self.config.max_session_duration_minutes = 45  # Conservative GitHub Actions limit
            
        elif os.getenv('AZURE_HTTP_USER_AGENT'):
            self.config.environment = CICDEnvironment.AZURE_DEVOPS
            self.config.max_session_duration_minutes = 60
            
        elif os.getenv('JENKINS_URL'):
            self.config.environment = CICDEnvironment.JENKINS
            self.config.max_session_duration_minutes = 120
            
        elif os.getenv('GITLAB_CI'):
            self.config.environment = CICDEnvironment.GITLAB_CI
            self.config.max_session_duration_minutes = 60
            
        elif os.getenv('BITBUCKET_COMMIT'):
            self.config.environment = CICDEnvironment.BITBUCKET
            self.config.max_session_duration_minutes = 50
            
        else:
            self.config.environment = CICDEnvironment.LOCAL
            self.config.max_session_duration_minutes = 240  # 4 hours for local development
        
        self.logger.info(f"Detected environment: {self.config.environment.value}")
    
    @asynccontextmanager
    async def session(self):
        """
        Async context manager for MCP session lifecycle
        
        Usage:
            async with session_manager.session() as mcp_servers:
                # Use mcp_servers dict for operations
                result = await mcp_servers['memory'].store(...)
        """
        try:
            # Initialize session
            servers = await self._initialize_session()
            self.state = MCPSessionState.ACTIVE
            
            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
            
            self.logger.info(f"MCP session {self.session_id} active with {len(servers)} servers")
            yield servers
            
        except Exception as e:
            self.state = MCPSessionState.FAILED
            self.logger.error(f"MCP session failed: {e}")
            raise
            
        finally:
            # Ensure cleanup always happens
            await self._cleanup_session()
    
    async def _initialize_session(self) -> Dict[str, Any]:
        """Initialize MCP session with all available servers"""
        self.logger.info("Initializing MCP session...")
        
        # First, try to connect to required servers
        connected_servers = {}
        
        for server_name in self.config.required_servers:
            success = await self._connect_to_server(server_name, required=True)
            if success:
                connected_servers[server_name] = self.active_connections[server_name]
            else:
                if not self.config.enable_offline_mode:
                    raise RuntimeError(f"Required MCP server '{server_name}' unavailable and offline mode disabled")
                
                self.logger.warning(f"Required server '{server_name}' unavailable - using offline fallback")
                connected_servers[server_name] = self._create_offline_fallback(server_name)
        
        # Then try optional servers
        for server_name in self.config.optional_servers:
            success = await self._connect_to_server(server_name, required=False)
            if success:
                connected_servers[server_name] = self.active_connections[server_name]
            else:
                self.logger.info(f"Optional server '{server_name}' unavailable - continuing without it")
                self.stats["fallback_activations"] += 1
        
        # Determine session state based on connectivity
        total_servers = len(self.config.required_servers) + len(self.config.optional_servers)
        if len(connected_servers) == total_servers:
            self.state = MCPSessionState.ACTIVE
        elif len(connected_servers) >= len(self.config.required_servers):
            self.state = MCPSessionState.DEGRADED
            self.logger.warning(f"Session in degraded mode - {len(self.failed_servers)} servers unavailable")
        else:
            self.state = MCPSessionState.FAILING
            self.logger.error(f"Session failing - insufficient servers available")
        
        return connected_servers
    
    async def _connect_to_server(self, server_name: str, required: bool = False) -> bool:
        """Connect to a specific MCP server with retries and timeout"""
        self.stats["servers_attempted"] += 1
        
        health = MCPServerHealth(server_name=server_name)
        self.server_health[server_name] = health
        
        for attempt in range(self.config.max_connection_retries):
            try:
                start_time = time.perf_counter()
                
                # Attempt connection based on server type
                connection = await self._establish_server_connection(server_name)
                
                if connection:
                    response_time = (time.perf_counter() - start_time) * 1000
                    
                    # Store connection
                    self.active_connections[server_name] = connection
                    
                    # Update health status
                    health.is_available = True
                    health.is_responsive = True
                    health.response_time_ms = response_time
                    health.consecutive_failures = 0
                    health.last_check = datetime.now()
                    
                    self.stats["servers_connected"] += 1
                    self.logger.info(f"Connected to MCP server '{server_name}' in {response_time:.1f}ms")
                    return True
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout connecting to '{server_name}' (attempt {attempt + 1})")
                health.error_count += 1
                health.consecutive_failures += 1
                health.last_error = "Connection timeout"
                
            except Exception as e:
                self.logger.warning(f"Failed to connect to '{server_name}' (attempt {attempt + 1}): {e}")
                health.error_count += 1
                health.consecutive_failures += 1
                health.last_error = str(e)
            
            # Wait before retry with exponential backoff
            if attempt < self.config.max_connection_retries - 1:
                delay = self.config.retry_delay_seconds * (self.config.backoff_multiplier ** attempt)
                await asyncio.sleep(delay)
        
        # Connection failed after all retries
        health.is_available = False
        health.is_responsive = False
        self.failed_servers.add(server_name)
        self.stats["servers_failed"] += 1
        
        if required:
            self.logger.error(f"Failed to connect to required MCP server '{server_name}'")
        else:
            self.logger.info(f"Optional MCP server '{server_name}' unavailable")
        
        return False
    
    async def _establish_server_connection(self, server_name: str) -> Optional[Any]:
        """Establish connection to specific server based on configuration"""
        # This is a placeholder for actual MCP server connection logic
        # In real implementation, this would:
        # 1. Read MCP configuration for the server
        # 2. Establish appropriate transport (stdio, websocket, etc.)
        # 3. Perform authentication
        # 4. Validate server capabilities
        
        # For now, simulate connection attempt
        if server_name == "memory":
            # Simulate memory MCP server connection
            await asyncio.sleep(0.1)  # Simulate connection time
            return {"type": "memory", "connection": f"memory_connection_{server_name}"}
            
        elif server_name == "sequential-thinking":
            # Simulate sequential thinking MCP server connection
            await asyncio.sleep(0.2)  # Simulate connection time
            return {"type": "sequential-thinking", "connection": f"st_connection_{server_name}"}
            
        elif server_name == "github":
            # Simulate GitHub MCP server connection (might fail in CI)
            if self.config.environment == CICDEnvironment.GITHUB_ACTIONS and not os.getenv('GITHUB_TOKEN'):
                raise ConnectionError("GitHub token required for GitHub Actions environment")
            await asyncio.sleep(0.3)  # Simulate connection time
            return {"type": "github", "connection": f"github_connection_{server_name}"}
            
        elif server_name == "hyperag":
            # Simulate HyperAG MCP server connection
            await asyncio.sleep(0.5)  # Simulate connection time
            if not Path("C:/Users/17175/Desktop/AIVillage/core/rag/mcp_servers/hyperag/mcp_server.py").exists():
                raise FileNotFoundError("HyperAG MCP server not found")
            return {"type": "hyperag", "connection": f"hyperag_connection_{server_name}"}
        
        else:
            raise ValueError(f"Unknown MCP server: {server_name}")
    
    def _create_offline_fallback(self, server_name: str) -> Dict[str, Any]:
        """Create offline fallback for unavailable required servers"""
        self.stats["fallback_activations"] += 1
        
        if server_name == "memory":
            # Memory fallback - in-memory storage
            return {
                "type": "memory_fallback",
                "storage": {},
                "store": lambda key, value: self._fallback_memory_store(key, value),
                "retrieve": lambda key: self._fallback_memory_retrieve(key),
                "is_fallback": True
            }
            
        elif server_name == "sequential-thinking":
            # Sequential thinking fallback - basic step-by-step processing
            return {
                "type": "sequential_thinking_fallback",
                "think": lambda prompt: self._fallback_sequential_think(prompt),
                "is_fallback": True
            }
        
        else:
            # Generic fallback
            return {
                "type": f"{server_name}_fallback",
                "execute": lambda *args, **kwargs: {"status": "fallback", "result": None},
                "is_fallback": True
            }
    
    def _fallback_memory_store(self, key: str, value: Any) -> Dict[str, Any]:
        """Fallback memory store implementation"""
        if not hasattr(self, '_fallback_storage'):
            self._fallback_storage = {}
        self._fallback_storage[key] = value
        return {"status": "stored", "key": key, "fallback": True}
    
    def _fallback_memory_retrieve(self, key: str) -> Dict[str, Any]:
        """Fallback memory retrieve implementation"""
        if not hasattr(self, '_fallback_storage'):
            self._fallback_storage = {}
        value = self._fallback_storage.get(key)
        return {"status": "retrieved", "key": key, "value": value, "fallback": True}
    
    def _fallback_sequential_think(self, prompt: str) -> Dict[str, Any]:
        """Fallback sequential thinking implementation"""
        # Basic step breakdown
        steps = [
            "1. Analyze the problem",
            "2. Break down into components", 
            "3. Evaluate options",
            "4. Make decision",
            "5. Plan implementation"
        ]
        return {
            "status": "processed",
            "steps": steps,
            "reasoning": f"Basic analysis of: {prompt[:100]}...",
            "fallback": True
        }
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring for active connections"""
        while self.state in [MCPSessionState.ACTIVE, MCPSessionState.DEGRADED]:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
            except asyncio.CancelledError:
                self.logger.info("Health monitoring cancelled")
                break
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.config.health_check_interval_seconds)
    
    async def _perform_health_checks(self):
        """Perform health checks on all active connections"""
        self.stats["health_checks_performed"] += 1
        
        for server_name, connection in self.active_connections.items():
            try:
                start_time = time.perf_counter()
                
                # Perform simple health check (ping/status)
                health_result = await self._check_server_health(server_name, connection)
                
                response_time = (time.perf_counter() - start_time) * 1000
                health = self.server_health[server_name]
                
                if health_result:
                    health.is_responsive = True
                    health.response_time_ms = response_time
                    health.consecutive_failures = 0
                    health.last_check = datetime.now()
                else:
                    health.is_responsive = False
                    health.consecutive_failures += 1
                    health.error_count += 1
                    health.last_error = "Health check failed"
                
                # Check if server should be marked as failed
                if health.consecutive_failures >= 5:
                    self.logger.warning(f"Server '{server_name}' marked as failed after {health.consecutive_failures} consecutive failures")
                    self.failed_servers.add(server_name)
                    
            except Exception as e:
                self.logger.warning(f"Health check failed for '{server_name}': {e}")
                health = self.server_health.get(server_name)
                if health:
                    health.consecutive_failures += 1
                    health.error_count += 1
                    health.last_error = str(e)
        
        # Update session state based on health
        self._update_session_state()
    
    async def _check_server_health(self, server_name: str, connection: Any) -> bool:
        """Check health of a specific server connection"""
        try:
            # This would perform actual health check based on server type
            # For now, simulate health check
            if connection.get("is_fallback"):
                return True  # Fallbacks are always "healthy"
            
            # Simulate health check with small probability of failure
            import random
            return random.random() > 0.05  # 95% success rate
            
        except Exception as e:
            self.logger.debug(f"Health check error for {server_name}: {e}")
            return False
    
    def _update_session_state(self):
        """Update session state based on server health"""
        healthy_servers = sum(1 for h in self.server_health.values() if h.is_healthy)
        total_servers = len(self.server_health)
        failed_required = len(self.failed_servers.intersection(self.config.required_servers))
        
        if failed_required > 0 and not self.config.enable_offline_mode:
            self.state = MCPSessionState.FAILED
        elif healthy_servers == total_servers:
            self.state = MCPSessionState.ACTIVE
        elif healthy_servers >= len(self.config.required_servers):
            self.state = MCPSessionState.DEGRADED
        else:
            self.state = MCPSessionState.FAILING
    
    async def _cleanup_session(self):
        """Clean up MCP session and all resources"""
        self.logger.info(f"Cleaning up MCP session {self.session_id}...")
        self.state = MCPSessionState.CLEANUP
        
        try:
            # Cancel health monitoring
            if self.health_check_task and not self.health_check_task.done():
                self.health_check_task.cancel()
                try:
                    await asyncio.wait_for(self.health_check_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Health monitoring task cleanup timeout")
            
            # Close all server connections
            for server_name, connection in self.active_connections.items():
                try:
                    await self._close_server_connection(server_name, connection)
                except Exception as e:
                    self.logger.warning(f"Error closing connection to '{server_name}': {e}")
            
            # Run registered cleanup handlers
            for handler in self.cleanup_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                    self.stats["cleanup_operations"] += 1
                except Exception as e:
                    self.logger.warning(f"Cleanup handler error: {e}")
            
            # Clear state
            self.active_connections.clear()
            self.server_health.clear()
            self.failed_servers.clear()
            
            duration = datetime.now() - self.start_time
            self.logger.info(f"MCP session cleanup completed in {duration.total_seconds():.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error during session cleanup: {e}")
        
        finally:
            self.state = MCPSessionState.TERMINATED
    
    async def _close_server_connection(self, server_name: str, connection: Any):
        """Close connection to a specific server"""
        if connection.get("is_fallback"):
            return  # Nothing to close for fallback connections
        
        # This would perform actual connection cleanup based on server type
        # For now, just log the closure
        self.logger.debug(f"Closing connection to '{server_name}'")
        await asyncio.sleep(0.01)  # Simulate cleanup time
    
    def register_cleanup_handler(self, handler: callable):
        """Register a cleanup handler to run during session termination"""
        self.cleanup_handlers.append(handler)
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get comprehensive session status information"""
        duration = datetime.now() - self.start_time
        
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "environment": self.config.environment.value,
            "duration_seconds": duration.total_seconds(),
            "server_health": {name: {
                "is_healthy": health.is_healthy,
                "is_available": health.is_available,
                "is_responsive": health.is_responsive,
                "response_time_ms": health.response_time_ms,
                "error_count": health.error_count,
                "consecutive_failures": health.consecutive_failures,
                "last_error": health.last_error
            } for name, health in self.server_health.items()},
            "active_connections": list(self.active_connections.keys()),
            "failed_servers": list(self.failed_servers),
            "statistics": self.stats.copy()
        }
    
    def is_healthy(self) -> bool:
        """Check if the session is in a healthy state"""
        return self.state in [MCPSessionState.ACTIVE, MCPSessionState.DEGRADED]
    
    def get_available_servers(self) -> Dict[str, Any]:
        """Get dictionary of available/healthy servers"""
        return {name: conn for name, conn in self.active_connections.items() 
                if name not in self.failed_servers}


# Example usage and testing functions
async def example_usage():
    """Example of how to use the MCP CI/CD Session Manager"""
    config = SessionConfiguration(
        required_servers={"memory", "sequential-thinking"},
        optional_servers={"github", "hyperag"},
        max_session_duration_minutes=30,
        enable_offline_mode=True
    )
    
    session_manager = MCPCICDSessionManager(config)
    
    async with session_manager.session() as mcp_servers:
        # Use MCP servers for operations
        print(f"Available servers: {list(mcp_servers.keys())}")
        
        # Example operations
        if "memory" in mcp_servers:
            memory_result = await mcp_servers["memory"].get("store", lambda k, v: print(f"Stored {k}: {v}"))("test_key", "test_value")
            print(f"Memory operation result: {memory_result}")
        
        if "sequential-thinking" in mcp_servers:
            thinking_result = await mcp_servers["sequential-thinking"].get("think", lambda p: print(f"Thinking about: {p}"))("How to optimize CI/CD pipeline")
            print(f"Sequential thinking result: {thinking_result}")
        
        # Check session health
        status = session_manager.get_session_status()
        print(f"Session status: {status['state']}")
        print(f"Healthy servers: {len([h for h in status['server_health'].values() if h['is_healthy']])}")


if __name__ == "__main__":
    asyncio.run(example_usage())