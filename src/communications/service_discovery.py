"""Service Discovery System for Agent Communications.

Provides automatic discovery and registration of agent services:
- Agent registration and heartbeat system
- Service capability advertisement
- Dynamic service discovery
- Network topology mapping
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ServiceInfo:
    """Information about a registered service."""
    
    agent_id: str
    service_type: str
    host: str
    port: int
    capabilities: List[str]
    metadata: Dict[str, str]
    last_heartbeat: float
    status: str = "active"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ServiceInfo":
        return cls(**data)
    
    def is_alive(self, timeout: float = 30.0) -> bool:
        """Check if service is considered alive based on heartbeat."""
        return time.time() - self.last_heartbeat < timeout


class ServiceRegistry:
    """Registry for tracking available services."""
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.service_types: Dict[str, Set[str]] = {}
        self.cleanup_interval = 30.0
        self.heartbeat_timeout = 60.0
        
    def register_service(self, service_info: ServiceInfo) -> bool:
        """Register a new service."""
        try:
            service_id = f"{service_info.agent_id}:{service_info.service_type}"
            service_info.last_heartbeat = time.time()
            
            self.services[service_id] = service_info
            
            # Update service type index
            if service_info.service_type not in self.service_types:
                self.service_types[service_info.service_type] = set()
            self.service_types[service_info.service_type].add(service_id)
            
            logger.info(f"Registered service: {service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service: {e}")
            return False
    
    def unregister_service(self, agent_id: str, service_type: str) -> bool:
        """Unregister a service."""
        service_id = f"{agent_id}:{service_type}"
        
        if service_id in self.services:
            service_info = self.services[service_id]
            del self.services[service_id]
            
            # Update service type index
            if service_type in self.service_types:
                self.service_types[service_type].discard(service_id)
                if not self.service_types[service_type]:
                    del self.service_types[service_type]
            
            logger.info(f"Unregistered service: {service_id}")
            return True
        
        return False
    
    def heartbeat(self, agent_id: str, service_type: str) -> bool:
        """Update service heartbeat."""
        service_id = f"{agent_id}:{service_type}"
        
        if service_id in self.services:
            self.services[service_id].last_heartbeat = time.time()
            self.services[service_id].status = "active"
            return True
        
        return False
    
    def discover_services(self, service_type: Optional[str] = None) -> List[ServiceInfo]:
        """Discover available services of a specific type or all services."""
        current_time = time.time()
        active_services = []
        
        if service_type:
            # Get services of specific type
            service_ids = self.service_types.get(service_type, set())
            for service_id in service_ids:
                if service_id in self.services:
                    service = self.services[service_id]
                    if service.is_alive(self.heartbeat_timeout):
                        active_services.append(service)
        else:
            # Get all active services
            for service in self.services.values():
                if service.is_alive(self.heartbeat_timeout):
                    active_services.append(service)
        
        return active_services
    
    def get_service_by_agent(self, agent_id: str) -> List[ServiceInfo]:
        """Get all services for a specific agent."""
        return [
            service for service in self.services.values()
            if service.agent_id == agent_id and service.is_alive(self.heartbeat_timeout)
        ]
    
    def cleanup_stale_services(self) -> int:
        """Remove services that haven't sent heartbeats."""
        current_time = time.time()
        stale_services = []
        
        for service_id, service in self.services.items():
            if not service.is_alive(self.heartbeat_timeout):
                stale_services.append(service_id)
        
        for service_id in stale_services:
            service = self.services[service_id]
            self.unregister_service(service.agent_id, service.service_type)
            logger.info(f"Cleaned up stale service: {service_id}")
        
        return len(stale_services)
    
    def get_registry_stats(self) -> Dict:
        """Get registry statistics."""
        active_count = len([s for s in self.services.values() if s.is_alive(self.heartbeat_timeout)])
        
        return {
            "total_services": len(self.services),
            "active_services": active_count,
            "stale_services": len(self.services) - active_count,
            "service_types": list(self.service_types.keys()),
            "service_type_counts": {
                stype: len([s for s in self.services.values() 
                           if s.service_type == stype and s.is_alive(self.heartbeat_timeout)])
                for stype in self.service_types.keys()
            }
        }


class ServiceDiscovery:
    """Main service discovery system."""
    
    def __init__(self, registry: Optional[ServiceRegistry] = None):
        self.registry = registry or ServiceRegistry()
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
    async def start(self):
        """Start the service discovery system."""
        if self.running:
            return
        
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Service discovery started")
    
    async def stop(self):
        """Stop the service discovery system."""
        self.running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Service discovery stopped")
    
    async def register_service(
        self,
        agent_id: str,
        service_type: str,
        host: str,
        port: int,
        capabilities: List[str] = None,
        metadata: Dict[str, str] = None
    ) -> bool:
        """Register a service with the discovery system."""
        service_info = ServiceInfo(
            agent_id=agent_id,
            service_type=service_type,
            host=host,
            port=port,
            capabilities=capabilities or [],
            metadata=metadata or {},
            last_heartbeat=time.time()
        )
        
        return self.registry.register_service(service_info)
    
    async def unregister_service(self, agent_id: str, service_type: str) -> bool:
        """Unregister a service."""
        return self.registry.unregister_service(agent_id, service_type)
    
    async def heartbeat(self, agent_id: str, service_type: str) -> bool:
        """Send heartbeat for a service."""
        return self.registry.heartbeat(agent_id, service_type)
    
    async def discover_services(self, service_type: Optional[str] = None) -> List[ServiceInfo]:
        """Discover available services."""
        return self.registry.discover_services(service_type)
    
    async def find_service_by_capability(self, capability: str) -> List[ServiceInfo]:
        """Find services that advertise a specific capability."""
        all_services = self.registry.discover_services()
        return [
            service for service in all_services
            if capability in service.capabilities
        ]
    
    async def get_best_service(
        self,
        service_type: str,
        prefer_local: bool = True
    ) -> Optional[ServiceInfo]:
        """Get the best available service of a given type."""
        services = self.registry.discover_services(service_type)
        
        if not services:
            return None
        
        # Simple selection strategy: prefer local services, then by load
        if prefer_local:
            local_services = [s for s in services if s.host in ['localhost', '127.0.0.1']]
            if local_services:
                return local_services[0]
        
        return services[0]
    
    async def _cleanup_loop(self):
        """Background loop to cleanup stale services."""
        while self.running:
            try:
                cleaned = self.registry.cleanup_stale_services()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} stale services")
                
                await asyncio.sleep(self.registry.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(5)


# Global service discovery instance
_service_discovery: Optional[ServiceDiscovery] = None


async def get_service_discovery() -> ServiceDiscovery:
    """Get global service discovery instance."""
    global _service_discovery
    
    if _service_discovery is None:
        _service_discovery = ServiceDiscovery()
        await _service_discovery.start()
    
    return _service_discovery


# Convenience functions for direct use
async def register_service(
    agent_id: str,
    service_type: str,
    host: str,
    port: int,
    capabilities: List[str] = None,
    metadata: Dict[str, str] = None
) -> bool:
    """Register a service using the global discovery system."""
    discovery = await get_service_discovery()
    return await discovery.register_service(
        agent_id, service_type, host, port, capabilities, metadata
    )


async def discover_services(service_type: Optional[str] = None) -> List[ServiceInfo]:
    """Discover services using the global discovery system."""
    discovery = await get_service_discovery()
    return await discovery.discover_services(service_type)


async def heartbeat_service(agent_id: str, service_type: str) -> bool:
    """Send heartbeat using the global discovery system."""
    discovery = await get_service_discovery()
    return await discovery.heartbeat(agent_id, service_type)


if __name__ == "__main__":
    async def test_service_discovery():
        """Test the service discovery system."""
        discovery = ServiceDiscovery()
        await discovery.start()
        
        # Register some test services
        await discovery.register_service(
            "agent1", "rag", "localhost", 8001, 
            ["embedding", "search"], {"model": "BERT"}
        )
        
        await discovery.register_service(
            "agent2", "p2p", "localhost", 8002,
            ["messaging", "discovery"], {"protocol": "libp2p"}
        )
        
        # Test discovery
        all_services = await discovery.discover_services()
        print(f"All services: {len(all_services)}")
        
        rag_services = await discovery.discover_services("rag")
        print(f"RAG services: {len(rag_services)}")
        
        # Test capability search
        embedding_services = await discovery.find_service_by_capability("embedding")
        print(f"Embedding services: {len(embedding_services)}")
        
        await discovery.stop()
    
    asyncio.run(test_service_discovery())