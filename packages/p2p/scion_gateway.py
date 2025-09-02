"""
SCION Gateway Implementation for P2P Communication
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class SCIONPathType(Enum):
    DIRECT = "direct"
    MULTI_HOP = "multi_hop"
    REDUNDANT = "redundant"


@dataclass
class SCIONPath:
    """SCION network path representation"""
    path_id: str
    interfaces: List[str]
    path_type: SCIONPathType = SCIONPathType.DIRECT
    latency_ms: int = 0
    bandwidth_mbps: int = 0
    reliability_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "interfaces": self.interfaces,
            "path_type": self.path_type.value,
            "latency_ms": self.latency_ms,
            "bandwidth_mbps": self.bandwidth_mbps,
            "reliability_score": self.reliability_score
        }


@dataclass 
class GatewayConfig:
    """Configuration for SCION Gateway"""
    local_address: str
    port: int = 8080
    max_connections: int = 100
    timeout_seconds: int = 30
    enable_encryption: bool = True
    enable_compression: bool = True
    log_level: str = "INFO"
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        return (
            self.port > 0 and self.port < 65536 and
            self.max_connections > 0 and
            self.timeout_seconds > 0 and
            self.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
        )


class SCIONGateway:
    """SCION Gateway for P2P network communication"""
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.active_paths: List[SCIONPath] = []
        self.connection_pool: Dict[str, Any] = {}
        self.is_running = False
        
        if not config.validate():
            raise ValueError("Invalid gateway configuration")
    
    async def start(self) -> bool:
        """Start the SCION gateway"""
        try:
            self.is_running = True
            return True
        except Exception as e:
            self.is_running = False
            raise e
    
    async def stop(self) -> bool:
        """Stop the SCION gateway"""
        self.is_running = False
        self.connection_pool.clear()
        return True
    
    def discover_paths(self, destination: str) -> List[SCIONPath]:
        """Discover available SCION paths to destination"""
        # Placeholder implementation
        return [
            SCIONPath(
                path_id=f"path_to_{destination}",
                interfaces=[self.config.local_address, destination],
                path_type=SCIONPathType.DIRECT
            )
        ]
    
    def get_best_path(self, destination: str) -> Optional[SCIONPath]:
        """Get the best available path to destination"""
        paths = self.discover_paths(destination)
        if not paths:
            return None
        
        # Simple best path selection - highest reliability
        return max(paths, key=lambda p: p.reliability_score)
    
    def is_healthy(self) -> bool:
        """Check if gateway is healthy and operational"""
        return self.is_running and len(self.active_paths) >= 0