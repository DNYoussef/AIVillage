# Federated Systems Integration Guide
## Implementation Manual for AIVillage

### Overview

This guide provides step-by-step instructions for implementing and integrating the federated inference and training systems into the AIVillage ecosystem. It covers practical integration patterns, configuration examples, and deployment strategies.

---

## 1. Quick Start Implementation

### 1.1 Basic Setup

```python
# File: infrastructure/distributed_inference/__init__.py

from .federated_inference_coordinator import FederatedInferenceCoordinator, NodeType, PrivacyLevel
from .enhanced_federated_training import EnhancedFederatedTrainingCoordinator, TrainingJobConfig

async def create_federated_systems():
    """Create and initialize federated systems"""
    
    # Initialize inference coordinator
    inference_coordinator = FederatedInferenceCoordinator(
        coordinator_id="aivillage_inference",
        integration_config={
            "p2p_network": True,
            "fog_computing": True, 
            "betanet_privacy": True
        }
    )
    
    # Initialize training coordinator
    training_coordinator = EnhancedFederatedTrainingCoordinator(
        coordinator_id="aivillage_training",
        enable_fog_integration=True,
        enable_p2p_integration=True,
        enable_privacy_protocols=True,
        enable_mobile_optimization=True
    )
    
    # Start both systems
    await inference_coordinator.initialize()
    await training_coordinator.initialize()
    
    return inference_coordinator, training_coordinator
```

### 1.2 Gateway Integration

```python
# File: core/gateway/federated_routes.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from infrastructure.distributed_inference import FederatedInferenceCoordinator
from infrastructure.distributed_inference import EnhancedFederatedTrainingCoordinator

router = APIRouter(prefix="/v1/federated", tags=["federated"])

class InferenceRequest(BaseModel):
    model_id: str
    input_data: Dict[str, Any]
    privacy_level: str = "medium"
    max_latency_ms: int = 5000
    preferred_node_types: List[str] = []

class TrainingJobRequest(BaseModel):
    job_name: str
    model_architecture: str
    total_rounds: int = 100
    min_participants: int = 3
    privacy_techniques: List[str] = []
    mobile_friendly: bool = True

# Global coordinators (initialized at startup)
inference_coordinator: Optional[FederatedInferenceCoordinator] = None
training_coordinator: Optional[EnhancedFederatedTrainingCoordinator] = None

@router.post("/inference/submit")
async def submit_inference(
    request: InferenceRequest,
    client_id: str = "anonymous"
):
    """Submit federated inference request"""
    
    if not inference_coordinator:
        raise HTTPException(status_code=503, detail="Inference coordinator not available")
    
    try:
        request_id = await inference_coordinator.submit_inference_request(
            client_id=client_id,
            model_id=request.model_id,
            input_data=request.input_data,
            preferences={
                "privacy_level": request.privacy_level,
                "max_latency_ms": request.max_latency_ms,
                "preferred_node_types": request.preferred_node_types
            }
        )
        
        return {"request_id": request_id, "status": "submitted"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/inference/result/{request_id}")
async def get_inference_result(request_id: str):
    """Get inference result"""
    
    if not inference_coordinator:
        raise HTTPException(status_code=503, detail="Inference coordinator not available")
    
    result = await inference_coordinator.get_inference_result(request_id)
    if not result:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return result

@router.post("/training/create")
async def create_training_job(request: TrainingJobRequest):
    """Create federated training job"""
    
    if not training_coordinator:
        raise HTTPException(status_code=503, detail="Training coordinator not available")
    
    try:
        from infrastructure.distributed_inference.enhanced_federated_training import (
            TrainingJobConfig, TrainingJobType, PrivacyTechnique
        )
        
        job_config = TrainingJobConfig(
            job_id=f"job_{uuid.uuid4().hex[:8]}",
            job_name=request.job_name,
            job_type=TrainingJobType.STANDARD,
            model_architecture=request.model_architecture,
            dataset_requirements={},
            total_rounds=request.total_rounds,
            min_participants=request.min_participants,
            privacy_techniques=[
                PrivacyTechnique(t) for t in request.privacy_techniques
                if t in [e.value for e in PrivacyTechnique]
            ],
            mobile_friendly=request.mobile_friendly
        )
        
        job_id = await training_coordinator.create_enhanced_training_job(job_config)
        
        return {"job_id": job_id, "status": "created"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/training/status/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status"""
    
    if not training_coordinator:
        raise HTTPException(status_code=503, detail="Training coordinator not available")
    
    status = await training_coordinator.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return status

@router.get("/status")
async def get_federated_status():
    """Get overall federated systems status"""
    
    status = {
        "inference": None,
        "training": None,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if inference_coordinator:
        status["inference"] = await inference_coordinator.get_coordinator_stats()
    
    if training_coordinator:
        status["training"] = await training_coordinator.get_enhanced_stats()
    
    return status
```

### 1.3 Integration with Existing Gateway

```python
# File: core/gateway/server.py (additions)

# Add federated routes to existing gateway
from .federated_routes import router as federated_router

# Include federated routes
app.include_router(federated_router)

# Global federated system instances
federated_systems = None

@app.on_event("startup")
async def initialize_federated_systems():
    """Initialize federated systems on startup"""
    global federated_systems
    
    try:
        from infrastructure.distributed_inference import create_federated_systems
        federated_systems = await create_federated_systems()
        
        # Update global coordinators in routes
        from .federated_routes import federated_router
        federated_router.inference_coordinator = federated_systems[0]
        federated_router.training_coordinator = federated_systems[1]
        
        logger.info("Federated systems initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize federated systems: {e}")

@app.on_event("shutdown")
async def shutdown_federated_systems():
    """Shutdown federated systems gracefully"""
    global federated_systems
    
    if federated_systems:
        try:
            await federated_systems[0].stop()  # Inference coordinator
            await federated_systems[1].stop()  # Training coordinator
            logger.info("Federated systems shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down federated systems: {e}")
```

---

## 2. P2P Network Integration

### 2.1 BitChat Integration

```python
# File: infrastructure/distributed_inference/integrations/p2p_integration.py

import asyncio
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class P2PFederatedManager:
    """Manages P2P integration for federated systems"""
    
    def __init__(self):
        self.bitchat_network = None
        self.discovered_peers = {}
        self.federated_peers = {}
    
    async def initialize(self):
        """Initialize P2P integration"""
        
        try:
            # Import BitChat components
            from infrastructure.p2p.bitchat.mesh_network import MeshNetwork
            
            self.bitchat_network = MeshNetwork()
            await self.bitchat_network.initialize()
            
            # Register federated learning service
            await self.bitchat_network.register_service(
                service_type="federated_learning",
                service_data={
                    "capabilities": ["inference", "training"],
                    "privacy_features": ["differential_privacy", "secure_aggregation"],
                    "node_types": ["coordinator", "participant"]
                }
            )
            
            # Start peer discovery
            asyncio.create_task(self._discover_federated_peers())
            
            logger.info("P2P federated manager initialized")
            return True
            
        except ImportError as e:
            logger.warning(f"BitChat not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize P2P integration: {e}")
            return False
    
    async def _discover_federated_peers(self):
        """Continuously discover federated learning peers"""
        
        while True:
            try:
                if self.bitchat_network:
                    peers = await self.bitchat_network.discover_service_peers("federated_learning")
                    
                    for peer in peers:
                        if peer.peer_id not in self.federated_peers:
                            # New federated peer discovered
                            self.federated_peers[peer.peer_id] = {
                                "peer_info": peer,
                                "capabilities": peer.service_data.get("capabilities", []),
                                "privacy_features": peer.service_data.get("privacy_features", []),
                                "node_type": peer.service_data.get("node_types", ["participant"])[0],
                                "last_seen": time.time()
                            }
                            
                            logger.info(f"Discovered federated peer: {peer.peer_id}")
                
                await asyncio.sleep(300)  # Discovery every 5 minutes
                
            except Exception as e:
                logger.error(f"Peer discovery error: {e}")
                await asyncio.sleep(60)
    
    async def send_federated_message(
        self,
        peer_id: str,
        message_type: str,
        payload: Dict[str, Any],
        privacy_level: str = "medium"
    ) -> bool:
        """Send message to federated peer"""
        
        try:
            if not self.bitchat_network:
                return False
            
            message = {
                "type": "federated_message",
                "subtype": message_type,
                "payload": payload,
                "privacy_level": privacy_level,
                "timestamp": time.time()
            }
            
            result = await self.bitchat_network.send_message(peer_id, message)
            return result
            
        except Exception as e:
            logger.error(f"Failed to send message to {peer_id}: {e}")
            return False
    
    async def broadcast_federated_message(
        self,
        message_type: str,
        payload: Dict[str, Any]
    ) -> int:
        """Broadcast message to all federated peers"""
        
        successful_sends = 0
        
        for peer_id in self.federated_peers.keys():
            if await self.send_federated_message(peer_id, message_type, payload):
                successful_sends += 1
        
        return successful_sends
    
    def get_federated_peers(self) -> List[Dict[str, Any]]:
        """Get list of discovered federated peers"""
        
        return list(self.federated_peers.values())
    
    def get_peers_by_capability(self, capability: str) -> List[str]:
        """Get peers with specific capability"""
        
        return [
            peer_id for peer_id, peer_data in self.federated_peers.items()
            if capability in peer_data.get("capabilities", [])
        ]
```

### 2.2 BetaNet Privacy Integration

```python
# File: infrastructure/distributed_inference/integrations/betanet_integration.py

import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BetaNetFederatedTransport:
    """BetaNet integration for federated systems"""
    
    def __init__(self):
        self.betanet_transport = None
        self.privacy_routing_enabled = False
    
    async def initialize(self, privacy_mode: str = "balanced"):
        """Initialize BetaNet transport"""
        
        try:
            from infrastructure.fog.bridges.betanet_integration import BetaNetFogTransport
            
            self.betanet_transport = BetaNetFogTransport(
                privacy_mode=privacy_mode,
                enable_covert=True,
                mobile_optimization=True
            )
            
            self.privacy_routing_enabled = True
            logger.info("BetaNet privacy transport initialized")
            return True
            
        except ImportError as e:
            logger.warning(f"BetaNet not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize BetaNet transport: {e}")
            return False
    
    async def send_private_federated_data(
        self,
        destination: str,
        data: Dict[str, Any],
        privacy_level: str = "high"
    ) -> Dict[str, Any]:
        """Send federated data with privacy protection"""
        
        if not self.betanet_transport:
            raise RuntimeError("BetaNet transport not initialized")
        
        # Serialize data
        import json
        serialized_data = json.dumps(data).encode('utf-8')
        
        # Send using BetaNet with appropriate priority
        priority = "high" if privacy_level in ["high", "ultra"] else "normal"
        
        result = await self.betanet_transport.send_job_data(
            serialized_data, 
            destination, 
            priority=priority
        )
        
        return result
    
    async def receive_private_federated_data(
        self,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Receive federated data with privacy protection"""
        
        if not self.betanet_transport:
            return None
        
        result = await self.betanet_transport.receive_job_data(timeout)
        
        if result and result.get("data"):
            try:
                import json
                deserialized_data = json.loads(result["data"].decode('utf-8'))
                return {
                    "data": deserialized_data,
                    "transport_info": result
                }
            except Exception as e:
                logger.error(f"Failed to deserialize received data: {e}")
        
        return None
    
    def get_transport_stats(self) -> Dict[str, Any]:
        """Get BetaNet transport statistics"""
        
        if self.betanet_transport:
            return self.betanet_transport.get_transport_stats()
        
        return {"transport_available": False}
```

---

## 3. Fog Computing Integration

### 3.1 Resource Discovery Integration

```python
# File: infrastructure/distributed_inference/integrations/fog_integration.py

import asyncio
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class FogFederatedIntegration:
    """Integration with fog computing infrastructure"""
    
    def __init__(self):
        self.fog_coordinator = None
        self.harvest_manager = None
        self.available_fog_resources = {}
        self.active_fog_sessions = {}
    
    async def initialize(self):
        """Initialize fog computing integration"""
        
        try:
            from infrastructure.fog.integration.fog_coordinator import FogCoordinator
            
            # Connect to existing fog coordinator
            # In production, would get reference to running fog coordinator
            self.fog_coordinator = None  # Placeholder for actual integration
            
            # If harvest manager is available
            try:
                from infrastructure.fog.compute.harvest_manager import FogHarvestManager
                # Connect to harvest manager for mobile device discovery
                pass
            except ImportError:
                logger.warning("Harvest manager not available")
            
            # Start resource monitoring
            asyncio.create_task(self._monitor_fog_resources())
            
            logger.info("Fog computing integration initialized")
            return True
            
        except ImportError as e:
            logger.warning(f"Fog computing not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize fog integration: {e}")
            return False
    
    async def _monitor_fog_resources(self):
        """Monitor available fog computing resources"""
        
        while True:
            try:
                # Discover fog compute resources
                fog_resources = await self._discover_fog_resources()
                
                # Update available resources
                for resource in fog_resources:
                    resource_id = resource["resource_id"]
                    self.available_fog_resources[resource_id] = {
                        **resource,
                        "last_updated": time.time(),
                        "federated_compatible": self._is_federated_compatible(resource)
                    }
                
                # Clean up stale resources
                current_time = time.time()
                stale_resources = [
                    rid for rid, rdata in self.available_fog_resources.items()
                    if current_time - rdata["last_updated"] > 600  # 10 minutes
                ]
                
                for rid in stale_resources:
                    del self.available_fog_resources[rid]
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Fog resource monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _discover_fog_resources(self) -> List[Dict[str, Any]]:
        """Discover available fog computing resources"""
        
        # Mock resource discovery - in production would query actual fog coordinator
        mock_resources = [
            {
                "resource_id": f"fog_edge_{i}",
                "resource_type": "edge_compute",
                "compute_gflops": 5.0 + i * 2,
                "memory_gb": 8.0 + i * 4,
                "network_mbps": 100.0,
                "location": f"edge_zone_{i}",
                "availability": 0.9,
                "cost_per_hour": 0.05
            }
            for i in range(3)
        ]
        
        # Add mobile harvest resources
        mock_resources.extend([
            {
                "resource_id": f"mobile_harvest_{i}",
                "resource_type": "mobile_device",
                "compute_gflops": 1.5 + i * 0.5,
                "memory_gb": 4.0 + i * 2,
                "battery_percent": 80.0,
                "charging_status": True,
                "wifi_connected": True,
                "temperature": 30.0,
                "availability": 0.7
            }
            for i in range(5)
        ])
        
        return mock_resources
    
    def _is_federated_compatible(self, resource: Dict[str, Any]) -> bool:
        """Check if fog resource is compatible with federated learning"""
        
        # Minimum requirements for federated participation
        min_compute = 1.0  # GFLOPS
        min_memory = 2.0   # GB
        
        compute = resource.get("compute_gflops", 0)
        memory = resource.get("memory_gb", 0)
        
        if compute < min_compute or memory < min_memory:
            return False
        
        # Mobile-specific checks
        if resource.get("resource_type") == "mobile_device":
            battery = resource.get("battery_percent", 0)
            temp = resource.get("temperature", 50)
            wifi = resource.get("wifi_connected", False)
            
            if battery < 30 or temp > 45 or not wifi:
                return False
        
        return True
    
    async def request_fog_compute_session(
        self,
        requirements: Dict[str, Any],
        duration_minutes: int = 60
    ) -> Optional[str]:
        """Request a fog computing session for federated training"""
        
        # Find suitable resource
        suitable_resources = [
            rid for rid, rdata in self.available_fog_resources.items()
            if (rdata["federated_compatible"] and 
                rdata["compute_gflops"] >= requirements.get("min_compute_gflops", 1.0) and
                rdata["memory_gb"] >= requirements.get("min_memory_gb", 2.0))
        ]
        
        if not suitable_resources:
            return None
        
        # Select best resource (highest compute capability)
        selected_resource_id = max(
            suitable_resources,
            key=lambda rid: self.available_fog_resources[rid]["compute_gflops"]
        )
        
        # Create session
        session_id = f"fog_session_{uuid.uuid4().hex[:8]}"
        
        self.active_fog_sessions[session_id] = {
            "session_id": session_id,
            "resource_id": selected_resource_id,
            "resource_data": self.available_fog_resources[selected_resource_id],
            "requirements": requirements,
            "start_time": time.time(),
            "duration_minutes": duration_minutes,
            "status": "active"
        }
        
        logger.info(f"Created fog compute session {session_id} on resource {selected_resource_id}")
        return session_id
    
    async def release_fog_compute_session(self, session_id: str) -> bool:
        """Release a fog computing session"""
        
        if session_id in self.active_fog_sessions:
            session = self.active_fog_sessions[session_id]
            session["status"] = "released"
            session["end_time"] = time.time()
            
            logger.info(f"Released fog compute session {session_id}")
            return True
        
        return False
    
    def get_available_fog_resources(
        self, 
        requirements: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get available fog resources matching requirements"""
        
        resources = list(self.available_fog_resources.values())
        
        if requirements:
            filtered_resources = []
            for resource in resources:
                if (resource["federated_compatible"] and
                    resource["compute_gflops"] >= requirements.get("min_compute_gflops", 0) and
                    resource["memory_gb"] >= requirements.get("min_memory_gb", 0)):
                    filtered_resources.append(resource)
            return filtered_resources
        
        return [r for r in resources if r["federated_compatible"]]
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get active fog computing sessions"""
        
        return [
            session for session in self.active_fog_sessions.values()
            if session["status"] == "active"
        ]
```

---

## 4. Configuration Management

### 4.1 Environment Configuration

```yaml
# File: config/federated.yaml

federated_systems:
  # Inference coordinator configuration
  inference:
    enabled: true
    coordinator_id: "aivillage_inference_coord"
    max_concurrent_requests: 1000
    cache_size_gb: 10.0
    load_balancing_strategy: "capability_aware"
    
    integration:
      p2p_network: true
      fog_computing: true
      betanet_privacy: true
      gateway_integration: true
    
    performance:
      request_timeout_ms: 30000
      model_distribution_timeout_ms: 10000
      result_aggregation_timeout_ms: 5000
    
    privacy:
      default_privacy_level: "medium"
      enable_covert_channels: true
      require_authentication: false

  # Training coordinator configuration
  training:
    enabled: true
    coordinator_id: "aivillage_training_coord"
    max_concurrent_jobs: 10
    default_rounds: 100
    round_timeout_minutes: 30
    
    participant_selection:
      strategy: "capability_aware"
      min_participants: 3
      max_participants: 50
      selection_timeout_ms: 10000
    
    privacy:
      default_epsilon: 1.0
      default_delta: 1e-5
      max_privacy_budget: 10.0
      enable_secure_aggregation: true
      differential_privacy_adaptive: true
    
    mobile:
      enabled: true
      min_battery_percent: 30
      max_temperature_celsius: 45
      require_wifi: true
      require_charging: false
      adaptive_batch_size: true
      thermal_management: true
    
    fog:
      enable_resource_harvesting: true
      prefer_edge_nodes: true
      dynamic_allocation: true
      harvest_during_idle: true
    
    aggregation:
      default_strategy: "federated_averaging"
      byzantine_tolerance: true
      enable_compression: true
      hierarchical_clusters: false

  # Network integration configuration
  network:
    p2p:
      enabled: true
      preferred_transport: "betanet"
      fallback_transport: "bitchat"
      discovery_interval_seconds: 300
      connection_timeout_seconds: 30
      enable_gossip_protocol: true
    
    fog:
      enabled: true
      resource_discovery_interval_seconds: 300
      session_timeout_minutes: 120
      auto_resource_allocation: true
    
    betanet:
      enabled: true
      privacy_mode: "balanced"
      enable_covert_channels: true
      mobile_optimization: true

  # Monitoring and metrics
  monitoring:
    enabled: true
    metrics_interval_seconds: 60
    health_check_interval_seconds: 30
    performance_analytics: true
    export_prometheus: true
    
    alerts:
      low_participant_threshold: 3
      high_failure_rate_threshold: 0.3
      privacy_budget_low_threshold: 1.0
```

### 4.2 Runtime Configuration

```python
# File: infrastructure/distributed_inference/config.py

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class FederatedConfig:
    """Configuration for federated systems"""
    
    # Load from YAML file
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'FederatedConfig':
        """Load configuration from YAML file"""
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls.from_dict(config_data.get('federated_systems', {}))
    
    @classmethod  
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FederatedConfig':
        """Create configuration from dictionary"""
        
        config = cls()
        
        # Parse inference configuration
        inference_config = config_dict.get('inference', {})
        config.inference_enabled = inference_config.get('enabled', True)
        config.inference_coordinator_id = inference_config.get('coordinator_id', 'inference_coord')
        config.max_concurrent_requests = inference_config.get('max_concurrent_requests', 1000)
        config.cache_size_gb = inference_config.get('cache_size_gb', 10.0)
        config.load_balancing_strategy = inference_config.get('load_balancing_strategy', 'capability_aware')
        
        # Parse training configuration  
        training_config = config_dict.get('training', {})
        config.training_enabled = training_config.get('enabled', True)
        config.training_coordinator_id = training_config.get('coordinator_id', 'training_coord')
        config.max_concurrent_jobs = training_config.get('max_concurrent_jobs', 10)
        config.default_rounds = training_config.get('default_rounds', 100)
        
        # Parse network configuration
        network_config = config_dict.get('network', {})
        config.p2p_enabled = network_config.get('p2p', {}).get('enabled', True)
        config.fog_enabled = network_config.get('fog', {}).get('enabled', True)
        config.betanet_enabled = network_config.get('betanet', {}).get('enabled', True)
        
        return config
    
    def __init__(self):
        # Default values
        self.inference_enabled = True
        self.inference_coordinator_id = "inference_coord"
        self.max_concurrent_requests = 1000
        self.cache_size_gb = 10.0
        self.load_balancing_strategy = "capability_aware"
        
        self.training_enabled = True  
        self.training_coordinator_id = "training_coord"
        self.max_concurrent_jobs = 10
        self.default_rounds = 100
        
        self.p2p_enabled = True
        self.fog_enabled = True
        self.betanet_enabled = True

# Global configuration instance
federated_config: Optional[FederatedConfig] = None

def get_federated_config() -> FederatedConfig:
    """Get global federated configuration"""
    global federated_config
    
    if federated_config is None:
        # Try to load from default locations
        config_paths = [
            Path("config/federated.yaml"),
            Path("../config/federated.yaml"),
            Path("/etc/aivillage/federated.yaml")
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                federated_config = FederatedConfig.load_from_file(config_path)
                break
        else:
            # Use default configuration
            federated_config = FederatedConfig()
    
    return federated_config
```

---

## 5. Testing and Validation

### 5.1 Integration Tests

```python
# File: tests/integration/test_federated_systems.py

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from infrastructure.distributed_inference import (
    FederatedInferenceCoordinator, 
    EnhancedFederatedTrainingCoordinator
)

class TestFederatedInferenceIntegration:
    """Integration tests for federated inference system"""
    
    @pytest.fixture
    async def inference_coordinator(self):
        """Create inference coordinator for testing"""
        
        coordinator = FederatedInferenceCoordinator(
            coordinator_id="test_inference",
            integration_config={
                "p2p_network": False,  # Disable for testing
                "fog_computing": False,
                "betanet_privacy": False
            }
        )
        
        await coordinator.initialize()
        yield coordinator
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_inference_request_flow(self, inference_coordinator):
        """Test complete inference request flow"""
        
        # Submit inference request
        request_id = await inference_coordinator.submit_inference_request(
            client_id="test_client",
            model_id="resnet50_imagenet",
            input_data={"image_tensor": [[0.1, 0.2], [0.3, 0.4]]},
            preferences={"privacy_level": "medium"}
        )
        
        assert request_id is not None
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Get result
        result = await inference_coordinator.get_inference_result(request_id)
        assert result is not None
        assert result["status"] in ["completed", "processing"]
        
        # Get status
        status = await inference_coordinator.get_inference_status(request_id)
        assert status is not None
        assert status["request_id"] == request_id
    
    @pytest.mark.asyncio
    async def test_model_registration(self, inference_coordinator):
        """Test model registration"""
        
        from infrastructure.distributed_inference.federated_inference_coordinator import (
            ModelInfo, NodeType, PrivacyLevel
        )
        
        model_info = ModelInfo(
            model_id="test_model",
            model_name="Test Model",
            model_version="1.0",
            model_type="classification",
            framework="pytorch",
            model_size_mb=100.0,
            memory_requirements_gb=2.0,
            compute_requirements_gflops=5.0,
            supported_node_types={NodeType.MOBILE, NodeType.EDGE},
            privacy_requirements=PrivacyLevel.MEDIUM
        )
        
        success = await inference_coordinator.register_model(model_info)
        assert success is True
        
        # Verify model is registered
        stats = await inference_coordinator.get_coordinator_stats()
        assert stats["statistics"]["registered_models"] >= 1


class TestFederatedTrainingIntegration:
    """Integration tests for federated training system"""
    
    @pytest.fixture
    async def training_coordinator(self):
        """Create training coordinator for testing"""
        
        coordinator = EnhancedFederatedTrainingCoordinator(
            coordinator_id="test_training",
            enable_fog_integration=False,  # Disable for testing
            enable_p2p_integration=False,
            enable_privacy_protocols=False
        )
        
        await coordinator.initialize()
        yield coordinator
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_training_job_creation(self, training_coordinator):
        """Test training job creation and execution"""
        
        from infrastructure.distributed_inference.enhanced_federated_training import (
            TrainingJobConfig, TrainingJobType
        )
        
        job_config = TrainingJobConfig(
            job_id="test_job_001",
            job_name="Test Training Job",
            job_type=TrainingJobType.STANDARD,
            model_architecture="simple_cnn",
            dataset_requirements={},
            total_rounds=5,  # Short test
            min_participants=1
        )
        
        job_id = await training_coordinator.create_enhanced_training_job(job_config)
        assert job_id == "test_job_001"
        
        # Wait for job to start
        await asyncio.sleep(1)
        
        # Check job status
        status = await training_coordinator.get_job_status(job_id)
        assert status is not None
        assert status["job_id"] == job_id
        assert status["status"] in ["running", "completed"]
    
    @pytest.mark.asyncio
    async def test_participant_registration(self, training_coordinator):
        """Test participant registration"""
        
        from infrastructure.distributed_inference.enhanced_federated_training import (
            EnhancedParticipant
        )
        
        participant = EnhancedParticipant(
            participant_id="test_participant",
            node_type="edge",
            capabilities={
                "compute_gflops": 10.0,
                "memory_gb": 16.0,
                "network_mbps": 100.0
            }
        )
        
        training_coordinator.enhanced_participants[participant.participant_id] = participant
        
        # Verify participant is registered
        stats = await training_coordinator.get_enhanced_stats()
        assert stats["participant_summary"]["total"] >= 1


class TestSystemIntegration:
    """Test integration between components"""
    
    @pytest.mark.asyncio
    async def test_federated_systems_coordination(self):
        """Test coordination between inference and training systems"""
        
        # Create both systems
        inference_coord = FederatedInferenceCoordinator(
            coordinator_id="test_inference_integration",
            integration_config={"p2p_network": False, "fog_computing": False, "betanet_privacy": False}
        )
        
        training_coord = EnhancedFederatedTrainingCoordinator(
            coordinator_id="test_training_integration",
            enable_fog_integration=False,
            enable_p2p_integration=False,
            enable_privacy_protocols=False
        )
        
        try:
            # Initialize both
            await inference_coord.initialize()
            await training_coord.initialize()
            
            # Test model sharing (inference model could be used for training)
            from infrastructure.distributed_inference.federated_inference_coordinator import (
                ModelInfo, NodeType, PrivacyLevel
            )
            
            shared_model = ModelInfo(
                model_id="shared_test_model",
                model_name="Shared Test Model", 
                model_version="1.0",
                model_type="classification",
                framework="pytorch",
                model_size_mb=50.0,
                memory_requirements_gb=1.0,
                compute_requirements_gflops=2.0,
                supported_node_types={NodeType.MOBILE, NodeType.EDGE},
                privacy_requirements=PrivacyLevel.LOW
            )
            
            # Register in inference system
            inference_success = await inference_coord.register_model(shared_model)
            assert inference_success is True
            
            # Training system could use this model for federated training
            from infrastructure.distributed_inference.enhanced_federated_training import (
                TrainingJobConfig, TrainingJobType
            )
            
            job_config = TrainingJobConfig(
                job_id="shared_model_job",
                job_name="Shared Model Training",
                job_type=TrainingJobType.STANDARD,
                model_architecture="shared_test_model",  # Reference to shared model
                dataset_requirements={},
                total_rounds=3,
                min_participants=1
            )
            
            training_job_id = await training_coord.create_enhanced_training_job(job_config)
            assert training_job_id is not None
            
        finally:
            # Cleanup
            await inference_coord.stop()
            await training_coord.stop()
```

### 5.2 Performance Testing

```python
# File: tests/performance/test_federated_performance.py

import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class TestFederatedPerformance:
    """Performance tests for federated systems"""
    
    @pytest.mark.asyncio
    async def test_inference_throughput(self):
        """Test inference system throughput"""
        
        from infrastructure.distributed_inference import FederatedInferenceCoordinator
        
        coordinator = FederatedInferenceCoordinator(
            coordinator_id="perf_test_inference",
            max_concurrent_requests=100,
            integration_config={"p2p_network": False, "fog_computing": False, "betanet_privacy": False}
        )
        
        await coordinator.initialize()
        
        try:
            # Submit multiple concurrent requests
            num_requests = 50
            start_time = time.time()
            
            request_tasks = []
            for i in range(num_requests):
                task = coordinator.submit_inference_request(
                    client_id=f"perf_client_{i}",
                    model_id="resnet50_imagenet",
                    input_data={"test_data": f"request_{i}"},
                    preferences={"privacy_level": "low"}  # Faster processing
                )
                request_tasks.append(task)
            
            # Wait for all requests to be submitted
            request_ids = await asyncio.gather(*request_tasks)
            
            submission_time = time.time() - start_time
            
            # Wait for processing to complete
            await asyncio.sleep(5)
            
            # Check completion rate
            completed_requests = 0
            for request_id in request_ids:
                result = await coordinator.get_inference_result(request_id)
                if result and result.get("status") == "completed":
                    completed_requests += 1
            
            completion_rate = completed_requests / num_requests
            throughput = num_requests / submission_time
            
            print(f"Submission throughput: {throughput:.2f} requests/second")
            print(f"Completion rate: {completion_rate:.2%}")
            
            # Performance assertions
            assert throughput > 10  # Should handle at least 10 requests/second
            assert completion_rate > 0.8  # At least 80% completion rate
            
        finally:
            await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_training_scalability(self):
        """Test training system scalability"""
        
        from infrastructure.distributed_inference.enhanced_federated_training import (
            EnhancedFederatedTrainingCoordinator, TrainingJobConfig, TrainingJobType
        )
        
        coordinator = EnhancedFederatedTrainingCoordinator(
            coordinator_id="perf_test_training",
            enable_fog_integration=False,
            enable_p2p_integration=False,
            enable_privacy_protocols=False
        )
        
        await coordinator.initialize()
        
        try:
            # Test with varying participant counts
            participant_counts = [5, 10, 20]
            performance_results = {}
            
            for participant_count in participant_counts:
                # Add mock participants
                from infrastructure.distributed_inference.enhanced_federated_training import EnhancedParticipant
                
                for i in range(participant_count):
                    participant = EnhancedParticipant(
                        participant_id=f"perf_participant_{i}",
                        node_type="edge",
                        capabilities={
                            "compute_gflops": 5.0,
                            "memory_gb": 8.0,
                            "network_mbps": 100.0
                        }
                    )
                    coordinator.enhanced_participants[participant.participant_id] = participant
                
                # Create training job
                job_config = TrainingJobConfig(
                    job_id=f"perf_job_{participant_count}",
                    job_name=f"Performance Test {participant_count} participants",
                    job_type=TrainingJobType.STANDARD,
                    model_architecture="test_model",
                    dataset_requirements={},
                    total_rounds=3,  # Short test
                    min_participants=min(3, participant_count),
                    max_participants=participant_count
                )
                
                start_time = time.time()
                job_id = await coordinator.create_enhanced_training_job(job_config)
                
                # Wait for job to complete
                max_wait_time = 60  # 1 minute max
                job_completed = False
                
                for _ in range(max_wait_time):
                    status = await coordinator.get_job_status(job_id)
                    if status and status.get("status") == "completed":
                        job_completed = True
                        break
                    await asyncio.sleep(1)
                
                completion_time = time.time() - start_time
                
                performance_results[participant_count] = {
                    "completion_time": completion_time,
                    "completed": job_completed,
                    "participants": participant_count
                }
                
                # Clear participants for next test
                coordinator.enhanced_participants.clear()
            
            # Analyze performance scaling
            for count, result in performance_results.items():
                print(f"Participants: {count}, Time: {result['completion_time']:.2f}s, Completed: {result['completed']}")
            
            # Performance assertions
            assert all(r["completed"] for r in performance_results.values())
            
            # Should scale reasonably (not exponentially worse)
            if len(performance_results) >= 2:
                times = [r["completion_time"] for r in performance_results.values()]
                scaling_factor = max(times) / min(times)
                assert scaling_factor < 5  # Should not be more than 5x slower with more participants
            
        finally:
            await coordinator.stop()
```

---

## 6. Deployment and Monitoring

### 6.1 Docker Deployment

```dockerfile
# File: docker/Dockerfile.federated

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY requirements-federated.txt .

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install -r requirements-federated.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/federated/status || exit 1

# Run command
CMD ["python", "-m", "uvicorn", "core.gateway.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# File: docker/docker-compose.federated.yml

version: '3.8'

services:
  aivillage-federated:
    build:
      context: ..
      dockerfile: docker/Dockerfile.federated
    ports:
      - "8000:8000"
      - "8001:8001" 
      - "8002:8002"
    environment:
      - ENVIRONMENT=production
      - API_KEY=${API_KEY}
      - SECRET_KEY=${SECRET_KEY}
      - ENABLE_FEDERATED=true
    volumes:
      - ../config:/app/config
      - ../logs:/app/logs
      - federated-cache:/app/cache
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/federated/status"]
      interval: 30s
      timeout: 10s
      retries: 3
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=aivillage
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    restart: unless-stopped

volumes:
  federated-cache:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:
```

### 6.2 Monitoring Configuration

```yaml
# File: docker/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'aivillage-federated'
    static_configs:
      - targets: ['aivillage-federated:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

rule_files:
  - "federated_alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

```yaml
# File: docker/federated_alerts.yml

groups:
  - name: federated_systems
    rules:
      - alert: HighInferenceLatency
        expr: federated_inference_latency_p95 > 5000
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "95th percentile inference latency is {{ $value }}ms"

      - alert: LowParticipantCount
        expr: federated_training_participants < 3
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low participant count for federated training"
          description: "Only {{ $value }} participants available"

      - alert: HighFailureRate
        expr: (federated_requests_failed / federated_requests_total) > 0.1
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High failure rate in federated systems"
          description: "Failure rate is {{ $value | humanizePercentage }}"

      - alert: PrivacyBudgetLow
        expr: federated_privacy_budget_remaining < 1.0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Privacy budget running low"
          description: "Privacy budget remaining: {{ $value }}"
```

### 6.3 Production Deployment Script

```bash
#!/bin/bash
# File: scripts/deploy_federated.sh

set -e

echo "🚀 Deploying AIVillage Federated Systems..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed"; exit 1; }

# Set environment variables
export ENVIRONMENT=${ENVIRONMENT:-production}
export API_KEY=${API_KEY:-$(openssl rand -hex 32)}
export SECRET_KEY=${SECRET_KEY:-$(openssl rand -hex 32)}
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(openssl rand -hex 16)}
export GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-$(openssl rand -hex 16)}

echo "✅ Environment configured"

# Create necessary directories
mkdir -p logs config cache

# Generate configuration if it doesn't exist
if [ ! -f "config/federated.yaml" ]; then
    echo "📝 Generating default federated configuration..."
    cat > config/federated.yaml << EOF
federated_systems:
  inference:
    enabled: true
    max_concurrent_requests: 1000
    cache_size_gb: 10.0
    
  training:
    enabled: true
    max_concurrent_jobs: 10
    default_rounds: 100
    
  network:
    p2p:
      enabled: true
    fog:
      enabled: true
    betanet:
      enabled: true
      
  monitoring:
    enabled: true
    export_prometheus: true
EOF
fi

# Pull latest images
echo "📦 Pulling Docker images..."
docker-compose -f docker/docker-compose.federated.yml pull

# Start services
echo "🔄 Starting federated services..."
docker-compose -f docker/docker-compose.federated.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
for service in aivillage-federated redis postgres; do
    if docker-compose -f docker/docker-compose.federated.yml ps $service | grep -q "Up.*healthy"; then
        echo "✅ $service is healthy"
    else
        echo "❌ $service is not healthy"
        docker-compose -f docker/docker-compose.federated.yml logs $service
        exit 1
    fi
done

# Test federated endpoints
echo "🧪 Testing federated endpoints..."
curl -f http://localhost:8000/v1/federated/status || {
    echo "❌ Federated status endpoint not responding"
    exit 1
}

echo "✅ Federated systems deployed successfully!"
echo ""
echo "📊 Access points:"
echo "  - API Gateway: http://localhost:8000"
echo "  - Federated Status: http://localhost:8000/v1/federated/status"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (admin/${GRAFANA_PASSWORD})"
echo ""
echo "🔑 Important credentials:"
echo "  - API Key: $API_KEY"
echo "  - Secret Key: $SECRET_KEY"
echo "  - Postgres Password: $POSTGRES_PASSWORD"
echo "  - Grafana Password: $GRAFANA_PASSWORD"
echo ""
echo "💾 Save these credentials securely!"
```

---

## 7. Best Practices and Troubleshooting

### 7.1 Best Practices

1. **Security**:
   - Always use secure random keys in production
   - Enable privacy features for sensitive workloads
   - Regularly rotate authentication credentials
   - Monitor for unusual patterns in federated traffic

2. **Performance**:
   - Configure appropriate cache sizes based on available memory
   - Tune participant selection strategies for your use case
   - Monitor resource utilization on mobile devices
   - Use hierarchical aggregation for large participant sets

3. **Reliability**:
   - Implement proper error handling and retry logic
   - Set reasonable timeouts for distributed operations
   - Monitor system health and set up alerts
   - Test failure scenarios regularly

4. **Privacy**:
   - Configure differential privacy parameters carefully
   - Monitor privacy budget consumption
   - Use secure aggregation for sensitive models
   - Audit privacy-preserving mechanisms regularly

### 7.2 Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Low participant count | Training jobs fail to start | Check node discovery, reduce min_participants |
| High inference latency | Slow response times | Optimize model distribution, check network |
| Privacy budget exhaustion | Participants excluded | Increase privacy budget or adjust epsilon |
| Memory issues | OOM errors | Reduce cache size, optimize model sizes |
| Network connectivity | P2P discovery fails | Check firewall, NAT configuration |

### 7.3 Performance Tuning

```python
# File: scripts/performance_tuning.py

"""
Performance tuning utilities for federated systems
"""

import asyncio
import logging
from typing import Dict, Any

async def tune_inference_system(coordinator, target_latency_ms: int = 1000):
    """Automatically tune inference system for target latency"""
    
    current_stats = await coordinator.get_coordinator_stats()
    current_latency = current_stats["statistics"].get("avg_response_time_ms", 0)
    
    if current_latency > target_latency_ms:
        # Increase cache size
        coordinator.cache_size_gb *= 1.5
        
        # Prefer faster nodes
        coordinator.load_balancing_strategy = LoadBalancingStrategy.LATENCY_OPTIMIZED
        
        # Reduce privacy overhead for non-sensitive requests
        # (implementation specific)
        
        logging.info(f"Tuned inference system for {target_latency_ms}ms target latency")

async def tune_training_system(coordinator, target_participants: int = 10):
    """Automatically tune training system for target participant count"""
    
    current_stats = await coordinator.get_enhanced_stats()
    current_participants = current_stats["participant_summary"]["total"]
    
    if current_participants < target_participants:
        # Relax participant requirements
        for participant in coordinator.enhanced_participants.values():
            if participant.trust_score > 0.3:  # Lower threshold
                participant.compute_availability = 1.0
        
        # Enable mobile-friendly settings
        # (implementation specific)
        
        logging.info(f"Tuned training system for {target_participants} target participants")

if __name__ == "__main__":
    # Example usage
    asyncio.run(tune_inference_system(inference_coordinator, target_latency_ms=500))
    asyncio.run(tune_training_system(training_coordinator, target_participants=20))
```

---

## Conclusion

This implementation guide provides a comprehensive foundation for deploying federated inference and training systems in the AIVillage ecosystem. The modular architecture allows for gradual rollout and integration with existing systems while maintaining flexibility for future enhancements.

Key implementation highlights:

1. **Seamless Integration**: Works with existing gateway, P2P, and fog systems
2. **Production Ready**: Includes monitoring, alerting, and deployment automation  
3. **Privacy Focused**: Multiple layers of privacy protection
4. **Mobile Optimized**: Efficient operation on resource-constrained devices
5. **Scalable Design**: Handles growth in participants and model complexity

For production deployment, follow the phased rollout approach:
- Start with basic inference coordinator
- Add training capabilities gradually
- Enable privacy features incrementally
- Scale to full distributed operation

The architecture provides a solid foundation for advanced federated AI operations while maintaining the security, privacy, and performance requirements of the AIVillage ecosystem.