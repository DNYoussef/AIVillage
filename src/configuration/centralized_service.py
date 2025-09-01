"""
Centralized Configuration Service with Distributed Replication
Main service orchestrating all configuration management components
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import hashlib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

# Import our configuration management components
from .unified_config_manager import UnifiedConfigurationManager, get_config_manager
from .distributed_cache_integration import DistributedCacheManager, create_distributed_cache_manager
from .sequential_config_analyzer import SequentialConfigurationAnalyzer, SequentialThinkingMCPIntegration
from .memory_mcp_integration import ConfigurationMemoryManager, create_memory_manager
from .github_mcp_deployment import GitHubMCPConfigurationDeployment, create_github_deployment_manager
from .validation_and_hot_reload import ConfigurationValidationSystem, create_validation_system

logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    """Health status of the configuration service"""
    service_name: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    error_message: Optional[str] = None
    response_time_ms: Optional[int] = None

@dataclass
class ConfigurationServiceMetrics:
    """Metrics for the configuration service"""
    total_configurations: int
    validation_pass_rate: float
    cache_hit_rate: float
    deployment_success_rate: float
    average_response_time_ms: float
    total_requests: int
    error_count: int
    uptime_seconds: int

class CentralizedConfigurationService:
    """Main centralized configuration service"""
    
    def __init__(self,
                 config_directories: List[str] = None,
                 redis_nodes: List[Dict[str, Any]] = None,
                 repo_owner: str = "aivillage",
                 repo_name: str = "AIVillage"):
        
        self.config_directories = config_directories or ["config"]
        self.redis_nodes = redis_nodes
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        
        # Core components
        self.unified_manager: Optional[UnifiedConfigurationManager] = None
        self.cache_manager: Optional[DistributedCacheManager] = None
        self.analyzer: Optional[SequentialConfigurationAnalyzer] = None
        self.memory_manager: Optional[ConfigurationMemoryManager] = None
        self.deployment_manager: Optional[GitHubMCPConfigurationDeployment] = None
        self.validation_system: Optional[ConfigurationValidationSystem] = None
        
        # Service state
        self.service_start_time = datetime.now()
        self._is_initialized = False
        self._health_status: Dict[str, ServiceHealth] = {}
        self._metrics: ConfigurationServiceMetrics = ConfigurationServiceMetrics(
            total_configurations=0,
            validation_pass_rate=0.0,
            cache_hit_rate=0.0,
            deployment_success_rate=0.0,
            average_response_time_ms=0.0,
            total_requests=0,
            error_count=0,
            uptime_seconds=0
        )
        
        # FastAPI app
        self.app = FastAPI(
            title="AIVillage Configuration Service",
            description="Centralized configuration management with distributed caching",
            version="1.0.0"
        )
        
        # Setup API routes
        self._setup_api_routes()
        
    async def initialize(self):
        """Initialize all configuration management components"""
        
        logger.info("Initializing Centralized Configuration Service")
        
        try:
            # Initialize unified configuration manager
            self.unified_manager = await get_config_manager()
            await self._update_component_health("unified_manager", "healthy")
            
            # Initialize distributed cache manager
            if self.redis_nodes:
                self.cache_manager = await create_distributed_cache_manager(
                    redis_nodes=self.redis_nodes,
                    replication_factor=2,
                    consistency_level="eventual"
                )
                await self._update_component_health("cache_manager", "healthy")
            else:
                logger.warning("No Redis nodes configured, cache manager disabled")
                
            # Initialize configuration analyzer
            self.analyzer = SequentialConfigurationAnalyzer()
            await self._update_component_health("analyzer", "healthy")
            
            # Initialize memory manager
            self.memory_manager = await create_memory_manager()
            await self._update_component_health("memory_manager", "healthy")
            
            # Initialize deployment manager
            self.deployment_manager = await create_github_deployment_manager(
                self.repo_owner, self.repo_name
            )
            await self._update_component_health("deployment_manager", "healthy")
            
            # Initialize validation system
            self.validation_system = await create_validation_system(self.config_directories)
            await self._update_component_health("validation_system", "healthy")
            
            # Register hot-reload callback
            await self.validation_system.register_reload_callback(self._on_configuration_changed)
            
            # Start background tasks
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._metrics_update_loop())
            
            self._is_initialized = True
            logger.info("Centralized Configuration Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration service: {e}")
            raise
            
    def _setup_api_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            
            if not self._is_initialized:
                raise HTTPException(status_code=503, detail="Service not initialized")
                
            overall_status = "healthy"
            unhealthy_components = [
                name for name, health in self._health_status.items()
                if health.status == "unhealthy"
            ]
            
            if unhealthy_components:
                overall_status = "degraded"
                
            return {
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "components": {name: asdict(health) for name, health in self._health_status.items()},
                "uptime_seconds": (datetime.now() - self.service_start_time).total_seconds()
            }
            
        @self.app.get("/config")
        async def get_configuration(key: Optional[str] = None, use_cache: bool = True):
            """Get configuration with optional key filtering"""
            
            if not self._is_initialized or not self.unified_manager:
                raise HTTPException(status_code=503, detail="Service not initialized")
                
            try:
                config = await self.unified_manager.get_configuration(key, use_cache)
                self._metrics.total_requests += 1
                return {"config": config, "timestamp": datetime.now().isoformat()}
                
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Configuration retrieval error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/config")
        async def set_configuration(config_update: Dict[str, Any]):
            """Update configuration"""
            
            if not self._is_initialized or not self.unified_manager:
                raise HTTPException(status_code=503, detail="Service not initialized")
                
            try:
                key = config_update.get("key")
                value = config_update.get("value")
                layer = config_update.get("layer", "runtime")
                
                if not key or value is None:
                    raise HTTPException(status_code=400, detail="Key and value required")
                    
                await self.unified_manager.set_configuration(key, value, layer)
                self._metrics.total_requests += 1
                
                return {"status": "updated", "timestamp": datetime.now().isoformat()}
                
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Configuration update error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/config/reload")
        async def reload_configuration():
            """Reload all configuration"""
            
            if not self._is_initialized or not self.unified_manager:
                raise HTTPException(status_code=503, detail="Service not initialized")
                
            try:
                await self.unified_manager.reload_configuration()
                return {"status": "reloaded", "timestamp": datetime.now().isoformat()}
                
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Configuration reload error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/config/validate")
        async def validate_configuration(file_path: Optional[str] = None):
            """Validate configuration files"""
            
            if not self._is_initialized or not self.validation_system:
                raise HTTPException(status_code=503, detail="Service not initialized")
                
            try:
                if file_path:
                    results = await self.validation_system.validate_configuration_file(file_path)
                    validation_data = [asdict(result) for result in results]
                else:
                    summary = await self.validation_system.get_validation_summary()
                    validation_data = summary
                    
                return {"validation": validation_data, "timestamp": datetime.now().isoformat()}
                
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Configuration validation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/config/analyze")
        async def analyze_configuration():
            """Analyze configuration hierarchy and patterns"""
            
            if not self._is_initialized or not self.analyzer:
                raise HTTPException(status_code=503, detail="Service not initialized")
                
            try:
                analysis = await self.analyzer.analyze_configuration_hierarchy(self.config_directories)
                return {"analysis": asdict(analysis), "timestamp": datetime.now().isoformat()}
                
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Configuration analysis error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/deployment/create")
        async def create_deployment(deployment_request: Dict[str, Any]):
            """Create configuration deployment"""
            
            if not self._is_initialized or not self.deployment_manager:
                raise HTTPException(status_code=503, detail="Service not initialized")
                
            try:
                environment = deployment_request.get("environment")
                config_files = deployment_request.get("config_files", [])
                strategy = deployment_request.get("strategy", "rolling")
                
                if not environment or not config_files:
                    raise HTTPException(status_code=400, detail="Environment and config_files required")
                    
                deployment = await self.deployment_manager.create_configuration_deployment(
                    environment, config_files, strategy
                )
                
                return {"deployment": asdict(deployment), "timestamp": datetime.now().isoformat()}
                
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Deployment creation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/deployment/{deployment_id}/deploy")
        async def deploy_configuration(deployment_id: str, force: bool = False):
            """Deploy configuration"""
            
            if not self._is_initialized or not self.deployment_manager:
                raise HTTPException(status_code=503, detail="Service not initialized")
                
            try:
                success = await self.deployment_manager.deploy_configuration(deployment_id, force)
                
                if success:
                    return {"status": "deployed", "deployment_id": deployment_id}
                else:
                    return {"status": "failed", "deployment_id": deployment_id}
                    
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Deployment error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/deployment/{deployment_id}/status")
        async def get_deployment_status(deployment_id: str):
            """Get deployment status"""
            
            if not self._is_initialized or not self.deployment_manager:
                raise HTTPException(status_code=503, detail="Service not initialized")
                
            try:
                status = await self.deployment_manager.get_deployment_status(deployment_id)
                return {"status": status, "timestamp": datetime.now().isoformat()}
                
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Deployment status error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/cache/stats")
        async def get_cache_stats():
            """Get cache statistics"""
            
            if not self._is_initialized or not self.cache_manager:
                return {"error": "Cache manager not available"}
                
            try:
                stats = await self.cache_manager.get_cache_stats()
                return {"cache_stats": stats, "timestamp": datetime.now().isoformat()}
                
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Cache stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/memory/patterns")
        async def get_memory_patterns():
            """Get learned configuration patterns"""
            
            if not self._is_initialized or not self.memory_manager:
                return {"error": "Memory manager not available"}
                
            try:
                trends = await self.memory_manager.analyze_configuration_trends()
                return {"patterns": trends, "timestamp": datetime.now().isoformat()}
                
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Memory patterns error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics"""
            
            self._metrics.uptime_seconds = int((datetime.now() - self.service_start_time).total_seconds())
            return {"metrics": asdict(self._metrics), "timestamp": datetime.now().isoformat()}
            
        @self.app.post("/snapshot/create")
        async def create_snapshot(snapshot_request: Dict[str, Any]):
            """Create configuration snapshot"""
            
            if not self._is_initialized or not self.validation_system:
                raise HTTPException(status_code=503, detail="Service not initialized")
                
            try:
                snapshot_name = snapshot_request.get("name", f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                await self.validation_system.create_configuration_snapshot(snapshot_name)
                
                return {"status": "created", "snapshot_name": snapshot_name}
                
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Snapshot creation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/snapshot/{snapshot_name}/rollback")
        async def rollback_to_snapshot(snapshot_name: str):
            """Rollback to configuration snapshot"""
            
            if not self._is_initialized or not self.validation_system:
                raise HTTPException(status_code=503, detail="Service not initialized")
                
            try:
                success = await self.validation_system.rollback_to_snapshot(snapshot_name)
                
                if success:
                    return {"status": "rolled_back", "snapshot_name": snapshot_name}
                else:
                    return {"status": "failed", "snapshot_name": snapshot_name}
                    
            except Exception as e:
                self._metrics.error_count += 1
                logger.error(f"Snapshot rollback error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
    async def _on_configuration_changed(self, file_path: str, config_data: Dict[str, Any]):
        """Handle configuration file changes"""
        
        logger.info(f"Configuration changed: {file_path}")
        
        # Invalidate caches
        if self.cache_manager:
            await self.cache_manager.invalidate_configuration(file_path, propagate=True)
            
        # Learn from configuration changes
        if self.memory_manager:
            # Record the configuration change pattern
            # This would be enhanced with actual learning logic
            pass
            
    async def _update_component_health(self, component_name: str, status: str, error_message: Optional[str] = None):
        """Update component health status"""
        
        self._health_status[component_name] = ServiceHealth(
            service_name=component_name,
            status=status,
            last_check=datetime.now(),
            error_message=error_message
        )
        
    async def _health_check_loop(self):
        """Background health check loop"""
        
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check unified manager
                if self.unified_manager:
                    try:
                        config = await self.unified_manager.get_configuration("health_check")
                        await self._update_component_health("unified_manager", "healthy")
                    except Exception as e:
                        await self._update_component_health("unified_manager", "unhealthy", str(e))
                        
                # Check cache manager
                if self.cache_manager:
                    try:
                        stats = await self.cache_manager.get_cache_stats()
                        await self._update_component_health("cache_manager", "healthy")
                    except Exception as e:
                        await self._update_component_health("cache_manager", "unhealthy", str(e))
                        
                # Check validation system
                if self.validation_system:
                    try:
                        summary = await self.validation_system.get_validation_summary()
                        await self._update_component_health("validation_system", "healthy")
                    except Exception as e:
                        await self._update_component_health("validation_system", "unhealthy", str(e))
                        
                logger.debug("Health check completed")
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                
    async def _metrics_update_loop(self):
        """Background metrics update loop"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Update validation metrics
                if self.validation_system:
                    try:
                        summary = await self.validation_system.get_validation_summary()
                        total_files = summary.get("total_files", 0)
                        files_clean = summary.get("files_clean", 0)
                        
                        if total_files > 0:
                            self._metrics.validation_pass_rate = files_clean / total_files
                        self._metrics.total_configurations = total_files
                        
                    except Exception as e:
                        logger.error(f"Metrics update error for validation: {e}")
                        
                # Update cache metrics
                if self.cache_manager:
                    try:
                        cache_stats = await self.cache_manager.get_cache_stats()
                        # Extract cache hit rate if available
                        node_stats = cache_stats.get("node_stats", {})
                        # This would be enhanced with actual hit rate calculation
                        
                    except Exception as e:
                        logger.error(f"Metrics update error for cache: {e}")
                        
                logger.debug("Metrics updated")
                
            except Exception as e:
                logger.error(f"Metrics update loop error: {e}")
                
    async def start_service(self, host: str = "0.0.0.0", port: int = 8090):
        """Start the configuration service"""
        
        if not self._is_initialized:
            await self.initialize()
            
        logger.info(f"Starting Configuration Service on {host}:{port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()

# Factory function
async def create_centralized_service(
    config_directories: List[str] = None,
    redis_nodes: List[Dict[str, Any]] = None,
    repo_owner: str = "aivillage", 
    repo_name: str = "AIVillage"
) -> CentralizedConfigurationService:
    """Create centralized configuration service"""
    
    service = CentralizedConfigurationService(
        config_directories=config_directories,
        redis_nodes=redis_nodes,
        repo_owner=repo_owner,
        repo_name=repo_name
    )
    
    await service.initialize()
    return service

# Docker deployment configuration
def create_docker_compose() -> str:
    """Generate docker-compose.yml for the configuration service"""
    
    return """
version: '3.8'

services:
  config-service:
    build:
      context: .
      dockerfile: src/configuration/Dockerfile
    ports:
      - "8090:8090"
    environment:
      - CONFIG_DIRECTORIES=config,tools/ci-cd/deployment/k8s
      - REDIS_NODES=redis-primary:6379,redis-replica:6379
      - REPO_OWNER=aivillage
      - REPO_NAME=AIVillage
      - LOG_LEVEL=INFO
    volumes:
      - ./config:/app/config:ro
      - ./tools:/app/tools:ro
      - ./data:/app/data
    depends_on:
      - redis-primary
      - redis-replica
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8090/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis-primary:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-primary-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  redis-replica:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    volumes:
      - redis-replica-data:/data
    command: redis-server --appendonly yes --slaveof redis-primary 6379
    depends_on:
      - redis-primary
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

volumes:
  redis-primary-data:
  redis-replica-data:
  grafana-data:

networks:
  default:
    name: config-service-network
"""

# Kubernetes deployment configuration
def create_k8s_manifests() -> Dict[str, str]:
    """Generate Kubernetes manifests for the configuration service"""
    
    manifests = {}
    
    # Deployment
    manifests["deployment.yaml"] = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: config-service
  namespace: aivillage-config
  labels:
    app: config-service
    version: v1
spec:
  replicas: 2
  selector:
    matchLabels:
      app: config-service
  template:
    metadata:
      labels:
        app: config-service
        version: v1
    spec:
      containers:
      - name: config-service
        image: aivillage/config-service:latest
        ports:
        - containerPort: 8090
          name: http
        env:
        - name: CONFIG_DIRECTORIES
          value: "config,tools/ci-cd/deployment/k8s"
        - name: REDIS_NODES
          value: "redis-primary:6379,redis-replica:6379"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            cpu: 200m
            memory: 512Mi
          limits:
            cpu: 500m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8090
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8090
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: config-volume
        configMap:
          name: aivillage-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: config-service-data
"""
    
    # Service
    manifests["service.yaml"] = """
apiVersion: v1
kind: Service
metadata:
  name: config-service
  namespace: aivillage-config
  labels:
    app: config-service
spec:
  selector:
    app: config-service
  ports:
  - name: http
    port: 8090
    targetPort: 8090
    protocol: TCP
  type: ClusterIP
"""
    
    # HPA
    manifests["hpa.yaml"] = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: config-service-hpa
  namespace: aivillage-config
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: config-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
    
    return manifests

if __name__ == "__main__":
    async def main():
        # Configuration for the service
        config_dirs = ["config", "tools/ci-cd/deployment/k8s"]
        redis_nodes = [
            {"node_id": "primary", "host": "localhost", "port": 6379, "weight": 100},
            {"node_id": "replica1", "host": "localhost", "port": 6380, "weight": 50},
        ]
        
        # Create and start service
        service = await create_centralized_service(
            config_directories=config_dirs,
            redis_nodes=redis_nodes
        )
        
        await service.start_service(host="0.0.0.0", port=8090)
        
    asyncio.run(main())