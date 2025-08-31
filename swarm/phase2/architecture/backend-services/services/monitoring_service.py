"""
Monitoring Service - Progress tracking and system health monitoring

This service is responsible for:
- System health tracking and metrics collection
- Performance monitoring and alerting
- Progress aggregation across services
- P2P/Fog infrastructure status monitoring
- Service discovery and health checks

Size Target: <400 lines
"""

import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid

from interfaces.service_contracts import (
    IMonitoringService, SystemMetrics, ServiceHealth, Alert,
    ServiceStatus, Event, ServiceHealthChangedEvent, AlertCreatedEvent
)

logger = logging.getLogger(__name__)


class HealthChecker:
    """Handles health checking of services."""
    
    def __init__(self, service_endpoints: Dict[str, str]):
        self.service_endpoints = service_endpoints
        self.health_cache: Dict[str, ServiceHealth] = {}
    
    async def check_service_health(self, service_name: str) -> ServiceHealth:
        """Check health of a specific service."""
        endpoint = self.service_endpoints.get(service_name)
        if not endpoint:
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time=0.0,
                error_rate=1.0,
                uptime_percentage=0.0
            )
        
        start_time = datetime.now()
        try:
            # In a real implementation, this would make an HTTP request to the service
            # For now, we'll simulate health checking
            await asyncio.sleep(0.01)  # Simulate network latency
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Simulate health status (in real implementation, check actual service)
            status = ServiceStatus.HEALTHY
            error_rate = 0.0
            uptime_percentage = 0.99
            
            return ServiceHealth(
                service_name=service_name,
                status=status,
                response_time=response_time,
                error_rate=error_rate,
                uptime_percentage=uptime_percentage
            )
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Health check failed for {service_name}: {e}")
            
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time=response_time,
                error_rate=1.0,
                uptime_percentage=0.0
            )


class MetricsCollector:
    """Collects system metrics."""
    
    @staticmethod
    def collect_system_metrics() -> SystemMetrics:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            
            return SystemMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent / 100.0,
                disk_usage=disk.percent / 100.0,
                network_io={
                    "bytes_sent": float(net_io.bytes_sent),
                    "bytes_recv": float(net_io.bytes_recv),
                    "packets_sent": float(net_io.packets_sent),
                    "packets_recv": float(net_io.packets_recv)
                }
            )
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={}
            )


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, event_publisher=None):
        self.event_publisher = event_publisher
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules = {
            "high_cpu": {"threshold": 0.8, "severity": "warning"},
            "high_memory": {"threshold": 0.9, "severity": "warning"},
            "service_down": {"threshold": 0.0, "severity": "error"},
            "high_error_rate": {"threshold": 0.1, "severity": "warning"}
        }
    
    async def check_alerts(self, service_name: str, metrics: SystemMetrics, 
                          health: ServiceHealth):
        """Check if any alert conditions are met."""
        alerts_to_create = []
        
        # CPU alert
        if metrics.cpu_usage > self.alert_rules["high_cpu"]["threshold"]:
            alerts_to_create.append(Alert(
                severity=self.alert_rules["high_cpu"]["severity"],
                service=service_name,
                message=f"High CPU usage: {metrics.cpu_usage:.1%}",
                metadata={"cpu_usage": metrics.cpu_usage}
            ))
        
        # Memory alert
        if metrics.memory_usage > self.alert_rules["high_memory"]["threshold"]:
            alerts_to_create.append(Alert(
                severity=self.alert_rules["high_memory"]["severity"],
                service=service_name,
                message=f"High memory usage: {metrics.memory_usage:.1%}",
                metadata={"memory_usage": metrics.memory_usage}
            ))
        
        # Service health alert
        if health.status == ServiceStatus.UNHEALTHY:
            alerts_to_create.append(Alert(
                severity="error",
                service=service_name,
                message=f"Service {service_name} is unhealthy",
                metadata={"status": health.status.value, "uptime": health.uptime_percentage}
            ))
        
        # Error rate alert
        if health.error_rate > self.alert_rules["high_error_rate"]["threshold"]:
            alerts_to_create.append(Alert(
                severity=self.alert_rules["high_error_rate"]["severity"],
                service=service_name,
                message=f"High error rate: {health.error_rate:.1%}",
                metadata={"error_rate": health.error_rate}
            ))
        
        # Create alerts
        for alert in alerts_to_create:
            await self.create_alert(alert)
    
    async def create_alert(self, alert: Alert):
        """Create a new alert."""
        self.alerts[alert.alert_id] = alert
        
        # Publish alert event
        if self.event_publisher:
            event = AlertCreatedEvent(
                source_service="monitoring_service",
                data=alert.dict()
            )
            await self.event_publisher.publish(event)
        
        logger.warning(f"Alert created: {alert.severity} - {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]


class MonitoringService(IMonitoringService):
    """Implementation of the Monitoring Service."""
    
    def __init__(self, 
                 service_endpoints: Dict[str, str] = None,
                 event_publisher=None,
                 check_interval: int = 30):
        self.service_endpoints = service_endpoints or {
            "training_service": "http://training:8001/health",
            "model_service": "http://model:8002/health",
            "websocket_service": "http://websocket:8003/health",
            "api_service": "http://api:8000/health"
        }
        self.event_publisher = event_publisher
        self.check_interval = check_interval
        
        # Components
        self.health_checker = HealthChecker(self.service_endpoints)
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(event_publisher)
        
        # Data storage
        self.service_metrics: Dict[str, List[SystemMetrics]] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.system_overview = {
            "overall_status": ServiceStatus.HEALTHY,
            "total_services": len(self.service_endpoints),
            "healthy_services": 0,
            "last_updated": datetime.now()
        }
        
        # Start monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def record_metrics(self, service_name: str, metrics: SystemMetrics) -> bool:
        """Record system metrics for a service."""
        try:
            if service_name not in self.service_metrics:
                self.service_metrics[service_name] = []
            
            # Keep only last 100 metrics to prevent memory bloat
            self.service_metrics[service_name].append(metrics)
            if len(self.service_metrics[service_name]) > 100:
                self.service_metrics[service_name] = self.service_metrics[service_name][-100:]
            
            logger.debug(f"Recorded metrics for {service_name}: CPU={metrics.cpu_usage:.1%}, Memory={metrics.memory_usage:.1%}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record metrics for {service_name}: {e}")
            return False
    
    async def get_service_health(self, service_name: str) -> ServiceHealth:
        """Get health status of a service."""
        if service_name in self.service_health:
            return self.service_health[service_name]
        
        # If not cached, perform health check
        health = await self.health_checker.check_service_health(service_name)
        self.service_health[service_name] = health
        return health
    
    async def create_alert(self, alert: Alert) -> str:
        """Create a new alert."""
        await self.alert_manager.create_alert(alert)
        return alert.alert_id
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get overall system health overview."""
        try:
            # Count healthy services
            healthy_count = sum(1 for health in self.service_health.values() 
                              if health.status == ServiceStatus.HEALTHY)
            
            # Determine overall status
            if healthy_count == len(self.service_endpoints):
                overall_status = ServiceStatus.HEALTHY
            elif healthy_count > len(self.service_endpoints) * 0.5:
                overall_status = ServiceStatus.DEGRADED
            else:
                overall_status = ServiceStatus.UNHEALTHY
            
            # Get system metrics
            current_metrics = self.metrics_collector.collect_system_metrics()
            
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts()
            
            return {
                "overall_status": overall_status.value,
                "services": {
                    "total": len(self.service_endpoints),
                    "healthy": healthy_count,
                    "degraded": sum(1 for h in self.service_health.values() 
                                  if h.status == ServiceStatus.DEGRADED),
                    "unhealthy": sum(1 for h in self.service_health.values() 
                                   if h.status == ServiceStatus.UNHEALTHY)
                },
                "system_metrics": current_metrics.dict(),
                "alerts": {
                    "total": len(self.alert_manager.alerts),
                    "active": len(active_alerts),
                    "critical": len([a for a in active_alerts if a.severity == "critical"]),
                    "warnings": len([a for a in active_alerts if a.severity == "warning"])
                },
                "service_health": {name: health.dict() 
                                 for name, health in self.service_health.items()},
                "last_updated": datetime.now().isoformat(),
                "uptime": (datetime.now() - self.system_overview["last_updated"]).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system overview: {e}")
            return {
                "overall_status": ServiceStatus.UNHEALTHY.value,
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    async def get_service_metrics_history(self, service_name: str, 
                                        hours: int = 1) -> List[SystemMetrics]:
        """Get metrics history for a service."""
        if service_name not in self.service_metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [metrics for metrics in self.service_metrics[service_name] 
                if metrics.timestamp >= cutoff_time]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        if alert_id in self.alert_manager.alerts:
            self.alert_manager.alerts[alert_id].resolved = True
            logger.info(f"Resolved alert {alert_id}")
            return True
        return False
    
    async def _monitoring_loop(self):
        """Main monitoring loop that runs periodically."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                
                # Check each service
                for service_name in self.service_endpoints.keys():
                    # Collect metrics
                    metrics = self.metrics_collector.collect_system_metrics()
                    await self.record_metrics(service_name, metrics)
                    
                    # Check health
                    previous_health = self.service_health.get(service_name)
                    current_health = await self.health_checker.check_service_health(service_name)
                    self.service_health[service_name] = current_health
                    
                    # Check for health status changes
                    if previous_health and previous_health.status != current_health.status:
                        if self.event_publisher:
                            event = ServiceHealthChangedEvent(
                                source_service="monitoring_service",
                                data={
                                    "service_name": service_name,
                                    "previous_status": previous_health.status.value,
                                    "current_status": current_health.status.value
                                }
                            )
                            await self.event_publisher.publish(event)
                    
                    # Check for alerts
                    await self.alert_manager.check_alerts(service_name, metrics, current_health)
                
                # Update system overview
                self.system_overview["last_updated"] = datetime.now()
                
                logger.debug(f"Monitoring check completed for {len(self.service_endpoints)} services")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def shutdown(self):
        """Shutdown the monitoring service gracefully."""
        logger.info("Shutting down monitoring service")
        self.monitoring_task.cancel()
        try:
            await self.monitoring_task
        except asyncio.CancelledError:
            pass


# Service factory
def create_monitoring_service(service_endpoints: Dict[str, str] = None,
                            event_publisher=None,
                            check_interval: int = 30) -> MonitoringService:
    """Create and configure the Monitoring Service."""
    return MonitoringService(
        service_endpoints=service_endpoints,
        event_publisher=event_publisher,
        check_interval=check_interval
    )