"""
Fog Monitoring Service

Manages system monitoring and health tracking including:
- Service health monitoring
- Performance metrics collection
- System resource tracking
- Alert generation and management
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime, UTC
import psutil
import json

from ..interfaces.base_service import BaseFogService, ServiceStatus, ServiceHealthCheck


class FogMonitoringService(BaseFogService):
    """Service for monitoring fog computing system health and performance"""
    
    def __init__(self, service_name: str, config: Dict[str, Any], event_bus):
        super().__init__(service_name, config, event_bus)
        
        # Monitoring configuration
        self.monitoring_config = config.get("monitoring", {})
        self.node_id = config.get("node_id", "default")
        
        # Tracked services
        self.tracked_services: Dict[str, Dict[str, Any]] = {}
        self.alert_thresholds = self.monitoring_config.get("alert_thresholds", {
            "cpu_usage_percent": 80.0,
            "memory_usage_percent": 85.0,
            "disk_usage_percent": 90.0,
            "error_rate_threshold": 0.05,
            "response_time_ms": 5000.0
        })
        
        # Service metrics
        self.metrics = {
            "total_alerts": 0,
            "critical_alerts": 0,
            "services_monitored": 0,
            "system_cpu_usage": 0.0,
            "system_memory_usage": 0.0,
            "system_disk_usage": 0.0,
            "network_io_mb": 0.0,
            "uptime_seconds": 0,
            "last_alert_timestamp": None
        }
        
        # Alert history
        self.alert_history: List[Dict[str, Any]] = []
        self.max_alert_history = 1000
    
    async def initialize(self) -> bool:
        """Initialize the monitoring service"""
        try:
            # Subscribe to all service events for monitoring
            self.subscribe_to_events("service_started", self._handle_service_started)
            self.subscribe_to_events("service_stopped", self._handle_service_stopped)
            self.subscribe_to_events("health_check", self._handle_health_check)
            self.subscribe_to_events("service_error", self._handle_service_error)
            
            # Start monitoring background tasks
            self.add_background_task(self._system_monitoring_task(), "system_monitor")
            self.add_background_task(self._service_monitoring_task(), "service_monitor")
            self.add_background_task(self._alert_processing_task(), "alert_processor")
            self.add_background_task(self._metrics_collection_task(), "metrics_collector")
            
            # Record start time
            self.start_time = datetime.now(UTC)
            
            self.logger.info("Fog monitoring service initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring service: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup monitoring service resources"""
        try:
            # Save alert history if configured
            if self.monitoring_config.get("save_alerts", False):
                await self._save_alert_history()
            
            self.logger.info("Fog monitoring service cleaned up")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up monitoring service: {e}")
            return False
    
    async def health_check(self) -> ServiceHealthCheck:
        """Perform health check on monitoring service"""
        try:
            error_messages = []
            
            # Check system resources
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            if cpu_usage > self.alert_thresholds["cpu_usage_percent"]:
                error_messages.append(f"High CPU usage: {cpu_usage:.1f}%")
            
            if memory.percent > self.alert_thresholds["memory_usage_percent"]:
                error_messages.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > self.alert_thresholds["disk_usage_percent"]:
                error_messages.append(f"High disk usage: {disk.percent:.1f}%")
            
            # Check for recent critical alerts
            recent_critical = sum(1 for alert in self.alert_history[-10:] 
                                if alert.get("severity") == "critical")
            if recent_critical > 3:
                error_messages.append(f"Multiple recent critical alerts: {recent_critical}")
            
            status = ServiceStatus.RUNNING if not error_messages else ServiceStatus.ERROR
            
            return ServiceHealthCheck(
                service_name=self.service_name,
                status=status,
                last_check=datetime.now(UTC),
                error_message="; ".join(error_messages) if error_messages else None,
                metrics=self.metrics.copy()
            )
            
        except Exception as e:
            return ServiceHealthCheck(
                service_name=self.service_name,
                status=ServiceStatus.ERROR,
                last_check=datetime.now(UTC),
                error_message=f"Health check failed: {e}",
                metrics=self.metrics.copy()
            )
    
    async def register_service_for_monitoring(
        self, 
        service_name: str, 
        service_type: str,
        monitoring_config: Optional[Dict[str, Any]] = None
    ):
        """Register a service for monitoring"""
        try:
            self.tracked_services[service_name] = {
                "service_type": service_type,
                "registered_at": datetime.now(UTC),
                "last_health_check": None,
                "health_status": "unknown",
                "error_count": 0,
                "monitoring_config": monitoring_config or {}
            }
            
            self.metrics["services_monitored"] = len(self.tracked_services)
            
            # Publish service registration event
            await self.publish_event("service_registered_for_monitoring", {
                "service_name": service_name,
                "service_type": service_type,
                "timestamp": datetime.now(UTC).isoformat()
            })
            
            self.logger.info(f"Registered service for monitoring: {service_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register service for monitoring: {e}")
    
    async def create_alert(
        self, 
        alert_type: str, 
        severity: str, 
        message: str, 
        source_service: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Create an alert"""
        try:
            alert = {
                "alert_id": f"alert_{datetime.now(UTC).timestamp()}",
                "alert_type": alert_type,
                "severity": severity,  # critical, warning, info
                "message": message,
                "source_service": source_service,
                "timestamp": datetime.now(UTC).isoformat(),
                "additional_data": additional_data or {}
            }
            
            # Add to alert history
            self.alert_history.append(alert)
            
            # Trim history if needed
            if len(self.alert_history) > self.max_alert_history:
                self.alert_history = self.alert_history[-self.max_alert_history:]
            
            # Update metrics
            self.metrics["total_alerts"] += 1
            if severity == "critical":
                self.metrics["critical_alerts"] += 1
            self.metrics["last_alert_timestamp"] = alert["timestamp"]
            
            # Publish alert event
            await self.publish_event("alert_created", alert)
            
            self.logger.warning(f"Alert created: {severity} - {message}")
            
            # Handle critical alerts immediately
            if severity == "critical":
                await self._handle_critical_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to create alert: {e}")
    
    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics"""
        try:
            # Get current system metrics
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Update metrics
            self.metrics.update({
                "system_cpu_usage": cpu_usage,
                "system_memory_usage": memory.percent,
                "system_disk_usage": disk.percent,
                "network_io_mb": (network.bytes_sent + network.bytes_recv) / (1024 * 1024)
            })
            
            if hasattr(self, 'start_time'):
                uptime = (datetime.now(UTC) - self.start_time).total_seconds()
                self.metrics["uptime_seconds"] = uptime
            
            stats = {
                "monitoring_metrics": self.metrics.copy(),
                "tracked_services": {
                    service_name: {
                        "service_type": info["service_type"],
                        "health_status": info["health_status"],
                        "error_count": info["error_count"],
                        "last_check": info["last_health_check"].isoformat() 
                                    if info["last_health_check"] else None
                    }
                    for service_name, info in self.tracked_services.items()
                },
                "recent_alerts": self.alert_history[-10:],  # Last 10 alerts
                "system_resources": {
                    "cpu_percent": cpu_usage,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "network_io_mb": self.metrics["network_io_mb"]
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get monitoring stats: {e}")
            return self.metrics.copy()
    
    async def get_service_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all tracked services"""
        try:
            summary = {
                "healthy_services": 0,
                "unhealthy_services": 0,
                "unknown_services": 0,
                "service_details": []
            }
            
            for service_name, info in self.tracked_services.items():
                health_status = info["health_status"]
                
                if health_status == "running":
                    summary["healthy_services"] += 1
                elif health_status in ["error", "stopped"]:
                    summary["unhealthy_services"] += 1
                else:
                    summary["unknown_services"] += 1
                
                summary["service_details"].append({
                    "service_name": service_name,
                    "service_type": info["service_type"],
                    "health_status": health_status,
                    "error_count": info["error_count"]
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get service health summary: {e}")
            return {"error": str(e)}
    
    async def _handle_service_started(self, event):
        """Handle service started events"""
        service_name = event.source_service
        
        if service_name in self.tracked_services:
            self.tracked_services[service_name]["health_status"] = "running"
            self.tracked_services[service_name]["last_health_check"] = datetime.now(UTC)
        
        self.logger.debug(f"Service started: {service_name}")
    
    async def _handle_service_stopped(self, event):
        """Handle service stopped events"""
        service_name = event.source_service
        
        if service_name in self.tracked_services:
            self.tracked_services[service_name]["health_status"] = "stopped"
        
        # Create alert for unexpected service stops
        if event.data.get("unexpected", False):
            await self.create_alert(
                "service_stopped",
                "warning",
                f"Service {service_name} stopped unexpectedly",
                service_name
            )
        
        self.logger.debug(f"Service stopped: {service_name}")
    
    async def _handle_health_check(self, event):
        """Handle health check events from services"""
        service_name = event.source_service
        health_data = event.data
        
        if service_name in self.tracked_services:
            self.tracked_services[service_name]["last_health_check"] = datetime.now(UTC)
            self.tracked_services[service_name]["health_status"] = health_data.get("status", "unknown")
            
            # Check for health issues
            if health_data.get("status") == "error":
                error_message = health_data.get("error_message", "Unknown error")
                await self.create_alert(
                    "health_check_failed",
                    "warning",
                    f"Health check failed for {service_name}: {error_message}",
                    service_name,
                    health_data
                )
    
    async def _handle_service_error(self, event):
        """Handle service error events"""
        service_name = event.source_service
        error_data = event.data
        
        if service_name in self.tracked_services:
            self.tracked_services[service_name]["error_count"] += 1
        
        # Create alert for errors
        await self.create_alert(
            "service_error",
            "warning",
            f"Error in service {service_name}: {error_data.get('error_message', 'Unknown error')}",
            service_name,
            error_data
        )
    
    async def _handle_critical_alert(self, alert: Dict[str, Any]):
        """Handle critical alerts with immediate action"""
        try:
            # Log critical alert
            self.logger.critical(f"CRITICAL ALERT: {alert['message']}")
            
            # Implement critical alert handling logic
            # This could include:
            # - Sending notifications
            # - Auto-remediation attempts
            # - Service restarts
            # - Circuit breakers
            
            # For now, just publish a critical alert event
            await self.publish_event("critical_alert", alert)
            
        except Exception as e:
            self.logger.error(f"Error handling critical alert: {e}")
    
    async def _system_monitoring_task(self):
        """Background task to monitor system resources"""
        while not self._shutdown_event.is_set():
            try:
                # Monitor CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)
                if cpu_usage > self.alert_thresholds["cpu_usage_percent"]:
                    await self.create_alert(
                        "high_cpu_usage",
                        "warning",
                        f"High CPU usage: {cpu_usage:.1f}%"
                    )
                
                # Monitor memory usage
                memory = psutil.virtual_memory()
                if memory.percent > self.alert_thresholds["memory_usage_percent"]:
                    await self.create_alert(
                        "high_memory_usage",
                        "warning",
                        f"High memory usage: {memory.percent:.1f}%"
                    )
                
                # Monitor disk usage
                disk = psutil.disk_usage('/')
                if disk.percent > self.alert_thresholds["disk_usage_percent"]:
                    await self.create_alert(
                        "high_disk_usage",
                        "critical",
                        f"High disk usage: {disk.percent:.1f}%"
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _service_monitoring_task(self):
        """Background task to monitor registered services"""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.now(UTC)
                
                # Check each tracked service
                for service_name, info in self.tracked_services.items():
                    last_check = info.get("last_health_check")
                    
                    # Check if service hasn't reported health recently
                    if last_check:
                        time_since_check = (current_time - last_check).total_seconds()
                        if time_since_check > 300:  # 5 minutes
                            await self.create_alert(
                                "service_unresponsive",
                                "warning",
                                f"Service {service_name} hasn't reported health in {time_since_check:.0f} seconds",
                                service_name
                            )
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Service monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _alert_processing_task(self):
        """Background task to process and aggregate alerts"""
        while not self._shutdown_event.is_set():
            try:
                # Process alert patterns and trends
                if len(self.alert_history) > 10:
                    recent_alerts = self.alert_history[-10:]
                    
                    # Check for alert storms
                    recent_time = datetime.now(UTC).timestamp() - 300  # Last 5 minutes
                    recent_alert_count = sum(1 for alert in recent_alerts 
                                           if datetime.fromisoformat(alert["timestamp"]).timestamp() > recent_time)
                    
                    if recent_alert_count > 5:
                        await self.create_alert(
                            "alert_storm",
                            "critical",
                            f"Alert storm detected: {recent_alert_count} alerts in 5 minutes"
                        )
                
                await asyncio.sleep(300)  # Process every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(120)
    
    async def _metrics_collection_task(self):
        """Background task to collect and publish metrics"""
        while not self._shutdown_event.is_set():
            try:
                # Collect current metrics
                stats = await self.get_monitoring_stats()
                
                # Publish metrics update
                await self.publish_event("monitoring_metrics_update", {
                    "metrics": stats["monitoring_metrics"],
                    "timestamp": datetime.now(UTC).isoformat()
                })
                
                await asyncio.sleep(60)  # Collect every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _save_alert_history(self):
        """Save alert history to persistent storage"""
        try:
            # Save to file (in production, would use proper database)
            alert_file = f"fog_alerts_{self.node_id}.json"
            with open(alert_file, 'w') as f:
                json.dump(self.alert_history, f, indent=2)
            
            self.logger.info(f"Saved {len(self.alert_history)} alerts to {alert_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save alert history: {e}")