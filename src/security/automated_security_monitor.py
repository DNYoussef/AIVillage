"""
Automated Security Monitoring System
Real-time security validation and continuous monitoring

This system provides:
- Continuous security posture monitoring
- Automated vulnerability scanning
- Real-time threat detection
- Compliance monitoring
- Security metrics collection
- Automated incident response
- Performance impact monitoring
"""

import asyncio
import json
import logging
from datetime import datetime, UTC, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from uuid import uuid4
import statistics

from .unified_security_framework import UnifiedSecurityFramework, UnifiedSecurityEvent, SecurityLevel, ThreatType
from .consolidated_security_config import ConsolidatedSecurityConfigService, ConfigurationCategory
from .mcp_security_coordinator import SecurityAutomationOrchestrator

logger = logging.getLogger(__name__)


class MonitoringLevel(Enum):
    """Security monitoring levels"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    COMPREHENSIVE = "comprehensive"
    FORENSIC = "forensic"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityMetric:
    """Security metric data structure"""
    metric_id: str
    name: str
    category: str
    value: float
    unit: str
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_within_bounds(self) -> bool:
        """Check if metric value is within acceptable bounds"""
        if self.threshold_low is not None and self.value < self.threshold_low:
            return False
        if self.threshold_high is not None and self.value > self.threshold_high:
            return False
        return True
    
    def get_deviation_score(self) -> float:
        """Get deviation score from acceptable bounds (0.0 = within bounds, 1.0+ = outside bounds)"""
        if self.is_within_bounds():
            return 0.0
        
        deviation = 0.0
        if self.threshold_low is not None and self.value < self.threshold_low:
            deviation = abs(self.value - self.threshold_low) / self.threshold_low
        elif self.threshold_high is not None and self.value > self.threshold_high:
            deviation = abs(self.value - self.threshold_high) / self.threshold_high
        
        return deviation


@dataclass
class SecurityAlert:
    """Security alert structure"""
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    description: str = ""
    severity: AlertSeverity = AlertSeverity.MEDIUM
    category: str = "general"
    
    # Alert details
    source_system: str = ""
    affected_resources: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Response
    automated_response: bool = False
    response_actions: List[str] = field(default_factory=list)
    escalated: bool = False
    
    # Metadata
    confidence: float = 1.0
    false_positive_probability: float = 0.0
    related_alerts: List[str] = field(default_factory=list)
    external_references: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category,
            "source_system": self.source_system,
            "affected_resources": self.affected_resources,
            "indicators": self.indicators,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "automated_response": self.automated_response,
            "response_actions": self.response_actions,
            "escalated": self.escalated,
            "confidence": self.confidence,
            "false_positive_probability": self.false_positive_probability,
            "related_alerts": self.related_alerts,
            "external_references": self.external_references
        }


class SecurityMetricsCollector:
    """Collect and analyze security metrics"""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[SecurityMetric]] = {}
        self.metric_definitions = {}
        self.collection_intervals = {}
        
    async def initialize(self):
        """Initialize metrics collector"""
        await self._define_security_metrics()
        await self._setup_collection_schedules()
        logger.info("Security metrics collector initialized")
    
    async def _define_security_metrics(self):
        """Define security metrics to collect"""
        self.metric_definitions = {
            # Authentication metrics
            "auth_success_rate": {
                "name": "Authentication Success Rate",
                "category": "authentication",
                "unit": "percentage",
                "threshold_low": 95.0,
                "threshold_high": None,
                "collection_interval": 60  # seconds
            },
            "auth_failure_rate": {
                "name": "Authentication Failure Rate", 
                "category": "authentication",
                "unit": "percentage",
                "threshold_low": None,
                "threshold_high": 5.0,
                "collection_interval": 60
            },
            "mfa_adoption_rate": {
                "name": "MFA Adoption Rate",
                "category": "authentication",
                "unit": "percentage", 
                "threshold_low": 80.0,
                "threshold_high": None,
                "collection_interval": 3600
            },
            
            # Authorization metrics
            "authorization_denial_rate": {
                "name": "Authorization Denial Rate",
                "category": "authorization",
                "unit": "percentage",
                "threshold_low": None,
                "threshold_high": 10.0,
                "collection_interval": 300
            },
            "privilege_escalation_attempts": {
                "name": "Privilege Escalation Attempts",
                "category": "authorization",
                "unit": "count",
                "threshold_low": None,
                "threshold_high": 5.0,
                "collection_interval": 300
            },
            
            # Threat detection metrics
            "threats_detected_per_hour": {
                "name": "Threats Detected Per Hour",
                "category": "threat_detection", 
                "unit": "count",
                "threshold_low": None,
                "threshold_high": 10.0,
                "collection_interval": 3600
            },
            "false_positive_rate": {
                "name": "False Positive Rate",
                "category": "threat_detection",
                "unit": "percentage",
                "threshold_low": None,
                "threshold_high": 15.0,
                "collection_interval": 3600
            },
            
            # System performance metrics
            "security_processing_latency": {
                "name": "Security Processing Latency",
                "category": "performance",
                "unit": "milliseconds",
                "threshold_low": None,
                "threshold_high": 500.0,
                "collection_interval": 60
            },
            "security_cpu_usage": {
                "name": "Security CPU Usage",
                "category": "performance",
                "unit": "percentage",
                "threshold_low": None,
                "threshold_high": 20.0,
                "collection_interval": 60
            },
            
            # Compliance metrics
            "policy_compliance_rate": {
                "name": "Policy Compliance Rate",
                "category": "compliance",
                "unit": "percentage",
                "threshold_low": 98.0,
                "threshold_high": None,
                "collection_interval": 3600
            },
            "audit_log_completeness": {
                "name": "Audit Log Completeness",
                "category": "compliance",
                "unit": "percentage",
                "threshold_low": 99.5,
                "threshold_high": None,
                "collection_interval": 3600
            }
        }
        
        logger.info(f"Defined {len(self.metric_definitions)} security metrics")
    
    async def _setup_collection_schedules(self):
        """Setup automatic metric collection schedules"""
        for metric_id, definition in self.metric_definitions.items():
            interval = definition["collection_interval"]
            self.collection_intervals[metric_id] = interval
            
            # Start collection task for this metric
            asyncio.create_task(self._collect_metric_periodically(metric_id, interval))
        
        logger.info("Setup metric collection schedules")
    
    async def _collect_metric_periodically(self, metric_id: str, interval: int):
        """Collect a specific metric periodically"""
        while True:
            try:
                await asyncio.sleep(interval)
                metric = await self.collect_metric(metric_id)
                
                if metric:
                    # Store metric
                    if metric_id not in self.metrics_history:
                        self.metrics_history[metric_id] = []
                    
                    self.metrics_history[metric_id].append(metric)
                    
                    # Keep only last 1000 entries per metric
                    if len(self.metrics_history[metric_id]) > 1000:
                        self.metrics_history[metric_id] = self.metrics_history[metric_id][-1000:]
                    
            except Exception as e:
                logger.error(f"Error collecting metric {metric_id}: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def collect_metric(self, metric_id: str) -> Optional[SecurityMetric]:
        """Collect a specific security metric"""
        definition = self.metric_definitions.get(metric_id)
        if not definition:
            return None
        
        # Simulate metric collection - in production, collect from actual systems
        value = await self._simulate_metric_collection(metric_id)
        
        metric = SecurityMetric(
            metric_id=metric_id,
            name=definition["name"],
            category=definition["category"],
            value=value,
            unit=definition["unit"],
            threshold_low=definition.get("threshold_low"),
            threshold_high=definition.get("threshold_high"),
            metadata={"collection_method": "simulated"}
        )
        
        return metric
    
    async def _simulate_metric_collection(self, metric_id: str) -> float:
        """Simulate metric collection for testing"""
        import random
        
        # Simulate realistic metric values
        base_values = {
            "auth_success_rate": 97.5,
            "auth_failure_rate": 2.5,
            "mfa_adoption_rate": 85.0,
            "authorization_denial_rate": 3.0,
            "privilege_escalation_attempts": 1.0,
            "threats_detected_per_hour": 5.0,
            "false_positive_rate": 8.0,
            "security_processing_latency": 150.0,
            "security_cpu_usage": 12.0,
            "policy_compliance_rate": 99.2,
            "audit_log_completeness": 99.8
        }
        
        base_value = base_values.get(metric_id, 50.0)
        # Add some realistic variation
        variation = random.uniform(-0.1, 0.1) * base_value
        return max(0.0, base_value + variation)
    
    async def get_metric_statistics(self, metric_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a metric over the specified time period"""
        if metric_id not in self.metrics_history:
            return {"error": "Metric not found"}
        
        # Filter metrics by time
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history[metric_id]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No data available for the specified time period"}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "metric_id": metric_id,
            "time_period_hours": hours,
            "data_points": len(values),
            "latest_value": values[-1] if values else None,
            "average": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "trend": self._calculate_trend(values),
            "threshold_violations": sum(1 for m in recent_metrics if not m.is_within_bounds()),
            "last_updated": recent_metrics[-1].timestamp.isoformat() if recent_metrics else None
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple trend calculation using first and last quartiles
        n = len(values)
        first_quartile = values[:n//4] if n >= 4 else values[:1]
        last_quartile = values[-n//4:] if n >= 4 else values[-1:]
        
        first_avg = statistics.mean(first_quartile)
        last_avg = statistics.mean(last_quartile)
        
        change_percent = ((last_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"


class VulnerabilityScanner:
    """Automated vulnerability scanning system"""
    
    def __init__(self):
        self.scan_history: List[Dict[str, Any]] = []
        self.vulnerability_database = {}
        self.scan_templates = {}
    
    async def initialize(self):
        """Initialize vulnerability scanner"""
        await self._load_vulnerability_database()
        await self._setup_scan_templates()
        logger.info("Vulnerability scanner initialized")
    
    async def _load_vulnerability_database(self):
        """Load vulnerability database"""
        # Simulate loading vulnerability database
        self.vulnerability_database = {
            "CVE-2023-12345": {
                "severity": "high",
                "description": "SQL injection vulnerability",
                "affected_systems": ["web_applications"],
                "mitigation": "Input validation and parameterized queries"
            },
            "CVE-2023-12346": {
                "severity": "critical",
                "description": "Remote code execution",
                "affected_systems": ["api_servers"],
                "mitigation": "Update to latest version and apply patches"
            },
            "AIVILLAGE-SEC-001": {
                "severity": "medium",
                "description": "Weak password policy",
                "affected_systems": ["authentication"],
                "mitigation": "Enforce stronger password requirements"
            }
        }
        
        logger.info(f"Loaded {len(self.vulnerability_database)} vulnerability signatures")
    
    async def _setup_scan_templates(self):
        """Setup vulnerability scan templates"""
        self.scan_templates = {
            "authentication_scan": {
                "name": "Authentication Security Scan",
                "description": "Scan authentication systems for vulnerabilities",
                "targets": ["authentication_service", "session_management"],
                "checks": [
                    "password_policy_strength",
                    "mfa_enforcement",
                    "session_security",
                    "brute_force_protection"
                ]
            },
            
            "authorization_scan": {
                "name": "Authorization Security Scan",
                "description": "Scan authorization systems for vulnerabilities",
                "targets": ["rbac_system", "permission_management"],
                "checks": [
                    "privilege_escalation_prevention",
                    "default_permissions",
                    "permission_boundaries",
                    "role_separation"
                ]
            },
            
            "data_protection_scan": {
                "name": "Data Protection Security Scan",
                "description": "Scan data protection mechanisms",
                "targets": ["data_storage", "data_transmission"],
                "checks": [
                    "encryption_at_rest",
                    "encryption_in_transit",
                    "data_classification",
                    "pii_protection"
                ]
            },
            
            "network_security_scan": {
                "name": "Network Security Scan", 
                "description": "Scan network security configurations",
                "targets": ["firewalls", "network_policies"],
                "checks": [
                    "default_deny_policy",
                    "network_segmentation",
                    "intrusion_detection",
                    "traffic_monitoring"
                ]
            }
        }
        
        logger.info(f"Setup {len(self.scan_templates)} scan templates")
    
    async def run_vulnerability_scan(self, scan_type: str = "comprehensive") -> Dict[str, Any]:
        """Run vulnerability scan"""
        scan_id = f"scan_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        
        scan_result = {
            "scan_id": scan_id,
            "scan_type": scan_type,
            "started_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "vulnerabilities_found": [],
            "scan_coverage": {},
            "recommendations": [],
            "metadata": {}
        }
        
        try:
            # Determine which templates to use
            templates_to_run = []
            if scan_type == "comprehensive":
                templates_to_run = list(self.scan_templates.keys())
            elif scan_type in self.scan_templates:
                templates_to_run = [scan_type]
            else:
                templates_to_run = ["authentication_scan"]  # Default
            
            # Run each scan template
            for template_name in templates_to_run:
                template = self.scan_templates[template_name]
                template_results = await self._run_scan_template(template)
                
                scan_result["vulnerabilities_found"].extend(template_results["vulnerabilities"])
                scan_result["scan_coverage"][template_name] = template_results["coverage"]
                scan_result["recommendations"].extend(template_results["recommendations"])
            
            # Calculate overall results
            scan_result["status"] = "completed"
            scan_result["completed_at"] = datetime.now(UTC).isoformat()
            scan_result["total_vulnerabilities"] = len(scan_result["vulnerabilities_found"])
            scan_result["severity_breakdown"] = self._calculate_severity_breakdown(scan_result["vulnerabilities_found"])
            
            # Store scan history
            self.scan_history.append(scan_result)
            
            logger.info(f"Vulnerability scan {scan_id} completed: {scan_result['total_vulnerabilities']} vulnerabilities found")
            
        except Exception as e:
            scan_result["status"] = "failed"
            scan_result["error"] = str(e)
            scan_result["completed_at"] = datetime.now(UTC).isoformat()
            logger.error(f"Vulnerability scan {scan_id} failed: {e}")
        
        return scan_result
    
    async def _run_scan_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific scan template"""
        template_results = {
            "vulnerabilities": [],
            "coverage": {"targets_scanned": len(template["targets"]), "checks_performed": len(template["checks"])},
            "recommendations": []
        }
        
        # Simulate vulnerability scanning
        for target in template["targets"]:
            for check in template["checks"]:
                # Simulate finding vulnerabilities (random for testing)
                import random
                if random.random() < 0.2:  # 20% chance of finding a vulnerability
                    vulnerability = {
                        "id": f"VULN_{uuid4().hex[:8]}",
                        "target": target,
                        "check": check,
                        "severity": random.choice(["low", "medium", "high", "critical"]),
                        "description": f"Vulnerability found in {target} during {check}",
                        "cvss_score": round(random.uniform(1.0, 10.0), 1),
                        "remediation": f"Address {check} issue in {target}",
                        "references": []
                    }
                    template_results["vulnerabilities"].append(vulnerability)
        
        # Generate recommendations
        if template_results["vulnerabilities"]:
            template_results["recommendations"].append(f"Address {len(template_results['vulnerabilities'])} vulnerabilities in {template['name']}")
        
        return template_results
    
    def _calculate_severity_breakdown(self, vulnerabilities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate severity breakdown of vulnerabilities"""
        breakdown = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "low")
            if severity in breakdown:
                breakdown[severity] += 1
        
        return breakdown
    
    async def get_scan_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get vulnerability scan history"""
        return self.scan_history[-limit:] if self.scan_history else []


class AutomatedSecurityMonitor:
    """Main automated security monitoring system"""
    
    def __init__(self):
        self.monitoring_level = MonitoringLevel.ENHANCED
        self.security_framework: Optional[UnifiedSecurityFramework] = None
        self.config_service: Optional[ConsolidatedSecurityConfigService] = None
        self.orchestrator: Optional[SecurityAutomationOrchestrator] = None
        
        # Monitoring components
        self.metrics_collector = SecurityMetricsCollector()
        self.vulnerability_scanner = VulnerabilityScanner()
        
        # Alert management
        self.active_alerts: List[SecurityAlert] = []
        self.alert_history: List[SecurityAlert] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.last_health_check = None
        self.health_status = "unknown"
        
        # Configuration
        self.monitoring_config = {
            "alert_threshold_critical": 0.9,
            "alert_threshold_high": 0.7,
            "alert_threshold_medium": 0.5,
            "max_alerts_per_hour": 100,
            "auto_resolve_timeout_hours": 24,
            "escalation_timeout_minutes": 30,
            "health_check_interval_seconds": 300  # 5 minutes
        }
    
    async def initialize(self):
        """Initialize automated security monitor"""
        logger.info("Initializing Automated Security Monitor...")
        
        try:
            # Initialize security framework
            from .unified_security_framework import get_security_framework
            self.security_framework = await get_security_framework()
            
            # Initialize configuration service
            from .consolidated_security_config import get_security_config_service
            self.config_service = await get_security_config_service()
            
            # Initialize orchestrator
            from .mcp_security_coordinator import get_security_orchestrator
            self.orchestrator = await get_security_orchestrator()
            
            # Initialize monitoring components
            await self.metrics_collector.initialize()
            await self.vulnerability_scanner.initialize()
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            self.monitoring_active = True
            self.health_status = "healthy"
            self.last_health_check = datetime.now(UTC)
            
            logger.info("Automated Security Monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Automated Security Monitor: {e}")
            self.health_status = "unhealthy"
            raise
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        # Health monitoring
        asyncio.create_task(self._health_monitor_loop())
        
        # Alert management
        asyncio.create_task(self._alert_management_loop())
        
        # Periodic vulnerability scanning
        asyncio.create_task(self._periodic_vulnerability_scanning())
        
        # Metric analysis
        asyncio.create_task(self._metric_analysis_loop())
        
        logger.info("Started background monitoring tasks")
    
    async def _health_monitor_loop(self):
        """Background task for health monitoring"""
        while self.monitoring_active:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.monitoring_config["health_check_interval_seconds"])
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        health_checks = {
            "security_framework": False,
            "config_service": False,
            "orchestrator": False,
            "metrics_collector": False,
            "vulnerability_scanner": False
        }
        
        try:
            # Check security framework
            if self.security_framework:
                status = await self.security_framework.get_security_status()
                health_checks["security_framework"] = status.get("framework_status") == "operational"
            
            # Check configuration service
            if self.config_service:
                status = await self.config_service.get_service_status()
                health_checks["config_service"] = status.get("service_status") == "operational"
            
            # Check orchestrator
            if self.orchestrator:
                status = await self.orchestrator.get_automation_status()
                health_checks["orchestrator"] = status.get("orchestrator_status") == "operational"
            
            # Check metrics collector
            health_checks["metrics_collector"] = len(self.metrics_collector.metric_definitions) > 0
            
            # Check vulnerability scanner
            health_checks["vulnerability_scanner"] = len(self.vulnerability_scanner.vulnerability_database) > 0
            
            # Calculate overall health
            healthy_components = sum(health_checks.values())
            total_components = len(health_checks)
            health_ratio = healthy_components / total_components
            
            if health_ratio >= 0.8:
                self.health_status = "healthy"
            elif health_ratio >= 0.6:
                self.health_status = "degraded"
            else:
                self.health_status = "unhealthy"
            
            self.last_health_check = datetime.now(UTC)
            
            # Generate alert for unhealthy status
            if self.health_status == "unhealthy":
                await self._generate_alert(
                    title="Security Monitor Health Check Failed",
                    description=f"Health check failed: {healthy_components}/{total_components} components healthy",
                    severity=AlertSeverity.HIGH,
                    category="system_health",
                    indicators=[f"{component}: {status}" for component, status in health_checks.items() if not status]
                )
            
        except Exception as e:
            self.health_status = "error"
            logger.error(f"Health check failed: {e}")
    
    async def _alert_management_loop(self):
        """Background task for alert management"""
        while self.monitoring_active:
            try:
                await self._process_alerts()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in alert management loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_alerts(self):
        """Process and manage active alerts"""
        current_time = datetime.now(UTC)
        
        # Auto-resolve old alerts
        auto_resolve_timeout = timedelta(hours=self.monitoring_config["auto_resolve_timeout_hours"])
        
        for alert in self.active_alerts[:]:  # Copy list to avoid modification during iteration
            if alert.resolved_at is None and current_time - alert.created_at > auto_resolve_timeout:
                alert.resolved_at = current_time
                alert.response_actions.append("auto_resolved_timeout")
                self.active_alerts.remove(alert)
                self.alert_history.append(alert)
                logger.info(f"Auto-resolved alert {alert.alert_id} due to timeout")
            
            # Check for escalation
            escalation_timeout = timedelta(minutes=self.monitoring_config["escalation_timeout_minutes"])
            if (not alert.escalated and 
                alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL] and
                alert.acknowledged_at is None and
                current_time - alert.created_at > escalation_timeout):
                
                await self._escalate_alert(alert)
    
    async def _escalate_alert(self, alert: SecurityAlert):
        """Escalate alert to higher severity or external systems"""
        alert.escalated = True
        alert.response_actions.append("escalated")
        
        # Create GitHub issue for escalated alerts
        if self.orchestrator:
            await self.orchestrator.handle_security_event({
                "event_type": "escalated_alert",
                "title": f"ESCALATED: {alert.title}",
                "severity": alert.severity.value,
                "details": alert.to_dict()
            })
        
        logger.warning(f"Escalated alert {alert.alert_id}: {alert.title}")
    
    async def _periodic_vulnerability_scanning(self):
        """Background task for periodic vulnerability scanning"""
        # Wait for initialization
        await asyncio.sleep(300)  # 5 minutes delay
        
        while self.monitoring_active:
            try:
                # Run vulnerability scan
                scan_result = await self.vulnerability_scanner.run_vulnerability_scan("comprehensive")
                
                # Generate alerts for critical/high vulnerabilities
                for vuln in scan_result.get("vulnerabilities_found", []):
                    if vuln["severity"] in ["critical", "high"]:
                        await self._generate_alert(
                            title=f"Vulnerability Detected: {vuln['id']}",
                            description=vuln["description"],
                            severity=AlertSeverity.CRITICAL if vuln["severity"] == "critical" else AlertSeverity.HIGH,
                            category="vulnerability",
                            indicators=[f"Target: {vuln['target']}", f"CVSS: {vuln['cvss_score']}"],
                            affected_resources=[vuln["target"]]
                        )
                
                # Wait 24 hours before next scan
                await asyncio.sleep(24 * 3600)
                
            except Exception as e:
                logger.error(f"Error in periodic vulnerability scanning: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def _metric_analysis_loop(self):
        """Background task for metric analysis"""
        while self.monitoring_active:
            try:
                await self._analyze_metrics()
                await asyncio.sleep(300)  # Analyze every 5 minutes
            except Exception as e:
                logger.error(f"Error in metric analysis loop: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_metrics(self):
        """Analyze security metrics and generate alerts"""
        for metric_id in self.metrics_collector.metric_definitions.keys():
            try:
                stats = await self.metrics_collector.get_metric_statistics(metric_id, hours=1)
                
                if "error" not in stats:
                    # Check for threshold violations
                    latest_value = stats.get("latest_value")
                    if latest_value is not None:
                        definition = self.metrics_collector.metric_definitions[metric_id]
                        threshold_high = definition.get("threshold_high")
                        threshold_low = definition.get("threshold_low")
                        
                        violation = False
                        violation_type = ""
                        
                        if threshold_high is not None and latest_value > threshold_high:
                            violation = True
                            violation_type = "high_threshold"
                        elif threshold_low is not None and latest_value < threshold_low:
                            violation = True
                            violation_type = "low_threshold"
                        
                        if violation:
                            # Calculate severity based on deviation
                            if threshold_high:
                                deviation = abs(latest_value - threshold_high) / threshold_high
                            else:
                                deviation = abs(latest_value - threshold_low) / threshold_low
                            
                            severity = AlertSeverity.MEDIUM
                            if deviation > 0.5:
                                severity = AlertSeverity.HIGH
                            elif deviation > 1.0:
                                severity = AlertSeverity.CRITICAL
                            
                            await self._generate_alert(
                                title=f"Metric Threshold Violation: {definition['name']}",
                                description=f"{definition['name']} is {latest_value} {definition['unit']}, violating {violation_type}",
                                severity=severity,
                                category="metric_violation",
                                indicators=[f"Current: {latest_value}", f"Threshold: {threshold_high or threshold_low}", f"Deviation: {deviation:.2f}"],
                                affected_resources=[f"metric:{metric_id}"]
                            )
                    
            except Exception as e:
                logger.error(f"Error analyzing metric {metric_id}: {e}")
    
    async def _generate_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        category: str,
        indicators: List[str] = None,
        affected_resources: List[str] = None,
        confidence: float = 1.0
    ) -> SecurityAlert:
        """Generate security alert"""
        alert = SecurityAlert(
            title=title,
            description=description,
            severity=severity,
            category=category,
            source_system="automated_monitor",
            indicators=indicators or [],
            affected_resources=affected_resources or [],
            confidence=confidence
        )
        
        # Check for duplicate alerts
        similar_alerts = [
            a for a in self.active_alerts
            if a.title == title and a.category == category and 
            (datetime.now(UTC) - a.created_at).total_seconds() < 3600  # Within last hour
        ]
        
        if not similar_alerts:
            self.active_alerts.append(alert)
            
            # Apply automated response
            await self._apply_automated_response(alert)
            
            logger.warning(f"Generated security alert {alert.alert_id}: {title}")
        
        return alert
    
    async def _apply_automated_response(self, alert: SecurityAlert):
        """Apply automated response to security alert"""
        response_actions = []
        
        # Rate limiting for DoS-related alerts
        if "dos" in alert.category.lower() or "abuse" in alert.description.lower():
            response_actions.append("rate_limiting_applied")
            alert.automated_response = True
        
        # Account restrictions for authentication issues
        if alert.category == "authentication" and alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            response_actions.append("account_restrictions_applied")
            alert.automated_response = True
        
        # Network isolation for intrusion attempts
        if "intrusion" in alert.description.lower() or "unauthorized" in alert.description.lower():
            response_actions.append("network_isolation_activated")
            alert.automated_response = True
        
        # Increase monitoring for vulnerability alerts
        if alert.category == "vulnerability":
            response_actions.append("monitoring_increased")
            alert.automated_response = True
        
        alert.response_actions.extend(response_actions)
        
        if response_actions:
            logger.info(f"Applied automated response to alert {alert.alert_id}: {response_actions}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged_at = datetime.now(UTC)
                alert.response_actions.append(f"acknowledged_by_{acknowledged_by}")
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        
        return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system", resolution_notes: str = "") -> bool:
        """Resolve an active alert"""
        for alert in self.active_alerts[:]:
            if alert.alert_id == alert_id:
                alert.resolved_at = datetime.now(UTC)
                alert.response_actions.append(f"resolved_by_{resolved_by}")
                if resolution_notes:
                    alert.response_actions.append(f"resolution:{resolution_notes}")
                
                self.active_alerts.remove(alert)
                self.alert_history.append(alert)
                
                logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
        
        return False
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        return {
            "monitoring_active": self.monitoring_active,
            "monitoring_level": self.monitoring_level.value,
            "health_status": self.health_status,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "active_alerts": len(self.active_alerts),
            "alerts_by_severity": {
                severity.value: len([a for a in self.active_alerts if a.severity == severity])
                for severity in AlertSeverity
            },
            "alert_history_count": len(self.alert_history),
            "metrics_tracked": len(self.metrics_collector.metric_definitions),
            "vulnerability_database_size": len(self.vulnerability_scanner.vulnerability_database),
            "scan_history_count": len(self.vulnerability_scanner.scan_history),
            "configuration": self.monitoring_config,
            "last_updated": datetime.now(UTC).isoformat()
        }
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data"""
        # Get recent metric statistics
        metric_stats = {}
        for metric_id in list(self.metrics_collector.metric_definitions.keys())[:10]:  # Top 10 metrics
            stats = await self.metrics_collector.get_metric_statistics(metric_id, hours=24)
            if "error" not in stats:
                metric_stats[metric_id] = {
                    "latest_value": stats["latest_value"],
                    "average": stats["average"],
                    "trend": stats["trend"],
                    "threshold_violations": stats["threshold_violations"]
                }
        
        # Get recent vulnerability scan
        recent_scans = await self.vulnerability_scanner.get_scan_history(limit=1)
        latest_scan = recent_scans[0] if recent_scans else None
        
        return {
            "dashboard_timestamp": datetime.now(UTC).isoformat(),
            "security_posture": self.health_status,
            "active_alerts": [alert.to_dict() for alert in self.active_alerts[-10:]],  # Latest 10 alerts
            "metric_highlights": metric_stats,
            "vulnerability_summary": {
                "latest_scan_id": latest_scan["scan_id"] if latest_scan else None,
                "total_vulnerabilities": latest_scan["total_vulnerabilities"] if latest_scan else 0,
                "severity_breakdown": latest_scan["severity_breakdown"] if latest_scan else {},
                "last_scan_date": latest_scan["completed_at"] if latest_scan else None
            },
            "system_health": {
                "monitoring_active": self.monitoring_active,
                "components_healthy": self.health_status == "healthy",
                "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
            }
        }


# Global monitor instance
_security_monitor: Optional[AutomatedSecurityMonitor] = None


async def get_security_monitor() -> AutomatedSecurityMonitor:
    """Get global automated security monitor instance"""
    global _security_monitor
    
    if _security_monitor is None:
        _security_monitor = AutomatedSecurityMonitor()
        await _security_monitor.initialize()
    
    return _security_monitor


# Convenience functions
async def get_security_dashboard() -> Dict[str, Any]:
    """Get security dashboard data"""
    monitor = await get_security_monitor()
    return await monitor.get_security_dashboard()


async def run_vulnerability_scan(scan_type: str = "comprehensive") -> Dict[str, Any]:
    """Run vulnerability scan"""
    monitor = await get_security_monitor()
    return await monitor.vulnerability_scanner.run_vulnerability_scan(scan_type)


async def get_security_metrics(metric_id: str, hours: int = 24) -> Dict[str, Any]:
    """Get security metric statistics"""
    monitor = await get_security_monitor()
    return await monitor.metrics_collector.get_metric_statistics(metric_id, hours)


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize monitor
        monitor = await get_security_monitor()
        
        # Get monitoring status
        status = await monitor.get_monitoring_status()
        print(f"Monitoring status: {json.dumps(status, indent=2)}")
        
        # Get security dashboard
        dashboard = await get_security_dashboard()
        print(f"Security dashboard: {json.dumps(dashboard, indent=2)}")
        
        # Run vulnerability scan
        scan_result = await run_vulnerability_scan("authentication_scan")
        print(f"Vulnerability scan: {json.dumps(scan_result, indent=2)}")
        
        # Get metrics
        auth_metrics = await get_security_metrics("auth_success_rate", hours=1)
        print(f"Auth metrics: {json.dumps(auth_metrics, indent=2)}")
    
    asyncio.run(main())