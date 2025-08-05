#!/usr/bin/env python3
"""AIVillage Security Monitoring System
Real-time security event detection and alerting.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
import logging
import os
import re
import time
from typing import Any

# Optional dependencies for enhanced monitoring
try:
    import sentry_sdk

    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("aivillage.security")


@dataclass
class SecurityEvent:
    """Security event data structure."""

    timestamp: datetime
    event_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    user_id: str
    source_ip: str
    details: dict[str, Any]
    threat_score: float = 0.0
    mitigated: bool = False


@dataclass
class ThreatIntelligence:
    """Threat intelligence data."""

    ip_address: str
    threat_type: str
    confidence: float
    last_seen: datetime
    source: str


class SecurityMetrics:
    """Security metrics collection."""

    def __init__(self) -> None:
        if PROMETHEUS_AVAILABLE:
            self.auth_failures = Counter(
                "auth_failures_total",
                "Authentication failures",
                ["user_id", "source_ip"],
            )
            self.security_events = Counter("security_events_total", "Security events", ["event_type", "severity"])
            self.request_rate = Histogram("request_duration_seconds", "Request duration")
            self.threat_score = Gauge("threat_score_current", "Current threat score", ["user_id"])

            # Start Prometheus metrics server
            start_http_server(8090)
            logger.info("Prometheus metrics server started on port 8090")

    def record_auth_failure(self, user_id: str, source_ip: str) -> None:
        """Record authentication failure."""
        if PROMETHEUS_AVAILABLE:
            self.auth_failures.labels(user_id=user_id, source_ip=source_ip).inc()

    def record_security_event(self, event_type: str, severity: str) -> None:
        """Record security event."""
        if PROMETHEUS_AVAILABLE:
            self.security_events.labels(event_type=event_type, severity=severity).inc()

    def update_threat_score(self, user_id: str, score: float) -> None:
        """Update threat score for user."""
        if PROMETHEUS_AVAILABLE:
            self.threat_score.labels(user_id=user_id).set(score)


class ThreatDetector:
    """Advanced threat detection algorithms."""

    def __init__(self) -> None:
        self.failed_attempts = defaultdict(deque)
        self.request_patterns = defaultdict(deque)
        self.sql_injection_patterns = [
            r"union\s+select",
            r"or\s+1\s*=\s*1",
            r"drop\s+table",
            r"exec\s*\(",
            r"script\s*>",
            r"javascript:",
            r"<\s*iframe",
        ]
        self.threat_ips = set()
        self.baseline_metrics = {}

    def detect_brute_force(self, user_id: str, source_ip: str) -> float:
        """Detect brute force attacks."""
        now = time.time()
        window = 300  # 5 minutes

        # Clean old attempts
        key = f"{user_id}:{source_ip}"
        while self.failed_attempts[key] and self.failed_attempts[key][0] < now - window:
            self.failed_attempts[key].popleft()

        # Add current attempt
        self.failed_attempts[key].append(now)

        # Calculate threat score
        attempt_count = len(self.failed_attempts[key])
        if attempt_count > 10:
            return 1.0  # Maximum threat
        if attempt_count > 5:
            return 0.7  # High threat
        if attempt_count > 3:
            return 0.4  # Medium threat

        return 0.1  # Low threat

    def detect_sql_injection(self, input_data: str) -> float:
        """Detect SQL injection attempts."""
        if not input_data:
            return 0.0

        input_lower = input_data.lower()
        threat_score = 0.0

        for pattern in self.sql_injection_patterns:
            if re.search(pattern, input_lower, re.IGNORECASE):
                threat_score += 0.3

        # Check for suspicious characters
        suspicious_chars = ["'", '"', ";", "--", "/*", "*/"]
        for char in suspicious_chars:
            if char in input_data:
                threat_score += 0.1

        return min(threat_score, 1.0)

    def detect_rate_limit_violation(self, user_id: str, endpoint: str) -> float:
        """Detect rate limiting violations."""
        now = time.time()
        window = 60  # 1 minute

        key = f"{user_id}:{endpoint}"

        # Clean old requests
        while self.request_patterns[key] and self.request_patterns[key][0] < now - window:
            self.request_patterns[key].popleft()

        # Add current request
        self.request_patterns[key].append(now)

        # Calculate threat score based on request rate
        request_count = len(self.request_patterns[key])
        normal_rate = 30  # requests per minute

        if request_count > normal_rate * 3:
            return 1.0  # DDoS-like behavior
        if request_count > normal_rate * 2:
            return 0.7  # High rate
        if request_count > normal_rate:
            return 0.4  # Elevated rate

        return 0.0

    def detect_anomalous_behavior(self, user_id: str, behavior_data: dict[str, Any]) -> float:
        """Detect anomalous user behavior."""
        threat_score = 0.0

        # Check for unusual access patterns
        if behavior_data.get("access_time"):
            hour = behavior_data["access_time"].hour
            if hour < 6 or hour > 22:  # Outside business hours
                threat_score += 0.2

        # Check for unusual operations
        if behavior_data.get("operation") in ["admin", "delete_all", "export_data"]:
            threat_score += 0.3

        # Check for multiple failed operations
        if behavior_data.get("failed_operations", 0) > 5:
            threat_score += 0.4

        return min(threat_score, 1.0)


class SecurityAlertManager:
    """Security alert and notification management."""

    def __init__(self) -> None:
        self.alert_thresholds = {
            "CRITICAL": 0.9,
            "HIGH": 0.7,
            "MEDIUM": 0.5,
            "LOW": 0.3,
        }

        self.notification_channels = []
        self.recent_alerts = deque(maxlen=1000)

        # Initialize Sentry if available
        if SENTRY_AVAILABLE and os.environ.get("SENTRY_DSN"):
            sentry_sdk.init(dsn=os.environ.get("SENTRY_DSN"))
            logger.info("Sentry integration enabled")

    async def send_alert(self, event: SecurityEvent) -> None:
        """Send security alert through configured channels."""
        alert_data = {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "severity": event.severity,
            "user_id": event.user_id,
            "source_ip": event.source_ip,
            "details": event.details,
            "threat_score": event.threat_score,
        }

        # Log alert
        logger.critical(f"SECURITY ALERT: {event.event_type} - {event.severity} - User: {event.user_id}")

        # Send to Sentry
        if SENTRY_AVAILABLE:
            sentry_sdk.capture_message(f"Security Alert: {event.event_type}", level="error", extra=alert_data)

        # Store alert
        self.recent_alerts.append(alert_data)

        # Send webhook notification
        webhook_url = os.environ.get("SECURITY_ALERT_WEBHOOK")
        if webhook_url:
            await self._send_webhook_alert(webhook_url, alert_data)

    async def _send_webhook_alert(self, webhook_url: str, alert_data: dict[str, Any]) -> None:
        """Send alert via webhook."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=alert_data) as response:
                    if response.status == 200:
                        logger.info("Webhook alert sent successfully")
                    else:
                        logger.error(f"Webhook alert failed: {response.status}")
        except Exception as e:
            logger.exception(f"Failed to send webhook alert: {e}")

    def get_recent_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent security alerts."""
        return list(self.recent_alerts)[-limit:]


class SecurityMonitor:
    """Main security monitoring system."""

    def __init__(self) -> None:
        self.metrics = SecurityMetrics()
        self.detector = ThreatDetector()
        self.alert_manager = SecurityAlertManager()
        self.event_queue = asyncio.Queue()
        self.running = False

        # Load threat intelligence
        self.threat_intel = self._load_threat_intelligence()

        logger.info("Security monitor initialized")

    def _load_threat_intelligence(self) -> list[ThreatIntelligence]:
        """Load threat intelligence data."""
        # In production, this would load from external threat feeds
        return []

    async def start(self) -> None:
        """Start security monitoring."""
        self.running = True
        logger.info("Starting security monitoring...")

        # Start event processing
        await asyncio.gather(
            self._process_events(),
            self._periodic_analysis(),
            self._threat_intel_update(),
        )

    async def stop(self) -> None:
        """Stop security monitoring."""
        self.running = False
        logger.info("Security monitoring stopped")

    async def log_security_event(self, event_type: str, user_id: str, source_ip: str, details: dict[str, Any]) -> None:
        """Log security event for analysis."""
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity="LOW",  # Will be updated by analysis
            user_id=user_id,
            source_ip=source_ip,
            details=details,
        )

        await self.event_queue.put(event)

    async def _process_events(self) -> None:
        """Process security events."""
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._analyze_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"Error processing security event: {e}")

    async def _analyze_event(self, event: SecurityEvent) -> None:
        """Analyze security event for threats."""
        threat_score = 0.0

        # Analyze based on event type
        if event.event_type == "auth_failure":
            threat_score = self.detector.detect_brute_force(event.user_id, event.source_ip)
            self.metrics.record_auth_failure(event.user_id, event.source_ip)

        elif event.event_type == "sql_injection_attempt":
            input_data = event.details.get("input_data", "")
            threat_score = self.detector.detect_sql_injection(input_data)

        elif event.event_type == "rate_limit_violation":
            endpoint = event.details.get("endpoint", "")
            threat_score = self.detector.detect_rate_limit_violation(event.user_id, endpoint)

        elif event.event_type == "anomalous_behavior":
            threat_score = self.detector.detect_anomalous_behavior(event.user_id, event.details)

        # Update event with threat score
        event.threat_score = threat_score

        # Determine severity
        if threat_score >= 0.9:
            event.severity = "CRITICAL"
        elif threat_score >= 0.7:
            event.severity = "HIGH"
        elif threat_score >= 0.5:
            event.severity = "MEDIUM"
        else:
            event.severity = "LOW"

        # Record metrics
        self.metrics.record_security_event(event.event_type, event.severity)
        self.metrics.update_threat_score(event.user_id, threat_score)

        # Send alert if necessary
        if threat_score >= 0.5:  # Medium threat or higher
            await self.alert_manager.send_alert(event)

        # Log event
        logger.info(f"Security event analyzed: {event.event_type} - Score: {threat_score:.2f}")

    async def _periodic_analysis(self) -> None:
        """Perform periodic security analysis."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Analyze trends
                await self._analyze_trends()

                # Update threat intelligence
                await self._update_threat_scores()

                logger.debug("Periodic security analysis completed")

            except Exception as e:
                logger.exception(f"Error in periodic analysis: {e}")

    async def _analyze_trends(self) -> None:
        """Analyze security trends."""
        # This would analyze patterns over time
        # For now, just log status
        recent_alerts = self.alert_manager.get_recent_alerts(10)
        if recent_alerts:
            logger.info(f"Recent security activity: {len(recent_alerts)} alerts in last period")

    async def _update_threat_scores(self) -> None:
        """Update threat scores for all users."""
        # This would update baseline threat scores

    async def _threat_intel_update(self) -> None:
        """Update threat intelligence data."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Every hour

                # In production, this would fetch from threat feeds
                logger.debug("Threat intelligence updated")

            except Exception as e:
                logger.exception(f"Error updating threat intelligence: {e}")

    def get_security_status(self) -> dict[str, Any]:
        """Get current security status."""
        recent_alerts = self.alert_manager.get_recent_alerts(10)

        return {
            "status": "monitoring",
            "uptime_seconds": time.time(),
            "recent_alerts_count": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a["severity"] == "CRITICAL"]),
            "high_alerts": len([a for a in recent_alerts if a["severity"] == "HIGH"]),
            "threat_intel_entries": len(self.threat_intel),
            "monitoring_active": self.running,
        }


# Example integration with MCP server


class MCPSecurityIntegration:
    """Integration with MCP server for security monitoring."""

    def __init__(self, security_monitor: SecurityMonitor) -> None:
        self.monitor = security_monitor

    async def log_authentication_attempt(
        self, user_id: str, source_ip: str, success: bool, details: dict[str, Any]
    ) -> None:
        """Log authentication attempt."""
        if not success:
            await self.monitor.log_security_event("auth_failure", user_id, source_ip, details)

    async def log_database_query(self, user_id: str, query: str, source_ip: str, details: dict[str, Any]) -> None:
        """Log database query for SQL injection detection."""
        await self.monitor.log_security_event("sql_injection_attempt", user_id, source_ip, {"query": query, **details})

    async def log_api_request(self, user_id: str, endpoint: str, source_ip: str, details: dict[str, Any]) -> None:
        """Log API request for rate limiting."""
        await self.monitor.log_security_event(
            "rate_limit_violation",
            user_id,
            source_ip,
            {"endpoint": endpoint, **details},
        )


async def main() -> None:
    """Main function to run security monitor."""
    monitor = SecurityMonitor()

    try:
        logger.info("Starting AIVillage Security Monitor...")
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())
