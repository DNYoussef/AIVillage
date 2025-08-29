# Security Monitoring Guide

## Introduction

The AIVillage Security Monitoring system provides comprehensive real-time threat detection, behavioral analysis, and automated incident response capabilities. This guide covers the implementation, configuration, and operational aspects of the security monitoring framework.

## Security Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Security Monitoring Platform                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │ Threat          │    │ Security Event  │    │ Alert       │  │
│  │ Detection       │    │ Processing      │    │ Management  │  │
│  │                 │    │                 │    │             │  │
│  │ • ML Analysis   │    │ • Event Queue   │    │ • Multi-    │  │
│  │ • Pattern Match │    │ • Correlation   │    │   channel   │  │
│  │ • Behavior      │    │ • Analysis      │    │ • Escalation│  │
│  │   Tracking      │    │ • Enrichment    │    │ • Response  │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
│           │                       │                     │        │
│           └───────────────────────┼─────────────────────┘        │
│                                   │                              │
│  ┌─────────────────────────────────┼─────────────────────────────┐  │
│  │                   Security Intelligence                      │  │
│  │                                                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────┐  │
│  │  │ Threat      │  │ User        │  │ Network     │  │ API │  │
│  │  │ Intel       │  │ Behavior    │  │ Traffic     │  │ Mon │  │
│  │  │ Feeds       │  │ Analysis    │  │ Analysis    │  │     │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────┘  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Security Event Flow

```
Security Event → Ingestion → Analysis → Threat Scoring → Alert Generation
      ↓             ↓          ↓           ↓               ↓
 Authentication  Event Queue  Pattern   Risk Assessment  Multi-channel
 API Requests    Buffering   Matching   ML Analysis      Dispatch
 User Actions    Filtering   Correlation Behavior Score  Response
```

## SecurityMonitor Implementation

### Core Class Structure

**Location**: `packages/monitoring/security_monitor.py:270-476`

```python
class SecurityMonitor:
    """Main security monitoring system with real-time threat detection."""
    
    def __init__(self):
        self.metrics = SecurityMetrics()           # Prometheus integration
        self.detector = ThreatDetector()           # ML-based threat detection
        self.alert_manager = SecurityAlertManager() # Multi-channel alerts
        self.event_queue = asyncio.Queue()        # Async event processing
        self.threat_intel = []                     # Threat intelligence feeds
        
    async def start(self):
        """Start security monitoring with concurrent processing."""
        await asyncio.gather(
            self._process_events(),        # Event processing loop
            self._periodic_analysis(),     # Trend analysis
            self._threat_intel_update(),   # Intelligence updates
        )
```

### Security Event Structure

```python
@dataclass
class SecurityEvent:
    """Security event data structure with threat scoring."""
    
    timestamp: datetime
    event_type: str          # auth_failure, sql_injection, rate_limit, anomalous_behavior
    severity: str            # CRITICAL, HIGH, MEDIUM, LOW
    user_id: str
    source_ip: str
    details: Dict[str, Any]  # Event-specific details
    threat_score: float = 0.0  # 0.0-1.0 threat probability
    mitigated: bool = False    # Mitigation status
```

### Event Types and Detection

#### 1. Authentication Security

**Brute Force Attack Detection** (`packages/monitoring/security_monitor.py:113-135`)

```python
def detect_brute_force(self, user_id: str, source_ip: str) -> float:
    """Detect brute force attacks using sliding window analysis."""
    now = time.time()
    window = 300  # 5-minute sliding window
    
    # Clean old attempts outside window
    key = f"{user_id}:{source_ip}"
    while (self.failed_attempts[key] and 
           self.failed_attempts[key][0] < now - window):
        self.failed_attempts[key].popleft()
    
    # Add current failed attempt
    self.failed_attempts[key].append(now)
    
    # Calculate threat score based on attempt frequency
    attempt_count = len(self.failed_attempts[key])
    if attempt_count > 10:
        return 1.0  # Maximum threat - potential brute force
    elif attempt_count > 5:
        return 0.7  # High threat - suspicious activity
    elif attempt_count > 3:
        return 0.4  # Medium threat - elevated risk
    
    return 0.1  # Low threat - normal failed login
```

**Usage Example**:
```python
# Log authentication failure
await security_monitor.log_security_event(
    event_type="auth_failure",
    user_id="user123",
    source_ip="192.168.1.100", 
    details={
        "reason": "invalid_password",
        "attempt_count": 3,
        "user_agent": "Mozilla/5.0...",
        "endpoint": "/api/login"
    }
)

# Threat score automatically calculated and alerts sent if > 0.5
```

**Threat Score Thresholds**:
- **1.0 (Critical)**: >10 attempts in 5 minutes - likely automated attack
- **0.7 (High)**: 6-10 attempts - suspicious activity requiring investigation  
- **0.4 (Medium)**: 4-5 attempts - elevated risk, monitor closely
- **0.1 (Low)**: 1-3 attempts - normal authentication failures

#### 2. SQL Injection Detection

**Pattern-Based Detection** (`packages/monitoring/security_monitor.py:137-155`)

```python
def detect_sql_injection(self, input_data: str) -> float:
    """Detect SQL injection attempts using pattern matching."""
    if not input_data:
        return 0.0
    
    input_lower = input_data.lower()
    threat_score = 0.0
    
    # Check for common SQL injection patterns
    sql_patterns = [
        r"union\s+select",      # Union-based injection
        r"or\s+1\s*=\s*1",     # Boolean-based injection
        r"drop\s+table",        # Destructive operations
        r"exec\s*\(",          # Code execution
        r"script\s*>",         # XSS attempts
        r"javascript:",        # JavaScript injection
        r"<\s*iframe",         # Frame injection
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, input_lower, re.IGNORECASE):
            threat_score += 0.3  # Each pattern adds 30% threat
    
    # Check for suspicious characters
    suspicious_chars = ["'", '"', ";", "--", "/*", "*/"]
    for char in suspicious_chars:
        if char in input_data:
            threat_score += 0.1  # Each char adds 10% threat
    
    return min(threat_score, 1.0)  # Cap at maximum threat
```

**Usage Example**:
```python
# Monitor database queries for injection attempts
async def log_database_query(user_id: str, query: str, source_ip: str):
    await security_monitor.log_security_event(
        "sql_injection_attempt",
        user_id,
        source_ip,
        {
            "query": query,
            "query_length": len(query),
            "endpoint": "/api/search",
            "timestamp": time.time()
        }
    )

# Example suspicious queries that trigger alerts:
# "' OR 1=1 --"                    -> Threat Score: 0.5
# "'; DROP TABLE users; --"        -> Threat Score: 0.8  
# "UNION SELECT * FROM passwords"  -> Threat Score: 0.6
```

#### 3. Rate Limiting Violations

**Request Rate Analysis** (`packages/monitoring/security_monitor.py:157-182`)

```python
def detect_rate_limit_violation(self, user_id: str, endpoint: str) -> float:
    """Detect rate limiting violations and potential DDoS."""
    now = time.time()
    window = 60  # 1-minute sliding window
    
    key = f"{user_id}:{endpoint}"
    
    # Clean old requests outside window
    while (self.request_patterns[key] and 
           self.request_patterns[key][0] < now - window):
        self.request_patterns[key].popleft()
    
    # Add current request
    self.request_patterns[key].append(now)
    
    # Calculate threat score based on request rate
    request_count = len(self.request_patterns[key])
    normal_rate = 30  # requests per minute baseline
    
    if request_count > normal_rate * 3:
        return 1.0  # DDoS-like behavior - 90+ requests/minute
    elif request_count > normal_rate * 2:
        return 0.7  # High rate - 60+ requests/minute
    elif request_count > normal_rate:
        return 0.4  # Elevated rate - 30+ requests/minute
    
    return 0.0  # Normal request rate
```

**Configuration Example**:
```python
# Per-endpoint rate limits
RATE_LIMITS = {
    "/api/login": 5,      # 5 requests/minute
    "/api/search": 60,    # 60 requests/minute  
    "/api/upload": 10,    # 10 requests/minute
    "/api/download": 20,  # 20 requests/minute
}

# Monitor API requests
async def track_api_request(user_id: str, endpoint: str, source_ip: str):
    await security_monitor.log_security_event(
        "rate_limit_violation",
        user_id,
        source_ip,
        {
            "endpoint": endpoint,
            "rate_limit": RATE_LIMITS.get(endpoint, 30),
            "user_agent": request.headers.get("User-Agent"),
            "method": request.method
        }
    )
```

#### 4. Behavioral Anomaly Detection

**User Behavior Analysis** (`packages/monitoring/security_monitor.py:184-202`)

```python
def detect_anomalous_behavior(self, user_id: str, behavior_data: Dict[str, Any]) -> float:
    """Detect anomalous user behavior patterns."""
    threat_score = 0.0
    
    # Time-based anomaly detection
    if behavior_data.get("access_time"):
        hour = behavior_data["access_time"].hour
        if hour < 6 or hour > 22:  # Outside business hours
            threat_score += 0.2
    
    # Operation risk scoring
    high_risk_operations = ["admin", "delete_all", "export_data", "user_management"]
    if behavior_data.get("operation") in high_risk_operations:
        threat_score += 0.3
    
    # Failed operation pattern analysis
    if behavior_data.get("failed_operations", 0) > 5:
        threat_score += 0.4
    
    # Geographic anomaly (if location data available)
    if behavior_data.get("location_anomaly"):
        threat_score += 0.3
    
    # Device fingerprint changes
    if behavior_data.get("device_change"):
        threat_score += 0.2
    
    return min(threat_score, 1.0)
```

**Behavioral Metrics Tracking**:
```python
# Track user behavior patterns
user_behavior = {
    "access_time": datetime.now(),
    "operation": "delete_user",
    "failed_operations": 3,
    "location_anomaly": True,  # Login from unusual location
    "device_change": False,    # Same device fingerprint
    "privilege_escalation": False,
    "data_access_volume": 1500  # MB accessed
}

await security_monitor.log_security_event(
    "anomalous_behavior",
    user_id,
    source_ip,
    user_behavior
)
```

## Security Metrics and Monitoring

### Prometheus Integration

**SecurityMetrics Class** (`packages/monitoring/security_monitor.py:61-93`)

```python
class SecurityMetrics:
    """Security metrics collection with Prometheus integration."""
    
    def __init__(self):
        if PROMETHEUS_AVAILABLE:
            # Authentication metrics
            self.auth_failures = Counter(
                "auth_failures_total",
                "Total authentication failures",
                ["user_id", "source_ip", "reason"]
            )
            
            # Security event metrics
            self.security_events = Counter(
                "security_events_total", 
                "Total security events",
                ["event_type", "severity"]
            )
            
            # Threat score distribution
            self.threat_score = Gauge(
                "threat_score_current",
                "Current threat score for users",
                ["user_id"]
            )
            
            # Response time metrics
            self.detection_latency = Histogram(
                "security_detection_duration_seconds",
                "Time taken for threat detection"
            )
            
            # Start metrics server
            start_http_server(8090)
```

### Key Security Metrics

**Authentication Metrics**
- `auth_failures_total`: Total failed authentication attempts by user/IP
- `auth_success_rate`: Authentication success rate over time
- `password_reset_requests`: Password reset request frequency
- `account_lockouts`: Number of accounts locked due to repeated failures

**Threat Detection Metrics**
- `threat_score_current`: Real-time threat scores for active users
- `security_events_total`: Total security events by type and severity
- `detection_latency`: Time taken for threat analysis (<50ms target)
- `false_positive_rate`: False positive rate for tuning thresholds

**Incident Response Metrics**
- `alert_response_time`: Time from detection to alert dispatch
- `incident_resolution_time`: Time to resolve security incidents
- `mitigation_success_rate`: Success rate of automated mitigations
- `escalation_rate`: Percentage of alerts requiring human intervention

### Dashboard Visualization

**Security Dashboard Components**:

```python
# Security overview dashboard
def create_security_dashboard():
    """Create comprehensive security monitoring dashboard."""
    
    # Real-time threat landscape
    threat_map = create_threat_heatmap(
        data=recent_threats,
        metrics=["source_ip", "threat_score", "event_type"]
    )
    
    # Authentication trends
    auth_trends = create_time_series(
        metrics=["auth_failures", "auth_success_rate"],
        time_range="24h",
        aggregation="5m"
    )
    
    # Top threats and attackers
    top_threats = create_ranked_list(
        data=threat_analysis,
        rank_by="threat_score",
        limit=10
    )
    
    # Incident timeline
    incident_timeline = create_timeline(
        events=security_incidents,
        time_range="7d"
    )
    
    return SecurityDashboard(
        threat_map=threat_map,
        auth_trends=auth_trends,
        top_threats=top_threats,
        incident_timeline=incident_timeline
    )
```

## Alert Management and Response

### Security Alert Configuration

**Alert Thresholds** (`packages/monitoring/security_monitor.py:208-244`)

```python
class SecurityAlertManager:
    """Security alert and notification management."""
    
    def __init__(self):
        self.alert_thresholds = {
            "CRITICAL": 0.9,  # Immediate response required
            "HIGH": 0.7,      # Investigation within 1 hour
            "MEDIUM": 0.5,    # Investigation within 4 hours
            "LOW": 0.3,       # Log and monitor
        }
        
        self.notification_channels = [
            {"type": "webhook", "url": "$SECURITY_WEBHOOK"},
            {"type": "email", "recipients": ["security@company.com"]},
            {"type": "sentry", "project": "security-monitoring"},
            {"type": "slack", "channel": "#security-alerts"}
        ]
```

### Multi-Channel Alert Dispatch

**Webhook Alerts** (`packages/monitoring/security_monitor.py:246-263`)
```python
async def _send_webhook_alert(self, webhook_url: str, alert_data: Dict[str, Any]):
    """Send security alert via webhook with retry logic."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=alert_data) as response:
                if response.status == 200:
                    logger.info("Security webhook alert sent successfully")
                else:
                    logger.error(f"Webhook failed with status {response.status}")
    except Exception as e:
        logger.exception(f"Failed to send webhook alert: {e}")
```

**Sentry Integration** (`packages/monitoring/security_monitor.py:240-241`)
```python
# Automatic Sentry error reporting for security events
if SENTRY_AVAILABLE:
    sentry_sdk.capture_message(
        f"Security Alert: {event.event_type}",
        level="error", 
        extra=alert_data
    )
```

### Automated Incident Response

**Response Actions by Threat Level**:

**Critical Threats (>0.9)**
- Immediate IP blocking (if configured)
- Account suspension for repeated offenders
- Security team notification via multiple channels
- Automatic ticket creation in ITSM system
- Enhanced logging and forensic data collection

**High Threats (0.7-0.9)**
- Rate limiting enforcement
- Enhanced monitoring for user/IP
- Security team alert within 1 hour
- Behavioral analysis and profiling
- Temporary restrictions on high-risk operations

**Medium Threats (0.5-0.7)**
- Increased authentication requirements (MFA)
- Session monitoring and analysis
- Alert to security team within 4 hours
- User behavior documentation
- Audit trail enhancement

**Low Threats (0.3-0.5)**
- Logging and trend analysis
- Baseline behavior update
- Weekly security report inclusion
- No immediate action required

### Incident Response Workflow

```python
async def handle_security_incident(event: SecurityEvent):
    """Automated incident response workflow."""
    
    # 1. Immediate Assessment
    threat_level = determine_threat_level(event.threat_score)
    
    # 2. Automated Mitigation
    if threat_level == "CRITICAL":
        await apply_immediate_mitigations(event)
    
    # 3. Evidence Collection
    forensic_data = await collect_forensic_evidence(event)
    
    # 4. Notification
    await notify_security_team(event, threat_level)
    
    # 5. Documentation
    await create_incident_record(event, forensic_data)
    
    # 6. Follow-up Actions
    await schedule_follow_up_analysis(event)

async def apply_immediate_mitigations(event: SecurityEvent):
    """Apply immediate security mitigations."""
    
    if event.event_type == "brute_force":
        # Temporary IP blocking
        await block_ip_address(event.source_ip, duration="1h")
        
    elif event.event_type == "sql_injection":
        # Block suspicious queries
        await add_waf_rule(pattern=event.details["query_pattern"])
        
    elif event.event_type == "rate_limit_violation":
        # Enhanced rate limiting
        await apply_strict_rate_limit(event.user_id, multiplier=0.1)
    
    event.mitigated = True
    logger.info(f"Applied mitigations for {event.event_type}")
```

## Threat Intelligence Integration

### External Threat Feeds

```python
class ThreatIntelligenceManager:
    """Manage external threat intelligence feeds."""
    
    def __init__(self):
        self.feeds = [
            {"name": "abuse_ch", "url": "https://feodotracker.abuse.ch/downloads/ipblocklist.txt"},
            {"name": "spamhaus", "url": "https://www.spamhaus.org/drop/drop.txt"},
            {"name": "malware_domains", "url": "http://www.malwaredomainlist.com/hostslist/hosts.txt"}
        ]
        self.threat_db = {}
        
    async def update_threat_intelligence(self):
        """Update threat intelligence from external feeds."""
        for feed in self.feeds:
            try:
                threat_data = await self.fetch_threat_feed(feed["url"])
                self.threat_db[feed["name"]] = self.parse_threat_data(threat_data)
                logger.info(f"Updated threat intel from {feed['name']}")
            except Exception as e:
                logger.error(f"Failed to update {feed['name']}: {e}")
    
    def check_ip_reputation(self, ip_address: str) -> float:
        """Check IP address against threat intelligence."""
        threat_score = 0.0
        
        for feed_name, threat_list in self.threat_db.items():
            if ip_address in threat_list:
                threat_score += 0.3  # Each feed match adds 30%
                
        return min(threat_score, 1.0)
```

### IOC (Indicators of Compromise) Management

```python
@dataclass
class IOC:
    """Indicator of Compromise data structure."""
    indicator: str           # IP, domain, hash, etc.
    ioc_type: str           # ip, domain, hash, url
    threat_type: str        # malware, phishing, botnet
    confidence: float       # 0.0-1.0 confidence score
    source: str            # Intelligence source
    first_seen: datetime
    last_seen: datetime
    description: str

class IOCManager:
    """Manage indicators of compromise."""
    
    def __init__(self):
        self.iocs = []
        
    def add_ioc(self, ioc: IOC):
        """Add new IOC to database."""
        self.iocs.append(ioc)
        logger.info(f"Added IOC: {ioc.indicator} ({ioc.threat_type})")
        
    def check_ioc_match(self, indicator: str) -> Optional[IOC]:
        """Check if indicator matches known IOCs."""
        for ioc in self.iocs:
            if ioc.indicator == indicator:
                return ioc
        return None
        
    def get_threat_context(self, indicator: str) -> Dict[str, Any]:
        """Get threat context for indicator."""
        ioc = self.check_ioc_match(indicator)
        if ioc:
            return {
                "threat_type": ioc.threat_type,
                "confidence": ioc.confidence,
                "source": ioc.source,
                "description": ioc.description,
                "first_seen": ioc.first_seen.isoformat(),
                "risk_level": "high" if ioc.confidence > 0.8 else "medium"
            }
        return {"risk_level": "unknown"}
```

## MCP Server Security Integration

### Authentication Monitoring

```python
class MCPSecurityIntegration:
    """Security integration for MCP servers."""
    
    def __init__(self, security_monitor: SecurityMonitor):
        self.monitor = security_monitor
        
    async def log_authentication_attempt(
        self, user_id: str, source_ip: str, success: bool, details: Dict[str, Any]
    ):
        """Log MCP server authentication attempts."""
        if not success:
            await self.monitor.log_security_event(
                "auth_failure",
                user_id,
                source_ip,
                {
                    "service": "mcp_server",
                    "reason": details.get("reason", "unknown"),
                    "mcp_method": details.get("method"),
                    "timestamp": time.time()
                }
            )
    
    async def log_privilege_escalation(
        self, user_id: str, source_ip: str, attempted_action: str
    ):
        """Log privilege escalation attempts."""
        await self.monitor.log_security_event(
            "privilege_escalation",
            user_id,
            source_ip,
            {
                "attempted_action": attempted_action,
                "service": "mcp_server",
                "severity": "high",
                "requires_investigation": True
            }
        )
```

### Tool Usage Monitoring

```python
async def monitor_mcp_tool_usage(
    tool_name: str, user_id: str, source_ip: str, arguments: Dict[str, Any]
):
    """Monitor MCP tool usage for security risks."""
    
    # Check for high-risk tools
    high_risk_tools = ["bash", "write_file", "delete_file", "network_access"]
    
    if tool_name in high_risk_tools:
        await security_monitor.log_security_event(
            "high_risk_tool_usage",
            user_id,
            source_ip,
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "risk_level": "high",
                "requires_approval": True
            }
        )
    
    # Monitor for suspicious patterns
    if "rm -rf" in str(arguments) or "sudo" in str(arguments):
        await security_monitor.log_security_event(
            "suspicious_command",
            user_id,
            source_ip,
            {
                "tool_name": tool_name,
                "command_pattern": "destructive_operation",
                "arguments": arguments,
                "blocked": True
            }
        )
```

## Configuration and Tuning

### Environment Configuration

```bash
# Security monitoring environment variables
export SECURITY_ALERT_WEBHOOK="https://hooks.slack.com/services/..."
export SENTRY_DSN="https://...@sentry.io/..."
export GITHUB_TOKEN="ghp_..."
export ALERT_EMAIL_PASSWORD="..."

# Threat detection tuning
export BRUTE_FORCE_THRESHOLD="5"           # Failed attempts before alert
export SQL_INJECTION_SENSITIVITY="0.7"     # Sensitivity for pattern matching
export RATE_LIMIT_MULTIPLIER="2.0"         # Rate limit threshold multiplier
export BEHAVIOR_ANALYSIS_ENABLED="true"    # Enable behavioral analysis

# Threat intelligence
export THREAT_INTEL_UPDATE_INTERVAL="3600" # Update interval in seconds
export THREAT_FEED_TIMEOUT="30"            # Feed fetch timeout
export IOC_RETENTION_DAYS="90"             # IOC retention period
```

### Security Configuration File

```yaml
# security_config.yaml
security:
  threat_detection:
    brute_force:
      window_minutes: 5
      threshold_low: 3
      threshold_medium: 5
      threshold_high: 10
      
    sql_injection:
      pattern_weight: 0.3
      char_weight: 0.1
      max_score: 1.0
      
    rate_limiting:
      window_minutes: 1
      normal_rate: 30
      multiplier_medium: 1.5
      multiplier_high: 2.0
      multiplier_critical: 3.0
      
    behavioral:
      off_hours_weight: 0.2
      high_risk_ops_weight: 0.3
      failed_ops_threshold: 5
      failed_ops_weight: 0.4
      
  alerting:
    channels:
      - type: webhook
        url: "${SECURITY_WEBHOOK}"
        timeout: 10
        
      - type: email
        smtp_server: "smtp.company.com"
        smtp_port: 587
        username: "security@company.com"
        to_emails: ["security-team@company.com", "soc@company.com"]
        
      - type: github
        repo: "company/security-incidents"
        labels: ["security-alert", "automated"]
        
      - type: sentry
        project: "security-monitoring"
        environment: "production"
        
  response:
    auto_mitigation:
      ip_blocking:
        enabled: true
        duration_minutes: 60
        threshold: 0.9
        
      rate_limiting:
        enabled: true
        strictness_multiplier: 0.1
        threshold: 0.7
        
      account_suspension:
        enabled: false  # Requires manual approval
        threshold: 0.95
        
  threat_intelligence:
    feeds:
      - name: "abuse_ch"
        url: "https://feodotracker.abuse.ch/downloads/ipblocklist.txt"
        format: "text"
        update_hours: 6
        
      - name: "malware_domains"
        url: "http://www.malwaredomainlist.com/hostslist/hosts.txt"
        format: "hosts"
        update_hours: 12
        
    ioc_retention_days: 90
    confidence_threshold: 0.6
```

### Performance Tuning

**Memory Optimization**:
```python
# Optimize memory usage for high-volume environments
SECURITY_CONFIG = {
    "event_queue_size": 10000,      # Max events in queue
    "failed_attempts_ttl": 300,     # TTL for failed attempts cache
    "request_patterns_ttl": 60,     # TTL for rate limiting cache
    "threat_intel_cache_size": 100000,  # Max IOCs in memory
    "alert_batch_size": 50,         # Batch alerts for efficiency
    "metrics_buffer_size": 1000,    # Metrics buffer before flush
}
```

**Processing Optimization**:
```python
# Async processing configuration
ASYNC_CONFIG = {
    "event_workers": 4,             # Concurrent event processors
    "analysis_timeout": 30,         # Max analysis time per event
    "alert_timeout": 10,            # Max alert dispatch time
    "queue_drain_interval": 1,      # Queue processing interval
    "batch_processing": True,       # Enable batch processing
}
```

## Testing and Validation

### Security Test Framework

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestSecurityMonitor:
    """Comprehensive security monitoring tests."""
    
    @pytest.fixture
    def security_monitor(self):
        """Create security monitor for testing."""
        monitor = SecurityMonitor()
        monitor.alert_manager = AsyncMock()
        return monitor
    
    @pytest.mark.asyncio
    async def test_brute_force_detection(self, security_monitor):
        """Test brute force attack detection."""
        user_id = "test_user"
        source_ip = "192.168.1.100"
        
        # Simulate multiple failed attempts
        for i in range(6):
            await security_monitor.log_security_event(
                "auth_failure", user_id, source_ip, 
                {"attempt": i, "reason": "invalid_password"}
            )
        
        # Check threat score increases with attempts
        threat_score = security_monitor.detector.detect_brute_force(user_id, source_ip)
        assert threat_score >= 0.7  # Should be high threat
        
        # Verify alert was triggered
        security_monitor.alert_manager.send_alert.assert_called()
    
    @pytest.mark.asyncio 
    async def test_sql_injection_detection(self, security_monitor):
        """Test SQL injection detection."""
        malicious_inputs = [
            "' OR 1=1 --",
            "'; DROP TABLE users; --", 
            "UNION SELECT * FROM passwords",
            "<script>alert('xss')</script>"
        ]
        
        for input_data in malicious_inputs:
            threat_score = security_monitor.detector.detect_sql_injection(input_data)
            assert threat_score > 0.3  # Should detect threat
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, security_monitor):
        """Test rate limiting detection."""
        user_id = "heavy_user"
        endpoint = "/api/search"
        
        # Simulate high request rate
        for i in range(100):  # 100 requests in quick succession
            await security_monitor.log_security_event(
                "rate_limit_violation", user_id, "192.168.1.200",
                {"endpoint": endpoint, "request_id": i}
            )
        
        threat_score = security_monitor.detector.detect_rate_limit_violation(user_id, endpoint)
        assert threat_score >= 0.7  # Should detect high rate
```

### Security Simulation Framework

```python
class SecuritySimulator:
    """Simulate security attacks for testing."""
    
    def __init__(self, security_monitor: SecurityMonitor):
        self.monitor = security_monitor
        
    async def simulate_brute_force_attack(
        self, target_user: str, source_ip: str, attempts: int = 15
    ):
        """Simulate brute force attack."""
        for i in range(attempts):
            await self.monitor.log_security_event(
                "auth_failure", target_user, source_ip,
                {
                    "attempt": i + 1,
                    "reason": "invalid_password",
                    "user_agent": "BruteForceBot/1.0",
                    "timestamp": time.time()
                }
            )
            await asyncio.sleep(0.1)  # Realistic timing
    
    async def simulate_sql_injection_scan(
        self, attacker_user: str, source_ip: str
    ):
        """Simulate SQL injection scanning."""
        payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT password FROM users --",
            "1' AND 1=1 --",
            "admin'--",
            "' OR 1=1#"
        ]
        
        for payload in payloads:
            await self.monitor.log_security_event(
                "sql_injection_attempt", attacker_user, source_ip,
                {
                    "input_data": payload,
                    "endpoint": "/api/search",
                    "method": "POST",
                    "timestamp": time.time()
                }
            )
            await asyncio.sleep(0.5)
    
    async def simulate_ddos_attack(
        self, source_ips: List[str], target_endpoint: str, rps: int = 100
    ):
        """Simulate distributed denial of service attack."""
        for minute in range(5):  # 5 minute attack
            tasks = []
            for ip in source_ips:
                for _ in range(rps):  # Requests per second
                    task = self.monitor.log_security_event(
                        "rate_limit_violation", f"bot_{ip}", ip,
                        {
                            "endpoint": target_endpoint,
                            "method": "GET",
                            "user_agent": "AttackBot/2.0",
                            "minute": minute
                        }
                    )
                    tasks.append(task)
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(1)  # Wait 1 second between bursts

# Usage example
async def run_security_simulation():
    """Run comprehensive security simulation."""
    monitor = SecurityMonitor()
    await monitor.start()
    
    simulator = SecuritySimulator(monitor)
    
    # Run multiple attack simulations
    await asyncio.gather(
        simulator.simulate_brute_force_attack("admin", "10.0.0.100"),
        simulator.simulate_sql_injection_scan("attacker", "10.0.0.101"),
        simulator.simulate_ddos_attack(
            ["10.0.0.102", "10.0.0.103", "10.0.0.104"], 
            "/api/login"
        )
    )
    
    # Analyze results
    status = monitor.get_security_status()
    print(f"Security simulation complete:")
    print(f"- Critical alerts: {status['critical_alerts']}")
    print(f"- High alerts: {status['high_alerts']}")
    print(f"- Total events processed: {status['recent_alerts_count']}")
```

## Best Practices and Recommendations

### Implementation Best Practices

**1. Layered Security Approach**
- Implement multiple detection methods for comprehensive coverage
- Use correlation between different event types for better accuracy
- Combine real-time detection with batch analysis for trend identification

**2. Tuning and Calibration**
- Start with conservative thresholds and adjust based on false positive rates
- Regularly review and update threat patterns based on attack trends
- Implement A/B testing for new detection algorithms

**3. Response Automation**
- Automate low-risk mitigations (rate limiting, enhanced monitoring)
- Require human approval for high-impact actions (account suspension)
- Implement rollback capabilities for automated responses

**4. Performance Optimization**
- Use async processing for all I/O operations
- Implement caching for frequently accessed threat intelligence
- Batch alerts and notifications to reduce overhead

### Operational Guidelines

**1. Alert Fatigue Prevention**
- Implement intelligent alert grouping and deduplication
- Use progressive alerting (escalate only if not addressed)
- Provide clear action items and runbooks for each alert type

**2. Incident Response Integration**
- Integrate with ITSM systems for ticket creation
- Provide automated forensic data collection
- Implement standardized incident response workflows

**3. Compliance and Auditing**
- Maintain comprehensive audit logs for all security events
- Implement data retention policies compliant with regulations
- Provide regular security posture reports and metrics

**4. Continuous Improvement**
- Regular threat model updates based on attack trends
- Performance monitoring and optimization of detection algorithms
- User feedback integration for false positive reduction

This comprehensive security monitoring guide provides the foundation for robust threat detection and automated incident response within the AIVillage platform. The system is designed to be both comprehensive and performant, providing real-time protection while maintaining operational efficiency.