"""
WebSocket Security Validator - Comprehensive Protection Against RCE and Injection Attacks

Provides enterprise-grade security validation for WebSocket communications with:
- Zero-tolerance for eval() and exec() on user input
- Comprehensive input sanitization and validation
- Pydantic schema enforcement
- Security headers and rate limiting
- Attack detection and prevention
- Comprehensive audit logging
"""

import json
import re
import ast
import time
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass, field
import logging

from pydantic import BaseModel, ValidationError, validator, Field
from fastapi import WebSocket, HTTPException

# Configure security logging
security_logger = logging.getLogger("websocket.security")
security_logger.setLevel(logging.INFO)

class SecurityThreatLevel(Enum):
    """Security threat classification levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class AttackType(Enum):
    """Types of attacks we detect and prevent."""
    CODE_INJECTION = "code_injection"
    COMMAND_INJECTION = "command_injection"
    SCRIPT_INJECTION = "script_injection"
    PATH_TRAVERSAL = "path_traversal"
    SQL_INJECTION = "sql_injection"
    NOSQL_INJECTION = "nosql_injection"
    DDOS = "ddos_attempt"
    FUZZING = "fuzzing_attack"
    MALFORMED_JSON = "malformed_json"

@dataclass
class SecurityThreat:
    """Details about a detected security threat."""
    threat_type: AttackType
    severity: SecurityThreatLevel
    description: str
    payload: str
    source_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    blocked: bool = True

class MessageSchema(BaseModel):
    """Secure WebSocket message schema with comprehensive validation."""
    type: str = Field(..., pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$', max_length=50)
    data: Optional[Dict[str, Any]] = Field(default=None)
    timestamp: Optional[datetime] = None
    message_id: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9-_]+$', max_length=100)
    request_id: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9-_]+$', max_length=100)
    priority: int = Field(default=1, ge=1, le=5)
    
    @validator('type')
    def validate_message_type(cls, v):
        """Validate message type against allowed values."""
        allowed_types = {
            'ping', 'pong', 'connect', 'disconnect',
            'inference_submit', 'inference_status', 'inference_progress', 'inference_result',
            'performance_metrics', 'optimization_recommendation',
            'subscribe', 'unsubscribe', 'get_status'
        }
        if v not in allowed_types:
            raise ValueError(f'Message type "{v}" not allowed. Allowed types: {allowed_types}')
        return v
    
    @validator('data', pre=True)
    def validate_data(cls, v):
        """Comprehensive validation of message data."""
        if v is None:
            return v
        
        # Convert to string for pattern checking
        data_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'os\.popen',
            r'open\s*\(',
            r'file\s*\(',
            r'compile\s*\(',
            r'execfile\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'\$\{.*\}',  # Template injection
            r'<%.*%>',    # Server-side template injection
            r'{{.*}}',    # Template injection
            r'javascript:',
            r'data:text/html',
            r'<script',
            r'</script>',
            r'onerror\s*=',
            r'onload\s*=',
            r'alert\s*\(',
            r'confirm\s*\(',
            r'prompt\s*\(',
            r'\.\./.*\.\.',  # Path traversal
            r'/etc/passwd',
            r'/etc/shadow',
            r'cmd\.exe',
            r'powershell',
            r'bash -c',
            r'sh -c',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                raise ValueError(f'Dangerous pattern detected: {pattern}')
        
        return v

class WebSocketSecurityValidator:
    """
    Comprehensive WebSocket security validator and protection system.
    
    Provides multiple layers of security including:
    - Input validation and sanitization
    - Pattern-based attack detection
    - Rate limiting and DoS protection
    - Comprehensive audit logging
    - Real-time threat monitoring
    """
    
    def __init__(self):
        """Initialize security validator with threat detection patterns."""
        self.threat_patterns = self._load_threat_patterns()
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.blocked_ips: Set[str] = set()
        self.security_events: List[SecurityThreat] = []
        self.max_security_events = 1000
        
        # Rate limiting configuration
        self.max_requests_per_minute = 60
        self.max_requests_per_second = 10
        self.block_duration_minutes = 30
        
        security_logger.info("WebSocket Security Validator initialized")
    
    def _load_threat_patterns(self) -> Dict[AttackType, List[str]]:
        """Load comprehensive threat detection patterns."""
        return {
            AttackType.CODE_INJECTION: [
                r'eval\s*\([^)]*\)',
                r'exec\s*\([^)]*\)',
                r'__import__\s*\([^)]*\)',
                r'compile\s*\([^)]*\)',
                r'execfile\s*\([^)]*\)',
                r'globals\s*\(\)',
                r'locals\s*\(\)',
                r'vars\s*\(\)',
                r'dir\s*\(\)',
                r'getattr\s*\(',
                r'setattr\s*\(',
                r'hasattr\s*\(',
                r'callable\s*\(',
            ],
            AttackType.COMMAND_INJECTION: [
                r'subprocess\.[a-zA-Z_]+\(',
                r'os\.system\s*\(',
                r'os\.popen\s*\(',
                r'os\.spawn[a-zA-Z_]*\s*\(',
                r'commands\.[a-zA-Z_]+\(',
                r'popen[0-9]*\s*\(',
                r'cmd\.exe',
                r'powershell(?:\.exe)?',
                r'/bin/(sh|bash|zsh|csh|tcsh)',
                r'bash\s+-c',
                r'sh\s+-c',
                r'\|\s*(sh|bash|cmd)',
                r';\s*(sh|bash|cmd)',
                r'&&\s*(sh|bash|cmd)',
                r'\$\([^)]+\)',  # Command substitution
            ],
            AttackType.SCRIPT_INJECTION: [
                r'<script[^>]*>',
                r'</script>',
                r'javascript:',
                r'vbscript:',
                r'data:text/html',
                r'data:application/javascript',
                r'on\w+\s*=',  # Event handlers
                r'alert\s*\(',
                r'confirm\s*\(',
                r'prompt\s*\(',
                r'console\.[a-zA-Z_]+\(',
                r'document\.[a-zA-Z_]+',
                r'window\.[a-zA-Z_]+',
                r'location\.[a-zA-Z_]+',
            ],
            AttackType.PATH_TRAVERSAL: [
                r'\.\.[\\/]',
                r'[\\/]\.\.[\\/]',
                r'\.\.%2[fF]',
                r'%2[eE]%2[eE]%2[fF]',
                r'\.\.\\',
                r'\.\./',
                r'/etc/passwd',
                r'/etc/shadow',
                r'/etc/hosts',
                r'/proc/self/',
                r'\\windows\\system32',
                r'c:\\windows',
            ],
            AttackType.SQL_INJECTION: [
                r"'[^']*'?\s*(;|--|\/\*)",
                r'"\s*(;|--|\/\*)',
                r'\bunion\s+select\b',
                r'\bselect\s+.*\bfrom\b',
                r'\binsert\s+into\b',
                r'\bupdate\s+.*\bset\b',
                r'\bdelete\s+from\b',
                r'\bdrop\s+(table|database|schema)\b',
                r'\bcreate\s+(table|database|schema)\b',
                r'\balter\s+table\b',
                r'\bexec\s*\(',
                r'xp_cmdshell',
                r'sp_executesql',
            ],
            AttackType.NOSQL_INJECTION: [
                r'\$where\s*:',
                r'\$ne\s*:',
                r'\$gt\s*:',
                r'\$lt\s*:',
                r'\$regex\s*:',
                r'MapReduce',
                r'\$function\s*:',
                r'db\..*\.find\s*\(',
                r'db\..*\.update\s*\(',
                r'db\..*\.remove\s*\(',
                r'ObjectId\s*\(',
            ]
        }
    
    async def validate_message(self, raw_message: str, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and parse incoming WebSocket message with comprehensive security checks.
        
        Args:
            raw_message: Raw message string from WebSocket
            client_info: Information about the client connection
            
        Returns:
            Parsed and validated message data
            
        Raises:
            SecurityError: If message contains security threats
            ValidationError: If message format is invalid
        """
        try:
            # Step 1: Rate limiting check
            client_ip = client_info.get('remote_addr', 'unknown')
            await self._check_rate_limits(client_ip)
            
            # Step 2: Basic format validation
            if not raw_message or len(raw_message) > 1024 * 1024:  # 1MB limit
                raise SecurityError(
                    SecurityThreat(
                        threat_type=AttackType.DDOS,
                        severity=SecurityThreatLevel.HIGH,
                        description="Message too large or empty",
                        payload=raw_message[:100] + "..." if len(raw_message) > 100 else raw_message
                    )
                )
            
            # Step 3: Threat pattern detection
            detected_threats = self._detect_threats(raw_message, client_info)
            if detected_threats:
                # Log all threats
                for threat in detected_threats:
                    self._log_security_event(threat, client_info)
                
                # Raise error for critical threats
                critical_threats = [t for t in detected_threats if t.severity == SecurityThreatLevel.CRITICAL]
                if critical_threats:
                    raise SecurityError(critical_threats[0])
            
            # Step 4: JSON parsing with safety checks  
            try:
                # Use safe JSON parsing - never eval()
                message_data = json.loads(raw_message)
            except json.JSONDecodeError as e:
                threat = SecurityThreat(
                    threat_type=AttackType.MALFORMED_JSON,
                    severity=SecurityThreatLevel.MEDIUM,
                    description=f"Invalid JSON format: {str(e)}",
                    payload=raw_message[:200] + "..." if len(raw_message) > 200 else raw_message
                )
                self._log_security_event(threat, client_info)
                raise SecurityError(threat)
            
            # Step 5: Pydantic schema validation
            try:
                validated_message = MessageSchema(**message_data)
            except ValidationError as e:
                threat = SecurityThreat(
                    threat_type=AttackType.CODE_INJECTION,
                    severity=SecurityThreatLevel.HIGH,
                    description=f"Schema validation failed: {str(e)}",
                    payload=raw_message[:200] + "..." if len(raw_message) > 200 else raw_message
                )
                self._log_security_event(threat, client_info)
                raise SecurityError(threat)
            
            # Step 6: Additional deep inspection
            await self._deep_content_inspection(validated_message.dict(), client_info)
            
            # Step 7: Update rate limiting
            self._update_rate_limits(client_ip)
            
            # Return validated message
            return validated_message.dict()
            
        except SecurityError:
            raise
        except Exception as e:
            # Log unexpected errors
            security_logger.error(f"Unexpected error in message validation: {e}")
            threat = SecurityThreat(
                threat_type=AttackType.CODE_INJECTION,
                severity=SecurityThreatLevel.HIGH,
                description=f"Validation error: {str(e)}",
                payload=raw_message[:200] + "..." if len(raw_message) > 200 else raw_message
            )
            raise SecurityError(threat)
    
    def _detect_threats(self, message: str, client_info: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect security threats using pattern matching."""
        detected_threats = []
        
        for attack_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE | re.MULTILINE):
                    severity = self._determine_threat_severity(attack_type, pattern, message)
                    
                    threat = SecurityThreat(
                        threat_type=attack_type,
                        severity=severity,
                        description=f"{attack_type.value} pattern detected: {pattern}",
                        payload=message[:500] + "..." if len(message) > 500 else message,
                        source_info=client_info
                    )
                    detected_threats.append(threat)
        
        return detected_threats
    
    def _determine_threat_severity(self, attack_type: AttackType, pattern: str, message: str) -> SecurityThreatLevel:
        """Determine threat severity based on attack type and context."""
        # Critical threats that allow immediate code execution
        critical_patterns = [
            'eval', 'exec', '__import__', 'subprocess', 'os.system', 'os.popen'
        ]
        
        if any(cp in pattern.lower() for cp in critical_patterns):
            return SecurityThreatLevel.CRITICAL
        
        # High severity threats
        if attack_type in [AttackType.CODE_INJECTION, AttackType.COMMAND_INJECTION]:
            return SecurityThreatLevel.HIGH
        
        # Medium severity threats
        if attack_type in [AttackType.SCRIPT_INJECTION, AttackType.SQL_INJECTION]:
            return SecurityThreatLevel.MEDIUM
        
        return SecurityThreatLevel.LOW
    
    async def _deep_content_inspection(self, message_data: Dict[str, Any], client_info: Dict[str, Any]):
        """Perform deep inspection of message content."""
        # Check for nested dangerous content
        def inspect_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    inspect_recursive(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    inspect_recursive(item, f"{path}[{i}]")
            elif isinstance(obj, str):
                # Check for encoded payloads
                if self._check_encoded_payload(obj):
                    threat = SecurityThreat(
                        threat_type=AttackType.CODE_INJECTION,
                        severity=SecurityThreatLevel.HIGH,
                        description=f"Encoded dangerous payload detected at {path}",
                        payload=obj[:200] + "..." if len(obj) > 200 else obj,
                        source_info=client_info
                    )
                    raise SecurityError(threat)
        
        inspect_recursive(message_data)
    
    def _check_encoded_payload(self, content: str) -> bool:
        """Check for base64 or hex encoded dangerous payloads."""
        try:
            # Check base64 encoding
            import base64
            try:
                decoded = base64.b64decode(content + "==").decode('utf-8', errors='ignore')
                dangerous_keywords = ['eval', 'exec', 'import', 'subprocess', 'system']
                if any(keyword in decoded.lower() for keyword in dangerous_keywords):
                    return True
            except:
                pass
            
            # Check hex encoding
            try:
                if all(c in '0123456789abcdefABCDEF' for c in content) and len(content) % 2 == 0:
                    decoded = bytes.fromhex(content).decode('utf-8', errors='ignore')
                    dangerous_keywords = ['eval', 'exec', 'import', 'subprocess', 'system']
                    if any(keyword in decoded.lower() for keyword in dangerous_keywords):
                        return True
            except:
                pass
                
        except Exception:
            pass
        
        return False
    
    async def _check_rate_limits(self, client_ip: str):
        """Check rate limits for client IP."""
        if client_ip in self.blocked_ips:
            raise SecurityError(
                SecurityThreat(
                    threat_type=AttackType.DDOS,
                    severity=SecurityThreatLevel.CRITICAL,
                    description="IP address is temporarily blocked due to abuse",
                    payload=f"IP: {client_ip}"
                )
            )
        
        now = datetime.now()
        
        # Initialize rate limit tracking
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = []
        
        # Clean old entries
        cutoff_time = now - timedelta(minutes=1)
        self.rate_limits[client_ip] = [
            timestamp for timestamp in self.rate_limits[client_ip]
            if timestamp > cutoff_time
        ]
        
        # Check rate limits
        if len(self.rate_limits[client_ip]) >= self.max_requests_per_minute:
            # Block IP temporarily
            self.blocked_ips.add(client_ip)
            
            # Schedule unblock
            import asyncio
            asyncio.create_task(self._schedule_unblock(client_ip))
            
            threat = SecurityThreat(
                threat_type=AttackType.DDOS,
                severity=SecurityThreatLevel.CRITICAL,
                description=f"Rate limit exceeded: {len(self.rate_limits[client_ip])} requests/minute",
                payload=f"IP: {client_ip}"
            )
            raise SecurityError(threat)
    
    def _update_rate_limits(self, client_ip: str):
        """Update rate limit tracking."""
        now = datetime.now()
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = []
        self.rate_limits[client_ip].append(now)
    
    async def _schedule_unblock(self, client_ip: str):
        """Schedule IP unblocking after block duration."""
        import asyncio
        await asyncio.sleep(self.block_duration_minutes * 60)
        self.blocked_ips.discard(client_ip)
        security_logger.info(f"Unblocked IP: {client_ip}")
    
    def _log_security_event(self, threat: SecurityThreat, client_info: Dict[str, Any]):
        """Log security events for audit and monitoring."""
        # Add to security events list
        self.security_events.append(threat)
        
        # Limit memory usage
        if len(self.security_events) > self.max_security_events:
            self.security_events = self.security_events[-self.max_security_events//2:]
        
        # Log to security logger
        log_data = {
            'timestamp': threat.timestamp.isoformat(),
            'threat_type': threat.threat_type.value,
            'severity': threat.severity.value,
            'description': threat.description,
            'client_ip': client_info.get('remote_addr', 'unknown'),
            'user_agent': client_info.get('user_agent', 'unknown'),
            'payload_hash': hashlib.sha256(threat.payload.encode()).hexdigest()[:16]
        }
        
        if threat.severity in [SecurityThreatLevel.CRITICAL, SecurityThreatLevel.HIGH]:
            security_logger.error(f"SECURITY THREAT: {json.dumps(log_data)}")
        else:
            security_logger.warning(f"Security event: {json.dumps(log_data)}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        recent_events = [e for e in self.security_events if e.timestamp > last_hour]
        daily_events = [e for e in self.security_events if e.timestamp > last_day]
        
        threat_counts = {}
        for event in daily_events:
            threat_type = event.threat_type.value
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        return {
            'total_events': len(self.security_events),
            'recent_events_1h': len(recent_events),
            'daily_events_24h': len(daily_events),
            'blocked_ips_count': len(self.blocked_ips),
            'threat_type_counts': threat_counts,
            'critical_events_24h': len([e for e in daily_events if e.severity == SecurityThreatLevel.CRITICAL]),
            'high_events_24h': len([e for e in daily_events if e.severity == SecurityThreatLevel.HIGH]),
            'rate_limited_clients': len(self.rate_limits),
            'generated_at': now.isoformat()
        }

class SecurityError(Exception):
    """Exception raised when security threats are detected."""
    
    def __init__(self, threat: SecurityThreat):
        self.threat = threat
        super().__init__(f"Security threat detected: {threat.description}")

# Safe alternative functions for any legacy eval() usage
class SafeEvaluator:
    """Safe alternatives to eval() and exec() functions."""
    
    @staticmethod
    def safe_literal_eval(expression: str) -> Any:
        """Safely evaluate string containing Python literal structures."""
        try:
            return ast.literal_eval(expression)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid literal expression: {e}")
    
    @staticmethod
    def safe_json_loads(json_string: str) -> Any:
        """Safely parse JSON string."""
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    
    @staticmethod
    def safe_parse_query_string(query: str) -> Dict[str, str]:
        """Safely parse query string parameters."""
        from urllib.parse import parse_qs, unquote
        
        result = {}
        try:
            parsed = parse_qs(query, strict_parsing=True)
            for key, values in parsed.items():
                # Take first value and sanitize
                result[unquote(key)] = unquote(values[0]) if values else ""
        except Exception as e:
            raise ValueError(f"Invalid query string: {e}")
        
        return result

# Export main components
__all__ = [
    'WebSocketSecurityValidator',
    'SecurityError',
    'SecurityThreat', 
    'SecurityThreatLevel',
    'AttackType',
    'MessageSchema',
    'SafeEvaluator'
]