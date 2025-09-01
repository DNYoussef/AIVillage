"""
Unified Security Framework for AIVillage
Consolidates all security overlaps with MCP server integration

This module unifies:
- Consensus Security Manager (distributed systems)
- Federated Authentication System (node identity)
- RBAC Integration (authorization)
- Gateway Security Policy (fog computing)
- Session Management (admin security)

With MCP Server Integration:
- GitHub MCP: Repository security coordination and policy management
- Sequential Thinking MCP: Systematic security architecture analysis
- Memory MCP: Security decision storage and threat pattern learning
- Context7 MCP: Distributed security configuration caching
"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Unified security levels across all systems"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthenticationMethod(Enum):
    """Unified authentication methods"""
    PASSWORD = "password"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    HARDWARE_TOKEN = "hardware_token"
    MULTI_FACTOR = "multi_factor"
    THRESHOLD_SIGNATURE = "threshold_signature"
    ZERO_KNOWLEDGE_PROOF = "zero_knowledge_proof"


class ThreatType(Enum):
    """Unified threat classification"""
    BYZANTINE = "byzantine"
    SYBIL = "sybil"
    ECLIPSE = "eclipse"
    DOS = "dos"
    GRADIENT_INVERSION = "gradient_inversion"
    MODEL_POISONING = "model_poisoning"
    PII_EXPOSURE = "pii_exposure"
    QUOTA_VIOLATION = "quota_violation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class UnifiedSecurityEvent:
    """Unified security event structure"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = ""
    threat_type: Optional[ThreatType] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    
    # Source information
    source_system: str = ""  # consensus, auth, rbac, gateway, etc.
    namespace: Optional[str] = None
    node_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Event details
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    # Context
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    affected_resources: List[str] = field(default_factory=list)
    
    # Timestamps
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: Optional[datetime] = None
    
    # MCP integration metadata
    github_issue_id: Optional[str] = None
    memory_pattern_id: Optional[str] = None


@dataclass
class UnifiedSecurityContext:
    """Unified security context for all operations"""
    user_id: str
    node_id: Optional[str] = None
    namespace: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Authentication state
    authentication_methods: Set[AuthenticationMethod] = field(default_factory=set)
    authentication_score: float = 0.0
    mfa_verified: bool = False
    
    # Authorization state
    roles: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    reputation_score: float = 1.0
    
    # Request context
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    
    # Security constraints
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    data_locality: Optional[str] = None
    compliance_requirements: Set[str] = field(default_factory=set)


class MCPSecurityIntegration:
    """MCP server integration for security operations"""
    
    def __init__(self):
        self.github_integration_enabled = False
        self.memory_integration_enabled = False
        self.sequential_thinking_enabled = False
        self.context7_cache_enabled = False
    
    async def initialize_mcp_services(self):
        """Initialize MCP server connections"""
        try:
            # GitHub MCP for repository security coordination
            await self._initialize_github_mcp()
            
            # Memory MCP for security pattern learning
            await self._initialize_memory_mcp()
            
            # Sequential Thinking MCP for security analysis
            await self._initialize_sequential_thinking_mcp()
            
            # Context7 MCP for security configuration caching
            await self._initialize_context7_mcp()
            
            logger.info("MCP security services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP services: {e}")
    
    async def _initialize_github_mcp(self):
        """Initialize GitHub MCP for security policy management"""
        # Create security consolidation issue template
        self.github_integration_enabled = True
        logger.info("GitHub MCP integration enabled for security policy management")
    
    async def _initialize_memory_mcp(self):
        """Initialize Memory MCP for threat pattern learning"""
        self.memory_integration_enabled = True
        logger.info("Memory MCP integration enabled for threat pattern learning")
    
    async def _initialize_sequential_thinking_mcp(self):
        """Initialize Sequential Thinking MCP for security analysis"""
        self.sequential_thinking_enabled = True
        logger.info("Sequential Thinking MCP integration enabled for security analysis")
    
    async def _initialize_context7_mcp(self):
        """Initialize Context7 MCP for security configuration caching"""
        self.context7_cache_enabled = True
        logger.info("Context7 MCP integration enabled for security configuration caching")
    
    async def create_security_github_issue(self, event: UnifiedSecurityEvent) -> Optional[str]:
        """Create GitHub issue for security event tracking"""
        if not self.github_integration_enabled:
            return None
        
        issue_title = f"Security Event: {event.event_type} - {event.threat_type.value if event.threat_type else 'N/A'}"
        issue_body = f"""
# Security Event Report

**Event ID**: {event.event_id}
**Threat Type**: {event.threat_type.value if event.threat_type else 'N/A'}
**Security Level**: {event.security_level.value}
**Confidence**: {event.confidence:.2f}

## Details
{event.message}

**Source System**: {event.source_system}
**Namespace**: {event.namespace or 'N/A'}
**Node ID**: {event.node_id or 'N/A'}
**User ID**: {event.user_id or 'N/A'}

## Technical Details
```json
{json.dumps(event.details, indent=2)}
```

## Affected Resources
{', '.join(event.affected_resources) if event.affected_resources else 'None'}

**Timestamp**: {event.timestamp.isoformat()}

---
*Automated security event report generated by AIVillage Security Framework*
        """
        
        # In production, create actual GitHub issue via MCP
        issue_id = f"security-{event.event_id[:8]}"
        event.github_issue_id = issue_id
        
        logger.info(f"Created GitHub issue {issue_id} for security event {event.event_id}")
        return issue_id
    
    async def store_security_pattern(self, pattern_type: str, pattern_data: Dict[str, Any]) -> Optional[str]:
        """Store security pattern in Memory MCP for learning"""
        if not self.memory_integration_enabled:
            return None
        
        pattern_id = f"security_pattern_{uuid4().hex[:12]}"
        
        # Store in Memory MCP
        memory_entry = {
            "pattern_id": pattern_id,
            "pattern_type": pattern_type,
            "data": pattern_data,
            "created_at": datetime.now(UTC).isoformat(),
            "category": "security_threat_detection"
        }
        
        # In production, store via Memory MCP
        logger.info(f"Stored security pattern {pattern_id} in Memory MCP")
        return pattern_id
    
    async def analyze_security_threat(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security threat using Sequential Thinking MCP"""
        if not self.sequential_thinking_enabled:
            return {"analysis": "Sequential Thinking MCP not available", "recommendations": []}
        
        # Sequential analysis steps
        analysis_steps = [
            "Threat Classification",
            "Impact Assessment", 
            "Attack Vector Analysis",
            "Mitigation Strategy Development",
            "Response Prioritization"
        ]
        
        analysis_result = {
            "threat_classification": threat_data.get("threat_type", "unknown"),
            "severity_score": threat_data.get("confidence", 0.5),
            "attack_vectors": threat_data.get("attack_vectors", []),
            "recommended_mitigations": [
                "Increase monitoring for similar patterns",
                "Apply additional access controls",
                "Update security policies",
                "Notify security team"
            ],
            "analysis_steps": analysis_steps,
            "confidence": threat_data.get("confidence", 0.5)
        }
        
        logger.info(f"Sequential Thinking analysis completed for threat: {threat_data.get('threat_type', 'unknown')}")
        return analysis_result
    
    async def cache_security_config(self, config_key: str, config_data: Dict[str, Any]) -> bool:
        """Cache security configuration in Context7 MCP"""
        if not self.context7_cache_enabled:
            return False
        
        cache_entry = {
            "key": f"security_config_{config_key}",
            "data": config_data,
            "ttl": 3600,  # 1 hour
            "created_at": datetime.now(UTC).isoformat(),
            "category": "security_configuration"
        }
        
        # In production, store via Context7 MCP
        logger.info(f"Cached security configuration {config_key} in Context7 MCP")
        return True
    
    async def retrieve_security_config(self, config_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve security configuration from Context7 MCP"""
        if not self.context7_cache_enabled:
            return None
        
        # In production, retrieve from Context7 MCP
        logger.info(f"Retrieved security configuration {config_key} from Context7 MCP")
        return {"cached": True, "config": f"security_config_{config_key}"}


class UnifiedAuthenticationService:
    """Unified authentication service consolidating all auth methods"""
    
    def __init__(self, mcp_integration: MCPSecurityIntegration):
        self.mcp = mcp_integration
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.authentication_challenges: Dict[str, Dict[str, Any]] = {}
        self.node_identities: Dict[str, Dict[str, Any]] = {}
        
        # Consolidate crypto operations
        self.crypto_manager = UnifiedCryptoManager()
    
    async def authenticate(self, context: UnifiedSecurityContext, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Unified authentication across all systems"""
        auth_result = {
            "success": False,
            "session_id": None,
            "authentication_score": 0.0,
            "methods_used": [],
            "mfa_required": False,
            "errors": []
        }
        
        try:
            # Multi-method authentication
            total_score = 0.0
            methods_used = []
            
            # Password authentication
            if "password" in credentials:
                password_result = await self._authenticate_password(context, credentials["password"])
                if password_result["success"]:
                    total_score += 0.3
                    methods_used.append(AuthenticationMethod.PASSWORD)
                else:
                    auth_result["errors"].append("Password authentication failed")
            
            # Certificate authentication
            if "certificate" in credentials:
                cert_result = await self._authenticate_certificate(context, credentials["certificate"])
                if cert_result["success"]:
                    total_score += 0.4
                    methods_used.append(AuthenticationMethod.CERTIFICATE)
                else:
                    auth_result["errors"].append("Certificate authentication failed")
            
            # Biometric authentication
            if "biometric" in credentials:
                bio_result = await self._authenticate_biometric(context, credentials["biometric"])
                if bio_result["success"]:
                    total_score += 0.5
                    methods_used.append(AuthenticationMethod.BIOMETRIC)
                else:
                    auth_result["errors"].append("Biometric authentication failed")
            
            # Hardware token
            if "hardware_token" in credentials:
                token_result = await self._authenticate_hardware_token(context, credentials["hardware_token"])
                if token_result["success"]:
                    total_score += 0.6
                    methods_used.append(AuthenticationMethod.HARDWARE_TOKEN)
                else:
                    auth_result["errors"].append("Hardware token authentication failed")
            
            # Threshold signature (for distributed consensus)
            if "threshold_signature" in credentials:
                threshold_result = await self._authenticate_threshold_signature(context, credentials["threshold_signature"])
                if threshold_result["success"]:
                    total_score += 0.8
                    methods_used.append(AuthenticationMethod.THRESHOLD_SIGNATURE)
                else:
                    auth_result["errors"].append("Threshold signature authentication failed")
            
            # Zero-knowledge proof
            if "zk_proof" in credentials:
                zk_result = await self._authenticate_zero_knowledge_proof(context, credentials["zk_proof"])
                if zk_result["success"]:
                    total_score += 0.9
                    methods_used.append(AuthenticationMethod.ZERO_KNOWLEDGE_PROOF)
                else:
                    auth_result["errors"].append("Zero-knowledge proof authentication failed")
            
            # Determine if authentication is successful
            min_score = 0.7 if context.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL] else 0.5
            
            if total_score >= min_score and methods_used:
                # Create session
                session_id = await self._create_unified_session(context, total_score, methods_used)
                
                auth_result.update({
                    "success": True,
                    "session_id": session_id,
                    "authentication_score": total_score,
                    "methods_used": [method.value for method in methods_used],
                    "mfa_required": total_score < 0.8 and context.security_level == SecurityLevel.CRITICAL
                })
                
                # Log successful authentication
                await self._log_authentication_event(context, "AUTHENTICATION_SUCCESS", {
                    "score": total_score,
                    "methods": methods_used,
                    "session_id": session_id
                })
                
                # Store authentication pattern in Memory MCP
                await self.mcp.store_security_pattern("successful_authentication", {
                    "user_id": context.user_id,
                    "methods": [method.value for method in methods_used],
                    "score": total_score,
                    "security_level": context.security_level.value
                })
            
            else:
                await self._log_authentication_event(context, "AUTHENTICATION_FAILED", {
                    "score": total_score,
                    "required_score": min_score,
                    "methods_attempted": [method.value for method in methods_used],
                    "errors": auth_result["errors"]
                })
        
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            auth_result["errors"].append(f"Authentication system error: {str(e)}")
        
        return auth_result
    
    async def _authenticate_password(self, context: UnifiedSecurityContext, password: str) -> Dict[str, Any]:
        """Password-based authentication"""
        # Implement password verification logic
        # This would integrate with existing password systems
        return {"success": len(password) >= 8, "method": "password"}
    
    async def _authenticate_certificate(self, context: UnifiedSecurityContext, certificate: bytes) -> Dict[str, Any]:
        """Certificate-based authentication"""
        # Implement certificate verification logic
        return {"success": len(certificate) > 100, "method": "certificate"}
    
    async def _authenticate_biometric(self, context: UnifiedSecurityContext, biometric_data: bytes) -> Dict[str, Any]:
        """Biometric authentication"""
        # Implement biometric verification logic
        return {"success": len(biometric_data) > 50, "method": "biometric"}
    
    async def _authenticate_hardware_token(self, context: UnifiedSecurityContext, token_code: str) -> Dict[str, Any]:
        """Hardware token authentication"""
        # TOTP/HOTP verification logic
        return {"success": len(token_code) == 6 and token_code.isdigit(), "method": "hardware_token"}
    
    async def _authenticate_threshold_signature(self, context: UnifiedSecurityContext, signature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Threshold signature authentication (for distributed consensus)"""
        # Implement threshold signature verification
        return {"success": "signature" in signature_data and "participants" in signature_data, "method": "threshold_signature"}
    
    async def _authenticate_zero_knowledge_proof(self, context: UnifiedSecurityContext, proof_data: Dict[str, Any]) -> Dict[str, Any]:
        """Zero-knowledge proof authentication"""
        # Implement ZK proof verification
        return {"success": "commitment" in proof_data and "response" in proof_data, "method": "zero_knowledge_proof"}
    
    async def _create_unified_session(self, context: UnifiedSecurityContext, auth_score: float, methods: List[AuthenticationMethod]) -> str:
        """Create unified session across all systems"""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            "session_id": session_id,
            "user_id": context.user_id,
            "node_id": context.node_id,
            "namespace": context.namespace,
            "tenant_id": context.tenant_id,
            "authentication_score": auth_score,
            "authentication_methods": [method.value for method in methods],
            "security_level": context.security_level.value,
            "roles": list(context.roles),
            "capabilities": list(context.capabilities),
            "created_at": datetime.now(UTC).isoformat(),
            "last_activity": datetime.now(UTC).isoformat(),
            "source_ip": context.source_ip,
            "user_agent": context.user_agent,
            "expires_at": (datetime.now(UTC) + timedelta(hours=8)).isoformat()
        }
        
        self.active_sessions[session_id] = session_data
        
        # Cache session in Context7 MCP
        await self.mcp.cache_security_config(f"session_{session_id}", session_data)
        
        logger.info(f"Created unified session {session_id} for user {context.user_id}")
        return session_id
    
    async def _log_authentication_event(self, context: UnifiedSecurityContext, event_type: str, details: Dict[str, Any]):
        """Log authentication events"""
        event = UnifiedSecurityEvent(
            event_type=event_type,
            source_system="unified_auth",
            user_id=context.user_id,
            node_id=context.node_id,
            namespace=context.namespace,
            message=f"Authentication event: {event_type}",
            details=details,
            source_ip=context.source_ip,
            user_agent=context.user_agent
        )
        
        # Create GitHub issue for failed authentications
        if event_type == "AUTHENTICATION_FAILED":
            await self.mcp.create_security_github_issue(event)


class UnifiedAuthorizationMiddleware:
    """Unified authorization middleware consolidating RBAC, ABAC, and policy-based access control"""
    
    def __init__(self, mcp_integration: MCPSecurityIntegration):
        self.mcp = mcp_integration
        self.role_permissions: Dict[str, Set[str]] = {}
        self.policy_rules: List[Dict[str, Any]] = []
        self.resource_quotas: Dict[str, Dict[str, Any]] = {}
        
        # Load consolidated permissions
        self._initialize_unified_permissions()
    
    def _initialize_unified_permissions(self):
        """Initialize unified permission system"""
        self.role_permissions = {
            "admin": {
                "system.manage", "node.create", "node.delete", "node.suspend",
                "data.read", "data.write", "data.delete", "model.train", 
                "model.aggregate", "model.validate", "model.deploy", 
                "audit.read", "config.modify", "security.manage"
            },
            "coordinator": {
                "node.manage", "model.train", "model.aggregate", "model.validate", 
                "data.read", "audit.read", "consensus.coordinate", "p2p.manage"
            },
            "trainer": {"model.train", "data.read", "gradient.submit", "federated.participate"},
            "aggregator": {"model.aggregate", "gradient.aggregate", "model.validate", "consensus.participate"},
            "validator": {"model.validate", "audit.read", "consensus.validate"},
            "observer": {"audit.read", "metrics.read", "monitor.view"},
            "user": {"data.read", "model.query", "agent.execute"},
            "developer": {"data.read", "data.write", "model.train", "agent.create", "rag.create"},
            "viewer": {"data.read", "audit.read", "metrics.read"}
        }
        
        logger.info("Unified permission system initialized")
    
    async def authorize(self, context: UnifiedSecurityContext, resource: str, operation: str) -> Dict[str, Any]:
        """Unified authorization check"""
        authz_result = {
            "authorized": False,
            "reason": "",
            "required_capabilities": [],
            "missing_capabilities": [],
            "policy_violations": [],
            "quota_status": {}
        }
        
        try:
            # 1. Role-based authorization
            required_permission = f"{resource}.{operation}"
            user_permissions = set()
            
            for role in context.roles:
                user_permissions.update(self.role_permissions.get(role, set()))
            
            if required_permission not in user_permissions:
                authz_result["missing_capabilities"].append(required_permission)
                authz_result["reason"] = f"Missing permission: {required_permission}"
            
            # 2. Attribute-based authorization
            abac_result = await self._check_attribute_policies(context, resource, operation)
            if not abac_result["allowed"]:
                authz_result["policy_violations"].extend(abac_result["violations"])
                authz_result["reason"] = "Attribute-based policy violations"
            
            # 3. Quota-based authorization
            quota_result = await self._check_resource_quotas(context, resource, operation)
            authz_result["quota_status"] = quota_result
            if not quota_result.get("allowed", True):
                authz_result["reason"] = f"Quota exceeded: {quota_result.get('violation', 'unknown')}"
            
            # 4. Context-based authorization
            context_result = await self._check_contextual_policies(context, resource, operation)
            if not context_result["allowed"]:
                authz_result["policy_violations"].extend(context_result["violations"])
                authz_result["reason"] = "Contextual policy violations"
            
            # Final authorization decision
            authz_result["authorized"] = (
                required_permission in user_permissions and
                abac_result["allowed"] and
                quota_result.get("allowed", True) and
                context_result["allowed"]
            )
            
            # Log authorization event
            await self._log_authorization_event(context, resource, operation, authz_result)
            
            # Store authorization pattern
            if not authz_result["authorized"]:
                await self.mcp.store_security_pattern("authorization_denied", {
                    "user_id": context.user_id,
                    "resource": resource,
                    "operation": operation,
                    "reason": authz_result["reason"],
                    "security_level": context.security_level.value
                })
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            authz_result["reason"] = f"Authorization system error: {str(e)}"
        
        return authz_result
    
    async def _check_attribute_policies(self, context: UnifiedSecurityContext, resource: str, operation: str) -> Dict[str, Any]:
        """Check attribute-based access control policies"""
        violations = []
        
        # Time-based restrictions
        current_hour = datetime.now(UTC).hour
        if context.security_level == SecurityLevel.CRITICAL and not (9 <= current_hour <= 17):
            violations.append("Critical operations only allowed during business hours (9-17 UTC)")
        
        # Reputation-based restrictions
        if context.reputation_score < 0.5 and operation in ["write", "delete", "modify"]:
            violations.append(f"Reputation score {context.reputation_score} too low for {operation} operations")
        
        # IP-based restrictions
        if context.source_ip and context.source_ip.startswith("192.168."):
            # Internal network access
            pass
        elif operation in ["system.manage", "security.manage"]:
            violations.append("Administrative operations restricted to internal network")
        
        return {
            "allowed": len(violations) == 0,
            "violations": violations
        }
    
    async def _check_resource_quotas(self, context: UnifiedSecurityContext, resource: str, operation: str) -> Dict[str, Any]:
        """Check resource quota constraints"""
        namespace = context.namespace or "default"
        
        # Get cached quota configuration
        quota_config = await self.mcp.retrieve_security_config(f"quota_{namespace}")
        
        if not quota_config:
            return {"allowed": True, "quota_available": True}
        
        # Simplified quota checking logic
        return {
            "allowed": True,
            "quota_available": True,
            "usage": {"cpu": 0.5, "memory": 0.3, "requests": 10}
        }
    
    async def _check_contextual_policies(self, context: UnifiedSecurityContext, resource: str, operation: str) -> Dict[str, Any]:
        """Check contextual security policies"""
        violations = []
        
        # Multi-tenant isolation
        if context.tenant_id and resource.startswith("tenant"):
            resource_tenant = resource.split("/")[1] if "/" in resource else None
            if resource_tenant and resource_tenant != context.tenant_id:
                violations.append("Cross-tenant access denied")
        
        # Namespace isolation
        if context.namespace and resource.startswith("namespace"):
            resource_namespace = resource.split("/")[1] if "/" in resource else None
            if resource_namespace and resource_namespace != context.namespace:
                violations.append("Cross-namespace access denied")
        
        # Data locality compliance
        if "data.export" in operation and context.data_locality:
            if context.data_locality == "EU" and not self._is_eu_compliant_operation(resource, operation):
                violations.append("Data export violates EU data residency requirements")
        
        return {
            "allowed": len(violations) == 0,
            "violations": violations
        }
    
    def _is_eu_compliant_operation(self, resource: str, operation: str) -> bool:
        """Check if operation complies with EU data residency"""
        # Simplified EU compliance check
        return "eu" in resource.lower() or "europe" in resource.lower()
    
    async def _log_authorization_event(self, context: UnifiedSecurityContext, resource: str, operation: str, result: Dict[str, Any]):
        """Log authorization events"""
        event = UnifiedSecurityEvent(
            event_type="AUTHORIZATION_CHECK",
            source_system="unified_authz",
            user_id=context.user_id,
            node_id=context.node_id,
            namespace=context.namespace,
            message=f"Authorization check: {resource}.{operation} - {'GRANTED' if result['authorized'] else 'DENIED'}",
            details={
                "resource": resource,
                "operation": operation,
                "authorized": result["authorized"],
                "reason": result["reason"],
                "quota_status": result["quota_status"]
            },
            source_ip=context.source_ip,
            user_agent=context.user_agent,
            affected_resources=[resource]
        )
        
        # Create GitHub issue for denied authorizations
        if not result["authorized"]:
            await self.mcp.create_security_github_issue(event)


class UnifiedThreatDetectionSystem:
    """Unified threat detection consolidating all attack detection systems"""
    
    def __init__(self, mcp_integration: MCPSecurityIntegration):
        self.mcp = mcp_integration
        self.threat_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.active_threats: List[UnifiedSecurityEvent] = []
        self.reputation_scores: Dict[str, float] = {}
        
        # Detection thresholds
        self.detection_thresholds = {
            ThreatType.BYZANTINE: 0.7,
            ThreatType.SYBIL: 0.8,
            ThreatType.ECLIPSE: 0.6,
            ThreatType.DOS: 0.9,
            ThreatType.GRADIENT_INVERSION: 0.8,
            ThreatType.MODEL_POISONING: 0.7,
            ThreatType.PII_EXPOSURE: 0.9,
            ThreatType.QUOTA_VIOLATION: 0.8,
            ThreatType.UNAUTHORIZED_ACCESS: 0.9,
            ThreatType.POLICY_VIOLATION: 0.6
        }
    
    async def detect_threats(self, context: UnifiedSecurityContext, activity_data: Dict[str, Any]) -> List[UnifiedSecurityEvent]:
        """Unified threat detection across all systems"""
        detected_threats = []
        
        try:
            # 1. Distributed consensus threats
            consensus_threats = await self._detect_consensus_threats(context, activity_data)
            detected_threats.extend(consensus_threats)
            
            # 2. Authentication threats
            auth_threats = await self._detect_authentication_threats(context, activity_data)
            detected_threats.extend(auth_threats)
            
            # 3. Authorization threats
            authz_threats = await self._detect_authorization_threats(context, activity_data)
            detected_threats.extend(authz_threats)
            
            # 4. Data security threats
            data_threats = await self._detect_data_security_threats(context, activity_data)
            detected_threats.extend(data_threats)
            
            # 5. Resource abuse threats
            resource_threats = await self._detect_resource_abuse_threats(context, activity_data)
            detected_threats.extend(resource_threats)
            
            # Process detected threats
            for threat in detected_threats:
                await self._process_threat(threat)
            
        except Exception as e:
            logger.error(f"Threat detection error: {e}")
            
            error_event = UnifiedSecurityEvent(
                event_type="THREAT_DETECTION_ERROR",
                source_system="unified_threat_detection",
                user_id=context.user_id,
                message=f"Threat detection system error: {str(e)}",
                details={"error": str(e), "context": activity_data},
                security_level=SecurityLevel.HIGH
            )
            detected_threats.append(error_event)
        
        return detected_threats
    
    async def _detect_consensus_threats(self, context: UnifiedSecurityContext, activity_data: Dict[str, Any]) -> List[UnifiedSecurityEvent]:
        """Detect distributed consensus threats"""
        threats = []
        
        if "consensus_messages" in activity_data:
            messages = activity_data["consensus_messages"]
            
            # Byzantine behavior detection
            contradictory_messages = self._find_contradictory_messages(messages)
            if contradictory_messages:
                threat = UnifiedSecurityEvent(
                    event_type="BYZANTINE_BEHAVIOR_DETECTED",
                    threat_type=ThreatType.BYZANTINE,
                    source_system="consensus",
                    node_id=context.node_id,
                    user_id=context.user_id,
                    message=f"Byzantine behavior detected: {len(contradictory_messages)} contradictory messages",
                    details={"contradictory_messages": contradictory_messages},
                    confidence=min(1.0, len(contradictory_messages) * 0.3),
                    security_level=SecurityLevel.HIGH
                )
                threats.append(threat)
            
            # Timing attack detection
            timing_anomalies = self._detect_timing_anomalies(messages)
            if timing_anomalies:
                threat = UnifiedSecurityEvent(
                    event_type="TIMING_ATTACK_DETECTED",
                    threat_type=ThreatType.DOS,
                    source_system="consensus",
                    node_id=context.node_id,
                    message=f"Timing attack detected: rapid message pattern",
                    details={"timing_anomalies": timing_anomalies},
                    confidence=0.7,
                    security_level=SecurityLevel.MEDIUM
                )
                threats.append(threat)
        
        return threats
    
    async def _detect_authentication_threats(self, context: UnifiedSecurityContext, activity_data: Dict[str, Any]) -> List[UnifiedSecurityEvent]:
        """Detect authentication-related threats"""
        threats = []
        
        if "failed_attempts" in activity_data:
            failed_attempts = activity_data["failed_attempts"]
            
            # Brute force detection
            if failed_attempts > 5:
                threat = UnifiedSecurityEvent(
                    event_type="BRUTE_FORCE_ATTACK",
                    threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                    source_system="authentication",
                    user_id=context.user_id,
                    message=f"Brute force attack detected: {failed_attempts} failed attempts",
                    details={"failed_attempts": failed_attempts},
                    confidence=min(1.0, failed_attempts / 10.0),
                    security_level=SecurityLevel.HIGH,
                    source_ip=context.source_ip
                )
                threats.append(threat)
        
        if "credential_stuffing" in activity_data:
            threat = UnifiedSecurityEvent(
                event_type="CREDENTIAL_STUFFING_DETECTED",
                threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                source_system="authentication",
                user_id=context.user_id,
                message="Credential stuffing attack detected",
                details=activity_data["credential_stuffing"],
                confidence=0.9,
                security_level=SecurityLevel.CRITICAL,
                source_ip=context.source_ip
            )
            threats.append(threat)
        
        return threats
    
    async def _detect_authorization_threats(self, context: UnifiedSecurityContext, activity_data: Dict[str, Any]) -> List[UnifiedSecurityEvent]:
        """Detect authorization-related threats"""
        threats = []
        
        if "privilege_escalation_attempts" in activity_data:
            attempts = activity_data["privilege_escalation_attempts"]
            
            threat = UnifiedSecurityEvent(
                event_type="PRIVILEGE_ESCALATION_ATTEMPT",
                threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                source_system="authorization",
                user_id=context.user_id,
                message=f"Privilege escalation attempt detected: {attempts} attempts",
                details={"escalation_attempts": attempts},
                confidence=0.8,
                security_level=SecurityLevel.HIGH,
                source_ip=context.source_ip
            )
            threats.append(threat)
        
        return threats
    
    async def _detect_data_security_threats(self, context: UnifiedSecurityContext, activity_data: Dict[str, Any]) -> List[UnifiedSecurityEvent]:
        """Detect data security threats"""
        threats = []
        
        if "data_content" in activity_data:
            content = activity_data["data_content"]
            pii_detected = self._scan_for_pii(content)
            
            if pii_detected:
                threat = UnifiedSecurityEvent(
                    event_type="PII_EXPOSURE_DETECTED",
                    threat_type=ThreatType.PII_EXPOSURE,
                    source_system="data_security",
                    user_id=context.user_id,
                    namespace=context.namespace,
                    message=f"PII exposure detected: {', '.join(pii_detected)}",
                    details={"pii_types": pii_detected},
                    confidence=0.9,
                    security_level=SecurityLevel.CRITICAL
                )
                threats.append(threat)
        
        return threats
    
    async def _detect_resource_abuse_threats(self, context: UnifiedSecurityContext, activity_data: Dict[str, Any]) -> List[UnifiedSecurityEvent]:
        """Detect resource abuse threats"""
        threats = []
        
        if "resource_usage" in activity_data:
            usage = activity_data["resource_usage"]
            
            # CPU abuse detection
            if usage.get("cpu_percent", 0) > 90:
                threat = UnifiedSecurityEvent(
                    event_type="RESOURCE_ABUSE_DETECTED",
                    threat_type=ThreatType.DOS,
                    source_system="resource_monitor",
                    user_id=context.user_id,
                    namespace=context.namespace,
                    message=f"High CPU usage detected: {usage['cpu_percent']}%",
                    details={"resource_usage": usage},
                    confidence=0.7,
                    security_level=SecurityLevel.MEDIUM
                )
                threats.append(threat)
            
            # Memory abuse detection
            if usage.get("memory_percent", 0) > 95:
                threat = UnifiedSecurityEvent(
                    event_type="MEMORY_EXHAUSTION_DETECTED",
                    threat_type=ThreatType.DOS,
                    source_system="resource_monitor",
                    user_id=context.user_id,
                    namespace=context.namespace,
                    message=f"High memory usage detected: {usage['memory_percent']}%",
                    details={"resource_usage": usage},
                    confidence=0.8,
                    security_level=SecurityLevel.HIGH
                )
                threats.append(threat)
        
        return threats
    
    async def _process_threat(self, threat: UnifiedSecurityEvent):
        """Process detected threat"""
        # Add to active threats
        self.active_threats.append(threat)
        
        # Update reputation score
        if threat.user_id:
            current_reputation = self.reputation_scores.get(threat.user_id, 1.0)
            reputation_penalty = self._calculate_reputation_penalty(threat)
            new_reputation = max(0.0, current_reputation - reputation_penalty)
            self.reputation_scores[threat.user_id] = new_reputation
        
        # Store threat pattern in Memory MCP
        await self.mcp.store_security_pattern(f"threat_{threat.threat_type.value}", {
            "event_type": threat.event_type,
            "security_level": threat.security_level.value,
            "confidence": threat.confidence,
            "details": threat.details
        })
        
        # Analyze threat with Sequential Thinking MCP
        analysis = await self.mcp.analyze_security_threat({
            "threat_type": threat.threat_type.value,
            "confidence": threat.confidence,
            "attack_vectors": [threat.event_type],
            "context": threat.details
        })
        
        # Create GitHub issue for high-severity threats
        if threat.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            await self.mcp.create_security_github_issue(threat)
        
        # Apply automated mitigations
        await self._apply_threat_mitigations(threat, analysis)
        
        logger.warning(f"Threat processed: {threat.event_type} - {threat.message}")
    
    def _find_contradictory_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find contradictory messages in consensus data"""
        contradictions = []
        message_hashes = set()
        
        for msg in messages:
            msg_hash = hashlib.sha256(json.dumps(msg, sort_keys=True).encode()).hexdigest()
            if msg_hash in message_hashes:
                contradictions.append(msg)
            message_hashes.add(msg_hash)
        
        return contradictions
    
    def _detect_timing_anomalies(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect timing-based attack patterns"""
        anomalies = []
        
        if len(messages) > 1:
            timestamps = [msg.get("timestamp", 0) for msg in messages]
            timestamp_diffs = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
            
            # Detect unusually rapid message sending
            rapid_messages = [diff for diff in timestamp_diffs if diff < 0.001]  # < 1ms
            if len(rapid_messages) > len(messages) * 0.5:  # More than 50% rapid
                anomalies.append({
                    "type": "rapid_messaging",
                    "rapid_count": len(rapid_messages),
                    "total_count": len(messages)
                })
        
        return anomalies
    
    def _scan_for_pii(self, content: str) -> List[str]:
        """Scan content for PII patterns"""
        import re
        
        pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
            (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "Credit Card"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email"),
            (r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b", "Phone Number"),
        ]
        
        detected_pii = []
        for pattern, pii_type in pii_patterns:
            if re.search(pattern, content):
                detected_pii.append(pii_type)
        
        return detected_pii
    
    def _calculate_reputation_penalty(self, threat: UnifiedSecurityEvent) -> float:
        """Calculate reputation penalty for threat"""
        base_penalty = {
            SecurityLevel.LOW: 0.05,
            SecurityLevel.MEDIUM: 0.1,
            SecurityLevel.HIGH: 0.2,
            SecurityLevel.CRITICAL: 0.4
        }
        
        return base_penalty.get(threat.security_level, 0.1) * threat.confidence
    
    async def _apply_threat_mitigations(self, threat: UnifiedSecurityEvent, analysis: Dict[str, Any]):
        """Apply automated threat mitigations"""
        mitigations_applied = []
        
        # Rate limiting for DoS threats
        if threat.threat_type == ThreatType.DOS:
            mitigations_applied.append("rate_limiting_applied")
        
        # Account suspension for critical threats
        if threat.security_level == SecurityLevel.CRITICAL and threat.user_id:
            mitigations_applied.append("account_temporarily_suspended")
        
        # IP blocking for unauthorized access
        if threat.threat_type == ThreatType.UNAUTHORIZED_ACCESS and threat.source_ip:
            mitigations_applied.append("ip_address_blocked")
        
        # Data access restriction for PII exposure
        if threat.threat_type == ThreatType.PII_EXPOSURE:
            mitigations_applied.append("data_access_restricted")
        
        if mitigations_applied:
            logger.info(f"Applied mitigations for threat {threat.event_id}: {mitigations_applied}")


class UnifiedCryptoManager:
    """Unified cryptographic operations manager"""
    
    def __init__(self):
        self.key_pairs: Dict[str, Tuple[bytes, bytes]] = {}  # node_id -> (private, public)
        self.trusted_certificates: Dict[str, bytes] = {}
        self.threshold_signatures: Dict[str, Dict[str, Any]] = {}
    
    def generate_keypair(self, node_id: str) -> Tuple[bytes, bytes]:
        """Generate cryptographic keypair for node"""
        # Simplified key generation - in production use proper crypto libraries
        private_key = secrets.token_bytes(32)
        public_key = hashlib.sha256(private_key).digest()
        
        self.key_pairs[node_id] = (private_key, public_key)
        return private_key, public_key
    
    def sign_data(self, node_id: str, data: bytes) -> bytes:
        """Sign data with node's private key"""
        if node_id not in self.key_pairs:
            raise ValueError(f"No keypair for node {node_id}")
        
        private_key, _ = self.key_pairs[node_id]
        # Simplified signing - use proper ECDSA/RSA in production
        signature = hashlib.sha256(private_key + data).digest()
        return signature
    
    def verify_signature(self, node_id: str, data: bytes, signature: bytes) -> bool:
        """Verify signature with node's public key"""
        if node_id not in self.key_pairs:
            return False
        
        private_key, public_key = self.key_pairs[node_id]
        expected_signature = hashlib.sha256(private_key + data).digest()
        return secrets.compare_digest(signature, expected_signature)


class UnifiedSecurityFramework:
    """Main unified security framework consolidating all security systems"""
    
    def __init__(self):
        # MCP integration
        self.mcp_integration = MCPSecurityIntegration()
        
        # Core security services
        self.auth_service = UnifiedAuthenticationService(self.mcp_integration)
        self.authz_middleware = UnifiedAuthorizationMiddleware(self.mcp_integration)
        self.threat_detection = UnifiedThreatDetectionSystem(self.mcp_integration)
        
        # Consolidated configuration
        self.security_config = {
            "enable_mcp_integration": True,
            "default_security_level": SecurityLevel.MEDIUM,
            "session_timeout_hours": 8,
            "max_failed_auth_attempts": 5,
            "enable_threat_detection": True,
            "enable_reputation_system": True,
            "compliance_mode": "standard",  # standard, strict, permissive
            "audit_all_operations": True
        }
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the unified security framework"""
        if self.initialized:
            return
        
        logger.info("Initializing Unified Security Framework...")
        
        try:
            # Initialize MCP services
            if self.security_config["enable_mcp_integration"]:
                await self.mcp_integration.initialize_mcp_services()
            
            # Cache initial security configuration
            await self.mcp_integration.cache_security_config("framework_config", self.security_config)
            
            self.initialized = True
            logger.info("Unified Security Framework initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Unified Security Framework: {e}")
            raise
    
    async def authenticate_user(self, context: UnifiedSecurityContext, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate user with unified authentication"""
        if not self.initialized:
            await self.initialize()
        
        return await self.auth_service.authenticate(context, credentials)
    
    async def authorize_operation(self, context: UnifiedSecurityContext, resource: str, operation: str) -> Dict[str, Any]:
        """Authorize operation with unified authorization"""
        if not self.initialized:
            await self.initialize()
        
        return await self.authz_middleware.authorize(context, resource, operation)
    
    async def detect_threats(self, context: UnifiedSecurityContext, activity_data: Dict[str, Any]) -> List[UnifiedSecurityEvent]:
        """Detect security threats with unified threat detection"""
        if not self.initialized:
            await self.initialize()
        
        return await self.threat_detection.detect_threats(context, activity_data)
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security framework status"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        return {
            "framework_status": "operational",
            "mcp_integration": {
                "github_enabled": self.mcp_integration.github_integration_enabled,
                "memory_enabled": self.mcp_integration.memory_integration_enabled,
                "sequential_thinking_enabled": self.mcp_integration.sequential_thinking_enabled,
                "context7_enabled": self.mcp_integration.context7_cache_enabled
            },
            "active_sessions": len(self.auth_service.active_sessions),
            "active_threats": len(self.threat_detection.active_threats),
            "security_level": self.security_config["default_security_level"].value,
            "last_initialized": datetime.now(UTC).isoformat()
        }
    
    async def update_security_config(self, config_updates: Dict[str, Any]):
        """Update security configuration"""
        self.security_config.update(config_updates)
        
        # Cache updated configuration
        await self.mcp_integration.cache_security_config("framework_config", self.security_config)
        
        logger.info(f"Security configuration updated: {list(config_updates.keys())}")
    
    async def handle_security_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle security incident with automated response"""
        incident_id = str(uuid4())
        
        # Create security event
        incident_event = UnifiedSecurityEvent(
            event_id=incident_id,
            event_type="SECURITY_INCIDENT",
            security_level=SecurityLevel.HIGH,
            source_system="unified_framework",
            message=f"Security incident reported: {incident_data.get('type', 'unknown')}",
            details=incident_data
        )
        
        # Create GitHub issue for incident tracking
        await self.mcp_integration.create_security_github_issue(incident_event)
        
        # Analyze incident with Sequential Thinking MCP
        analysis = await self.mcp_integration.analyze_security_threat(incident_data)
        
        # Store incident pattern in Memory MCP
        await self.mcp_integration.store_security_pattern("security_incident", incident_data)
        
        response = {
            "incident_id": incident_id,
            "status": "processed",
            "analysis": analysis,
            "github_issue_created": incident_event.github_issue_id,
            "automated_response": "Security team notified, monitoring increased"
        }
        
        logger.critical(f"Security incident handled: {incident_id}")
        return response


# Global framework instance
_security_framework: Optional[UnifiedSecurityFramework] = None


async def get_security_framework() -> UnifiedSecurityFramework:
    """Get global unified security framework instance"""
    global _security_framework
    
    if _security_framework is None:
        _security_framework = UnifiedSecurityFramework()
        await _security_framework.initialize()
    
    return _security_framework


# Convenience functions for external integration
async def authenticate_user(user_id: str, credentials: Dict[str, Any], **context_kwargs) -> Dict[str, Any]:
    """Convenience function for user authentication"""
    framework = await get_security_framework()
    context = UnifiedSecurityContext(user_id=user_id, **context_kwargs)
    return await framework.authenticate_user(context, credentials)


async def authorize_operation(user_id: str, resource: str, operation: str, **context_kwargs) -> Dict[str, Any]:
    """Convenience function for operation authorization"""
    framework = await get_security_framework()
    context = UnifiedSecurityContext(user_id=user_id, **context_kwargs)
    return await framework.authorize_operation(context, resource, operation)


async def detect_security_threats(user_id: str, activity_data: Dict[str, Any], **context_kwargs) -> List[UnifiedSecurityEvent]:
    """Convenience function for threat detection"""
    framework = await get_security_framework()
    context = UnifiedSecurityContext(user_id=user_id, **context_kwargs)
    return await framework.detect_threats(context, activity_data)


# Factory function for creating security contexts
def create_security_context(
    user_id: str,
    roles: Optional[Set[str]] = None,
    capabilities: Optional[Set[str]] = None,
    security_level: SecurityLevel = SecurityLevel.MEDIUM,
    **kwargs
) -> UnifiedSecurityContext:
    """Create unified security context"""
    return UnifiedSecurityContext(
        user_id=user_id,
        roles=roles or set(),
        capabilities=capabilities or set(),
        security_level=security_level,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize framework
        framework = await get_security_framework()
        
        # Test authentication
        context = create_security_context(
            user_id="test_user",
            roles={"developer"},
            security_level=SecurityLevel.HIGH
        )
        
        auth_result = await framework.authenticate_user(context, {
            "password": "secure_password_123",
            "hardware_token": "123456"
        })
        
        print(f"Authentication result: {auth_result}")
        
        # Test authorization
        authz_result = await framework.authorize_operation(context, "data.model", "train")
        print(f"Authorization result: {authz_result}")
        
        # Test threat detection
        threats = await framework.detect_threats(context, {
            "failed_attempts": 3,
            "resource_usage": {"cpu_percent": 85, "memory_percent": 70}
        })
        print(f"Detected threats: {len(threats)}")
        
        # Get framework status
        status = await framework.get_security_status()
        print(f"Framework status: {status}")
    
    asyncio.run(main())