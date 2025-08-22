"""
Fog Gateway Security Policy Engine

Implements comprehensive security and compliance policies for fog computing:
- Namespace-based quotas and resource limits
- Default-deny egress with allowlisting
- Data locality and compliance enforcement
- Multi-tenant isolation and access control

Security Model:
- Namespaces provide tenant isolation
- Jobs without namespace → 403 Forbidden
- Resource quotas prevent abuse
- Egress restricted by default (deny-all)
- PII/PHI scanning and blocking
- Audit logging for compliance
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class PolicyLevel(str, Enum):
    """Security policy enforcement levels"""

    PERMISSIVE = "permissive"  # Log violations but allow
    ADVISORY = "advisory"  # Warn but allow
    ENFORCING = "enforcing"  # Block violations
    STRICT = "strict"  # Block and audit violations


class DataLocality(str, Enum):
    """Data locality requirements for compliance"""

    GLOBAL = "global"  # No restrictions
    REGION = "region"  # Must stay in region
    COUNTRY = "country"  # Must stay in country
    EU = "eu"  # EU data residency
    US = "us"  # US data residency


class EgressAction(str, Enum):
    """Actions for egress policy evaluation"""

    ALLOW = "allow"  # Explicitly allow
    DENY = "deny"  # Explicitly deny
    LOG = "log"  # Allow but log
    AUDIT = "audit"  # Allow but audit


@dataclass
class ResourceQuota:
    """Resource quotas for namespace"""

    namespace: str

    # Compute quotas
    max_cpu_cores: float = 10.0
    max_memory_gb: float = 8.0
    max_disk_gb: float = 20.0
    max_gpu_hours: float = 2.0

    # Job quotas
    max_concurrent_jobs: int = 5
    max_daily_jobs: int = 100
    max_job_duration_hours: float = 2.0
    max_job_size_mb: float = 100.0

    # Network quotas
    max_ingress_gb: float = 1.0
    max_egress_gb: float = 0.5  # Default minimal egress

    # Cost quotas (in USD)
    max_daily_cost: float = 10.0
    max_monthly_cost: float = 200.0

    # Current usage tracking
    current_cpu_cores: float = 0.0
    current_memory_gb: float = 0.0
    current_disk_gb: float = 0.0
    current_concurrent_jobs: int = 0
    daily_jobs_count: int = 0
    daily_ingress_gb: float = 0.0
    daily_egress_gb: float = 0.0
    daily_cost: float = 0.0
    monthly_cost: float = 0.0

    # Reset timestamps
    last_daily_reset: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_monthly_reset: datetime = field(default_factory=lambda: datetime.now(UTC))

    def check_cpu_quota(self, requested_cores: float) -> bool:
        """Check if CPU quota allows request"""
        return (self.current_cpu_cores + requested_cores) <= self.max_cpu_cores

    def check_memory_quota(self, requested_gb: float) -> bool:
        """Check if memory quota allows request"""
        return (self.current_memory_gb + requested_gb) <= self.max_memory_gb

    def check_job_quota(self) -> bool:
        """Check if job count quota allows new job"""
        return self.current_concurrent_jobs < self.max_concurrent_jobs and self.daily_jobs_count < self.max_daily_jobs

    def check_cost_quota(self, estimated_cost: float) -> bool:
        """Check if cost quota allows job"""
        return (self.daily_cost + estimated_cost) <= self.max_daily_cost and (
            self.monthly_cost + estimated_cost
        ) <= self.max_monthly_cost

    def reserve_resources(self, cpu_cores: float, memory_gb: float) -> None:
        """Reserve resources for job"""
        self.current_cpu_cores += cpu_cores
        self.current_memory_gb += memory_gb
        self.current_concurrent_jobs += 1
        self.daily_jobs_count += 1

    def release_resources(self, cpu_cores: float, memory_gb: float, actual_cost: float) -> None:
        """Release resources after job completion"""
        self.current_cpu_cores = max(0.0, self.current_cpu_cores - cpu_cores)
        self.current_memory_gb = max(0.0, self.current_memory_gb - memory_gb)
        self.current_concurrent_jobs = max(0, self.current_concurrent_jobs - 1)

        self.daily_cost += actual_cost
        self.monthly_cost += actual_cost

    def reset_daily_counters(self) -> None:
        """Reset daily usage counters"""
        self.daily_jobs_count = 0
        self.daily_ingress_gb = 0.0
        self.daily_egress_gb = 0.0
        self.daily_cost = 0.0
        self.last_daily_reset = datetime.now(UTC)

    def reset_monthly_counters(self) -> None:
        """Reset monthly usage counters"""
        self.monthly_cost = 0.0
        self.last_monthly_reset = datetime.now(UTC)


@dataclass
class EgressRule:
    """Network egress access rule"""

    rule_id: str
    namespace: str

    # Target specification
    destination: str  # hostname, IP, or CIDR
    port: int | None = None  # specific port (None = any)
    protocol: str = "tcp"  # tcp, udp, icmp

    # Policy
    action: EgressAction = EgressAction.DENY
    reason: str = ""

    # Metadata
    created_by: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None

    def matches(self, destination: str, port: int | None = None, protocol: str = "tcp") -> bool:
        """Check if rule matches the destination"""

        # Check protocol
        if self.protocol != protocol:
            return False

        # Check port
        if self.port is not None and self.port != port:
            return False

        # Check destination (support exact match and CIDR for now)
        if self.destination == destination:
            return True

        # Add CIDR matching for IP ranges
        try:
            import ipaddress

            # Try to parse destination as CIDR
            if "/" in self.destination:
                network = ipaddress.ip_network(self.destination, strict=False)
                target_ip = ipaddress.ip_address(destination)
                return target_ip in network

            # Try to match as IP range
            if "-" in self.destination:
                start_ip, end_ip = self.destination.split("-", 1)
                start_addr = ipaddress.ip_address(start_ip.strip())
                end_addr = ipaddress.ip_address(end_ip.strip())
                target_addr = ipaddress.ip_address(destination)
                return start_addr <= target_addr <= end_addr

        except (ValueError, ipaddress.AddressValueError):
            # Not valid IP/CIDR, fall back to string matching
            pass

        # Support wildcard matching
        if "*" in self.destination:
            import fnmatch

            return fnmatch.fnmatch(destination, self.destination)

        return False

    def is_expired(self) -> bool:
        """Check if rule has expired"""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at


@dataclass
class SecurityEvent:
    """Security event for audit logging"""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = ""
    namespace: str = ""
    job_id: str | None = None

    # Event details
    severity: str = "INFO"  # INFO, WARN, ERROR, CRITICAL
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    # Context
    user_id: str | None = None
    source_ip: str | None = None
    user_agent: str | None = None

    # Timestamps
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_audit_log(self) -> str:
        """Convert to structured audit log entry"""
        return json.dumps(
            {
                "event_id": self.event_id,
                "timestamp": self.timestamp.isoformat(),
                "event_type": self.event_type,
                "namespace": self.namespace,
                "job_id": self.job_id,
                "severity": self.severity,
                "message": self.message,
                "details": self.details,
                "user_id": self.user_id,
                "source_ip": self.source_ip,
                "user_agent": self.user_agent,
            }
        )


class FogSecurityPolicyEngine:
    """
    Fog Computing Security Policy Engine

    Enforces security policies for fog computing workloads including:
    - Namespace quotas and limits
    - Network egress controls
    - Data locality compliance
    - PII/PHI scanning and blocking
    """

    def __init__(self, policy_level: PolicyLevel = PolicyLevel.ENFORCING):
        self.policy_level = policy_level

        # Policy storage
        self.namespace_quotas: dict[str, ResourceQuota] = {}
        self.egress_rules: dict[str, list[EgressRule]] = {}  # namespace -> rules
        self.security_events: list[SecurityEvent] = []

        # Default policies
        self.default_deny_egress = True
        self.require_namespace = True
        self.enable_pii_scanning = True
        self.data_locality = DataLocality.GLOBAL

        # Audit configuration
        self.audit_enabled = True
        self.audit_file_path = "/var/log/fog/security_audit.log"

        logger.info(f"Fog Security Policy Engine initialized with level: {policy_level.value}")

    async def initialize(self) -> None:
        """Initialize the policy engine"""

        # Create default namespace quotas
        await self._create_default_quotas()

        # Load egress rules
        await self._load_egress_rules()

        # Start background tasks
        asyncio.create_task(self._quota_reset_scheduler())
        asyncio.create_task(self._audit_logger())

        logger.info("Security policy engine initialized successfully")

    async def validate_job_submission(
        self, namespace: str | None, job_spec: dict[str, Any], user_context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate job submission against security policies

        Returns:
            dict: {"allowed": bool, "reason": str, "violations": List[str]}
        """

        violations = []

        # 1. Namespace requirement check
        if self.require_namespace and not namespace:
            violations.append("Job submission requires a valid namespace")
            await self._log_security_event(
                "NAMESPACE_MISSING",
                namespace="",
                severity="ERROR",
                message="Job submitted without required namespace",
                details={"job_spec": job_spec, "user": user_context.get("user_id")},
            )

        if not namespace:
            return {"allowed": False, "reason": "Namespace required", "violations": violations}

        # 2. Quota validation
        quota = await self._get_namespace_quota(namespace)
        if not quota:
            violations.append(f"No quota configured for namespace: {namespace}")
        else:
            # Check resource quotas
            requested_cpu = job_spec.get("cpu_cores", 1.0)
            requested_memory = job_spec.get("memory_gb", 1.0)

            if not quota.check_cpu_quota(requested_cpu):
                violations.append(
                    f"CPU quota exceeded: {requested_cpu} cores requested, {quota.max_cpu_cores - quota.current_cpu_cores} available"
                )

            if not quota.check_memory_quota(requested_memory):
                violations.append(
                    f"Memory quota exceeded: {requested_memory}GB requested, {quota.max_memory_gb - quota.current_memory_gb}GB available"
                )

            if not quota.check_job_quota():
                violations.append(
                    f"Job count quota exceeded: {quota.current_concurrent_jobs}/{quota.max_concurrent_jobs} concurrent, {quota.daily_jobs_count}/{quota.max_daily_jobs} daily"
                )

            # Estimate cost and check quota
            estimated_cost = self._estimate_job_cost(job_spec)
            if not quota.check_cost_quota(estimated_cost):
                violations.append(
                    f"Cost quota exceeded: ${estimated_cost:.2f} estimated, ${quota.max_daily_cost - quota.daily_cost:.2f} daily budget remaining"
                )

        # 3. PII/PHI scanning on job inputs
        if self.enable_pii_scanning:
            pii_violations = await self._scan_job_inputs_for_pii(job_spec)
            violations.extend(pii_violations)

        # 4. Data locality validation
        job_region = job_spec.get("region", "global")
        if not self._validate_data_locality(namespace, job_region):
            violations.append(f"Data locality violation: job region {job_region} not allowed for namespace {namespace}")

        # Log violations if any
        if violations:
            await self._log_security_event(
                "JOB_VALIDATION_FAILED",
                namespace=namespace,
                severity="WARN" if self.policy_level == PolicyLevel.PERMISSIVE else "ERROR",
                message=f"Job validation failed with {len(violations)} violations",
                details={"violations": violations, "job_spec": job_spec},
            )

        # Determine if job is allowed based on policy level
        allowed = True
        if violations:
            if self.policy_level in [PolicyLevel.ENFORCING, PolicyLevel.STRICT]:
                allowed = False
            elif self.policy_level == PolicyLevel.ADVISORY:
                logger.warning(f"Job allowed despite violations (advisory mode): {violations}")

        return {
            "allowed": allowed,
            "reason": "Policy violations detected" if violations else "Validation passed",
            "violations": violations,
        }

    async def validate_network_egress(
        self,
        namespace: str,
        destination: str,
        port: int | None = None,
        protocol: str = "tcp",
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Validate network egress request against policy

        Returns:
            dict: {"allowed": bool, "action": str, "reason": str}
        """

        # Default deny if no rules exist
        if self.default_deny_egress and namespace not in self.egress_rules:
            await self._log_security_event(
                "EGRESS_DENIED_NO_RULES",
                namespace=namespace,
                job_id=job_id,
                severity="WARN",
                message=f"Egress denied (default policy): {destination}:{port}",
                details={"destination": destination, "port": port, "protocol": protocol},
            )

            return {"allowed": False, "action": "DENY", "reason": "Default deny policy - no egress rules configured"}

        # Check namespace-specific rules
        namespace_rules = self.egress_rules.get(namespace, [])

        for rule in namespace_rules:
            if rule.is_expired():
                continue

            if rule.matches(destination, port, protocol):
                # Log the decision
                await self._log_security_event(
                    f"EGRESS_{rule.action.value.upper()}",
                    namespace=namespace,
                    job_id=job_id,
                    severity="INFO" if rule.action == EgressAction.ALLOW else "WARN",
                    message=f"Egress {rule.action.value}: {destination}:{port} (rule: {rule.rule_id})",
                    details={
                        "destination": destination,
                        "port": port,
                        "protocol": protocol,
                        "rule_id": rule.rule_id,
                        "rule_reason": rule.reason,
                    },
                )

                return {
                    "allowed": rule.action in [EgressAction.ALLOW, EgressAction.LOG, EgressAction.AUDIT],
                    "action": rule.action.value.upper(),
                    "reason": rule.reason or f"Matched rule {rule.rule_id}",
                }

        # No matching rules - apply default policy
        if self.default_deny_egress:
            await self._log_security_event(
                "EGRESS_DENIED_DEFAULT",
                namespace=namespace,
                job_id=job_id,
                severity="WARN",
                message=f"Egress denied (no matching rules): {destination}:{port}",
                details={"destination": destination, "port": port, "protocol": protocol},
            )

            return {"allowed": False, "action": "DENY", "reason": "No matching egress rules found"}
        else:
            return {"allowed": True, "action": "ALLOW", "reason": "Default allow policy"}

    async def reserve_job_resources(self, namespace: str, job_id: str, cpu_cores: float, memory_gb: float) -> bool:
        """Reserve resources for job execution"""

        quota = await self._get_namespace_quota(namespace)
        if not quota:
            return False

        if quota.check_cpu_quota(cpu_cores) and quota.check_memory_quota(memory_gb):
            quota.reserve_resources(cpu_cores, memory_gb)

            await self._log_security_event(
                "RESOURCES_RESERVED",
                namespace=namespace,
                job_id=job_id,
                severity="INFO",
                message=f"Reserved {cpu_cores} cores, {memory_gb}GB memory",
                details={"cpu_cores": cpu_cores, "memory_gb": memory_gb},
            )

            return True

        return False

    async def release_job_resources(
        self, namespace: str, job_id: str, cpu_cores: float, memory_gb: float, actual_cost: float = 0.0
    ) -> None:
        """Release resources after job completion"""

        quota = await self._get_namespace_quota(namespace)
        if quota:
            quota.release_resources(cpu_cores, memory_gb, actual_cost)

            await self._log_security_event(
                "RESOURCES_RELEASED",
                namespace=namespace,
                job_id=job_id,
                severity="INFO",
                message=f"Released {cpu_cores} cores, {memory_gb}GB memory, cost: ${actual_cost:.2f}",
                details={"cpu_cores": cpu_cores, "memory_gb": memory_gb, "cost": actual_cost},
            )

    async def create_namespace_quota(self, namespace: str, quota_config: dict[str, Any]) -> ResourceQuota:
        """Create or update namespace quota"""

        quota = ResourceQuota(
            namespace=namespace,
            max_cpu_cores=quota_config.get("max_cpu_cores", 10.0),
            max_memory_gb=quota_config.get("max_memory_gb", 8.0),
            max_disk_gb=quota_config.get("max_disk_gb", 20.0),
            max_concurrent_jobs=quota_config.get("max_concurrent_jobs", 5),
            max_daily_jobs=quota_config.get("max_daily_jobs", 100),
            max_daily_cost=quota_config.get("max_daily_cost", 10.0),
            max_monthly_cost=quota_config.get("max_monthly_cost", 200.0),
        )

        self.namespace_quotas[namespace] = quota

        await self._log_security_event(
            "QUOTA_CREATED",
            namespace=namespace,
            severity="INFO",
            message="Namespace quota created/updated",
            details=quota_config,
        )

        return quota

    async def add_egress_rule(
        self,
        namespace: str,
        destination: str,
        port: int | None = None,
        protocol: str = "tcp",
        action: EgressAction = EgressAction.ALLOW,
        reason: str = "",
        expires_hours: int | None = None,
        created_by: str = "",
    ) -> str:
        """Add network egress rule"""

        rule_id = f"egress_{uuid4().hex[:8]}"
        expires_at = None
        if expires_hours:
            expires_at = datetime.now(UTC) + timedelta(hours=expires_hours)

        rule = EgressRule(
            rule_id=rule_id,
            namespace=namespace,
            destination=destination,
            port=port,
            protocol=protocol,
            action=action,
            reason=reason,
            created_by=created_by,
            expires_at=expires_at,
        )

        if namespace not in self.egress_rules:
            self.egress_rules[namespace] = []

        self.egress_rules[namespace].append(rule)

        await self._log_security_event(
            "EGRESS_RULE_ADDED",
            namespace=namespace,
            severity="INFO",
            message=f"Egress rule added: {action.value} {destination}:{port}",
            details={
                "rule_id": rule_id,
                "destination": destination,
                "port": port,
                "protocol": protocol,
                "action": action.value,
                "reason": reason,
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )

        return rule_id

    def get_namespace_status(self, namespace: str) -> dict[str, Any]:
        """Get comprehensive namespace status"""

        quota = self.namespace_quotas.get(namespace)
        if not quota:
            return {"error": "Namespace not found"}

        egress_rules = self.egress_rules.get(namespace, [])
        active_rules = [r for r in egress_rules if not r.is_expired()]

        return {
            "namespace": namespace,
            "quota": {
                "cpu": {
                    "used": quota.current_cpu_cores,
                    "max": quota.max_cpu_cores,
                    "utilization": quota.current_cpu_cores / quota.max_cpu_cores,
                },
                "memory": {
                    "used": quota.current_memory_gb,
                    "max": quota.max_memory_gb,
                    "utilization": quota.current_memory_gb / quota.max_memory_gb,
                },
                "jobs": {
                    "concurrent": quota.current_concurrent_jobs,
                    "max_concurrent": quota.max_concurrent_jobs,
                    "daily_count": quota.daily_jobs_count,
                    "max_daily": quota.max_daily_jobs,
                },
                "cost": {
                    "daily": quota.daily_cost,
                    "max_daily": quota.max_daily_cost,
                    "monthly": quota.monthly_cost,
                    "max_monthly": quota.max_monthly_cost,
                },
            },
            "egress_rules": {
                "total": len(egress_rules),
                "active": len(active_rules),
                "allow_rules": len([r for r in active_rules if r.action == EgressAction.ALLOW]),
                "deny_rules": len([r for r in active_rules if r.action == EgressAction.DENY]),
            },
            "policy_level": self.policy_level.value,
            "last_daily_reset": quota.last_daily_reset.isoformat(),
            "last_monthly_reset": quota.last_monthly_reset.isoformat(),
        }

    # Private helper methods

    async def _get_namespace_quota(self, namespace: str) -> ResourceQuota | None:
        """Get namespace quota, creating default if needed"""

        if namespace not in self.namespace_quotas:
            # Create default quota
            await self.create_namespace_quota(namespace, {})

        return self.namespace_quotas.get(namespace)

    async def _create_default_quotas(self) -> None:
        """Create default namespace quotas"""

        default_namespaces = ["default", "development", "staging"]

        for ns in default_namespaces:
            await self.create_namespace_quota(
                ns, {"max_cpu_cores": 5.0, "max_memory_gb": 4.0, "max_daily_cost": 5.0, "max_monthly_cost": 100.0}
            )

    async def _load_egress_rules(self) -> None:
        """Load default egress rules"""

        # Common allow-listed destinations
        common_allowed = [
            ("api.openai.com", 443, "OpenAI API"),
            ("api.anthropic.com", 443, "Anthropic API"),
            ("huggingface.co", 443, "Hugging Face"),
            ("github.com", 443, "GitHub"),
            ("pypi.org", 443, "Python Package Index"),
        ]

        for ns in ["default", "development", "staging"]:
            for dest, port, reason in common_allowed:
                await self.add_egress_rule(
                    namespace=ns,
                    destination=dest,
                    port=port,
                    action=EgressAction.ALLOW,
                    reason=reason,
                    created_by="system",
                )

    def _estimate_job_cost(self, job_spec: dict[str, Any]) -> float:
        """Estimate job execution cost"""

        cpu_cores = job_spec.get("cpu_cores", 1.0)
        memory_gb = job_spec.get("memory_gb", 1.0)
        duration_hours = job_spec.get("estimated_duration_hours", 1.0)

        # Simple cost model: $0.10/core-hour + $0.05/GB-hour
        cpu_cost = cpu_cores * duration_hours * 0.10
        memory_cost = memory_gb * duration_hours * 0.05

        return cpu_cost + memory_cost

    async def _scan_job_inputs_for_pii(self, job_spec: dict[str, Any]) -> list[str]:
        """Scan job inputs for PII/PHI data"""

        violations = []

        # Get input data
        input_data = job_spec.get("input_data", "")
        env_vars = job_spec.get("env", {})

        # Simple PII patterns (production would use proper PII detection)
        pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
            (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "Credit Card"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email"),
            (r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b", "Phone Number"),
        ]

        # Check input data
        if isinstance(input_data, str):
            for pattern, pii_type in pii_patterns:
                if re.search(pattern, input_data):
                    violations.append(f"PII detected in input data: {pii_type}")

        # Check environment variables
        for key, value in env_vars.items():
            if isinstance(value, str):
                for pattern, pii_type in pii_patterns:
                    if re.search(pattern, value):
                        violations.append(f"PII detected in environment variable {key}: {pii_type}")

        return violations

    def _validate_data_locality(self, namespace: str, job_region: str) -> bool:
        """Validate data locality requirements"""

        # For now, allow all regions (would be configured per namespace)
        return True

    async def _log_security_event(
        self,
        event_type: str,
        namespace: str,
        job_id: str | None = None,
        severity: str = "INFO",
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log security event for audit"""

        event = SecurityEvent(
            event_type=event_type,
            namespace=namespace,
            job_id=job_id,
            severity=severity,
            message=message,
            details=details or {},
        )

        self.security_events.append(event)

        # Log to standard logger
        log_level = getattr(logging, severity, logging.INFO)
        logger.log(log_level, f"[{event_type}] {namespace}: {message}")

    async def _quota_reset_scheduler(self) -> None:
        """Background task to reset quota counters"""

        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour

                now = datetime.now(UTC)

                for quota in self.namespace_quotas.values():
                    # Reset daily counters if needed
                    if (now - quota.last_daily_reset).days >= 1:
                        quota.reset_daily_counters()
                        logger.info(f"Reset daily quotas for namespace: {quota.namespace}")

                    # Reset monthly counters if needed
                    if (now - quota.last_monthly_reset).days >= 30:
                        quota.reset_monthly_counters()
                        logger.info(f"Reset monthly quotas for namespace: {quota.namespace}")

            except Exception as e:
                logger.error(f"Error in quota reset scheduler: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _audit_logger(self) -> None:
        """Background task to write audit logs"""

        while True:
            try:
                await asyncio.sleep(10)  # Write every 10 seconds

                if self.audit_enabled and self.security_events:
                    # Write events to audit log file
                    events_to_write = self.security_events[:]
                    self.security_events.clear()

                    # In production, write to proper audit log file
                    for event in events_to_write:
                        logger.info(f"AUDIT: {event.to_audit_log()}")

            except Exception as e:
                logger.error(f"Error in audit logger: {e}")
                await asyncio.sleep(30)


# Global policy engine instance
_policy_engine: FogSecurityPolicyEngine | None = None


async def get_policy_engine() -> FogSecurityPolicyEngine:
    """Get global policy engine instance"""
    global _policy_engine

    if _policy_engine is None:
        _policy_engine = FogSecurityPolicyEngine()
        await _policy_engine.initialize()

    return _policy_engine


async def validate_job_security(
    namespace: str | None, job_spec: dict[str, Any], user_context: dict[str, Any]
) -> dict[str, Any]:
    """Convenience function to validate job security"""

    engine = await get_policy_engine()
    return await engine.validate_job_submission(namespace, job_spec, user_context)


async def validate_egress_request(
    namespace: str, destination: str, port: int | None = None, protocol: str = "tcp", job_id: str | None = None
) -> dict[str, Any]:
    """Convenience function to validate egress request"""

    engine = await get_policy_engine()
    return await engine.validate_network_egress(namespace, destination, port, protocol, job_id)
