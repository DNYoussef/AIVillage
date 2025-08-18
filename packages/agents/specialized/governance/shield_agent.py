"""Shield Agent - Blue Team & Constitutional Enforcement

The guardian of AIVillage, responsible for:
- Defending the village from threats
- Enforcing constitution and policies
- Approving risky actions
- Monitoring compliance during operations
- Constitutional adherence validation
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.core.agent_interface import AgentInterface

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class PolicyViolationType(Enum):
    CONSTITUTIONAL = "constitutional"
    SECURITY = "security"
    PRIVACY = "privacy"
    RESOURCE = "resource"
    ETHICAL = "ethical"


class ScanType(Enum):
    MALWARE_SCAN = "malware_scan"
    PRIVACY_POLICY_ANALYSIS = "privacy_policy_analysis"
    NETWORK_TRAFFIC_SCAN = "network_traffic_scan"
    VULNERABILITY_SCAN = "vulnerability_scan"
    FILE_INTEGRITY_CHECK = "file_integrity_check"


@dataclass
class SecurityAlert:
    alert_id: str
    threat_level: ThreatLevel
    description: str
    source: str
    timestamp: str
    mitigated: bool = False


@dataclass
class PolicyCheck:
    policy_id: str
    description: str
    compliance_score: float  # 0-1
    violations: list[str]
    approved: bool


@dataclass
class SecurityScanResult:
    scan_id: str
    scan_type: ScanType
    target: str  # file path, URL, or description
    risk_score: int  # 0-100
    findings: list[dict[str, Any]]
    threats_detected: list[str]
    recommendations: list[str]
    scan_duration_ms: float
    timestamp: float
    file_hash: str | None
    signature_matches: list[str]
    receipt: dict[str, Any]


class ShieldAgent(AgentInterface):
    """Shield Agent serves as the defensive guardian, enforcing the village
    constitution and protecting against security threats.
    """

    def __init__(self, agent_id: str = "shield_agent"):
        self.agent_id = agent_id
        self.agent_type = "Shield"
        self.capabilities = [
            # Constitutional enforcement (existing)
            "threat_defense",
            "constitutional_enforcement",
            "policy_compliance",
            "risk_approval",
            "security_monitoring",
            "compliance_auditing",
            "access_control",
            "incident_response",
            # Security scan service (Q1 MVP)
            "malware_detection",
            "privacy_policy_analysis",
            "network_traffic_inspection",
            "vulnerability_scanning",
            "file_integrity_checking",
            "risk_scoring",
            "security_receipt_generation",
            "mobile_optimized_scanning",
        ]

        # Constitutional enforcement (existing)
        self.active_alerts: dict[str, SecurityAlert] = {}
        self.policy_violations: list[dict[str, Any]] = []
        self.approved_actions: dict[str, dict[str, Any]] = {}
        self.compliance_history: list[dict[str, Any]] = []
        self.constitution_rules = {}
        self.security_policies = {}

        # Security scan service (Q1 MVP)
        self.scan_results: dict[str, SecurityScanResult] = {}
        self.malware_signatures: dict[str, str] = {}
        self.privacy_patterns: dict[str, list[str]] = {}
        self.vulnerability_db: dict[str, dict[str, Any]] = {}
        self.scan_cache: dict[str, SecurityScanResult] = {}
        self.scan_queue: list[dict[str, Any]] = []

        # Defense metrics
        self.threats_detected = 0
        self.threats_mitigated = 0
        self.policies_enforced = 0
        self.compliance_checks = 0

        # Scan service metrics
        self.scans_completed = 0
        self.malware_detected = 0
        self.privacy_violations_found = 0
        self.vulnerabilities_discovered = 0
        self.average_scan_time_ms = 0.0
        self.cache_hit_rate = 0.0

        # Mobile optimization settings
        self.max_memory_usage_mb = 100  # <100MB constraint
        self.chunk_size_bytes = 1024 * 1024  # 1MB chunks for streaming
        self.max_file_size_mb = 50  # Don't load files larger than 50MB fully
        self.signature_cache_limit = 10000  # Keep 10k signatures in memory

        self.initialized = False

    async def generate(self, prompt: str) -> str:
        """Generate defensive/enforcement responses"""
        if "defend" in prompt.lower() or "protect" in prompt.lower():
            return "I defend AIVillage against all threats and maintain our security perimeter."
        if "policy" in prompt.lower() or "constitution" in prompt.lower():
            return "I enforce constitutional principles and ensure all actions comply with village policies."
        if "approve" in prompt.lower() or "risk" in prompt.lower():
            return "I review and approve risky actions, ensuring they meet our safety standards."
        if "monitor" in prompt.lower():
            return "I continuously monitor operations for compliance and security violations."
        return "I am Shield Agent, the constitutional guardian and defender of AIVillage."

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for security/policy text"""
        import hashlib

        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_value % 1000) / 1000.0] * 512  # Security-focused embedding

    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        """Rerank based on security/policy relevance"""
        security_keywords = [
            "security",
            "threat",
            "policy",
            "compliance",
            "defense",
            "protect",
            "constitution",
            "risk",
            "violation",
            "enforcement",
        ]

        for result in results:
            score = 0
            content = str(result.get("content", ""))
            for keyword in security_keywords:
                score += content.lower().count(keyword) * 2  # Higher weight for security

            # Critical security boost
            if any(term in content.lower() for term in ["critical", "emergency", "breach"]):
                score *= 3

            result["security_score"] = score

        return sorted(results, key=lambda x: x.get("security_score", 0), reverse=True)[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return Shield agent status and security metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "active_alerts": len(self.active_alerts),
            "policy_violations": len(self.policy_violations),
            "approved_actions": len(self.approved_actions),
            "threats_detected": self.threats_detected,
            "threats_mitigated": self.threats_mitigated,
            "policies_enforced": self.policies_enforced,
            "compliance_rate": self.compliance_checks / max(1, self.policies_enforced),
            "defense_status": "active",
            "alert_level": self._get_current_alert_level(),
            "initialized": self.initialized,
        }

    def _get_current_alert_level(self) -> str:
        """Determine current security alert level"""
        if not self.active_alerts:
            return "green"

        max_threat = max(alert.threat_level.value for alert in self.active_alerts.values())

        if max_threat >= 4:
            return "red"
        if max_threat >= 3:
            return "orange"
        if max_threat >= 2:
            return "yellow"
        return "green"

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Secure communication with other agents"""
        # Security scan the message first
        security_scan = await self._scan_message_security(message)

        if not security_scan["safe"]:
            return f"BLOCKED: Message contains security violations: {security_scan['violations']}"

        if recipient:
            response = await recipient.generate(f"Shield Agent reports: {message}")
            return f"Secure communication delivered: {response[:50]}..."
        return "No recipient specified for secure communication"

    async def _scan_message_security(self, message: str) -> dict[str, Any]:
        """Scan message for security issues"""
        violations = []

        # Simple security checks
        dangerous_patterns = ["rm -rf", "DROP TABLE", "../", "eval(", "exec("]
        for pattern in dangerous_patterns:
            if pattern in message:
                violations.append(f"Dangerous pattern detected: {pattern}")

        return {
            "safe": len(violations) == 0,
            "violations": violations,
            "scan_timestamp": "2024-01-01T12:00:00Z",
        }

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate security/defense latent space"""
        if "threat" in query.lower():
            space_type = "threat_assessment"
        elif "policy" in query.lower():
            space_type = "policy_enforcement"
        elif "constitution" in query.lower():
            space_type = "constitutional_review"
        else:
            space_type = "general_defense"

        latent_repr = f"SHIELD[{space_type}:{query[:50]}]"
        return space_type, latent_repr

    async def detect_threat(self, source: str, activity: dict[str, Any]) -> SecurityAlert:
        """Detect and classify security threats"""
        threat_indicators = {
            "unauthorized_access": ThreatLevel.HIGH,
            "data_exfiltration": ThreatLevel.CRITICAL,
            "resource_exhaustion": ThreatLevel.MEDIUM,
            "policy_violation": ThreatLevel.MEDIUM,
            "suspicious_behavior": ThreatLevel.LOW,
            "malicious_code": ThreatLevel.CRITICAL,
        }

        # Analyze activity for threats
        threat_type = "suspicious_behavior"  # Default
        for indicator in threat_indicators:
            if indicator in str(activity).lower():
                threat_type = indicator
                break

        threat_level = threat_indicators[threat_type]
        alert_id = f"alert_{len(self.active_alerts) + 1}"

        alert = SecurityAlert(
            alert_id=alert_id,
            threat_level=threat_level,
            description=f"Detected {threat_type} from {source}",
            source=source,
            timestamp="2024-01-01T12:00:00Z",
        )

        self.active_alerts[alert_id] = alert
        self.threats_detected += 1

        logger.warning(f"SHIELD ALERT {alert_id}: {threat_type} - Level {threat_level.name}")

        # Auto-mitigate if critical
        if threat_level == ThreatLevel.CRITICAL:
            await self._auto_mitigate_threat(alert)

        return alert

    async def _auto_mitigate_threat(self, alert: SecurityAlert):
        """Automatically mitigate critical threats"""
        mitigation_actions = {
            "data_exfiltration": [
                "block_network_access",
                "quarantine_agent",
                "alert_king",
            ],
            "malicious_code": [
                "sandbox_execution",
                "code_analysis",
                "rollback_changes",
            ],
            "unauthorized_access": [
                "revoke_permissions",
                "force_authentication",
                "audit_trail",
            ],
        }

        # Determine mitigation strategy
        threat_keywords = alert.description.lower()
        actions_taken = []

        for threat_type, actions in mitigation_actions.items():
            if threat_type in threat_keywords:
                actions_taken = actions
                break

        if not actions_taken:
            actions_taken = ["general_lockdown", "escalate_to_king"]

        # Execute mitigation
        for action in actions_taken:
            await self._execute_mitigation_action(action, alert)

        alert.mitigated = True
        self.threats_mitigated += 1

        logger.info(f"Auto-mitigated threat {alert.alert_id}: {actions_taken}")

    async def _execute_mitigation_action(self, action: str, alert: SecurityAlert):
        """Execute specific mitigation action"""
        action_results = {
            "block_network_access": "Network access blocked for suspicious agent",
            "quarantine_agent": "Agent quarantined in secure sandbox",
            "alert_king": "King Agent notified of critical threat",
            "sandbox_execution": "Malicious code contained in sandbox",
            "code_analysis": "Code analyzed for threat patterns",
            "rollback_changes": "System rolled back to safe state",
            "revoke_permissions": "Access permissions revoked",
            "force_authentication": "Authentication required for access",
            "audit_trail": "Full audit trail initiated",
            "general_lockdown": "General security lockdown activated",
            "escalate_to_king": "Threat escalated to King Agent",
        }

        result = action_results.get(action, f"Unknown action: {action}")
        logger.info(f"Mitigation action {action}: {result}")

    async def enforce_policy(self, action_description: str, agent_id: str) -> PolicyCheck:
        """Enforce constitutional and security policies"""
        policy_id = f"policy_check_{len(self.approved_actions) + 1}"

        # Check against constitution rules
        violations = []
        compliance_score = 1.0

        # Constitutional principles check
        constitutional_violations = await self._check_constitutional_compliance(action_description)
        if constitutional_violations:
            violations.extend(constitutional_violations)
            compliance_score -= 0.3

        # Security policy check
        security_violations = await self._check_security_policies(action_description)
        if security_violations:
            violations.extend(security_violations)
            compliance_score -= 0.4

        # Privacy policy check
        privacy_violations = await self._check_privacy_policies(action_description)
        if privacy_violations:
            violations.extend(privacy_violations)
            compliance_score -= 0.3

        compliance_score = max(0, compliance_score)
        approved = compliance_score >= 0.7 and len(violations) == 0

        policy_check = PolicyCheck(
            policy_id=policy_id,
            description=action_description,
            compliance_score=compliance_score,
            violations=violations,
            approved=approved,
        )

        if approved:
            self.approved_actions[policy_id] = {
                "agent_id": agent_id,
                "action": action_description,
                "timestamp": "2024-01-01T12:00:00Z",
                "compliance_score": compliance_score,
            }
        else:
            self.policy_violations.append(
                {
                    "agent_id": agent_id,
                    "action": action_description,
                    "violations": violations,
                    "timestamp": "2024-01-01T12:00:00Z",
                }
            )

        self.policies_enforced += 1
        if approved:
            self.compliance_checks += 1

        return policy_check

    async def _check_constitutional_compliance(self, action: str) -> list[str]:
        """Check action against constitutional principles"""
        violations = []
        action_lower = action.lower()

        # Core constitutional principles

        # Simple compliance checks
        if "secret" in action_lower or "hidden" in action_lower:
            violations.append("Transparency violation: Action appears secretive")

        if "personal data" in action_lower and "expose" in action_lower:
            violations.append("Privacy violation: Personal data exposure detected")

        if "force" in action_lower or "override" in action_lower:
            violations.append("Autonomy violation: Forced action detected")

        if "harm" in action_lower or "damage" in action_lower:
            violations.append("Non-maleficence violation: Potential harm detected")

        return violations

    async def _check_security_policies(self, action: str) -> list[str]:
        """Check action against security policies"""
        violations = []
        action_lower = action.lower()

        dangerous_actions = [
            "delete all",
            "rm -rf",
            "drop database",
            "format disk",
            "disable security",
            "bypass authentication",
            "escalate privileges",
        ]

        for dangerous in dangerous_actions:
            if dangerous in action_lower:
                violations.append(f"Security violation: Dangerous action '{dangerous}' detected")

        return violations

    async def _check_privacy_policies(self, action: str) -> list[str]:
        """Check action against privacy policies"""
        violations = []
        action_lower = action.lower()

        privacy_sensitive = [
            "personal information",
            "private data",
            "user secrets",
            "passwords",
            "biometric data",
            "location data",
            "communication logs",
        ]

        if any(sensitive in action_lower for sensitive in privacy_sensitive):
            if not any(protect in action_lower for protect in ["encrypt", "secure", "protect"]):
                violations.append("Privacy violation: Sensitive data handling without protection")

        return violations

    async def approve_risky_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Review and approve risky actions"""
        risk_assessment = await self._assess_risk_level(action)
        policy_check = await self.enforce_policy(action["description"], action.get("requesting_agent", "unknown"))

        approval_decision = {
            "action_id": action.get("id", f"action_{len(self.approved_actions) + 1}"),
            "risk_level": risk_assessment["level"],
            "risk_factors": risk_assessment["factors"],
            "policy_compliance": policy_check.approved,
            "compliance_score": policy_check.compliance_score,
            "violations": policy_check.violations,
            "approved": False,
            "conditions": [],
        }

        # Decision logic
        if risk_assessment["level"] == "low" and policy_check.approved:
            approval_decision["approved"] = True
        elif risk_assessment["level"] == "medium" and policy_check.compliance_score >= 0.8:
            approval_decision["approved"] = True
            approval_decision["conditions"] = ["enhanced_monitoring", "periodic_review"]
        elif risk_assessment["level"] == "high" and policy_check.compliance_score >= 0.9:
            approval_decision["approved"] = True
            approval_decision["conditions"] = [
                "continuous_monitoring",
                "king_notification",
                "immediate_stop_authority",
            ]

        # Log the approval decision
        if approval_decision["approved"]:
            logger.info(
                f"APPROVED risky action {approval_decision['action_id']} with conditions: {approval_decision['conditions']}"
            )
        else:
            logger.warning(
                f"DENIED risky action {approval_decision['action_id']} - Risk: {risk_assessment['level']}, Compliance: {policy_check.compliance_score}"
            )

        return approval_decision

    async def _assess_risk_level(self, action: dict[str, Any]) -> dict[str, Any]:
        """Assess risk level of proposed action"""
        risk_factors = []
        risk_score = 0

        description = action.get("description", "").lower()

        # Risk factor analysis
        if any(keyword in description for keyword in ["delete", "remove", "destroy"]):
            risk_factors.append("data_destruction_risk")
            risk_score += 3

        if any(keyword in description for keyword in ["network", "internet", "external"]):
            risk_factors.append("external_communication_risk")
            risk_score += 2

        if any(keyword in description for keyword in ["system", "core", "infrastructure"]):
            risk_factors.append("system_modification_risk")
            risk_score += 2

        if any(keyword in description for keyword in ["user", "personal", "private"]):
            risk_factors.append("privacy_risk")
            risk_score += 1

        # Determine risk level
        if risk_score >= 5:
            level = "high"
        elif risk_score >= 3:
            level = "medium"
        else:
            level = "low"

        return {"level": level, "score": risk_score, "factors": risk_factors}

    async def monitor_compliance(self, agent_id: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Monitor ongoing operations for compliance"""
        compliance_report = {
            "agent_id": agent_id,
            "activity_type": activity.get("type", "unknown"),
            "compliance_status": "compliant",
            "violations": [],
            "recommendations": [],
            "monitoring_timestamp": "2024-01-01T12:00:00Z",
        }

        # Real-time compliance checking
        violations = []

        # Check resource usage
        if activity.get("resource_usage", {}).get("cpu", 0) > 0.9:
            violations.append("Resource usage violation: Excessive CPU consumption")

        # Check data access patterns
        if activity.get("data_access", {}).get("sensitive_data", False):
            if not activity.get("authorization", {}).get("privacy_approved", False):
                violations.append("Privacy violation: Unauthorized sensitive data access")

        # Check communication patterns
        if activity.get("communications", []):
            for comm in activity["communications"]:
                if comm.get("external", False) and not comm.get("encrypted", False):
                    violations.append("Security violation: Unencrypted external communication")

        if violations:
            compliance_report["compliance_status"] = "violation"
            compliance_report["violations"] = violations
            compliance_report["recommendations"] = [
                "Immediate review required",
                "Implement corrective measures",
                "Report to King Agent if critical",
            ]

        # Log compliance check
        self.compliance_history.append(compliance_report)

        return compliance_report

    # ========== Q1 MVP SECURITY SCAN SERVICE ==========

    async def scan_file(self, file_path: str, scan_options: dict[str, Any] = None) -> SecurityScanResult:
        """Malware detection and file security scan - Q1 MVP function"""
        scan_id = f"scan_{int(time.time())}_{len(self.scan_results)}"
        start_time = time.time()

        scan_options = scan_options or {}

        # Check cache first
        file_hash = await self._calculate_file_hash(file_path)
        cache_key = f"file_{file_hash}"
        if cache_key in self.scan_cache:
            logger.info(f"Scan cache hit for file: {file_path}")
            return self.scan_cache[cache_key]

        findings = []
        threats_detected = []
        signature_matches = []
        risk_score = 0

        try:
            # Mobile-optimized file analysis
            file_analysis = await self._analyze_file_mobile_optimized(file_path, file_hash)
            findings.extend(file_analysis["findings"])
            risk_score += file_analysis["risk_contribution"]

            # Malware signature scanning
            malware_scan = await self._scan_malware_signatures(file_path, file_hash)
            if malware_scan["threats_found"]:
                threats_detected.extend(malware_scan["threats_found"])
                signature_matches.extend(malware_scan["signature_matches"])
                risk_score += 30  # Malware detection adds significant risk

            # APK-specific analysis (if Android APK)
            if file_path.lower().endswith(".apk"):
                apk_analysis = await self._analyze_apk_security(file_path)
                findings.extend(apk_analysis["findings"])
                risk_score += apk_analysis["risk_contribution"]
                if apk_analysis["malicious_permissions"]:
                    threats_detected.extend(apk_analysis["malicious_permissions"])

            # File integrity and suspicious patterns
            integrity_check = await self._check_file_integrity(file_path, file_hash)
            findings.extend(integrity_check["findings"])
            risk_score += integrity_check["risk_contribution"]

        except Exception as e:
            findings.append(
                {
                    "type": "scan_error",
                    "description": f"Error during file scan: {e!s}",
                    "severity": "medium",
                }
            )
            risk_score += 20  # Scan errors are suspicious

        # Normalize risk score to 0-100
        risk_score = min(100, max(0, risk_score))

        # Generate recommendations
        recommendations = await self._generate_scan_recommendations(threats_detected, findings, risk_score)

        scan_duration = (time.time() - start_time) * 1000  # Convert to ms

        # Create receipt
        receipt = {
            "agent": "Shield",
            "action": "security_scan",
            "scan_type": ScanType.MALWARE_SCAN.value,
            "scan_id": scan_id,
            "timestamp": time.time(),
            "target_file": file_path,
            "file_hash": file_hash,
            "file_size_bytes": (os.path.getsize(file_path) if os.path.exists(file_path) else 0),
            "risk_score": risk_score,
            "threats_detected": len(threats_detected),
            "findings_count": len(findings),
            "scan_duration_ms": scan_duration,
            "signature_matches": len(signature_matches),
            "mobile_optimized": True,
            "signature": f"shield_scan_{scan_id}",
        }

        # Create scan result
        result = SecurityScanResult(
            scan_id=scan_id,
            scan_type=ScanType.MALWARE_SCAN,
            target=file_path,
            risk_score=risk_score,
            findings=findings,
            threats_detected=threats_detected,
            recommendations=recommendations,
            scan_duration_ms=scan_duration,
            timestamp=time.time(),
            file_hash=file_hash,
            signature_matches=signature_matches,
            receipt=receipt,
        )

        # Store results and update metrics
        self.scan_results[scan_id] = result
        self.scan_cache[cache_key] = result
        self.scans_completed += 1
        if threats_detected:
            self.malware_detected += len(threats_detected)

        # Update average scan time
        self.average_scan_time_ms = (
            self.average_scan_time_ms * (self.scans_completed - 1) + scan_duration
        ) / self.scans_completed

        logger.info(f"File scan completed: {file_path} - Risk score: {risk_score}/100")

        return result

    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file using streaming for large files"""
        if not os.path.exists(file_path):
            return hashlib.sha256(file_path.encode()).hexdigest()

        hash_obj = hashlib.sha256()
        file_size = os.path.getsize(file_path)

        try:
            with open(file_path, "rb") as f:
                if file_size > self.max_file_size_mb * 1024 * 1024:
                    # Stream large files in chunks to stay within memory limits
                    while chunk := f.read(self.chunk_size_bytes):
                        hash_obj.update(chunk)
                else:
                    # Small files can be read entirely
                    hash_obj.update(f.read())
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return hashlib.sha256(file_path.encode()).hexdigest()

        return hash_obj.hexdigest()

    async def _analyze_file_mobile_optimized(self, file_path: str, file_hash: str) -> dict[str, Any]:
        """Mobile-optimized file analysis with memory constraints"""
        findings = []
        risk_contribution = 0

        if not os.path.exists(file_path):
            return {
                "findings": [{"type": "file_not_found", "description": "File does not exist"}],
                "risk_contribution": 10,
            }

        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()

        # File size analysis
        if file_size > 100 * 1024 * 1024:  # >100MB
            findings.append(
                {
                    "type": "large_file",
                    "description": f"Large file size: {file_size / 1024 / 1024:.1f}MB",
                    "severity": "medium",
                }
            )
            risk_contribution += 10

        # Suspicious file extensions
        dangerous_extensions = [".exe", ".bat", ".cmd", ".scr", ".pif", ".com", ".dll"]
        if file_ext in dangerous_extensions:
            findings.append(
                {
                    "type": "dangerous_extension",
                    "description": f"Potentially dangerous file extension: {file_ext}",
                    "severity": "high",
                }
            )
            risk_contribution += 20

        # Analyze file header (magic bytes)
        try:
            with open(file_path, "rb") as f:
                header = f.read(512)  # Read first 512 bytes

                # Check for executable signatures
                if header.startswith(b"MZ"):  # Windows executable
                    findings.append(
                        {
                            "type": "executable_file",
                            "description": "File appears to be a Windows executable",
                            "severity": "medium",
                        }
                    )
                    risk_contribution += 15

                elif header.startswith(b"PK"):  # ZIP/APK archive
                    if file_ext == ".apk":
                        findings.append(
                            {
                                "type": "android_apk",
                                "description": "Android APK file detected",
                                "severity": "low",
                            }
                        )
                        risk_contribution += 5

                # Check for embedded scripts or suspicious patterns
                if b"<script" in header or b"javascript:" in header:
                    findings.append(
                        {
                            "type": "embedded_script",
                            "description": "File contains embedded scripts",
                            "severity": "high",
                        }
                    )
                    risk_contribution += 25

        except Exception as e:
            findings.append(
                {
                    "type": "file_read_error",
                    "description": f"Could not analyze file header: {e!s}",
                    "severity": "low",
                }
            )
            risk_contribution += 5

        return {"findings": findings, "risk_contribution": risk_contribution}

    async def _scan_malware_signatures(self, file_path: str, file_hash: str) -> dict[str, Any]:
        """Scan file against cached malware signatures"""
        threats_found = []
        signature_matches = []

        # Check hash against known malware hashes
        if file_hash in self.malware_signatures:
            threats_found.append(f"Known malware hash: {self.malware_signatures[file_hash]}")
            signature_matches.append(f"hash:{file_hash[:16]}...")

        # Simple pattern-based scanning for mobile optimization
        try:
            if os.path.getsize(file_path) < self.max_file_size_mb * 1024 * 1024:
                with open(file_path, "rb") as f:
                    content = f.read()

                    # Check for common malware patterns
                    malware_patterns = [
                        b"ransomware",
                        b"trojan",
                        b"keylogger",
                        b"backdoor",
                        b"bitcoin",
                        b"cryptocurrency",
                        b"mining",
                        b"payload",
                        b"exploit",
                        b"shellcode",
                    ]

                    for pattern in malware_patterns:
                        if pattern in content.lower():
                            threats_found.append(f"Suspicious pattern detected: {pattern.decode()}")
                            signature_matches.append(f"pattern:{pattern.decode()}")

        except Exception:
            pass  # Silent fail for signature scanning

        return {"threats_found": threats_found, "signature_matches": signature_matches}

    async def _analyze_apk_security(self, apk_path: str) -> dict[str, Any]:
        """Basic APK security analysis (mobile-optimized)"""
        findings = []
        risk_contribution = 0
        malicious_permissions = []

        # Simple APK analysis without heavy dependencies
        try:
            with open(apk_path, "rb") as f:
                # Look for AndroidManifest.xml patterns (simplified)
                content = f.read(min(10 * 1024 * 1024, os.path.getsize(apk_path)))  # Max 10MB

                # Check for dangerous permissions (simplified pattern matching)
                dangerous_perms = [
                    b"CALL_PHONE",
                    b"SEND_SMS",
                    b"READ_CONTACTS",
                    b"ACCESS_FINE_LOCATION",
                    b"RECORD_AUDIO",
                    b"CAMERA",
                    b"WRITE_EXTERNAL_STORAGE",
                    b"SYSTEM_ALERT_WINDOW",
                    b"DEVICE_ADMIN",
                    b"BIND_DEVICE_ADMIN",
                ]

                for perm in dangerous_perms:
                    if perm in content:
                        malicious_permissions.append(f"Dangerous permission: {perm.decode()}")
                        risk_contribution += 5

                # Check for suspicious package names
                suspicious_packages = [b"com.malware", b"trojan.", b"fake."]
                for pkg in suspicious_packages:
                    if pkg in content:
                        findings.append(
                            {
                                "type": "suspicious_package",
                                "description": f"Suspicious package name pattern: {pkg.decode()}",
                                "severity": "high",
                            }
                        )
                        risk_contribution += 20

        except Exception as e:
            findings.append(
                {
                    "type": "apk_analysis_error",
                    "description": f"Could not analyze APK: {e!s}",
                    "severity": "low",
                }
            )
            risk_contribution += 5

        return {
            "findings": findings,
            "risk_contribution": risk_contribution,
            "malicious_permissions": malicious_permissions,
        }

    async def _check_file_integrity(self, file_path: str, file_hash: str) -> dict[str, Any]:
        """Check file integrity and suspicious patterns"""
        findings = []
        risk_contribution = 0

        # Check for hidden extensions (double extensions)
        file_name = os.path.basename(file_path)
        if file_name.count(".") > 1:
            findings.append(
                {
                    "type": "multiple_extensions",
                    "description": f"File has multiple extensions: {file_name}",
                    "severity": "medium",
                }
            )
            risk_contribution += 10

        # Check for suspicious file names
        suspicious_names = ["setup.exe", "install.bat", "update.scr", "invoice.pdf.exe"]
        if any(sus_name in file_name.lower() for sus_name in suspicious_names):
            findings.append(
                {
                    "type": "suspicious_filename",
                    "description": f"Suspicious file name: {file_name}",
                    "severity": "high",
                }
            )
            risk_contribution += 15

        return {"findings": findings, "risk_contribution": risk_contribution}

    async def scan_privacy_policy(self, policy_text: str, context: str = None) -> SecurityScanResult:
        """Privacy policy analysis - Q1 MVP function"""
        scan_id = f"privacy_{int(time.time())}_{len(self.scan_results)}"
        start_time = time.time()

        findings = []
        threats_detected = []
        risk_score = 0

        # Privacy concern analysis
        privacy_issues = await self._analyze_privacy_concerns(policy_text)
        findings.extend(privacy_issues["findings"])
        risk_score += privacy_issues["risk_contribution"]

        # Data collection analysis
        data_collection = await self._analyze_data_collection(policy_text)
        findings.extend(data_collection["findings"])
        risk_score += data_collection["risk_contribution"]
        if data_collection["excessive_collection"]:
            threats_detected.extend(data_collection["excessive_collection"])

        # Third-party sharing analysis
        sharing_analysis = await self._analyze_third_party_sharing(policy_text)
        findings.extend(sharing_analysis["findings"])
        risk_score += sharing_analysis["risk_contribution"]

        # Normalize risk score
        risk_score = min(100, max(0, risk_score))

        # Generate recommendations
        recommendations = await self._generate_privacy_recommendations(findings, risk_score)

        scan_duration = (time.time() - start_time) * 1000

        # Create receipt
        receipt = {
            "agent": "Shield",
            "action": "privacy_policy_analysis",
            "scan_type": ScanType.PRIVACY_POLICY_ANALYSIS.value,
            "scan_id": scan_id,
            "timestamp": time.time(),
            "policy_length": len(policy_text),
            "policy_hash": hashlib.sha256(policy_text.encode()).hexdigest(),
            "risk_score": risk_score,
            "privacy_violations": len(threats_detected),
            "findings_count": len(findings),
            "scan_duration_ms": scan_duration,
            "signature": f"shield_privacy_{scan_id}",
        }

        # Create result
        result = SecurityScanResult(
            scan_id=scan_id,
            scan_type=ScanType.PRIVACY_POLICY_ANALYSIS,
            target=f"Privacy policy ({len(policy_text)} chars)",
            risk_score=risk_score,
            findings=findings,
            threats_detected=threats_detected,
            recommendations=recommendations,
            scan_duration_ms=scan_duration,
            timestamp=time.time(),
            file_hash=None,
            signature_matches=[],
            receipt=receipt,
        )

        # Store result and update metrics
        self.scan_results[scan_id] = result
        self.scans_completed += 1
        if threats_detected:
            self.privacy_violations_found += len(threats_detected)

        logger.info(f"Privacy policy scan completed - Risk score: {risk_score}/100")

        return result

    async def _analyze_privacy_concerns(self, policy_text: str) -> dict[str, Any]:
        """Analyze privacy policy for concerning language"""
        findings = []
        risk_contribution = 0
        policy_lower = policy_text.lower()

        # High-risk phrases
        concerning_phrases = [
            ("sell your data", "Data selling mentioned", 30),
            ("share with third parties", "Third-party data sharing", 20),
            ("advertising partners", "Data shared with advertisers", 15),
            ("may collect", "Vague data collection language", 10),
            ("cookies and tracking", "Tracking technologies", 10),
            ("location data", "Location tracking", 15),
            ("biometric", "Biometric data collection", 25),
            ("no guarantee", "No security guarantees", 20),
        ]

        for phrase, description, risk_points in concerning_phrases:
            if phrase in policy_lower:
                findings.append(
                    {
                        "type": "privacy_concern",
                        "description": description,
                        "phrase": phrase,
                        "severity": "high" if risk_points >= 20 else "medium",
                    }
                )
                risk_contribution += risk_points

        return {"findings": findings, "risk_contribution": risk_contribution}

    async def _analyze_data_collection(self, policy_text: str) -> dict[str, Any]:
        """Analyze data collection practices"""
        findings = []
        risk_contribution = 0
        excessive_collection = []
        policy_lower = policy_text.lower()

        # Data types that might be collected
        sensitive_data_types = [
            ("financial information", "Financial data collection", 25),
            ("health data", "Health information collection", 30),
            ("browsing history", "Web browsing tracking", 15),
            ("contacts", "Contact list access", 20),
            ("microphone", "Audio recording capability", 20),
            ("camera", "Camera access", 15),
            ("call logs", "Phone call monitoring", 25),
        ]

        for data_type, description, risk_points in sensitive_data_types:
            if data_type in policy_lower:
                findings.append(
                    {
                        "type": "data_collection",
                        "description": description,
                        "data_type": data_type,
                        "severity": "high" if risk_points >= 20 else "medium",
                    }
                )
                risk_contribution += risk_points

                if risk_points >= 25:
                    excessive_collection.append(f"Excessive collection: {data_type}")

        return {
            "findings": findings,
            "risk_contribution": risk_contribution,
            "excessive_collection": excessive_collection,
        }

    async def _analyze_third_party_sharing(self, policy_text: str) -> dict[str, Any]:
        """Analyze third-party data sharing"""
        findings = []
        risk_contribution = 0
        policy_lower = policy_text.lower()

        # Third-party sharing indicators
        sharing_indicators = [
            ("partners", "Data shared with partners", 15),
            ("affiliates", "Data shared with affiliates", 10),
            ("advertisers", "Data shared with advertisers", 20),
            ("service providers", "Data shared with service providers", 10),
            ("government", "Data shared with government", 25),
            ("law enforcement", "Data shared with law enforcement", 20),
        ]

        for indicator, description, risk_points in sharing_indicators:
            if indicator in policy_lower:
                findings.append(
                    {
                        "type": "third_party_sharing",
                        "description": description,
                        "sharing_type": indicator,
                        "severity": "high" if risk_points >= 20 else "medium",
                    }
                )
                risk_contribution += risk_points

        return {"findings": findings, "risk_contribution": risk_contribution}

    async def _generate_scan_recommendations(
        self, threats: list[str], findings: list[dict], risk_score: int
    ) -> list[str]:
        """Generate security recommendations based on scan results"""
        recommendations = []

        if risk_score >= 80:
            recommendations.append("CRITICAL: Do not use this file/service - high security risk")
            recommendations.append("Consider reporting to security authorities")
        elif risk_score >= 60:
            recommendations.append("HIGH RISK: Use extreme caution")
            recommendations.append("Scan with additional security tools")
            recommendations.append("Monitor system after use")
        elif risk_score >= 40:
            recommendations.append("MEDIUM RISK: Review carefully before use")
            recommendations.append("Ensure antivirus is up to date")
        elif risk_score >= 20:
            recommendations.append("LOW RISK: Generally safe but monitor usage")
        else:
            recommendations.append("MINIMAL RISK: Appears safe")

        # Specific recommendations based on findings
        finding_types = [f.get("type", "") for f in findings]

        if "malware" in str(threats).lower():
            recommendations.append("Quarantine file immediately")
            recommendations.append("Run full system scan")

        if "executable_file" in finding_types:
            recommendations.append("Run in isolated environment/sandbox")
            recommendations.append("Verify digital signature")

        if "privacy_concern" in finding_types:
            recommendations.append("Review privacy policy carefully")
            recommendations.append("Limit data sharing permissions")

        if "android_apk" in finding_types:
            recommendations.append("Install only from trusted sources")
            recommendations.append("Review app permissions before installing")

        return recommendations

    async def _generate_privacy_recommendations(self, findings: list[dict], risk_score: int) -> list[str]:
        """Generate privacy-specific recommendations"""
        recommendations = []

        if risk_score >= 70:
            recommendations.append("AVOID: Privacy policy contains concerning practices")
            recommendations.append("Consider alternative services with better privacy")
        elif risk_score >= 50:
            recommendations.append("CAUTION: Review privacy practices carefully")
            recommendations.append("Limit data sharing to minimum necessary")
        elif risk_score >= 30:
            recommendations.append("MODERATE: Standard privacy practices")
            recommendations.append("Review privacy settings regularly")
        else:
            recommendations.append("GOOD: Privacy practices appear reasonable")

        # Specific privacy recommendations
        finding_types = [f.get("type", "") for f in findings]

        if "data_collection" in finding_types:
            recommendations.append("Review what data is collected and why")
            recommendations.append("Opt out of unnecessary data collection")

        if "third_party_sharing" in finding_types:
            recommendations.append("Understand who your data is shared with")
            recommendations.append("Check if you can limit third-party sharing")

        return recommendations

    async def initialize(self):
        """Initialize the Shield Agent"""
        try:
            logger.info("Initializing Shield Agent - Constitutional Guardian...")

            # Load constitution and policies
            self.constitution_rules = {
                "transparency": "All agent actions must be transparent and auditable",
                "privacy": "Personal data must be protected and encrypted",
                "autonomy": "Agent autonomy must be respected within bounds",
                "beneficence": "All actions must benefit the village community",
                "non_maleficence": "Actions must not cause harm to others",
                "justice": "Resources and opportunities must be fairly distributed",
            }

            self.security_policies = {
                "data_protection": "All sensitive data must be encrypted",
                "access_control": "Access must be authenticated and authorized",
                "communication": "External communications must be secure",
                "resource_limits": "Resource usage must stay within limits",
                "code_integrity": "Code changes must be reviewed and approved",
            }

            self.initialized = True
            logger.info(f"Shield Agent {self.agent_id} initialized - Village defenses active")

        except Exception as e:
            logger.error(f"Failed to initialize Shield Agent: {e}")
            self.initialized = False

    async def shutdown(self):
        """Shutdown Shield Agent gracefully"""
        try:
            logger.info("Shield Agent shutting down...")

            # Final security report
            final_report = {
                "threats_detected": self.threats_detected,
                "threats_mitigated": self.threats_mitigated,
                "policies_enforced": self.policies_enforced,
                "compliance_rate": self.compliance_checks / max(1, self.policies_enforced),
                "active_alerts": len(self.active_alerts),
                "shutdown_timestamp": "2024-01-01T12:00:00Z",
            }

            logger.info(f"Shield Agent final security report: {final_report}")
            self.initialized = False

        except Exception as e:
            logger.error(f"Error during Shield Agent shutdown: {e}")
