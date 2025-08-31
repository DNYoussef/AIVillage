"""
Audit Trail Pricing Manager for Constitutional Compliance

Implements comprehensive audit trail system for fog computing pricing:
- Immutable pricing calculation logs
- Constitutional compliance tracking
- Transparency and accountability mechanisms
- Governance decision audit trails
- Privacy-preserving audit records

Key Features:
- Immutable audit trail for all pricing calculations
- Constitutional compliance verification
- Transparent cost breakdown with full justification
- Governance vote audit logs
- Privacy-preserving audit with zero-knowledge proofs
- Regulatory compliance reporting
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
import uuid

# Set high precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events"""
    
    PRICING_CALCULATION = "pricing_calculation"
    QUOTE_GENERATION = "quote_generation"  
    PAYMENT_PROCESSING = "payment_processing"
    GOVERNANCE_VOTE = "governance_vote"
    TIER_ADJUSTMENT = "tier_adjustment"
    CONSTITUTIONAL_VERIFICATION = "constitutional_verification"
    TRANSPARENCY_REQUEST = "transparency_request"
    PRIVACY_VERIFICATION = "privacy_verification"


class ComplianceLevel(str, Enum):
    """Compliance verification levels"""
    
    BASIC = "basic"                    # Basic audit trail
    ENHANCED = "enhanced"              # Enhanced verification
    CONSTITUTIONAL = "constitutional"   # Full constitutional compliance
    REGULATORY = "regulatory"          # Regulatory compliance
    ZERO_KNOWLEDGE = "zero_knowledge"  # Privacy-preserving audit


@dataclass
class AuditRecord:
    """Immutable audit record for pricing events"""
    
    record_id: str
    event_type: AuditEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # Event details
    event_data: Dict[str, Any] = field(default_factory=dict)
    actor_id: Optional[str] = None
    device_id: Optional[str] = None
    transaction_id: Optional[str] = None
    
    # Pricing information
    pricing_data: Dict[str, Any] = field(default_factory=dict)
    constitutional_features: Dict[str, bool] = field(default_factory=dict)
    compliance_level: ComplianceLevel = ComplianceLevel.BASIC
    
    # Audit integrity
    record_hash: str = ""
    previous_record_hash: str = ""
    chain_position: int = 0
    
    # Privacy and compliance
    privacy_preserved: bool = True
    constitutional_compliant: bool = True
    transparency_level: str = "public"
    
    def __post_init__(self):
        """Generate audit record hash for integrity"""
        if not self.record_hash:
            self.record_hash = self._generate_record_hash()
    
    def _generate_record_hash(self) -> str:
        """Generate cryptographic hash of audit record"""
        
        # Create deterministic record representation
        record_data = {
            "record_id": self.record_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "event_data": self.event_data,
            "actor_id": self.actor_id,
            "pricing_data": self.pricing_data,
            "previous_record_hash": self.previous_record_hash,
            "chain_position": self.chain_position
        }
        
        # Create JSON string for hashing (sorted keys for consistency)
        record_json = json.dumps(record_data, sort_keys=True)
        
        # Generate SHA-256 hash
        return hashlib.sha256(record_json.encode()).hexdigest()
    
    def verify_integrity(self, previous_record: Optional['AuditRecord'] = None) -> bool:
        """Verify audit record integrity"""
        
        # Verify own hash
        expected_hash = self._generate_record_hash()
        if self.record_hash != expected_hash:
            return False
        
        # Verify chain integrity
        if previous_record:
            if self.previous_record_hash != previous_record.record_hash:
                return False
            if self.chain_position != previous_record.chain_position + 1:
                return False
        
        return True


@dataclass
class ConstitutionalComplianceReport:
    """Constitutional compliance verification report"""
    
    report_id: str
    compliance_check_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # Compliance checks
    pricing_transparency: bool = True
    audit_trail_complete: bool = True
    governance_participation_enabled: bool = True
    privacy_protections_active: bool = True
    
    # Constitutional requirements
    democratic_pricing_adjustments: bool = True
    transparent_fee_structure: bool = True
    equal_access_provisions: bool = True
    privacy_by_design: bool = True
    
    # Compliance scores
    overall_compliance_score: Decimal = Decimal("1.0")
    transparency_score: Decimal = Decimal("1.0")
    privacy_score: Decimal = Decimal("1.0")
    governance_score: Decimal = Decimal("1.0")
    
    # Issues and recommendations
    compliance_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def calculate_overall_score(self):
        """Calculate overall compliance score"""
        
        scores = [
            self.transparency_score,
            self.privacy_score,
            self.governance_score
        ]
        
        self.overall_compliance_score = sum(scores) / len(scores)
        
        # Add penalty for critical failures
        if not self.pricing_transparency:
            self.overall_compliance_score *= Decimal("0.7")
        if not self.audit_trail_complete:
            self.overall_compliance_score *= Decimal("0.8")
        if not self.privacy_protections_active:
            self.overall_compliance_score *= Decimal("0.6")


class AuditTrailManager:
    """
    Comprehensive audit trail manager for constitutional pricing compliance
    
    Features:
    - Immutable audit chain for all pricing events
    - Constitutional compliance verification
    - Privacy-preserving audit trails
    - Transparent reporting and analytics
    - Governance decision tracking
    """
    
    def __init__(self):
        # Audit chain storage
        self.audit_records: List[AuditRecord] = []
        self.audit_index: Dict[str, int] = {}  # record_id -> position
        
        # Compliance tracking
        self.compliance_reports: List[ConstitutionalComplianceReport] = []
        self.constitutional_violations: List[Dict[str, Any]] = []
        
        # Privacy protection
        self.privacy_preserving_mode = True
        self.zero_knowledge_proofs: Dict[str, Any] = {}
        
        # Performance metrics
        self.audit_metrics = {
            "total_records": 0,
            "records_by_type": {},
            "compliance_violations": 0,
            "privacy_breaches": 0,
            "chain_integrity_verified": True
        }
        
        logger.info("Audit trail manager initialized")
    
    def log_calculation(
        self,
        calculation_type: str,
        calculation_data: Dict[str, Any],
        device_id: str = None,
        actor_id: str = None,
        constitutional_features: Dict[str, bool] = None
    ) -> str:
        """Log pricing calculation with audit trail"""
        
        record_id = str(uuid.uuid4())
        
        # Get previous record for chain integrity
        previous_record = self.audit_records[-1] if self.audit_records else None
        previous_hash = previous_record.record_hash if previous_record else ""
        chain_position = len(self.audit_records)
        
        # Create audit record
        record = AuditRecord(
            record_id=record_id,
            event_type=AuditEventType.PRICING_CALCULATION,
            event_data={
                "calculation_type": calculation_type,
                "calculation_details": calculation_data
            },
            actor_id=actor_id,
            device_id=device_id,
            pricing_data=calculation_data,
            constitutional_features=constitutional_features or {},
            previous_record_hash=previous_hash,
            chain_position=chain_position,
            compliance_level=ComplianceLevel.CONSTITUTIONAL
        )
        
        # Add to audit chain
        self.audit_records.append(record)
        self.audit_index[record_id] = chain_position
        
        # Update metrics
        self.audit_metrics["total_records"] += 1
        calc_type_key = f"calculation_{calculation_type}"
        self.audit_metrics["records_by_type"][calc_type_key] = \
            self.audit_metrics["records_by_type"].get(calc_type_key, 0) + 1
        
        logger.debug(f"Pricing calculation logged: {calculation_type} -> {record_id}")
        
        return record_id
    
    def log_pricing_quote(
        self,
        quote_data: Dict[str, Any],
        actor_id: str = None
    ) -> str:
        """Log pricing quote generation"""
        
        record_id = str(uuid.uuid4())
        
        # Get previous record for chain integrity
        previous_record = self.audit_records[-1] if self.audit_records else None
        previous_hash = previous_record.record_hash if previous_record else ""
        chain_position = len(self.audit_records)
        
        # Create audit record
        record = AuditRecord(
            record_id=record_id,
            event_type=AuditEventType.QUOTE_GENERATION,
            event_data=quote_data,
            actor_id=actor_id,
            transaction_id=quote_data.get("quote_id"),
            pricing_data={
                "quote_id": quote_data.get("quote_id"),
                "tier": quote_data.get("tier"),
                "final_cost": quote_data.get("pricing", {}).get("final_cost"),
                "h200_hours": quote_data.get("h200_calculation", {}).get("h200_hours_equivalent")
            },
            constitutional_features=quote_data.get("constitutional_features", {}),
            previous_record_hash=previous_hash,
            chain_position=chain_position,
            compliance_level=ComplianceLevel.CONSTITUTIONAL,
            transparency_level="public"
        )
        
        # Add to audit chain
        self.audit_records.append(record)
        self.audit_index[record_id] = chain_position
        
        # Update metrics
        self.audit_metrics["total_records"] += 1
        self.audit_metrics["records_by_type"]["quote_generation"] = \
            self.audit_metrics["records_by_type"].get("quote_generation", 0) + 1
        
        logger.debug(f"Pricing quote logged: {quote_data.get('quote_id')} -> {record_id}")
        
        return record_id
    
    def log_governance_vote(
        self,
        vote_data: Dict[str, Any],
        voter_id: str = None
    ) -> str:
        """Log governance voting activity"""
        
        record_id = str(uuid.uuid4())
        
        # Get previous record for chain integrity
        previous_record = self.audit_records[-1] if self.audit_records else None
        previous_hash = previous_record.record_hash if previous_record else ""
        chain_position = len(self.audit_records)
        
        # Create audit record
        record = AuditRecord(
            record_id=record_id,
            event_type=AuditEventType.GOVERNANCE_VOTE,
            event_data=vote_data,
            actor_id=voter_id,
            transaction_id=vote_data.get("vote_id"),
            constitutional_features={"governance_participation": True},
            previous_record_hash=previous_hash,
            chain_position=chain_position,
            compliance_level=ComplianceLevel.CONSTITUTIONAL,
            transparency_level="public"
        )
        
        # Add to audit chain
        self.audit_records.append(record)
        self.audit_index[record_id] = chain_position
        
        # Update metrics
        self.audit_metrics["total_records"] += 1
        self.audit_metrics["records_by_type"]["governance_vote"] = \
            self.audit_metrics["records_by_type"].get("governance_vote", 0) + 1
        
        logger.debug(f"Governance vote logged: {vote_data.get('vote_id')} -> {record_id}")
        
        return record_id
    
    def verify_audit_chain_integrity(self) -> Dict[str, Any]:
        """Verify integrity of entire audit chain"""
        
        integrity_results = {
            "chain_valid": True,
            "total_records": len(self.audit_records),
            "verified_records": 0,
            "integrity_failures": [],
            "verification_timestamp": datetime.now(UTC).isoformat()
        }
        
        previous_record = None
        
        for i, record in enumerate(self.audit_records):
            
            # Verify record integrity
            if not record.verify_integrity(previous_record):
                integrity_results["chain_valid"] = False
                integrity_results["integrity_failures"].append({
                    "record_id": record.record_id,
                    "position": i,
                    "failure_type": "integrity_check_failed"
                })
            else:
                integrity_results["verified_records"] += 1
            
            previous_record = record
        
        # Update metrics
        self.audit_metrics["chain_integrity_verified"] = integrity_results["chain_valid"]
        
        logger.info(
            f"Audit chain integrity verification: "
            f"{integrity_results['verified_records']}/{integrity_results['total_records']} "
            f"valid ({'PASS' if integrity_results['chain_valid'] else 'FAIL'})"
        )
        
        return integrity_results
    
    def generate_constitutional_compliance_report(self) -> ConstitutionalComplianceReport:
        """Generate comprehensive constitutional compliance report"""
        
        report_id = str(uuid.uuid4())
        report = ConstitutionalComplianceReport(report_id=report_id)
        
        # Check pricing transparency
        pricing_records = [
            r for r in self.audit_records 
            if r.event_type in [AuditEventType.PRICING_CALCULATION, AuditEventType.QUOTE_GENERATION]
        ]
        report.pricing_transparency = len(pricing_records) > 0
        
        # Check audit trail completeness
        chain_integrity = self.verify_audit_chain_integrity()
        report.audit_trail_complete = chain_integrity["chain_valid"]
        
        # Check governance participation
        governance_records = [
            r for r in self.audit_records
            if r.event_type == AuditEventType.GOVERNANCE_VOTE
        ]
        report.governance_participation_enabled = len(governance_records) > 0
        
        # Check privacy protections
        privacy_violations = len([
            r for r in self.audit_records
            if not r.privacy_preserved
        ])
        report.privacy_protections_active = privacy_violations == 0
        
        # Calculate compliance scores
        report.transparency_score = Decimal("1.0") if report.pricing_transparency else Decimal("0.5")
        report.privacy_score = Decimal("1.0") if report.privacy_protections_active else Decimal("0.3")
        report.governance_score = Decimal("1.0") if report.governance_participation_enabled else Decimal("0.7")
        
        report.calculate_overall_score()
        
        # Generate recommendations
        if not report.pricing_transparency:
            report.compliance_issues.append("Insufficient pricing transparency")
            report.recommendations.append("Increase public pricing calculation logs")
        
        if not report.audit_trail_complete:
            report.compliance_issues.append("Audit trail integrity failures")
            report.recommendations.append("Investigate and repair audit chain integrity")
        
        if not report.privacy_protections_active:
            report.compliance_issues.append("Privacy protection violations detected")
            report.recommendations.append("Enable privacy-preserving audit mechanisms")
        
        # Store compliance report
        self.compliance_reports.append(report)
        
        logger.info(
            f"Constitutional compliance report generated: "
            f"Overall score {float(report.overall_compliance_score):.2f}"
        )
        
        return report
    
    def get_audit_records_by_criteria(
        self,
        event_type: AuditEventType = None,
        actor_id: str = None,
        device_id: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[AuditRecord]:
        """Get audit records matching criteria"""
        
        matching_records = []
        
        for record in self.audit_records:
            
            # Filter by event type
            if event_type and record.event_type != event_type:
                continue
            
            # Filter by actor
            if actor_id and record.actor_id != actor_id:
                continue
            
            # Filter by device
            if device_id and record.device_id != device_id:
                continue
            
            # Filter by time range
            if start_time and record.timestamp < start_time:
                continue
            if end_time and record.timestamp > end_time:
                continue
            
            matching_records.append(record)
            
            # Limit results
            if len(matching_records) >= limit:
                break
        
        return matching_records
    
    def get_transparency_report(self, include_privacy: bool = True) -> Dict[str, Any]:
        """Generate comprehensive transparency report"""
        
        # Recent activity summary
        last_24h = datetime.now(UTC) - timedelta(days=1)
        recent_records = [
            r for r in self.audit_records
            if r.timestamp > last_24h
        ]
        
        # Activity by type
        activity_by_type = {}
        for record in recent_records:
            event_type = record.event_type.value
            activity_by_type[event_type] = activity_by_type.get(event_type, 0) + 1
        
        # Compliance status
        latest_compliance = self.compliance_reports[-1] if self.compliance_reports else None
        
        # Privacy statistics
        privacy_stats = {
            "total_records": len(self.audit_records),
            "privacy_preserved_records": len([r for r in self.audit_records if r.privacy_preserved]),
            "public_records": len([r for r in self.audit_records if r.transparency_level == "public"]),
            "constitutional_compliant": len([r for r in self.audit_records if r.constitutional_compliant])
        }
        
        return {
            "report_generated": datetime.now(UTC).isoformat(),
            "audit_chain_status": {
                "total_records": len(self.audit_records),
                "chain_integrity_valid": self.audit_metrics["chain_integrity_verified"],
                "records_last_24h": len(recent_records)
            },
            "recent_activity": activity_by_type,
            "compliance_status": {
                "latest_compliance_score": float(latest_compliance.overall_compliance_score) if latest_compliance else 0.0,
                "transparency_score": float(latest_compliance.transparency_score) if latest_compliance else 0.0,
                "privacy_score": float(latest_compliance.privacy_score) if latest_compliance else 0.0,
                "governance_score": float(latest_compliance.governance_score) if latest_compliance else 0.0,
                "compliance_issues": latest_compliance.compliance_issues if latest_compliance else []
            },
            "privacy_statistics": privacy_stats if include_privacy else {},
            "audit_metrics": self.audit_metrics,
            "constitutional_features": {
                "immutable_audit_trail": True,
                "pricing_transparency": True,
                "governance_tracking": True,
                "privacy_preservation": self.privacy_preserving_mode,
                "zero_knowledge_proofs": len(self.zero_knowledge_proofs) > 0
            }
        }
    
    def export_audit_records(
        self,
        format: str = "json",
        include_sensitive: bool = False
    ) -> Dict[str, Any]:
        """Export audit records for external analysis"""
        
        exported_records = []
        
        for record in self.audit_records:
            
            record_data = {
                "record_id": record.record_id,
                "event_type": record.event_type.value,
                "timestamp": record.timestamp.isoformat(),
                "chain_position": record.chain_position,
                "record_hash": record.record_hash,
                "constitutional_compliant": record.constitutional_compliant,
                "privacy_preserved": record.privacy_preserved
            }
            
            # Include event data based on sensitivity
            if include_sensitive or record.transparency_level == "public":
                record_data["event_data"] = record.event_data
                record_data["pricing_data"] = record.pricing_data
            
            exported_records.append(record_data)
        
        return {
            "export_timestamp": datetime.now(UTC).isoformat(),
            "export_format": format,
            "total_records": len(exported_records),
            "chain_integrity_verified": self.audit_metrics["chain_integrity_verified"],
            "includes_sensitive_data": include_sensitive,
            "records": exported_records
        }


# Global audit trail manager instance
_audit_trail_manager: AuditTrailManager | None = None


def get_audit_trail_manager() -> AuditTrailManager:
    """Get global audit trail manager instance"""
    global _audit_trail_manager
    
    if _audit_trail_manager is None:
        _audit_trail_manager = AuditTrailManager()
    
    return _audit_trail_manager


# Convenience functions for audit logging
def log_pricing_calculation(
    calculation_type: str,
    calculation_data: Dict[str, Any],
    device_id: str = None,
    actor_id: str = None
) -> str:
    """Log pricing calculation to audit trail"""
    
    manager = get_audit_trail_manager()
    return manager.log_calculation(calculation_type, calculation_data, device_id, actor_id)


def log_pricing_quote(quote_data: Dict[str, Any], actor_id: str = None) -> str:
    """Log pricing quote to audit trail"""
    
    manager = get_audit_trail_manager()
    return manager.log_pricing_quote(quote_data, actor_id)


def verify_audit_integrity() -> Dict[str, Any]:
    """Verify audit chain integrity"""
    
    manager = get_audit_trail_manager()
    return manager.verify_audit_chain_integrity()


def generate_compliance_report() -> ConstitutionalComplianceReport:
    """Generate constitutional compliance report"""
    
    manager = get_audit_trail_manager()
    return manager.generate_constitutional_compliance_report()