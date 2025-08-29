"""
Anti-Griefing and Market Validation System

Comprehensive protection against market manipulation and griefing:
- Bid validation and authentication
- Reputation-based filtering
- Market manipulation detection
- Coordinated attack prevention
- Quality assurance mechanisms

Key Features:
- Multi-layer validation pipeline
- Real-time manipulation detection
- Automated response systems
- Reputation scoring and filtering
- Economic incentives for good behavior
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
import logging
from typing import Any
import uuid

# Set precision for calculations
getcontext().prec = 18

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationStatus(str, Enum):
    """Validation result status"""

    PASSED = "passed"
    WARNING = "warning"
    BLOCKED = "blocked"
    REQUIRES_REVIEW = "requires_review"


class GriefingType(str, Enum):
    """Types of griefing behavior"""

    SYBIL_ATTACK = "sybil_attack"
    BID_SPAM = "bid_spam"
    PRICE_MANIPULATION = "price_manipulation"
    REPUTATION_ABUSE = "reputation_abuse"
    RESOURCE_HOARDING = "resource_hoarding"
    FAKE_RESOURCES = "fake_resources"
    COORDINATION_ATTACK = "coordination_attack"


@dataclass
class ValidationResult:
    """Result of anti-griefing validation"""

    participant_id: str
    status: ValidationStatus
    threat_level: ThreatLevel

    # Validation details
    passed_checks: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    warning_checks: list[str] = field(default_factory=list)

    # Risk assessment
    risk_score: Decimal = Decimal("0")  # 0-1 risk score
    confidence: Decimal = Decimal("0")  # 0-1 confidence in assessment

    # Detected issues
    detected_griefing_types: list[GriefingType] = field(default_factory=list)
    suspicious_patterns: list[str] = field(default_factory=list)

    # Recommendations
    recommended_actions: list[str] = field(default_factory=list)
    monitoring_recommendations: list[str] = field(default_factory=list)

    # Metadata
    validation_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    validator_version: str = "1.0.0"


@dataclass
class ParticipantProfile:
    """Participant profile for reputation and behavior tracking"""

    participant_id: str

    # Identity verification
    identity_verified: bool = False
    verification_level: str = "none"  # none, basic, verified, premium
    kyc_completed: bool = False

    # Historical behavior
    total_bids: int = 0
    successful_bids: int = 0
    failed_bids: int = 0
    cancelled_bids: int = 0

    # Performance metrics
    average_trust_score: Decimal = Decimal("0.5")
    reliability_score: Decimal = Decimal("0.5")  # Based on fulfillment history
    quality_score: Decimal = Decimal("0.5")  # Service quality

    # Financial behavior
    total_volume: Decimal = Decimal("0")
    average_bid_size: Decimal = Decimal("0")
    deposit_compliance_rate: Decimal = Decimal("1.0")
    payment_history_score: Decimal = Decimal("0.5")

    # Risk indicators
    suspicious_activity_count: int = 0
    last_suspicious_activity: datetime | None = None
    blocked_until: datetime | None = None
    warning_level: int = 0  # 0-5 escalating warnings

    # Network analysis
    connected_participants: set[str] = field(default_factory=set)
    similar_behavior_groups: list[str] = field(default_factory=list)

    # Timestamps
    first_seen: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))

    def calculate_reputation_score(self) -> Decimal:
        """Calculate overall reputation score"""

        # Success rate component
        if self.total_bids > 0:
            success_rate = Decimal(str(self.successful_bids / self.total_bids))
        else:
            success_rate = Decimal("0.5")  # Neutral for new participants

        # Trust and quality components
        trust_component = self.average_trust_score * Decimal("0.3")
        quality_component = self.quality_score * Decimal("0.2")
        reliability_component = self.reliability_score * Decimal("0.2")
        success_component = success_rate * Decimal("0.3")

        reputation = trust_component + quality_component + reliability_component + success_component

        # Penalty for suspicious activity
        if self.suspicious_activity_count > 0:
            penalty = min(Decimal("0.5"), Decimal(str(self.suspicious_activity_count)) * Decimal("0.1"))
            reputation = max(Decimal("0"), reputation - penalty)

        return reputation

    def is_blocked(self) -> bool:
        """Check if participant is currently blocked"""
        if not self.blocked_until:
            return False
        return datetime.now(UTC) < self.blocked_until

    def needs_enhanced_monitoring(self) -> bool:
        """Check if participant needs enhanced monitoring"""
        return (
            self.warning_level >= 2
            or self.suspicious_activity_count >= 3
            or self.calculate_reputation_score() < Decimal("0.3")
        )


@dataclass
class MarketAnomalyDetection:
    """Detection of market-wide anomalies and coordinated attacks"""

    detection_id: str
    anomaly_type: str

    # Detection details
    affected_participants: list[str] = field(default_factory=list)
    suspicious_patterns: dict[str, Any] = field(default_factory=dict)
    confidence_score: Decimal = Decimal("0")

    # Time analysis
    detection_window: timedelta = timedelta(hours=1)
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Impact assessment
    market_impact_score: Decimal = Decimal("0")  # 0-1 impact on market
    affected_auctions: list[str] = field(default_factory=list)
    estimated_damage: Decimal = Decimal("0")

    # Response
    automated_response_taken: bool = False
    response_actions: list[str] = field(default_factory=list)
    requires_human_review: bool = False


class AntiGriefingSystem:
    """
    Comprehensive anti-griefing and market validation system

    Protects against:
    - Sybil attacks and fake identities
    - Bid spam and resource manipulation
    - Price manipulation schemes
    - Coordinated market attacks
    - Quality/reputation abuse
    """

    def __init__(self, token_system=None, auction_engine=None, pricing_manager=None):
        self.token_system = token_system
        self.auction_engine = auction_engine
        self.pricing_manager = pricing_manager

        # Participant tracking
        self.participant_profiles: dict[str, ParticipantProfile] = {}
        self.validation_history: list[ValidationResult] = []

        # Anomaly detection
        self.detected_anomalies: list[MarketAnomalyDetection] = []
        self.pattern_cache: dict[str, Any] = {}

        # Configuration
        self.config = {
            # Reputation thresholds
            "min_reputation_for_bidding": Decimal("0.1"),
            "min_reputation_for_large_bids": Decimal("0.5"),
            "reputation_decay_rate": Decimal("0.01"),  # Per day
            # Behavioral limits
            "max_bids_per_hour": 20,
            "max_bid_frequency_per_minute": 3,
            "min_deposit_compliance_rate": Decimal("0.8"),
            # Anomaly detection
            "coordination_detection_window": timedelta(minutes=5),
            "min_coordination_participants": 3,
            "price_manipulation_threshold": Decimal("0.2"),  # 20% sudden change
            # Response thresholds
            "auto_block_risk_threshold": Decimal("0.8"),
            "warning_risk_threshold": Decimal("0.5"),
            "enhanced_monitoring_threshold": Decimal("0.3"),
            # Economic parameters
            "min_stake_for_high_trust": Decimal("1000"),  # FOG tokens
            "quality_bond_amount": Decimal("100"),  # Quality assurance bond
        }

        # Real-time monitoring
        self.recent_activities: dict[str, list[tuple[datetime, str]]] = {}  # participant_id -> activities
        self.coordination_tracker: dict[str, list[tuple[datetime, str]]] = {}  # pattern -> participants

        # Background tasks
        self._monitoring_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        logger.info("Anti-griefing system initialized")

    async def start(self):
        """Start anti-griefing monitoring systems"""

        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Anti-griefing system started")

    async def stop(self):
        """Stop anti-griefing monitoring systems"""

        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        logger.info("Anti-griefing system stopped")

    async def validate_participant(
        self, participant_id: str, action_type: str, context: dict[str, Any]
    ) -> ValidationResult:
        """Comprehensive participant validation"""

        # Get or create participant profile
        if participant_id not in self.participant_profiles:
            self.participant_profiles[participant_id] = ParticipantProfile(participant_id=participant_id)

        profile = self.participant_profiles[participant_id]

        # Initialize validation result
        result = ValidationResult(
            participant_id=participant_id, status=ValidationStatus.PASSED, threat_level=ThreatLevel.LOW
        )

        # Run validation checks
        await self._validate_identity(profile, result, context)
        await self._validate_behavior_patterns(profile, result, context, action_type)
        await self._validate_financial_requirements(profile, result, context)
        await self._validate_reputation_requirements(profile, result, context, action_type)
        await self._detect_coordination_patterns(profile, result, context)

        # Calculate overall risk score
        result.risk_score = self._calculate_risk_score(profile, result)

        # Determine final status and threat level
        self._determine_final_status(result)

        # Take automated actions if needed
        await self._handle_validation_result(profile, result)

        # Store validation history
        self.validation_history.append(result)

        # Update participant activity
        self._update_participant_activity(participant_id, action_type)

        logger.info(
            f"Validation completed for {participant_id}: "
            f"status={result.status.value}, "
            f"threat_level={result.threat_level.value}, "
            f"risk_score={float(result.risk_score):.3f}"
        )

        return result

    async def validate_bid_submission(
        self, bidder_id: str, auction_id: str, bid_data: dict[str, Any]
    ) -> ValidationResult:
        """Validate bid submission for anti-griefing"""

        context = {
            "auction_id": auction_id,
            "bid_amount": bid_data.get("bid_price", 0),
            "resources": bid_data.get("available_resources", {}),
            "bid_type": bid_data.get("bid_type", "spot"),
        }

        return await self.validate_participant(bidder_id, "bid_submission", context)

    async def detect_market_manipulation(self, time_window: timedelta = None) -> list[MarketAnomalyDetection]:
        """Detect market-wide manipulation patterns"""

        if time_window is None:
            time_window = self.config["coordination_detection_window"]

        anomalies = []
        current_time = datetime.now(UTC)
        window_start = current_time - time_window

        # Price manipulation detection
        price_anomaly = await self._detect_price_manipulation(window_start, current_time)
        if price_anomaly:
            anomalies.append(price_anomaly)

        # Coordination attack detection
        coordination_anomaly = await self._detect_coordination_attacks(window_start, current_time)
        if coordination_anomaly:
            anomalies.append(coordination_anomaly)

        # Volume anomaly detection
        volume_anomaly = await self._detect_volume_anomalies(window_start, current_time)
        if volume_anomaly:
            anomalies.append(volume_anomaly)

        # Store detected anomalies
        self.detected_anomalies.extend(anomalies)

        if anomalies:
            logger.warning(f"Detected {len(anomalies)} market anomalies in {time_window}")

        return anomalies

    async def get_participant_risk_assessment(self, participant_id: str) -> dict[str, Any]:
        """Get comprehensive risk assessment for participant"""

        if participant_id not in self.participant_profiles:
            return {"error": "Participant not found"}

        profile = self.participant_profiles[participant_id]
        reputation_score = profile.calculate_reputation_score()

        # Recent validation results
        recent_validations = [
            v for v in self.validation_history[-100:] if v.participant_id == participant_id  # Last 100 validations
        ]

        # Risk factors
        risk_factors = []
        if profile.suspicious_activity_count > 0:
            risk_factors.append(f"{profile.suspicious_activity_count} suspicious activities")
        if profile.warning_level > 0:
            risk_factors.append(f"Warning level {profile.warning_level}")
        if profile.is_blocked():
            risk_factors.append("Currently blocked")

        return {
            "participant_id": participant_id,
            "reputation_score": float(reputation_score),
            "trust_score": float(profile.average_trust_score),
            "quality_score": float(profile.quality_score),
            "reliability_score": float(profile.reliability_score),
            "verification_status": {
                "identity_verified": profile.identity_verified,
                "verification_level": profile.verification_level,
                "kyc_completed": profile.kyc_completed,
            },
            "activity_summary": {
                "total_bids": profile.total_bids,
                "success_rate": profile.successful_bids / max(1, profile.total_bids),
                "total_volume": float(profile.total_volume),
                "average_bid_size": float(profile.average_bid_size),
            },
            "risk_assessment": {
                "risk_factors": risk_factors,
                "is_blocked": profile.is_blocked(),
                "needs_enhanced_monitoring": profile.needs_enhanced_monitoring(),
                "suspicious_activity_count": profile.suspicious_activity_count,
                "warning_level": profile.warning_level,
            },
            "recent_validations": {
                "total_validations": len(recent_validations),
                "passed": len([v for v in recent_validations if v.status == ValidationStatus.PASSED]),
                "warnings": len([v for v in recent_validations if v.status == ValidationStatus.WARNING]),
                "blocked": len([v for v in recent_validations if v.status == ValidationStatus.BLOCKED]),
            },
        }

    async def get_market_security_analytics(self) -> dict[str, Any]:
        """Get comprehensive market security analytics"""

        # Participant statistics
        total_participants = len(self.participant_profiles)
        blocked_participants = len([p for p in self.participant_profiles.values() if p.is_blocked()])
        high_risk_participants = len(
            [p for p in self.participant_profiles.values() if p.calculate_reputation_score() < Decimal("0.3")]
        )

        # Validation statistics
        recent_validations = [
            v for v in self.validation_history if (datetime.now(UTC) - v.validation_timestamp).days <= 1
        ]

        validation_stats = {
            "total_validations_24h": len(recent_validations),
            "passed": len([v for v in recent_validations if v.status == ValidationStatus.PASSED]),
            "warnings": len([v for v in recent_validations if v.status == ValidationStatus.WARNING]),
            "blocked": len([v for v in recent_validations if v.status == ValidationStatus.BLOCKED]),
        }

        # Anomaly statistics
        recent_anomalies = [a for a in self.detected_anomalies if (datetime.now(UTC) - a.detected_at).days <= 1]

        anomaly_types = {}
        for anomaly in recent_anomalies:
            anomaly_types[anomaly.anomaly_type] = anomaly_types.get(anomaly.anomaly_type, 0) + 1

        # Threat level distribution
        threat_levels = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for validation in recent_validations:
            threat_levels[validation.threat_level.value] += 1

        return {
            "participant_security": {
                "total_participants": total_participants,
                "blocked_participants": blocked_participants,
                "high_risk_participants": high_risk_participants,
                "block_rate": blocked_participants / max(1, total_participants),
                "average_reputation": float(
                    sum(p.calculate_reputation_score() for p in self.participant_profiles.values())
                    / max(1, total_participants)
                ),
            },
            "validation_metrics": validation_stats,
            "anomaly_detection": {
                "total_anomalies_24h": len(recent_anomalies),
                "anomaly_types": anomaly_types,
                "critical_anomalies": len([a for a in recent_anomalies if a.confidence_score > Decimal("0.8")]),
            },
            "threat_assessment": {
                "threat_level_distribution": threat_levels,
                "overall_threat_level": self._calculate_overall_threat_level(),
                "market_safety_score": self._calculate_market_safety_score(),
            },
            "system_status": {
                "monitoring_active": self._monitoring_task is not None,
                "detection_accuracy": self._calculate_detection_accuracy(),
                "response_time_seconds": self._calculate_average_response_time(),
            },
        }

    # Private validation methods

    async def _validate_identity(self, profile: ParticipantProfile, result: ValidationResult, context: dict[str, Any]):
        """Validate participant identity and verification status"""

        # Basic identity checks
        if not profile.identity_verified and profile.total_bids > 10:
            result.warning_checks.append("High activity without identity verification")
            result.recommended_actions.append("Complete identity verification")

        # KYC requirements for large transactions
        bid_amount = Decimal(str(context.get("bid_amount", 0)))
        if bid_amount > Decimal("1000") and not profile.kyc_completed:
            result.failed_checks.append("KYC required for large transactions")
            result.recommended_actions.append("Complete KYC verification")

        # Verification level requirements
        if profile.verification_level == "none" and profile.total_volume > Decimal("10000"):
            result.warning_checks.append("High volume without verification")

        result.passed_checks.append("Identity validation completed")

    async def _validate_behavior_patterns(
        self, profile: ParticipantProfile, result: ValidationResult, context: dict[str, Any], action_type: str
    ):
        """Validate behavioral patterns for suspicious activity"""

        participant_id = profile.participant_id
        current_time = datetime.now(UTC)

        # Check bid frequency
        if action_type == "bid_submission":
            recent_activities = self.recent_activities.get(participant_id, [])

            # Hourly limit
            hour_ago = current_time - timedelta(hours=1)
            recent_bids = len(
                [
                    activity
                    for activity in recent_activities
                    if activity[0] > hour_ago and activity[1] == "bid_submission"
                ]
            )

            if recent_bids >= self.config["max_bids_per_hour"]:
                result.failed_checks.append("Exceeded hourly bid limit")
                result.detected_griefing_types.append(GriefingType.BID_SPAM)
                return

            # Per-minute frequency
            minute_ago = current_time - timedelta(minutes=1)
            very_recent_bids = len(
                [
                    activity
                    for activity in recent_activities
                    if activity[0] > minute_ago and activity[1] == "bid_submission"
                ]
            )

            if very_recent_bids >= self.config["max_bid_frequency_per_minute"]:
                result.failed_checks.append("Bid frequency too high")
                result.detected_griefing_types.append(GriefingType.BID_SPAM)
                return

        # Check for pattern anomalies
        if profile.total_bids > 0:
            success_rate = profile.successful_bids / profile.total_bids
            if success_rate < 0.1 and profile.total_bids > 20:
                result.warning_checks.append("Unusually low success rate")
                result.suspicious_patterns.append("Potential fake bidding")

        result.passed_checks.append("Behavioral pattern validation completed")

    async def _validate_financial_requirements(
        self, profile: ParticipantProfile, result: ValidationResult, context: dict[str, Any]
    ):
        """Validate financial requirements and deposit compliance"""

        # Check deposit compliance rate
        if profile.deposit_compliance_rate < self.config["min_deposit_compliance_rate"]:
            result.warning_checks.append("Low deposit compliance rate")
            result.recommended_actions.append("Improve deposit payment reliability")

        # Validate account balance for deposits
        if self.token_system and context.get("bid_amount"):
            participant_id = profile.participant_id
            account_info = self.token_system.get_account_balance(participant_id)

            if account_info.get("error"):
                result.failed_checks.append("Cannot verify account balance")
                return

            available_balance = Decimal(str(account_info.get("balance", 0)))
            required_deposit = Decimal(str(context["bid_amount"])) * Decimal("0.1")  # 10% deposit

            if available_balance < required_deposit:
                result.failed_checks.append("Insufficient balance for required deposit")
                return

        # Check for financial manipulation patterns
        if profile.average_bid_size > Decimal("0"):
            current_bid = Decimal(str(context.get("bid_amount", 0)))
            if current_bid > profile.average_bid_size * Decimal("10"):
                result.warning_checks.append("Bid significantly larger than historical average")
                result.suspicious_patterns.append("Potential financial manipulation")

        result.passed_checks.append("Financial validation completed")

    async def _validate_reputation_requirements(
        self, profile: ParticipantProfile, result: ValidationResult, context: dict[str, Any], action_type: str
    ):
        """Validate reputation-based requirements"""

        reputation_score = profile.calculate_reputation_score()

        # Minimum reputation for bidding
        if reputation_score < self.config["min_reputation_for_bidding"]:
            result.failed_checks.append("Reputation too low for market participation")
            result.recommended_actions.append("Improve reputation through successful transactions")
            return

        # Higher reputation requirements for large bids
        bid_amount = Decimal(str(context.get("bid_amount", 0)))
        if bid_amount > Decimal("500") and reputation_score < self.config["min_reputation_for_large_bids"]:
            result.warning_checks.append("Higher reputation recommended for large bids")
            result.recommended_actions.append("Build reputation with smaller transactions first")

        # Trust score requirements
        if profile.average_trust_score < Decimal("0.2"):
            result.warning_checks.append("Very low trust score")
            result.monitoring_recommendations.append("Enhanced monitoring due to low trust")

        result.passed_checks.append("Reputation validation completed")

    async def _detect_coordination_patterns(
        self, profile: ParticipantProfile, result: ValidationResult, context: dict[str, Any]
    ):
        """Detect coordinated attack patterns"""

        participant_id = profile.participant_id
        current_time = datetime.now(UTC)

        # Look for coordination in timing
        auction_id = context.get("auction_id")
        if auction_id:
            # Check if multiple related participants are bidding on same auction
            coordination_key = f"auction:{auction_id}"

            if coordination_key not in self.coordination_tracker:
                self.coordination_tracker[coordination_key] = []

            self.coordination_tracker[coordination_key].append((current_time, participant_id))

            # Check for suspicious coordination
            window_start = current_time - self.config["coordination_detection_window"]
            recent_participants = [
                pid for timestamp, pid in self.coordination_tracker[coordination_key] if timestamp > window_start
            ]

            if len(set(recent_participants)) >= self.config["min_coordination_participants"]:
                # Check if participants have similar profiles or connections
                similar_profiles = self._find_similar_profiles(recent_participants)

                if len(similar_profiles) >= 2:
                    result.warning_checks.append("Potential coordinated bidding detected")
                    result.detected_griefing_types.append(GriefingType.COORDINATION_ATTACK)
                    result.suspicious_patterns.append("Multiple similar profiles bidding simultaneously")

        result.passed_checks.append("Coordination detection completed")

    def _calculate_risk_score(self, profile: ParticipantProfile, result: ValidationResult) -> Decimal:
        """Calculate overall risk score based on validation results"""

        base_risk = Decimal("0.1")  # Base risk for all participants

        # Failed checks contribute significantly to risk
        failed_weight = Decimal(str(len(result.failed_checks))) * Decimal("0.3")

        # Warning checks contribute moderately
        warning_weight = Decimal(str(len(result.warning_checks))) * Decimal("0.1")

        # Detected griefing types are high risk
        griefing_weight = Decimal(str(len(result.detected_griefing_types))) * Decimal("0.2")

        # Suspicious patterns add risk
        pattern_weight = Decimal(str(len(result.suspicious_patterns))) * Decimal("0.1")

        # Historical factors
        history_risk = Decimal("0")
        if profile.suspicious_activity_count > 0:
            history_risk = min(Decimal("0.4"), Decimal(str(profile.suspicious_activity_count)) * Decimal("0.1"))

        # Reputation factor (inverse relationship)
        reputation_score = profile.calculate_reputation_score()
        reputation_risk = max(Decimal("0"), (Decimal("0.5") - reputation_score) * Decimal("0.4"))

        total_risk = (
            base_risk
            + failed_weight
            + warning_weight
            + griefing_weight
            + pattern_weight
            + history_risk
            + reputation_risk
        )

        return min(Decimal("1"), total_risk)

    def _determine_final_status(self, result: ValidationResult):
        """Determine final validation status and threat level"""

        risk_score = result.risk_score

        if risk_score >= self.config["auto_block_risk_threshold"]:
            result.status = ValidationStatus.BLOCKED
            result.threat_level = ThreatLevel.CRITICAL
        elif risk_score >= self.config["warning_risk_threshold"]:
            result.status = ValidationStatus.WARNING
            result.threat_level = ThreatLevel.HIGH if risk_score >= Decimal("0.7") else ThreatLevel.MEDIUM
        elif risk_score >= self.config["enhanced_monitoring_threshold"]:
            result.status = ValidationStatus.REQUIRES_REVIEW
            result.threat_level = ThreatLevel.MEDIUM
        else:
            result.status = ValidationStatus.PASSED
            result.threat_level = ThreatLevel.LOW

        # Override for failed checks
        if result.failed_checks:
            result.status = ValidationStatus.BLOCKED
            result.threat_level = max(result.threat_level, ThreatLevel.HIGH)

        result.confidence = min(Decimal("1"), result.risk_score + Decimal("0.2"))

    async def _handle_validation_result(self, profile: ParticipantProfile, result: ValidationResult):
        """Handle validation result with appropriate actions"""

        if result.status == ValidationStatus.BLOCKED:
            # Block participant temporarily
            block_duration = timedelta(hours=24)  # Base block duration

            # Escalate block duration for repeat offenders
            if profile.warning_level > 2:
                block_duration = timedelta(days=profile.warning_level)

            profile.blocked_until = datetime.now(UTC) + block_duration
            profile.suspicious_activity_count += 1
            profile.warning_level += 1

            logger.warning(
                f"Blocked participant {profile.participant_id} for {block_duration} "
                f"due to {result.threat_level.value} risk (score: {float(result.risk_score):.3f})"
            )

        elif result.status == ValidationStatus.WARNING:
            profile.warning_level += 1
            profile.suspicious_activity_count += 1

            logger.info(
                f"Warning issued to participant {profile.participant_id} " f"(warning level: {profile.warning_level})"
            )

        # Update last activity time
        profile.last_activity = datetime.now(UTC)

    def _update_participant_activity(self, participant_id: str, action_type: str):
        """Update participant activity tracking"""

        current_time = datetime.now(UTC)

        if participant_id not in self.recent_activities:
            self.recent_activities[participant_id] = []

        self.recent_activities[participant_id].append((current_time, action_type))

        # Keep only last 24 hours of activity
        cutoff = current_time - timedelta(hours=24)
        self.recent_activities[participant_id] = [
            activity for activity in self.recent_activities[participant_id] if activity[0] > cutoff
        ]

    async def _detect_price_manipulation(
        self, window_start: datetime, window_end: datetime
    ) -> MarketAnomalyDetection | None:
        """Detect price manipulation patterns"""

        if not self.pricing_manager:
            return None

        try:
            analytics = await self.pricing_manager.get_market_analytics()
            price_volatility = analytics.get("market_overview", {}).get("volatility", 0)

            if price_volatility > float(self.config["price_manipulation_threshold"]):
                return MarketAnomalyDetection(
                    detection_id=f"price_manip_{uuid.uuid4().hex[:8]}",
                    anomaly_type="price_manipulation",
                    confidence_score=Decimal(str(min(1.0, price_volatility / 0.5))),
                    suspicious_patterns={"volatility": price_volatility},
                    market_impact_score=Decimal(str(price_volatility)),
                    automated_response_taken=False,
                    requires_human_review=True,
                )

        except Exception as e:
            logger.error(f"Error detecting price manipulation: {e}")

        return None

    async def _detect_coordination_attacks(
        self, window_start: datetime, window_end: datetime
    ) -> MarketAnomalyDetection | None:
        """Detect coordinated attack patterns"""

        suspicious_groups = []

        for pattern, participants in self.coordination_tracker.items():
            # Filter to time window
            window_participants = [pid for timestamp, pid in participants if window_start <= timestamp <= window_end]

            unique_participants = list(set(window_participants))

            if len(unique_participants) >= self.config["min_coordination_participants"]:
                # Check for profile similarities
                similar_groups = self._find_similar_profiles(unique_participants)

                if len(similar_groups) >= 2:
                    suspicious_groups.append(
                        {
                            "pattern": pattern,
                            "participants": unique_participants,
                            "similarity_score": len(similar_groups) / len(unique_participants),
                        }
                    )

        if suspicious_groups:
            return MarketAnomalyDetection(
                detection_id=f"coordination_{uuid.uuid4().hex[:8]}",
                anomaly_type="coordination_attack",
                affected_participants=[p for group in suspicious_groups for p in group["participants"]],
                confidence_score=Decimal(str(max(group["similarity_score"] for group in suspicious_groups))),
                suspicious_patterns={"coordination_groups": len(suspicious_groups)},
                automated_response_taken=False,
                requires_human_review=True,
            )

        return None

    async def _detect_volume_anomalies(
        self, window_start: datetime, window_end: datetime
    ) -> MarketAnomalyDetection | None:
        """Detect unusual volume patterns"""

        # This would analyze transaction volumes for anomalies
        # Simplified implementation for now

        return None

    def _find_similar_profiles(self, participant_ids: list[str]) -> list[list[str]]:
        """Find groups of similar participant profiles"""

        similar_groups = []

        for i, pid1 in enumerate(participant_ids):
            for j, pid2 in enumerate(participant_ids[i + 1 :], i + 1):
                if self._are_profiles_similar(pid1, pid2):
                    # Find existing group or create new one
                    group_found = False
                    for group in similar_groups:
                        if pid1 in group or pid2 in group:
                            group.extend([pid1, pid2])
                            group_found = True
                            break

                    if not group_found:
                        similar_groups.append([pid1, pid2])

        # Remove duplicates within groups
        for group in similar_groups:
            group[:] = list(set(group))

        return similar_groups

    def _are_profiles_similar(self, pid1: str, pid2: str) -> bool:
        """Check if two participant profiles are suspiciously similar"""

        if pid1 not in self.participant_profiles or pid2 not in self.participant_profiles:
            return False

        profile1 = self.participant_profiles[pid1]
        profile2 = self.participant_profiles[pid2]

        # Check for similar timing patterns
        time_diff = abs((profile1.first_seen - profile2.first_seen).total_seconds())
        if time_diff < 3600:  # Created within 1 hour
            similarity_score = 1
        else:
            similarity_score = 0

        # Check for similar behavior patterns
        if profile1.total_bids > 0 and profile2.total_bids > 0:
            success_rate1 = profile1.successful_bids / profile1.total_bids
            success_rate2 = profile2.successful_bids / profile2.total_bids

            if abs(success_rate1 - success_rate2) < 0.1:  # Very similar success rates
                similarity_score += 1

        # Check for similar volume patterns
        if abs(float(profile1.average_bid_size - profile2.average_bid_size)) < 10.0:
            similarity_score += 1

        return similarity_score >= 2

    def _calculate_overall_threat_level(self) -> str:
        """Calculate overall market threat level"""

        recent_critical = len(
            [
                v
                for v in self.validation_history[-100:]  # Last 100 validations
                if v.threat_level == ThreatLevel.CRITICAL
            ]
        )

        if recent_critical > 10:
            return "critical"
        elif recent_critical > 5:
            return "high"
        elif recent_critical > 2:
            return "medium"
        else:
            return "low"

    def _calculate_market_safety_score(self) -> float:
        """Calculate overall market safety score (0-1)"""

        if not self.participant_profiles:
            return 0.5  # Neutral for no data

        # Average reputation score
        avg_reputation = sum(p.calculate_reputation_score() for p in self.participant_profiles.values()) / len(
            self.participant_profiles
        )

        # Blocked participant rate
        blocked_rate = len([p for p in self.participant_profiles.values() if p.is_blocked()]) / len(
            self.participant_profiles
        )

        # Recent validation success rate
        recent_validations = self.validation_history[-100:]
        if recent_validations:
            success_rate = len([v for v in recent_validations if v.status == ValidationStatus.PASSED]) / len(
                recent_validations
            )
        else:
            success_rate = 0.5

        # Weighted safety score
        safety_score = float(avg_reputation) * 0.4 + (1 - blocked_rate) * 0.3 + success_rate * 0.3

        return safety_score

    def _calculate_detection_accuracy(self) -> float:
        """Calculate detection accuracy (simplified)"""

        # This would be calculated based on confirmed vs false positives
        # For now, return a placeholder
        return 0.85

    def _calculate_average_response_time(self) -> float:
        """Calculate average response time for validation (simplified)"""

        # This would track actual response times
        # For now, return a placeholder
        return 1.2  # 1.2 seconds average

    async def _monitoring_loop(self):
        """Background monitoring for anomalies"""

        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Run market manipulation detection
                await self.detect_market_manipulation()

                # Update participant reputation decay
                await self._update_reputation_decay()

                # Clean up coordination tracking
                self._cleanup_coordination_tracking()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(600)

    async def _cleanup_loop(self):
        """Background cleanup of old data"""

        while True:
            try:
                await asyncio.sleep(3600)  # Clean up hourly

                # Clean up old validation history
                cutoff = datetime.now(UTC) - timedelta(days=7)
                self.validation_history = [
                    v for v in self.validation_history if v.validation_timestamp.replace(tzinfo=UTC) > cutoff
                ]

                # Clean up old anomalies
                self.detected_anomalies = [
                    a for a in self.detected_anomalies if a.detected_at.replace(tzinfo=UTC) > cutoff
                ]

                # Clean up recent activities
                activity_cutoff = datetime.now(UTC) - timedelta(hours=24)
                for participant_id in self.recent_activities:
                    self.recent_activities[participant_id] = [
                        activity for activity in self.recent_activities[participant_id] if activity[0] > activity_cutoff
                    ]

                logger.info("Completed anti-griefing system cleanup")

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(1800)

    async def _update_reputation_decay(self):
        """Apply reputation decay over time"""

        decay_rate = self.config["reputation_decay_rate"]

        for profile in self.participant_profiles.values():
            days_inactive = (datetime.now(UTC) - profile.last_activity).days

            if days_inactive > 0:
                # Apply decay to trust and quality scores
                decay_factor = (Decimal("1") - decay_rate) ** days_inactive
                profile.average_trust_score *= decay_factor
                profile.quality_score *= decay_factor

                # Reduce warning levels over time for good behavior
                if days_inactive > 30 and profile.warning_level > 0:
                    profile.warning_level = max(0, profile.warning_level - 1)

    def _cleanup_coordination_tracking(self):
        """Clean up old coordination tracking data"""

        cutoff = datetime.now(UTC) - timedelta(hours=1)

        for pattern in self.coordination_tracker:
            self.coordination_tracker[pattern] = [
                entry for entry in self.coordination_tracker[pattern] if entry[0] > cutoff
            ]


# Global anti-griefing system instance
_anti_griefing_system: AntiGriefingSystem | None = None


async def get_anti_griefing_system() -> AntiGriefingSystem:
    """Get global anti-griefing system instance"""
    global _anti_griefing_system

    if _anti_griefing_system is None:
        _anti_griefing_system = AntiGriefingSystem()
        await _anti_griefing_system.start()

    return _anti_griefing_system


# Convenience functions
async def validate_market_participant(
    participant_id: str, action_type: str, context: dict[str, Any]
) -> ValidationResult:
    """Validate market participant for anti-griefing"""

    system = await get_anti_griefing_system()
    return await system.validate_participant(participant_id, action_type, context)


async def check_participant_risk(participant_id: str) -> dict[str, Any]:
    """Check risk assessment for participant"""

    system = await get_anti_griefing_system()
    return await system.get_participant_risk_assessment(participant_id)


async def detect_market_anomalies() -> list[MarketAnomalyDetection]:
    """Detect market-wide anomalies and manipulation"""

    system = await get_anti_griefing_system()
    return await system.detect_market_manipulation()
