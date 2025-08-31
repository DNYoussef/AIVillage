"""
Constitutional Decision Logging System
Comprehensive logging for constitutional governance and decision rationale
"""

import json
import time
import hashlib
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
from collections import defaultdict, deque

class ConstitutionalDecisionType(Enum):
    """Types of constitutional decisions to log"""
    CONTENT_MODERATION = "content_moderation"
    USER_TIER_CHANGE = "user_tier_change"
    GOVERNANCE_VOTE = "governance_vote"
    POLICY_UPDATE = "policy_update"
    CONSTITUTIONAL_AMENDMENT = "constitutional_amendment"
    APPEAL_RESOLUTION = "appeal_resolution"
    HARM_CLASSIFICATION = "harm_classification"
    PRICING_ADJUSTMENT = "pricing_adjustment"
    DEMOCRATIC_PARTICIPATION = "democratic_participation"
    SYSTEM_OVERRIDE = "system_override"

class DecisionOutcome(Enum):
    """Outcomes of constitutional decisions"""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    ESCALATED = "escalated"
    DEFERRED = "deferred"
    APPEALED = "appealed"

class GovernanceLevel(Enum):
    """Levels of governance involvement"""
    AUTOMATED = "automated"        # System-automated decision
    MODERATED = "moderated"       # Human moderator involved
    COMMUNITY = "community"       # Community vote
    CONSTITUTIONAL = "constitutional"  # Constitutional committee
    SUPREME = "supreme"           # Supreme constitutional authority

@dataclass
class ConstitutionalEvidence:
    """Evidence supporting constitutional decision"""
    evidence_type: str
    confidence_score: float
    data_hash: str
    source: str
    timestamp: float
    supporting_documents: List[str]
    expert_opinions: List[str]

@dataclass
class ConstitutionalRationale:
    """Detailed rationale for constitutional decision"""
    primary_reasoning: str
    constitutional_principles_applied: List[str]
    precedents_cited: List[str]
    harm_analysis: Dict[str, Any]
    alternative_considerations: List[str]
    minority_opinions: List[str]
    democratic_input_summary: str

@dataclass
class AppealInformation:
    """Information about appeals process"""
    appeal_id: Optional[str]
    appealing_party: Optional[str]
    appeal_grounds: List[str]
    appeal_status: str
    appeal_deadline: float
    appeal_response: Optional[str]

@dataclass
class ConstitutionalDecisionLog:
    """Comprehensive constitutional decision log entry"""
    log_id: str
    timestamp: float
    decision_type: ConstitutionalDecisionType
    governance_level: GovernanceLevel
    decision_maker: str  # System, moderator ID, or committee
    user_tier: str
    user_id_hash: str  # Privacy-preserving user identifier
    
    # Decision details
    decision_summary: str
    decision_outcome: DecisionOutcome
    rationale: ConstitutionalRationale
    evidence: List[ConstitutionalEvidence]
    
    # Context and metadata
    constitutional_context: Dict[str, Any]
    policy_version: str
    system_state: Dict[str, Any]
    environmental_factors: Dict[str, Any]
    
    # Appeal and oversight
    appeal_info: Optional[AppealInformation]
    oversight_required: bool
    oversight_completed: bool
    
    # Transparency and privacy
    public_disclosure_level: str
    privacy_preserving_hash: str
    audit_trail_hash: str
    
    def to_audit_format(self) -> Dict[str, Any]:
        """Convert to format suitable for audit logging"""
        return {
            'log_id': self.log_id,
            'timestamp': self.timestamp,
            'decision_type': self.decision_type.value,
            'governance_level': self.governance_level.value,
            'decision_outcome': self.decision_outcome.value,
            'rationale_summary': self.rationale.primary_reasoning,
            'constitutional_principles': self.rationale.constitutional_principles_applied,
            'evidence_count': len(self.evidence),
            'appeal_status': self.appeal_info.appeal_status if self.appeal_info else 'not_appealed',
            'public_disclosure_level': self.public_disclosure_level,
            'audit_trail_hash': self.audit_trail_hash
        }

class ConstitutionalDecisionLogger:
    """
    Comprehensive logging system for constitutional decisions
    Maintains detailed records with full rationale and appeal tracking
    """
    
    def __init__(self, storage_path: str = "constitutional_decisions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.decision_logs: List[ConstitutionalDecisionLog] = []
        self.decision_index: Dict[str, ConstitutionalDecisionLog] = {}
        
        # Analytics and metrics
        self.decision_metrics = defaultdict(int)
        self.governance_metrics = defaultdict(int)
        self.appeal_metrics = defaultdict(int)
        
        # Recent decisions cache for fast lookup
        self.recent_decisions = deque(maxlen=1000)
        
        self.logger = logging.getLogger(__name__)
        
        self._initialize_logging_system()
    
    def _initialize_logging_system(self):
        """Initialize the constitutional decision logging system"""
        self.logger.info("Initializing Constitutional Decision Logger")
        
        # Load existing logs
        self._load_existing_logs()
        
        # Initialize metrics
        self._calculate_metrics()
    
    def _load_existing_logs(self):
        """Load existing decision logs from storage"""
        try:
            log_files = list(self.storage_path.glob("decisions_*.json"))
            for log_file in sorted(log_files):
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    for log_data in data.get('decisions', []):
                        # Reconstruct decision log
                        decision = self._reconstruct_decision_log(log_data)
                        if decision:
                            self.decision_logs.append(decision)
                            self.decision_index[decision.log_id] = decision
                            self.recent_decisions.append(decision.log_id)
            
            self.logger.info(f"Loaded {len(self.decision_logs)} constitutional decisions")
            
        except Exception as e:
            self.logger.error(f"Error loading decision logs: {e}")
    
    def _reconstruct_decision_log(self, log_data: Dict[str, Any]) -> Optional[ConstitutionalDecisionLog]:
        """Reconstruct decision log from stored data"""
        try:
            # Reconstruct rationale
            rationale_data = log_data.get('rationale', {})
            rationale = ConstitutionalRationale(
                primary_reasoning=rationale_data.get('primary_reasoning', ''),
                constitutional_principles_applied=rationale_data.get('constitutional_principles_applied', []),
                precedents_cited=rationale_data.get('precedents_cited', []),
                harm_analysis=rationale_data.get('harm_analysis', {}),
                alternative_considerations=rationale_data.get('alternative_considerations', []),
                minority_opinions=rationale_data.get('minority_opinions', []),
                democratic_input_summary=rationale_data.get('democratic_input_summary', '')
            )
            
            # Reconstruct evidence
            evidence_list = []
            for ev_data in log_data.get('evidence', []):
                evidence = ConstitutionalEvidence(**ev_data)
                evidence_list.append(evidence)
            
            # Reconstruct appeal info
            appeal_info = None
            if log_data.get('appeal_info'):
                appeal_info = AppealInformation(**log_data['appeal_info'])
            
            # Create decision log
            decision = ConstitutionalDecisionLog(
                log_id=log_data['log_id'],
                timestamp=log_data['timestamp'],
                decision_type=ConstitutionalDecisionType(log_data['decision_type']),
                governance_level=GovernanceLevel(log_data['governance_level']),
                decision_maker=log_data['decision_maker'],
                user_tier=log_data['user_tier'],
                user_id_hash=log_data['user_id_hash'],
                decision_summary=log_data['decision_summary'],
                decision_outcome=DecisionOutcome(log_data['decision_outcome']),
                rationale=rationale,
                evidence=evidence_list,
                constitutional_context=log_data.get('constitutional_context', {}),
                policy_version=log_data['policy_version'],
                system_state=log_data.get('system_state', {}),
                environmental_factors=log_data.get('environmental_factors', {}),
                appeal_info=appeal_info,
                oversight_required=log_data.get('oversight_required', False),
                oversight_completed=log_data.get('oversight_completed', False),
                public_disclosure_level=log_data['public_disclosure_level'],
                privacy_preserving_hash=log_data['privacy_preserving_hash'],
                audit_trail_hash=log_data['audit_trail_hash']
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error reconstructing decision log: {e}")
            return None
    
    def _calculate_metrics(self):
        """Calculate decision metrics from loaded logs"""
        for decision in self.decision_logs:
            self.decision_metrics[decision.decision_type.value] += 1
            self.governance_metrics[decision.governance_level.value] += 1
            
            if decision.appeal_info:
                self.appeal_metrics[decision.appeal_info.appeal_status] += 1
    
    async def log_constitutional_decision(self,
                                        decision_type: ConstitutionalDecisionType,
                                        governance_level: GovernanceLevel,
                                        decision_maker: str,
                                        user_tier: str,
                                        user_id: str,
                                        decision_summary: str,
                                        decision_outcome: DecisionOutcome,
                                        rationale_data: Dict[str, Any],
                                        evidence_data: List[Dict[str, Any]],
                                        constitutional_context: Dict[str, Any],
                                        policy_version: str = "1.0",
                                        oversight_required: bool = False) -> str:
        """
        Log comprehensive constitutional decision with full documentation
        """
        log_id = f"const_decision_{int(time.time() * 1000000)}"
        timestamp = time.time()
        
        # Create privacy-preserving user ID hash
        user_id_hash = hashlib.sha256(f"{user_id}_{timestamp}".encode('utf-8')).hexdigest()[:16]
        
        # Construct rationale
        rationale = ConstitutionalRationale(
            primary_reasoning=rationale_data.get('primary_reasoning', ''),
            constitutional_principles_applied=rationale_data.get('constitutional_principles', []),
            precedents_cited=rationale_data.get('precedents', []),
            harm_analysis=rationale_data.get('harm_analysis', {}),
            alternative_considerations=rationale_data.get('alternatives', []),
            minority_opinions=rationale_data.get('minority_opinions', []),
            democratic_input_summary=rationale_data.get('democratic_input', '')
        )
        
        # Construct evidence list
        evidence_list = []
        for ev_data in evidence_data:
            evidence = ConstitutionalEvidence(
                evidence_type=ev_data.get('type', 'unknown'),
                confidence_score=ev_data.get('confidence', 0.0),
                data_hash=hashlib.sha256(
                    json.dumps(ev_data.get('data', {}), sort_keys=True).encode('utf-8')
                ).hexdigest(),
                source=ev_data.get('source', 'system'),
                timestamp=timestamp,
                supporting_documents=ev_data.get('documents', []),
                expert_opinions=ev_data.get('expert_opinions', [])
            )
            evidence_list.append(evidence)
        
        # Determine public disclosure level based on tier and decision type
        public_disclosure_level = self._determine_disclosure_level(
            user_tier, decision_type, governance_level
        )
        
        # Create privacy-preserving hash
        privacy_hash = hashlib.sha256(
            json.dumps({
                'decision_type': decision_type.value,
                'outcome': decision_outcome.value,
                'tier': user_tier,
                'timestamp': timestamp
            }, sort_keys=True).encode('utf-8')
        ).hexdigest()
        
        # Create audit trail hash
        audit_hash = hashlib.sha256(
            json.dumps({
                'log_id': log_id,
                'decision_maker': decision_maker,
                'rationale': rationale.primary_reasoning,
                'evidence_count': len(evidence_list),
                'constitutional_context': constitutional_context
            }, sort_keys=True).encode('utf-8')
        ).hexdigest()
        
        # Get system state
        system_state = await self._capture_system_state()
        
        # Get environmental factors
        environmental_factors = await self._capture_environmental_factors()
        
        # Create decision log
        decision_log = ConstitutionalDecisionLog(
            log_id=log_id,
            timestamp=timestamp,
            decision_type=decision_type,
            governance_level=governance_level,
            decision_maker=decision_maker,
            user_tier=user_tier,
            user_id_hash=user_id_hash,
            decision_summary=decision_summary,
            decision_outcome=decision_outcome,
            rationale=rationale,
            evidence=evidence_list,
            constitutional_context=constitutional_context,
            policy_version=policy_version,
            system_state=system_state,
            environmental_factors=environmental_factors,
            appeal_info=None,
            oversight_required=oversight_required,
            oversight_completed=not oversight_required,
            public_disclosure_level=public_disclosure_level,
            privacy_preserving_hash=privacy_hash,
            audit_trail_hash=audit_hash
        )
        
        # Add to logs
        self.decision_logs.append(decision_log)
        self.decision_index[log_id] = decision_log
        self.recent_decisions.append(log_id)
        
        # Update metrics
        self.decision_metrics[decision_type.value] += 1
        self.governance_metrics[governance_level.value] += 1
        
        # Persist to storage
        await self._persist_decision_log(decision_log)
        
        self.logger.info(f"Logged constitutional decision {log_id}: {decision_type.value}")
        
        return log_id
    
    def _determine_disclosure_level(self, 
                                   user_tier: str,
                                   decision_type: ConstitutionalDecisionType,
                                   governance_level: GovernanceLevel) -> str:
        """Determine appropriate public disclosure level"""
        # Bronze tier - full transparency
        if user_tier == 'bronze':
            return 'full_public'
        
        # Silver tier - selective transparency
        elif user_tier == 'silver':
            if decision_type in [ConstitutionalDecisionType.GOVERNANCE_VOTE, 
                               ConstitutionalDecisionType.POLICY_UPDATE]:
                return 'summary_public'
            else:
                return 'statistical_only'
        
        # Gold tier - privacy-preserving transparency
        elif user_tier == 'gold':
            if governance_level in [GovernanceLevel.COMMUNITY, GovernanceLevel.CONSTITUTIONAL]:
                return 'anonymized_summary'
            else:
                return 'aggregate_only'
        
        # Platinum tier - minimal disclosure
        elif user_tier == 'platinum':
            if decision_type == ConstitutionalDecisionType.CONSTITUTIONAL_AMENDMENT:
                return 'anonymized_summary'
            else:
                return 'commitment_only'
        
        return 'statistical_only'
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture relevant system state at decision time"""
        return {
            'active_policies': ['content_moderation_v1', 'harm_classification_v2'],
            'system_load': 0.65,  # Placeholder
            'active_moderators': 12,  # Placeholder
            'pending_appeals': len([d for d in self.decision_logs 
                                  if d.appeal_info and d.appeal_info.appeal_status == 'pending']),
            'governance_session_active': True,  # Placeholder
            'constitutional_amendments_pending': 2  # Placeholder
        }
    
    async def _capture_environmental_factors(self) -> Dict[str, Any]:
        """Capture environmental factors affecting decision"""
        return {
            'peak_usage_period': False,  # Placeholder
            'community_sentiment': 'stable',  # Placeholder
            'recent_policy_changes': [],
            'external_events': [],
            'democratic_participation_rate': 0.75,  # Placeholder
            'constitutional_crisis_level': 'normal'
        }
    
    async def register_appeal(self, 
                            original_decision_id: str,
                            appealing_party: str,
                            appeal_grounds: List[str],
                            appeal_deadline_hours: int = 72) -> str:
        """Register an appeal for a constitutional decision"""
        if original_decision_id not in self.decision_index:
            raise ValueError("Original decision not found")
        
        appeal_id = f"appeal_{int(time.time() * 1000000)}"
        appeal_deadline = time.time() + (appeal_deadline_hours * 3600)
        
        appeal_info = AppealInformation(
            appeal_id=appeal_id,
            appealing_party=appealing_party,
            appeal_grounds=appeal_grounds,
            appeal_status='pending',
            appeal_deadline=appeal_deadline,
            appeal_response=None
        )
        
        # Update original decision
        original_decision = self.decision_index[original_decision_id]
        original_decision.appeal_info = appeal_info
        
        # Update metrics
        self.appeal_metrics['pending'] += 1
        
        # Log appeal registration
        await self.log_constitutional_decision(
            ConstitutionalDecisionType.APPEAL_RESOLUTION,
            GovernanceLevel.COMMUNITY,
            'appeal_system',
            original_decision.user_tier,
            appealing_party,
            f"Appeal registered for decision {original_decision_id}",
            DecisionOutcome.DEFERRED,
            {
                'primary_reasoning': f"Appeal registered with grounds: {', '.join(appeal_grounds)}",
                'constitutional_principles': ['due_process', 'right_to_appeal'],
                'precedents': [],
                'harm_analysis': {},
                'alternatives': [],
                'minority_opinions': [],
                'democratic_input': f'Appeal requested by {appealing_party}'
            },
            [],
            {
                'original_decision_id': original_decision_id,
                'appeal_grounds': appeal_grounds,
                'appeal_deadline': appeal_deadline
            }
        )
        
        self.logger.info(f"Registered appeal {appeal_id} for decision {original_decision_id}")
        
        return appeal_id
    
    async def resolve_appeal(self, 
                           appeal_id: str,
                           resolution: DecisionOutcome,
                           resolution_rationale: str,
                           resolver: str) -> str:
        """Resolve a pending appeal"""
        # Find decision with this appeal
        original_decision = None
        for decision in self.decision_logs:
            if decision.appeal_info and decision.appeal_info.appeal_id == appeal_id:
                original_decision = decision
                break
        
        if not original_decision:
            raise ValueError("Appeal not found")
        
        # Update appeal status
        original_decision.appeal_info.appeal_status = resolution.value
        original_decision.appeal_info.appeal_response = resolution_rationale
        
        # Update metrics
        self.appeal_metrics['pending'] -= 1
        self.appeal_metrics[resolution.value] += 1
        
        # Log appeal resolution
        resolution_log_id = await self.log_constitutional_decision(
            ConstitutionalDecisionType.APPEAL_RESOLUTION,
            GovernanceLevel.CONSTITUTIONAL,
            resolver,
            original_decision.user_tier,
            original_decision.appeal_info.appealing_party,
            f"Appeal {appeal_id} resolved: {resolution.value}",
            resolution,
            {
                'primary_reasoning': resolution_rationale,
                'constitutional_principles': ['due_process', 'constitutional_review'],
                'precedents': [],
                'harm_analysis': {},
                'alternatives': [],
                'minority_opinions': [],
                'democratic_input': f'Appeal resolved by {resolver}'
            },
            [],
            {
                'original_decision_id': original_decision.log_id,
                'appeal_id': appeal_id,
                'appeal_outcome': resolution.value
            }
        )
        
        self.logger.info(f"Resolved appeal {appeal_id}: {resolution.value}")
        
        return resolution_log_id
    
    async def _persist_decision_log(self, decision: ConstitutionalDecisionLog):
        """Persist decision log to storage"""
        date = time.strftime("%Y-%m-%d", time.localtime(decision.timestamp))
        log_file = self.storage_path / f"decisions_{date}.json"
        
        # Load existing data
        existing_data = {'decisions': [], 'metadata': {}}
        if log_file.exists():
            with open(log_file, 'r') as f:
                existing_data = json.load(f)
        
        # Add new decision
        decision_dict = asdict(decision)
        
        # Convert enums to strings for JSON serialization
        decision_dict['decision_type'] = decision.decision_type.value
        decision_dict['governance_level'] = decision.governance_level.value
        decision_dict['decision_outcome'] = decision.decision_outcome.value
        
        existing_data['decisions'].append(decision_dict)
        
        # Update metadata
        existing_data['metadata'] = {
            'last_updated': time.time(),
            'decision_count': len(existing_data['decisions']),
            'file_hash': hashlib.sha256(
                json.dumps(existing_data['decisions'], sort_keys=True).encode('utf-8')
            ).hexdigest()
        }
        
        # Write to file
        with open(log_file, 'w') as f:
            json.dump(existing_data, f, indent=2, default=str)
    
    def get_decision_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get analytics for constitutional decisions"""
        cutoff_time = time.time() - (time_range_hours * 3600)
        recent_decisions = [d for d in self.decision_logs if d.timestamp >= cutoff_time]
        
        # Decision type breakdown
        type_breakdown = defaultdict(int)
        outcome_breakdown = defaultdict(int)
        governance_breakdown = defaultdict(int)
        tier_breakdown = defaultdict(int)
        
        for decision in recent_decisions:
            type_breakdown[decision.decision_type.value] += 1
            outcome_breakdown[decision.decision_outcome.value] += 1
            governance_breakdown[decision.governance_level.value] += 1
            tier_breakdown[decision.user_tier] += 1
        
        # Appeal statistics
        appeals_pending = len([d for d in recent_decisions 
                             if d.appeal_info and d.appeal_info.appeal_status == 'pending'])
        appeals_resolved = len([d for d in recent_decisions 
                              if d.appeal_info and d.appeal_info.appeal_status != 'pending'])
        
        # Constitutional compliance
        constitutional_decisions = len([d for d in recent_decisions 
                                      if d.governance_level == GovernanceLevel.CONSTITUTIONAL])
        
        return {
            'time_range_hours': time_range_hours,
            'total_decisions': len(recent_decisions),
            'decision_types': dict(type_breakdown),
            'decision_outcomes': dict(outcome_breakdown),
            'governance_levels': dict(governance_breakdown),
            'user_tier_distribution': dict(tier_breakdown),
            'appeal_statistics': {
                'pending_appeals': appeals_pending,
                'resolved_appeals': appeals_resolved,
                'appeal_rate': (appeals_pending + appeals_resolved) / len(recent_decisions) * 100 if recent_decisions else 0
            },
            'constitutional_oversight': {
                'constitutional_decisions': constitutional_decisions,
                'oversight_rate': constitutional_decisions / len(recent_decisions) * 100 if recent_decisions else 0
            },
            'analysis_timestamp': time.time()
        }
    
    def get_constitutional_precedents(self, 
                                    decision_type: ConstitutionalDecisionType,
                                    limit: int = 10) -> List[Dict[str, Any]]:
        """Get constitutional precedents for specific decision type"""
        relevant_decisions = [
            d for d in self.decision_logs 
            if d.decision_type == decision_type 
            and d.governance_level in [GovernanceLevel.CONSTITUTIONAL, GovernanceLevel.SUPREME]
        ]
        
        # Sort by timestamp (most recent first)
        relevant_decisions.sort(key=lambda x: x.timestamp, reverse=True)
        
        precedents = []
        for decision in relevant_decisions[:limit]:
            precedent = {
                'log_id': decision.log_id,
                'timestamp': decision.timestamp,
                'decision_summary': decision.decision_summary,
                'outcome': decision.decision_outcome.value,
                'rationale_summary': decision.rationale.primary_reasoning,
                'constitutional_principles': decision.rationale.constitutional_principles_applied,
                'precedents_cited': decision.rationale.precedents_cited,
                'governance_level': decision.governance_level.value,
                'public_disclosure_level': decision.public_disclosure_level
            }
            precedents.append(precedent)
        
        return precedents
    
    def get_public_decision_summary(self, log_id: str) -> Optional[Dict[str, Any]]:
        """Get public summary of decision based on disclosure level"""
        if log_id not in self.decision_index:
            return None
        
        decision = self.decision_index[log_id]
        disclosure_level = decision.public_disclosure_level
        
        if disclosure_level == 'full_public':
            return decision.to_audit_format()
        
        elif disclosure_level == 'summary_public':
            return {
                'log_id': decision.log_id,
                'timestamp': decision.timestamp,
                'decision_type': decision.decision_type.value,
                'decision_outcome': decision.decision_outcome.value,
                'decision_summary': decision.decision_summary,
                'governance_level': decision.governance_level.value,
                'constitutional_principles': decision.rationale.constitutional_principles_applied
            }
        
        elif disclosure_level in ['statistical_only', 'aggregate_only']:
            return {
                'log_id': decision.log_id,
                'timestamp': decision.timestamp,
                'decision_type': decision.decision_type.value,
                'decision_outcome': decision.decision_outcome.value,
                'governance_level': decision.governance_level.value
            }
        
        elif disclosure_level in ['anonymized_summary', 'commitment_only']:
            return {
                'decision_type': decision.decision_type.value,
                'decision_outcome': decision.decision_outcome.value,
                'privacy_preserving_hash': decision.privacy_preserving_hash
            }
        
        return None

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_constitutional_logging():
        logger = ConstitutionalDecisionLogger()
        
        # Test logging a constitutional decision
        decision_id = await logger.log_constitutional_decision(
            ConstitutionalDecisionType.CONTENT_MODERATION,
            GovernanceLevel.MODERATED,
            'moderator_001',
            'bronze',
            'user_12345',
            'Content removed for constitutional harm violation',
            DecisionOutcome.APPROVED,
            {
                'primary_reasoning': 'Content contained H2-level harm that violates constitutional principles',
                'constitutional_principles': ['harm_prevention', 'democratic_discourse'],
                'precedents': ['case_001', 'case_012'],
                'harm_analysis': {'level': 'H2', 'confidence': 0.95, 'categories': ['misinformation']},
                'alternatives': ['warning_issued', 'content_flagged'],
                'minority_opinions': [],
                'democratic_input': 'Community reported content through democratic reporting system'
            },
            [
                {
                    'type': 'harm_classification',
                    'confidence': 0.95,
                    'data': {'classification': 'H2', 'categories': ['misinformation']},
                    'source': 'constitutional_classifier',
                    'documents': ['harm_policy_v2.1'],
                    'expert_opinions': ['constitutional_expert_001']
                }
            ],
            {
                'policy_context': 'content_moderation_policy_v1.0',
                'user_history': 'no_previous_violations',
                'community_sentiment': 'supportive_of_moderation'
            }
        )
        
        print(f"Logged decision: {decision_id}")
        
        # Test appeal registration
        appeal_id = await logger.register_appeal(
            decision_id,
            'user_12345',
            ['procedural_error', 'disproportionate_response']
        )
        
        print(f"Registered appeal: {appeal_id}")
        
        # Get decision analytics
        analytics = logger.get_decision_analytics()
        print(f"Decision analytics: {json.dumps(analytics, indent=2)}")
        
        # Get constitutional precedents
        precedents = logger.get_constitutional_precedents(ConstitutionalDecisionType.CONTENT_MODERATION)
        print(f"Constitutional precedents: {len(precedents)} found")
        
        # Get public summary
        public_summary = logger.get_public_decision_summary(decision_id)
        print(f"Public summary: {json.dumps(public_summary, indent=2)}")
    
    # Run test
    # asyncio.run(test_constitutional_logging())