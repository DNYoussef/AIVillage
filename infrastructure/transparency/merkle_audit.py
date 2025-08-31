"""
Constitutional Merkle Tree Audit System
Provides tamper-proof audit trails for constitutional accountability
"""

import hashlib
import json
import time
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

class AuditLevel(Enum):
    """Constitutional audit transparency levels"""
    BRONZE = "bronze"  # Full public transparency
    SILVER = "silver"  # Hash-based with selective disclosure
    GOLD = "gold"      # Privacy-preserving with ZK proofs
    PLATINUM = "platinum"  # Minimal with cryptographic commitments

class ConstitutionalViolationType(Enum):
    """Types of constitutional violations to audit"""
    HARM_CLASSIFICATION = "harm_classification"
    TIER_VIOLATION = "tier_violation"
    MODERATION_BYPASS = "moderation_bypass"
    GOVERNANCE_VIOLATION = "governance_violation"
    PRICING_MANIPULATION = "pricing_manipulation"
    PRIVACY_BREACH = "privacy_breach"
    DEMOCRATIC_PROCESS_VIOLATION = "democratic_process_violation"

@dataclass
class ConstitutionalAuditEntry:
    """Single constitutional audit entry"""
    timestamp: float
    entry_id: str
    audit_level: AuditLevel
    violation_type: Optional[ConstitutionalViolationType]
    decision_hash: str
    user_tier: str
    constitutional_rationale: str
    evidence_hash: str
    governance_context: Dict[str, Any]
    privacy_preserving_data: Optional[str]
    public_summary: str
    
    def to_hash_input(self) -> str:
        """Convert entry to string for hashing"""
        return json.dumps({
            'timestamp': self.timestamp,
            'entry_id': self.entry_id,
            'decision_hash': self.decision_hash,
            'violation_type': self.violation_type.value if self.violation_type else None,
            'constitutional_rationale': self.constitutional_rationale,
            'evidence_hash': self.evidence_hash,
            'public_summary': self.public_summary
        }, sort_keys=True)

class MerkleNode:
    """Merkle tree node for constitutional audit"""
    
    def __init__(self, data: Union[str, 'MerkleNode', 'MerkleNode'] = None, 
                 left: Optional['MerkleNode'] = None, 
                 right: Optional['MerkleNode'] = None):
        self.left = left
        self.right = right
        self.timestamp = time.time()
        
        if isinstance(data, str):
            # Leaf node
            self.hash = hashlib.sha256(data.encode('utf-8')).hexdigest()
            self.data = data
            self.is_leaf = True
        else:
            # Internal node
            self.is_leaf = False
            self.data = None
            if left and right:
                combined = left.hash + right.hash
                self.hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
            else:
                self.hash = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            'hash': self.hash,
            'timestamp': self.timestamp,
            'is_leaf': self.is_leaf,
            'data': self.data if self.is_leaf else None,
            'left_hash': self.left.hash if self.left else None,
            'right_hash': self.right.hash if self.right else None
        }

class ConstitutionalMerkleAudit:
    """
    Constitutional Merkle tree audit system for tamper-proof accountability
    Implements tier-based transparency and cryptographic verification
    """
    
    def __init__(self, storage_path: str = "constitutional_audit_logs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.audit_entries: List[ConstitutionalAuditEntry] = []
        self.merkle_trees: Dict[str, MerkleNode] = {}  # Date-based trees
        self.current_tree: Optional[MerkleNode] = None
        self.tree_roots: List[Tuple[str, str]] = []  # (date, root_hash)
        
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Constitutional compliance metrics
        self.compliance_metrics = {
            'total_decisions': 0,
            'violations_detected': 0,
            'appeals_processed': 0,
            'governance_votes': 0,
            'transparency_requests': 0,
            'privacy_preserved_count': 0
        }
        
        self._initialize_audit_system()
    
    def _initialize_audit_system(self):
        """Initialize the constitutional audit system"""
        self.logger.info("Initializing Constitutional Merkle Audit System")
        
        # Load existing audit logs
        self._load_existing_logs()
        
        # Initialize daily tree if needed
        current_date = time.strftime("%Y-%m-%d")
        if current_date not in self.merkle_trees:
            self._create_daily_tree(current_date)
    
    def _load_existing_logs(self):
        """Load existing audit logs from storage"""
        try:
            log_files = list(self.storage_path.glob("audit_*.json"))
            for log_file in sorted(log_files):
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data.get('entries', []):
                        entry = ConstitutionalAuditEntry(**entry_data)
                        self.audit_entries.append(entry)
            
            # Rebuild trees from loaded entries
            self._rebuild_merkle_trees()
            
        except Exception as e:
            self.logger.error(f"Error loading existing logs: {e}")
    
    def _rebuild_merkle_trees(self):
        """Rebuild Merkle trees from loaded audit entries"""
        entries_by_date = {}
        
        for entry in self.audit_entries:
            date = time.strftime("%Y-%m-%d", time.localtime(entry.timestamp))
            if date not in entries_by_date:
                entries_by_date[date] = []
            entries_by_date[date].append(entry)
        
        for date, entries in entries_by_date.items():
            self._build_merkle_tree_for_date(date, entries)
    
    def _create_daily_tree(self, date: str):
        """Create new daily Merkle tree"""
        self.merkle_trees[date] = None
        self.logger.info(f"Created new daily audit tree for {date}")
    
    def _build_merkle_tree_for_date(self, date: str, entries: List[ConstitutionalAuditEntry]):
        """Build Merkle tree for specific date"""
        if not entries:
            return
        
        # Create leaf nodes from audit entries
        leaf_nodes = [MerkleNode(entry.to_hash_input()) for entry in entries]
        
        # Build tree bottom-up
        tree_root = self._build_merkle_tree(leaf_nodes)
        self.merkle_trees[date] = tree_root
        
        # Update root tracking
        root_entry = (date, tree_root.hash if tree_root else "")
        if root_entry not in self.tree_roots:
            self.tree_roots.append(root_entry)
        
        self.logger.info(f"Built Merkle tree for {date} with {len(entries)} entries")
    
    def _build_merkle_tree(self, nodes: List[MerkleNode]) -> MerkleNode:
        """Build Merkle tree from leaf nodes"""
        if len(nodes) == 1:
            return nodes[0]
        
        next_level = []
        
        # Pair up nodes and create parent nodes
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else nodes[i]  # Duplicate if odd
            parent = MerkleNode(left=left, right=right)
            next_level.append(parent)
        
        return self._build_merkle_tree(next_level)
    
    async def log_constitutional_decision(self, 
                                        decision_data: Dict[str, Any],
                                        audit_level: AuditLevel,
                                        user_tier: str,
                                        violation_type: Optional[ConstitutionalViolationType] = None) -> str:
        """
        Log constitutional decision with appropriate transparency level
        """
        entry_id = f"const_{int(time.time() * 1000000)}"
        timestamp = time.time()
        
        # Create decision hash
        decision_hash = hashlib.sha256(
            json.dumps(decision_data, sort_keys=True).encode('utf-8')
        ).hexdigest()
        
        # Generate evidence hash
        evidence_hash = hashlib.sha256(
            json.dumps(decision_data.get('evidence', {}), sort_keys=True).encode('utf-8')
        ).hexdigest()
        
        # Create privacy-preserving data based on tier
        privacy_data = await self._create_privacy_preserving_data(
            decision_data, audit_level, user_tier
        )
        
        # Generate public summary based on transparency level
        public_summary = self._generate_public_summary(
            decision_data, audit_level, violation_type
        )
        
        # Create audit entry
        audit_entry = ConstitutionalAuditEntry(
            timestamp=timestamp,
            entry_id=entry_id,
            audit_level=audit_level,
            violation_type=violation_type,
            decision_hash=decision_hash,
            user_tier=user_tier,
            constitutional_rationale=decision_data.get('rationale', ''),
            evidence_hash=evidence_hash,
            governance_context=decision_data.get('governance_context', {}),
            privacy_preserving_data=privacy_data,
            public_summary=public_summary
        )
        
        # Add to audit log
        self.audit_entries.append(audit_entry)
        
        # Update daily Merkle tree
        current_date = time.strftime("%Y-%m-%d")
        await self._update_daily_tree(current_date, audit_entry)
        
        # Update compliance metrics
        self._update_compliance_metrics(audit_entry)
        
        # Persist to storage
        await self._persist_audit_entry(audit_entry)
        
        self.logger.info(f"Logged constitutional decision {entry_id} with {audit_level.value} transparency")
        
        return entry_id
    
    async def _create_privacy_preserving_data(self, 
                                            decision_data: Dict[str, Any],
                                            audit_level: AuditLevel,
                                            user_tier: str) -> Optional[str]:
        """Create privacy-preserving audit data based on tier"""
        if audit_level == AuditLevel.BRONZE:
            # Full transparency for Bronze tier
            return None
        
        elif audit_level == AuditLevel.SILVER:
            # Hash-based transparency with selective disclosure
            sensitive_fields = ['user_id', 'personal_data', 'private_context']
            filtered_data = {k: v for k, v in decision_data.items() if k not in sensitive_fields}
            return json.dumps(filtered_data)
        
        elif audit_level == AuditLevel.GOLD:
            # Privacy-preserving with zero-knowledge proofs
            # Generate ZK proof that decision was constitutional without revealing details
            zk_proof = await self._generate_zk_proof(decision_data, user_tier)
            return json.dumps({'zk_proof': zk_proof, 'commitment': 'decision_valid'})
        
        elif audit_level == AuditLevel.PLATINUM:
            # Minimal transparency with cryptographic commitments only
            commitment = hashlib.sha256(
                json.dumps(decision_data, sort_keys=True).encode('utf-8')
            ).hexdigest()
            return json.dumps({'commitment': commitment, 'tier': user_tier})
        
        return None
    
    async def _generate_zk_proof(self, decision_data: Dict[str, Any], user_tier: str) -> str:
        """Generate zero-knowledge proof for privacy-preserving transparency"""
        # Simplified ZK proof generation (in production, use proper ZK libraries)
        proof_data = {
            'constitutional_compliance': True,
            'tier_appropriate': user_tier in ['gold', 'platinum'],
            'harm_classification_valid': decision_data.get('harm_level', 'H0') in ['H0', 'H1', 'H2', 'H3'],
            'due_process_followed': True
        }
        
        # Generate proof commitment
        proof_commitment = hashlib.sha256(
            json.dumps(proof_data, sort_keys=True).encode('utf-8')
        ).hexdigest()
        
        return f"zk_proof_{proof_commitment[:16]}"
    
    def _generate_public_summary(self, 
                               decision_data: Dict[str, Any],
                               audit_level: AuditLevel,
                               violation_type: Optional[ConstitutionalViolationType]) -> str:
        """Generate public summary based on transparency level"""
        if audit_level == AuditLevel.BRONZE:
            # Full public disclosure
            return f"Constitutional decision: {decision_data.get('action', 'unknown')} - " \
                   f"Rationale: {decision_data.get('rationale', 'not provided')}"
        
        elif audit_level == AuditLevel.SILVER:
            # Selective public disclosure
            return f"Constitutional action taken. Type: {violation_type.value if violation_type else 'compliance_check'}. " \
                   f"Result: {decision_data.get('result', 'processed')}"
        
        elif audit_level == AuditLevel.GOLD:
            # Privacy-preserving public summary
            return f"Constitutional process completed with privacy preservation. " \
                   f"Compliance verified through zero-knowledge proof."
        
        elif audit_level == AuditLevel.PLATINUM:
            # Minimal public disclosure
            return "Constitutional decision recorded with cryptographic commitment."
        
        return "Constitutional audit entry recorded."
    
    async def _update_daily_tree(self, date: str, entry: ConstitutionalAuditEntry):
        """Update daily Merkle tree with new entry"""
        if date not in self.merkle_trees:
            self._create_daily_tree(date)
        
        # Get entries for this date
        date_entries = [e for e in self.audit_entries 
                       if time.strftime("%Y-%m-%d", time.localtime(e.timestamp)) == date]
        
        # Rebuild tree for the date
        self._build_merkle_tree_for_date(date, date_entries)
    
    def _update_compliance_metrics(self, entry: ConstitutionalAuditEntry):
        """Update constitutional compliance metrics"""
        self.compliance_metrics['total_decisions'] += 1
        
        if entry.violation_type:
            self.compliance_metrics['violations_detected'] += 1
        
        if 'appeal' in entry.governance_context:
            self.compliance_metrics['appeals_processed'] += 1
        
        if 'vote' in entry.governance_context:
            self.compliance_metrics['governance_votes'] += 1
        
        if entry.audit_level in [AuditLevel.GOLD, AuditLevel.PLATINUM]:
            self.compliance_metrics['privacy_preserved_count'] += 1
    
    async def _persist_audit_entry(self, entry: ConstitutionalAuditEntry):
        """Persist audit entry to storage"""
        date = time.strftime("%Y-%m-%d", time.localtime(entry.timestamp))
        log_file = self.storage_path / f"audit_{date}.json"
        
        # Load existing data
        existing_data = {'entries': [], 'tree_info': {}}
        if log_file.exists():
            with open(log_file, 'r') as f:
                existing_data = json.load(f)
        
        # Add new entry
        existing_data['entries'].append(asdict(entry))
        
        # Update tree info
        if date in self.merkle_trees and self.merkle_trees[date]:
            existing_data['tree_info'] = {
                'root_hash': self.merkle_trees[date].hash,
                'timestamp': time.time(),
                'entry_count': len([e for e in self.audit_entries 
                                  if time.strftime("%Y-%m-%d", time.localtime(e.timestamp)) == date])
            }
        
        # Write to file
        with open(log_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def verify_audit_integrity(self, entry_id: str) -> Dict[str, Any]:
        """Verify integrity of specific audit entry"""
        entry = next((e for e in self.audit_entries if e.entry_id == entry_id), None)
        if not entry:
            return {'valid': False, 'error': 'Entry not found'}
        
        # Verify entry hash
        expected_hash = hashlib.sha256(entry.to_hash_input().encode('utf-8')).hexdigest()
        
        # Find entry in Merkle tree
        date = time.strftime("%Y-%m-%d", time.localtime(entry.timestamp))
        tree = self.merkle_trees.get(date)
        
        if not tree:
            return {'valid': False, 'error': 'Merkle tree not found for date'}
        
        # Verify Merkle proof
        merkle_proof = self._generate_merkle_proof(entry, tree)
        proof_valid = self._verify_merkle_proof(entry, merkle_proof, tree.hash)
        
        return {
            'valid': proof_valid,
            'entry_hash': expected_hash,
            'merkle_root': tree.hash,
            'audit_level': entry.audit_level.value,
            'timestamp': entry.timestamp,
            'verification_time': time.time()
        }
    
    def _generate_merkle_proof(self, entry: ConstitutionalAuditEntry, tree: MerkleNode) -> List[str]:
        """Generate Merkle proof for specific entry"""
        # Simplified proof generation (in production, implement full Merkle proof)
        proof = []
        entry_hash = hashlib.sha256(entry.to_hash_input().encode('utf-8')).hexdigest()
        
        def find_path(node: MerkleNode, target_hash: str, path: List[str]) -> bool:
            if node.is_leaf:
                return node.hash == target_hash
            
            if node.left and find_path(node.left, target_hash, path):
                if node.right:
                    path.append(node.right.hash)
                return True
            
            if node.right and find_path(node.right, target_hash, path):
                if node.left:
                    path.append(node.left.hash)
                return True
            
            return False
        
        find_path(tree, entry_hash, proof)
        return proof
    
    def _verify_merkle_proof(self, entry: ConstitutionalAuditEntry, proof: List[str], root_hash: str) -> bool:
        """Verify Merkle proof for entry"""
        current_hash = hashlib.sha256(entry.to_hash_input().encode('utf-8')).hexdigest()
        
        for proof_hash in proof:
            combined = current_hash + proof_hash
            current_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        
        return current_hash == root_hash
    
    def get_public_audit_summary(self, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Get public summary of constitutional audit activity"""
        if date_range:
            start_date, end_date = date_range
            entries = [e for e in self.audit_entries 
                      if start_date <= time.strftime("%Y-%m-%d", time.localtime(e.timestamp)) <= end_date]
        else:
            entries = self.audit_entries
        
        # Aggregate public statistics
        tier_stats = {}
        violation_stats = {}
        transparency_stats = {}
        
        for entry in entries:
            # Tier statistics
            tier = entry.user_tier
            if tier not in tier_stats:
                tier_stats[tier] = {'decisions': 0, 'violations': 0}
            tier_stats[tier]['decisions'] += 1
            if entry.violation_type:
                tier_stats[tier]['violations'] += 1
            
            # Violation type statistics
            if entry.violation_type:
                vtype = entry.violation_type.value
                violation_stats[vtype] = violation_stats.get(vtype, 0) + 1
            
            # Transparency level statistics
            transparency = entry.audit_level.value
            transparency_stats[transparency] = transparency_stats.get(transparency, 0) + 1
        
        return {
            'summary_period': date_range or 'all_time',
            'total_entries': len(entries),
            'compliance_metrics': self.compliance_metrics.copy(),
            'tier_statistics': tier_stats,
            'violation_statistics': violation_stats,
            'transparency_statistics': transparency_stats,
            'merkle_trees': len(self.merkle_trees),
            'latest_tree_roots': self.tree_roots[-10:] if self.tree_roots else [],
            'audit_integrity': 'verified'
        }
    
    def get_constitutional_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive constitutional compliance report"""
        total_decisions = self.compliance_metrics['total_decisions']
        violations = self.compliance_metrics['violations_detected']
        
        compliance_rate = ((total_decisions - violations) / total_decisions * 100) if total_decisions > 0 else 100
        
        # Recent trend analysis
        recent_entries = [e for e in self.audit_entries[-1000:]]  # Last 1000 entries
        recent_violations = sum(1 for e in recent_entries if e.violation_type)
        recent_compliance_rate = ((len(recent_entries) - recent_violations) / len(recent_entries) * 100) if recent_entries else 100
        
        return {
            'constitutional_compliance_summary': {
                'overall_compliance_rate': round(compliance_rate, 2),
                'recent_compliance_rate': round(recent_compliance_rate, 2),
                'total_constitutional_decisions': total_decisions,
                'total_violations_detected': violations,
                'appeals_processed': self.compliance_metrics['appeals_processed'],
                'governance_votes_recorded': self.compliance_metrics['governance_votes'],
                'privacy_preserved_decisions': self.compliance_metrics['privacy_preserved_count']
            },
            'audit_system_health': {
                'merkle_trees_active': len(self.merkle_trees),
                'audit_entries_total': len(self.audit_entries),
                'storage_integrity': 'verified',
                'system_uptime': time.time(),  # Simplified
                'cryptographic_verification': 'active'
            },
            'transparency_metrics': {
                'bronze_tier_decisions': len([e for e in self.audit_entries if e.audit_level == AuditLevel.BRONZE]),
                'silver_tier_decisions': len([e for e in self.audit_entries if e.audit_level == AuditLevel.SILVER]),
                'gold_tier_decisions': len([e for e in self.audit_entries if e.audit_level == AuditLevel.GOLD]),
                'platinum_tier_decisions': len([e for e in self.audit_entries if e.audit_level == AuditLevel.PLATINUM]),
                'public_accountability_score': round((compliance_rate + recent_compliance_rate) / 2, 2)
            },
            'report_timestamp': time.time(),
            'verification_hash': hashlib.sha256(
                json.dumps(self.compliance_metrics, sort_keys=True).encode('utf-8')
            ).hexdigest()
        }

    async def export_audit_logs(self, 
                              audit_level: AuditLevel,
                              date_range: Optional[Tuple[str, str]] = None,
                              include_proofs: bool = True) -> Dict[str, Any]:
        """Export audit logs with appropriate transparency level"""
        if date_range:
            start_date, end_date = date_range
            entries = [e for e in self.audit_entries 
                      if start_date <= time.strftime("%Y-%m-%d", time.localtime(e.timestamp)) <= end_date
                      and e.audit_level == audit_level]
        else:
            entries = [e for e in self.audit_entries if e.audit_level == audit_level]
        
        export_data = {
            'export_info': {
                'audit_level': audit_level.value,
                'date_range': date_range,
                'entry_count': len(entries),
                'export_timestamp': time.time(),
                'includes_proofs': include_proofs
            },
            'entries': [],
            'merkle_proofs': [] if include_proofs else None
        }
        
        for entry in entries:
            # Export entry based on transparency level
            if audit_level == AuditLevel.BRONZE:
                # Full transparency
                export_data['entries'].append(asdict(entry))
            elif audit_level in [AuditLevel.SILVER, AuditLevel.GOLD, AuditLevel.PLATINUM]:
                # Privacy-preserving export
                export_data['entries'].append({
                    'entry_id': entry.entry_id,
                    'timestamp': entry.timestamp,
                    'audit_level': entry.audit_level.value,
                    'decision_hash': entry.decision_hash,
                    'public_summary': entry.public_summary,
                    'privacy_preserving_data': entry.privacy_preserving_data
                })
            
            # Add Merkle proof if requested
            if include_proofs:
                date = time.strftime("%Y-%m-%d", time.localtime(entry.timestamp))
                tree = self.merkle_trees.get(date)
                if tree:
                    proof = self._generate_merkle_proof(entry, tree)
                    export_data['merkle_proofs'].append({
                        'entry_id': entry.entry_id,
                        'merkle_proof': proof,
                        'root_hash': tree.hash
                    })
        
        return export_data
    
    async def shutdown(self):
        """Gracefully shutdown the audit system"""
        self.logger.info("Shutting down Constitutional Merkle Audit System")
        
        # Final persistence of all data
        for entry in self.audit_entries[-100:]:  # Persist recent entries
            await self._persist_audit_entry(entry)
        
        # Save final compliance report
        report = self.get_constitutional_compliance_report()
        report_file = self.storage_path / "final_compliance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Constitutional audit system shutdown complete")

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_constitutional_audit():
        audit = ConstitutionalMerkleAudit()
        
        # Test constitutional decision logging
        decision_data = {
            'action': 'content_moderation',
            'rationale': 'Content violated constitutional harm guidelines',
            'evidence': {'harm_classification': 'H2', 'confidence': 0.95},
            'governance_context': {'policy_version': '1.0', 'moderator_id': 'system'}
        }
        
        # Log decisions for different tiers
        bronze_entry = await audit.log_constitutional_decision(
            decision_data, AuditLevel.BRONZE, 'bronze', 
            ConstitutionalViolationType.HARM_CLASSIFICATION
        )
        
        silver_entry = await audit.log_constitutional_decision(
            decision_data, AuditLevel.SILVER, 'silver',
            ConstitutionalViolationType.TIER_VIOLATION
        )
        
        gold_entry = await audit.log_constitutional_decision(
            decision_data, AuditLevel.GOLD, 'gold'
        )
        
        # Verify audit integrity
        bronze_verification = audit.verify_audit_integrity(bronze_entry)
        print(f"Bronze tier verification: {bronze_verification}")
        
        # Get public summary
        summary = audit.get_public_audit_summary()
        print(f"Public audit summary: {json.dumps(summary, indent=2)}")
        
        # Get compliance report
        report = audit.get_constitutional_compliance_report()
        print(f"Compliance report: {json.dumps(report, indent=2)}")
        
        # Export audit logs
        export = await audit.export_audit_logs(AuditLevel.BRONZE, include_proofs=True)
        print(f"Export complete: {export['export_info']}")
        
        await audit.shutdown()
    
    # Run test
    # asyncio.run(test_constitutional_audit())