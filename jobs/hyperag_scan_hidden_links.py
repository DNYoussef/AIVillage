#!/usr/bin/env python3
"""HypeRAG Hidden-Link Batch Scanner

Nightly cron job to surface candidate missing edges through co-mention analysis
and divergent retrieval. Processes high co-mention entity pairs and validates
them through the Innovator → Guardian pipeline.

Usage:
    python jobs/hyperag_scan_hidden_links.py [--config CONFIG_PATH] [--dry-run]
"""

import argparse
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

# HypeRAG imports
try:
    from mcp_servers.hyperag.gdc.specs import Violation
    from mcp_servers.hyperag.guardian.gate import CreativeBridge, GuardianGate
    from mcp_servers.hyperag.repair.innovator_agent import (
        InnovatorAgent,
        RepairOperation,
    )
    from mcp_servers.hyperag.retrieval.hybrid_retriever import HybridRetriever
except ImportError as e:
    print(f"Warning: HypeRAG modules not available: {e}")

    # Mock classes for development
    class HybridRetriever:
        pass

    class InnovatorAgent:
        pass

    class GuardianGate:
        pass

    class CreativeBridge:
        pass

    class Violation:
        pass


@dataclass
class CoMentionPair:
    """Entity pair with co-mention statistics."""

    entity1: str
    entity2: str
    co_mention_count: int
    confidence: float
    contexts: list[str] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.now)

    @property
    def pair_key(self) -> str:
        """Normalized pair key for deduplication."""
        entities = sorted([self.entity1, self.entity2])
        return f"{entities[0]}|{entities[1]}"


@dataclass
class CandidateEdge:
    """Candidate missing edge discovered by scanner."""

    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    scan_id: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class ScanMetrics:
    """Metrics for hidden link scan run."""

    scan_id: str
    start_time: datetime
    end_time: datetime | None = None

    # Input metrics
    hippo_log_entries_processed: int = 0
    co_mention_pairs_found: int = 0
    high_confidence_pairs: int = 0

    # Retrieval metrics
    divergent_retrieval_calls: int = 0
    candidate_edges_discovered: int = 0

    # Pipeline metrics
    innovator_proposals_generated: int = 0
    guardian_evaluations: int = 0
    guardian_approved: int = 0
    guardian_quarantined: int = 0
    guardian_rejected: int = 0

    # Performance
    total_time_seconds: float = 0.0
    avg_retrieval_time_ms: float = 0.0

    # Errors
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "scan_id": self.scan_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "input_metrics": {
                "hippo_log_entries_processed": self.hippo_log_entries_processed,
                "co_mention_pairs_found": self.co_mention_pairs_found,
                "high_confidence_pairs": self.high_confidence_pairs,
            },
            "retrieval_metrics": {
                "divergent_retrieval_calls": self.divergent_retrieval_calls,
                "candidate_edges_discovered": self.candidate_edges_discovered,
            },
            "pipeline_metrics": {
                "innovator_proposals_generated": self.innovator_proposals_generated,
                "guardian_evaluations": self.guardian_evaluations,
                "guardian_approved": self.guardian_approved,
                "guardian_quarantined": self.guardian_quarantined,
                "guardian_rejected": self.guardian_rejected,
            },
            "performance": {
                "total_time_seconds": self.total_time_seconds,
                "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }


class HippoIndexAnalyzer:
    """Analyzes Hippo-Index logs for high co-mention entity pairs."""

    def __init__(self, log_path: Path, lookback_hours: int = 24):
        """Initialize analyzer.

        Args:
            log_path: Path to Hippo-Index log directory
            lookback_hours: Hours to look back for log analysis
        """
        self.log_path = Path(log_path)
        self.lookback_hours = lookback_hours
        self.logger = logging.getLogger(f"{__name__}.HippoAnalyzer")

        # Co-mention patterns
        self.entity_pattern = re.compile(r"\[ENTITY:([^\]]+)\]")
        self.co_mention_pattern = re.compile(r"\[COMENTION:([^\]]+)\|([^\]]+)\]")

    def analyze_logs(self, min_co_mentions: int = 3) -> list[CoMentionPair]:
        """Analyze Hippo-Index logs for entity co-mentions.

        Args:
            min_co_mentions: Minimum co-mentions to consider a pair

        Returns:
            List of high co-mention entity pairs
        """
        self.logger.info(f"Analyzing Hippo-Index logs from {self.log_path}")

        cutoff_time = datetime.now() - timedelta(hours=self.lookback_hours)
        co_mention_counts = defaultdict(int)
        pair_contexts = defaultdict(list)
        pair_last_seen = {}

        # Find log files in time range
        log_files = self._find_recent_log_files(cutoff_time)
        self.logger.info(f"Found {len(log_files)} recent log files")

        for log_file in log_files:
            try:
                self._process_log_file(
                    log_file, co_mention_counts, pair_contexts, pair_last_seen
                )
            except Exception as e:
                self.logger.warning(f"Failed to process {log_file}: {e}")

        # Convert to CoMentionPair objects
        pairs = []
        for pair_key, count in co_mention_counts.items():
            if count >= min_co_mentions:
                entity1, entity2 = pair_key.split("|")
                confidence = min(1.0, count / 10.0)  # Simple confidence scoring

                pair = CoMentionPair(
                    entity1=entity1,
                    entity2=entity2,
                    co_mention_count=count,
                    confidence=confidence,
                    contexts=pair_contexts[pair_key][:5],  # Keep top 5 contexts
                    last_seen=pair_last_seen.get(pair_key, datetime.now()),
                )
                pairs.append(pair)

        # Sort by co-mention count descending
        pairs.sort(key=lambda p: p.co_mention_count, reverse=True)

        self.logger.info(
            f"Found {len(pairs)} high co-mention pairs (min: {min_co_mentions})"
        )
        return pairs

    def _find_recent_log_files(self, cutoff_time: datetime) -> list[Path]:
        """Find log files modified since cutoff time."""
        if not self.log_path.exists():
            self.logger.warning(f"Log path does not exist: {self.log_path}")
            return []

        log_files = []
        for log_file in self.log_path.glob("*.log"):
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if mtime >= cutoff_time:
                    log_files.append(log_file)
            except OSError:
                continue

        return sorted(log_files, key=lambda f: f.stat().st_mtime, reverse=True)

    def _process_log_file(
        self,
        log_file: Path,
        co_mention_counts: dict,
        pair_contexts: dict,
        pair_last_seen: dict,
    ):
        """Process a single log file for co-mentions."""
        with open(log_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse timestamp
                    if not line.strip():
                        continue

                    # Look for co-mention patterns
                    co_mentions = self.co_mention_pattern.findall(line)
                    for entity1, entity2 in co_mentions:
                        # Normalize pair key
                        entities = sorted([entity1.strip(), entity2.strip()])
                        pair_key = f"{entities[0]}|{entities[1]}"

                        co_mention_counts[pair_key] += 1

                        # Extract context (surrounding text)
                        context = line.strip()[:200]  # First 200 chars as context
                        if context not in pair_contexts[pair_key]:
                            pair_contexts[pair_key].append(context)

                        # Update last seen
                        pair_last_seen[pair_key] = datetime.now()

                    # Also look for entities mentioned together in same line
                    entities = self.entity_pattern.findall(line)
                    if len(entities) >= 2:
                        # Create pairs from entities in same line
                        for i in range(len(entities)):
                            for j in range(i + 1, len(entities)):
                                entity1, entity2 = sorted(
                                    [entities[i].strip(), entities[j].strip()]
                                )
                                pair_key = f"{entity1}|{entity2}"

                                co_mention_counts[
                                    pair_key
                                ] += 0.5  # Lower weight for implicit co-mention

                                context = line.strip()[:200]
                                if context not in pair_contexts[pair_key]:
                                    pair_contexts[pair_key].append(context)

                                pair_last_seen[pair_key] = datetime.now()

                except Exception as e:
                    self.logger.debug(
                        f"Error processing line {line_num} in {log_file}: {e}"
                    )


class DivergentRetrieverScanner:
    """Scanner using DivergentRetriever in scan mode."""

    def __init__(self, retriever: HybridRetriever):
        """Initialize scanner.

        Args:
            retriever: HybridRetriever instance
        """
        self.retriever = retriever
        self.logger = logging.getLogger(f"{__name__}.DivergentScanner")

    async def scan_entity_pairs(
        self, pairs: list[CoMentionPair], n_candidates: int = 3
    ) -> list[CandidateEdge]:
        """Scan entity pairs using DivergentRetriever.

        Args:
            pairs: Co-mention pairs to scan
            n_candidates: Number of candidates per pair

        Returns:
            List of candidate edges
        """
        self.logger.info(f"Scanning {len(pairs)} entity pairs with DivergentRetriever")

        candidate_edges = []

        for pair in pairs:
            try:
                start_time = time.time()

                # Create scan query for entity pair
                query = f"relationship between {pair.entity1} and {pair.entity2}"

                # Call DivergentRetriever in scan mode
                # Note: This would be the actual retriever call in production
                candidates = await self._mock_divergent_retrieval(
                    query, pair, n_candidates
                )

                candidate_edges.extend(candidates)

                elapsed_ms = (time.time() - start_time) * 1000
                self.logger.debug(
                    f"Scanned pair {pair.pair_key}: {len(candidates)} candidates ({elapsed_ms:.1f}ms)"
                )

            except Exception as e:
                self.logger.error(f"Failed to scan pair {pair.pair_key}: {e}")

        self.logger.info(f"Found {len(candidate_edges)} total candidate edges")
        return candidate_edges

    async def _mock_divergent_retrieval(
        self, query: str, pair: CoMentionPair, n_candidates: int
    ) -> list[CandidateEdge]:
        """Mock divergent retrieval (replace with actual implementation).

        Args:
            query: Scan query
            pair: Entity pair
            n_candidates: Number of candidates to return

        Returns:
            List of candidate edges
        """
        # Simulate divergent retrieval
        await asyncio.sleep(0.01)  # Simulate processing time

        # Mock relationship types based on entity names
        relationship_types = ["RELATED_TO", "ASSOCIATED_WITH", "INFLUENCES"]

        # For medical entities, add medical relationships
        if any(
            term in f"{pair.entity1} {pair.entity2}".lower()
            for term in ["drug", "patient", "treatment", "disease"]
        ):
            relationship_types.extend(["TREATS", "PRESCRIBED_FOR", "DIAGNOSED_WITH"])

        candidates = []
        for i in range(min(n_candidates, len(relationship_types))):
            confidence = max(0.5, pair.confidence - (i * 0.1))

            candidate = CandidateEdge(
                source_entity=pair.entity1,
                target_entity=pair.entity2,
                relationship_type=relationship_types[i],
                confidence=confidence,
                evidence=pair.contexts[:2],  # Use co-mention contexts as evidence
                scan_id=f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            candidates.append(candidate)

        return candidates


class HiddenLinkScanner:
    """Main hidden link scanner orchestrating the full pipeline."""

    def __init__(self, config: dict[str, Any]):
        """Initialize scanner.

        Args:
            config: Scanner configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.hippo_analyzer = HippoIndexAnalyzer(
            log_path=Path(config.get("hippo_log_path", "data/hippo_logs")),
            lookback_hours=config.get("lookback_hours", 24),
        )

        # Initialize HypeRAG components (would be real in production)
        self.retriever = None  # HybridRetriever()
        self.innovator = None  # InnovatorAgent.create_default()
        self.guardian = None  # GuardianGate()

        self.divergent_scanner = DivergentRetrieverScanner(self.retriever)

        # Metrics
        self.metrics = ScanMetrics(
            scan_id=f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
        )

    async def run_scan(self, dry_run: bool = False) -> ScanMetrics:
        """Run complete hidden link scan.

        Args:
            dry_run: If True, don't actually apply changes

        Returns:
            Scan metrics
        """
        self.logger.info(f"Starting hidden link scan {self.metrics.scan_id}")

        try:
            # Step 1: Analyze Hippo-Index logs
            await self._step1_analyze_logs()

            # Step 2: Scan with DivergentRetriever
            candidate_edges = await self._step2_divergent_scan()

            # Step 3: Process through Innovator → Guardian pipeline
            await self._step3_pipeline_processing(candidate_edges, dry_run)

            # Step 4: Generate metrics and reports
            await self._step4_generate_metrics()

        except Exception as e:
            self.logger.error(f"Scan failed: {e}")
            self.metrics.errors.append(str(e))

        finally:
            self.metrics.end_time = datetime.now()
            self.metrics.total_time_seconds = (
                self.metrics.end_time - self.metrics.start_time
            ).total_seconds()

        self.logger.info(
            f"Scan {self.metrics.scan_id} completed in {self.metrics.total_time_seconds:.1f}s"
        )
        return self.metrics

    async def _step1_analyze_logs(self):
        """Step 1: Query Hippo-Index logs for co-mention pairs."""
        self.logger.info("Step 1: Analyzing Hippo-Index logs")

        min_co_mentions = self.config.get("min_co_mentions", 3)
        self.co_mention_pairs = self.hippo_analyzer.analyze_logs(min_co_mentions)

        self.metrics.co_mention_pairs_found = len(self.co_mention_pairs)
        self.metrics.high_confidence_pairs = len(
            [p for p in self.co_mention_pairs if p.confidence >= 0.7]
        )

        self.logger.info(f"Found {len(self.co_mention_pairs)} co-mention pairs")

    async def _step2_divergent_scan(self) -> list[CandidateEdge]:
        """Step 2: Call DivergentRetriever in scan mode."""
        self.logger.info("Step 2: Scanning with DivergentRetriever")

        # Limit to top pairs for performance
        max_pairs = self.config.get("max_pairs_to_scan", 50)
        pairs_to_scan = self.co_mention_pairs[:max_pairs]

        n_candidates = self.config.get("candidates_per_pair", 3)
        candidate_edges = await self.divergent_scanner.scan_entity_pairs(
            pairs_to_scan, n_candidates
        )

        self.metrics.divergent_retrieval_calls = len(pairs_to_scan)
        self.metrics.candidate_edges_discovered = len(candidate_edges)

        return candidate_edges

    async def _step3_pipeline_processing(
        self, candidate_edges: list[CandidateEdge], dry_run: bool
    ):
        """Step 3: Process candidates through Innovator -> Guardian pipeline."""
        self.logger.info("Step 3: Processing through Innovator -> Guardian pipeline")

        if not candidate_edges:
            self.logger.info("No candidate edges to process")
            return

        for candidate in candidate_edges:
            try:
                # Convert candidate to mock proposal for pipeline
                proposal = await self._create_mock_proposal(candidate)
                self.metrics.innovator_proposals_generated += 1

                # Evaluate with Guardian Gate
                decision = await self._evaluate_with_guardian(
                    candidate, proposal, dry_run
                )
                self.metrics.guardian_evaluations += 1

                # Track decision
                if decision == "APPLY":
                    self.metrics.guardian_approved += 1
                elif decision == "QUARANTINE":
                    self.metrics.guardian_quarantined += 1
                else:
                    self.metrics.guardian_rejected += 1

            except Exception as e:
                self.logger.warning(
                    f"Failed to process candidate {candidate.source_entity}->{candidate.target_entity}: {e}"
                )
                self.metrics.errors.append(str(e))

    async def _step4_generate_metrics(self):
        """Step 4: Generate metrics and write reports."""
        self.logger.info("Step 4: Generating metrics and reports")

        # Write metrics to file
        metrics_path = (
            Path("data/scan_metrics") / f"{self.metrics.scan_id}_metrics.json"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_path, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

        self.logger.info(f"Metrics written to {metrics_path}")

        # Log summary
        self._log_summary()

    def _log_summary(self):
        """Log scan summary."""
        summary = f"""
Hidden Link Scan Summary ({self.metrics.scan_id}):
  Input: {self.metrics.co_mention_pairs_found} co-mention pairs
  Candidates: {self.metrics.candidate_edges_discovered} edges discovered
  Guardian Results:
    - Approved: {self.metrics.guardian_approved}
    - Quarantined: {self.metrics.guardian_quarantined}
    - Rejected: {self.metrics.guardian_rejected}
  Performance: {self.metrics.total_time_seconds:.1f}s total
  Errors: {len(self.metrics.errors)}
        """.strip()

        self.logger.info(summary)

    async def _create_mock_proposal(self, candidate: CandidateEdge):
        """Create mock repair proposal from candidate edge."""
        # Mock implementation - would create actual RepairOperation in production
        return {
            "operation_type": "add_edge",
            "source_entity": candidate.source_entity,
            "target_entity": candidate.target_entity,
            "relationship_type": candidate.relationship_type,
            "confidence": candidate.confidence,
            "rationale": f"Hidden link discovered via co-mention analysis: {candidate.evidence[:1]}",
        }

    async def _evaluate_with_guardian(
        self, candidate: CandidateEdge, proposal: dict, dry_run: bool
    ) -> str:
        """Evaluate candidate with Guardian Gate."""
        # Mock implementation - would use actual GuardianGate in production
        if dry_run:
            self.logger.debug(
                f"DRY RUN: Would evaluate {candidate.relationship_type} edge"
            )

        # Mock decision based on confidence
        if candidate.confidence >= 0.8:
            return "APPLY"
        if candidate.confidence >= 0.5:
            return "QUARANTINE"
        return "REJECT"


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load scanner configuration."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return json.load(f)

    # Default configuration
    return {
        "hippo_log_path": "../data/hippo_logs",
        "lookback_hours": 24,
        "min_co_mentions": 3,
        "max_pairs_to_scan": 50,
        "candidates_per_pair": 3,
        "guardian": {"policy_path": "mcp_servers/hyperag/guardian/policies.yaml"},
        "dry_run": False,
    }


async def main():
    """Main entry point for hidden link scanner."""
    parser = argparse.ArgumentParser(description="HypeRAG Hidden-Link Batch Scanner")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--dry-run", action="store_true", help="Don't apply changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Create log directory
    log_dir = Path.cwd().parent / "data" / "scan_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                str(
                    log_dir
                    / f"hidden_link_scan_{datetime.now().strftime('%Y%m%d')}.log"
                )
            ),
        ],
    )

    # Load configuration
    config = load_config(args.config)
    if args.dry_run:
        config["dry_run"] = True

    # Run scanner
    scanner = HiddenLinkScanner(config)
    metrics = await scanner.run_scan(dry_run=config.get("dry_run", False))

    # Exit with error code if scan had errors
    if metrics.errors:
        print(f"Scan completed with {len(metrics.errors)} errors")
        return 1

    print(
        f"Scan completed successfully: {metrics.guardian_approved} approved, {metrics.guardian_quarantined} quarantined"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
