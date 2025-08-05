#!/usr/bin/env python3
"""Test script for HypeRAG Hidden-Link Scanner

Validates scanner components with mock data to ensure functionality.
"""

import asyncio
from datetime import datetime
from pathlib import Path
import tempfile

from hyperag_scan_hidden_links import (
    CoMentionPair,
    DivergentRetrieverScanner,
    HiddenLinkScanner,
    HippoIndexAnalyzer,
    load_config,
)


def create_mock_hippo_logs(log_dir: Path):
    """Create mock Hippo-Index log files for testing."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create mock log file with co-mentions
    log_file = log_dir / f"hippo_{datetime.now().strftime('%Y%m%d')}.log"

    mock_entries = [
        "2025-07-23T01:15:00Z [ENTITY:aspirin] mentioned in context with [ENTITY:headache]",
        "2025-07-23T01:16:00Z [COMENTION:aspirin|headache] found in medical query",
        "2025-07-23T01:17:00Z Patient reported [ENTITY:aspirin] allergy, also has [ENTITY:headache]",
        "2025-07-23T01:18:00Z [COMENTION:aspirin|headache] [COMENTION:headache|treatment]",
        "2025-07-23T01:19:00Z [ENTITY:ibuprofen] alternative to [ENTITY:aspirin] for [ENTITY:headache]",
        "2025-07-23T01:20:00Z [COMENTION:ibuprofen|headache] recommended by doctor",
        "2025-07-23T01:21:00Z Multiple mentions: [ENTITY:aspirin] [ENTITY:ibuprofen] [ENTITY:headache]",
    ]

    with open(log_file, "w") as f:
        for entry in mock_entries:
            f.write(entry + "\n")

    print(f"Created mock log file: {log_file}")


async def test_hippo_analyzer():
    """Test HippoIndexAnalyzer with mock data."""
    print("Testing HippoIndexAnalyzer...")

    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir) / "hippo_logs"
        create_mock_hippo_logs(log_dir)

        analyzer = HippoIndexAnalyzer(log_dir, lookback_hours=24)
        pairs = analyzer.analyze_logs(min_co_mentions=1)

        print(f"Found {len(pairs)} co-mention pairs:")
        for pair in pairs:
            print(
                f"  {pair.entity1} <-> {pair.entity2}: {pair.co_mention_count} mentions (confidence: {pair.confidence:.2f})"
            )

        assert len(pairs) >= 2, "Should find at least aspirin-headache and ibuprofen-headache pairs"
        print("+ HippoIndexAnalyzer test passed")


async def test_divergent_scanner():
    """Test DivergentRetrieverScanner with mock pairs."""
    print("Testing DivergentRetrieverScanner...")

    # Create mock co-mention pairs
    pairs = [
        CoMentionPair(entity1="aspirin", entity2="headache", co_mention_count=3, confidence=0.8),
        CoMentionPair(entity1="ibuprofen", entity2="headache", co_mention_count=2, confidence=0.6),
    ]

    scanner = DivergentRetrieverScanner(None)  # Mock retriever
    candidates = await scanner.scan_entity_pairs(pairs, n_candidates=2)

    print(f"Found {len(candidates)} candidate edges:")
    for candidate in candidates:
        print(
            f"  {candidate.source_entity} -{candidate.relationship_type}-> {candidate.target_entity} (confidence: {candidate.confidence:.2f})"
        )

    assert len(candidates) >= 2, "Should find candidates for each pair"
    print("+ DivergentRetrieverScanner test passed")


async def test_full_scanner():
    """Test complete HiddenLinkScanner pipeline."""
    print("Testing complete HiddenLinkScanner pipeline...")

    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir) / "hippo_logs"
        create_mock_hippo_logs(log_dir)

        config = {
            "hippo_log_path": str(log_dir),
            "lookback_hours": 24,
            "min_co_mentions": 1,
            "max_pairs_to_scan": 10,
            "candidates_per_pair": 2,
            "dry_run": True,
        }

        scanner = HiddenLinkScanner(config)
        metrics = await scanner.run_scan(dry_run=True)

        print("Scan metrics:")
        print(f"  Co-mention pairs found: {metrics.co_mention_pairs_found}")
        print(f"  Candidate edges discovered: {metrics.candidate_edges_discovered}")
        print(f"  Guardian evaluations: {metrics.guardian_evaluations}")
        print(f"  Guardian approved: {metrics.guardian_approved}")
        print(f"  Guardian quarantined: {metrics.guardian_quarantined}")
        print(f"  Guardian rejected: {metrics.guardian_rejected}")
        print(f"  Total time: {metrics.total_time_seconds:.2f}s")
        print(f"  Errors: {len(metrics.errors)}")

        assert metrics.co_mention_pairs_found > 0, "Should find co-mention pairs"
        assert metrics.candidate_edges_discovered > 0, "Should discover candidate edges"
        print("+ Complete scanner test passed")


async def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")

    # Test default config
    config = load_config()
    assert "hippo_log_path" in config
    assert "lookback_hours" in config
    print("+ Default config loaded")

    # Test with actual config file
    config_path = Path(__file__).parent.parent / "config" / "scanner_config.json"
    if config_path.exists():
        config = load_config(str(config_path))
        assert "jobs" in config
        assert "scanner" in config
        print("+ Scanner config file loaded")

    print("+ Configuration loading test passed")


async def main():
    """Run all tests."""
    print("Running HypeRAG Hidden-Link Scanner tests...\n")

    try:
        await test_config_loading()
        print()

        await test_hippo_analyzer()
        print()

        await test_divergent_scanner()
        print()

        await test_full_scanner()
        print()

        print("All tests passed!")

    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
