#!/usr/bin/env python3
"""GDC Scanner Demo.

Demonstrates how to use the HypeRAG Graph-Doctor constraint detection system.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.hyperag.gdc.registry import GDC_REGISTRY


async def demo_gdc_scanning() -> None:
    """Demonstrate GDC scanning functionality."""
    print("🔍 HypeRAG Graph-Doctor Demo")
    print("=" * 50)

    # Show loaded GDCs
    print(f"📋 Loaded {len(GDC_REGISTRY)} GDC rules:")
    for gdc_id, gdc in sorted(GDC_REGISTRY.items()):
        status = "✅" if gdc.enabled else "❌"
        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[gdc.severity]
        print(f"  {status} {severity_emoji} {gdc_id}: {gdc.description}")

    print("\n🔧 Example CLI Usage:")
    print("=" * 30)

    cli_examples = [
        "# List all available GDCs",
        "python scripts/hyperag_scan_gdc.py --list-gdcs",
        "",
        "# Scan all GDCs (requires Neo4j running)",
        "python scripts/hyperag_scan_gdc.py --neo4j-uri bolt://localhost:7687",
        "",
        "# Scan specific GDC with custom limit",
        "python scripts/hyperag_scan_gdc.py --gdc GDC_CONFIDENCE_VIOLATION --limit 100",
        "",
        "# Scan high-severity GDCs and save to CSV",
        "python scripts/hyperag_scan_gdc.py --severity high --format csv --out violations.csv",
        "",
        "# Scan with verbose output and custom Neo4j credentials",
        "python scripts/hyperag_scan_gdc.py --verbose --neo4j-user admin --neo4j-password secret",
    ]

    for example in cli_examples:
        if example.startswith("#"):
            print(f"\n{example}")
        else:
            print(f"  {example}")

    # Demonstrate programmatic usage (mock since we don't have Neo4j)
    print("\n🐍 Programmatic Usage:")
    print("=" * 25)

    try:
        # This would normally connect to Neo4j
        print("# Example: Connect to Neo4j and scan for violations")
        print("async with GDCExtractorContext('bolt://localhost:7687', ('neo4j', 'password')) as extractor:")
        print("    violations = await extractor.scan_all(limit=50)")
        print("    print(f'Found {len(violations)} violations')")
        print("")
        print("# Example: Scan specific GDC")
        print("violations = await extractor.scan_gdc('GDC_CONFIDENCE_VIOLATION')")
        print("")
        print("# Example: Health check")
        print("health = await extractor.health_check()")
        print("print(f'Neo4j status: {health[\"status\"]}')")

    except Exception as e:
        print(f"⚠️  Mock demo (Neo4j not running): {e}")

    # Show sample violation structure
    print("\n📊 Sample Violation Structure:")
    print("=" * 33)

    sample_violation = {
        "violation_id": "12345678-1234-5678-9abc-123456789abc",
        "gdc_id": "GDC_CONFIDENCE_VIOLATION",
        "nodes": [
            {
                "id": "node-123",
                "confidence": -0.5,  # Invalid confidence
                "_labels": ["SemanticNode"],
                "_neo4j_id": 12345,
            }
        ],
        "edges": [],
        "relationships": [],
        "severity": "high",
        "detected_at": "2025-07-22T18:30:00Z",
        "suggested_repair": "normalize_confidence",
        "confidence_score": 1.0,
        "metadata": {
            "gdc_description": "Node or edge has confidence value outside valid range [0,1]",
            "gdc_category": "data_quality",
        },
    }

    print(json.dumps(sample_violation, indent=2))

    print("\n✅ Demo complete!")
    print("\nNext steps:")
    print("1. Start Neo4j database")
    print("2. Run: python scripts/hyperag_scan_gdc.py --list-gdcs")
    print("3. Configure Neo4j connection in CLI arguments")
    print("4. Execute full scan to detect violations")


if __name__ == "__main__":
    asyncio.run(demo_gdc_scanning())
