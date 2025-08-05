#!/usr/bin/env python3
"""HypeRAG Graph-Doctor CLI Scanner

Command-line interface for detecting Graph Denial Constraint (GDC) violations
in Neo4j knowledge graphs.

Usage:
    python scripts/hyperag_scan_gdc.py --gdc ALL --limit 1000 --out violations.json
    python scripts/hyperag_scan_gdc.py --gdc GDC_CONFIDENCE_VIOLATION --limit 100
    python scripts/hyperag_scan_gdc.py --severity high --format csv
"""

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

from mcp_servers.hyperag.gdc.extractor import GDCExtractorContext
from mcp_servers.hyperag.gdc.registry import GDC_REGISTRY, validate_registry
from mcp_servers.hyperag.gdc.specs import Violation

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the CLI"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"gdc_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        ],
    )


def format_violation_summary(violations: list[Violation]) -> str:
    """Create a human-readable summary of violations"""
    if not violations:
        return "✅ No violations detected!"

    # Group by GDC ID and severity
    by_gdc = {}
    by_severity = {"high": 0, "medium": 0, "low": 0}

    for violation in violations:
        gdc_id = violation.gdc_id
        if gdc_id not in by_gdc:
            by_gdc[gdc_id] = []
        by_gdc[gdc_id].append(violation)
        by_severity[violation.severity] += 1

    summary = f"🚨 Detected {len(violations)} violations:\n\n"

    # Severity breakdown
    summary += "Severity Breakdown:\n"
    for severity, count in by_severity.items():
        if count > 0:
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[severity]
            summary += f"  {emoji} {severity.title()}: {count}\n"

    summary += "\nViolations by GDC:\n"
    for gdc_id, gdc_violations in sorted(by_gdc.items()):
        summary += f"  📋 {gdc_id}: {len(gdc_violations)} violations\n"

        # Show first violation details
        if gdc_violations:
            v = gdc_violations[0]
            summary += f"     └─ Example: {len(v.nodes)} nodes, {len(v.relationships)} relationships\n"

    return summary


def save_violations_json(violations: list[Violation], output_path: Path) -> None:
    """Save violations to JSON file"""
    data = {
        "scan_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_violations": len(violations),
        "violations": [v.to_dict() for v in violations],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"💾 Saved {len(violations)} violations to {output_path}")


def save_violations_csv(violations: list[Violation], output_path: Path) -> None:
    """Save violations to CSV file"""
    if not violations:
        print("No violations to save")
        return

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(
            [
                "violation_id",
                "gdc_id",
                "severity",
                "detected_at",
                "suggested_repair",
                "confidence_score",
                "node_count",
                "relationship_count",
                "description",
            ]
        )

        # Write violation data
        for v in violations:
            writer.writerow(
                [
                    v.violation_id,
                    v.gdc_id,
                    v.severity,
                    v.detected_at.isoformat(),
                    v.suggested_repair,
                    v.confidence_score,
                    len(v.nodes),
                    len(v.relationships),
                    v.metadata.get("gdc_description", ""),
                ]
            )

    print(f"💾 Saved {len(violations)} violations to {output_path}")


async def run_scan(args: argparse.Namespace) -> None:
    """Execute the GDC scan"""
    # Initialize extractor
    async with GDCExtractorContext(
        neo4j_uri=args.neo4j_uri,
        neo4j_auth=(args.neo4j_user, args.neo4j_password),
        max_concurrent_queries=args.max_concurrent,
        default_limit=args.limit,
    ) as extractor:
        # Health check
        health = await extractor.health_check()
        if health["status"] != "healthy":
            print(f"❌ Neo4j health check failed: {health}")
            sys.exit(1)

        print(f"✅ Connected to Neo4j: {args.neo4j_uri}")

        # Get graph statistics
        if args.verbose:
            stats = await extractor.get_graph_stats()
            if stats and "total_nodes" in stats:
                print(f"📊 Graph stats: {stats['total_nodes']} nodes, {stats['total_relationships']} relationships")

        # Execute scan
        print("🔍 Starting GDC scan...")
        start_time = datetime.now()

        if args.gdc and args.gdc != "ALL":
            # Scan specific GDC
            violations = await extractor.scan_gdc(args.gdc, args.limit)
        else:
            # Scan all GDCs
            violations = await extractor.scan_all(
                limit=args.limit,
                enabled_only=not args.include_disabled,
                severity_filter=args.severity,
            )

        scan_duration = (datetime.now() - start_time).total_seconds()
        print(f"⏱️  Scan completed in {scan_duration:.2f} seconds")

        # Display summary
        print("\n" + format_violation_summary(violations))

        # Save results
        if args.output:
            output_path = Path(args.output)

            if args.format == "json":
                save_violations_json(violations, output_path)
            elif args.format == "csv":
                save_violations_csv(violations, output_path)
            else:
                print(f"❌ Unknown output format: {args.format}")
                sys.exit(1)

        # Exit with error code if violations found
        if violations and args.fail_on_violations:
            sys.exit(1)


def validate_gdc_config() -> None:
    """Validate the GDC configuration"""
    if not GDC_REGISTRY:
        print("❌ No GDC rules loaded. Check config/gdc_rules.yaml")
        sys.exit(1)

    issues = validate_registry(GDC_REGISTRY)
    if issues:
        print("⚠️  GDC configuration issues:")
        for issue in issues:
            print(f"   - {issue}")
        print()

    print(f"📋 Loaded {len(GDC_REGISTRY)} GDC rules")

    if any(not gdc.enabled for gdc in GDC_REGISTRY.values()):
        disabled_count = sum(1 for gdc in GDC_REGISTRY.values() if not gdc.enabled)
        print(f"⏸️  {disabled_count} GDCs are disabled")


def list_gdcs() -> None:
    """List all available GDCs"""
    if not GDC_REGISTRY:
        print("No GDCs configured")
        return

    print("Available GDCs:")
    print("=" * 80)

    for gdc_id, gdc in sorted(GDC_REGISTRY.items()):
        status = "✅" if gdc.enabled else "❌"
        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[gdc.severity]

        print(f"{status} {gdc_id}")
        print(f"   {severity_emoji} Severity: {gdc.severity}")
        print(f"   📝 {gdc.description}")
        print(f"   🔧 Repair: {gdc.suggested_action}")
        print(f"   🏷️  Category: {gdc.category}")
        print()


def main() -> None:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="HypeRAG Graph-Doctor GDC Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all GDCs with default settings
  python scripts/hyperag_scan_gdc.py

  # Scan specific GDC with custom limit
  python scripts/hyperag_scan_gdc.py --gdc GDC_CONFIDENCE_VIOLATION --limit 500

  # Scan only high-severity GDCs and save to CSV
  python scripts/hyperag_scan_gdc.py --severity high --format csv --out violations.csv

  # List all available GDCs
  python scripts/hyperag_scan_gdc.py --list-gdcs
        """,
    )

    # GDC selection
    parser.add_argument("--gdc", default="ALL", help="GDC ID to scan (default: ALL)")
    parser.add_argument(
        "--severity",
        choices=["low", "medium", "high"],
        help="Only scan GDCs with specified severity",
    )
    parser.add_argument("--include-disabled", action="store_true", help="Include disabled GDCs in scan")

    # Scan parameters
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum violations per GDC (default: 1000)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent queries (default: 5)",
    )

    # Neo4j connection
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)",
    )
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username (default: neo4j)")
    parser.add_argument(
        "--neo4j-password",
        default="password",
        help="Neo4j password (default: password)",
    )

    # Output options
    parser.add_argument("--output", "--out", help="Output file path")
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )

    # Control options
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit with error code if violations found",
    )
    parser.add_argument("--list-gdcs", action="store_true", help="List all available GDCs and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    # Handle list command
    if args.list_gdcs:
        list_gdcs()
        return

    # Validate configuration
    validate_gdc_config()

    # Run scan
    try:
        asyncio.run(run_scan(args))
    except KeyboardInterrupt:
        print("\n🛑 Scan interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Scan failed: {e}")
        logging.exception("Scan failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
