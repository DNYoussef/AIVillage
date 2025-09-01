"""
Command Line Interface for PII/PHI Compliance Management.

Provides comprehensive CLI commands for data privacy compliance including
PII/PHI discovery, retention management, and compliance reporting.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys

try:
    import click
    import tabulate

    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    import argparse

from .pii_phi_manager import DataClassification, PIIPHIManager, RetentionPolicy


class ComplianceCLI:
    """PII/PHI compliance command line interface."""

    def __init__(self):
        """Initialize CLI with manager."""
        self.pii_manager = None

    async def _initialize_manager(self):
        """Initialize PII/PHI manager."""
        if not self.pii_manager:
            self.pii_manager = PIIPHIManager()

    async def discover_pii_phi(
        self, scan_type: str = "all", path: str = None, tenant_id: str = None, output_format: str = "table"
    ):
        """Discover PII/PHI in data sources."""
        await self._initialize_manager()

        try:
            if scan_type == "database" and path:
                discoveries = await self.pii_manager.discover_pii_phi_in_database(path, tenant_id)
            elif scan_type == "files" and path:
                discoveries = await self.pii_manager.discover_pii_phi_in_files(Path(path), tenant_id)
            elif scan_type == "all":
                discovery_results = await self.pii_manager.scan_all_known_locations()
                discoveries = []
                for locations in discovery_results.values():
                    discoveries.extend(locations)
            else:
                self._error("Invalid scan type or missing path")
                return

            if output_format == "json":
                discovery_data = []
                for location in discoveries:
                    discovery_data.append(
                        {
                            "location_id": location.location_id,
                            "source_type": location.source_type,
                            "path": location.path,
                            "classification": location.classification.value,
                            "confidence": location.confidence_score,
                            "tenant_id": location.tenant_id,
                            "estimated_records": location.estimated_records,
                        }
                    )
                self._output_json({"discoveries": discovery_data})

            elif output_format == "table":
                if not discoveries:
                    self._info("No PII/PHI discoveries found")
                    return

                headers = ["Location ID", "Type", "Path", "Classification", "Confidence", "Records", "Tenant"]
                rows = []

                for location in discoveries:
                    rows.append(
                        [
                            (
                                location.location_id[:20] + "..."
                                if len(location.location_id) > 20
                                else location.location_id
                            ),
                            location.source_type,
                            (location.path[:30] + "...") if len(location.path) > 30 else location.path,
                            location.classification.value,
                            f"{location.confidence_score:.2f}",
                            str(location.estimated_records),
                            location.tenant_id or "N/A",
                        ]
                    )

                table = tabulate.tabulate(rows, headers=headers, tablefmt="grid")
                print(table)

                # Summary
                by_classification = {}
                for location in discoveries:
                    cls = location.classification.value
                    by_classification[cls] = by_classification.get(cls, 0) + 1

                print(f"\nSummary: {len(discoveries)} locations discovered")
                for cls, count in by_classification.items():
                    print(f"  - {cls}: {count}")

        except Exception as e:
            self._error(f"Discovery failed: {e}")

    async def list_locations(
        self, classification: str = None, tenant_id: str = None, limit: int = 50, output_format: str = "table"
    ):
        """List discovered PII/PHI locations."""
        await self._initialize_manager()

        try:
            locations = list(self.pii_manager.data_locations.values())

            # Apply filters
            if classification:
                classification_enum = DataClassification(classification)
                locations = [loc for loc in locations if loc.classification == classification_enum]

            if tenant_id:
                locations = [loc for loc in locations if loc.tenant_id == tenant_id]

            # Limit results
            locations = locations[:limit]

            if output_format == "json":
                location_data = []
                for location in locations:
                    location_data.append(
                        {
                            "location_id": location.location_id,
                            "source_type": location.source_type,
                            "path": location.path,
                            "classification": location.classification.value,
                            "retention_policy": location.retention_policy.value,
                            "tenant_id": location.tenant_id,
                            "compliant": location.compliant,
                            "last_audit": location.last_audit.isoformat() if location.last_audit else None,
                            "estimated_records": location.estimated_records,
                            "confidence": location.confidence_score,
                        }
                    )
                self._output_json({"locations": location_data})

            elif output_format == "table":
                if not locations:
                    self._info("No locations found")
                    return

                headers = ["Location ID", "Type", "Path", "Classification", "Policy", "Compliant", "Records"]
                rows = []

                for location in locations:
                    compliance_status = "✓" if location.compliant else "✗"
                    rows.append(
                        [
                            (
                                location.location_id[:15] + "..."
                                if len(location.location_id) > 15
                                else location.location_id
                            ),
                            location.source_type,
                            (location.path[:25] + "...") if len(location.path) > 25 else location.path,
                            location.classification.value,
                            location.retention_policy.value,
                            compliance_status,
                            str(location.estimated_records),
                        ]
                    )

                table = tabulate.tabulate(rows, headers=headers, tablefmt="grid")
                print(table)

        except Exception as e:
            self._error(f"Failed to list locations: {e}")

    async def location_info(self, location_id: str, output_format: str = "text"):
        """Show detailed information about a PII/PHI location."""
        await self._initialize_manager()

        try:
            if location_id not in self.pii_manager.data_locations:
                self._error(f"Location {location_id} not found")
                return

            location = self.pii_manager.data_locations[location_id]

            if output_format == "json":
                data = {
                    "location_id": location.location_id,
                    "source_type": location.source_type,
                    "path": location.path,
                    "table_name": location.table_name,
                    "column_name": location.column_name,
                    "field_name": location.field_name,
                    "tenant_id": location.tenant_id,
                    "classification": location.classification.value,
                    "retention_policy": location.retention_policy.value,
                    "regulations": [r.value for r in location.regulations],
                    "discovered_at": location.discovered_at.isoformat(),
                    "last_verified": location.last_verified.isoformat() if location.last_verified else None,
                    "sample_count": location.sample_count,
                    "confidence_score": location.confidence_score,
                    "compliant": location.compliant,
                    "last_audit": location.last_audit.isoformat() if location.last_audit else None,
                    "violations": location.violations,
                    "estimated_records": location.estimated_records,
                    "data_size_bytes": location.data_size_bytes,
                    "access_frequency": location.access_frequency,
                    "last_accessed": location.last_accessed.isoformat() if location.last_accessed else None,
                }
                self._output_json(data)

            else:
                self._info(f"Location: {location.location_id}")
                self._info(f"Source: {location.source_type}")
                self._info(f"Path: {location.path}")

                if location.table_name:
                    self._info(f"Table: {location.table_name}")
                if location.column_name:
                    self._info(f"Column: {location.column_name}")
                if location.field_name:
                    self._info(f"Field: {location.field_name}")
                if location.tenant_id:
                    self._info(f"Tenant: {location.tenant_id}")

                self._info(f"Classification: {location.classification.value}")
                self._info(f"Retention Policy: {location.retention_policy.value}")

                if location.regulations:
                    regulations = ", ".join([r.value for r in location.regulations])
                    self._info(f"Regulations: {regulations}")

                self._info(f"Discovered: {location.discovered_at}")
                if location.last_verified:
                    self._info(f"Last Verified: {location.last_verified}")

                self._info(f"Confidence: {location.confidence_score:.2f}")
                self._info(f"Sample Count: {location.sample_count}")
                self._info(f"Estimated Records: {location.estimated_records}")

                if location.data_size_bytes > 0:
                    size_mb = location.data_size_bytes / (1024 * 1024)
                    self._info(f"Data Size: {size_mb:.2f} MB")

                compliance_status = "Compliant" if location.compliant else "Non-compliant"
                self._info(f"Compliance: {compliance_status}")

                if location.last_audit:
                    self._info(f"Last Audit: {location.last_audit}")

                if location.violations:
                    self._warning("Violations:")
                    for violation in location.violations:
                        self._warning(f"  - {violation}")

        except Exception as e:
            self._error(f"Failed to get location info: {e}")

    async def create_retention_job(
        self,
        name: str,
        description: str,
        location_ids: list,
        retention_policy: str,
        custom_days: int = None,
        schedule: str = "0 2 * * 0",
    ):
        """Create a new data retention job."""
        await self._initialize_manager()

        try:
            policy_enum = RetentionPolicy(retention_policy)

            job_id = await self.pii_manager.create_retention_job(
                name=name,
                description=description,
                location_ids=location_ids,
                retention_policy=policy_enum,
                custom_retention_days=custom_days,
                schedule_cron=schedule,
            )

            self._success(f"Retention job created: {job_id}")
            self._info(f"Name: {name}")
            self._info(f"Policy: {retention_policy}")
            if custom_days:
                self._info(f"Custom retention: {custom_days} days")
            self._info(f"Locations: {len(location_ids)}")
            self._info(f"Schedule: {schedule}")

        except Exception as e:
            self._error(f"Failed to create retention job: {e}")

    async def list_retention_jobs(self, output_format: str = "table"):
        """List retention jobs."""
        await self._initialize_manager()

        try:
            jobs = list(self.pii_manager.retention_jobs.values())

            if output_format == "json":
                job_data = []
                for job in jobs:
                    job_data.append(
                        {
                            "job_id": job.job_id,
                            "name": job.name,
                            "description": job.description,
                            "retention_policy": job.retention_policy.value,
                            "enabled": job.enabled,
                            "location_count": len(job.location_ids),
                            "last_run": job.last_run.isoformat() if job.last_run else None,
                            "next_run": job.next_run.isoformat() if job.next_run else None,
                            "success_rate": job.success_count / max(job.run_count, 1) * 100 if job.run_count > 0 else 0,
                            "approval_status": job.approval_status,
                        }
                    )
                self._output_json({"retention_jobs": job_data})

            elif output_format == "table":
                if not jobs:
                    self._info("No retention jobs found")
                    return

                headers = ["Job ID", "Name", "Policy", "Enabled", "Locations", "Last Run", "Success Rate", "Status"]
                rows = []

                for job in jobs:
                    success_rate = job.success_count / max(job.run_count, 1) * 100 if job.run_count > 0 else 0
                    last_run = job.last_run.strftime("%Y-%m-%d") if job.last_run else "Never"

                    rows.append(
                        [
                            job.job_id[:15] + "..." if len(job.job_id) > 15 else job.job_id,
                            job.name[:20] + "..." if len(job.name) > 20 else job.name,
                            job.retention_policy.value,
                            "Yes" if job.enabled else "No",
                            str(len(job.location_ids)),
                            last_run,
                            f"{success_rate:.1f}%",
                            job.approval_status,
                        ]
                    )

                table = tabulate.tabulate(rows, headers=headers, tablefmt="grid")
                print(table)

        except Exception as e:
            self._error(f"Failed to list retention jobs: {e}")

    async def execute_retention_job(self, job_id: str):
        """Execute a retention job."""
        await self._initialize_manager()

        try:
            self._info(f"Executing retention job: {job_id}")
            results = await self.pii_manager.execute_retention_job(job_id)

            self._success("Retention job completed")
            self._info(f"Locations processed: {results['locations_processed']}")
            self._info(f"Records deleted: {results['records_deleted']}")

            if results["errors"]:
                self._warning("Errors occurred:")
                for error in results["errors"]:
                    self._warning(f"  - {error}")

        except Exception as e:
            self._error(f"Failed to execute retention job: {e}")

    async def compliance_summary(self, output_format: str = "text"):
        """Show compliance summary."""
        await self._initialize_manager()

        try:
            summary = await self.pii_manager.get_compliance_summary()

            if output_format == "json":
                self._output_json(summary)

            else:
                self._info("Compliance Summary")
                self._info("=" * 50)

                self._info(f"Total locations: {summary['total_locations']}")

                print("\nBy Classification:")
                for classification, count in summary["by_classification"].items():
                    print(f"  - {classification}: {count}")

                print("\nBy Regulation:")
                for regulation, count in summary["by_regulation"].items():
                    print(f"  - {regulation}: {count}")

                print("\nCompliance Status:")
                cs = summary["compliance_status"]
                print(f"  - Compliant: {cs['compliant']}")
                print(f"  - Violations: {cs['violations']}")
                print(f"  - Unaudited: {cs['unaudited']}")

                print("\nRetention Jobs:")
                rj = summary["retention_jobs"]
                print(f"  - Total: {rj['total']}")
                print(f"  - Enabled: {rj['enabled']}")
                print(f"  - Pending approval: {rj['pending_approval']}")

                # Calculate compliance percentage
                total_audited = cs["compliant"] + cs["violations"]
                if total_audited > 0:
                    compliance_rate = (cs["compliant"] / total_audited) * 100
                    print(f"\nCompliance Rate: {compliance_rate:.1f}%")

        except Exception as e:
            self._error(f"Failed to get compliance summary: {e}")

    async def approve_retention_job(self, job_id: str, approver: str = "system"):
        """Approve a retention job."""
        await self._initialize_manager()

        try:
            if job_id not in self.pii_manager.retention_jobs:
                self._error(f"Retention job {job_id} not found")
                return

            job = self.pii_manager.retention_jobs[job_id]
            job.approval_status = "approved"
            job.approver = approver
            job.approved_at = datetime.utcnow()

            await self.pii_manager._save_retention_job(job)

            self._success(f"Retention job {job_id} approved by {approver}")

        except Exception as e:
            self._error(f"Failed to approve retention job: {e}")

    async def generate_compliance_report(self, output_file: str = None, format: str = "json"):
        """Generate comprehensive compliance report."""
        await self._initialize_manager()

        try:
            report = {
                "generated_at": datetime.utcnow().isoformat(),
                "summary": await self.pii_manager.get_compliance_summary(),
                "locations": [],
                "retention_jobs": [],
                "audit_recommendations": [],
            }

            # Add location details
            for location in self.pii_manager.data_locations.values():
                report["locations"].append(
                    {
                        "location_id": location.location_id,
                        "source_type": location.source_type,
                        "path": location.path,
                        "classification": location.classification.value,
                        "retention_policy": location.retention_policy.value,
                        "regulations": [r.value for r in location.regulations],
                        "compliant": location.compliant,
                        "confidence_score": location.confidence_score,
                        "estimated_records": location.estimated_records,
                        "tenant_id": location.tenant_id,
                    }
                )

            # Add retention job details
            for job in self.pii_manager.retention_jobs.values():
                report["retention_jobs"].append(
                    {
                        "job_id": job.job_id,
                        "name": job.name,
                        "retention_policy": job.retention_policy.value,
                        "enabled": job.enabled,
                        "location_count": len(job.location_ids),
                        "approval_status": job.approval_status,
                        "run_count": job.run_count,
                        "success_count": job.success_count,
                        "total_records_deleted": job.total_records_deleted,
                    }
                )

            # Generate audit recommendations
            recommendations = []

            # Check for unaudited locations
            unaudited = [loc for loc in self.pii_manager.data_locations.values() if not loc.last_audit]
            if unaudited:
                recommendations.append(
                    {
                        "type": "audit_required",
                        "priority": "high",
                        "message": f"{len(unaudited)} locations require audit",
                        "affected_locations": [loc.location_id for loc in unaudited[:5]],
                    }
                )

            # Check for non-compliant locations
            violations = [loc for loc in self.pii_manager.data_locations.values() if not loc.compliant]
            if violations:
                recommendations.append(
                    {
                        "type": "compliance_violation",
                        "priority": "critical",
                        "message": f"{len(violations)} locations have compliance violations",
                        "affected_locations": [loc.location_id for loc in violations[:5]],
                    }
                )

            # Check for pending retention jobs
            pending_jobs = [job for job in self.pii_manager.retention_jobs.values() if job.approval_status == "pending"]
            if pending_jobs:
                recommendations.append(
                    {
                        "type": "approval_required",
                        "priority": "medium",
                        "message": f"{len(pending_jobs)} retention jobs need approval",
                        "affected_jobs": [job.job_id for job in pending_jobs],
                    }
                )

            report["audit_recommendations"] = recommendations

            if output_file:
                with open(output_file, "w") as f:
                    if format == "json":
                        json.dump(report, f, indent=2, default=str)
                    else:
                        # Generate human-readable report
                        f.write("AIVillage Privacy Compliance Report\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"Generated: {report['generated_at']}\n\n")

                        summary = report["summary"]
                        f.write(f"Total locations: {summary['total_locations']}\n")
                        f.write(
                            f"Compliance rate: {summary['compliance_status']['compliant']} / {summary['total_locations']}\n\n"
                        )

                        f.write("Classifications:\n")
                        for cls, count in summary["by_classification"].items():
                            f.write(f"  - {cls}: {count}\n")

                        f.write("\nRecommendations:\n")
                        for rec in recommendations:
                            f.write(f"  - {rec['priority'].upper()}: {rec['message']}\n")

                self._success(f"Compliance report generated: {output_file}")
            else:
                self._output_json(report)

        except Exception as e:
            self._error(f"Failed to generate compliance report: {e}")

    def _success(self, message: str):
        """Print success message."""
        print(f"✓ {message}")

    def _info(self, message: str):
        """Print info message."""
        print(f"ℹ {message}")

    def _warning(self, message: str):
        """Print warning message."""
        print(f"⚠ {message}")

    def _error(self, message: str):
        """Print error message."""
        print(f"✗ {message}", file=sys.stderr)

    def _output_json(self, data):
        """Output data as JSON."""
        print(json.dumps(data, indent=2, default=str))


# CLI command definitions
if CLICK_AVAILABLE:
    # Click-based CLI
    @click.group()
    def compliance_cli():
        """AIVillage PII/PHI Compliance Management"""
        pass

    @compliance_cli.command("discover")
    @click.option(
        "--type",
        "scan_type",
        default="all",
        type=click.Choice(["all", "database", "files"]),
        help="Type of scan to perform",
    )
    @click.option("--path", help="Path to scan (for database or files)")
    @click.option("--tenant", help="Tenant ID to associate with discoveries")
    @click.option(
        "--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format"
    )
    def discover(scan_type, path, tenant, output_format):
        """Discover PII/PHI in data sources"""
        cli = ComplianceCLI()
        asyncio.run(cli.discover_pii_phi(scan_type, path, tenant, output_format))

    @compliance_cli.command("list")
    @click.option("--classification", help="Filter by classification")
    @click.option("--tenant", help="Filter by tenant ID")
    @click.option("--limit", default=50, help="Maximum number to show")
    @click.option(
        "--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format"
    )
    def list_locations(classification, tenant, limit, output_format):
        """List discovered PII/PHI locations"""
        cli = ComplianceCLI()
        asyncio.run(cli.list_locations(classification, tenant, limit, output_format))

    @compliance_cli.command("info")
    @click.argument("location_id")
    @click.option(
        "--format", "output_format", default="text", type=click.Choice(["text", "json"]), help="Output format"
    )
    def location_info(location_id, output_format):
        """Show detailed location information"""
        cli = ComplianceCLI()
        asyncio.run(cli.location_info(location_id, output_format))

    @compliance_cli.command("create-job")
    @click.option("--name", required=True, help="Job name")
    @click.option("--description", required=True, help="Job description")
    @click.option("--locations", required=True, help="Comma-separated location IDs")
    @click.option(
        "--policy",
        "retention_policy",
        required=True,
        type=click.Choice(["immediate", "short_term", "standard", "long_term", "healthcare", "financial", "custom"]),
        help="Retention policy",
    )
    @click.option("--custom-days", type=int, help="Custom retention days (for custom policy)")
    @click.option("--schedule", default="0 2 * * 0", help="Cron schedule")
    def create_retention_job(name, description, locations, retention_policy, custom_days, schedule):
        """Create data retention job"""
        cli = ComplianceCLI()
        location_list = [loc.strip() for loc in locations.split(",")]
        asyncio.run(cli.create_retention_job(name, description, location_list, retention_policy, custom_days, schedule))

    @compliance_cli.command("list-jobs")
    @click.option(
        "--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format"
    )
    def list_jobs(output_format):
        """List retention jobs"""
        cli = ComplianceCLI()
        asyncio.run(cli.list_retention_jobs(output_format))

    @compliance_cli.command("execute-job")
    @click.argument("job_id")
    def execute_job(job_id):
        """Execute retention job"""
        cli = ComplianceCLI()
        asyncio.run(cli.execute_retention_job(job_id))

    @compliance_cli.command("approve-job")
    @click.argument("job_id")
    @click.option("--approver", default="system", help="Approver name")
    def approve_job(job_id, approver):
        """Approve retention job"""
        cli = ComplianceCLI()
        asyncio.run(cli.approve_retention_job(job_id, approver))

    @compliance_cli.command("summary")
    @click.option(
        "--format", "output_format", default="text", type=click.Choice(["text", "json"]), help="Output format"
    )
    def summary(output_format):
        """Show compliance summary"""
        cli = ComplianceCLI()
        asyncio.run(cli.compliance_summary(output_format))

    @compliance_cli.command("report")
    @click.option("--output", help="Output file path")
    @click.option("--format", default="json", type=click.Choice(["json", "text"]), help="Report format")
    def generate_report(output, format):
        """Generate compliance report"""
        cli = ComplianceCLI()
        asyncio.run(cli.generate_compliance_report(output, format))

    if __name__ == "__main__":
        compliance_cli()

else:
    # Fallback argparse-based CLI
    def main():
        parser = argparse.ArgumentParser(description="AIVillage PII/PHI Compliance Management")
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Discovery command
        discover_parser = subparsers.add_parser("discover", help="Discover PII/PHI")
        discover_parser.add_argument("--type", choices=["all", "database", "files"], default="all")
        discover_parser.add_argument("--path", help="Path to scan")
        discover_parser.add_argument("--tenant", help="Tenant ID")

        # List command
        list_parser = subparsers.add_parser("list", help="List locations")
        list_parser.add_argument("--limit", type=int, default=50)

        # Summary command
        subparsers.add_parser("summary", help="Show compliance summary")

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        cli = ComplianceCLI()

        if args.command == "discover":
            asyncio.run(cli.discover_pii_phi(args.type, args.path, args.tenant))
        elif args.command == "list":
            asyncio.run(cli.list_locations(limit=args.limit))
        elif args.command == "summary":
            asyncio.run(cli.compliance_summary())
        else:
            print(f"Unknown command: {args.command}")

    if __name__ == "__main__":
        main()
