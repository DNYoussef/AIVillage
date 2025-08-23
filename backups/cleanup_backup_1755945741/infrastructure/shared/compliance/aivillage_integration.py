"""
AIVillage Integration for PII/PHI Compliance Management.

This module provides deep integration between the PII/PHI compliance system
and existing AIVillage infrastructure including RBAC, backup systems,
agent coordination, and the unified MCP governance dashboard.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .compliance_cli import ComplianceCLI
from .pii_phi_manager import ComplianceRegulation, DataClassification, PIIPHIManager, RetentionPolicy

# Import existing AIVillage infrastructure
try:
    from ...agents.governance.mcp_governance_dashboard import MCPGovernanceDashboard
    from ..backup.backup_manager import BackupManager
    from ..backup.restore_manager import RestoreManager
    from ..security.rbac_manager import RBACManager

    AIVILLAGE_INTEGRATION_AVAILABLE = True
except ImportError:
    logging.warning("AIVillage infrastructure not available for PII/PHI integration")
    AIVILLAGE_INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class AIVillageComplianceIntegration:
    """
    Comprehensive integration between PII/PHI compliance and AIVillage infrastructure.

    This class provides seamless integration with:
    - RBAC system for permission management
    - Backup system for compliance data protection
    - Agent governance for democratic data decisions
    - MCP dashboard for unified compliance management
    """

    def __init__(self, config_path: Path | None = None):
        """Initialize AIVillage compliance integration."""
        self.pii_manager = PIIPHIManager(config_path)
        self.compliance_cli = ComplianceCLI()

        # Initialize AIVillage infrastructure if available
        if AIVILLAGE_INTEGRATION_AVAILABLE:
            self.rbac_manager = RBACManager()
            self.backup_manager = BackupManager()
            self.restore_manager = RestoreManager(self.backup_manager)
            self.governance_dashboard = MCPGovernanceDashboard()
        else:
            self.rbac_manager = None
            self.backup_manager = None
            self.restore_manager = None
            self.governance_dashboard = None

        logger.info("AIVillage compliance integration initialized")

    async def initialize_compliance_infrastructure(self):
        """Initialize complete compliance infrastructure within AIVillage."""
        logger.info("Setting up AIVillage compliance infrastructure...")

        # 1. Set up RBAC permissions for compliance management
        await self._setup_rbac_permissions()

        # 2. Configure backup integration for compliance data
        await self._setup_backup_integration()

        # 3. Register compliance workflows with MCP governance
        await self._setup_governance_integration()

        # 4. Set up automated compliance monitoring
        await self._setup_compliance_monitoring()

        logger.info("AIVillage compliance infrastructure setup complete")

    async def _setup_rbac_permissions(self):
        """Set up RBAC permissions for compliance management."""
        if not self.rbac_manager:
            logger.warning("RBAC manager not available")
            return

        # Define compliance roles
        compliance_roles = [
            {
                "role_name": "compliance_admin",
                "description": "Full compliance management access",
                "permissions": [
                    "compliance:discover:all",
                    "compliance:manage_locations:all",
                    "compliance:create_retention_jobs:all",
                    "compliance:execute_retention_jobs:all",
                    "compliance:approve_retention_jobs:all",
                    "compliance:generate_reports:all",
                    "compliance:audit_trail:read",
                ],
            },
            {
                "role_name": "compliance_analyst",
                "description": "Compliance monitoring and reporting",
                "permissions": [
                    "compliance:discover:tenant",
                    "compliance:manage_locations:tenant",
                    "compliance:create_retention_jobs:tenant",
                    "compliance:generate_reports:tenant",
                    "compliance:audit_trail:read",
                ],
            },
            {
                "role_name": "compliance_viewer",
                "description": "Read-only compliance access",
                "permissions": ["compliance:locations:read", "compliance:reports:read", "compliance:audit_trail:read"],
            },
            {
                "role_name": "data_protection_officer",
                "description": "DPO role with special oversight capabilities",
                "permissions": [
                    "compliance:discover:all",
                    "compliance:manage_locations:all",
                    "compliance:oversight:all",
                    "compliance:breach_response:all",
                    "compliance:generate_reports:all",
                    "compliance:audit_trail:read",
                    "compliance:emergency_actions:all",
                ],
            },
        ]

        # Create roles
        for role_config in compliance_roles:
            try:
                role_id = await self.rbac_manager.create_role(
                    name=role_config["role_name"],
                    description=role_config["description"],
                    permissions=role_config["permissions"],
                )
                logger.info(f"Created compliance role: {role_config['role_name']} ({role_id})")
            except Exception as e:
                logger.warning(f"Failed to create role {role_config['role_name']}: {e}")

    async def _setup_backup_integration(self):
        """Integrate PII/PHI compliance with backup system."""
        if not self.backup_manager:
            logger.warning("Backup manager not available")
            return

        # Create specialized backup jobs for compliance data
        compliance_backup_jobs = [
            {
                "name": "PII/PHI Location Registry Backup",
                "description": "Daily backup of PII/PHI location database",
                "paths": ["data/compliance/pii_phi.db"],
                "schedule": "0 1 * * *",  # Daily at 1 AM
                "retention_days": 2555,  # 7 years for compliance
                "encryption": True,
                "tenant_id": None,  # System-wide
            },
            {
                "name": "Compliance Audit Trail Backup",
                "description": "Hourly backup of compliance audit logs",
                "paths": ["data/compliance/audit_log.db", "logs/compliance/"],
                "schedule": "0 * * * *",  # Hourly
                "retention_days": 2555,  # 7 years for compliance
                "encryption": True,
                "tenant_id": None,  # System-wide
            },
            {
                "name": "Tenant Compliance Data Backup",
                "description": "Daily backup of tenant-specific compliance data",
                "paths": ["data/tenants/*/compliance/"],
                "schedule": "0 2 * * *",  # Daily at 2 AM
                "retention_days": 2555,  # 7 years for compliance
                "encryption": True,
                "tenant_specific": True,
            },
        ]

        # Register backup jobs
        for job_config in compliance_backup_jobs:
            try:
                backup_id = await self.backup_manager.create_scheduled_backup(
                    name=job_config["name"],
                    description=job_config["description"],
                    paths=job_config["paths"],
                    schedule=job_config["schedule"],
                    retention_days=job_config["retention_days"],
                    encryption=job_config["encryption"],
                )
                logger.info(f"Created compliance backup job: {backup_id}")
            except Exception as e:
                logger.warning(f"Failed to create backup job {job_config['name']}: {e}")

    async def _setup_governance_integration(self):
        """Integrate compliance workflows with MCP governance dashboard."""
        if not self.governance_dashboard:
            logger.warning("Governance dashboard not available")
            return

        # Register compliance MCP tools with governance dashboard
        compliance_tools = [
            {
                "tool_id": "compliance_discover",
                "name": "PII/PHI Discovery",
                "description": "Discover PII/PHI in AIVillage data sources",
                "category": "compliance",
                "requires_approval": True,
                "approval_roles": ["compliance_admin", "data_protection_officer"],
                "handler": self._handle_discovery_request,
            },
            {
                "tool_id": "retention_job_create",
                "name": "Create Data Retention Job",
                "description": "Create automated data retention job",
                "category": "compliance",
                "requires_approval": True,
                "approval_roles": ["compliance_admin", "data_protection_officer"],
                "handler": self._handle_retention_job_creation,
            },
            {
                "tool_id": "retention_job_execute",
                "name": "Execute Data Retention Job",
                "description": "Execute approved data retention job",
                "category": "compliance",
                "requires_approval": True,
                "approval_roles": ["compliance_admin", "data_protection_officer"],
                "handler": self._handle_retention_job_execution,
            },
            {
                "tool_id": "compliance_report",
                "name": "Generate Compliance Report",
                "description": "Generate comprehensive compliance report",
                "category": "compliance",
                "requires_approval": False,
                "approval_roles": [],
                "handler": self._handle_compliance_report,
            },
        ]

        # Register tools with governance dashboard
        for tool_config in compliance_tools:
            try:
                await self.governance_dashboard.register_mcp_tool(
                    tool_id=tool_config["tool_id"],
                    name=tool_config["name"],
                    description=tool_config["description"],
                    category=tool_config["category"],
                    requires_approval=tool_config["requires_approval"],
                    approval_roles=tool_config["approval_roles"],
                    handler=tool_config["handler"],
                )
                logger.info(f"Registered compliance tool: {tool_config['tool_id']}")
            except Exception as e:
                logger.warning(f"Failed to register tool {tool_config['tool_id']}: {e}")

    async def _setup_compliance_monitoring(self):
        """Set up automated compliance monitoring and alerting."""
        # Set up periodic compliance checks
        monitoring_jobs = [
            {
                "name": "Daily Compliance Scan",
                "description": "Daily scan for new PII/PHI in system",
                "handler": self._daily_compliance_scan,
                "schedule": "0 3 * * *",  # Daily at 3 AM
            },
            {
                "name": "Weekly Compliance Audit",
                "description": "Weekly compliance audit and violation check",
                "handler": self._weekly_compliance_audit,
                "schedule": "0 4 * * 0",  # Weekly on Sunday at 4 AM
            },
            {
                "name": "Monthly Compliance Report",
                "description": "Monthly comprehensive compliance report",
                "handler": self._monthly_compliance_report,
                "schedule": "0 5 1 * *",  # Monthly on 1st at 5 AM
            },
        ]

        # Schedule monitoring jobs (would use proper scheduler in production)
        for job in monitoring_jobs:
            logger.info(f"Scheduled compliance monitoring: {job['name']}")

    # MCP Tool Handlers

    async def _handle_discovery_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle PII/PHI discovery request via MCP governance."""
        scan_type = request.get("scan_type", "all")
        path = request.get("path")
        tenant_id = request.get("tenant_id")

        try:
            if scan_type == "all":
                results = await self.pii_manager.scan_all_known_locations()
                discoveries = []
                for category_results in results.values():
                    discoveries.extend(category_results)
            elif scan_type == "database" and path:
                discoveries = await self.pii_manager.discover_pii_phi_in_database(path, tenant_id)
            elif scan_type == "files" and path:
                discoveries = await self.pii_manager.discover_pii_phi_in_files(Path(path), tenant_id)
            else:
                return {"status": "error", "message": "Invalid scan parameters"}

            return {
                "status": "success",
                "discoveries_count": len(discoveries),
                "classifications": {
                    cls.value: len([d for d in discoveries if d.classification == cls]) for cls in DataClassification
                },
                "message": f"Discovered {len(discoveries)} PII/PHI locations",
            }

        except Exception as e:
            logger.error(f"Discovery request failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _handle_retention_job_creation(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle retention job creation request via MCP governance."""
        name = request.get("name")
        description = request.get("description")
        location_ids = request.get("location_ids", [])
        retention_policy = request.get("retention_policy", "standard")
        custom_days = request.get("custom_retention_days")

        try:
            policy_enum = RetentionPolicy(retention_policy)

            job_id = await self.pii_manager.create_retention_job(
                name=name,
                description=description,
                location_ids=location_ids,
                retention_policy=policy_enum,
                custom_retention_days=custom_days,
            )

            return {"status": "success", "job_id": job_id, "message": f"Created retention job: {job_id}"}

        except Exception as e:
            logger.error(f"Retention job creation failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _handle_retention_job_execution(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle retention job execution request via MCP governance."""
        job_id = request.get("job_id")

        try:
            results = await self.pii_manager.execute_retention_job(job_id)

            return {"status": "success", "results": results, "message": f"Executed retention job: {job_id}"}

        except Exception as e:
            logger.error(f"Retention job execution failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _handle_compliance_report(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle compliance report generation request via MCP governance."""
        report_type = request.get("report_type", "summary")
        tenant_id = request.get("tenant_id")

        try:
            if report_type == "summary":
                summary = await self.pii_manager.get_compliance_summary()
                return {"status": "success", "report_type": "summary", "data": summary}
            elif report_type == "detailed":
                # Generate detailed report
                report = {
                    "generated_at": datetime.utcnow().isoformat(),
                    "summary": await self.pii_manager.get_compliance_summary(),
                    "locations": [],
                    "retention_jobs": [],
                }

                # Add location details
                locations = list(self.pii_manager.data_locations.values())
                if tenant_id:
                    locations = [loc for loc in locations if loc.tenant_id == tenant_id]

                for location in locations:
                    report["locations"].append(
                        {
                            "location_id": location.location_id,
                            "path": location.path,
                            "classification": location.classification.value,
                            "compliant": location.compliant,
                            "tenant_id": location.tenant_id,
                        }
                    )

                return {"status": "success", "report_type": "detailed", "data": report}
            else:
                return {"status": "error", "message": "Invalid report type"}

        except Exception as e:
            logger.error(f"Compliance report generation failed: {e}")
            return {"status": "error", "message": str(e)}

    # Monitoring Job Handlers

    async def _daily_compliance_scan(self):
        """Perform daily compliance scan."""
        logger.info("Starting daily compliance scan...")

        try:
            # Run comprehensive scan
            results = await self.pii_manager.scan_all_known_locations()

            total_discoveries = sum(len(locations) for locations in results.values())

            # Check for compliance violations
            violations = [loc for loc in self.pii_manager.data_locations.values() if not loc.compliant]

            if violations:
                logger.warning(f"Daily scan found {len(violations)} compliance violations")
                # Send alerts via governance dashboard if available
                if self.governance_dashboard:
                    await self.governance_dashboard.send_alert(
                        "compliance_violation",
                        f"Daily compliance scan found {len(violations)} violations",
                        severity="high",
                    )

            logger.info(f"Daily compliance scan completed: {total_discoveries} locations monitored")

        except Exception as e:
            logger.error(f"Daily compliance scan failed: {e}")

    async def _weekly_compliance_audit(self):
        """Perform weekly comprehensive compliance audit."""
        logger.info("Starting weekly compliance audit...")

        try:
            summary = await self.pii_manager.get_compliance_summary()

            # Calculate compliance metrics
            total_locations = summary["total_locations"]
            compliant_locations = summary["compliance_status"]["compliant"]
            violation_locations = summary["compliance_status"]["violations"]
            unaudited_locations = summary["compliance_status"]["unaudited"]

            if total_locations > 0:
                compliance_rate = (compliant_locations / total_locations) * 100
            else:
                compliance_rate = 100

            {
                "audit_date": datetime.utcnow().isoformat(),
                "total_locations": total_locations,
                "compliance_rate": compliance_rate,
                "violations": violation_locations,
                "unaudited": unaudited_locations,
                "retention_jobs": summary["retention_jobs"],
            }

            # Log audit results
            logger.info(f"Weekly audit: {compliance_rate:.1f}% compliance rate")

            if compliance_rate < 90:
                logger.warning(f"Compliance rate below threshold: {compliance_rate:.1f}%")
                if self.governance_dashboard:
                    await self.governance_dashboard.send_alert(
                        "low_compliance_rate",
                        f"Weekly audit shows {compliance_rate:.1f}% compliance rate",
                        severity="medium",
                    )

        except Exception as e:
            logger.error(f"Weekly compliance audit failed: {e}")

    async def _monthly_compliance_report(self):
        """Generate monthly compliance report."""
        logger.info("Generating monthly compliance report...")

        try:
            # Generate comprehensive report
            report = {
                "report_date": datetime.utcnow().isoformat(),
                "report_type": "monthly_compliance",
                "summary": await self.pii_manager.get_compliance_summary(),
                "recommendations": [],
            }

            # Add recommendations
            summary = report["summary"]

            if summary["compliance_status"]["violations"] > 0:
                report["recommendations"].append(
                    {
                        "type": "compliance_violations",
                        "priority": "high",
                        "message": f"{summary['compliance_status']['violations']} locations need attention",
                        "action": "Review and remediate compliance violations",
                    }
                )

            if summary["compliance_status"]["unaudited"] > 0:
                report["recommendations"].append(
                    {
                        "type": "audit_required",
                        "priority": "medium",
                        "message": f"{summary['compliance_status']['unaudited']} locations need audit",
                        "action": "Schedule comprehensive audit",
                    }
                )

            pending_jobs = summary["retention_jobs"]["pending_approval"]
            if pending_jobs > 0:
                report["recommendations"].append(
                    {
                        "type": "pending_approvals",
                        "priority": "medium",
                        "message": f"{pending_jobs} retention jobs awaiting approval",
                        "action": "Review and approve retention jobs",
                    }
                )

            # Save report
            report_path = Path(f"reports/compliance/monthly_report_{datetime.utcnow().strftime('%Y_%m')}.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Monthly compliance report saved: {report_path}")

        except Exception as e:
            logger.error(f"Monthly compliance report failed: {e}")

    # Public API Methods

    async def get_tenant_compliance_status(self, tenant_id: str) -> dict[str, Any]:
        """Get compliance status for specific tenant."""
        locations = await self.pii_manager.get_locations_by_tenant(tenant_id)

        classifications = {}
        violations = []

        for location in locations:
            cls = location.classification.value
            classifications[cls] = classifications.get(cls, 0) + 1

            if not location.compliant:
                violations.append(
                    {"location_id": location.location_id, "path": location.path, "violations": location.violations}
                )

        return {
            "tenant_id": tenant_id,
            "total_locations": len(locations),
            "classifications": classifications,
            "compliance_violations": len(violations),
            "violation_details": violations,
            "compliant": len(violations) == 0,
        }

    async def emergency_data_retention(
        self, regulation: ComplianceRegulation, tenant_id: str | None = None
    ) -> dict[str, Any]:
        """Trigger emergency data retention for regulatory compliance."""
        logger.warning(f"Emergency data retention triggered for {regulation.value}")

        # Find all locations under the regulation
        applicable_locations = []
        for location in self.pii_manager.data_locations.values():
            if regulation in location.regulations:
                if tenant_id is None or location.tenant_id == tenant_id:
                    applicable_locations.append(location.location_id)

        if not applicable_locations:
            return {
                "status": "success",
                "message": f"No locations found for {regulation.value}",
                "locations_processed": 0,
            }

        # Create emergency retention job
        job_id = await self.pii_manager.create_retention_job(
            name=f"Emergency {regulation.value} Retention",
            description=f"Emergency data retention for {regulation.value} compliance",
            location_ids=applicable_locations,
            retention_policy=RetentionPolicy.IMMEDIATE,
        )

        # Auto-approve emergency jobs
        job = self.pii_manager.retention_jobs[job_id]
        job.approval_status = "approved"
        job.approver = "emergency_system"
        job.approved_at = datetime.utcnow()
        await self.pii_manager._save_retention_job(job)

        # Execute immediately
        results = await self.pii_manager.execute_retention_job(job_id)

        return {
            "status": "success",
            "job_id": job_id,
            "regulation": regulation.value,
            "locations_processed": results["locations_processed"],
            "records_deleted": results["records_deleted"],
            "message": f"Emergency retention completed for {regulation.value}",
        }


# Convenience function for easy integration
async def setup_aivillage_compliance_integration(config_path: Path | None = None) -> AIVillageComplianceIntegration:
    """Set up complete AIVillage compliance integration."""
    integration = AIVillageComplianceIntegration(config_path)
    await integration.initialize_compliance_infrastructure()
    return integration


if __name__ == "__main__":
    # Example usage
    async def main():
        # Set up complete compliance integration
        integration = await setup_aivillage_compliance_integration()

        # Run initial compliance scan
        print("Running initial compliance scan...")
        discovery_results = await integration.pii_manager.scan_all_known_locations()

        total_discoveries = sum(len(locations) for locations in discovery_results.values())
        print(f"Discovered {total_discoveries} PII/PHI locations")

        # Get compliance summary
        summary = await integration.pii_manager.get_compliance_summary()
        print(f"Compliance summary: {summary}")

        # Generate compliance report via MCP governance
        report = await integration._handle_compliance_report({"report_type": "summary"})
        print(f"Generated report: {report['status']}")

    asyncio.run(main())
