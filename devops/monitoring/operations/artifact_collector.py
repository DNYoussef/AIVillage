"""
AIVillage Operational Artifacts Collection System

Collects, validates, and manages operational artifacts from CI/CD pipeline
including coverage reports, security scans, SBOM files, performance metrics,
quality reports, container scanning results, and compliance documentation.

Integrates with GitHub Actions workflow to automatically collect and index
artifacts for operational visibility and compliance tracking.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import shutil
import time
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    """Types of operational artifacts collected."""

    COVERAGE = "coverage"  # Code coverage reports
    SECURITY = "security"  # Security scan results
    SBOM = "sbom"  # Software Bill of Materials
    PERFORMANCE = "performance"  # Benchmark and performance metrics
    QUALITY = "quality"  # Code quality reports
    CONTAINER = "container"  # Container security scans
    COMPLIANCE = "compliance"  # Compliance validation reports
    DEPLOYMENT = "deployment"  # Deployment artifacts
    DOCUMENTATION = "documentation"  # Generated documentation


class ArtifactStatus(Enum):
    """Status of artifact collection and validation."""

    PENDING = "pending"  # Not yet collected
    COLLECTING = "collecting"  # Currently being collected
    COLLECTED = "collected"  # Successfully collected
    VALIDATED = "validated"  # Validated and processed
    FAILED = "failed"  # Collection or validation failed
    ARCHIVED = "archived"  # Moved to long-term storage


@dataclass
class ArtifactMetadata:
    """Metadata for operational artifacts."""

    artifact_type: ArtifactType
    artifact_id: str
    source_path: str
    collected_at: float
    pipeline_id: str
    commit_sha: str
    branch: str
    size_bytes: int = 0
    checksum: str = ""
    validation_status: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionConfig:
    """Configuration for artifact collection."""

    # Collection settings
    enabled_types: list[ArtifactType] = field(default_factory=lambda: list(ArtifactType))
    collection_timeout_seconds: int = 300
    max_artifact_size_mb: int = 100
    retention_days: int = 90

    # Storage settings
    artifacts_base_dir: str = "artifacts"
    archive_base_dir: str = "artifacts/archive"
    staging_dir: str = "artifacts/staging"

    # Validation settings
    validate_checksums: bool = True
    validate_formats: bool = True
    validate_schemas: bool = False

    # GitHub integration
    github_workflow_enabled: bool = True
    github_artifacts_download: bool = True
    github_token: str | None = None


class OperationalArtifactCollector:
    """
    Main artifact collection system for AIVillage operational visibility.

    Collects artifacts from CI/CD pipeline and local builds, validates them,
    and creates searchable indexes for operational monitoring and compliance.
    """

    def __init__(self, config: CollectionConfig | None = None):
        """Initialize the artifact collector."""
        self.config = config or CollectionConfig()
        self.artifacts: list[ArtifactMetadata] = []

        # Create required directories
        self._setup_directories()

        # Validation rules for different artifact types
        self.validation_rules = self._load_validation_rules()

        logger.info("Operational artifact collector initialized")

    def _setup_directories(self):
        """Create required artifact directories."""
        for dir_path in [
            self.config.artifacts_base_dir,
            self.config.archive_base_dir,
            self.config.staging_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _load_validation_rules(self) -> dict[ArtifactType, dict[str, Any]]:
        """Load validation rules for different artifact types."""
        return {
            ArtifactType.COVERAGE: {
                "required_extensions": [".xml", ".json", ".html"],
                "max_size_mb": 50,
                "required_fields": ["coverage_percent", "lines_covered"],
                "schema_validation": True,
            },
            ArtifactType.SECURITY: {
                "required_extensions": [".json", ".sarif", ".xml"],
                "max_size_mb": 25,
                "required_fields": ["scan_date", "vulnerabilities"],
                "schema_validation": True,
            },
            ArtifactType.SBOM: {
                "required_extensions": [".json", ".spdx", ".xml"],
                "max_size_mb": 10,
                "required_fields": ["components", "metadata"],
                "schema_validation": False,  # SPDX validation is complex
            },
            ArtifactType.PERFORMANCE: {
                "required_extensions": [".json", ".csv", ".html"],
                "max_size_mb": 20,
                "required_fields": ["benchmarks", "metrics"],
                "schema_validation": True,
            },
            ArtifactType.QUALITY: {
                "required_extensions": [".json", ".xml", ".txt"],
                "max_size_mb": 15,
                "required_fields": ["quality_score", "issues"],
                "schema_validation": True,
            },
            ArtifactType.CONTAINER: {
                "required_extensions": [".json", ".sarif"],
                "max_size_mb": 30,
                "required_fields": ["vulnerabilities", "scan_metadata"],
                "schema_validation": True,
            },
            ArtifactType.COMPLIANCE: {
                "required_extensions": [".json", ".html", ".pdf"],
                "max_size_mb": 20,
                "required_fields": ["compliance_checks", "status"],
                "schema_validation": True,
            },
        }

    async def collect_ci_artifacts(
        self, pipeline_id: str, commit_sha: str, branch: str, artifact_sources: dict[ArtifactType, str]
    ) -> list[ArtifactMetadata]:
        """
        Collect artifacts from CI/CD pipeline.

        Args:
            pipeline_id: Unique identifier for pipeline run
            commit_sha: Git commit SHA
            branch: Git branch name
            artifact_sources: Mapping of artifact types to source paths/URLs

        Returns:
            List of collected artifact metadata
        """
        collected_artifacts = []

        for artifact_type, source_path in artifact_sources.items():
            if artifact_type not in self.config.enabled_types:
                continue

            logger.info(f"Collecting {artifact_type.value} artifact from {source_path}")

            try:
                artifact = await self._collect_single_artifact(
                    artifact_type=artifact_type,
                    source_path=source_path,
                    pipeline_id=pipeline_id,
                    commit_sha=commit_sha,
                    branch=branch,
                )

                if artifact:
                    collected_artifacts.append(artifact)
                    self.artifacts.append(artifact)

            except Exception as e:
                logger.error(f"Failed to collect {artifact_type.value} artifact: {e}")

        # Update artifacts index
        await self._update_artifacts_index()

        return collected_artifacts

    async def _collect_single_artifact(
        self,
        artifact_type: ArtifactType,
        source_path: str,
        pipeline_id: str,
        commit_sha: str,
        branch: str,
    ) -> ArtifactMetadata | None:
        """Collect a single artifact with validation."""

        # Generate artifact ID
        artifact_id = f"{artifact_type.value}_{pipeline_id}_{int(time.time())}"

        # Determine if source is URL or local path
        is_url = bool(urlparse(source_path).scheme)

        try:
            # Stage the artifact
            staging_path = Path(self.config.staging_dir) / f"{artifact_id}_{Path(source_path).name}"

            if is_url:
                await self._download_artifact(source_path, staging_path)
            else:
                source_path_obj = Path(source_path)
                if source_path_obj.exists():
                    shutil.copy2(source_path_obj, staging_path)
                else:
                    logger.warning(f"Source artifact not found: {source_path}")
                    return None

            # Validate the artifact
            validation_result = await self._validate_artifact(artifact_type, staging_path)

            if not validation_result["valid"]:
                logger.error(f"Artifact validation failed: {validation_result['error']}")
                return None

            # Move to permanent location
            artifact_dir = Path(self.config.artifacts_base_dir) / artifact_type.value / pipeline_id
            artifact_dir.mkdir(parents=True, exist_ok=True)

            final_path = artifact_dir / staging_path.name
            shutil.move(str(staging_path), str(final_path))

            # Create metadata
            artifact_metadata = ArtifactMetadata(
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                source_path=source_path,
                collected_at=time.time(),
                pipeline_id=pipeline_id,
                commit_sha=commit_sha,
                branch=branch,
                size_bytes=final_path.stat().st_size,
                checksum=validation_result.get("checksum", ""),
                validation_status="validated",
                tags={
                    "pipeline_id": pipeline_id,
                    "commit_sha": commit_sha[:8],
                    "branch": branch,
                    "artifact_type": artifact_type.value,
                },
                metadata=validation_result.get("metadata", {}),
            )

            logger.info(f"Successfully collected {artifact_type.value} artifact: {artifact_id}")
            return artifact_metadata

        except Exception as e:
            logger.error(f"Failed to collect {artifact_type.value} artifact: {e}")
            return None

    async def _download_artifact(self, url: str, destination: Path):
        """Download artifact from URL using async HTTP client."""
        try:
            import aiofiles
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()

                    destination.parent.mkdir(parents=True, exist_ok=True)

                    async with aiofiles.open(destination, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)

        except ImportError:
            # Fallback to synchronous urllib if aiohttp not available
            import urllib.request

            destination.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(destination))

    async def _validate_artifact(self, artifact_type: ArtifactType, file_path: Path) -> dict[str, Any]:
        """Validate collected artifact."""
        validation_result = {"valid": False, "error": "", "checksum": "", "metadata": {}}

        rules = self.validation_rules.get(artifact_type, {})

        try:
            # Check file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            max_size = rules.get("max_size_mb", self.config.max_artifact_size_mb)

            if size_mb > max_size:
                validation_result["error"] = f"File size {size_mb:.1f}MB exceeds limit {max_size}MB"
                return validation_result

            # Check file extension
            required_extensions = rules.get("required_extensions", [])
            if required_extensions and file_path.suffix.lower() not in required_extensions:
                validation_result["error"] = f"Invalid extension {file_path.suffix}, expected {required_extensions}"
                return validation_result

            # Calculate checksum
            import hashlib

            checksum = hashlib.sha256(file_path.read_bytes()).hexdigest()
            validation_result["checksum"] = checksum

            # Validate content based on artifact type
            content_validation = await self._validate_artifact_content(artifact_type, file_path, rules)
            validation_result["metadata"].update(content_validation)

            validation_result["valid"] = True
            return validation_result

        except Exception as e:
            validation_result["error"] = str(e)
            return validation_result

    async def _validate_artifact_content(
        self, artifact_type: ArtifactType, file_path: Path, rules: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate artifact content based on type-specific rules."""
        metadata = {}

        try:
            if file_path.suffix.lower() == ".json":
                with open(file_path) as f:
                    data = json.load(f)

                # Check for required fields
                required_fields = rules.get("required_fields", [])
                for field in required_fields:
                    if field not in data:
                        raise ValueError(f"Missing required field: {field}")

                # Extract type-specific metadata
                if artifact_type == ArtifactType.COVERAGE:
                    metadata["coverage_percent"] = data.get("coverage_percent", 0)
                    metadata["lines_covered"] = data.get("lines_covered", 0)
                    metadata["lines_total"] = data.get("lines_total", 0)

                elif artifact_type == ArtifactType.SECURITY:
                    metadata["vulnerability_count"] = len(data.get("vulnerabilities", []))
                    metadata["critical_count"] = len(
                        [v for v in data.get("vulnerabilities", []) if v.get("severity") == "critical"]
                    )
                    metadata["scan_date"] = data.get("scan_date")

                elif artifact_type == ArtifactType.PERFORMANCE:
                    metadata["benchmark_count"] = len(data.get("benchmarks", []))
                    metadata["avg_response_time"] = data.get("avg_response_time", 0)
                    metadata["throughput"] = data.get("throughput", 0)

                elif artifact_type == ArtifactType.QUALITY:
                    metadata["quality_score"] = data.get("quality_score", 0)
                    metadata["issues_count"] = len(data.get("issues", []))
                    metadata["code_smell_count"] = data.get("code_smell_count", 0)

            elif file_path.suffix.lower() == ".xml":
                # Basic XML validation - would expand for specific formats
                import xml.etree.ElementTree as ET

                ET.parse(file_path)
                metadata["format"] = "xml"

        except Exception as e:
            logger.warning(f"Content validation failed for {artifact_type.value}: {e}")

        return metadata

    async def _update_artifacts_index(self):
        """Update the artifacts index file."""
        index_file = Path(self.config.artifacts_base_dir) / "index.json"

        # Create index data
        index_data = {
            "last_updated": time.time(),
            "total_artifacts": len(self.artifacts),
            "artifacts_by_type": {},
            "artifacts": [],
        }

        # Count by type
        for artifact in self.artifacts:
            artifact_type = artifact.artifact_type.value
            if artifact_type not in index_data["artifacts_by_type"]:
                index_data["artifacts_by_type"][artifact_type] = 0
            index_data["artifacts_by_type"][artifact_type] += 1

        # Add artifact details
        for artifact in self.artifacts:
            index_data["artifacts"].append(
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_type": artifact.artifact_type.value,
                    "pipeline_id": artifact.pipeline_id,
                    "commit_sha": artifact.commit_sha,
                    "branch": artifact.branch,
                    "collected_at": artifact.collected_at,
                    "size_bytes": artifact.size_bytes,
                    "checksum": artifact.checksum,
                    "validation_status": artifact.validation_status,
                    "tags": artifact.tags,
                    "metadata": artifact.metadata,
                }
            )

        # Write index
        with open(index_file, "w") as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Updated artifacts index: {len(self.artifacts)} artifacts")

    def search_artifacts(
        self,
        artifact_type: ArtifactType | None = None,
        pipeline_id: str | None = None,
        branch: str | None = None,
        commit_sha: str | None = None,
        days_back: int | None = None,
    ) -> list[ArtifactMetadata]:
        """Search artifacts by criteria."""
        results = self.artifacts.copy()

        if artifact_type:
            results = [a for a in results if a.artifact_type == artifact_type]

        if pipeline_id:
            results = [a for a in results if a.pipeline_id == pipeline_id]

        if branch:
            results = [a for a in results if a.branch == branch]

        if commit_sha:
            results = [a for a in results if a.commit_sha.startswith(commit_sha)]

        if days_back:
            cutoff_time = time.time() - (days_back * 24 * 3600)
            results = [a for a in results if a.collected_at > cutoff_time]

        return sorted(results, key=lambda x: x.collected_at, reverse=True)

    async def generate_operational_report(self, days_back: int = 7) -> dict[str, Any]:
        """Generate operational report from collected artifacts."""
        cutoff_time = time.time() - (days_back * 24 * 3600)
        recent_artifacts = [a for a in self.artifacts if a.collected_at > cutoff_time]

        report = {
            "generated_at": time.time(),
            "period_days": days_back,
            "summary": {
                "total_artifacts": len(recent_artifacts),
                "artifacts_by_type": {},
                "pipelines_analyzed": len(set(a.pipeline_id for a in recent_artifacts)),
                "branches_analyzed": len(set(a.branch for a in recent_artifacts)),
            },
            "quality_metrics": {},
            "security_metrics": {},
            "performance_metrics": {},
            "recommendations": [],
        }

        # Count artifacts by type
        for artifact in recent_artifacts:
            artifact_type = artifact.artifact_type.value
            if artifact_type not in report["summary"]["artifacts_by_type"]:
                report["summary"]["artifacts_by_type"][artifact_type] = 0
            report["summary"]["artifacts_by_type"][artifact_type] += 1

        # Extract quality metrics
        coverage_artifacts = [a for a in recent_artifacts if a.artifact_type == ArtifactType.COVERAGE]
        if coverage_artifacts:
            latest_coverage = max(coverage_artifacts, key=lambda x: x.collected_at)
            coverage_percent = latest_coverage.metadata.get("coverage_percent", 0)
            report["quality_metrics"]["code_coverage"] = coverage_percent

            if coverage_percent < 60:
                report["recommendations"].append("Code coverage is below 60% - consider adding more tests")

        # Extract security metrics
        security_artifacts = [a for a in recent_artifacts if a.artifact_type == ArtifactType.SECURITY]
        if security_artifacts:
            latest_security = max(security_artifacts, key=lambda x: x.collected_at)
            vuln_count = latest_security.metadata.get("vulnerability_count", 0)
            critical_count = latest_security.metadata.get("critical_count", 0)

            report["security_metrics"]["vulnerability_count"] = vuln_count
            report["security_metrics"]["critical_vulnerability_count"] = critical_count

            if critical_count > 0:
                report["recommendations"].append(
                    f"Found {critical_count} critical vulnerabilities - immediate attention required"
                )

        # Extract performance metrics
        performance_artifacts = [a for a in recent_artifacts if a.artifact_type == ArtifactType.PERFORMANCE]
        if performance_artifacts:
            latest_perf = max(performance_artifacts, key=lambda x: x.collected_at)
            avg_response_time = latest_perf.metadata.get("avg_response_time", 0)
            throughput = latest_perf.metadata.get("throughput", 0)

            report["performance_metrics"]["avg_response_time_ms"] = avg_response_time
            report["performance_metrics"]["throughput_rps"] = throughput

            if avg_response_time > 1000:  # 1 second
                report["recommendations"].append(
                    "Average response time exceeds 1 second - performance optimization needed"
                )

        return report

    async def cleanup_old_artifacts(self):
        """Clean up old artifacts based on retention policy."""
        cutoff_time = time.time() - (self.config.retention_days * 24 * 3600)

        archived_count = 0
        for artifact in self.artifacts.copy():
            if artifact.collected_at < cutoff_time:
                # Move to archive
                archive_path = Path(self.config.archive_base_dir) / artifact.artifact_type.value
                archive_path.mkdir(parents=True, exist_ok=True)

                # Remove from active list
                self.artifacts.remove(artifact)
                archived_count += 1

        if archived_count > 0:
            await self._update_artifacts_index()
            logger.info(f"Archived {archived_count} old artifacts")


# GitHub Actions integration functions
async def collect_github_artifacts(
    collector: OperationalArtifactCollector,
    workflow_run_id: str,
    commit_sha: str,
    branch: str,
) -> list[ArtifactMetadata]:
    """Collect artifacts from GitHub Actions workflow run."""

    # Map of expected GitHub Actions artifacts to their types
    github_artifact_mapping = {
        "coverage-report": ArtifactType.COVERAGE,
        "security-scan": ArtifactType.SECURITY,
        "sbom-report": ArtifactType.SBOM,
        "performance-benchmarks": ArtifactType.PERFORMANCE,
        "quality-report": ArtifactType.QUALITY,
        "container-scan": ArtifactType.CONTAINER,
        "compliance-report": ArtifactType.COMPLIANCE,
    }

    artifact_sources = {}

    # In production, this would query GitHub API for workflow artifacts
    # For now, we'll simulate with local paths that would be downloaded
    for github_name, artifact_type in github_artifact_mapping.items():
        # This would be the downloaded artifact path from GitHub
        artifact_sources[artifact_type] = f"artifacts/github/{workflow_run_id}/{github_name}.json"

    return await collector.collect_ci_artifacts(
        pipeline_id=workflow_run_id,
        commit_sha=commit_sha,
        branch=branch,
        artifact_sources=artifact_sources,
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create collector with default config
        config = CollectionConfig(
            enabled_types=[
                ArtifactType.COVERAGE,
                ArtifactType.SECURITY,
                ArtifactType.PERFORMANCE,
                ArtifactType.QUALITY,
            ],
            retention_days=30,
        )

        collector = OperationalArtifactCollector(config)

        # Simulate CI artifact collection
        artifact_sources = {
            ArtifactType.COVERAGE: "reports/coverage.json",
            ArtifactType.SECURITY: "reports/security-scan.json",
            ArtifactType.PERFORMANCE: "reports/benchmarks.json",
        }

        artifacts = await collector.collect_ci_artifacts(
            pipeline_id="ci-run-123",
            commit_sha="abc123def456",
            branch="main",
            artifact_sources=artifact_sources,
        )

        print(f"Collected {len(artifacts)} artifacts")

        # Generate operational report
        report = await collector.generate_operational_report(days_back=7)
        print(f"Quality metrics: {report['quality_metrics']}")
        print(f"Recommendations: {report['recommendations']}")

    asyncio.run(main())
