#!/usr/bin/env python3
"""
Enhanced Artifact Collector
Collects comprehensive security, quality, and operational artifacts from CI/CD pipelines.
"""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ArtifactInfo:
    name: str
    path: str
    size: int
    checksum: str
    collected: bool
    category: str
    priority: str  # critical, high, medium, low


class EnhancedArtifactCollector:
    """Enhanced artifact collection with security and compliance focus."""

    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.collection_manifest = []

    def collect_security_artifacts(self) -> list[ArtifactInfo]:
        """Collect security-related artifacts."""
        artifacts = []

        # Security scan reports
        security_files = {
            "bandit-report.json": "critical",
            "enhanced-security-report.json": "critical",
            "safety-report.json": "high",
            "p2p-safety-report.json": "high",
            "semgrep-report.json": "high",
            "p2p-sast-report.json": "high",
            "trivy-results*.json": "high",
            "trivy-results*.sarif": "medium",
            "pip-audit-report.json": "medium",
            "sbom-security.json": "medium",
            "arch-security-precheck.json": "medium",
            "p2p-security-precheck.json": "high",
        }

        for pattern, priority in security_files.items():
            artifacts.extend(self._collect_files_by_pattern(pattern, "security", priority, "security/"))

        # Secret scanning artifacts
        secret_files = [".secrets.baseline", ".secrets.audit"]
        for file_path in secret_files:
            if Path(file_path).exists():
                artifacts.append(self._create_artifact_info(file_path, "security", "critical", "security/"))

        return artifacts

    def collect_quality_artifacts(self) -> list[ArtifactInfo]:
        """Collect code quality and architectural artifacts."""
        artifacts = []

        # Quality reports
        quality_files = {
            "quality_gate_result.json": "high",
            "fitness_report.md": "medium",
            "coupling_report.json": "high",
            "antipatterns_report.json": "medium",
            "connascence_report.json": "medium",
            "god_objects.json": "low",
            "complexity_report.md": "medium",
            "architectural-quality-reports-*.zip": "medium",
        }

        for pattern, priority in quality_files.items():
            artifacts.extend(self._collect_files_by_pattern(pattern, "quality", priority, "quality/"))

        return artifacts

    def collect_test_artifacts(self) -> list[ArtifactInfo]:
        """Collect testing artifacts."""
        artifacts = []

        # Test results and coverage
        test_files = {
            "coverage.xml": "high",
            "coverage.json": "medium",
            ".coverage": "medium",
            "pytest-results.xml": "medium",
            "test-results*.xml": "medium",
            ".benchmarks/": "low",
            "performance-reports/": "low",
        }

        for pattern, priority in test_files.items():
            artifacts.extend(self._collect_files_by_pattern(pattern, "testing", priority, "testing/"))

        return artifacts

    def collect_compliance_artifacts(self) -> list[ArtifactInfo]:
        """Collect compliance and audit artifacts."""
        artifacts = []

        # Compliance reports
        compliance_files = {
            "compliance-reports/": "critical",
            "security-compliance/": "critical",
            "scion-security-compliance-*/": "critical",
            "consolidation-report.md": "medium",
            "security-gate-*.json": "high",
            "deployment-gate-*.json": "high",
        }

        for pattern, priority in compliance_files.items():
            artifacts.extend(self._collect_files_by_pattern(pattern, "compliance", priority, "compliance/"))

        return artifacts

    def collect_operational_artifacts(self) -> list[ArtifactInfo]:
        """Collect operational artifacts."""
        artifacts = []

        # Build and deployment artifacts
        operational_files = {
            "build.log": "medium",
            "deploy.log": "medium",
            "metrics-snapshot/": "medium",
            "bench-results/": "low",
            "lint-reports/": "low",
            "dist/": "low",
            "sbom/": "medium",
        }

        for pattern, priority in operational_files.items():
            artifacts.extend(self._collect_files_by_pattern(pattern, "operational", priority, "operational/"))

        return artifacts

    def _collect_files_by_pattern(self, pattern: str, category: str, priority: str, subdir: str) -> list[ArtifactInfo]:
        """Collect files matching a pattern."""
        artifacts = []

        if "*" in pattern or "?" in pattern:
            # Use glob for patterns
            from glob import glob

            files = glob(pattern, recursive=True)
        elif pattern.endswith("/"):
            # Directory pattern
            dir_path = Path(pattern)
            if dir_path.exists() and dir_path.is_dir():
                files = [str(dir_path)]
            else:
                files = []
        else:
            # Single file
            files = [pattern] if Path(pattern).exists() else []

        for file_path in files:
            artifacts.append(self._create_artifact_info(file_path, category, priority, subdir))

        return artifacts

    def _create_artifact_info(self, file_path: str, category: str, priority: str, subdir: str) -> ArtifactInfo:
        """Create artifact info for a file."""
        path = Path(file_path)

        if path.is_file():
            size = path.stat().st_size
            checksum = self._calculate_checksum(path)
        elif path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            checksum = self._calculate_dir_checksum(path)
        else:
            size = 0
            checksum = "missing"

        return ArtifactInfo(
            name=path.name,
            path=str(path),
            size=size,
            checksum=checksum,
            collected=False,
            category=category,
            priority=priority,
        )

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate checksum for {file_path}: {e}")
            return "error"

    def _calculate_dir_checksum(self, dir_path: Path) -> str:
        """Calculate combined checksum for a directory."""
        try:
            hash_sha256 = hashlib.sha256()
            for file_path in sorted(dir_path.rglob("*")):
                if file_path.is_file():
                    hash_sha256.update(str(file_path).encode())
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate directory checksum for {dir_path}: {e}")
            return "error"

    def copy_artifact(self, artifact: ArtifactInfo, target_subdir: str) -> bool:
        """Copy artifact to collection directory."""
        try:
            source_path = Path(artifact.path)
            target_dir = self.output_dir / target_subdir
            target_dir.mkdir(parents=True, exist_ok=True)

            if source_path.is_file():
                target_path = target_dir / artifact.name
                shutil.copy2(source_path, target_path)
                logger.debug(f"Copied file {source_path} to {target_path}")
            elif source_path.is_dir():
                target_path = target_dir / artifact.name
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                logger.debug(f"Copied directory {source_path} to {target_path}")
            else:
                logger.warning(f"Artifact not found: {source_path}")
                return False

            artifact.collected = True
            return True

        except Exception as e:
            logger.error(f"Failed to copy artifact {artifact.path}: {e}")
            return False

    def generate_manifest(self, artifacts: list[ArtifactInfo]) -> dict:
        """Generate collection manifest."""
        manifest = {
            "collection_timestamp": self.timestamp,
            "collection_id": f"artifacts_{self.timestamp}",
            "total_artifacts": len(artifacts),
            "collected_artifacts": len([a for a in artifacts if a.collected]),
            "failed_artifacts": len([a for a in artifacts if not a.collected]),
            "categories": {},
            "priorities": {},
            "artifacts": [],
        }

        # Categorize artifacts
        for artifact in artifacts:
            # By category
            if artifact.category not in manifest["categories"]:
                manifest["categories"][artifact.category] = {"total": 0, "collected": 0}
            manifest["categories"][artifact.category]["total"] += 1
            if artifact.collected:
                manifest["categories"][artifact.category]["collected"] += 1

            # By priority
            if artifact.priority not in manifest["priorities"]:
                manifest["priorities"][artifact.priority] = {"total": 0, "collected": 0}
            manifest["priorities"][artifact.priority]["total"] += 1
            if artifact.collected:
                manifest["priorities"][artifact.priority]["collected"] += 1

            # Add to artifact list
            manifest["artifacts"].append(
                {
                    "name": artifact.name,
                    "path": artifact.path,
                    "size": artifact.size,
                    "checksum": artifact.checksum,
                    "collected": artifact.collected,
                    "category": artifact.category,
                    "priority": artifact.priority,
                }
            )

        return manifest

    def create_archive(self, include_artifacts: bool = True) -> str:
        """Create compressed archive of collected artifacts."""
        archive_name = f"aivillage_artifacts_{self.timestamp}.zip"
        archive_path = self.output_dir / archive_name

        try:
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                if include_artifacts:
                    # Add all files in output directory
                    for file_path in self.output_dir.rglob("*"):
                        if file_path.is_file() and file_path.name != archive_name:
                            arcname = file_path.relative_to(self.output_dir)
                            zipf.write(file_path, arcname)

                # Always add manifest
                manifest_path = self.output_dir / "collection_manifest.json"
                if manifest_path.exists():
                    zipf.write(manifest_path, "collection_manifest.json")

            logger.info(f"Created archive: {archive_path}")
            return str(archive_path)

        except Exception as e:
            logger.error(f"Failed to create archive: {e}")
            return ""

    def collect_all_artifacts(self, create_archive: bool = True) -> dict:
        """Collect all artifacts and generate manifest."""
        logger.info("Starting enhanced artifact collection...")

        # Collect artifacts by category
        all_artifacts = []
        all_artifacts.extend(self.collect_security_artifacts())
        all_artifacts.extend(self.collect_quality_artifacts())
        all_artifacts.extend(self.collect_test_artifacts())
        all_artifacts.extend(self.collect_compliance_artifacts())
        all_artifacts.extend(self.collect_operational_artifacts())

        logger.info(f"Found {len(all_artifacts)} artifacts to collect")

        # Copy artifacts
        collected_count = 0
        failed_count = 0

        # Sort by priority (critical first)
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_artifacts.sort(key=lambda a: priority_order.get(a.priority, 99))

        for artifact in all_artifacts:
            target_subdir = artifact.category
            if self.copy_artifact(artifact, target_subdir):
                collected_count += 1
            else:
                failed_count += 1

        logger.info(f"Collection complete: {collected_count} collected, {failed_count} failed")

        # Generate manifest
        manifest = self.generate_manifest(all_artifacts)

        # Save manifest
        manifest_path = self.output_dir / "collection_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Create summary report
        self._create_summary_report(manifest)

        # Create archive if requested
        if create_archive:
            archive_path = self.create_archive()
            manifest["archive_path"] = archive_path

        return manifest

    def _create_summary_report(self, manifest: dict):
        """Create human-readable summary report."""
        report_path = self.output_dir / "collection_summary.md"

        content = f"""# Artifact Collection Summary

**Collection ID:** {manifest['collection_id']}
**Timestamp:** {manifest['collection_timestamp']}
**Status:** {"✅ SUCCESS" if manifest['failed_artifacts'] == 0 else "⚠️ PARTIAL"}

## Statistics
- **Total Artifacts:** {manifest['total_artifacts']}
- **Collected:** {manifest['collected_artifacts']}
- **Failed:** {manifest['failed_artifacts']}
- **Success Rate:** {(manifest['collected_artifacts'] / manifest['total_artifacts'] * 100) if manifest['total_artifacts'] > 0 else 0:.1f}%

## By Category
"""

        for category, stats in manifest["categories"].items():
            success_rate = (stats["collected"] / stats["total"] * 100) if stats["total"] > 0 else 0
            content += f"- **{category.title()}:** {stats['collected']}/{stats['total']} ({success_rate:.1f}%)\n"

        content += "\n## By Priority\n"
        for priority, stats in manifest["priorities"].items():
            success_rate = (stats["collected"] / stats["total"] * 100) if stats["total"] > 0 else 0
            content += f"- **{priority.title()}:** {stats['collected']}/{stats['total']} ({success_rate:.1f}%)\n"

        # Critical failures
        failed_critical = [a for a in manifest["artifacts"] if not a["collected"] and a["priority"] == "critical"]
        if failed_critical:
            content += "\n## ⚠️ Critical Artifacts Failed\n"
            for artifact in failed_critical:
                content += f"- **{artifact['name']}** ({artifact['category']})\n"

        with open(report_path, "w") as f:
            f.write(content)

        logger.info(f"Summary report created: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Artifact Collector")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory for collected artifacts")
    parser.add_argument("--no-archive", action="store_true", help="Skip creating compressed archive")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel collection (future enhancement)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create collector and run collection
    collector = EnhancedArtifactCollector(args.output_dir)
    manifest = collector.collect_all_artifacts(create_archive=not args.no_archive)

    # Print summary
    print("\nArtifact Collection Complete!")
    print(f"Collection ID: {manifest['collection_id']}")
    print(f"Total Artifacts: {manifest['total_artifacts']}")
    print(f"Collected: {manifest['collected_artifacts']}")
    print(f"Failed: {manifest['failed_artifacts']}")

    if manifest["failed_artifacts"] > 0:
        critical_failed = len([a for a in manifest["artifacts"] if not a["collected"] and a["priority"] == "critical"])
        if critical_failed > 0:
            print(f"⚠️ Critical artifacts failed: {critical_failed}")
            return 1
        else:
            print("⚠️ Some non-critical artifacts failed")
            return 0
    else:
        print("✅ All artifacts collected successfully")
        return 0


if __name__ == "__main__":
    sys.exit(main())
