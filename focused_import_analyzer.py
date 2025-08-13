#!/usr/bin/env python3
"""
Focused Import Impact Analyzer for High-Priority AIVillage Consolidation

Analyzes import usage for the top duplicate files to assess migration risk.
Focuses on high-priority items from the consolidation plan.
"""

import json
import re
from pathlib import Path


class FocusedImportAnalyzer:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.high_priority_files = [
            "src/core/p2p/p2p_node.py",
            "src/infrastructure/p2p/p2p_node.py",
            "src/production/communications/p2p/p2p_node.py",
            "src/core/resources/device_profiler.py",
            "src/production/monitoring/mobile/device_profiler.py",
            "src/production/rag/rag_system/core/intelligent_chunking.py",
            "src/production/rag/rag_system/core/intelligent_chunking_simple.py",
            "scripts/implement_mesh_protocol.py",
            "scripts/implement_mesh_protocol_fixed.py",
        ]

    def search_file_references(self, target_file: str) -> dict[str, int]:
        """Search for references to a specific file across the codebase."""
        file_path = Path(target_file)
        filename = file_path.stem  # Get filename without extension

        # Generate search patterns
        patterns = [
            filename,
            str(file_path).replace("\\", "/"),
            str(file_path).replace("/", ".").replace("\\", ".").replace(".py", ""),
        ]

        results = {
            "total_references": 0,
            "import_statements": 0,
            "string_references": 0,
            "referencing_files": [],
        }

        print(f"  Searching for references to: {target_file}")

        # Search through Python files
        for py_file in self.repo_root.rglob("*.py"):
            try:
                # Skip virtual environment and cache directories
                if any(
                    part in str(py_file)
                    for part in [
                        "new_env",
                        "old_env",
                        "venv",
                        "__pycache__",
                        "site-packages",
                    ]
                ):
                    continue

                # Skip the target file itself
                if str(py_file.relative_to(self.repo_root)) == target_file:
                    continue

                with open(py_file, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                file_has_refs = False
                import_count = 0
                string_count = 0

                # Check each pattern
                for pattern in patterns:
                    if pattern and pattern in content:
                        # Count import vs string references
                        lines = content.split("\n")
                        for line in lines:
                            line = line.strip()
                            if pattern in line:
                                if line.startswith(
                                    ("import ", "from ")
                                ) and not line.strip().startswith("#"):
                                    import_count += 1
                                else:
                                    string_count += 1
                        file_has_refs = True

                if file_has_refs:
                    relative_path = str(py_file.relative_to(self.repo_root))
                    results["referencing_files"].append(
                        {
                            "file": relative_path,
                            "import_refs": import_count,
                            "string_refs": string_count,
                        }
                    )
                    results["import_statements"] += import_count
                    results["string_references"] += string_count

            except Exception:
                continue

        results["total_references"] = (
            results["import_statements"] + results["string_references"]
        )
        print(
            f"    Found {results['total_references']} references in {len(results['referencing_files'])} files"
        )

        return results

    def analyze_file_apis(self, file_path: str) -> dict[str, list[str]]:
        """Analyze the public APIs exposed by a file."""
        full_path = self.repo_root / file_path
        apis = {"classes": [], "functions": [], "exports": [], "imports": []}

        try:
            if full_path.exists() and full_path.suffix == ".py":
                with open(full_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Extract classes
                class_matches = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
                apis["classes"] = class_matches

                # Extract public functions (not starting with _)
                func_matches = re.findall(
                    r"^def\s+([a-zA-Z][a-zA-Z0-9_]*)", content, re.MULTILINE
                )
                apis["functions"] = [f for f in func_matches if not f.startswith("_")]

                # Extract __all__ exports if present
                all_match = re.search(r"__all__\s*=\s*\[(.*?)\]", content, re.DOTALL)
                if all_match:
                    exports = re.findall(r'["\']([^"\']+)["\']', all_match.group(1))
                    apis["exports"] = exports

                # Extract key imports
                import_lines = re.findall(
                    r"^(import\s+\w+|from\s+\w+.*import.*)", content, re.MULTILINE
                )
                apis["imports"] = import_lines[:10]  # Limit to first 10

        except Exception as e:
            print(f"    Warning: Could not analyze APIs for {file_path}: {e}")

        return apis

    def assess_consolidation_risk(self, file_data: dict) -> dict[str, str]:
        """Assess the risk level for consolidating this file."""
        ref_count = file_data["references"]["total_references"]
        import_count = file_data["references"]["import_statements"]
        file_count = len(file_data["references"]["referencing_files"])

        risk_factors = []

        # Analyze risk factors
        if ref_count > 20:
            risk_factors.append("high_reference_count")
        if import_count > 10:
            risk_factors.append("many_imports")
        if file_count > 8:
            risk_factors.append("widely_used")
        if file_data["apis"]["exports"]:
            risk_factors.append("explicit_exports")
        if len(file_data["apis"]["classes"]) > 3:
            risk_factors.append("complex_api")

        # Determine overall risk
        if len(risk_factors) >= 3 or ref_count > 20:
            risk_level = "HIGH"
        elif len(risk_factors) >= 2 or ref_count > 5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendation": self._get_recommendation(risk_level, ref_count),
        }

    def _get_recommendation(self, risk_level: str, ref_count: int) -> str:
        """Get consolidation recommendation based on risk."""
        if risk_level == "HIGH":
            return "Gradual deprecation with import redirects"
        elif risk_level == "MEDIUM":
            return "Careful consolidation with compatibility shims"
        else:
            return "Direct consolidation - low risk"

    def analyze_high_priority_files(self) -> dict:
        """Analyze import impact for high-priority duplicate files."""
        results = {}

        print("=== Analyzing High-Priority Files ===\n")

        for i, file_path in enumerate(self.high_priority_files):
            print(f"[{i + 1}/{len(self.high_priority_files)}] Analyzing: {file_path}")

            # Check if file exists
            full_path = self.repo_root / file_path
            if not full_path.exists():
                print(f"  File does not exist: {file_path}")
                continue

            # Search for references
            references = self.search_file_references(file_path)

            # Analyze APIs
            apis = self.analyze_file_apis(file_path)

            # Assess risk
            file_data = {"references": references, "apis": apis}
            risk_assessment = self.assess_consolidation_risk(file_data)

            results[file_path] = {
                "exists": True,
                "references": references,
                "apis": apis,
                "risk_assessment": risk_assessment,
            }

            print(f"    Risk Level: {risk_assessment['risk_level']}")
            print(f"    Recommendation: {risk_assessment['recommendation']}\n")

        return results

    def generate_focused_report(self, analysis_results: dict) -> str:
        """Generate focused impact report for high-priority files."""
        report = []
        report.append("# High-Priority Import Impact Analysis")
        report.append("")
        report.append("## Executive Summary")
        report.append("")

        # Count by risk level
        high_risk = sum(
            1
            for r in analysis_results.values()
            if r.get("risk_assessment", {}).get("risk_level") == "HIGH"
        )
        medium_risk = sum(
            1
            for r in analysis_results.values()
            if r.get("risk_assessment", {}).get("risk_level") == "MEDIUM"
        )
        low_risk = sum(
            1
            for r in analysis_results.values()
            if r.get("risk_assessment", {}).get("risk_level") == "LOW"
        )

        report.append(f"- **High-Priority Files Analyzed**: {len(analysis_results)}")
        report.append(f"- **High Risk**: {high_risk} files")
        report.append(f"- **Medium Risk**: {medium_risk} files")
        report.append(f"- **Low Risk**: {low_risk} files")
        report.append("")

        # Detailed analysis by group
        groups = {
            "P2P Implementation": [
                "src/core/p2p/p2p_node.py",
                "src/infrastructure/p2p/p2p_node.py",
                "src/production/communications/p2p/p2p_node.py",
            ],
            "Device Profiler": [
                "src/core/resources/device_profiler.py",
                "src/production/monitoring/mobile/device_profiler.py",
            ],
            "RAG Chunking": [
                "src/production/rag/rag_system/core/intelligent_chunking.py",
                "src/production/rag/rag_system/core/intelligent_chunking_simple.py",
            ],
            "Mesh Protocol": [
                "scripts/implement_mesh_protocol.py",
                "scripts/implement_mesh_protocol_fixed.py",
            ],
        }

        for group_name, file_list in groups.items():
            report.append(f"## {group_name}")
            report.append("")
            report.append("| File | References | Risk Level | Recommendation |")
            report.append("|------|------------|------------|----------------|")

            for file_path in file_list:
                if file_path in analysis_results:
                    data = analysis_results[file_path]
                    refs = data["references"]["total_references"]
                    risk = data["risk_assessment"]["risk_level"]
                    rec = data["risk_assessment"]["recommendation"]
                    report.append(f"| `{file_path}` | {refs} | {risk} | {rec} |")
                else:
                    report.append(f"| `{file_path}` | N/A | N/A | File not found |")

            report.append("")

            # Group-specific recommendations
            group_files_data = [
                analysis_results[f] for f in file_list if f in analysis_results
            ]
            if group_files_data:
                max_refs = max(
                    d["references"]["total_references"] for d in group_files_data
                )
                canonical_file = None
                for f in file_list:
                    if (
                        f in analysis_results
                        and analysis_results[f]["references"]["total_references"]
                        == max_refs
                    ):
                        canonical_file = f
                        break

                if canonical_file:
                    report.append(
                        f"**Recommended Canonical**: `{canonical_file}` (most referenced)"
                    )
                    report.append("")

        # Migration strategy
        report.append("## Migration Strategy")
        report.append("")
        report.append("### Implementation Order")
        report.append("")

        # Sort by risk level for implementation order
        sorted_files = sorted(
            analysis_results.items(),
            key=lambda x: (
                x[1]["risk_assessment"]["risk_level"] == "LOW",
                x[1]["risk_assessment"]["risk_level"] == "MEDIUM",
                x[1]["references"]["total_references"],
            ),
        )

        for i, (file_path, data) in enumerate(sorted_files):
            risk = data["risk_assessment"]["risk_level"]
            refs = data["references"]["total_references"]
            report.append(f"{i + 1}. `{file_path}` - {risk} risk ({refs} references)")

        report.append("")

        # Testing recommendations
        report.append("## Testing Requirements")
        report.append("")
        report.append("### Pre-Migration")
        report.append("```bash")
        report.append("# Baseline test run")
        report.append("python -m pytest tests/ -v")
        report.append("")
        report.append("# Check import structure")
        for file_path in self.high_priority_files[:3]:
            if file_path in analysis_results:
                filename = Path(file_path).stem
                report.append(f"grep -r '{filename}' src/ --include='*.py'")
        report.append("```")
        report.append("")

        report.append("### Post-Migration")
        report.append("```bash")
        report.append("# Verify no import errors")
        report.append(
            'python -c \'import sys; sys.path.append("src"); import importlib; importlib.import_module("core.p2p.p2p_node")\''
        )
        report.append("")
        report.append("# Integration tests")
        report.append("python -m pytest tests/integration/ -v")
        report.append("```")
        report.append("")

        return "\n".join(report)

    def generate_migration_map(self, analysis_results: dict) -> dict:
        """Generate migration map for high-priority files."""
        groups = {
            "P2P Implementation": [
                "src/core/p2p/p2p_node.py",
                "src/infrastructure/p2p/p2p_node.py",
                "src/production/communications/p2p/p2p_node.py",
            ],
            "Device Profiler": [
                "src/core/resources/device_profiler.py",
                "src/production/monitoring/mobile/device_profiler.py",
            ],
            "RAG Chunking": [
                "src/production/rag/rag_system/core/intelligent_chunking.py",
                "src/production/rag/rag_system/core/intelligent_chunking_simple.py",
            ],
            "Mesh Protocol": [
                "scripts/implement_mesh_protocol.py",
                "scripts/implement_mesh_protocol_fixed.py",
            ],
        }

        migration_map = {}

        for group_name, file_list in groups.items():
            # Find canonical file (most referenced or production version)
            canonical = None
            max_refs = -1

            for file_path in file_list:
                if file_path in analysis_results:
                    refs = analysis_results[file_path]["references"]["total_references"]
                    # Prefer production versions
                    if "production" in file_path or refs > max_refs:
                        canonical = file_path
                        max_refs = refs

            if canonical:
                for file_path in file_list:
                    if file_path != canonical and file_path in analysis_results:
                        data = analysis_results[file_path]
                        migration_map[file_path] = {
                            "action": "deprecate"
                            if data["risk_assessment"]["risk_level"] == "HIGH"
                            else "merge",
                            "canonical": canonical,
                            "group": group_name,
                            "dependents_count": data["references"]["total_references"],
                            "risk_level": data["risk_assessment"]["risk_level"],
                            "migration_priority": data["risk_assessment"][
                                "risk_level"
                            ].lower(),
                            "shim_strategy": "import_redirect"
                            if data["references"]["total_references"] > 5
                            else "warning",
                        }

        return migration_map

    def run_focused_analysis(self):
        """Run focused analysis on high-priority files."""
        print("=== Focused Import Impact Analysis ===")
        print("Analyzing high-priority files from consolidation plan\n")

        # Analyze files
        analysis_results = self.analyze_high_priority_files()

        # Generate reports
        report = self.generate_focused_report(analysis_results)
        migration_map = self.generate_migration_map(analysis_results)

        # Write output files
        report_file = self.repo_root / "IMPORT_IMPACT.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Generated: {report_file}")

        map_file = self.repo_root / "RENAME_MAP.json"
        with open(map_file, "w", encoding="utf-8") as f:
            json.dump(migration_map, f, indent=2)
        print(f"Generated: {map_file}")

        # Summary
        total_refs = sum(
            r["references"]["total_references"] for r in analysis_results.values()
        )
        high_risk_count = sum(
            1
            for r in analysis_results.values()
            if r["risk_assessment"]["risk_level"] == "HIGH"
        )

        print("\n=== Analysis Complete ===")
        print(f"Files analyzed: {len(analysis_results)}")
        print(f"Total references found: {total_refs}")
        print(f"High-risk files: {high_risk_count}")
        print(f"Migration mappings: {len(migration_map)}")


def main():
    analyzer = FocusedImportAnalyzer()
    analyzer.run_focused_analysis()


if __name__ == "__main__":
    main()
