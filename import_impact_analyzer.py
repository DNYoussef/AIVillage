#!/usr/bin/env python3
"""
Import Impact Analyzer for AIVillage Consolidation Plan

Analyzes import usage and dependencies for all duplicate files to assess
migration risk and effort required for consolidation.
"""

import json
import re
from collections import defaultdict
from pathlib import Path


class ImportImpactAnalyzer:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.duplicates_data = {}
        self.import_patterns = {}
        self.dependency_map = defaultdict(list)
        self.risk_assessment = {}

    def load_duplicates(self, duplicates_file: str = "tmp_bloat_audit/DUPLICATES.json"):
        """Load duplicate groups from analysis file."""
        duplicates_path = self.repo_root / duplicates_file
        with open(duplicates_path, encoding="utf-8") as f:
            self.duplicates_data = json.load(f)
        print(
            f"Loaded {len(self.duplicates_data.get('duplicates', []))} duplicate groups"
        )

    def extract_all_duplicate_files(self) -> list[dict]:
        """Extract all individual files from duplicate groups."""
        all_files = []
        for group in self.duplicates_data.get("duplicates", []):
            for member in group.get("members", []):
                file_info = {
                    "path": member["path"],
                    "group_label": group.get("task_label", "Unknown"),
                    "priority": group.get("priority", "UNKNOWN"),
                    "role": member.get("role", "unknown"),
                    "loc": member.get("loc", 0),
                    "key_apis": member.get("key_apis", []),
                    "score": member.get("score", 0),
                }
                all_files.append(file_info)

        print(f"Extracted {len(all_files)} duplicate files for analysis")
        return all_files

    def search_python_imports(self, target_file: str) -> dict[str, list[str]]:
        """Search for Python import statements referencing the target file."""
        # Convert file path to import patterns
        file_path = Path(target_file)

        # Generate possible import patterns
        import_patterns = []

        # Direct module import
        if file_path.suffix == ".py":
            module_path = str(file_path.with_suffix(""))
            module_path = module_path.replace("/", ".").replace("\\", ".")

            # Remove common prefixes that wouldn't be in imports
            for prefix in ["src.", "scripts.", "experimental."]:
                if module_path.startswith(prefix):
                    module_path = module_path[len(prefix) :]
                    break

            import_patterns.extend(
                [
                    f"from {module_path}",
                    f"import {module_path}",
                    module_path.split(".")[-1],  # Just the filename
                ]
            )

        # Use native Python search for import patterns
        results = {
            "direct_imports": [],
            "from_imports": [],
            "string_references": [],
            "dynamic_imports": [],
        }

        # Search through all Python files
        for py_file in self.repo_root.rglob("*.py"):
            try:
                # Skip virtual environment directories
                if any(
                    part in str(py_file)
                    for part in ["new_env", "old_env", "venv", "__pycache__"]
                ):
                    continue

                with open(py_file, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Check for import patterns
                for pattern in import_patterns:
                    if pattern and pattern in content:
                        relative_path = str(py_file.relative_to(self.repo_root))
                        if "from" in pattern:
                            results["from_imports"].append(relative_path)
                        elif "import" in pattern:
                            results["direct_imports"].append(relative_path)
                        else:
                            results["string_references"].append(relative_path)

                # Also check for filename references
                filename = file_path.name
                if filename and filename in content:
                    relative_path = str(py_file.relative_to(self.repo_root))
                    if relative_path not in results["string_references"]:
                        results["string_references"].append(relative_path)

            except Exception:
                # Skip files that can't be read
                continue

        # Remove duplicates and the file itself
        for key in results:
            results[key] = list(set(results[key]))
            if target_file in results[key]:
                results[key].remove(target_file)

        return results

    def analyze_public_apis(self, file_path: str) -> dict[str, list[str]]:
        """Analyze if file exposes public APIs via __all__ or __init__.py."""
        full_path = self.repo_root / file_path
        public_apis = {
            "exports": [],
            "init_reexports": [],
            "class_apis": [],
            "function_apis": [],
        }

        try:
            if full_path.exists() and full_path.suffix == ".py":
                with open(full_path, encoding="utf-8") as f:
                    content = f.read()

                # Look for __all__ declarations
                all_match = re.search(r"__all__\s*=\s*\[(.*?)\]", content, re.DOTALL)
                if all_match:
                    exports = re.findall(r'["\']([^"\']+)["\']', all_match.group(1))
                    public_apis["exports"] = exports

                # Look for class definitions
                class_matches = re.findall(r"class\s+(\w+)", content)
                public_apis["class_apis"] = class_matches

                # Look for function definitions (excluding private ones)
                func_matches = re.findall(r"def\s+([a-zA-Z][a-zA-Z0-9_]*)", content)
                public_apis["function_apis"] = [
                    f for f in func_matches if not f.startswith("_")
                ]

            # Check if this module is re-exported in parent __init__.py
            parent_dir = full_path.parent
            init_file = parent_dir / "__init__.py"
            if init_file.exists():
                with open(init_file, encoding="utf-8") as f:
                    init_content = f.read()
                    module_name = full_path.stem
                    if module_name in init_content:
                        public_apis["init_reexports"].append(str(init_file))

        except Exception as e:
            print(f"Warning: Error analyzing APIs for {file_path}: {e}")

        return public_apis

    def count_dependents(self, import_results: dict[str, list[str]]) -> int:
        """Count total number of files that depend on this file."""
        all_dependents = set()
        for _result_type, files in import_results.items():
            all_dependents.update(files)
        return len(all_dependents)

    def assess_migration_risk(
        self, file_info: dict, import_results: dict, api_info: dict
    ) -> dict:
        """Assess the migration risk for consolidating this file."""
        dependent_count = self.count_dependents(import_results)

        # Risk factors
        risk_factors = []

        if dependent_count > 10:
            risk_factors.append("high_dependents")
        if api_info["exports"]:
            risk_factors.append("explicit_exports")
        if api_info["init_reexports"]:
            risk_factors.append("reexported")
        if file_info["priority"] == "HIGH":
            risk_factors.append("high_priority")
        if import_results["dynamic_imports"]:
            risk_factors.append("dynamic_imports")

        # Calculate risk level
        if dependent_count > 10 or len(risk_factors) > 3:
            risk_level = "HIGH"
        elif dependent_count > 3 or len(risk_factors) > 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "dependent_count": dependent_count,
            "risk_factors": risk_factors,
            "migration_complexity": "HIGH"
            if dependent_count > 10
            else "MEDIUM"
            if dependent_count > 3
            else "LOW",
        }

    def generate_migration_strategy(self, file_info: dict, risk_info: dict) -> dict:
        """Generate migration strategy based on risk assessment."""
        strategy = {
            "action": "merge",  # Default
            "shim_strategy": "none",
            "migration_priority": risk_info["risk_level"].lower(),
            "breaking_changes": [],
        }

        # Determine action
        if file_info["role"] == "canonical":
            strategy["action"] = "keep"
        elif risk_info["dependent_count"] == 0:
            strategy["action"] = "archive"
        elif risk_info["risk_level"] == "HIGH":
            strategy["action"] = "deprecate"
            strategy["shim_strategy"] = "import_redirect"
        else:
            strategy["action"] = "merge"

        # Determine shim strategy
        if risk_info["dependent_count"] > 5:
            strategy["shim_strategy"] = "import_redirect"
        elif risk_info["dependent_count"] > 0:
            strategy["shim_strategy"] = "warning"

        # Identify potential breaking changes
        if "explicit_exports" in risk_info["risk_factors"]:
            strategy["breaking_changes"].append("__all__ exports may change")
        if "reexported" in risk_info["risk_factors"]:
            strategy["breaking_changes"].append("init reexports need updating")
        if "dynamic_imports" in risk_info["risk_factors"]:
            strategy["breaking_changes"].append("dynamic imports may break")

        return strategy

    def analyze_all_files(self) -> dict:
        """Analyze import impact for all duplicate files."""
        duplicate_files = self.extract_all_duplicate_files()
        results = {}

        print(f"\nAnalyzing import impact for {len(duplicate_files)} files...")

        for i, file_info in enumerate(duplicate_files):
            file_path = file_info["path"]
            print(f"  [{i + 1}/{len(duplicate_files)}] Analyzing {file_path}")

            # Search for imports
            import_results = self.search_python_imports(file_path)

            # Analyze APIs
            api_info = self.analyze_public_apis(file_path)

            # Assess risk
            risk_info = self.assess_migration_risk(file_info, import_results, api_info)

            # Generate strategy
            strategy = self.generate_migration_strategy(file_info, risk_info)

            results[file_path] = {
                "file_info": file_info,
                "import_results": import_results,
                "api_info": api_info,
                "risk_assessment": risk_info,
                "migration_strategy": strategy,
            }

        return results

    def generate_impact_report(self, analysis_results: dict) -> str:
        """Generate IMPORT_IMPACT.md report."""
        report = []
        report.append("# Import Impact Analysis for AIVillage Consolidation")
        report.append("")
        report.append("## Executive Summary")
        report.append("")

        # Summary statistics
        total_files = len(analysis_results)
        high_impact = sum(
            1
            for r in analysis_results.values()
            if r["risk_assessment"]["risk_level"] == "HIGH"
        )
        medium_impact = sum(
            1
            for r in analysis_results.values()
            if r["risk_assessment"]["risk_level"] == "MEDIUM"
        )
        low_impact = sum(
            1
            for r in analysis_results.values()
            if r["risk_assessment"]["risk_level"] == "LOW"
        )

        report.append(f"- **Total Files Analyzed**: {total_files}")
        report.append(f"- **High Impact Files**: {high_impact} (>10 dependents)")
        report.append(f"- **Medium Impact Files**: {medium_impact} (3-10 dependents)")
        report.append(f"- **Low Impact Files**: {low_impact} (<3 dependents)")
        report.append("")

        # High Impact Files section
        report.append("## High Impact Files (>10 dependents)")
        report.append("")
        report.append("| File | Dependents | Risk Factors | Migration Strategy |")
        report.append("|------|------------|--------------|-------------------|")

        high_impact_files = [
            (path, data)
            for path, data in analysis_results.items()
            if data["risk_assessment"]["risk_level"] == "HIGH"
        ]
        high_impact_files.sort(
            key=lambda x: x[1]["risk_assessment"]["dependent_count"], reverse=True
        )

        for file_path, data in high_impact_files:
            dependents = data["risk_assessment"]["dependent_count"]
            risk_factors = ", ".join(data["risk_assessment"]["risk_factors"])
            strategy = data["migration_strategy"]["action"]
            report.append(
                f"| `{file_path}` | {dependents} | {risk_factors} | {strategy} |"
            )

        report.append("")

        # Medium Impact Files section
        report.append("## Medium Impact Files (3-10 dependents)")
        report.append("")
        report.append("| File | Dependents | Group | Priority |")
        report.append("|------|------------|-------|----------|")

        medium_impact_files = [
            (path, data)
            for path, data in analysis_results.items()
            if data["risk_assessment"]["risk_level"] == "MEDIUM"
        ]
        medium_impact_files.sort(
            key=lambda x: x[1]["risk_assessment"]["dependent_count"], reverse=True
        )

        for file_path, data in medium_impact_files[:20]:  # Limit to top 20
            dependents = data["risk_assessment"]["dependent_count"]
            group = data["file_info"]["group_label"]
            priority = data["file_info"]["priority"]
            report.append(f"| `{file_path}` | {dependents} | {group} | {priority} |")

        report.append("")

        # Migration Strategy section
        report.append("## Migration Strategy by Risk Level")
        report.append("")

        report.append("### High Risk (>10 dependents)")
        report.append("- **Strategy**: Gradual deprecation with import redirects")
        report.append("- **Timeline**: 2-3 releases")
        report.append("- **Testing**: Comprehensive integration testing required")
        report.append("- **Rollback**: Keep deprecated files for 1 release cycle")
        report.append("")

        report.append("### Medium Risk (3-10 dependents)")
        report.append("- **Strategy**: Direct consolidation with warnings")
        report.append("- **Timeline**: 1-2 releases")
        report.append("- **Testing**: Unit and integration tests")
        report.append("- **Rollback**: Standard git revert capabilities")
        report.append("")

        report.append("### Low Risk (<3 dependents)")
        report.append("- **Strategy**: Immediate consolidation")
        report.append("- **Timeline**: Single release")
        report.append("- **Testing**: Basic smoke tests")
        report.append("- **Rollback**: Standard procedures")
        report.append("")

        # Testing Requirements
        report.append("## Testing Requirements")
        report.append("")
        report.append("### Pre-Migration Testing")
        report.append("```bash")
        report.append("# Verify current import structure")
        for file_path, data in high_impact_files[:5]:
            report.append(
                f"rg 'from.*{Path(file_path).stem}|import.*{Path(file_path).stem}' --type py"
            )
        report.append("")
        report.append("# Run baseline tests")
        report.append("python -m pytest tests/ -v")
        report.append("```")
        report.append("")

        report.append("### Post-Migration Validation")
        report.append("```bash")
        report.append("# Verify no import errors")
        report.append(
            'python -c \'import sys; sys.path.insert(0, "src"); import pkgutil; [__import__(name) for _, name, _ in pkgutil.iter_modules(["src"])]\''
        )
        report.append("")
        report.append("# Run integration tests")
        report.append("python -m pytest tests/integration/ -v")
        report.append("```")
        report.append("")

        return "\n".join(report)

    def generate_rename_map(self, analysis_results: dict) -> dict:
        """Generate RENAME_MAP.json with migration details."""
        rename_map = {}

        # Group files by duplicate group
        groups = defaultdict(list)
        for file_path, data in analysis_results.items():
            group_label = data["file_info"]["group_label"]
            groups[group_label].append((file_path, data))

        # Process each group
        for group_label, files in groups.items():
            # Find canonical file (highest score or marked as canonical)
            canonical_file = None
            canonical_data = None

            for file_path, data in files:
                if (
                    data["file_info"]["role"] == "canonical"
                    or data["migration_strategy"]["action"] == "keep"
                ):
                    canonical_file = file_path
                    canonical_data = data
                    break

            if not canonical_file:
                # Choose file with highest score
                files.sort(key=lambda x: x[1]["file_info"]["score"], reverse=True)
                canonical_file, canonical_data = files[0]

            # Map all other files to canonical
            for file_path, data in files:
                if file_path != canonical_file:
                    rename_map[file_path] = {
                        "action": data["migration_strategy"]["action"],
                        "canonical": canonical_file,
                        "shim_strategy": data["migration_strategy"]["shim_strategy"],
                        "dependents_count": data["risk_assessment"]["dependent_count"],
                        "migration_priority": data["migration_strategy"][
                            "migration_priority"
                        ],
                        "breaking_changes": data["migration_strategy"][
                            "breaking_changes"
                        ],
                        "group_label": group_label,
                        "risk_level": data["risk_assessment"]["risk_level"],
                    }

        return rename_map

    def run_analysis(self):
        """Run complete import impact analysis."""
        print("=== AIVillage Import Impact Analysis ===")

        # Load duplicate data
        self.load_duplicates()

        # Analyze all files
        analysis_results = self.analyze_all_files()

        # Generate reports
        impact_report = self.generate_impact_report(analysis_results)
        rename_map = self.generate_rename_map(analysis_results)

        # Write output files
        impact_file = self.repo_root / "IMPORT_IMPACT.md"
        with open(impact_file, "w", encoding="utf-8") as f:
            f.write(impact_report)
        print(f"\nGenerated: {impact_file}")

        rename_file = self.repo_root / "RENAME_MAP.json"
        with open(rename_file, "w", encoding="utf-8") as f:
            json.dump(rename_map, f, indent=2)
        print(f"Generated: {rename_file}")

        # Generate summary
        total_files = len(analysis_results)
        high_risk = sum(
            1
            for r in analysis_results.values()
            if r["risk_assessment"]["risk_level"] == "HIGH"
        )
        total_dependents = sum(
            r["risk_assessment"]["dependent_count"] for r in analysis_results.values()
        )

        print("\n=== Analysis Complete ===")
        print(f"Files analyzed: {total_files}")
        print(f"High-risk files: {high_risk}")
        print(f"Total dependents found: {total_dependents}")
        print(f"Rename mappings: {len(rename_map)}")


def main():
    analyzer = ImportImpactAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
