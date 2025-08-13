#!/usr/bin/env python3
"""
Enhanced Import Impact Analyzer

Extends the focused analysis with more detailed import pattern analysis
and provides specific recommendations for each duplicate group.
"""

import json
from pathlib import Path
import re


class EnhancedImportAnalyzer:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)

    def analyze_import_patterns(self, target_file: str) -> dict[str, list[str]]:
        """Analyze specific import patterns for a target file."""
        file_path = Path(target_file)
        filename = file_path.stem

        patterns = {
            "direct_imports": [],  # import module
            "from_imports": [],  # from module import ...
            "relative_imports": [],  # from .module import ...
            "star_imports": [],  # from module import *
            "alias_imports": [],  # import module as alias
            "string_references": [],  # String references in code
        }

        # Search patterns
        search_terms = [
            filename,
            str(file_path).replace("\\", "/"),
            str(file_path).replace("/", ".").replace("\\", ".").replace(".py", ""),
        ]

        for py_file in self.repo_root.rglob("*.py"):
            try:
                if any(part in str(py_file) for part in ["new_env", "old_env", "venv", "__pycache__"]):
                    continue

                if str(py_file.relative_to(self.repo_root)) == target_file:
                    continue

                with open(py_file, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    lines = content.split("\n")

                file_rel_path = str(py_file.relative_to(self.repo_root))

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()

                    for term in search_terms:
                        if term and term in line:
                            # Classify the import type
                            if line.startswith("import ") and term in line:
                                if " as " in line:
                                    patterns["alias_imports"].append(f"{file_rel_path}:{line_num}: {line}")
                                else:
                                    patterns["direct_imports"].append(f"{file_rel_path}:{line_num}: {line}")
                            elif line.startswith("from ") and term in line:
                                if "import *" in line:
                                    patterns["star_imports"].append(f"{file_rel_path}:{line_num}: {line}")
                                elif line.startswith("from ."):
                                    patterns["relative_imports"].append(f"{file_rel_path}:{line_num}: {line}")
                                else:
                                    patterns["from_imports"].append(f"{file_rel_path}:{line_num}: {line}")
                            elif not line.startswith("#") and term in line:
                                patterns["string_references"].append(f"{file_rel_path}:{line_num}: {line}")

            except Exception:
                continue

        return patterns

    def analyze_breaking_changes(self, file_path: str) -> list[str]:
        """Analyze potential breaking changes for a file."""
        full_path = self.repo_root / file_path
        breaking_changes = []

        try:
            if full_path.exists():
                with open(full_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Check for __all__ exports
                if "__all__" in content:
                    breaking_changes.append("__all__ exports - API contract may change")

                # Check for class inheritance
                if re.search(r"class\s+\w+\([^)]+\):", content):
                    breaking_changes.append("Class inheritance - subclasses may break")

                # Check for decorators
                if "@" in content and re.search(r"@\w+", content):
                    breaking_changes.append("Decorators present - behavior changes possible")

                # Check for async/await
                if "async def" in content or "await " in content:
                    breaking_changes.append("Async code - concurrency behavior changes")

                # Check for global variables
                if re.search(r"^[A-Z_][A-Z0-9_]*\s*=", content, re.MULTILINE):
                    breaking_changes.append("Module-level constants - value changes possible")

        except Exception:
            pass

        return breaking_changes

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive import impact report."""
        # Load the basic analysis results
        try:
            with open(self.repo_root / "RENAME_MAP.json") as f:
                rename_map = json.load(f)
        except:
            rename_map = {}

        report = []
        report.append("# Comprehensive Import Impact Analysis for AIVillage Consolidation")
        report.append("")
        report.append("## Executive Summary")
        report.append("")
        report.append("This analysis examines the import usage and migration impact for all files identified in the")
        report.append("consolidation plan. The assessment includes direct imports, string references, and potential")
        report.append("breaking changes to provide a complete migration strategy.")
        report.append("")

        # High-level statistics
        high_priority_files = [
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

        total_files = len(high_priority_files)
        high_risk = len([f for f in rename_map.values() if f["risk_level"] == "HIGH"])
        total_dependents = sum(f["dependents_count"] for f in rename_map.values())

        report.append(f"- **Files Analyzed**: {total_files}")
        report.append(f"- **High-Risk Migrations**: {high_risk}")
        report.append(f"- **Total Dependencies**: {total_dependents}")
        report.append("- **Migration Complexity**: HIGH (requires phased approach)")
        report.append("")

        # Detailed analysis by group
        groups = {
            "P2P Implementation": {
                "files": [
                    "src/core/p2p/p2p_node.py",
                    "src/infrastructure/p2p/p2p_node.py",
                    "src/production/communications/p2p/p2p_node.py",
                ],
                "impact": "CRITICAL - Core networking functionality",
                "canonical": "src/production/communications/p2p/p2p_node.py",
                "reason": "Production-ready implementation with complete P2P features",
            },
            "Device Profiler": {
                "files": [
                    "src/core/resources/device_profiler.py",
                    "src/production/monitoring/mobile/device_profiler.py",
                ],
                "impact": "HIGH - Mobile optimization and resource management",
                "canonical": "src/core/resources/device_profiler.py",
                "reason": "Core implementation with evolution integration",
            },
            "RAG Chunking": {
                "files": [
                    "src/production/rag/rag_system/core/intelligent_chunking.py",
                    "src/production/rag/rag_system/core/intelligent_chunking_simple.py",
                ],
                "impact": "MEDIUM - RAG system functionality",
                "canonical": "src/production/rag/rag_system/core/intelligent_chunking.py",
                "reason": "Full implementation encompasses simple version",
            },
            "Mesh Protocol": {
                "files": [
                    "scripts/implement_mesh_protocol.py",
                    "scripts/implement_mesh_protocol_fixed.py",
                ],
                "impact": "LOW - Development scripts only",
                "canonical": "scripts/implement_mesh_protocol_fixed.py",
                "reason": "Fixed version with improvements",
            },
        }

        for group_name, group_data in groups.items():
            report.append(f"## {group_name}")
            report.append("")
            report.append(f"**Impact Level**: {group_data['impact']}")
            report.append(f"**Canonical Choice**: `{group_data['canonical']}`")
            report.append(f"**Rationale**: {group_data['reason']}")
            report.append("")

            # Analyze import patterns for each file
            for file_path in group_data["files"]:
                if (self.repo_root / file_path).exists():
                    patterns = self.analyze_import_patterns(file_path)
                    breaking_changes = self.analyze_breaking_changes(file_path)

                    report.append(f"### {file_path}")
                    report.append("")

                    # Import pattern summary
                    total_imports = sum(
                        len(patterns[key])
                        for key in [
                            "direct_imports",
                            "from_imports",
                            "relative_imports",
                        ]
                    )
                    total_refs = len(patterns["string_references"])

                    report.append(f"- **Import Statements**: {total_imports}")
                    report.append(f"- **String References**: {total_refs}")
                    report.append(f"- **Star Imports**: {len(patterns['star_imports'])}")

                    if breaking_changes:
                        report.append(f"- **Breaking Change Risks**: {len(breaking_changes)}")
                        for change in breaking_changes:
                            report.append(f"  - {change}")

                    report.append("")

                    # Migration strategy for this file
                    if file_path in rename_map:
                        strategy = rename_map[file_path]
                        report.append(f"**Migration Strategy**: {strategy['action'].title()}")
                        report.append(f"**Risk Level**: {strategy['risk_level']}")
                        report.append(f"**Shim Strategy**: {strategy['shim_strategy']}")
                        report.append("")

        # Migration roadmap
        report.append("## Migration Roadmap")
        report.append("")

        # Phase 1: Low risk items
        report.append("### Phase 1: Low-Risk Consolidations (Week 1)")
        report.append("")
        low_risk_items = [
            ("Mesh Protocol Scripts", "Development tools only", "Zero runtime impact"),
            (
                "Exact duplicates from full analysis",
                "Zero functional differences",
                "Immediate cleanup",
            ),
        ]

        for item, description, benefit in low_risk_items:
            report.append(f"- **{item}**: {description} - {benefit}")
        report.append("")

        # Phase 2: Medium risk items
        report.append("### Phase 2: Medium-Risk Consolidations (Week 2-3)")
        report.append("")
        report.append("- **RAG Chunking**: Consolidate simple and full implementations")
        report.append("- **Add comprehensive test coverage before migration**")
        report.append("- **Create compatibility shims for import transitions**")
        report.append("")

        # Phase 3: High risk items
        report.append("### Phase 3: High-Risk Consolidations (Week 4-6)")
        report.append("")
        report.append("- **P2P Implementation**: Critical infrastructure - requires careful planning")
        report.append("- **Device Profiler**: Mobile subsystem core - extensive testing needed")
        report.append("- **Implement gradual deprecation with import redirects**")
        report.append("- **Maintain backwards compatibility for 1-2 releases**")
        report.append("")

        # Testing strategy
        report.append("## Testing Strategy")
        report.append("")

        report.append("### Pre-Migration Testing")
        report.append("```bash")
        report.append("# 1. Baseline functionality tests")
        report.append("python -m pytest tests/ -v --tb=short")
        report.append("")
        report.append("# 2. Import dependency mapping")
        report.append('python -c "')
        report.append("import ast")
        report.append("import os")
        report.append("for root, dirs, files in os.walk('src'):")
        report.append("    for file in files:")
        report.append("        if file.endswith('.py'):")
        report.append("            print(f'Analyzing {os.path.join(root, file)}')")
        report.append('"')
        report.append("")
        report.append("# 3. P2P system verification")
        report.append("python -m pytest tests/core/p2p/ -v")
        report.append("python -m pytest tests/infrastructure/p2p/ -v")
        report.append("python -m pytest tests/production/communications/ -v")
        report.append("```")
        report.append("")

        report.append("### Migration Testing")
        report.append("```bash")
        report.append("# 1. Gradual import replacement verification")
        report.append("python scripts/validate_imports.py")
        report.append("")
        report.append("# 2. Integration testing at each phase")
        report.append("python -m pytest tests/integration/ -x")
        report.append("")
        report.append("# 3. Performance regression testing")
        report.append("python benchmarks/run_all.py")
        report.append("```")
        report.append("")

        report.append("### Post-Migration Validation")
        report.append("```bash")
        report.append("# 1. No import errors")
        report.append("python -c 'import src; print(\"Import check passed\")'")
        report.append("")
        report.append("# 2. All tests still pass")
        report.append("python -m pytest tests/ -v")
        report.append("")
        report.append("# 3. System health check")
        report.append("python scripts/run_health_check.py")
        report.append("```")
        report.append("")

        # Risk mitigation
        report.append("## Risk Mitigation")
        report.append("")

        report.append("### High-Risk Mitigations")
        report.append("1. **P2P Node Consolidation**:")
        report.append("   - Implement feature flags for gradual transition")
        report.append("   - Create import proxy modules for backward compatibility")
        report.append("   - Extensive integration testing with real P2P scenarios")
        report.append("   - Rollback plan: Keep deprecated modules for 2 releases")
        report.append("")

        report.append("2. **Device Profiler Merge**:")
        report.append("   - Test on multiple mobile device configurations")
        report.append("   - Validate evolution suitability score calculations")
        report.append("   - Monitor performance impact on resource-constrained devices")
        report.append("   - Rollback plan: Core/production separation restoration")
        report.append("")

        report.append("### Medium-Risk Mitigations")
        report.append("1. **RAG Chunking Consolidation**:")
        report.append("   - Verify all simple chunking use cases are covered")
        report.append("   - Performance testing with large document sets")
        report.append("   - Configuration validation for different chunk strategies")
        report.append("")

        # Success metrics
        report.append("## Success Metrics")
        report.append("")
        report.append("### Technical Metrics")
        report.append("- [ ] Zero import errors after migration")
        report.append("- [ ] All existing tests continue to pass")
        report.append("- [ ] P2P connection success rate maintained")
        report.append("- [ ] Mobile device profiling accuracy preserved")
        report.append("- [ ] RAG chunking performance within 5% of baseline")
        report.append("- [ ] System startup time improved by >10%")
        report.append("")

        report.append("### Organizational Metrics")
        report.append("- [ ] 5,000+ LOC reduction achieved")
        report.append("- [ ] Developer onboarding time reduced")
        report.append("- [ ] Code review cycle time improved")
        report.append("- [ ] Maintenance overhead reduced")
        report.append("- [ ] Architecture documentation accuracy increased")
        report.append("")

        # Implementation checklist
        report.append("## Implementation Checklist")
        report.append("")
        report.append("### Preparation Phase")
        report.append("- [ ] Create feature branch for consolidation work")
        report.append("- [ ] Set up automated testing pipeline")
        report.append("- [ ] Create rollback procedures")
        report.append("- [ ] Notify team of upcoming changes")
        report.append("")

        report.append("### Execution Phase")
        report.append("- [ ] Phase 1: Low-risk consolidations")
        report.append("- [ ] Phase 2: Medium-risk consolidations")
        report.append("- [ ] Phase 3: High-risk consolidations")
        report.append("- [ ] Documentation updates")
        report.append("- [ ] Final validation testing")
        report.append("")

        report.append("### Completion Phase")
        report.append("- [ ] Performance benchmarking")
        report.append("- [ ] User acceptance testing")
        report.append("- [ ] Production deployment")
        report.append("- [ ] Post-deployment monitoring")
        report.append("- [ ] Cleanup of deprecated modules (after grace period)")
        report.append("")

        report.append("---")
        report.append("")
        report.append("*This analysis provides a comprehensive roadmap for safely implementing*")
        report.append("*the AIVillage consolidation plan while minimizing risk and maintaining*")
        report.append("*system functionality throughout the migration process.*")

        return "\n".join(report)

    def enhance_rename_map(self) -> dict:
        """Enhance the rename map with additional migration details."""
        try:
            with open(self.repo_root / "RENAME_MAP.json") as f:
                rename_map = json.load(f)
        except:
            return {}

        # Add detailed migration steps for each file
        enhanced_map = {}

        for old_path, mapping in rename_map.items():
            enhanced_mapping = mapping.copy()

            # Add migration steps
            if mapping["risk_level"] == "HIGH":
                enhanced_mapping["migration_steps"] = [
                    "1. Create import proxy module",
                    "2. Add deprecation warnings",
                    "3. Update documentation",
                    "4. Gradual import replacement",
                    "5. Remove after grace period",
                ]
                enhanced_mapping["timeline"] = "4-6 weeks"
                enhanced_mapping["testing_requirements"] = "Comprehensive integration testing"
            elif mapping["risk_level"] == "MEDIUM":
                enhanced_mapping["migration_steps"] = [
                    "1. Add compatibility shims",
                    "2. Update direct imports",
                    "3. Test integration points",
                    "4. Remove old module",
                ]
                enhanced_mapping["timeline"] = "2-3 weeks"
                enhanced_mapping["testing_requirements"] = "Unit and integration testing"
            else:
                enhanced_mapping["migration_steps"] = [
                    "1. Direct replacement",
                    "2. Update imports",
                    "3. Basic smoke testing",
                ]
                enhanced_mapping["timeline"] = "1 week"
                enhanced_mapping["testing_requirements"] = "Smoke testing"

            # Add rollback strategy
            enhanced_mapping["rollback_strategy"] = {
                "method": "Git revert" if mapping["risk_level"] == "LOW" else "Restore deprecated modules",
                "recovery_time": "1 hour" if mapping["risk_level"] == "LOW" else "4 hours",
                "data_loss_risk": "None",
            }

            enhanced_map[old_path] = enhanced_mapping

        return enhanced_map

    def run_enhanced_analysis(self):
        """Run enhanced analysis and generate comprehensive reports."""
        print("=== Enhanced Import Impact Analysis ===")

        # Generate comprehensive report
        comprehensive_report = self.generate_comprehensive_report()

        # Enhance rename map
        enhanced_map = self.enhance_rename_map()

        # Write enhanced outputs
        report_file = self.repo_root / "IMPORT_IMPACT.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(comprehensive_report)
        print(f"Enhanced: {report_file}")

        map_file = self.repo_root / "RENAME_MAP.json"
        with open(map_file, "w", encoding="utf-8") as f:
            json.dump(enhanced_map, f, indent=2)
        print(f"Enhanced: {map_file}")

        print("\n=== Enhanced Analysis Complete ===")
        print("Generated comprehensive migration strategy with:")
        print("- Detailed import pattern analysis")
        print("- Breaking change risk assessment")
        print("- Phased migration roadmap")
        print("- Testing and rollback strategies")
        print("- Success metrics and checklists")


def main():
    analyzer = EnhancedImportAnalyzer()
    analyzer.run_enhanced_analysis()


if __name__ == "__main__":
    main()
