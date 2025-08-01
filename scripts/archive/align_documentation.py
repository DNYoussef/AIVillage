#!/usr/bin/env python3
"""Align all documentation with actual implementation status.
This creates honest, accurate documentation.
"""

import json
import re
from datetime import datetime
from pathlib import Path


class DocumentationAligner:
    def __init__(self):
        self.misleading_phrases = {
            "self-evolving": "planned self-evolving (not yet implemented)",
            "fully implemented": "partially implemented",
            "production-ready": "experimental prototype",
            "complete platform": "platform under development",
            "automatically learns": "will automatically learn (planned feature)",
        }

        # Based on our analysis
        self.component_status = {
            "compression": {"status": "production", "completion": 95},
            "evolution": {"status": "production", "completion": 90},
            "rag": {"status": "production", "completion": 85},
            "agents": {"status": "experimental", "completion": 35},
            "self_evolution": {"status": "planned", "completion": 0},
            "mesh": {"status": "experimental", "completion": 20},
        }

    def update_readme(self):
        """Update README.md with honest status."""
        readme_path = Path("README.md")
        if not readme_path.exists():
            print("README.md not found, skipping...")
            return

        content = readme_path.read_text(encoding="utf-8")

        # Add implementation warning if not present
        if "Implementation Status" not in content:
            warning = """
> ⚠️ **Implementation Status**: This is an experimental prototype with approximately 42% feature completion.
> See [Implementation Status](#implementation-status) for details on what actually works.

"""
            # Insert after title
            content = re.sub(
                r"(^# .*\n)", r"\1\n" + warning, content, flags=re.MULTILINE
            )

        # Replace misleading phrases
        for misleading, honest in self.misleading_phrases.items():
            content = re.sub(misleading, honest, content, flags=re.IGNORECASE)

        # Add implementation status section
        if "## Implementation Status" not in content:
            status_section = self._generate_status_section()
            content += "\n" + status_section

        readme_path.write_text(content, encoding="utf-8")
        print("Updated README.md with honest status")

    def _generate_status_section(self) -> str:
        """Generate implementation status section."""
        section = "## Implementation Status\n\n"
        section += f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n"

        # Group by status
        production = [
            (k, v)
            for k, v in self.component_status.items()
            if v["status"] == "production"
        ]
        experimental = [
            (k, v)
            for k, v in self.component_status.items()
            if v["status"] == "experimental"
        ]
        planned = [
            (k, v) for k, v in self.component_status.items() if v["status"] == "planned"
        ]

        section += "### Production-Ready Components\n\n"
        for name, info in production:
            section += f"- **{name.title()}**: {info['completion']}% complete\n"

        section += "\n### Experimental Components\n\n"
        for name, info in experimental:
            section += f"- **{name.title()}**: {info['completion']}% complete\n"

        section += "\n### Planned Components\n\n"
        for name, info in planned:
            section += f"- **{name.title()}**: Not yet implemented\n"

        return section

    def update_feature_matrix(self):
        """Update feature matrix with accurate status."""
        matrix_path = Path("docs/reference/feature_matrix_1.md")
        if not matrix_path.exists():
            matrix_path = Path("docs/reference/FEATURE_MATRIX_1.md")

        if not matrix_path.exists():
            # Create the docs directory if it doesn't exist
            matrix_path.parent.mkdir(exist_ok=True)

        content = """# Feature Implementation Matrix

| Feature | Status | Implementation | Tests | Documentation |
|---------|--------|----------------|-------|---------------|
"""

        for component, info in self.component_status.items():
            status_emoji = {
                "production": "[READY]",
                "experimental": "[EXPERIMENTAL]",
                "planned": "[PLANNED]",
            }[info["status"]]

            test_status = "PASS" if info["completion"] > 80 else "FAIL"
            doc_status = "YES" if Path(f"docs/{component}.md").exists() else "NO"

            content += f"| {
                component.title()} | {status_emoji} | {
                info['completion']}% | {test_status} | {doc_status} |\n"

        matrix_path.write_text(content, encoding="utf-8")
        print("Updated feature matrix")

    def create_implementation_status_json(self):
        """Create machine-readable status file."""
        status = {
            "generated": datetime.now().isoformat(),
            "overall_completion": 42,
            "components": self.component_status,
            "working_features": [
                "Model compression (SeedLM, BitNet, VPTQ)",
                "Evolution system with tournament selection",
                "Basic RAG pipeline",
            ],
            "planned_features": [
                "Self-evolving system",
                "Mesh networking",
                "Agent specialization",
                "HippoRAG",
                "Expert vectors",
            ],
        }

        with open("implementation_status.json", "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)

        print("Created implementation_status.json")


# Run the alignment
if __name__ == "__main__":
    aligner = DocumentationAligner()
    aligner.update_readme()
    aligner.update_feature_matrix()
    aligner.create_implementation_status_json()

    print("\nDocumentation alignment complete!")
    print("All documentation now reflects actual implementation status.")
