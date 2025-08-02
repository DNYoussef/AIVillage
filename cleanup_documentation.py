#!/usr/bin/env python3
"""
Comprehensive Documentation Cleanup Script for AIVillage
========================================================

This script consolidates and organizes scattered markdown files into a clean,
hierarchical documentation structure.

Goals:
1. Archive reports and summaries to deprecated/old_reports/
2. Reorganize docs/ into logical structure
3. Remove duplicates
4. Clean hidden directories
5. Create master documentation index
6. Update cross-references
"""

import os
import shutil
import re
from pathlib import Path
from collections import defaultdict
import json

class DocumentationCleanup:
    def __init__(self, base_path=None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.moved_files = []
        self.duplicates_removed = []
        self.structure_created = []

        # Define target structure
        self.target_structure = {
            'docs/architecture/': [
                'architecture.md', 'architecture_updated.md', 'system_overview.md',
                'design/', 'hyperag_mcp_architecture.md'
            ],
            'docs/guides/': [
                'onboarding.md', 'advanced_setup.md', 'usage_examples.md',
                'migration_notes.md', 'EVOMERGE_GUIDE.md', 'QUIETSTAR_GUIDE.md',
                'compression_guide.md', 'process_standardization_guide.md',
                'interface_standardization_guide.md'
            ],
            'docs/api/': [
                'API_DOCUMENTATION.md', 'specs/', 'hyperag_api.md'
            ],
            'docs/components/': [
                'mesh/', 'rag/', 'twin/', 'agent_forge_pipeline_overview.md',
                'complete_agent_forge_pipeline.md', 'AGENT_FORGE_ANALYSIS.md'
            ],
            'docs/development/': [
                'BRANCHING_STRATEGY.md', 'testing-best-practices.md',
                'test-discovered-behaviors.md', 'COMPRESSION_INTEGRATION.md',
                'SMOKE_TEST_INTEGRATION.md'
            ],
            'docs/reference/': [
                'TODO_1.md', 'roadmap.md', 'benchmark_results.md',
                'DIRECTORY_STRUCTURE_1.md', 'ENTRY_POINTS.md', 'ENTRY_POINT_MAPPING_1.md'
            ],
            'deprecated/old_reports/': [
                # All report files will go here
            ]
        }

        # Files to keep in root
        self.root_files = [
            'README.md', 'CONTRIBUTING.md', 'CHANGELOG.md', 'CLAUDE.local.md'
        ]

        # Report patterns to archive
        self.report_patterns = [
            r'.*_REPORT\.md$', r'.*REPORT\.md$', r'.*_SUMMARY\.md$',
            r'.*SUMMARY\.md$', r'.*_COMPLETE\.md$', r'.*COMPLETE\.md$',
            r'.*_STATUS\.md$', r'.*STATUS\.md$', r'.*_PLAN\.md$',
            r'.*PLAN\.md$', r'.*_CHECKLIST\.md$', r'.*CHECKLIST\.md$',
            r'.*_DASHBOARD\.md$', r'.*DASHBOARD\.md$', r'.*_ANALYSIS\.md$',
            r'.*ANALYSIS\.md$', r'.*_AUDIT\.md$', r'.*AUDIT\.md$',
            r'.*_ROADMAP\.md$', r'.*ROADMAP\.md$', r'.*_GUIDE\.md$'
        ]

    def create_directory_structure(self):
        """Create the target directory structure"""
        print("Creating directory structure...")

        for dir_path in self.target_structure.keys():
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.structure_created.append(str(full_path))
            print(f"  Created: {dir_path}")

        # Create deprecated directory
        deprecated_path = self.base_path / 'deprecated' / 'old_reports'
        deprecated_path.mkdir(parents=True, exist_ok=True)
        self.structure_created.append(str(deprecated_path))
        print(f"  Created: deprecated/old_reports/")

    def is_report_file(self, filename):
        """Check if a file matches report patterns"""
        for pattern in self.report_patterns:
            if re.match(pattern, filename, re.IGNORECASE):
                return True
        return False

    def find_markdown_files(self):
        """Find all markdown files in the project (excluding virtual environments)"""
        md_files = []

        for root, dirs, files in os.walk(self.base_path):
            # Skip virtual environments and node_modules
            dirs[:] = [d for d in dirs if not d.startswith(('env', 'venv', 'node_modules', '__pycache__'))]

            for file in files:
                if file.endswith('.md'):
                    full_path = Path(root) / file
                    relative_path = full_path.relative_to(self.base_path)
                    md_files.append((full_path, relative_path))

        return md_files

    def categorize_file(self, file_path, relative_path):
        """Determine the appropriate category for a file"""
        filename = file_path.name
        path_str = str(relative_path).lower()

        # Keep certain files in root
        if filename in self.root_files:
            return 'root'

        # Archive reports and temporary files
        if self.is_report_file(filename):
            return 'archive'

        # Check existing location for hints
        if 'docs/architecture' in path_str or 'docs/design' in path_str:
            return 'architecture'
        elif 'docs/guides' in path_str or 'guide' in filename.lower():
            return 'guides'
        elif 'docs/api' in path_str or 'api' in filename.lower() or 'docs/specs' in path_str:
            return 'api'
        elif any(comp in path_str for comp in ['mesh', 'rag', 'twin', 'agent_forge']):
            return 'components'
        elif any(dev in path_str for dev in ['test', 'branch', 'development', 'smoke']):
            return 'development'
        elif any(ref in filename.lower() for ref in ['todo', 'roadmap', 'feature', 'benchmark', 'directory', 'entry']):
            return 'reference'

        # Default categorization based on content keywords
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()

            if any(arch in content for arch in ['architecture', 'system design', 'component diagram']):
                return 'architecture'
            elif any(guide in content for guide in ['setup', 'installation', 'how to', 'tutorial']):
                return 'guides'
            elif any(api in content for api in ['endpoint', 'api', 'interface', 'protocol']):
                return 'api'
            elif any(comp in content for comp in ['component', 'module', 'service']):
                return 'components'
            elif any(dev in content for dev in ['test', 'development', 'testing', 'ci/cd']):
                return 'development'
            else:
                return 'reference'
        except Exception:
            return 'reference'

    def find_duplicates(self, md_files):
        """Find duplicate files based on filename"""
        filename_groups = defaultdict(list)

        for file_path, relative_path in md_files:
            base_name = file_path.stem.lower()
            # Handle README variations
            if base_name.startswith('readme'):
                base_name = 'readme'
            filename_groups[base_name].append((file_path, relative_path))

        duplicates = {}
        for base_name, files in filename_groups.items():
            if len(files) > 1:
                # Sort by modification time, keep newest
                files.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
                duplicates[base_name] = files

        return duplicates

    def move_file(self, source_path, target_dir, new_name=None):
        """Move a file to target directory"""
        target_path = self.base_path / target_dir
        target_path.mkdir(parents=True, exist_ok=True)

        filename = new_name if new_name else source_path.name
        destination = target_path / filename

        # Handle conflicts
        counter = 1
        original_destination = destination
        while destination.exists():
            stem = original_destination.stem
            suffix = original_destination.suffix
            destination = target_path / f"{stem}_{counter}{suffix}"
            counter += 1

        try:
            shutil.move(str(source_path), str(destination))
            self.moved_files.append((str(source_path), str(destination)))
            return destination
        except Exception as e:
            print(f"  Error moving {source_path}: {e}")
            return None

    def cleanup_hidden_directories(self):
        """Remove hidden cleanup directories"""
        hidden_dirs = [
            '.claude_analysis', '.claude_cleanup', '.cleanup_analysis',
            '.cleanup_backups', '.test_repair_backup'
        ]

        for hidden_dir in hidden_dirs:
            dir_path = self.base_path / hidden_dir
            if dir_path.exists():
                try:
                    shutil.rmtree(str(dir_path))
                    print(f"  Removed: {hidden_dir}/")
                except Exception as e:
                    print(f"  Error removing {hidden_dir}: {e}")

    def organize_files(self):
        """Main file organization logic"""
        print("\nFinding markdown files...")
        md_files = self.find_markdown_files()
        print(f"Found {len(md_files)} markdown files")

        print("\nFinding duplicates...")
        duplicates = self.find_duplicates(md_files)

        # Remove duplicates (keep newest)
        for base_name, files in duplicates.items():
            print(f"  Duplicate group '{base_name}': {len(files)} files")
            keep_file = files[0]  # Newest
            for file_path, relative_path in files[1:]:
                try:
                    file_path.unlink()
                    self.duplicates_removed.append(str(relative_path))
                    print(f"    Removed duplicate: {relative_path}")
                except Exception as e:
                    print(f"    Error removing {relative_path}: {e}")

        # Re-scan after duplicate removal
        md_files = self.find_markdown_files()

        print(f"\nOrganizing {len(md_files)} files...")

        category_mapping = {
            'architecture': 'docs/architecture/',
            'guides': 'docs/guides/',
            'api': 'docs/api/',
            'components': 'docs/components/',
            'development': 'docs/development/',
            'reference': 'docs/reference/',
            'archive': 'deprecated/old_reports/'
        }

        for file_path, relative_path in md_files:
            if not file_path.exists():
                continue

            category = self.categorize_file(file_path, relative_path)

            if category == 'root':
                print(f"  Keeping in root: {file_path.name}")
                continue

            target_dir = category_mapping.get(category, 'docs/reference/')

            print(f"  Moving {relative_path} -> {target_dir}")
            moved_to = self.move_file(file_path, target_dir)

            if moved_to:
                print(f"    Successfully moved to: {moved_to.relative_to(self.base_path)}")

    def create_documentation_index(self):
        """Create a master documentation index"""
        print("\nCreating documentation index...")

        index_content = """# AIVillage Documentation

Welcome to the AIVillage documentation. This directory contains comprehensive documentation for the AIVillage project, organized into logical categories.

## Documentation Structure

### 📐 Architecture
Core system architecture, design decisions, and component relationships.
- [System Overview](architecture/system_overview.md)
- [Architecture Documentation](architecture/architecture.md)
- [Design Documents](architecture/design/)

### 📚 Guides
Step-by-step guides for setup, configuration, and usage.
- [Onboarding Guide](guides/onboarding.md)
- [Advanced Setup](guides/advanced_setup.md)
- [Usage Examples](guides/usage_examples.md)
- [Migration Notes](guides/migration_notes.md)

### 🔌 API Reference
API documentation, specifications, and interface definitions.
- [API Documentation](api/API_DOCUMENTATION.md)
- [Specifications](api/specs/)

### 🧩 Components
Documentation for individual system components and modules.
- [Mesh Network](components/mesh/)
- [RAG System](components/rag/)
- [Agent Forge Pipeline](components/agent_forge_pipeline_overview.md)

### 🛠️ Development
Development workflows, testing, and contribution guidelines.
- [Branching Strategy](development/BRANCHING_STRATEGY.md)
- [Testing Best Practices](development/testing-best-practices.md)
- [Integration Testing](development/SMOKE_TEST_INTEGRATION.md)

### 📋 Reference
Reference materials, roadmaps, and administrative documentation.
    - [Project Roadmap](reference/roadmap.md)
    - [Feature Matrix](../feature_matrix.md)
    - [Directory Structure](reference/DIRECTORY_STRUCTURE_1.md)
    - [TODO List](reference/TODO_1.md)

## Quick Links

- [Main README](../README.md) - Project overview and quick start
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute to the project
- [Changelog](../CHANGELOG.md) - Project history and releases

## Historical Documentation

Older reports, summaries, and historical documentation have been archived in [`deprecated/old_reports/`](../deprecated/old_reports/) to maintain project history while keeping the main documentation clean and focused.

## Documentation Standards

All documentation in this project follows our [Style Guide](../STYLE_GUIDE.md) for consistency and maintainability.

---

*Last updated: {date}*
"""

        from datetime import datetime
        index_content = index_content.format(date=datetime.now().strftime("%Y-%m-%d"))

        index_path = self.base_path / 'docs' / 'README.md'

        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(index_content)
            print(f"  Created: {index_path.relative_to(self.base_path)}")
        except Exception as e:
            print(f"  Error creating index: {e}")

    def generate_cleanup_report(self):
        """Generate a comprehensive cleanup report"""
        report_content = f"""# Documentation Cleanup Report

## Summary
Successfully reorganized AIVillage documentation structure.

## Actions Taken

### Directory Structure Created
{len(self.structure_created)} directories created:
"""

        for directory in self.structure_created:
            report_content += f"- {Path(directory).relative_to(self.base_path)}\n"

        report_content += f"""
### Files Moved
{len(self.moved_files)} files moved:
"""

        for source, target in self.moved_files:
            source_rel = Path(source).relative_to(self.base_path) if Path(source).is_absolute() else source
            target_rel = Path(target).relative_to(self.base_path) if Path(target).is_absolute() else target
            report_content += f"- {source_rel} → {target_rel}\n"

        report_content += f"""
### Duplicates Removed
{len(self.duplicates_removed)} duplicate files removed:
"""

        for duplicate in self.duplicates_removed:
            report_content += f"- {duplicate}\n"

        report_content += """
### Hidden Directories Cleaned
- .claude_analysis/
- .claude_cleanup/
- .cleanup_analysis/
- .cleanup_backups/
- .test_repair_backup/

## New Documentation Structure

```
docs/
├── README.md                 # Master documentation index
├── architecture/            # System architecture and design
├── guides/                  # User and developer guides
├── api/                     # API documentation
├── components/              # Component-specific docs
├── development/             # Development workflows
└── reference/               # Reference materials

deprecated/
└── old_reports/            # Archived reports and summaries
```

## Next Steps

1. Review moved files for accuracy
2. Update any remaining broken links
3. Validate new documentation structure
4. Update CI/CD to reflect new paths

---
*Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

        from datetime import datetime
        report_path = self.base_path / 'DOCUMENTATION_CLEANUP_REPORT.md'

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"\nCleanup report generated: {report_path.relative_to(self.base_path)}")
        except Exception as e:
            print(f"Error generating report: {e}")

    def run_cleanup(self):
        """Execute the complete cleanup process"""
        print("=" * 60)
        print("AIVillage Documentation Cleanup")
        print("=" * 60)

        try:
            # Step 1: Create directory structure
            self.create_directory_structure()

            # Step 2: Clean hidden directories
            print("\nCleaning hidden directories...")
            self.cleanup_hidden_directories()

            # Step 3: Organize files
            self.organize_files()

            # Step 4: Create documentation index
            self.create_documentation_index()

            # Step 5: Generate report
            self.generate_cleanup_report()

            print("\n" + "=" * 60)
            print("CLEANUP COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"Directories created: {len(self.structure_created)}")
            print(f"Files moved: {len(self.moved_files)}")
            print(f"Duplicates removed: {len(self.duplicates_removed)}")
            print("\nNew documentation structure is ready!")

        except Exception as e:
            print(f"\nERROR during cleanup: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    cleanup = DocumentationCleanup()
    cleanup.run_cleanup()
