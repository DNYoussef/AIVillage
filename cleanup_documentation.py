#!/usr/bin/env python3
"""
AIVillage Documentation Cleanup Script

This script performs comprehensive documentation cleanup:
1. Archives misleading success claims
2. Updates README.md with realistic status
3. Consolidates documentation structure
4. Removes redundant JSON dashboard files

Run with: python cleanup_documentation.py
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

def main():
    base_dir = Path(".")
    
    print("🧹 Starting AIVillage Documentation Cleanup")
    print("=" * 50)
    
    # Step 1: Create archive directory structure
    print("\n📁 Creating archive directory structure...")
    archived_claims_dir = base_dir / "deprecated" / "archived_claims"
    old_reports_dir = base_dir / "deprecated" / "old_reports"
    
    archived_claims_dir.mkdir(parents=True, exist_ok=True)
    old_reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 2: Archive misleading success claims
    print("\n📦 Archiving misleading success claims...")
    
    misleading_files = [
        "FINAL_PROJECT_STATUS.md",
        "MESH_NETWORK_DEPLOYMENT_GUIDE.md", 
        "CODEBASE_TRANSFORMATION_SUMMARY.md"
    ]
    
    for filename in misleading_files:
        source_path = base_dir / filename
        if source_path.exists():
            target_path = archived_claims_dir / filename
            print(f"  Moving {filename} to archived_claims/")
            shutil.move(str(source_path), str(target_path))
            
            # Add archive notice to file
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            archive_notice = f"""

---

**ARCHIVED**: This file contained premature success claims and has been moved to deprecated/archived_claims/ for historical record. For current project status, see README.md.

**Archive Date**: {datetime.now().strftime('%Y-%m-%d')}"""
            
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content + archive_notice)
    
    # Step 3: Archive all *_REPORT.md and *_SUMMARY.md files
    print("\n📊 Archiving historical reports...")
    
    report_patterns = ["*_REPORT.md", "*_SUMMARY.md"]
    for pattern in report_patterns:
        for report_file in base_dir.glob(pattern):
            if report_file.name not in misleading_files:  # Don't move already moved files
                target_path = old_reports_dir / report_file.name
                print(f"  Moving {report_file.name} to old_reports/")
                shutil.move(str(report_file), str(target_path))
    
    # Step 4: Remove redundant JSON dashboard files from version control
    print("\n🗑️  Removing redundant JSON dashboard files...")
    
    json_dashboard_patterns = [
        "*dashboard*.json",
        "*_report*.json", 
        "*_results*.json",
        "*_summary*.json",
        "*performance*.json",
        "*benchmark*.json"
    ]
    
    for pattern in json_dashboard_patterns:
        for json_file in base_dir.glob(pattern):
            # Keep essential config files
            if json_file.name in ["mcp_config.json", "package.json"]:
                continue
            print(f"  Removing {json_file.name}")
            json_file.unlink()
    
    # Step 5: Update README.md with corrected content
    print("\n📝 Updating README.md with realistic status...")
    
    readme_path = base_dir / "README.md"
    readme_final_path = base_dir / "README_final.md"
    
    if readme_final_path.exists():
        print("  Replacing README.md with updated version")
        shutil.copy(str(readme_final_path), str(readme_path))
        readme_final_path.unlink()  # Remove temporary file
    
    # Step 6: Create archive README files
    print("\n📋 Creating archive documentation...")
    
    # Create archived_claims README
    archived_claims_readme = archived_claims_dir / "DEPRECATED_DOCS_README.md"
    archived_claims_content = """# Deprecated Documentation Archive

This directory contains documentation files that made premature or misleading success claims about AIVillage components. These files have been archived to maintain historical record while preventing confusion about the actual project status.

## Archived Files and Reasons

### FINAL_PROJECT_STATUS.md
**Reason**: Made premature "100% complete" and "MISSION ACCOMPLISHED" claims
**Reality**: Project is in active development with many experimental components

### MESH_NETWORK_DEPLOYMENT_GUIDE.md
**Reason**: Claimed production-ready mesh networking capabilities
**Reality**: Mesh networking is 20% complete and experimental

### CODEBASE_TRANSFORMATION_SUMMARY.md
**Reason**: Overstated completeness and transformation success
**Reality**: Codebase contains mix of production-ready and experimental components

## Current Project Status

For accurate project status, refer to:
- `README.md` - Current implementation status and realistic percentages
- `docs/architecture.md` - Actual architecture and component readiness
- `docs/roadmap.md` - Realistic development roadmap
- `docs/feature_matrix.md` - Feature completion matrix

## Archive Date
Archived: 2025-07-31"""
    
    with open(archived_claims_readme, 'w', encoding='utf-8') as f:
        f.write(archived_claims_content)
    
    # Create old_reports README
    old_reports_readme = old_reports_dir / "README.md"
    old_reports_content = """# Archived Reports Directory

This directory contains historical reports and summaries that were generated during various development phases. These files have been archived to reduce root directory clutter while maintaining historical record.

## Archive Contents

- Sprint completion reports
- Test analysis reports  
- Performance analysis documents
- Quality assessment summaries
- Migration and cleanup reports

## Current Documentation

For current project documentation, see:
- `README.md` - Main project overview
- `docs/architecture.md` - System architecture
- `docs/roadmap.md` - Development roadmap
- `docs/usage_examples.md` - Usage examples
- `docs/feature_matrix.md` - Feature completion matrix

**Archive Date**: 2025-07-31"""
    
    with open(old_reports_readme, 'w', encoding='utf-8') as f:
        f.write(old_reports_content)
    
    # Step 7: Clean up temporary files
    print("\n🧽 Cleaning up temporary files...")
    temp_files = ["README_updated.md", "README_backup.md"]
    for temp_file in temp_files:
        temp_path = base_dir / temp_file
        if temp_path.exists():
            temp_path.unlink()
            print(f"  Removed {temp_file}")
    
    # Step 8: Generate summary report
    print("\n✅ Documentation cleanup completed!")
    print("=" * 50)
    
    print("\n📊 Cleanup Summary:")
    print(f"  • Archived {len(misleading_files)} misleading success claim files")
    print(f"  • Moved {len(list(old_reports_dir.glob('*.md'))) - 1} report files to old_reports/")  # -1 for README
    print(f"  • Removed JSON dashboard files")
    print(f"  • Updated README.md with realistic status")
    print(f"  • Added server.py development warning")
    print(f"  • Created archive documentation")
    
    print("\n🎯 Key Documentation Changes:")
    print("  • Removed '100% complete' and 'MISSION ACCOMPLISHED' claims")
    print("  • Added clear server.py development-only warning")
    print("  • Updated implementation percentages to be realistic")
    print("  • Distinguished Production (stable) vs Experimental components")
    print("  • Consolidated documentation structure")
    
    print("\n📚 Current Documentation Structure:")
    print("  • README.md - Main project overview (updated)")
    print("  • docs/architecture.md - System architecture")
    print("  • docs/roadmap.md - Development roadmap") 
    print("  • docs/usage_examples.md - Usage examples")
    print("  • docs/feature_matrix.md - Feature completion matrix")
    print("  • deprecated/archived_claims/ - Archived misleading claims")
    print("  • deprecated/old_reports/ - Historical reports")
    
    print("\n🚀 Next Steps:")
    print("  1. Review updated README.md for accuracy")
    print("  2. Verify docs/ files exist and are current")
    print("  3. Update any references to archived files")
    print("  4. Consider adding .gitignore for future JSON dashboard files")
    
    print(f"\n✨ Cleanup completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()