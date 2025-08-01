#\!/bin/bash
# AIVillage Archive Repository Creation Script
# This script creates the aivillage-archive repository with git history

set -e

echo "🗄️  Creating AIVillage Archive Repository..."
echo "=============================================="

# Validate prerequisites
if [ \! -d "deprecated" ]; then
    echo "❌ Error: deprecated/ directory not found\!"
    echo "Please run this script from the AIVillage root directory."
    exit 1
fi

echo "✅ Prerequisites validated"
echo "📊 Analyzing content..."

# Show content analysis
echo "Content to archive:"
echo "  - Total files: $(find deprecated -type f | wc -l)"
echo "  - Total size: $(du -sh deprecated | cut -f1)"
echo "  - Categories: $(find deprecated -maxdepth 1 -type d | wc -l) directories"

echo ""
read -p "Continue with archive creation? (y/N): " -n 1 -r
echo
if [[ \! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Archive creation cancelled"
    exit 1
fi

# Create archive directory
echo "🏗️  Step 1: Creating archive directory..."
cd ..
mkdir -p aivillage-archive
cd aivillage-archive

# Initialize git repository
echo "🔧 Step 2: Initializing git repository..."
git init
git branch -m main

# Extract deprecated/ with history
echo "📦 Step 3: Extracting deprecated/ with git history..."
cd ../AIVillage
git subtree push --prefix=deprecated ../aivillage-archive main

# Return to archive and create documentation
cd ../aivillage-archive
echo "📝 Step 4: Creating archive documentation..."

# Create README.md
cat > README.md << 'READMEEOF'
# AIVillage Archive Repository

This repository contains deprecated components from the AIVillage project.

## Archive Contents

- **Total files**: 101
- **Total size**: ~2.2 MB
- **Categories**: 8 directories with legacy components

### Directory Structure

- `archived_claims/` - Documentation with misleading status claims
- `docs_archive/` - Outdated analysis from pre-Sprint 2 development  
- `legacy/` - Legacy code superseded by production/experimental components
- `backups/` - Deprecated development utilities
- Other directories with legacy type stubs and reports

## Purpose

This archive preserves the development history of deprecated components while keeping the main repository focused on active development.

## Important Notes

- **Read-Only**: This repository is for historical reference only
- **No Updates**: Archived components will not receive updates
- **Migration Required**: Use current components in the main repository

## Migration Guidance

See `DEPRECATION_GUIDE.md` in the main AIVillage repository for detailed migration paths and alternatives to deprecated components.

---
*Archive created: $(date)*
*Source: AIVillage deprecated/ directory*
READMEEOF

# Commit the documentation
git add README.md
git commit -m "docs: Add archive repository documentation

This commit adds documentation explaining the purpose and contents
of the AIVillage archive repository.

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

echo ""
echo "✅ Archive repository created successfully\!"
echo ""
echo "📋 Summary:"
echo "  - Location: ../aivillage-archive"
echo "  - Files: $(find . -type f | wc -l) files"
echo "  - Size: $(du -sh . | cut -f1)"
echo "  - Git history: Preserved"
echo ""
echo "🔗 Next steps:"
echo "1. Create GitHub repository: https://github.com/new"
echo "   - Name: aivillage-archive"
echo "   - Description: Archived components from AIVillage project"
echo "   - Public repository"
echo "   - Do NOT initialize with README"
echo ""
echo "2. Push to GitHub:"
echo "   cd ../aivillage-archive"
echo "   git remote add origin https://github.com/YOUR_USERNAME/aivillage-archive.git"
echo "   git push -u origin main"
echo ""
echo "3. Clean main repository:"
echo "   cd ../AIVillage"
echo "   git rm -r deprecated/"
echo "   git commit -m 'feat: Archive deprecated components'"
echo ""
echo "🎯 Archive creation complete\!"
EOF < /dev/null
