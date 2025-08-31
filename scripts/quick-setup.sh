#!/bin/bash

# GitHub Claude Code Automation - Quick Setup Script
# Automates 90% of GitHub workflow setup in minutes

set -e

echo "🚀 GitHub Claude Code Automation Setup"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v)
    print_status "Node.js installed: $NODE_VERSION"
else
    print_error "Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Check GitHub CLI
if command -v gh &> /dev/null; then
    print_status "GitHub CLI installed"
else
    print_warning "GitHub CLI not found. Installing..."
    # Attempt to install based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install gh
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
        sudo apt update
        sudo apt install gh
    else
        print_error "Please install GitHub CLI manually: https://cli.github.com/"
        exit 1
    fi
fi

# Check if in git repository
if git rev-parse --git-dir > /dev/null 2>&1; then
    REPO_NAME=$(basename `git rev-parse --show-toplevel`)
    print_status "Git repository detected: $REPO_NAME"
else
    print_error "Not in a git repository. Please run from your project root."
    exit 1
fi

echo ""
echo "🔧 Step 1: Installing Claude Flow"
echo "---------------------------------"

# Install Claude Flow globally
npm install -g claude-flow@alpha || {
    print_error "Failed to install Claude Flow"
    exit 1
}
print_status "Claude Flow installed"

echo ""
echo "🎯 Step 2: Initializing Claude Code"
echo "-----------------------------------"

# Initialize Claude Code
npx claude-flow init --yes-all || {
    print_error "Failed to initialize Claude Code"
    exit 1
}
print_status "Claude Code initialized"

echo ""
echo "📦 Step 3: Setting up GitHub Integration"
echo "----------------------------------------"

# Create .github directory if it doesn't exist
mkdir -p .github/workflows

# Check if workflow already exists
if [ -f ".github/workflows/claude-code-integration.yml" ]; then
    print_status "GitHub workflow already exists"
else
    print_warning "Creating GitHub workflow..."
    # Workflow will be created by the automation script
fi

echo ""
echo "🔐 Step 4: Configuring Secrets"
echo "------------------------------"

# Check for Claude API key
if [ -z "$CLAUDE_API_KEY" ]; then
    print_warning "CLAUDE_API_KEY not found in environment"
    echo ""
    echo "Please add your Claude API key to GitHub Secrets:"
    echo "1. Go to: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/settings/secrets/actions"
    echo "2. Click 'New repository secret'"
    echo "3. Name: CLAUDE_API_KEY"
    echo "4. Value: Your API key from https://console.anthropic.com/"
    echo ""
    read -p "Press Enter once you've added the secret..."
else
    print_status "CLAUDE_API_KEY found in environment"
    
    # Optionally add to GitHub secrets
    read -p "Add CLAUDE_API_KEY to GitHub Secrets? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gh secret set CLAUDE_API_KEY --body "$CLAUDE_API_KEY"
        print_status "Secret added to GitHub"
    fi
fi

echo ""
echo "📂 Step 5: Creating Project Structure"
echo "-------------------------------------"

# Create necessary directories
directories=(
    "config"
    "docs/templates"
    "scripts"
    ".claude/rules"
    ".claude/agents"
    ".hive-mind"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    print_status "Created $dir"
done

echo ""
echo "🚀 Step 6: Installing Dependencies"
echo "----------------------------------"

# Navigate to scripts directory and install dependencies
cd scripts
if [ -f "package.json" ]; then
    npm install
    print_status "Dependencies installed"
else
    print_warning "package.json not found in scripts directory"
fi
cd ..

echo ""
echo "✅ Step 7: Running Initial Setup"
echo "--------------------------------"

# Run the automation initialization
if [ -f "scripts/github-claude-automation.js" ]; then
    node scripts/github-claude-automation.js init
    print_status "Automation initialized"
else
    print_warning "Automation script not found"
fi

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps:"
echo "1. Create a test issue with '@claude' mention"
echo "2. Example: '@claude create a hello world endpoint'"
echo "3. Watch Claude create a PR automatically!"
echo ""
echo "📚 Documentation:"
echo "- Setup Guide: docs/GITHUB_CLAUDE_AUTOMATION_SETUP.md"
echo "- Templates: docs/templates/"
echo "- Configuration: config/claude-rules.json"
echo ""
echo "💡 Quick Commands:"
echo "- Monitor automation: node scripts/github-claude-automation.js monitor"
echo "- View stats: node scripts/github-claude-automation.js stats"
echo "- Generate PRD: node scripts/github-claude-automation.js generate prd"
echo ""
echo "🔗 Useful Links:"
echo "- GitHub Actions: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions"
echo "- Issues: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/issues"
echo "- Settings: https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/settings"
echo ""
print_status "Happy automating! 🤖"