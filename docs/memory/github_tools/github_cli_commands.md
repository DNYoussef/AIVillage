# GitHub CLI Commands Reference

## Core GitHub CLI Commands for Claude Code Integration

### Repository Management
```bash
# List recent GitHub Actions runs
gh run list --limit 10

# List available workflows
gh workflow list

# View specific run details
gh run view [RUN_ID]

# Rerun failed workflows
gh run rerun [RUN_ID]
```

### Pull Request Management
```bash
# Create pull request with HEREDOC body
gh pr create --title "title" --body "$(cat <<'EOF'
## Summary
- Key changes summary

## Test plan
- Validation steps

🤖 Generated with [Claude Code](https://claude.ai/code)
EOF
)"

# View PR comments via API
gh api repos/owner/repo/pulls/123/comments

# List PRs
gh pr list

# Check PR status
gh pr checks
```

### Issue Management
```bash
# List issues
gh issue list

# Create issue
gh issue create --title "title" --body "body"

# View issue details
gh issue view [ISSUE_NUMBER]
```

### Repository Analysis
```bash
# Clone repository
gh repo clone owner/repo

# View repository info
gh repo view owner/repo

# Fork repository
gh repo fork owner/repo
```

## Authentication & Configuration
```bash
# Check authentication status
gh auth status

# Login to GitHub
gh auth login

# Set default repository
gh repo set-default owner/repo
```

## Advanced API Usage
```bash
# Get repository information
gh api repos/owner/repo

# Get workflow runs
gh api repos/owner/repo/actions/runs

# Get specific workflow
gh api repos/owner/repo/actions/workflows/workflow.yml
```

## Integration with Claude Code Tools
These commands work seamlessly with:
- Bash tool for execution
- WebFetch for GitHub web interface analysis
- Git operations for local repository management
- File operations for workflow file modifications