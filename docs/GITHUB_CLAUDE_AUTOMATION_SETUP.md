# GitHub Claude Code Automation Setup Guide

## 🚀 Overview

This guide walks you through setting up automated GitHub workflows with Claude Code integration, enabling you to automate 90% of your development workflow directly from GitHub issues and PRs.

## 📋 Prerequisites

- GitHub repository with admin access
- Claude API key from Anthropic
- Node.js 18+ installed
- GitHub CLI (`gh`) installed

## 🔧 Quick Setup

### Step 1: Initialize Claude Code

```bash
# In your repository root
npx claude-flow init --yes-all
```

### Step 2: Install GitHub App

Run the following command in Claude Code:
```
/github install github app
```

This will:
- Open GitHub app installation page
- Create necessary workflow files
- Set up PR automation

### Step 3: Configure Repository Secrets

1. Go to your repository on GitHub
2. Navigate to: **Settings → Secrets and variables → Actions**
3. Add the following secrets:

| Secret Name | Description | How to Get |
|------------|-------------|------------|
| `CLAUDE_API_KEY` | Your Anthropic API key | [Console](https://console.anthropic.com/) |

### Step 4: Merge Initial PR

After running the GitHub app installation, a PR will be created automatically:
- Review the PR (it adds `.github/workflows/claude-code-integration.yml`)
- Merge it to enable automation

## 🎯 Features & Usage

### 1. Issue Automation

Create issues with `@claude` mentions to trigger automation:

```markdown
@claude create a REST API endpoint for user authentication with JWT tokens
```

Claude will:
- Analyze the requirement
- Create a new branch
- Implement the feature
- Write tests
- Create a PR

### 2. PR Code Reviews

When you create a PR, Claude automatically:
- Reviews all changed files
- Suggests improvements
- Checks for best practices
- Validates test coverage

### 3. Content Generation

In issues or comments, use:

```markdown
@claude generate prd for payment processing feature
```

```markdown
@claude generate xml prompt for data validation logic
```

### 4. Task Management

Claude automatically:
- Updates project boards
- Tracks progress
- Creates todo lists
- Manages milestones

## 📝 Command Reference

### Issue Commands

| Command | Description | Example |
|---------|-------------|---------|
| `@claude create pr` | Create PR from issue | `@claude create pr for user authentication` |
| `@claude update` | Update existing feature | `@claude update login validation` |
| `@claude test` | Run tests | `@claude test authentication module` |
| `@claude generate prd` | Generate PRD | `@claude generate prd for checkout flow` |
| `@claude generate xml` | Generate XML prompt | `@claude generate xml for API endpoint` |
| `@claude generate api` | Generate API docs | `@claude generate api documentation` |
| `@claude review` | Request code review | `@claude review this implementation` |

### PR Commands

| Command | Description | Example |
|---------|-------------|---------|
| `@claude approve` | Approve PR after fixes | `@claude approve after addressing comments` |
| `@claude suggest` | Get improvement suggestions | `@claude suggest optimizations` |
| `@claude benchmark` | Run performance tests | `@claude benchmark this change` |

## 🔄 Workflow Examples

### Example 1: Feature Implementation

1. Create issue:
```markdown
Title: Add dark mode toggle

@claude I want to add a dark mode toggle to the application settings. 
Make sure to:
- Store preference in localStorage
- Apply theme on page load
- Include smooth transitions
- Add tests
```

2. Claude will:
   - Create branch `claude-issue-123`
   - Implement dark mode feature
   - Write unit and integration tests
   - Create PR with full implementation

### Example 2: Bug Fix

1. Create issue:
```markdown
Title: Fix login timeout issue

@claude The login session times out after 5 minutes instead of 30 minutes.
Debug and fix this issue.
```

2. Claude will:
   - Investigate the issue
   - Identify root cause
   - Implement fix
   - Add regression tests
   - Create PR with fix

### Example 3: Documentation

1. Comment on issue:
```markdown
@claude generate prd for the new reporting dashboard feature including:
- Real-time data updates
- Export functionality
- Custom date ranges
- Multiple chart types
```

2. Claude will generate complete PRD with:
   - Executive summary
   - User stories
   - Technical requirements
   - Success metrics

## 📊 Performance & Costs

### Performance Improvements
- **Token Reduction**: 32.3%
- **Speed Improvement**: 2.8-4.4x
- **Automation Coverage**: 90%

### GitHub Actions Costs
- **Free tier**: 2,000 minutes/month
- **Average PR**: ~3 minutes
- **Cost per PR**: ~$0.01-0.03

### Best Practices for Cost Optimization
1. Use `@claude` mentions selectively
2. Batch related issues together
3. Use clear, specific instructions
4. Leverage caching for dependencies

## 🛠️ Advanced Configuration

### Custom Rules

Edit `config/claude-rules.json` to customize:
- Task management preferences
- Code quality standards
- Testing requirements
- Documentation formats

### Workflow Customization

Modify `.github/workflows/claude-code-integration.yml` to:
- Add custom triggers
- Integrate with other tools
- Set up notifications
- Configure branch protection

### Agent Configuration

Customize agent behavior:
```json
{
  "agents": {
    "auto_spawn": true,
    "preferred_topology": "adaptive",
    "max_concurrent": 8,
    "specializations": ["coder", "reviewer", "tester"]
  }
}
```

## 🔍 Monitoring & Debugging

### Check Workflow Status

```bash
# List recent workflow runs
gh run list

# View specific run details
gh run view <run-id>

# Watch live logs
gh run watch
```

### Debug Issues

1. Check workflow logs:
```bash
gh run view --log
```

2. Verify secrets:
```bash
gh secret list
```

3. Test locally:
```bash
node scripts/github-claude-automation.js init
```

## 📈 Metrics & Reporting

Claude tracks:
- Issues processed
- PRs created
- Code review comments
- Test coverage changes
- Performance improvements

View metrics:
```bash
node scripts/github-claude-automation.js stats
```

## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Claude not responding | Check API key and GitHub app installation |
| PR creation fails | Verify branch permissions and conflicts |
| Tests failing | Check test environment and dependencies |
| High costs | Optimize trigger conditions and caching |

### Getting Help

1. Check logs: `gh run view --log`
2. Review documentation: `docs/templates/`
3. Create issue with `@claude help`

## 🎓 Learning from Claude

Claude creates rules to avoid repeating mistakes:
- Automatically adds to `.claude/rules/`
- Learns from code review feedback
- Improves over time

## 🔐 Security Considerations

- Never commit API keys
- Use repository secrets
- Review generated code
- Set up branch protection
- Enable required reviews for sensitive repos

## 📚 Additional Resources

- [Claude API Documentation](https://docs.anthropic.com/)
- [GitHub Actions Documentation](https://docs.github.com/actions)
- [Claude Flow Documentation](https://github.com/ruvnet/claude-flow)

## 🎉 Success Tips

1. **Start Small**: Begin with simple issues to test the setup
2. **Be Specific**: Clear instructions yield better results
3. **Review Output**: Always review generated code
4. **Iterate**: Refine your prompts based on results
5. **Monitor Costs**: Track GitHub Actions usage

---

## Quick Test

After setup, create a test issue:

```markdown
Title: Test Claude Integration

@claude create a simple hello world endpoint that returns the current timestamp
```

If Claude responds and creates a PR, your setup is complete! 🎉

---

*Last Updated: 2024*
*Version: 1.0.0*