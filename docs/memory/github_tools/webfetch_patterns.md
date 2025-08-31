# WebFetch GitHub Analysis Patterns

## WebFetch Integration for GitHub Operations

### Core WebFetch Patterns for GitHub

#### 1. GitHub Actions Analysis
```python
# Access GitHub Actions dashboard
WebFetch("https://github.com/user/repo/actions", "Show recent workflow runs and their status")

# Analyze specific workflow
WebFetch("https://github.com/user/repo/actions/workflows/security-scan.yml", "Get detailed information about security scan workflow")

# View specific run details
WebFetch("https://github.com/user/repo/actions/runs/123456", "Show detailed logs and failure information for this workflow run")
```

#### 2. Pull Request Analysis
```python
# Analyze pull request details
WebFetch("https://github.com/user/repo/pull/123", "Show pull request details, comments, and review status")

# View PR files changed
WebFetch("https://github.com/user/repo/pull/123/files", "Show all files changed in this pull request")

# Check PR conversations
WebFetch("https://github.com/user/repo/pull/123#discussion_r456789", "Show specific review comments and discussions")
```

#### 3. Issue Tracking
```python
# View issue details
WebFetch("https://github.com/user/repo/issues/456", "Show issue details, labels, and discussion thread")

# Analyze issue patterns
WebFetch("https://github.com/user/repo/issues?q=is:issue+is:open+label:bug", "Show all open bug issues")
```

#### 4. Repository Health
```python
# Repository overview
WebFetch("https://github.com/user/repo", "Show repository overview, recent activity, and health indicators")

# Security tab analysis
WebFetch("https://github.com/user/repo/security", "Show security advisories and vulnerability information")

# Insights and analytics
WebFetch("https://github.com/user/repo/pulse", "Show repository activity and contribution patterns")
```

## Integration with Workflow Debugging

### Pattern 1: Workflow Failure Investigation
```javascript
[WorkflowDebug]:
  // First, get repository URL
  Bash("git remote get-url origin")
  
  // Use WebFetch to access GitHub Actions
  WebFetch("https://github.com/user/repo/actions", "Show recent workflow runs and identify failures")
  
  // Get specific workflow details
  WebFetch("https://github.com/user/repo/actions/workflows/failing-workflow.yml", "Show workflow configuration and recent runs")
  
  // Read local workflow file for comparison
  Read(".github/workflows/failing-workflow.yml")
  
  // Test components locally
  Bash("python scripts/architectural_analysis.py --project-root . --output-format json")
```

### Pattern 2: PR Review Enhancement
```javascript
[PRReviewWorkflow]:
  // Analyze PR via web interface
  WebFetch("https://github.com/user/repo/pull/123", "Show PR details and current review status")
  
  // Check changed files
  WebFetch("https://github.com/user/repo/pull/123/files", "Show all files changed in PR")
  
  // Read changed files locally for detailed analysis
  Read("src/changed/file.py")
  Read("tests/changed/test_file.py")
  
  // Provide comprehensive review
  Write("review_comments.md")
```

## Advanced WebFetch Patterns

### 1. GitHub API Data Extraction
```python
# Repository statistics
WebFetch("https://github.com/user/repo/graphs/contributors", "Show contribution statistics and patterns")

# Dependency analysis
WebFetch("https://github.com/user/repo/network/dependencies", "Show repository dependencies and security status")

# Branch protection rules
WebFetch("https://github.com/user/repo/settings/branches", "Show branch protection rules and policies")
```

### 2. Workflow Pattern Analysis
```python
# Compare multiple workflow runs
WebFetch("https://github.com/user/repo/actions/workflows/test.yml?query=event:push", "Show test workflow runs for push events")

# Analyze workflow timing patterns
WebFetch("https://github.com/user/repo/actions", "Show workflow execution times and identify performance bottlenecks")
```

### 3. Security Analysis
```python
# Security advisories
WebFetch("https://github.com/user/repo/security/advisories", "Show security advisories and vulnerability reports")

# Dependabot alerts
WebFetch("https://github.com/user/repo/security/dependabot", "Show automated security update information")

# Code scanning results
WebFetch("https://github.com/user/repo/security/code-scanning", "Show code scanning results and security issues")
```

## Best Practices for WebFetch GitHub Integration

### 1. URL Construction Patterns
```python
# Dynamic URL construction
base_url = "https://github.com/user/repo"
actions_url = f"{base_url}/actions"
pr_url = f"{base_url}/pull/{pr_number}"
workflow_url = f"{base_url}/actions/workflows/{workflow_name}"
```

### 2. Prompt Optimization
```python
# Specific, actionable prompts
WebFetch(url, "Show specific workflow failure logs and identify the exact commands that failed")

# Context-aware prompts
WebFetch(url, "Analyze this PR for architectural changes and potential coupling issues")

# Result-focused prompts
WebFetch(url, "Extract the list of failing tests and their error messages")
```

### 3. Error Handling
```python
# Graceful degradation
try:
    WebFetch("https://github.com/user/repo/actions", "Show workflow status")
except:
    # Fallback to gh CLI
    Bash("gh run list --limit 10")
```

## Integration with Other Tools

### Combining with Bash Commands
```javascript
[GitHubAnalysis]:
  // Get repo info locally
  Bash("git remote get-url origin")
  Bash("gh repo view")
  
  // Web-based analysis
  WebFetch("github-url", "Show detailed web interface information")
  
  // Local verification
  Bash("gh workflow list")
  Read(".github/workflows/main.yml")
```

### Combining with File Operations
```javascript
[WorkflowFix]:
  // Web analysis
  WebFetch("https://github.com/user/repo/actions/runs/123", "Show failure details")
  
  // Local file analysis
  Read(".github/workflows/failing.yml")
  
  // Apply fixes
  Edit(".github/workflows/failing.yml")
  
  // Verify locally
  Bash("yamllint .github/workflows/failing.yml")
```

## Common URL Patterns for WebFetch

### Repository URLs
- Main: `https://github.com/user/repo`
- Actions: `https://github.com/user/repo/actions`
- Security: `https://github.com/user/repo/security`
- Settings: `https://github.com/user/repo/settings`
- Insights: `https://github.com/user/repo/pulse`

### Workflow URLs
- All workflows: `https://github.com/user/repo/actions`
- Specific workflow: `https://github.com/user/repo/actions/workflows/name.yml`
- Workflow run: `https://github.com/user/repo/actions/runs/123456`

### Pull Request URLs
- PR overview: `https://github.com/user/repo/pull/123`
- PR files: `https://github.com/user/repo/pull/123/files`
- PR commits: `https://github.com/user/repo/pull/123/commits`
- PR checks: `https://github.com/user/repo/pull/123/checks`

### Issue URLs
- Issue details: `https://github.com/user/repo/issues/456`
- Issue list: `https://github.com/user/repo/issues`
- Filtered issues: `https://github.com/user/repo/issues?q=is:open+label:bug`

## Performance Considerations

### 1. Rate Limiting
WebFetch is subject to GitHub's web interface rate limits. Use judiciously for complex analysis.

### 2. Data Freshness
WebFetch provides real-time data from GitHub's web interface, more current than cached API responses.

### 3. Content Richness
Web interface often provides more contextual information than API endpoints, especially for visual elements and computed statistics.