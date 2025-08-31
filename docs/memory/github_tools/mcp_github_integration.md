# MCP GitHub Integration Tools Reference

## Claude Flow MCP GitHub Tools

### Available MCP Tools for GitHub Integration

#### 1. GitHub Swarm Coordination
```bash
mcp__claude-flow__github_swarm
```
**Purpose**: Initialize GitHub workflow swarm coordination
**Parameters**:
- `repository`: Target repository name
- `workflow`: Specific workflow type (pr-review-enhancement, issue-triage, etc.)
- `coordination_level`: Level of automation (basic, advanced, full)

**Usage Pattern**:
```javascript
mcp__claude-flow__github_swarm {
  repository: "AIVillage",
  workflow: "pr-review-enhancement",
  coordination_level: "advanced"
}
```

#### 2. Repository Analysis
```bash
mcp__claude-flow__repo_analyze
```
**Purpose**: Comprehensive repository analysis and insights
**Parameters**:
- `repository_url`: Full GitHub repository URL
- `analysis_depth`: Depth of analysis (surface, deep, comprehensive)
- `focus_areas`: Areas to focus on (architecture, security, performance, etc.)

**Usage Pattern**:
```javascript
mcp__claude-flow__repo_analyze {
  repository_url: "https://github.com/user/AIVillage",
  analysis_depth: "comprehensive",
  focus_areas: ["architecture", "security", "workflow-health"]
}
```

#### 3. Enhanced Pull Request Management
```bash
mcp__claude-flow__pr_enhance
```
**Purpose**: AI-powered pull request enhancement and management
**Parameters**:
- `pr_number`: Pull request number
- `analysis_type`: Type of enhancement (review, optimization, validation)
- `auto_review`: Enable automatic review generation
- `coupling_check`: Enable connascence/coupling analysis

**Usage Pattern**:
```javascript
mcp__claude-flow__pr_enhance {
  pr_number: 123,
  analysis_type: "comprehensive",
  auto_review: true,
  coupling_check: true
}
```

#### 4. Intelligent Issue Triage
```bash
mcp__claude-flow__issue_triage
```
**Purpose**: Automated issue classification and prioritization
**Parameters**:
- `issue_number`: Specific issue number (optional)
- `triage_mode`: Triage approach (automatic, assisted, manual)
- `labeling`: Enable automatic labeling
- `assignment`: Enable automatic assignment

**Usage Pattern**:
```javascript
mcp__claude-flow__issue_triage {
  triage_mode: "assisted",
  labeling: true,
  assignment: false
}
```

#### 5. AI-Powered Code Review
```bash
mcp__claude-flow__code_review
```
**Purpose**: Comprehensive AI-powered code review
**Parameters**:
- `files`: Array of files or directories to review
- `focus`: Review focus areas
- `standards`: Code standards to enforce
- `architectural_analysis`: Enable architectural review

**Usage Pattern**:
```javascript
mcp__claude-flow__code_review {
  files: ["src/auth/", "tests/auth/"],
  focus: ["connascence-violations", "architectural-fitness", "security"],
  standards: "project-specific",
  architectural_analysis: true
}
```

## Integration Patterns with Claude Code

### Pattern 1: Workflow Failure Resolution
```javascript
[GitHubWorkflowFix]:
  // Analyze repository and failing workflows
  mcp__claude-flow__repo_analyze {
    repository_url: "current-repo",
    analysis_depth: "deep",
    focus_areas: ["workflow-health", "ci-cd"]
  }

  // Use WebFetch for additional GitHub analysis
  WebFetch("https://github.com/user/repo/actions", "Show recent workflow failures")

  // Traditional Claude Code tools for fixes
  Read(".github/workflows/security-scan.yml")
  Bash("mkdir -p security/reports security/sboms")
  Edit(".github/workflows/security-scan.yml")
```

### Pattern 2: PR Enhancement Workflow
```javascript
[PREnhancement]:
  // Initialize GitHub swarm for PR management
  mcp__claude-flow__github_swarm {
    repository: "current-repo",
    workflow: "pr-review-enhancement"
  }

  // Enhance specific PR
  mcp__claude-flow__pr_enhance {
    pr_number: 42,
    analysis_type: "comprehensive",
    coupling_check: true
  }

  // Code review with architectural focus
  mcp__claude-flow__code_review {
    files: ["changed-files"],
    focus: ["architectural-fitness", "connascence-violations"]
  }
```

### Pattern 3: Repository Health Assessment
```javascript
[RepoHealthCheck]:
  // Comprehensive repository analysis
  mcp__claude-flow__repo_analyze {
    analysis_depth: "comprehensive",
    focus_areas: ["architecture", "security", "performance", "workflow-health"]
  }

  // Issue triage for health-related issues
  mcp__claude-flow__issue_triage {
    triage_mode: "automatic",
    labeling: true
  }

  // Follow up with traditional tools
  Bash("gh workflow list")
  WebFetch("github-actions-url", "Get workflow status")
```

## Best Practices for MCP GitHub Integration

### 1. Batch Operations
Always combine MCP GitHub tools with traditional Claude Code tools in single messages for maximum efficiency.

### 2. Context Preservation
Use MCP tools to maintain context across GitHub operations, then use Claude Code tools for implementation.

### 3. Error Handling
MCP tools provide higher-level error handling and recovery for GitHub operations.

### 4. Coordination Benefits
MCP GitHub tools excel at coordinating complex multi-step GitHub workflows that involve multiple repositories or complex automation.

## Limitations & Fallbacks

### When to Use Traditional Tools Instead:
- Simple GitHub CLI operations (`gh pr list`, `gh run list`)
- Direct file modifications in `.github/workflows/`
- Quick repository cloning or basic Git operations
- Local testing of workflow components

### When MCP Tools Excel:
- Complex multi-repository operations
- Intelligent analysis requiring AI coordination
- Long-running GitHub automation workflows
- Cross-tool coordination (GitHub + Slack + Jira, etc.)
- Advanced pattern recognition in repositories

## Error Recovery Patterns

### GitHub API Rate Limiting:
MCP tools include automatic rate limiting handling and retry mechanisms.

### Authentication Issues:
MCP tools can coordinate with multiple authentication methods and provide fallback strategies.

### Workflow Complexity:
For complex workflows, MCP tools can break down operations into manageable chunks and coordinate execution across multiple agents.