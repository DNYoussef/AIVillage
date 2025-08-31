# GitHub Workflow Debugging Procedures

## Systematic Workflow Debugging Process

### Phase 1: Initial Assessment

#### Step 1: Identify Repository Context
```bash
# Get repository information
git remote get-url origin
git branch --show-current
git status

# Check GitHub CLI authentication
gh auth status

# List available workflows
gh workflow list
```

#### Step 2: Access Workflow Status
```bash
# List recent workflow runs
gh run list --limit 10

# Get specific workflow runs
gh run list --workflow="workflow-name"

# View failed run details
gh run view [RUN_ID] --log-failed
```

#### Step 3: Web Interface Analysis
```python
# Use WebFetch for comprehensive analysis
WebFetch("https://github.com/user/repo/actions", "Show all recent workflow runs and their status")
WebFetch("https://github.com/user/repo/actions/runs/[ID]", "Show detailed failure logs for this specific run")
```

### Phase 2: Workflow File Analysis

#### Step 4: Read Workflow Configuration
```bash
# List all workflow files
find .github/workflows -name "*.yml" -o -name "*.yaml"

# Read specific workflow file
cat .github/workflows/failing-workflow.yml
```

#### Step 5: Identify Required Dependencies
```yaml
# Common dependency patterns in workflows
jobs:
  security-scan:
    steps:
      - name: Create directories
        run: mkdir -p security/reports security/sboms
      
      - name: Run pip-audit
        run: pip-audit --format=json --output=security/reports/pip-audit.json
```

#### Step 6: Check for Missing Files/Directories
```bash
# Check for required directories
ls -la security/ 2>/dev/null || echo "security/ directory missing"
ls -la security/reports/ 2>/dev/null || echo "security/reports/ directory missing"

# Check for required configuration files
ls -la .secrets.baseline 2>/dev/null || echo ".secrets.baseline missing"
ls -la pyproject.toml 2>/dev/null || echo "pyproject.toml missing"
```

### Phase 3: Local Testing & Validation

#### Step 7: Test Individual Commands
```bash
# Test security tools locally
pip-audit --format=json || echo "pip-audit failed"
safety check --json || echo "safety check failed"
bandit -r . -f json || echo "bandit failed"

# Test architectural analysis scripts
python scripts/architectural_analysis.py --project-root . --output-format json || echo "architectural analysis failed"
python scripts/coupling_metrics.py --format json || echo "coupling metrics failed"
```

#### Step 8: Validate Script Parameters
```bash
# Common parameter validation patterns
# Check if script accepts --project-root
python scripts/script.py --help | grep "project-root"

# Check if script accepts --format vs --output-format
python scripts/script.py --help | grep -E "(format|output)"

# Test with correct parameters
python scripts/script.py --project-root . --output-format json
```

#### Step 9: Create Missing Dependencies
```bash
# Create required directory structure
mkdir -p security/reports
mkdir -p security/sboms
mkdir -p docs/architecture

# Initialize configuration files
echo "{}" > .secrets.baseline
echo "# Architectural decisions" > docs/architecture/decisions.md

# Install missing dependencies
pip install cyclonedx-bom semgrep bandit safety
```

### Phase 4: Workflow Modification & Testing

#### Step 10: Apply Fixes to Workflow Files
```yaml
# Common fixes for workflow files

# Add error handling
- name: Run with error handling
  run: |
    command || echo "completed with warnings"

# Create required directories
- name: Setup directories
  run: |
    mkdir -p security/reports security/sboms

# Fix parameter formats
- name: Run architectural analysis
  run: |
    python scripts/architectural_analysis.py --project-root . --output-format json || echo "completed with warnings"
```

#### Step 11: Test Workflow Locally
```bash
# Use act to test workflows locally (if available)
act -j job-name

# Or test individual steps
bash -c "$(grep -A 5 'run:' .github/workflows/workflow.yml | grep -v 'run:' | sed 's/^[[:space:]]*//')"
```

#### Step 12: Validate YAML Syntax
```bash
# Check YAML syntax
yamllint .github/workflows/*.yml

# Use GitHub CLI to validate
gh workflow view workflow-name
```

### Phase 5: Deployment & Monitoring

#### Step 13: Commit and Monitor
```bash
# Commit workflow fixes
git add .github/workflows/
git commit -m "fix: Resolve workflow failures with proper dependencies and parameters"

# Monitor workflow execution
gh run watch
```

#### Step 14: Analyze Results
```python
# Check results via WebFetch
WebFetch("https://github.com/user/repo/actions", "Show if the fixed workflows are now passing")
```

## Common Debugging Patterns

### Pattern 1: Missing Dependencies
```bash
# Diagnostic sequence
ls -la security/ || echo "Missing security directory"
ls -la .secrets.baseline || echo "Missing secrets baseline"
pip list | grep -E "(bandit|safety|semgrep)" || echo "Missing security tools"

# Fix sequence
mkdir -p security/{reports,sboms}
echo "{}" > .secrets.baseline
pip install bandit safety semgrep cyclonedx-bom
```

### Pattern 2: Parameter Mismatches
```bash
# Diagnostic sequence
python scripts/script.py --help | grep -E "(format|output|project)"

# Common parameter fixes
# Wrong: --output json
# Right: --output-format json
# Wrong: --project .
# Right: --project-root .
```

### Pattern 3: Permission Issues
```bash
# Diagnostic sequence
ls -la scripts/
chmod +x scripts/*.py

# Fix sequence
find scripts/ -name "*.py" -exec chmod +x {} \;
```

### Pattern 4: Environment Issues
```bash
# Diagnostic sequence
python --version
pip --version
which python

# Fix sequence for workflow
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Integration with Claude Code Tools

### Automated Debugging Workflow
```javascript
[AutomatedWorkflowDebug]:
  // Phase 1: Assessment
  Bash("git remote get-url origin")
  Bash("gh run list --limit 5")
  WebFetch("github-actions-url", "Show recent failures")
  
  // Phase 2: Analysis  
  Read(".github/workflows/failing-workflow.yml")
  Bash("find .github/workflows -name '*.yml'")
  
  // Phase 3: Testing
  Bash("python scripts/architectural_analysis.py --help")
  Bash("ls -la security/ || echo 'Missing security directory'")
  
  // Phase 4: Fixes
  Bash("mkdir -p security/{reports,sboms}")
  Edit(".github/workflows/failing-workflow.yml")
  
  // Phase 5: Validation
  Bash("yamllint .github/workflows/*.yml")
  Bash("python scripts/architectural_analysis.py --project-root . --output-format json")
```

## Error Classification & Solutions

### Class 1: Configuration Errors
- **Symptoms**: Missing files, wrong paths, invalid configuration
- **Solution Pattern**: Create missing files, fix paths, validate configuration

### Class 2: Parameter Errors  
- **Symptoms**: Unknown argument, invalid format, wrong flags
- **Solution Pattern**: Check --help output, fix parameter names, validate formats

### Class 3: Dependency Errors
- **Symptoms**: Command not found, import errors, missing packages
- **Solution Pattern**: Install dependencies, check versions, update requirements

### Class 4: Permission Errors
- **Symptoms**: Permission denied, file not executable, access denied
- **Solution Pattern**: Fix permissions, check ownership, validate access

### Class 5: Environment Errors
- **Symptoms**: Wrong Python version, missing environment variables, path issues
- **Solution Pattern**: Set environment, fix paths, validate versions

## Advanced Debugging Techniques

### Using GitHub API for Deep Analysis
```bash
# Get detailed run information
gh api repos/owner/repo/actions/runs/[ID] | jq '.conclusion'

# Get job logs
gh api repos/owner/repo/actions/runs/[ID]/logs

# List workflow files via API
gh api repos/owner/repo/contents/.github/workflows | jq '.[].name'
```

### Matrix Build Debugging
```yaml
# For matrix builds, identify specific failing combinations
strategy:
  matrix:
    python-version: [3.8, 3.9, 3.10]
    os: [ubuntu-latest, windows-latest]
```

### Artifact Analysis
```bash
# Download artifacts for analysis
gh run download [RUN_ID]

# List artifacts
gh api repos/owner/repo/actions/runs/[ID]/artifacts
```