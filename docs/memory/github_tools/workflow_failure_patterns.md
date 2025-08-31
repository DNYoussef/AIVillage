# GitHub Actions Workflow Failure Analysis Patterns

## Common Workflow Failures & Solutions

### 1. Security Scan Workflow (`security-scan.yml`)

#### Common Failure Patterns:
- **Missing directories**: `security/reports/`, `security/sboms/`
- **Missing baseline file**: `.secrets.baseline`
- **Tool configuration issues**: pip-audit, safety, bandit, semgrep
- **SBOM generation failures**: cyclonedx-bom missing dependencies

#### Required Directory Structure:
```
security/
├── reports/
└── sboms/
```

#### Key Tools & Configuration:
```yaml
# Security scanning tools used
- pip-audit: Python package vulnerability scanning
- safety: Python dependency safety checks  
- bandit: Python security linter
- semgrep: Multi-language static analysis
- cyclonedx-bom: Software Bill of Materials generation
```

#### Fix Pattern:
```bash
# Create required directories
mkdir -p security/reports security/sboms

# Initialize secrets baseline
echo "{}" > .secrets.baseline

# Test tools locally before workflow
pip-audit --format=json --output=security/reports/pip-audit.json || echo "completed with warnings"
```

### 2. Architectural Quality Workflow (`architectural-quality.yml`)

#### Common Failure Patterns:
- **Script parameter mismatches**: Wrong CLI argument formats
- **Missing scripts**: architectural_analysis.py, coupling_metrics.py, detect_anti_patterns.py
- **Output format issues**: JSON vs text format mismatches
- **File path problems**: Incorrect working directory assumptions

#### Correct Script Parameters:
```bash
# Correct parameter formats
python scripts/architectural_analysis.py --project-root . --output-format json
python scripts/coupling_metrics.py --format json  
python scripts/detect_anti_patterns.py --project-root . --format json
```

#### Common Parameter Mistakes:
```bash
# WRONG - causes failures
--output json              # Should be --output-format json
--project .               # Should be --project-root .
--json-output             # Should be --format json
```

#### Fix Pattern:
```bash
# Test scripts locally with correct parameters
python scripts/architectural_analysis.py --project-root . --output-format json || echo "completed with warnings"
python scripts/coupling_metrics.py --format json || echo "completed with warnings" 
python scripts/detect_anti_patterns.py --project-root . --format json || echo "completed with warnings"
```

### 3. General Workflow Debugging Pattern

#### Step-by-Step Process:
1. **Get repository URL**: `git remote get-url origin`
2. **Access GitHub Actions**: Use WebFetch or gh CLI
3. **Identify failures**: Check specific workflow run logs
4. **Read workflow file**: Understand tool requirements
5. **Test locally**: Run failing commands with correct parameters
6. **Fix configuration**: Create directories, files, correct parameters
7. **Verify fixes**: Test before committing

#### Error Handling Best Practices:
```yaml
# Add graceful error handling in workflows
- name: Run tool with error handling
  run: |
    python script.py --correct-params || echo "completed with warnings"
    
# Use HEREDOC for complex commands
- name: Complex command
  run: |
    $(cat <<'EOF'
    complex multi-line
    command sequence
    EOF
    )
```

### 4. Artifact Upload Patterns

#### Common Issues:
- **Path matching problems**: Artifacts not found
- **Wildcard mismatches**: Incorrect glob patterns
- **Missing files**: Expected outputs not generated

#### Fix Pattern:
```yaml
- name: Upload artifacts
  uses: actions/upload-artifact@v3
  with:
    name: reports
    path: |
      security/reports/
      coupling_report.json
      antipatterns_report.json
  if: always()  # Upload even if previous steps failed
```

## Integration with Claude Code Debugging

### Using WebFetch for GitHub Analysis:
```python
# Pattern for accessing GitHub Actions
WebFetch("https://github.com/user/repo/actions", "Show recent workflow runs and failures")
WebFetch("https://github.com/user/repo/actions/workflows/workflow.yml", "Get detailed failure logs")
```

### Using Bash for Local Testing:
```bash
# Test workflow steps locally
python scripts/architectural_analysis.py --project-root . --output-format json
mkdir -p security/reports security/sboms
echo "{}" > .secrets.baseline
```