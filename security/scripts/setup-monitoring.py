#!/usr/bin/env python3
"""
Dependency Monitoring Setup Script for AIVillage
Configures continuous dependency monitoring, alerting, and automated responses
across all ecosystems in the project.
"""

import os
import sys
import yaml
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DependencyMonitoringSetup:
    """Sets up comprehensive dependency monitoring for AIVillage"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.security_dir = self.project_root / "security"
        self.config_dir = self.security_dir / "configs"
        self.scripts_dir = self.security_dir / "scripts"

        # Ensure directories exist
        self.security_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        self.scripts_dir.mkdir(exist_ok=True)

    def setup_monitoring(self):
        """Set up complete dependency monitoring system"""
        logger.info("Setting up comprehensive dependency monitoring...")

        # Create monitoring configurations
        self._create_monitoring_configs()

        # Setup GitHub Actions workflows
        self._setup_github_workflows()

        # Create alerting configurations
        self._setup_alerting()

        # Setup dashboard configuration
        self._setup_dashboard()

        # Create maintenance scripts
        self._create_maintenance_scripts()

        # Setup pre-commit hooks
        self._setup_precommit_hooks()

        logger.info("Dependency monitoring setup complete!")

    def _create_monitoring_configs(self):
        """Create comprehensive monitoring configuration files"""
        logger.info("Creating monitoring configurations...")

        # Dependabot configuration
        dependabot_config = {
            "version": 2,
            "updates": [
                {
                    "package-ecosystem": "pip",
                    "directory": "/",
                    "schedule": {"interval": "daily"},
                    "reviewers": ["security-team"],
                    "assignees": ["security-team"],
                    "labels": ["dependencies", "security"],
                    "open-pull-requests-limit": 10,
                    "rebase-strategy": "auto",
                    "allow": [
                        {"dependency-type": "direct"},
                        {"dependency-type": "indirect", "update-type": "security"},
                    ],
                    "ignore": [{"dependency-name": "*", "update-type": "version-update:semver-major"}],
                },
                {
                    "package-ecosystem": "npm",
                    "directory": "/apps/web",
                    "schedule": {"interval": "daily"},
                    "reviewers": ["security-team"],
                    "assignees": ["security-team"],
                    "labels": ["dependencies", "security", "javascript"],
                    "open-pull-requests-limit": 10,
                },
                {
                    "package-ecosystem": "cargo",
                    "directory": "/",
                    "schedule": {"interval": "daily"},
                    "reviewers": ["security-team"],
                    "assignees": ["security-team"],
                    "labels": ["dependencies", "security", "rust"],
                    "open-pull-requests-limit": 10,
                },
                {
                    "package-ecosystem": "gomod",
                    "directory": "/integrations/clients/rust/scion-sidecar",
                    "schedule": {"interval": "daily"},
                    "reviewers": ["security-team"],
                    "assignees": ["security-team"],
                    "labels": ["dependencies", "security", "go"],
                    "open-pull-requests-limit": 10,
                },
                {
                    "package-ecosystem": "docker",
                    "directory": "/",
                    "schedule": {"interval": "weekly"},
                    "reviewers": ["devops-team"],
                    "labels": ["dependencies", "docker"],
                    "open-pull-requests-limit": 5,
                },
            ],
        }

        dependabot_file = self.project_root / ".github" / "dependabot.yml"
        dependabot_file.parent.mkdir(exist_ok=True)
        with open(dependabot_file, "w") as f:
            yaml.dump(dependabot_config, f, indent=2)

        # CodeQL configuration for dependency scanning
        codeql_config = {
            "name": "CodeQL Analysis",
            "queries": ["security-and-quality", "security-experimental"],
            "paths-ignore": ["**/node_modules/**", "**/target/**", "**/.venv/**", "**/build/**"],
            "disable-default-path-filters": False,
        }

        codeql_file = self.project_root / ".github" / "codeql" / "codeql-config.yml"
        codeql_file.parent.mkdir(exist_ok=True)
        with open(codeql_file, "w") as f:
            yaml.dump(codeql_config, f, indent=2)

    def _setup_github_workflows(self):
        """Create GitHub Actions workflows for monitoring"""
        logger.info("Setting up GitHub Actions workflows...")

        # Create dependency monitoring workflow
        monitoring_workflow = {
            "name": "Dependency Monitoring",
            "on": {
                "schedule": [{"cron": "0 */6 * * *"}],  # Every 6 hours
                "workflow_dispatch": {},
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main", "develop"]},
            },
            "env": {
                "SECURITY_ALERTS_WEBHOOK": "${{ secrets.SECURITY_ALERTS_WEBHOOK }}",
                "DASHBOARD_API_KEY": "${{ secrets.DASHBOARD_API_KEY }}",
            },
            "jobs": {
                "monitor-dependencies": {
                    "name": "Monitor Dependencies",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.12"},
                        },
                        {
                            "name": "Install monitoring tools",
                            "run": "pip install pip-audit safety bandit semgrep requests pyyaml",
                        },
                        {"name": "Run dependency monitoring", "run": "python security/scripts/monitor-dependencies.py"},
                        {"name": "Update security dashboard", "run": "python security/scripts/update-dashboard.py"},
                        {
                            "name": "Send alerts if needed",
                            "if": "failure()",
                            "run": "python security/scripts/send-alerts.py",
                        },
                    ],
                }
            },
        }

        workflows_dir = self.project_root / ".github" / "workflows"
        workflows_dir.mkdir(exist_ok=True)

        with open(workflows_dir / "dependency-monitoring.yml", "w") as f:
            yaml.dump(monitoring_workflow, f, indent=2, sort_keys=False)

    def _setup_alerting(self):
        """Setup alerting configuration"""
        logger.info("Setting up alerting system...")

        alerting_config = {
            "alerting": {
                "channels": [
                    {
                        "name": "slack",
                        "type": "webhook",
                        "url": "${SECURITY_SLACK_WEBHOOK}",
                        "severity_levels": ["critical", "high", "medium"],
                        "message_template": {
                            "text": "🚨 AIVillage Security Alert",
                            "blocks": [
                                {
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": "*Severity:* {severity}\n*Package:* {package}\n*Vulnerability:* {vulnerability_id}\n*Description:* {description}",
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "name": "github_issues",
                        "type": "github",
                        "severity_levels": ["critical", "high"],
                        "auto_assign": ["security-team"],
                        "labels": ["security", "vulnerability", "automated"],
                        "issue_template": {
                            "title": "🚨 {severity} Security Vulnerability: {package}",
                            "body": "## Security Alert\n\n**Package:** {package}\n**Version:** {version}\n**Vulnerability ID:** {vulnerability_id}\n**Severity:** {severity}\n**CVSS Score:** {cvss_score}\n\n## Description\n{description}\n\n## Recommendation\n{recommendation}\n\n## References\n{references}\n\n---\n*This issue was created automatically by the dependency monitoring system.*",
                        },
                    },
                    {
                        "name": "email",
                        "type": "smtp",
                        "severity_levels": ["critical"],
                        "recipients": ["security@aivillage.dev"],
                        "subject_template": "🚨 CRITICAL: Security Vulnerability in {package}",
                    },
                ],
                "rules": [
                    {
                        "name": "critical_vulnerabilities",
                        "condition": "severity == 'critical'",
                        "actions": ["slack", "github_issues", "email"],
                        "immediate": True,
                    },
                    {
                        "name": "high_vulnerabilities",
                        "condition": "severity == 'high'",
                        "actions": ["slack", "github_issues"],
                        "delay_minutes": 30,
                    },
                    {
                        "name": "medium_vulnerabilities",
                        "condition": "severity == 'medium'",
                        "actions": ["slack"],
                        "batch": True,
                        "batch_size": 10,
                        "batch_timeout_hours": 4,
                    },
                ],
            }
        }

        with open(self.config_dir / "alerting.yml", "w") as f:
            yaml.dump(alerting_config, f, indent=2)

    def _setup_dashboard(self):
        """Setup security dashboard configuration"""
        logger.info("Setting up security dashboard...")

        dashboard_config = {
            "dashboard": {
                "title": "AIVillage Dependency Security Dashboard",
                "refresh_interval": "5m",
                "data_sources": [
                    {
                        "name": "vulnerability_scans",
                        "type": "json_files",
                        "path": "security/reports/*.json",
                        "refresh": "1h",
                    },
                    {"name": "dependency_inventory", "type": "sbom", "path": "security/sboms/*.json", "refresh": "24h"},
                ],
                "widgets": [
                    {
                        "title": "Vulnerability Overview",
                        "type": "stat_cards",
                        "metrics": ["critical", "high", "medium", "low"],
                    },
                    {"title": "Vulnerability Trends", "type": "time_series", "timeframe": "30d"},
                    {
                        "title": "Top Vulnerable Packages",
                        "type": "table",
                        "columns": ["package", "ecosystem", "severity", "age"],
                    },
                    {"title": "Ecosystem Breakdown", "type": "pie_chart", "data": "vulnerabilities_by_ecosystem"},
                    {
                        "title": "Patching Performance",
                        "type": "bar_chart",
                        "metrics": ["mean_time_to_patch", "vulnerabilities_patched"],
                    },
                ],
            }
        }

        with open(self.config_dir / "dashboard.yml", "w") as f:
            yaml.dump(dashboard_config, f, indent=2)

    def _create_maintenance_scripts(self):
        """Create maintenance and monitoring scripts"""
        logger.info("Creating maintenance scripts...")

        # Monitor dependencies script
        monitor_script = '''#!/usr/bin/env python3
"""
Continuous dependency monitoring script
Runs vulnerability scans and checks for new security advisories
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    print(f"Starting dependency monitoring at {datetime.now()}")
    
    # Run security scans
    run_python_scans()
    run_nodejs_scans()
    run_rust_scans()
    run_go_scans()
    
    # Aggregate results
    aggregate_results()
    
    # Check for critical issues
    check_critical_issues()
    
    print("Monitoring complete")

def run_python_scans():
    """Run Python security scans"""
    try:
        subprocess.run(['pip-audit', '--desc', '--format=json', '--output=security/reports/pip-audit-monitor.json'], check=False)
        subprocess.run(['safety', 'check', '--json', '--output', 'security/reports/safety-monitor.json'], check=False)
    except Exception as e:
        print(f"Error running Python scans: {e}")

def run_nodejs_scans():
    """Run Node.js security scans"""
    try:
        subprocess.run(['npm', 'audit', '--json'], 
                      stdout=open('security/reports/npm-audit-monitor.json', 'w'), 
                      cwd='apps/web', check=False)
    except Exception as e:
        print(f"Error running Node.js scans: {e}")

def run_rust_scans():
    """Run Rust security scans"""
    try:
        subprocess.run(['cargo', 'audit', '--json'], 
                      stdout=open('security/reports/cargo-audit-monitor.json', 'w'), 
                      check=False)
    except Exception as e:
        print(f"Error running Rust scans: {e}")

def run_go_scans():
    """Run Go security scans"""
    try:
        subprocess.run(['govulncheck', '-json', './...'], 
                      stdout=open('security/reports/govulncheck-monitor.json', 'w'),
                      cwd='integrations/clients/rust/scion-sidecar', check=False)
    except Exception as e:
        print(f"Error running Go scans: {e}")

def aggregate_results():
    """Aggregate scan results"""
    try:
        subprocess.run(['python', 'security/scripts/aggregate-security-results.py'], check=False)
    except Exception as e:
        print(f"Error aggregating results: {e}")

def check_critical_issues():
    """Check for critical vulnerabilities and exit with error code if found"""
    try:
        with open('security/reports/security-summary.json', 'r') as f:
            summary = json.load(f)
        
        if summary.get('critical', 0) > 0:
            print(f"CRITICAL: {summary['critical']} critical vulnerabilities found!")
            sys.exit(1)
    except Exception as e:
        print(f"Error checking critical issues: {e}")

if __name__ == "__main__":
    main()
'''

        with open(self.scripts_dir / "monitor-dependencies.py", "w") as f:
            f.write(monitor_script)

        # Make script executable
        os.chmod(self.scripts_dir / "monitor-dependencies.py", 0o755)

        # Dashboard update script
        dashboard_script = '''#!/usr/bin/env python3
"""
Dashboard update script
Updates the security dashboard with latest scan results
"""

import json
import requests
import os
from pathlib import Path
from datetime import datetime

def main():
    """Update security dashboard"""
    api_key = os.getenv('DASHBOARD_API_KEY')
    if not api_key:
        print("DASHBOARD_API_KEY not set, skipping dashboard update")
        return
    
    # Load latest security report
    try:
        with open('security/reports/security-report.json', 'r') as f:
            report = json.load(f)
        
        # Update dashboard
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'vulnerabilities': report.get('summary', {}),
            'total_components': report.get('total_dependencies', 0),
            'ecosystems': report.get('ecosystems', {})
        }
        
        # Send to dashboard API (placeholder)
        print(f"Dashboard data prepared: {dashboard_data}")
        
    except Exception as e:
        print(f"Error updating dashboard: {e}")

if __name__ == "__main__":
    main()
'''

        with open(self.scripts_dir / "update-dashboard.py", "w") as f:
            f.write(dashboard_script)
        os.chmod(self.scripts_dir / "update-dashboard.py", 0o755)

        # Alert sender script
        alert_script = '''#!/usr/bin/env python3
"""
Alert sender script
Sends security alerts based on vulnerability findings
"""

import json
import requests
import os
from datetime import datetime

def main():
    """Send security alerts"""
    webhook_url = os.getenv('SECURITY_ALERTS_WEBHOOK')
    if not webhook_url:
        print("SECURITY_ALERTS_WEBHOOK not set, skipping alerts")
        return
    
    try:
        # Check for critical vulnerabilities
        with open('security/reports/critical-vulnerabilities.json', 'r') as f:
            critical_vulns = json.load(f)
        
        if critical_vulns.get('critical', 0) > 0:
            send_slack_alert(webhook_url, critical_vulns)
            
    except FileNotFoundError:
        print("No critical vulnerabilities file found")
    except Exception as e:
        print(f"Error sending alerts: {e}")

def send_slack_alert(webhook_url, vuln_data):
    """Send Slack alert"""
    message = {
        "text": "🚨 AIVillage Critical Security Alert",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Critical vulnerabilities detected:* {vuln_data['critical']}\\n*High severity:* {vuln_data.get('high', 0)}\\n*Scan time:* {datetime.now()}"
                }
            }
        ]
    }
    
    try:
        response = requests.post(webhook_url, json=message, timeout=10)
        response.raise_for_status()
        print("Alert sent successfully")
    except Exception as e:
        print(f"Failed to send alert: {e}")

if __name__ == "__main__":
    main()
'''

        with open(self.scripts_dir / "send-alerts.py", "w") as f:
            f.write(alert_script)
        os.chmod(self.scripts_dir / "send-alerts.py", 0o755)

    def _setup_precommit_hooks(self):
        """Setup pre-commit hooks for dependency security"""
        logger.info("Setting up pre-commit hooks...")

        precommit_config = {
            "repos": [
                {
                    "repo": "local",
                    "hooks": [
                        {
                            "id": "dependency-security-check",
                            "name": "Dependency Security Check",
                            "entry": "python security/scripts/monitor-dependencies.py",
                            "language": "system",
                            "files": r"(requirements.*\.txt|package\.json|Cargo\.toml|go\.mod)$",
                            "pass_filenames": False,
                        },
                        {
                            "id": "sbom-update",
                            "name": "Update SBOM",
                            "entry": "python security/scripts/generate-sbom.py",
                            "language": "system",
                            "files": r"(requirements.*\.txt|package\.json|Cargo\.toml|go\.mod)$",
                            "pass_filenames": False,
                        },
                    ],
                }
            ]
        }

        with open(self.project_root / ".pre-commit-config.yaml", "w") as f:
            yaml.dump(precommit_config, f, indent=2)


def main():
    """Main setup function"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."

    setup = DependencyMonitoringSetup(project_root)
    setup.setup_monitoring()

    print("✅ Dependency monitoring setup complete!")
    print("📊 Configured components:")
    print("  - GitHub Actions workflows")
    print("  - Dependabot automation")
    print("  - Security alerting")
    print("  - Dashboard integration")
    print("  - Pre-commit hooks")
    print("  - Monitoring scripts")
    print("")
    print("🔧 Next steps:")
    print("  1. Configure webhook URLs in GitHub secrets")
    print("  2. Set up Slack/email notifications")
    print("  3. Install and configure pre-commit hooks")
    print("  4. Test the monitoring pipeline")


if __name__ == "__main__":
    main()
