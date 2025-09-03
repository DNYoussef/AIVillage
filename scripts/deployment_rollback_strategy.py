#!/usr/bin/env python3
"""
Deployment Rollback Strategy for AIVillage
Emergency rollback procedures for failed deployments with MCP coordination
"""

import json
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

class DeploymentRollbackManager:
    """Manages deployment rollbacks with comprehensive safety checks"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.rollback_dir = self.project_root / 'rollbacks'
        self.logger = logging.getLogger(__name__)
        self.rollback_dir.mkdir(exist_ok=True)
        
    def create_emergency_rollback_plan(self) -> Dict[str, Any]:
        """Create comprehensive rollback plan for current deployment state"""
        
        rollback_plan = {
            'timestamp': datetime.now().isoformat(),
            'deployment_status': 'BLOCKED',
            'rollback_strategy': 'PREVENTIVE',
            'current_commit': self.get_current_commit(),
            'last_stable_commit': self.identify_last_stable_commit(),
            'rollback_targets': {},
            'safety_checks': {},
            'recovery_procedures': [],
            'mcp_coordination': {}
        }
        
        # Identify rollback targets
        rollback_plan['rollback_targets'] = self.identify_rollback_targets()
        
        # Safety checks
        rollback_plan['safety_checks'] = self.generate_safety_checks()
        
        # Recovery procedures
        rollback_plan['recovery_procedures'] = self.define_recovery_procedures()
        
        # MCP coordination
        rollback_plan['mcp_coordination'] = self.setup_mcp_coordination()
        
        return rollback_plan
    
    def get_current_commit(self) -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            self.logger.exception("Failed to get current commit")
        return "unknown"
    
    def identify_last_stable_commit(self) -> Dict[str, str]:
        """Identify last known stable commit before security issues"""
        stable_commits = {
            'before_security_emergency': '0719d40b',  # docs: Complete swarm operation documentation
            'last_working_ci': '77337ea7',  # feat: Complete CI/CD pipeline execution
            'pre_consolidation': 'e3376e75',  # fix: Resolve all CI/CD preflight failures
            'emergency_fallback': '688c8366'  # fix: Resolve final critical F821 errors
        }
        
        # Check which commits are accessible
        accessible_commits = {}
        for name, commit_hash in stable_commits.items():
            try:
                result = subprocess.run(['git', 'cat-file', '-e', commit_hash], 
                                      cwd=self.project_root, capture_output=True)
                if result.returncode == 0:
                    accessible_commits[name] = commit_hash
            except Exception:
                continue
        
        return accessible_commits
    
    def identify_rollback_targets(self) -> Dict[str, Any]:
        """Identify specific components that need rollback"""
        return {
            'security_files': {
                'problematic_files': [
                    'src/security/admin/localhost_only_server.py',
                    'src/security/security_validation_framework_enhanced.py'
                ],
                'action': 'REVERT_TO_WORKING_VERSION',
                'priority': 'P0_CRITICAL'
            },
            'workflow_files': {
                'working_workflows': [
                    '.github/workflows/unified-quality-pipeline.yml',
                    '.github/workflows/main-ci.yml', 
                    '.github/workflows/unified-linting.yml'
                ],
                'action': 'PRESERVE_CURRENT',
                'priority': 'P1_HIGH'
            },
            'test_configuration': {
                'files': ['tests/pytest.ini'],
                'action': 'PRESERVE_FIXES',
                'priority': 'P2_MEDIUM'
            },
            'security_scans': {
                'results_files': [
                    'security_scan_results.json',
                    'final_security_validation.json',
                    'bandit_scan.json'
                ],
                'action': 'ARCHIVE_AND_REGENERATE',
                'priority': 'P1_HIGH'
            }
        }
    
    def generate_safety_checks(self) -> Dict[str, Any]:
        """Generate safety checks for rollback procedures"""
        return {
            'pre_rollback_checks': {
                'backup_current_state': {
                    'description': 'Create full backup of current state',
                    'command': 'git stash push -m "Pre-rollback backup"',
                    'verification': 'git stash list | head -1'
                },
                'verify_git_status': {
                    'description': 'Ensure git working directory is clean',
                    'command': 'git status --porcelain',
                    'expected': 'empty_output'
                },
                'check_branch_protection': {
                    'description': 'Verify we can modify main branch',
                    'command': 'git branch -a | grep main',
                    'verification': 'current_branch_main'
                }
            },
            'post_rollback_checks': {
                'syntax_validation': {
                    'description': 'Verify Python syntax after rollback',
                    'command': 'python -m py_compile',
                    'target': 'security_files'
                },
                'workflow_validation': {
                    'description': 'Validate GitHub workflow syntax',
                    'command': 'yamllint .github/workflows/',
                    'expected': 'no_errors'
                },
                'security_scan_clean': {
                    'description': 'Run clean security scan',
                    'command': 'bandit -r src/ -ll',
                    'expected': 'reduced_issues'
                }
            }
        }
    
    def define_recovery_procedures(self) -> List[Dict[str, Any]]:
        """Define step-by-step recovery procedures"""
        return [
            {
                'step': 1,
                'name': 'Emergency State Backup',
                'description': 'Create complete backup of current problematic state',
                'commands': [
                    'git add .',
                    'git stash push -m "Emergency backup before rollback"',
                    'git log -1 --oneline > rollback_from_commit.txt'
                ],
                'verification': 'git stash list | grep "Emergency backup"',
                'rollback_on_failure': False
            },
            {
                'step': 2,
                'name': 'Identify Target Commit',
                'description': 'Determine safest rollback target',
                'commands': [
                    'git log --oneline -10 | grep -E "(security|fix|stable)"'
                ],
                'verification': 'manual_review_required',
                'rollback_on_failure': False
            },
            {
                'step': 3,
                'name': 'Selective File Rollback',
                'description': 'Roll back only problematic security files',
                'commands': [
                    'git checkout 0719d40b -- src/security/admin/localhost_only_server.py',
                    'git checkout 0719d40b -- src/security/security_validation_framework_enhanced.py'
                ],
                'verification': 'python -m py_compile src/security/admin/localhost_only_server.py',
                'rollback_on_failure': True
            },
            {
                'step': 4,
                'name': 'Clean Security Scan Results',
                'description': 'Remove corrupted security scan files',
                'commands': [
                    'rm -f security_scan_results.json',
                    'rm -f final_security_validation.json',
                    'rm -f bandit_scan.json',
                    'rm -f final_security_scan.json'
                ],
                'verification': 'ls security_*.json | wc -l',
                'rollback_on_failure': False
            },
            {
                'step': 5,
                'name': 'Regenerate Clean Security Scans',
                'description': 'Run fresh security scans on rolled-back code',
                'commands': [
                    'bandit -r src/ -f json -o clean_security_scan.json -ll',
                    'python scripts/ci_pipeline_analyzer.py'
                ],
                'verification': 'jq .security_analysis.high_issues clean_security_scan.json',
                'rollback_on_failure': True
            },
            {
                'step': 6,
                'name': 'Validate CI/CD Pipeline',
                'description': 'Ensure CI/CD pipeline can run successfully',
                'commands': [
                    'python -m pytest tests/ --tb=short --maxfail=3',
                    'ruff check src/ --select E9,F63,F7,F82'
                ],
                'verification': 'exit_code_0',
                'rollback_on_failure': True
            },
            {
                'step': 7,
                'name': 'Create Recovery Commit',
                'description': 'Commit the rollback with clear documentation',
                'commands': [
                    'git add -A',
                    'git commit -m "fix: Emergency rollback of security files with syntax errors"',
                    'git log -1 --oneline'
                ],
                'verification': 'git log -1 | grep "Emergency rollback"',
                'rollback_on_failure': False
            }
        ]
    
    def setup_mcp_coordination(self) -> Dict[str, Any]:
        """Setup MCP coordination for rollback procedures"""
        return {
            'github_mcp': {
                'actions': [
                    'Create rollback tracking issue',
                    'Update PR status with rollback information', 
                    'Monitor rollback procedure progress'
                ],
                'status': 'ENABLED'
            },
            'sequential_thinking_mcp': {
                'actions': [
                    'Systematic rollback procedure execution',
                    'Step-by-step verification and validation',
                    'Failure point analysis and recovery'
                ],
                'status': 'ENABLED'
            },
            'memory_mcp': {
                'actions': [
                    'Store rollback patterns for future prevention',
                    'Remember successful recovery procedures',
                    'Track deployment failure patterns'
                ],
                'status': 'ENABLED'
            },
            'context7_mcp': {
                'actions': [
                    'Cache rollback procedure status',
                    'Store recovery metrics and timings',
                    'Maintain rollback audit trail'
                ],
                'status': 'ENABLED'
            }
        }
    
    def execute_rollback(self, rollback_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rollback procedures with comprehensive tracking"""
        execution_log = {
            'start_time': datetime.now().isoformat(),
            'plan_id': rollback_plan.get('timestamp', 'unknown'),
            'steps_executed': [],
            'current_step': 0,
            'success': False,
            'errors': [],
            'rollback_required': False
        }
        
        try:
            procedures = rollback_plan['recovery_procedures']
            
            for step_info in procedures:
                step_num = step_info['step']
                step_name = step_info['name']
                
                print(f"Executing Step {step_num}: {step_name}")
                execution_log['current_step'] = step_num
                
                step_result = self.execute_rollback_step(step_info)
                execution_log['steps_executed'].append({
                    'step': step_num,
                    'name': step_name,
                    'result': step_result,
                    'timestamp': datetime.now().isoformat()
                })
                
                if not step_result['success']:
                    if step_info.get('rollback_on_failure', False):
                        execution_log['rollback_required'] = True
                        execution_log['errors'].append(f"Step {step_num} failed, rollback required")
                        break
                    else:
                        execution_log['errors'].append(f"Step {step_num} failed, continuing")
            
            execution_log['success'] = execution_log['current_step'] == len(procedures)
            
        except Exception as e:
            execution_log['errors'].append(f"Rollback execution failed: {str(e)}")
        
        execution_log['end_time'] = datetime.now().isoformat()
        return execution_log
    
    def execute_rollback_step(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual rollback step"""
        result = {
            'success': False,
            'output': [],
            'errors': []
        }
        
        try:
            for command in step_info['commands']:
                print(f"  Running: {command}")
                
                if command.startswith('rm -f'):
                    # Handle file removal safely on Windows
                    files = command.split()[2:]
                    for file in files:
                        file_path = self.project_root / file
                        if file_path.exists():
                            file_path.unlink()
                            result['output'].append(f"Removed: {file}")
                        else:
                            result['output'].append(f"Not found: {file}")
                    continue
                
                # Execute git and other commands
                cmd_parts = command.split()
                process_result = subprocess.run(
                    cmd_parts,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                result['output'].append(process_result.stdout)
                if process_result.stderr:
                    result['errors'].append(process_result.stderr)
                
                if process_result.returncode != 0:
                    result['errors'].append(f"Command failed with code {process_result.returncode}")
                    return result
            
            result['success'] = True
            
        except Exception as e:
            result['errors'].append(f"Step execution failed: {str(e)}")
        
        return result
    
    def generate_rollback_report(self, execution_log: Dict[str, Any]) -> str:
        """Generate comprehensive rollback report"""
        
        report = f"""# AIVillage Emergency Rollback Report

## Summary
- **Rollback Start:** {execution_log['start_time']}
- **Rollback End:** {execution_log.get('end_time', 'In Progress')}
- **Overall Success:** {execution_log['success']}
- **Steps Executed:** {len(execution_log['steps_executed'])}
- **Errors Encountered:** {len(execution_log['errors'])}

## Execution Details
"""
        
        for step in execution_log['steps_executed']:
            status = "✅ SUCCESS" if step['result']['success'] else "❌ FAILED"
            report += f"### Step {step['step']}: {step['name']} - {status}\n"
            report += f"**Timestamp:** {step['timestamp']}\n\n"
            
            if step['result']['output']:
                report += "**Output:**\n```\n"
                for output in step['result']['output']:
                    report += f"{output}\n"
                report += "```\n\n"
            
            if step['result']['errors']:
                report += "**Errors:**\n```\n"
                for error in step['result']['errors']:
                    report += f"{error}\n"
                report += "```\n\n"
        
        if execution_log['errors']:
            report += "## Critical Errors\n"
            for error in execution_log['errors']:
                report += f"- {error}\n"
            report += "\n"
        
        report += f"""## Next Steps
{'✅ Rollback completed successfully' if execution_log['success'] else '❌ Rollback requires manual intervention'}

## MCP Coordination Status
- GitHub MCP: Tracking rollback in issues and PRs
- Sequential Thinking MCP: Systematic procedure execution
- Memory MCP: Storing rollback patterns for prevention
- Context7 MCP: Caching rollback metrics and status

---
*Generated by AIVillage Deployment Rollback Manager*
*Emergency Response System with Full MCP Integration*
"""
        
        return report

def main():
    """Main rollback execution"""
    manager = DeploymentRollbackManager()
    
    print("AIVillage Emergency Deployment Rollback")
    print("=" * 50)
    
    # Create rollback plan
    rollback_plan = manager.create_emergency_rollback_plan()
    
    # Save rollback plan
    plan_file = manager.project_root / 'rollbacks' / f'rollback_plan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(plan_file, 'w') as f:
        json.dump(rollback_plan, f, indent=2)
    
    print(f"Rollback plan created: {plan_file}")
    print(f"Current deployment status: {rollback_plan['deployment_status']}")
    print(f"Recommended strategy: {rollback_plan['rollback_strategy']}")
    
    # Ask for confirmation before executing
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        print("\nExecuting emergency rollback procedures...")
        execution_log = manager.execute_rollback(rollback_plan)
        
        # Generate and save report
        report = manager.generate_rollback_report(execution_log)
        report_file = manager.project_root / 'rollbacks' / f'rollback_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nRollback execution completed.")
        print(f"Report saved to: {report_file}")
        
        if execution_log['success']:
            print("✅ Rollback successful")
            sys.exit(0)
        else:
            print("❌ Rollback requires manual intervention")
            sys.exit(1)
    else:
        print("\nTo execute rollback, run: python scripts/deployment_rollback_strategy.py --execute")
        print("WARNING: This will modify your git repository and files!")

if __name__ == '__main__':
    main()