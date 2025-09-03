#!/usr/bin/env python3
"""
CI/CD Pipeline Analyzer with MCP Integration
Comprehensive analysis of pipeline status, failures, and deployment readiness
"""

import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

logger = logging.getLogger(__name__)


class CIPipelineAnalyzer:
    """Analyzes CI/CD pipeline status and generates reports"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.analysis_results = {}
        
    def analyze_security_scans(self) -> Dict[str, Any]:
        """Analyze all security scan results"""
        security_files = [
            'security_scan_results.json',
            'final_security_validation.json', 
            'bandit_scan.json',
            'final_security_scan.json'
        ]
        
        security_analysis = {
            'total_files_analyzed': 0,
            'critical_issues': 0,
            'high_issues': 0,
            'medium_issues': 0,
            'low_issues': 0,
            'syntax_errors': [],
            'deployment_blockers': [],
            'security_score': 0
        }
        
        for file in security_files:
            file_path = self.project_root / file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Parse metrics from different scan formats
                    if 'metrics' in data and '_totals' in data['metrics']:
                        totals = data['metrics']['_totals']
                        security_analysis['high_issues'] += totals.get('SEVERITY.HIGH', 0)
                        security_analysis['medium_issues'] += totals.get('SEVERITY.MEDIUM', 0)
                        security_analysis['low_issues'] += totals.get('SEVERITY.LOW', 0)
                        security_analysis['total_files_analyzed'] += totals.get('loc', 0)
                    
                    # Check for syntax errors
                    if 'errors' in data:
                        for error in data['errors']:
                            security_analysis['syntax_errors'].append({
                                'file': error.get('filename', 'unknown'),
                                'reason': error.get('reason', 'unknown error')
                            })
                    
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse {file}")
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        
        # Calculate security score (0-100)
        total_issues = security_analysis['high_issues'] + security_analysis['medium_issues'] + security_analysis['low_issues']
        if total_issues == 0:
            security_analysis['security_score'] = 100
        else:
            # Penalize high severity issues more
            penalty = (security_analysis['high_issues'] * 10) + (security_analysis['medium_issues'] * 3) + (security_analysis['low_issues'] * 1)
            security_analysis['security_score'] = max(0, 100 - penalty)
        
        # Determine deployment blockers
        if security_analysis['high_issues'] > 0:
            security_analysis['deployment_blockers'].append(f"HIGH_SEVERITY_SECURITY: {security_analysis['high_issues']} critical security issues")
        
        if len(security_analysis['syntax_errors']) > 0:
            security_analysis['deployment_blockers'].append(f"SYNTAX_ERRORS: {len(security_analysis['syntax_errors'])} files with syntax errors")
        
        return security_analysis
    
    def analyze_test_status(self) -> Dict[str, Any]:
        """Analyze test execution status"""
        test_analysis = {
            'pytest_config_status': 'unknown',
            'modified_test_files': 0,
            'test_cache_files': 0,
            'critical_test_issues': []
        }
        
        # Check pytest configuration
        pytest_ini = self.project_root / 'tests' / 'pytest.ini'
        if pytest_ini.exists():
            try:
                with open(pytest_ini, 'r') as f:
                    content = f.read()
                    if ']' in content and '[coverage:run]' in content:
                        test_analysis['pytest_config_status'] = 'syntax_error_fixed'
                    else:
                        test_analysis['pytest_config_status'] = 'needs_review'
            except Exception as e:
                test_analysis['pytest_config_status'] = f'error: {e}'
                test_analysis['critical_test_issues'].append('pytest.ini configuration error')
        
        # Count test-related files
        try:
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        if 'test' in line.lower():
                            test_analysis['modified_test_files'] += 1
                        if '__pycache__' in line or '.pyc' in line:
                            test_analysis['test_cache_files'] += 1
        except Exception as e:
            print(f"Error checking git status: {e}")
        
        return test_analysis
    
    def analyze_workflow_status(self) -> Dict[str, Any]:
        """Analyze GitHub workflow configurations"""
        workflow_analysis = {
            'total_workflows': 0,
            'unified_quality_pipeline': False,
            'main_ci_pipeline': False,
            'unified_linting': False,
            'security_workflows': 0,
            'workflow_issues': []
        }
        
        workflows_dir = self.project_root / '.github' / 'workflows'
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob('*.yml'))
            workflow_analysis['total_workflows'] = len(workflow_files)
            
            for workflow_file in workflow_files:
                filename = workflow_file.name
                
                if 'unified-quality-pipeline' in filename:
                    workflow_analysis['unified_quality_pipeline'] = True
                
                if 'main-ci' in filename:
                    workflow_analysis['main_ci_pipeline'] = True
                
                if 'unified-linting' in filename:
                    workflow_analysis['unified_linting'] = True
                
                if 'security' in filename:
                    workflow_analysis['security_workflows'] += 1
                
                # Check for potential workflow issues
                try:
                    with open(workflow_file, 'r') as f:
                        content = f.read()
                        if 'sudo' in content and 'apt' in content:
                            workflow_analysis['workflow_issues'].append(f"{filename}: Contains sudo/apt commands that may fail on Windows runners")
                except Exception:
                    logger.exception("Workflow analysis failed")
        
        return workflow_analysis
    
    def generate_deployment_readiness_assessment(self) -> Dict[str, Any]:
        """Generate comprehensive deployment readiness assessment"""
        security_analysis = self.analyze_security_scans()
        test_analysis = self.analyze_test_status()
        workflow_analysis = self.analyze_workflow_status()
        
        # Calculate overall readiness score
        security_weight = 0.4
        test_weight = 0.3
        workflow_weight = 0.3
        
        security_score = security_analysis['security_score']
        test_score = 80 if test_analysis['pytest_config_status'] == 'syntax_error_fixed' else 60
        workflow_score = 90 if workflow_analysis['unified_quality_pipeline'] else 70
        
        overall_score = (security_score * security_weight) + (test_score * test_weight) + (workflow_score * workflow_weight)
        
        # Determine deployment status
        if security_analysis['deployment_blockers']:
            deployment_status = 'BLOCKED'
            deployment_reason = f"Security issues: {', '.join(security_analysis['deployment_blockers'])}"
        elif overall_score >= 80:
            deployment_status = 'READY'
            deployment_reason = 'All quality gates passed'
        elif overall_score >= 60:
            deployment_status = 'WARNING'
            deployment_reason = 'Minor issues present, review recommended'
        else:
            deployment_status = 'NOT_READY'
            deployment_reason = 'Multiple quality gate failures'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_score': round(overall_score, 2),
            'deployment_status': deployment_status,
            'deployment_reason': deployment_reason,
            'security_analysis': security_analysis,
            'test_analysis': test_analysis,
            'workflow_analysis': workflow_analysis,
            'recommendations': self.generate_recommendations(security_analysis, test_analysis, workflow_analysis)
        }
    
    def generate_recommendations(self, security_analysis: Dict, test_analysis: Dict, workflow_analysis: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if security_analysis['high_issues'] > 0:
            recommendations.append(f"CRITICAL: Address {security_analysis['high_issues']} high-severity security issues immediately")
        
        if security_analysis['syntax_errors']:
            recommendations.append(f"Fix {len(security_analysis['syntax_errors'])} syntax errors in security-related files")
        
        if test_analysis['critical_test_issues']:
            recommendations.append("Resolve pytest configuration issues to enable proper testing")
        
        if not workflow_analysis['unified_quality_pipeline']:
            recommendations.append("Enable unified quality pipeline for comprehensive CI/CD")
        
        if security_analysis['security_score'] < 70:
            recommendations.append("Improve security posture - current score below acceptable threshold")
        
        if test_analysis['modified_test_files'] > 10:
            recommendations.append("Review and commit outstanding test file changes")
        
        return recommendations
    
    def generate_github_issue_content(self, assessment: Dict[str, Any]) -> str:
        """Generate GitHub issue content for critical failures"""
        if assessment['deployment_status'] not in ['BLOCKED', 'NOT_READY']:
            return None
        
        issue_content = f"""# CI/CD Pipeline Critical Issues

## Summary
**Deployment Status:** {assessment['deployment_status']}  
**Overall Quality Score:** {assessment['overall_score']}/100  
**Assessment Date:** {assessment['timestamp']}

## Critical Issues

### Security Analysis
- **High Severity Issues:** {assessment['security_analysis']['high_issues']}
- **Security Score:** {assessment['security_analysis']['security_score']}/100
- **Syntax Errors:** {len(assessment['security_analysis']['syntax_errors'])}

### Deployment Blockers
"""
        
        for blocker in assessment['security_analysis']['deployment_blockers']:
            issue_content += f"- {blocker}\n"
        
        issue_content += f"""
### Test Status
- **Pytest Config:** {assessment['test_analysis']['pytest_config_status']}
- **Modified Test Files:** {assessment['test_analysis']['modified_test_files']}

## Recommendations
"""
        
        for rec in assessment['recommendations']:
            issue_content += f"- [ ] {rec}\n"
        
        issue_content += f"""
## Next Steps
1. Address all HIGH severity security issues
2. Fix syntax errors in security scan files  
3. Resolve pytest configuration issues
4. Re-run unified quality pipeline
5. Verify all quality gates pass

---
*Generated by AIVillage CI/CD Pipeline Analyzer*
*Workflow Integration: GitHub MCP + Sequential Thinking MCP*
"""
        
        return issue_content

def main():
    """Main analysis execution"""
    analyzer = CIPipelineAnalyzer()
    
    print("AIVillage CI/CD Pipeline Analysis")
    print("=" * 50)
    
    # Generate comprehensive assessment
    assessment = analyzer.generate_deployment_readiness_assessment()
    
    # Save results
    results_file = analyzer.project_root / 'ci_pipeline_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(assessment, f, indent=2)
    
    # Print summary
    print(f"Overall Quality Score: {assessment['overall_score']}/100")
    print(f"Deployment Status: {assessment['deployment_status']}")
    print(f"Reason: {assessment['deployment_reason']}")
    print()
    
    print("Security Analysis:")
    sec = assessment['security_analysis']
    print(f"  - Security Score: {sec['security_score']}/100")
    print(f"  - High Issues: {sec['high_issues']}")
    print(f"  - Syntax Errors: {len(sec['syntax_errors'])}")
    print(f"  - Deployment Blockers: {len(sec['deployment_blockers'])}")
    print()
    
    print("Workflow Analysis:")
    wf = assessment['workflow_analysis']
    print(f"  - Total Workflows: {wf['total_workflows']}")
    print(f"  - Unified Quality Pipeline: {wf['unified_quality_pipeline']}")
    print(f"  - Security Workflows: {wf['security_workflows']}")
    print()
    
    if assessment['recommendations']:
        print("Recommendations:")
        for rec in assessment['recommendations']:
            print(f"  - {rec}")
        print()
    
    # Generate GitHub issue if needed
    if assessment['deployment_status'] in ['BLOCKED', 'NOT_READY']:
        issue_content = analyzer.generate_github_issue_content(assessment)
        if issue_content:
            issue_file = analyzer.project_root / 'github_issue_content.md'
            with open(issue_file, 'w') as f:
                f.write(issue_content)
            print(f"GitHub issue content generated: {issue_file}")
    
    print(f"Full analysis saved to: {results_file}")
    
    # Exit with appropriate code
    if assessment['deployment_status'] == 'BLOCKED':
        sys.exit(1)
    elif assessment['deployment_status'] == 'NOT_READY':
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()