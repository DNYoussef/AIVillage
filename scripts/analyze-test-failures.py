#!/usr/bin/env python3
"""
Test Failure Analysis Script
Analyzes GitHub workflow failures and categorizes them for auto-fixing
"""

import os
import sys
import json
import re
import sqlite3
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestFailureAnalyzer:
    """Analyzes test failures from GitHub workflows"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.claude_dir = self.project_root / '.claude'
        self.test_failures_dir = self.claude_dir / 'test-failures'
        self.failures_db = self.test_failures_dir / 'failures.db'
        
        # Ensure directories exist
        self.test_failures_dir.mkdir(parents=True, exist_ok=True)
        
        # GitHub API setup
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo = self._get_repo_info()
        
        # Initialize database
        self._init_db()
    
    def _get_repo_info(self) -> Optional[str]:
        """Get repository info from git remote"""
        try:
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                # Extract owner/repo from GitHub URL
                if 'github.com' in url:
                    parts = url.split('/')
                    if len(parts) >= 2:
                        owner = parts[-2]
                        repo = parts[-1].replace('.git', '')
                        return f"{owner}/{repo}"
            return None
        except Exception:
            return None
    
    def _init_db(self):
        """Initialize failures database"""
        with sqlite3.connect(self.failures_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    pr_number INTEGER,
                    commit_hash TEXT,
                    test_name TEXT NOT NULL,
                    failure_message TEXT NOT NULL,
                    stack_trace TEXT,
                    file_path TEXT,
                    failure_type TEXT,
                    severity TEXT DEFAULT 'medium',
                    status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved_at DATETIME,
                    fix_attempt_count INTEGER DEFAULT 0
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_failures_run ON test_failures(run_id)
            ''')
    
    def analyze_workflow_run(self, run_id: str, pr_number: Optional[int] = None, 
                           commit_sha: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a specific workflow run for failures"""
        failures = []
        
        try:
            # Get workflow run details
            run_details = self._get_workflow_run_details(run_id)
            if not run_details:
                return {'failures': [], 'error': 'Could not fetch workflow run details'}
            
            # Get job details and logs
            jobs = self._get_workflow_jobs(run_id)
            
            for job in jobs:
                if job.get('conclusion') == 'failure':
                    job_failures = self._analyze_job_failure(job, run_id)
                    failures.extend(job_failures)
            
            # Store failures in database
            for failure in failures:
                self._store_failure(
                    run_id=run_id,
                    pr_number=pr_number,
                    commit_hash=commit_sha,
                    **failure
                )
            
            # Categorize failures
            categorized = self._categorize_failures(failures)
            
            # Save current failures for processing
            current_failures = {
                'run_id': run_id,
                'pr_number': pr_number,
                'commit_sha': commit_sha,
                'timestamp': datetime.now().isoformat(),
                'total_failures': len(failures),
                'categorized': categorized,
                'raw_failures': failures
            }
            
            current_file = self.test_failures_dir / 'current-failures.json'
            with open(current_file, 'w') as f:
                json.dump(current_failures, f, indent=2)
            
            print(f"✅ Analyzed {len(failures)} test failures")
            return current_failures
            
        except Exception as e:
            error_msg = f"❌ Analysis failed: {e}"
            print(error_msg)
            return {'failures': [], 'error': error_msg}
    
    def _get_workflow_run_details(self, run_id: str) -> Optional[Dict]:
        """Get workflow run details from GitHub API"""
        if not self.github_token or not self.repo:
            return None
        
        url = f"https://api.github.com/repos/{self.repo}/actions/runs/{run_id}"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching workflow run: {e}")
        
        return None
    
    def _get_workflow_jobs(self, run_id: str) -> List[Dict]:
        """Get jobs for a workflow run"""
        if not self.github_token or not self.repo:
            return []
        
        url = f"https://api.github.com/repos/{self.repo}/actions/runs/{run_id}/jobs"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json().get('jobs', [])
        except Exception as e:
            print(f"Error fetching jobs: {e}")
        
        return []
    
    def _analyze_job_failure(self, job: Dict, run_id: str) -> List[Dict]:
        """Analyze a specific job failure"""
        failures = []
        
        # Get job logs
        logs = self._get_job_logs(job['id'])
        if not logs:
            return failures
        
        # Parse logs for test failures
        parsed_failures = self._parse_test_logs(logs, job['name'])
        
        for failure in parsed_failures:
            failure_data = {
                'test_name': failure.get('test_name', job['name']),
                'failure_message': failure.get('message', 'Unknown failure'),
                'stack_trace': failure.get('stack_trace', ''),
                'file_path': failure.get('file_path', ''),
                'failure_type': self._classify_failure_type(failure),
                'severity': self._determine_severity(failure)
            }
            failures.append(failure_data)
        
        # If no specific test failures found, create a general job failure
        if not failures:
            failures.append({
                'test_name': job['name'],
                'failure_message': f"Job failed: {job.get('conclusion', 'Unknown')}",
                'stack_trace': logs[:2000],  # First 2000 chars of logs
                'file_path': '',
                'failure_type': 'build_failure',
                'severity': 'high'
            })
        
        return failures
    
    def _get_job_logs(self, job_id: str) -> Optional[str]:
        """Get logs for a specific job"""
        if not self.github_token or not self.repo:
            return None
        
        url = f"https://api.github.com/repos/{self.repo}/actions/jobs/{job_id}/logs"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Error fetching logs: {e}")
        
        return None
    
    def _parse_test_logs(self, logs: str, job_name: str) -> List[Dict]:
        """Parse test logs to extract specific failures"""
        failures = []
        
        # Python/pytest patterns
        pytest_patterns = [
            r'FAILED\s+(.+?)::\s*(.+?)\s*-\s*(.+?)(?=\n|$)',
            r'ERROR\s+(.+?)::\s*(.+?)\s*-\s*(.+?)(?=\n|$)',
            r'=+\s*FAILURES\s*=+(.*?)=+\s*(?:ERRORS|short test summary)',
            r'AssertionError:\s*(.+?)(?=\n|$)'
        ]
        
        # JavaScript/Node.js patterns
        js_patterns = [
            r'✗\s*(.+?)(?=\n|$)',
            r'Error:\s*(.+?)(?=\n|$)',
            r'Test failed:\s*(.+?)(?=\n|$)',
            r'AssertionError.*?:\s*(.+?)(?=\n|$)'
        ]
        
        # General error patterns
        general_patterns = [
            r'ERROR:\s*(.+?)(?=\n|$)',
            r'FAIL:\s*(.+?)(?=\n|$)',
            r'Exception:\s*(.+?)(?=\n|$)'
        ]
        
        all_patterns = pytest_patterns + js_patterns + general_patterns
        
        for pattern in all_patterns:
            matches = re.finditer(pattern, logs, re.MULTILINE | re.DOTALL)
            for match in matches:
                groups = match.groups()
                if groups:
                    failure = {
                        'test_name': groups[1] if len(groups) > 1 else groups[0],
                        'message': groups[-1] if len(groups) > 1 else groups[0],
                        'file_path': groups[0] if len(groups) > 1 else '',
                        'stack_trace': self._extract_stack_trace(logs, match.start())
                    }
                    failures.append(failure)
        
        return failures
    
    def _extract_stack_trace(self, logs: str, failure_pos: int) -> str:
        """Extract stack trace from logs starting at failure position"""
        lines = logs[failure_pos:].split('\n')
        stack_trace = []
        
        for line in lines[:20]:  # Look at next 20 lines
            if any(keyword in line.lower() for keyword in ['traceback', 'error', 'fail', 'exception']):
                stack_trace.append(line)
            elif len(stack_trace) > 0 and line.strip() == '':
                break
        
        return '\n'.join(stack_trace[:10])  # Limit to 10 lines
    
    def _classify_failure_type(self, failure: Dict) -> str:
        """Classify the type of failure"""
        message = failure.get('message', '').lower()
        stack_trace = failure.get('stack_trace', '').lower()
        combined = f"{message} {stack_trace}"
        
        # Syntax errors
        if any(keyword in combined for keyword in ['syntaxerror', 'indentationerror', 'invalid syntax']):
            return 'syntax'
        
        # Import/dependency errors
        elif any(keyword in combined for keyword in ['modulenotfounderror', 'importerror', 'cannot resolve']):
            return 'dependency'
        
        # Assertion/logic errors
        elif any(keyword in combined for keyword in ['assertionerror', 'expected', 'actual']):
            return 'logic'
        
        # Performance issues
        elif any(keyword in combined for keyword in ['timeout', 'slow', 'performance']):
            return 'performance'
        
        # Integration issues
        elif any(keyword in combined for keyword in ['connection', 'network', 'database', 'api']):
            return 'integration'
        
        return 'unknown'
    
    def _determine_severity(self, failure: Dict) -> str:
        """Determine failure severity"""
        message = failure.get('message', '').lower()
        
        # High severity
        if any(keyword in message for keyword in ['critical', 'fatal', 'security', 'crash']):
            return 'high'
        
        # Low severity
        elif any(keyword in message for keyword in ['warning', 'deprecated', 'style']):
            return 'low'
        
        return 'medium'
    
    def _categorize_failures(self, failures: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize failures by type"""
        categorized = {
            'syntax': [],
            'logic': [],
            'integration': [],
            'performance': [],
            'dependency': [],
            'unknown': []
        }
        
        for failure in failures:
            failure_type = failure.get('failure_type', 'unknown')
            categorized[failure_type].append(failure)
        
        return categorized
    
    def _store_failure(self, run_id: str, pr_number: Optional[int], 
                      commit_hash: Optional[str], **failure_data):
        """Store failure in database"""
        with sqlite3.connect(self.failures_db) as conn:
            conn.execute('''
                INSERT INTO test_failures 
                (run_id, pr_number, commit_hash, test_name, failure_message, 
                 stack_trace, file_path, failure_type, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                pr_number,
                commit_hash,
                failure_data.get('test_name', ''),
                failure_data.get('failure_message', ''),
                failure_data.get('stack_trace', ''),
                failure_data.get('file_path', ''),
                failure_data.get('failure_type', 'unknown'),
                failure_data.get('severity', 'medium')
            ))


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze test failures from GitHub workflows')
    parser.add_argument('--run-id', required=True, help='GitHub workflow run ID')
    parser.add_argument('--pr-number', type=int, help='Pull request number')
    parser.add_argument('--commit-sha', help='Commit SHA')
    
    args = parser.parse_args()
    
    analyzer = TestFailureAnalyzer()
    result = analyzer.analyze_workflow_run(
        run_id=args.run_id,
        pr_number=args.pr_number,
        commit_sha=args.commit_sha
    )
    
    if result.get('error'):
        print(f"Analysis failed: {result['error']}")
        sys.exit(1)
    
    print(f"✅ Analysis complete: {result['total_failures']} failures found")
    
    # Print summary
    for failure_type, failures in result['categorized'].items():
        if failures:
            print(f"  {failure_type}: {len(failures)} failures")


if __name__ == '__main__':
    main()