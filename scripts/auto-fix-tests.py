#!/usr/bin/env python3
"""
Automated Test Fix Script
Uses Claude to analyze and fix failing tests automatically
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import sqlite3
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class AutoTestFixer:
    """Automatically fixes test failures using Claude"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.claude_dir = self.project_root / '.claude'
        self.test_failures_dir = self.claude_dir / 'test-failures'
        self.failures_db = self.test_failures_dir / 'failures.db'
        
        # Claude API setup
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        if not self.claude_api_key:
            raise ValueError("CLAUDE_API_KEY environment variable required")
    
    def fix_failures(self, failure_type: str, failure_data: str, 
                    pr_number: Optional[int] = None, auto_commit: bool = False) -> Dict[str, Any]:
        """Fix failures of a specific type"""
        try:
            # Parse failure data
            failures = json.loads(failure_data) if isinstance(failure_data, str) else failure_data
            
            # Filter failures by type
            type_failures = self._filter_failures_by_type(failures, failure_type)
            
            if not type_failures:
                return {'status': 'no_failures', 'message': f'No {failure_type} failures found'}
            
            print(f"🔧 Fixing {len(type_failures)} {failure_type} failures...")
            
            # Process each failure
            fix_results = []
            for failure in type_failures:
                result = self._fix_single_failure(failure, failure_type)
                fix_results.append(result)
                
                if result.get('fixed'):
                    print(f"✅ Fixed: {failure.get('test_name', 'Unknown test')}")
                else:
                    print(f"❌ Failed to fix: {failure.get('test_name', 'Unknown test')}")
            
            # Commit changes if requested
            if auto_commit and any(r.get('fixed') for r in fix_results):
                self._commit_fixes(failure_type, fix_results)
            
            # Update database
            self._update_fix_attempts(type_failures, fix_results)
            
            # Return summary
            fixed_count = sum(1 for r in fix_results if r.get('fixed'))
            return {
                'status': 'completed',
                'failure_type': failure_type,
                'total_failures': len(type_failures),
                'fixed_count': fixed_count,
                'success_rate': fixed_count / len(type_failures) if type_failures else 0,
                'results': fix_results
            }
            
        except Exception as e:
            error_msg = f"Auto-fix failed: {e}"
            print(f"❌ {error_msg}")
            return {'status': 'error', 'error': error_msg}
    
    def _filter_failures_by_type(self, failures: Dict[str, Any], failure_type: str) -> List[Dict]:
        """Filter failures by specific type"""
        if isinstance(failures, dict) and 'categorized' in failures:
            return failures['categorized'].get(failure_type, [])
        elif isinstance(failures, dict) and 'raw_failures' in failures:
            # Filter raw failures by type
            return [f for f in failures['raw_failures'] if f.get('failure_type') == failure_type]
        elif isinstance(failures, list):
            return [f for f in failures if f.get('failure_type') == failure_type]
        else:
            return []
    
    def _fix_single_failure(self, failure: Dict[str, Any], failure_type: str) -> Dict[str, Any]:
        """Fix a single test failure using Claude"""
        try:
            # Prepare context for Claude
            context = self._prepare_fix_context(failure, failure_type)
            
            # Generate fix using Claude
            fix_result = self._generate_fix_with_claude(context, failure_type)
            
            if fix_result.get('success'):
                # Apply the fix
                applied = self._apply_fix(fix_result['fix_code'], failure.get('file_path', ''))
                
                if applied:
                    # Validate the fix by running tests
                    validation_result = self._validate_fix(failure)
                    
                    return {
                        'fixed': validation_result.get('passed', False),
                        'fix_applied': True,
                        'validation_result': validation_result,
                        'fix_code': fix_result.get('fix_code', ''),
                        'failure': failure
                    }
                else:
                    return {
                        'fixed': False,
                        'fix_applied': False,
                        'error': 'Failed to apply fix',
                        'failure': failure
                    }
            else:
                return {
                    'fixed': False,
                    'fix_applied': False,
                    'error': fix_result.get('error', 'Failed to generate fix'),
                    'failure': failure
                }
                
        except Exception as e:
            return {
                'fixed': False,
                'fix_applied': False,
                'error': str(e),
                'failure': failure
            }
    
    def _prepare_fix_context(self, failure: Dict[str, Any], failure_type: str) -> Dict[str, Any]:
        """Prepare context for Claude to generate fix"""
        context = {
            'failure_type': failure_type,
            'test_name': failure.get('test_name', ''),
            'failure_message': failure.get('failure_message', ''),
            'stack_trace': failure.get('stack_trace', ''),
            'file_path': failure.get('file_path', ''),
            'severity': failure.get('severity', 'medium')
        }
        
        # Add file content if available
        if failure.get('file_path'):
            file_path = self.project_root / failure['file_path']
            if file_path.exists():
                try:
                    context['file_content'] = file_path.read_text()
                except Exception:
                    context['file_content'] = 'Could not read file content'
        
        # Add related test files
        context['related_files'] = self._find_related_files(failure.get('file_path', ''))
        
        return context
    
    def _generate_fix_with_claude(self, context: Dict[str, Any], failure_type: str) -> Dict[str, Any]:
        """Generate fix using Claude via Claude Flow"""
        try:
            # Use Claude Flow to generate fix
            fix_prompt = self._create_fix_prompt(context, failure_type)
            
            # Call Claude Flow
            result = subprocess.run([
                'npx', 'claude-flow', 'sparc', 'run', 'coder', fix_prompt
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                # Parse Claude's response
                fix_code = self._extract_fix_code(result.stdout)
                return {
                    'success': True,
                    'fix_code': fix_code,
                    'explanation': result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': f"Claude Flow failed: {result.stderr}"
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_fix_prompt(self, context: Dict[str, Any], failure_type: str) -> str:
        """Create prompt for Claude to generate fix"""
        prompt = f"""
Fix the following {failure_type} test failure:

**Test Name**: {context.get('test_name', 'Unknown')}
**File Path**: {context.get('file_path', 'Unknown')}
**Failure Message**: {context.get('failure_message', 'Unknown')}

**Stack Trace**:
```
{context.get('stack_trace', 'No stack trace available')}
```

**Current File Content**:
```
{context.get('file_content', 'File content not available')[:2000]}
```

Please provide a complete fix for this {failure_type} failure. 

Requirements:
1. Analyze the root cause of the failure
2. Provide the corrected code
3. Ensure the fix maintains existing functionality
4. Include any necessary imports or dependencies
5. Make the fix minimal and focused

Respond with the corrected code that should replace the problematic section.
"""
        return prompt
    
    def _extract_fix_code(self, claude_response: str) -> str:
        """Extract fix code from Claude's response"""
        # Look for code blocks
        import re
        
        # Find code blocks
        code_blocks = re.findall(r'```(?:python|javascript|js|py)?\n?(.*?)```', 
                               claude_response, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, return the response (might be just the fix)
        return claude_response.strip()
    
    def _apply_fix(self, fix_code: str, file_path: str) -> bool:
        """Apply the generated fix to the file"""
        if not file_path or not fix_code:
            return False
        
        try:
            full_path = self.project_root / file_path
            if not full_path.exists():
                print(f"Warning: File not found: {full_path}")
                return False
            
            # For now, we'll create a backup and apply the fix
            # In a more sophisticated version, we'd do intelligent merging
            
            # Create backup
            backup_path = full_path.with_suffix(full_path.suffix + '.backup')
            backup_path.write_text(full_path.read_text())
            
            # Apply fix (simple replacement for now)
            # TODO: Implement smarter patching
            current_content = full_path.read_text()
            
            # If the fix_code looks like a complete file, replace entire content
            if ('def ' in fix_code and 'import ' in fix_code) or len(fix_code) > len(current_content) * 0.8:
                full_path.write_text(fix_code)
            else:
                # Try to do intelligent replacement
                # For now, append the fix
                updated_content = current_content + '\n\n# Auto-generated fix:\n' + fix_code
                full_path.write_text(updated_content)
            
            return True
            
        except Exception as e:
            print(f"Error applying fix: {e}")
            return False
    
    def _validate_fix(self, failure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the fix works by running the test"""
        try:
            test_name = failure.get('test_name', '')
            file_path = failure.get('file_path', '')
            
            # Run the specific test
            if 'pytest' in test_name.lower() or file_path.endswith('.py'):
                # Python test
                cmd = ['python', '-m', 'pytest', file_path, '-v']
            elif file_path.endswith('.js') or file_path.endswith('.ts'):
                # JavaScript test
                cmd = ['npm', 'test', '--', '--testNamePattern=' + test_name]
            else:
                # Generic test run
                cmd = ['python', '-m', 'pytest', '-k', test_name]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=self.project_root, timeout=60)
            
            return {
                'passed': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {'passed': False, 'error': 'Test execution timeout'}
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _find_related_files(self, file_path: str) -> List[str]:
        """Find related files that might be relevant for fixing"""
        if not file_path:
            return []
        
        related = []
        base_path = Path(file_path)
        
        # Look for related test files
        if 'test' in base_path.name:
            # Look for the source file
            source_name = base_path.name.replace('test_', '').replace('_test', '')
            for ext in ['.py', '.js', '.ts']:
                potential = base_path.parent / f"{source_name.split('.')[0]}{ext}"
                if potential.exists():
                    related.append(str(potential))
        
        return related
    
    def _commit_fixes(self, failure_type: str, fix_results: List[Dict]) -> bool:
        """Commit the applied fixes"""
        try:
            fixed_count = sum(1 for r in fix_results if r.get('fixed'))
            
            # Add changed files
            subprocess.run(['git', 'add', '.'], cwd=self.project_root)
            
            # Commit with descriptive message
            commit_msg = f"""fix: Auto-fix {fixed_count} {failure_type} test failures

- Automatically analyzed and fixed {failure_type} test failures
- Applied intelligent fixes using Claude Code automation
- Validated fixes by running tests

Co-authored-by: Claude <claude@anthropic.com>"""
            
            result = subprocess.run([
                'git', 'commit', '-m', commit_msg
            ], capture_output=True, text=True, cwd=self.project_root)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error committing fixes: {e}")
            return False
    
    def _update_fix_attempts(self, failures: List[Dict], results: List[Dict]):
        """Update database with fix attempts"""
        try:
            with sqlite3.connect(self.failures_db) as conn:
                for failure, result in zip(failures, results):
                    # Update fix attempt count and status
                    status = 'resolved' if result.get('fixed') else 'failed'
                    
                    conn.execute('''
                        UPDATE test_failures 
                        SET fix_attempt_count = fix_attempt_count + 1,
                            status = ?,
                            resolved_at = ?
                        WHERE test_name = ? AND failure_message = ?
                    ''', (
                        status,
                        datetime.now().isoformat() if result.get('fixed') else None,
                        failure.get('test_name', ''),
                        failure.get('failure_message', '')
                    ))
                    
                    # Insert fix attempt record
                    conn.execute('''
                        INSERT INTO fix_attempts 
                        (failure_id, attempted_fix, result, success)
                        SELECT id, ?, ?, ?
                        FROM test_failures 
                        WHERE test_name = ? AND failure_message = ?
                        ORDER BY id DESC LIMIT 1
                    ''', (
                        result.get('fix_code', ''),
                        json.dumps(result),
                        result.get('fixed', False),
                        failure.get('test_name', ''),
                        failure.get('failure_message', '')
                    ))
                    
        except Exception as e:
            print(f"Error updating database: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Auto-fix test failures using Claude')
    parser.add_argument('--failure-type', required=True, 
                       choices=['syntax', 'logic', 'integration', 'performance', 'dependency'],
                       help='Type of failures to fix')
    parser.add_argument('--failure-data', required=True, 
                       help='JSON string or file path containing failure data')
    parser.add_argument('--pr-number', type=int, help='Pull request number')
    parser.add_argument('--auto-commit', action='store_true', 
                       help='Automatically commit successful fixes')
    
    args = parser.parse_args()
    
    # Load failure data
    if os.path.isfile(args.failure_data):
        with open(args.failure_data, 'r') as f:
            failure_data = json.load(f)
    else:
        failure_data = args.failure_data
    
    # Fix failures
    fixer = AutoTestFixer()
    result = fixer.fix_failures(
        failure_type=args.failure_type,
        failure_data=failure_data,
        pr_number=args.pr_number,
        auto_commit=args.auto_commit
    )
    
    # Print results
    if result.get('status') == 'completed':
        print(f"\n✅ Auto-fix completed for {args.failure_type} failures:")
        print(f"  Total failures: {result['total_failures']}")
        print(f"  Successfully fixed: {result['fixed_count']}")
        print(f"  Success rate: {result['success_rate']:.1%}")
    elif result.get('status') == 'no_failures':
        print(f"ℹ️ No {args.failure_type} failures found to fix")
    else:
        print(f"❌ Auto-fix failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()