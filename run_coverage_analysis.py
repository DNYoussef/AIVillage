#!/usr/bin/env python3
"""
Comprehensive Test Coverage Analysis for AIVillage
Analyze current test coverage and identify gaps for improvement.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple
import glob

def run_coverage_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)

def find_python_files() -> List[Path]:
    """Find all Python files in the project."""
    project_root = Path(__file__).parent
    python_files = []
    
    # Exclude common directories we don't want to test
    exclude_dirs = {
        'new_env', 'evomerge_env', '__pycache__', '.git', 
        'htmlcov', 'node_modules', 'migrations'
    }
    
    for py_file in project_root.rglob("*.py"):
        # Skip if in excluded directory
        if any(exclude_dir in py_file.parts for exclude_dir in exclude_dirs):
            continue
        # Skip test files themselves
        if 'test' in py_file.name or py_file.parts[-2:] == ('tests',):
            continue
        python_files.append(py_file)
    
    return python_files

def analyze_test_coverage() -> Dict:
    """Run coverage analysis and return results."""
    print("🔍 Running comprehensive test coverage analysis...")
    
    # Run pytest with coverage
    coverage_cmd = [
        sys.executable, "-m", "pytest", 
        "--cov=.", 
        "--cov-report=json:coverage.json",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "-v", "--tb=short"
    ]
    
    print(f"Running command: {' '.join(coverage_cmd)}")
    exit_code, stdout, stderr = run_coverage_command(coverage_cmd)
    
    print(f"Coverage command exit code: {exit_code}")
    if stderr:
        print(f"Stderr: {stderr}")
    
    # Try to load coverage results
    coverage_data = {}
    coverage_file = Path("coverage.json")
    
    if coverage_file.exists():
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
                print("✅ Coverage data loaded successfully")
        except Exception as e:
            print(f"❌ Error loading coverage data: {e}")
    else:
        print("❌ Coverage file not found")
    
    return {
        'exit_code': exit_code,
        'stdout': stdout,
        'stderr': stderr,
        'coverage_data': coverage_data
    }

def identify_critical_untested_components() -> List[Dict]:
    """Identify critical components that lack tests."""
    project_root = Path(__file__).parent
    critical_components = []
    
    # Critical directories to analyze
    critical_dirs = [
        'mcp_servers',
        'production',
        'agent_forge',
        'communications',
        'experimental/agents',
        'digital_twin'
    ]
    
    for critical_dir in critical_dirs:
        dir_path = project_root / critical_dir
        if not dir_path.exists():
            continue
            
        # Find Python files in this critical directory
        for py_file in dir_path.rglob("*.py"):
            if '__pycache__' in str(py_file) or 'test' in py_file.name:
                continue
                
            # Check if corresponding test exists
            relative_path = py_file.relative_to(project_root)
            test_patterns = [
                project_root / "tests" / f"test_{py_file.stem}.py",
                project_root / "tests" / relative_path.parent / f"test_{py_file.stem}.py",
                py_file.parent / f"test_{py_file.stem}.py"
            ]
            
            has_test = any(test_path.exists() for test_path in test_patterns)
            
            critical_components.append({
                'file': str(relative_path),
                'has_test': has_test,
                'category': critical_dir,
                'size': py_file.stat().st_size if py_file.exists() else 0
            })
    
    return critical_components

def analyze_existing_tests() -> Dict:
    """Analyze the existing test suite structure."""
    project_root = Path(__file__).parent
    test_files = list(project_root.rglob("*test*.py"))
    
    test_analysis = {
        'total_test_files': len(test_files),
        'test_categories': {},
        'test_distribution': {}
    }
    
    for test_file in test_files:
        if 'new_env' in str(test_file) or 'evomerge_env' in str(test_file):
            continue
            
        # Categorize by directory
        relative_path = test_file.relative_to(project_root)
        category = str(relative_path.parts[0]) if relative_path.parts else 'root'
        
        if category not in test_analysis['test_categories']:
            test_analysis['test_categories'][category] = []
        
        test_analysis['test_categories'][category].append(str(relative_path))
    
    return test_analysis

def generate_coverage_report(coverage_results: Dict, critical_components: List[Dict], test_analysis: Dict):
    """Generate comprehensive coverage report."""
    
    report = f"""# AIVillage Test Coverage Analysis Report
Generated: {os.popen('date').read().strip()}

## Executive Summary

### Current Test Infrastructure Status
- **Test Files Discovered**: {test_analysis['total_test_files']}
- **Coverage Analysis Status**: {'✅ Success' if coverage_results['exit_code'] == 0 else '❌ Failed'}
- **Critical Components Analyzed**: {len(critical_components)}

### Coverage Overview
"""
    
    if coverage_results['coverage_data']:
        coverage_data = coverage_results['coverage_data']
        if 'totals' in coverage_data:
            totals = coverage_data['totals']
            coverage_percent = (totals.get('covered_lines', 0) / max(totals.get('num_lines', 1), 1)) * 100
            
            report += f"""
**Overall Coverage**: {coverage_percent:.1f}%
- **Lines Covered**: {totals.get('covered_lines', 0)}
- **Total Lines**: {totals.get('num_lines', 0)}
- **Missing Lines**: {totals.get('num_lines', 0) - totals.get('covered_lines', 0)}
"""
    else:
        report += "\n**Coverage Data**: Not available (tests may not have run successfully)\n"
    
    # Critical components analysis
    untested_critical = [c for c in critical_components if not c['has_test']]
    tested_critical = [c for c in critical_components if c['has_test']]
    
    report += f"""
## Critical Components Analysis

### Test Coverage by Category
- **Total Critical Components**: {len(critical_components)}
- **Components with Tests**: {len(tested_critical)} ({len(tested_critical)/len(critical_components)*100:.1f}%)
- **Components without Tests**: {len(untested_critical)} ({len(untested_critical)/len(critical_components)*100:.1f}%)

### High Priority Untested Components
"""
    
    # Group untested by category
    untested_by_category = {}
    for component in untested_critical:
        category = component['category']
        if category not in untested_by_category:
            untested_by_category[category] = []
        untested_by_category[category].append(component)
    
    for category, components in untested_by_category.items():
        report += f"\n#### {category.title()} ({len(components)} untested)\n"
        # Sort by file size (larger files are higher priority)
        components.sort(key=lambda x: x['size'], reverse=True)
        
        for component in components[:10]:  # Top 10 per category
            report += f"- `{component['file']}` ({component['size']} bytes)\n"
        
        if len(components) > 10:
            report += f"- ... and {len(components) - 10} more files\n"
    
    # Test distribution analysis
    report += f"""
## Existing Test Distribution

### Tests by Category
"""
    for category, files in test_analysis['test_categories'].items():
        report += f"- **{category}**: {len(files)} test files\n"
    
    # Recommendations
    report += """
## Recommendations for 90%+ Coverage

### Immediate Priority (Week 1)
1. **MCP Server Testing** - Critical for production
   - Create integration tests for server.py and protocol.py
   - Add unit tests for memory, planning, and retrieval modules
   - Test authentication and security components

2. **Production Pipeline Testing** - Essential for reliability
   - Comprehensive compression pipeline tests
   - Evolution system integration tests
   - RAG system component tests

### Medium Priority (Week 2)
3. **Agent Forge Core Testing**
   - Test orchestration and workflow management
   - Add benchmark and evaluation tests
   - Test model loading and deployment

4. **Communication System Testing**
   - Test message queue and protocol handling
   - Add integration tests for cross-component communication
   - Test credit system and community hub features

### Long-term Priority (Week 3+)
5. **Experimental Component Testing**
   - Test agent implementations where stable
   - Add integration tests for multi-agent scenarios
   - Performance and load testing

6. **Digital Twin Testing**
   - Test personalization and tutoring components
   - Add security and privacy tests
   - Test edge deployment scenarios

### Test Infrastructure Improvements
- Set up automated coverage reporting in CI/CD
- Create comprehensive integration test framework
- Add performance benchmarking tests
- Implement test fixtures for complex scenarios
- Set up test data management and cleanup

## Coverage Monitoring Plan
- Weekly coverage reports
- Coverage gates for new code (85%+ for production components)
- Integration with GitHub Actions for PR testing
- Performance regression testing
"""
    
    return report

def main():
    """Main execution function."""
    print("🚀 Starting AIVillage Test Coverage Analysis")
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Step 1: Run coverage analysis
    coverage_results = analyze_test_coverage()
    
    # Step 2: Identify critical untested components
    print("🔍 Identifying critical untested components...")
    critical_components = identify_critical_untested_components()
    
    # Step 3: Analyze existing tests
    print("📊 Analyzing existing test structure...")
    test_analysis = analyze_existing_tests()
    
    # Step 4: Generate comprehensive report
    print("📝 Generating coverage report...")
    report = generate_coverage_report(coverage_results, critical_components, test_analysis)
    
    # Save report
    report_file = Path("COVERAGE_ANALYSIS_REPORT.md")
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Coverage analysis complete! Report saved to {report_file}")
    print(f"📊 Summary:")
    print(f"   - Test files discovered: {test_analysis['total_test_files']}")
    print(f"   - Critical components analyzed: {len(critical_components)}")
    print(f"   - Components without tests: {len([c for c in critical_components if not c['has_test']])}")
    
    # Show top priority untested files
    untested = [c for c in critical_components if not c['has_test']]
    untested.sort(key=lambda x: x['size'], reverse=True)
    
    print("\n🎯 Top 10 priority files needing tests:")
    for i, component in enumerate(untested[:10], 1):
        print(f"   {i}. {component['file']} ({component['category']})")

if __name__ == "__main__":
    main()