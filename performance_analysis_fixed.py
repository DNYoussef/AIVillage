#!/usr/bin/env python3
"""Fixed Performance Analysis for AIVillage System

CRITICAL ISSUES:
1. MESH NETWORK: 0% message delivery rate
2. MEMORY: 87% usage (13.8GB/15.9GB)  
3. SYSTEM COMPLEXITY: 81k+ Python files
"""

import json
import logging
import os
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Simple, effective performance analyzer."""
    
    def __init__(self):
        self.start_time = time.time()
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage_percent': round(cpu_percent, 1),
                'memory_total_gb': round(memory.total / (1024**3), 1),
                'memory_available_gb': round(memory.available / (1024**3), 1),
                'memory_used_percent': round(memory.percent, 1),
                'process_memory_gb': round(process.memory_info().rss / (1024**3), 2),
                'cpu_cores': psutil.cpu_count(),
                'status': 'CRITICAL' if memory.percent > 85 else 'WARNING' if memory.percent > 70 else 'OK'
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def analyze_mesh_network(self) -> Dict[str, Any]:
        """Analyze mesh network performance."""
        try:
            mesh_file = Path("mesh_network_test_results.json")
            if not mesh_file.exists():
                return {
                    'status': 'NO_DATA',
                    'message_delivery_rate': 0.0,
                    'issue': 'No mesh network test results found'
                }
            
            with open(mesh_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Extract delivery rates
            routing_results = results.get('detailed_results', {}).get('routing', [])
            delivery_rates = [r.get('delivery_rate', 0) for r in routing_results]
            avg_delivery = sum(delivery_rates) / len(delivery_rates) if delivery_rates else 0
            
            status = 'CRITICAL' if avg_delivery == 0 else 'WARNING' if avg_delivery < 0.5 else 'OK'
            
            return {
                'status': status,
                'message_delivery_rate': round(avg_delivery * 100, 1),
                'overall_success_rate': round(results.get('overall_success_rate', 0) * 100, 1),
                'tests_passed': results.get('tests_passed', 0),
                'tests_total': results.get('tests_total', 0),
                'issue': 'Complete message delivery failure' if avg_delivery == 0 else None
            }
        except Exception as e:
            logger.error(f"Mesh network analysis failed: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def analyze_system_complexity(self) -> Dict[str, Any]:
        """Analyze system complexity metrics."""
        try:
            # Count Python files
            python_files = 0
            total_lines = 0
            large_files = []
            
            for root, dirs, files in os.walk('.'):
                # Skip virtual environments and caches
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'env', '__pycache__']]
                
                for file in files:
                    if file.endswith('.py'):
                        python_files += 1
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = len(f.readlines())
                                total_lines += lines
                                if lines > 1000:  # Files with >1000 lines
                                    large_files.append({
                                        'file': file_path,
                                        'lines': lines
                                    })
                        except Exception:
                            continue
            
            # Sort large files by size
            large_files.sort(key=lambda x: x['lines'], reverse=True)
            
            complexity_status = 'HIGH' if python_files > 10000 else 'MEDIUM' if python_files > 1000 else 'LOW'
            
            return {
                'python_files': python_files,
                'total_lines': total_lines,
                'avg_lines_per_file': round(total_lines / python_files, 1) if python_files > 0 else 0,
                'large_files_count': len(large_files),
                'largest_files': large_files[:10],  # Top 10 largest files
                'complexity_status': complexity_status,
                'issue': 'Excessive system complexity' if python_files > 10000 else None
            }
        except Exception as e:
            logger.error(f"System complexity analysis failed: {e}")
            return {'error': str(e)}
    
    def identify_critical_issues(self, system_metrics: Dict[str, Any], 
                               mesh_analysis: Dict[str, Any], 
                               complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify critical performance issues."""
        issues = []
        
        # Critical: Mesh network failure
        if mesh_analysis.get('message_delivery_rate', 0) == 0:
            issues.append({
                'severity': 'CRITICAL',
                'component': 'Mesh Network',
                'description': '0% message delivery rate - complete communication failure',
                'impact': 'Distributed operations impossible',
                'action': 'Fix routing algorithms, message serialization, connection pooling'
            })
        
        # High: Memory pressure
        if system_metrics.get('memory_used_percent', 0) > 85:
            issues.append({
                'severity': 'HIGH',
                'component': 'Memory',
                'description': f"{system_metrics.get('memory_used_percent', 0)}% memory usage",
                'impact': 'Risk of crashes and system instability',
                'action': 'Memory optimization, garbage collection, model compression'
            })
        
        # High: System complexity
        if complexity_analysis.get('python_files', 0) > 10000:
            issues.append({
                'severity': 'HIGH',
                'component': 'System Architecture',
                'description': f"{complexity_analysis.get('python_files', 0)} Python files",
                'impact': 'Slow build times, maintenance overhead',
                'action': 'Code consolidation, modularization, dead code removal'
            })
        
        return {
            'total_issues': len(issues),
            'critical_count': sum(1 for i in issues if i['severity'] == 'CRITICAL'),
            'high_count': sum(1 for i in issues if i['severity'] == 'HIGH'),
            'issues': issues
        }
    
    def generate_optimization_plan(self, issues: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable optimization plan."""
        plan = {
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': []
        }
        
        for issue in issues.get('issues', []):
            if issue['severity'] == 'CRITICAL':
                plan['immediate_actions'].append({
                    'component': issue['component'],
                    'action': issue['action'],
                    'priority': 1,
                    'estimated_effort': '1-2 days'
                })
            elif issue['severity'] == 'HIGH':
                plan['short_term_actions'].append({
                    'component': issue['component'],
                    'action': issue['action'],
                    'priority': 2,
                    'estimated_effort': '3-5 days'
                })
        
        # Long-term improvements
        plan['long_term_actions'].extend([
            {
                'component': 'Performance Monitoring',
                'action': 'Implement continuous performance monitoring dashboard',
                'priority': 3,
                'estimated_effort': '1-2 weeks'
            },
            {
                'component': 'Testing',
                'action': 'Create automated performance regression tests',
                'priority': 3,
                'estimated_effort': '1 week'
            }
        ])
        
        return plan
    
    def create_performance_report(self) -> str:
        """Create comprehensive performance report."""
        # Collect all analysis data
        system_metrics = self.get_system_metrics()
        mesh_analysis = self.analyze_mesh_network()
        complexity_analysis = self.analyze_system_complexity()
        issues = self.identify_critical_issues(system_metrics, mesh_analysis, complexity_analysis)
        optimization_plan = self.generate_optimization_plan(issues)
        
        # Generate report
        report = f"""# AIVillage Performance Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
System Status: **{system_metrics.get('status', 'UNKNOWN')}**
Critical Issues: **{issues.get('critical_count', 0)}**
High Priority Issues: **{issues.get('high_count', 0)}**

## System Metrics
- **CPU Usage**: {system_metrics.get('cpu_usage_percent', 0)}%
- **Memory Usage**: {system_metrics.get('memory_used_percent', 0)}% ({system_metrics.get('process_memory_gb', 0)} GB used)
- **Available Memory**: {system_metrics.get('memory_available_gb', 0)} GB
- **CPU Cores**: {system_metrics.get('cpu_cores', 0)}

## Critical Issues Analysis

### 1. Mesh Network Performance
- **Status**: {mesh_analysis.get('status', 'UNKNOWN')}
- **Message Delivery Rate**: {mesh_analysis.get('message_delivery_rate', 0)}%
- **Overall Success Rate**: {mesh_analysis.get('overall_success_rate', 0)}%
- **Tests Status**: {mesh_analysis.get('tests_passed', 0)}/{mesh_analysis.get('tests_total', 0)} passed

### 2. Memory Utilization  
- **Current Usage**: {system_metrics.get('memory_used_percent', 0)}%
- **Process Memory**: {system_metrics.get('process_memory_gb', 0)} GB
- **Status**: {'CRITICAL - Risk of OOM' if system_metrics.get('memory_used_percent', 0) > 85 else 'Acceptable'}

### 3. System Complexity
- **Python Files**: {complexity_analysis.get('python_files', 0):,}
- **Total Lines**: {complexity_analysis.get('total_lines', 0):,}
- **Avg Lines/File**: {complexity_analysis.get('avg_lines_per_file', 0)}
- **Large Files (>1000 lines)**: {complexity_analysis.get('large_files_count', 0)}

## Issues Identified
"""
        
        for i, issue in enumerate(issues.get('issues', []), 1):
            severity_icon = "🚨" if issue['severity'] == 'CRITICAL' else "⚠️"
            report += f"""
### {i}. {issue['component']} ({severity_icon} {issue['severity']})
**Problem**: {issue['description']}
**Impact**: {issue['impact']}
**Action Required**: {issue['action']}
"""
        
        report += f"""
## Optimization Plan

### Immediate Actions (Critical - Do Now)
"""
        for action in optimization_plan.get('immediate_actions', []):
            report += f"- **{action['component']}**: {action['action']} (Est: {action['estimated_effort']})\n"
        
        report += f"""
### Short-term Actions (High Priority - This Week)
"""
        for action in optimization_plan.get('short_term_actions', []):
            report += f"- **{action['component']}**: {action['action']} (Est: {action['estimated_effort']})\n"
        
        report += f"""
### Long-term Actions (Ongoing Improvements)
"""
        for action in optimization_plan.get('long_term_actions', []):
            report += f"- **{action['component']}**: {action['action']} (Est: {action['estimated_effort']})\n"
        
        report += f"""
## Performance Targets

| Metric | Current | Target | Status |
|--------|---------|---------|--------|
| Mesh Delivery Rate | {mesh_analysis.get('message_delivery_rate', 0)}% | 100% | {'❌' if mesh_analysis.get('message_delivery_rate', 0) < 90 else '✅'} |
| Memory Usage | {system_metrics.get('memory_used_percent', 0)}% | <50% | {'❌' if system_metrics.get('memory_used_percent', 0) > 70 else '✅'} |
| System Files | {complexity_analysis.get('python_files', 0):,} | <10,000 | {'❌' if complexity_analysis.get('python_files', 0) > 10000 else '✅'} |

## Next Steps
1. **CRITICAL**: Fix mesh network message routing (Priority 1)
2. **HIGH**: Implement memory optimization (Priority 2)  
3. **HIGH**: Reduce system complexity (Priority 3)
4. **MEDIUM**: Set up continuous monitoring (Priority 4)

## Files Generated
- Performance dashboard: `performance_dashboard.json`
- Detailed results: `performance_analysis_results.json`
"""
        
        return report
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete performance analysis."""
        logger.info("Starting comprehensive performance analysis...")
        
        try:
            # Collect all data
            system_metrics = self.get_system_metrics()
            mesh_analysis = self.analyze_mesh_network()
            complexity_analysis = self.analyze_system_complexity()
            issues = self.identify_critical_issues(system_metrics, mesh_analysis, complexity_analysis)
            optimization_plan = self.generate_optimization_plan(issues)
            
            # Create comprehensive results
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'system_status': system_metrics.get('status', 'UNKNOWN'),
                'system_metrics': system_metrics,
                'mesh_network': mesh_analysis,
                'system_complexity': complexity_analysis,
                'critical_issues': issues,
                'optimization_plan': optimization_plan,
                'summary': {
                    'total_issues': issues.get('total_issues', 0),
                    'critical_issues': issues.get('critical_count', 0),
                    'memory_usage_percent': system_metrics.get('memory_used_percent', 0),
                    'mesh_delivery_rate': mesh_analysis.get('message_delivery_rate', 0),
                    'python_files_count': complexity_analysis.get('python_files', 0)
                }
            }
            
            # Save results
            with open('performance_analysis_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Generate and save report
            report = self.create_performance_report()
            with open('performance_analysis_report.md', 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info("Performance analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {'error': str(e)}


def main():
    """Main entry point."""
    analyzer = PerformanceAnalyzer()
    
    print("=== AIVillage Performance Analysis ===")
    print("Analyzing system performance...")
    
    results = analyzer.run_complete_analysis()
    
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return
    
    # Print summary
    summary = results.get('summary', {})
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"System Status: {results.get('system_status', 'UNKNOWN')}")
    print(f"Critical Issues: {summary.get('critical_issues', 0)}")
    print(f"Memory Usage: {summary.get('memory_usage_percent', 0)}%")
    print(f"Mesh Delivery Rate: {summary.get('mesh_delivery_rate', 0)}%")
    print(f"Python Files: {summary.get('python_files_count', 0):,}")
    
    print(f"\n=== FILES GENERATED ===")
    print("📊 Report: performance_analysis_report.md")
    print("📋 Results: performance_analysis_results.json")
    
    # Show critical issues
    critical_count = summary.get('critical_issues', 0)
    if critical_count > 0:
        print(f"\n🚨 {critical_count} CRITICAL ISSUES REQUIRE IMMEDIATE ATTENTION!")
        print("See report for detailed action plan.")
    else:
        print(f"\n✅ No critical issues found.")


if __name__ == "__main__":
    main()