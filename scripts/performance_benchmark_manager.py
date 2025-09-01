#!/usr/bin/env python3
"""
Performance Benchmark Manager for Security Validation

Provides before/after performance benchmarking to ensure security fixes
don't degrade system performance beyond acceptable thresholds.

Features:
- Baseline performance measurement
- Post-fix performance comparison
- Regression detection and alerting
- Performance trend analysis
"""

import asyncio
import json
import logging
import os
import psutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Individual performance benchmark"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.results: List[Dict[str, Any]] = []
        
    def add_result(self, result: Dict[str, Any]):
        """Add a benchmark result"""
        result["timestamp"] = datetime.now().isoformat()
        self.results.append(result)
        
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent result"""
        return self.results[-1] if self.results else None
        
    def get_baseline_result(self) -> Optional[Dict[str, Any]]:
        """Get the baseline result (first result)"""
        return self.results[0] if self.results else None

class PerformanceBenchmarkManager:
    """Manager for performance benchmarking across security fixes"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or os.getcwd())
        self.benchmarks_dir = self.base_path / "benchmarks" / "performance"
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        
    def create_benchmark(self, name: str, description: str = "") -> PerformanceBenchmark:
        """Create a new benchmark"""
        benchmark = PerformanceBenchmark(name, description)
        self.benchmarks[name] = benchmark
        return benchmark
        
    def run_system_resource_benchmark(self) -> Dict[str, Any]:
        """Benchmark system resource usage"""
        logger.info("Running system resource benchmark...")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage_mb = memory.used / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_gb = disk.used / (1024 * 1024 * 1024)
        disk_total_gb = disk.total / (1024 * 1024 * 1024)
        disk_percent = (disk.used / disk.total) * 100
        
        # Load average (if available)
        load_avg = None
        try:
            load_avg = os.getloadavg()
        except (OSError, AttributeError):
            pass  # Not available on Windows
        
        result = {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": {
                "used_mb": round(memory_usage_mb, 2),
                "total_mb": round(memory_total_mb, 2),
                "percent": memory_percent
            },
            "disk": {
                "used_gb": round(disk_usage_gb, 2),
                "total_gb": round(disk_total_gb, 2),
                "percent": round(disk_percent, 2)
            }
        }
        
        if load_avg:
            result["load_average"] = {
                "1min": load_avg[0],
                "5min": load_avg[1],
                "15min": load_avg[2]
            }
        
        return result
        
    def run_module_import_benchmark(self) -> Dict[str, Any]:
        """Benchmark module import times"""
        logger.info("Running module import benchmark...")
        
        modules_to_test = [
            "core.agent_forge",
            "core.rag", 
            "core.monitoring",
            "core.p2p"
        ]
        
        import_times = {}
        successful_imports = 0
        
        for module in modules_to_test:
            try:
                start_time = time.time()
                
                # Test import in subprocess to get clean timing
                result = subprocess.run([
                    sys.executable, "-c", 
                    f"import time; start=time.time(); import {module}; print(time.time()-start)"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    import_time = float(result.stdout.strip())
                    import_times[module] = import_time
                    successful_imports += 1
                else:
                    import_times[module] = "FAILED"
                    
            except (subprocess.TimeoutExpired, ValueError):
                import_times[module] = "TIMEOUT"
            except Exception as e:
                import_times[module] = f"ERROR: {str(e)}"
        
        avg_import_time = None
        successful_times = [t for t in import_times.values() if isinstance(t, float)]
        if successful_times:
            avg_import_time = statistics.mean(successful_times)
        
        return {
            "import_times": import_times,
            "successful_imports": successful_imports,
            "total_modules": len(modules_to_test),
            "average_import_time": avg_import_time,
            "success_rate": successful_imports / len(modules_to_test)
        }
        
    def run_test_execution_benchmark(self) -> Dict[str, Any]:
        """Benchmark test execution performance"""
        logger.info("Running test execution benchmark...")
        
        try:
            # Run a subset of tests with timing
            start_time = time.time()
            
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/", "--collect-only", "-q", "--tb=no"
            ], capture_output=True, text=True, timeout=120)
            
            collection_time = time.time() - start_time
            
            test_count = 0
            if result.returncode == 0 and "collected" in result.stdout:
                # Extract test count
                for line in result.stdout.split('\n'):
                    if "collected" in line:
                        try:
                            test_count = int(line.split()[0])
                            break
                        except (ValueError, IndexError):
                            pass
            
            return {
                "collection_time": round(collection_time, 3),
                "test_count": test_count,
                "collection_rate": round(test_count / collection_time, 2) if collection_time > 0 else 0,
                "collection_successful": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "collection_time": 120.0,
                "test_count": 0,
                "collection_rate": 0,
                "collection_successful": False,
                "error": "Timeout"
            }
        except Exception as e:
            return {
                "collection_time": 0,
                "test_count": 0,
                "collection_rate": 0,
                "collection_successful": False,
                "error": str(e)
            }
            
    def run_gateway_startup_benchmark(self) -> Dict[str, Any]:
        """Benchmark gateway startup performance"""
        logger.info("Running gateway startup benchmark...")
        
        try:
            # Test gateway config loading time (without actually starting server)
            test_code = '''
import os
import time
os.environ["API_KEY"] = "test_key_for_benchmark"
os.environ["SECRET_KEY"] = "test_secret_key_for_benchmark_32chars"
start_time = time.time()
from core.gateway.server import GatewayConfig
config = GatewayConfig()
config_time = time.time() - start_time
print(f"CONFIG_TIME:{config_time}")
'''
            
            start_time = time.time()
            result = subprocess.run([
                sys.executable, "-c", test_code
            ], capture_output=True, text=True, timeout=30)
            total_time = time.time() - start_time
            
            config_time = None
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith("CONFIG_TIME:"):
                        try:
                            config_time = float(line.split(":")[1])
                            break
                        except (ValueError, IndexError):
                            pass
            
            return {
                "total_startup_time": round(total_time, 3),
                "config_load_time": round(config_time, 3) if config_time else None,
                "startup_successful": result.returncode == 0,
                "error": result.stderr if result.returncode != 0 else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "total_startup_time": 30.0,
                "config_load_time": None,
                "startup_successful": False,
                "error": "Timeout"
            }
        except Exception as e:
            return {
                "total_startup_time": 0,
                "config_load_time": None,
                "startup_successful": False,
                "error": str(e)
            }
    
    def run_comprehensive_benchmark_suite(self) -> Dict[str, Any]:
        """Run all benchmark tests"""
        logger.info("Starting comprehensive performance benchmark suite...")
        
        suite_start_time = time.time()
        
        # System resources
        system_benchmark = self.create_benchmark(
            "system_resources",
            "System resource usage baseline"
        )
        system_result = self.run_system_resource_benchmark()
        system_benchmark.add_result(system_result)
        
        # Module imports
        import_benchmark = self.create_benchmark(
            "module_imports",
            "Module import performance"
        )
        import_result = self.run_module_import_benchmark()
        import_benchmark.add_result(import_result)
        
        # Test execution
        test_benchmark = self.create_benchmark(
            "test_execution",
            "Test suite execution performance"
        )
        test_result = self.run_test_execution_benchmark()
        test_benchmark.add_result(test_result)
        
        # Gateway startup
        gateway_benchmark = self.create_benchmark(
            "gateway_startup",
            "Gateway startup performance"
        )
        gateway_result = self.run_gateway_startup_benchmark()
        gateway_benchmark.add_result(gateway_result)
        
        suite_duration = time.time() - suite_start_time
        
        return {
            "suite_duration": round(suite_duration, 3),
            "benchmarks": {
                "system_resources": system_result,
                "module_imports": import_result,
                "test_execution": test_result,
                "gateway_startup": gateway_result
            },
            "completed_at": datetime.now().isoformat()
        }
        
    def save_benchmark_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.benchmarks_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
        return filepath
        
    def load_benchmark_results(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load benchmark results from file"""
        filepath = self.benchmarks_dir / filename
        
        if not filepath.exists():
            logger.error(f"Benchmark file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load benchmark results: {e}")
            return None
            
    def compare_benchmark_results(self, baseline_file: str, current_file: str) -> Dict[str, Any]:
        """Compare two benchmark results"""
        baseline = self.load_benchmark_results(baseline_file)
        current = self.load_benchmark_results(current_file)
        
        if not baseline or not current:
            return {"error": "Failed to load benchmark files"}
        
        comparison = {
            "baseline_file": baseline_file,
            "current_file": current_file,
            "comparison_time": datetime.now().isoformat(),
            "regressions": [],
            "improvements": [],
            "summary": {}
        }
        
        # Compare system resources
        if "system_resources" in baseline["benchmarks"] and "system_resources" in current["benchmarks"]:
            baseline_sys = baseline["benchmarks"]["system_resources"]
            current_sys = current["benchmarks"]["system_resources"]
            
            # CPU comparison
            cpu_change = current_sys["cpu"]["percent"] - baseline_sys["cpu"]["percent"]
            if cpu_change > 10:  # 10% increase is concerning
                comparison["regressions"].append({
                    "metric": "CPU usage",
                    "change": f"+{cpu_change:.1f}%",
                    "baseline": baseline_sys["cpu"]["percent"],
                    "current": current_sys["cpu"]["percent"]
                })
            elif cpu_change < -5:  # 5% decrease is good
                comparison["improvements"].append({
                    "metric": "CPU usage",
                    "change": f"{cpu_change:.1f}%",
                    "baseline": baseline_sys["cpu"]["percent"],
                    "current": current_sys["cpu"]["percent"]
                })
            
            # Memory comparison
            memory_change = current_sys["memory"]["percent"] - baseline_sys["memory"]["percent"]
            if memory_change > 15:  # 15% increase is concerning
                comparison["regressions"].append({
                    "metric": "Memory usage",
                    "change": f"+{memory_change:.1f}%",
                    "baseline": baseline_sys["memory"]["percent"],
                    "current": current_sys["memory"]["percent"]
                })
        
        # Compare import times
        if "module_imports" in baseline["benchmarks"] and "module_imports" in current["benchmarks"]:
            baseline_imp = baseline["benchmarks"]["module_imports"]
            current_imp = current["benchmarks"]["module_imports"]
            
            if baseline_imp.get("average_import_time") and current_imp.get("average_import_time"):
                time_change = current_imp["average_import_time"] - baseline_imp["average_import_time"]
                if time_change > 0.5:  # 500ms increase is concerning
                    comparison["regressions"].append({
                        "metric": "Average import time",
                        "change": f"+{time_change:.3f}s",
                        "baseline": baseline_imp["average_import_time"],
                        "current": current_imp["average_import_time"]
                    })
        
        # Compare test execution
        if "test_execution" in baseline["benchmarks"] and "test_execution" in current["benchmarks"]:
            baseline_test = baseline["benchmarks"]["test_execution"]
            current_test = current["benchmarks"]["test_execution"]
            
            if baseline_test["collection_rate"] and current_test["collection_rate"]:
                rate_change = current_test["collection_rate"] - baseline_test["collection_rate"]
                if rate_change < -10:  # 10 tests/sec decrease is concerning
                    comparison["regressions"].append({
                        "metric": "Test collection rate",
                        "change": f"{rate_change:.1f} tests/sec",
                        "baseline": baseline_test["collection_rate"],
                        "current": current_test["collection_rate"]
                    })
        
        # Summary
        comparison["summary"] = {
            "total_regressions": len(comparison["regressions"]),
            "total_improvements": len(comparison["improvements"]),
            "performance_impact": "HIGH" if len(comparison["regressions"]) > 2 else 
                                 "MEDIUM" if len(comparison["regressions"]) > 0 else "LOW"
        }
        
        return comparison
        
    def generate_performance_report(self, comparison: Dict[str, Any]) -> str:
        """Generate human-readable performance report"""
        report = []
        report.append("="*60)
        report.append("PERFORMANCE IMPACT ANALYSIS")
        report.append("="*60)
        
        report.append(f"Baseline: {comparison['baseline_file']}")
        report.append(f"Current:  {comparison['current_file']}")
        report.append(f"Analysis Time: {comparison['comparison_time']}")
        report.append("")
        
        # Summary
        summary = comparison["summary"]
        report.append("SUMMARY:")
        report.append(f"  Performance Impact: {summary['performance_impact']}")
        report.append(f"  Regressions Found: {summary['total_regressions']}")
        report.append(f"  Improvements Found: {summary['total_improvements']}")
        report.append("")
        
        # Regressions
        if comparison["regressions"]:
            report.append("PERFORMANCE REGRESSIONS:")
            for reg in comparison["regressions"]:
                report.append(f"  ❌ {reg['metric']}: {reg['change']}")
                report.append(f"     Baseline: {reg['baseline']} → Current: {reg['current']}")
            report.append("")
        
        # Improvements
        if comparison["improvements"]:
            report.append("PERFORMANCE IMPROVEMENTS:")
            for imp in comparison["improvements"]:
                report.append(f"  ✅ {imp['metric']}: {imp['change']}")
                report.append(f"     Baseline: {imp['baseline']} → Current: {imp['current']}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if summary["performance_impact"] == "HIGH":
            report.append("  🚨 HIGH IMPACT: Review security fixes for performance bottlenecks")
            report.append("  📋 Consider rollback if regressions are unacceptable")
        elif summary["performance_impact"] == "MEDIUM":
            report.append("  ⚠️ MEDIUM IMPACT: Monitor performance in production")
            report.append("  🔍 Investigate specific regressions")
        else:
            report.append("  ✅ LOW IMPACT: Performance impact within acceptable limits")
            report.append("  🚀 Proceed with deployment")
        
        return "\n".join(report)

def main():
    """Main CLI for performance benchmark manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Benchmark Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run benchmarks
    run_parser = subparsers.add_parser("run", help="Run performance benchmarks")
    run_parser.add_argument("-o", "--output", help="Output filename for results")
    
    # Compare results
    compare_parser = subparsers.add_parser("compare", help="Compare two benchmark results")
    compare_parser.add_argument("baseline", help="Baseline benchmark file")
    compare_parser.add_argument("current", help="Current benchmark file")
    compare_parser.add_argument("-r", "--report", action="store_true", help="Generate detailed report")
    
    # List results
    list_parser = subparsers.add_parser("list", help="List available benchmark results")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = PerformanceBenchmarkManager()
    
    if args.command == "run":
        results = manager.run_comprehensive_benchmark_suite()
        filename = args.output or None
        filepath = manager.save_benchmark_results(results, filename)
        
        print("Performance Benchmark Results:")
        print(f"Suite Duration: {results['suite_duration']:.3f}s")
        print(f"Results saved to: {filepath}")
        
    elif args.command == "compare":
        comparison = manager.compare_benchmark_results(args.baseline, args.current)
        
        if "error" in comparison:
            print(f"Error: {comparison['error']}")
            sys.exit(1)
        
        if args.report:
            report = manager.generate_performance_report(comparison)
            print(report)
        else:
            print(f"Regressions: {comparison['summary']['total_regressions']}")
            print(f"Improvements: {comparison['summary']['total_improvements']}")
            print(f"Impact: {comparison['summary']['performance_impact']}")
        
    elif args.command == "list":
        benchmark_files = list(manager.benchmarks_dir.glob("*.json"))
        if benchmark_files:
            print("Available benchmark results:")
            for file in sorted(benchmark_files):
                print(f"  {file.name}")
        else:
            print("No benchmark results found")

if __name__ == "__main__":
    main()