"""
Security Overhead Performance Tests

Tests performance impact of security controls and validates acceptable overhead levels.
Ensures security implementations maintain system performance within acceptable limits.

Focus: Performance testing of security mechanisms without compromising security effectiveness.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import threading
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable
import statistics
import psutil
import json

from core.domain.security_constants import SecurityLevel


class SecurityPerformanceBenchmark:
    """Benchmarks performance impact of security controls."""
    
    def __init__(self):
        self.benchmark_results = {}
        self.performance_thresholds = {
            "authentication_latency_ms": 100,
            "authorization_latency_ms": 50,
            "encryption_throughput_mbps": 100,
            "vulnerability_scan_time_s": 30,
            "sbom_generation_time_s": 10,
            "audit_logging_overhead_percent": 5
        }
    
    def benchmark_authentication_performance(self, iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark authentication process performance."""
        from tests.security.unit.test_admin_security import AdminInterfaceServer
        
        admin_server = AdminInterfaceServer(require_mfa=True)
        admin_server.start_server()
        
        # Benchmark authentication timing
        auth_times = []
        successful_auths = 0
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            try:
                result = admin_server.authenticate_user(
                    user_id=f"test_user_{i % 10}",  # Cycle through 10 users
                    password="test_password_456",  # pragma: allowlist secret
                    mfa_token="123456",  # pragma: allowlist secret
                    source_ip="127.0.0.1"
                )
                
                end_time = time.perf_counter()
                auth_time_ms = (end_time - start_time) * 1000
                auth_times.append(auth_time_ms)
                
                if result.get("authenticated", False):
                    successful_auths += 1
                    
            except Exception as e:
                # Some auth attempts may fail due to user limits, that's expected
                import logging
                logging.exception("Authentication performance test iteration failed (expected): %s", str(e))
        
        benchmark_result = {
            "test_type": "authentication_performance",
            "iterations": iterations,
            "successful_authentications": successful_auths,
            "average_latency_ms": statistics.mean(auth_times) if auth_times else 0,
            "median_latency_ms": statistics.median(auth_times) if auth_times else 0,
            "p95_latency_ms": statistics.quantiles(auth_times, n=20)[18] if len(auth_times) >= 20 else 0,
            "max_latency_ms": max(auth_times) if auth_times else 0,
            "min_latency_ms": min(auth_times) if auth_times else 0,
            "meets_threshold": (statistics.mean(auth_times) if auth_times else float('inf')) <= self.performance_thresholds["authentication_latency_ms"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.benchmark_results["authentication"] = benchmark_result
        return benchmark_result
    
    def benchmark_authorization_performance(self, iterations: int = 5000) -> Dict[str, Any]:
        """Benchmark authorization check performance."""
        from tests.security.unit.test_boundary_security import SecurityBoundary, SecurityBoundaryType, TestSecurityContext
        
        # Create security boundary for testing
        test_boundary = SecurityBoundary(
            "performance-test-boundary",
            SecurityBoundaryType.MODULE_BOUNDARY,
            SecurityLevel.INTERNAL
        )
        
        test_context = TestSecurityContext(
            SecurityLevel.INTERNAL,
            ["read", "write", "execute"]
        )
        
        # Benchmark authorization timing
        auth_times = []
        successful_checks = 0
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            result = test_boundary.validate_access(test_context, "read")
            
            end_time = time.perf_counter()
            auth_time_ms = (end_time - start_time) * 1000
            auth_times.append(auth_time_ms)
            
            if result.get("access_granted", False):
                successful_checks += 1
        
        benchmark_result = {
            "test_type": "authorization_performance",
            "iterations": iterations,
            "successful_authorizations": successful_checks,
            "average_latency_ms": statistics.mean(auth_times),
            "median_latency_ms": statistics.median(auth_times),
            "p95_latency_ms": statistics.quantiles(auth_times, n=20)[18] if len(auth_times) >= 20 else max(auth_times),
            "max_latency_ms": max(auth_times),
            "min_latency_ms": min(auth_times),
            "meets_threshold": statistics.mean(auth_times) <= self.performance_thresholds["authorization_latency_ms"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.benchmark_results["authorization"] = benchmark_result
        return benchmark_result
    
    def benchmark_encryption_performance(self, data_size_mb: int = 10) -> Dict[str, Any]:
        """Benchmark cryptographic operations performance."""
        from tests.security.unit.test_sbom_generation import CryptographicSigner
        
        signer = CryptographicSigner(algorithm="RSA-SHA256")
        
        # Generate test data
        test_data = "X" * (data_size_mb * 1024 * 1024)  # MB of data
        
        # Benchmark signing performance
        signing_times = []
        verification_times = []
        
        for i in range(10):  # 10 iterations for encryption benchmark
            # Benchmark signing
            start_time = time.perf_counter()
            signature_metadata = signer.sign_document(test_data)
            signing_time = (time.perf_counter() - start_time) * 1000
            signing_times.append(signing_time)
            
            # Benchmark verification
            start_time = time.perf_counter()
            verification_result = signer.verify_signature(test_data, signature_metadata)
            verification_time = (time.perf_counter() - start_time) * 1000
            verification_times.append(verification_time)
        
        # Calculate throughput
        avg_signing_time_s = statistics.mean(signing_times) / 1000
        avg_verification_time_s = statistics.mean(verification_times) / 1000
        signing_throughput_mbps = data_size_mb / avg_signing_time_s if avg_signing_time_s > 0 else 0
        verification_throughput_mbps = data_size_mb / avg_verification_time_s if avg_verification_time_s > 0 else 0
        
        benchmark_result = {
            "test_type": "encryption_performance",
            "data_size_mb": data_size_mb,
            "iterations": 10,
            "signing": {
                "average_time_ms": statistics.mean(signing_times),
                "throughput_mbps": signing_throughput_mbps,
                "meets_threshold": signing_throughput_mbps >= self.performance_thresholds["encryption_throughput_mbps"]
            },
            "verification": {
                "average_time_ms": statistics.mean(verification_times),
                "throughput_mbps": verification_throughput_mbps,
                "meets_threshold": verification_throughput_mbps >= self.performance_thresholds["encryption_throughput_mbps"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.benchmark_results["encryption"] = benchmark_result
        return benchmark_result
    
    def benchmark_vulnerability_scanning_performance(self) -> Dict[str, Any]:
        """Benchmark vulnerability scanning performance."""
        from tests.security.unit.test_dependency_auditing import DependencyScanner, DependencyEcosystem
        
        scanner = DependencyScanner()
        
        # Benchmark scanning performance across ecosystems
        ecosystems = [DependencyEcosystem.PYTHON, DependencyEcosystem.JAVASCRIPT, DependencyEcosystem.RUST]
        scan_times = []
        total_dependencies = 0
        total_vulnerabilities = 0
        
        for ecosystem in ecosystems:
            start_time = time.perf_counter()
            
            scan_result = scanner.scan_ecosystem(ecosystem, f"{ecosystem.value}_test")
            
            scan_time = time.perf_counter() - start_time
            scan_times.append(scan_time)
            
            total_dependencies += scan_result.get("dependencies_scanned", 0)
            total_vulnerabilities += scan_result.get("vulnerabilities_found", 0)
        
        average_scan_time = statistics.mean(scan_times)
        
        benchmark_result = {
            "test_type": "vulnerability_scanning_performance",
            "ecosystems_scanned": len(ecosystems),
            "total_dependencies": total_dependencies,
            "total_vulnerabilities": total_vulnerabilities,
            "average_scan_time_s": average_scan_time,
            "total_scan_time_s": sum(scan_times),
            "dependencies_per_second": total_dependencies / sum(scan_times) if sum(scan_times) > 0 else 0,
            "meets_threshold": average_scan_time <= self.performance_thresholds["vulnerability_scan_time_s"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.benchmark_results["vulnerability_scanning"] = benchmark_result
        return benchmark_result
    
    def benchmark_sbom_generation_performance(self, component_count: int = 100) -> Dict[str, Any]:
        """Benchmark SBOM generation performance."""
        from tests.security.unit.test_sbom_generation import SBOMGenerator, SBOMComponent, ComponentType, CryptographicSigner
        
        signer = CryptographicSigner()
        sbom_generator = SBOMGenerator(signer=signer)
        
        # Add test components
        for i in range(component_count):
            component = SBOMComponent(
                name=f"test-component-{i}",
                version=f"1.0.{i}",
                component_type=ComponentType.LIBRARY
            )
            sbom_generator.add_component(component)
        
        # Benchmark SBOM generation
        start_time = time.perf_counter()
        sbom_content = sbom_generator.generate_sbom()
        generation_time = time.perf_counter() - start_time
        
        # Benchmark SBOM signing
        start_time = time.perf_counter()
        signed_sbom = sbom_generator.sign_sbom(sbom_content)
        signing_time = time.perf_counter() - start_time
        
        total_time = generation_time + signing_time
        
        benchmark_result = {
            "test_type": "sbom_generation_performance",
            "component_count": component_count,
            "generation_time_s": generation_time,
            "signing_time_s": signing_time,
            "total_time_s": total_time,
            "components_per_second": component_count / total_time if total_time > 0 else 0,
            "meets_threshold": total_time <= self.performance_thresholds["sbom_generation_time_s"],
            "signed_successfully": signed_sbom.get("signed", False),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.benchmark_results["sbom_generation"] = benchmark_result
        return benchmark_result
    
    def benchmark_audit_logging_overhead(self, operations_count: int = 10000) -> Dict[str, Any]:
        """Benchmark audit logging performance overhead."""
        from tests.security.unit.test_vulnerability_reporting import VulnerabilityReportingWorkflow
        
        workflow = VulnerabilityReportingWorkflow()
        
        # Benchmark operations without audit logging
        start_time = time.perf_counter()
        
        for i in range(operations_count // 2):
            # Simulate lightweight operations
            test_data = {"operation": f"test_op_{i}", "data": "test_data"}
        
        baseline_time = time.perf_counter() - start_time
        
        # Benchmark operations with audit logging
        start_time = time.perf_counter()
        
        for i in range(operations_count // 2):
            # Simulate operations with logging (using workflow's audit capabilities)
            workflow._audit_log(f"test_event_{i}", {"data": f"test_data_{i}"})
        
        logging_time = time.perf_counter() - start_time
        
        # Calculate overhead
        overhead_percent = ((logging_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
        
        benchmark_result = {
            "test_type": "audit_logging_overhead",
            "operations_count": operations_count,
            "baseline_time_s": baseline_time,
            "logging_time_s": logging_time,
            "overhead_percent": overhead_percent,
            "meets_threshold": overhead_percent <= self.performance_thresholds["audit_logging_overhead_percent"],
            "operations_per_second_baseline": (operations_count // 2) / baseline_time if baseline_time > 0 else 0,
            "operations_per_second_logging": (operations_count // 2) / logging_time if logging_time > 0 else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.benchmark_results["audit_logging"] = benchmark_result
        return benchmark_result
    
    def benchmark_concurrent_security_operations(self, thread_count: int = 10, 
                                               operations_per_thread: int = 100) -> Dict[str, Any]:
        """Benchmark security operations under concurrent load."""
        from tests.security.unit.test_boundary_security import SecurityBoundary, SecurityBoundaryType, TestSecurityContext
        
        # Create test boundary
        test_boundary = SecurityBoundary(
            "concurrent-test-boundary",
            SecurityBoundaryType.MODULE_BOUNDARY,
            SecurityLevel.INTERNAL
        )
        
        def security_operation_thread(thread_id: int) -> Dict[str, Any]:
            """Execute security operations in thread."""
            context = TestSecurityContext(SecurityLevel.INTERNAL, ["read", "write"])
            thread_times = []
            successful_operations = 0
            
            for i in range(operations_per_thread):
                start_time = time.perf_counter()
                
                result = test_boundary.validate_access(context, "read")
                
                operation_time = (time.perf_counter() - start_time) * 1000
                thread_times.append(operation_time)
                
                if result.get("access_granted", False):
                    successful_operations += 1
            
            return {
                "thread_id": thread_id,
                "operation_times": thread_times,
                "successful_operations": successful_operations,
                "average_time_ms": statistics.mean(thread_times),
                "total_time_s": sum(thread_times) / 1000
            }
        
        # Execute concurrent operations
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(security_operation_thread, i) for i in range(thread_count)]
            thread_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.perf_counter() - start_time
        
        # Aggregate results
        total_operations = thread_count * operations_per_thread
        total_successful = sum(result["successful_operations"] for result in thread_results)
        all_times = []
        for result in thread_results:
            all_times.extend(result["operation_times"])
        
        benchmark_result = {
            "test_type": "concurrent_security_operations",
            "thread_count": thread_count,
            "operations_per_thread": operations_per_thread,
            "total_operations": total_operations,
            "successful_operations": total_successful,
            "total_time_s": total_time,
            "average_operation_time_ms": statistics.mean(all_times) if all_times else 0,
            "operations_per_second": total_operations / total_time if total_time > 0 else 0,
            "concurrent_efficiency": (total_successful / total_operations * 100) if total_operations > 0 else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.benchmark_results["concurrent_operations"] = benchmark_result
        return benchmark_result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "overall_performance_score": 0,
            "benchmarks_completed": len(self.benchmark_results),
            "performance_thresholds": self.performance_thresholds,
            "benchmark_results": self.benchmark_results,
            "recommendations": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Calculate overall performance score
        threshold_met_count = 0
        total_benchmarks = 0
        
        for benchmark_name, results in self.benchmark_results.items():
            if "meets_threshold" in results:
                total_benchmarks += 1
                if results["meets_threshold"]:
                    threshold_met_count += 1
            elif benchmark_name == "encryption" and "signing" in results:
                # Handle encryption benchmark structure
                total_benchmarks += 2
                if results["signing"]["meets_threshold"]:
                    threshold_met_count += 1
                if results["verification"]["meets_threshold"]:
                    threshold_met_count += 1
        
        if total_benchmarks > 0:
            summary["overall_performance_score"] = (threshold_met_count / total_benchmarks) * 100
        
        # Generate recommendations
        if summary["overall_performance_score"] < 80:
            summary["recommendations"].append("Consider optimizing security implementations for better performance")
        
        for benchmark_name, results in self.benchmark_results.items():
            if not results.get("meets_threshold", True):
                summary["recommendations"].append(f"Optimize {benchmark_name} performance")
        
        return summary


class SecurityOverheadTest(unittest.TestCase):
    """
    Performance tests for security overhead validation.
    
    Tests that security controls maintain acceptable performance levels
    without compromising security effectiveness.
    """
    
    def setUp(self):
        """Set up performance benchmarking fixtures."""
        self.performance_benchmark = SecurityPerformanceBenchmark()
    
    def test_authentication_performance_overhead(self):
        """
        Security Contract: Authentication must complete within acceptable time limits.
        Tests authentication performance under normal load conditions.
        """
        # Act
        result = self.performance_benchmark.benchmark_authentication_performance(iterations=100)
        
        # Assert - Performance requirements
        self.assertTrue(result["meets_threshold"],
                       f"Authentication latency must be <= {self.performance_benchmark.performance_thresholds['authentication_latency_ms']}ms")
        
        self.assertLess(result["average_latency_ms"], 150,
                       "Average authentication latency should be reasonable")
        
        self.assertGreater(result["successful_authentications"], 0,
                          "Some authentication attempts must succeed")
        
        # P95 should be acceptable
        if result["p95_latency_ms"] > 0:
            self.assertLess(result["p95_latency_ms"], 200,
                           "P95 authentication latency should be acceptable")
    
    def test_authorization_performance_overhead(self):
        """
        Security Contract: Authorization checks must have minimal performance impact.
        Tests authorization performance for high-frequency operations.
        """
        # Act
        result = self.performance_benchmark.benchmark_authorization_performance(iterations=1000)
        
        # Assert - Performance requirements
        self.assertTrue(result["meets_threshold"],
                       f"Authorization latency must be <= {self.performance_benchmark.performance_thresholds['authorization_latency_ms']}ms")
        
        self.assertLess(result["average_latency_ms"], 10,
                       "Authorization should be very fast for frequent checks")
        
        self.assertEqual(result["successful_authorizations"], result["iterations"],
                        "All authorization checks should succeed in this test")
        
        # Verify consistent performance
        latency_variance = result["max_latency_ms"] - result["min_latency_ms"]
        self.assertLess(latency_variance, 50,
                       "Authorization latency should be consistent")
    
    def test_encryption_performance_throughput(self):
        """
        Security Contract: Cryptographic operations must maintain acceptable throughput.
        Tests encryption/signing performance for data processing workflows.
        """
        # Act
        result = self.performance_benchmark.benchmark_encryption_performance(data_size_mb=1)
        
        # Assert - Throughput requirements
        signing_result = result["signing"]
        verification_result = result["verification"]
        
        self.assertTrue(signing_result["meets_threshold"],
                       f"Signing throughput must be >= {self.performance_benchmark.performance_thresholds['encryption_throughput_mbps']} MB/s")
        
        self.assertTrue(verification_result["meets_threshold"],
                       f"Verification throughput must be >= {self.performance_benchmark.performance_thresholds['encryption_throughput_mbps']} MB/s")
        
        # Verification should be faster than signing
        self.assertLess(verification_result["average_time_ms"], 
                       signing_result["average_time_ms"],
                       "Signature verification should be faster than signing")
    
    def test_vulnerability_scanning_performance(self):
        """
        Security Contract: Vulnerability scanning must complete within reasonable time limits.
        Tests dependency scanning performance for CI/CD integration.
        """
        # Act
        result = self.performance_benchmark.benchmark_vulnerability_scanning_performance()
        
        # Assert - Scanning performance
        self.assertTrue(result["meets_threshold"],
                       f"Average scan time must be <= {self.performance_benchmark.performance_thresholds['vulnerability_scan_time_s']}s")
        
        self.assertGreater(result["dependencies_per_second"], 10,
                          "Should scan at least 10 dependencies per second")
        
        self.assertGreater(result["total_dependencies"], 0,
                          "Should scan some dependencies")
        
        # Total scan time should be reasonable for CI/CD
        self.assertLess(result["total_scan_time_s"], 60,
                       "Total scan time should be acceptable for CI/CD")
    
    def test_sbom_generation_performance(self):
        """
        Security Contract: SBOM generation must scale with component count.
        Tests SBOM generation performance for large dependency sets.
        """
        # Act
        result = self.performance_benchmark.benchmark_sbom_generation_performance(component_count=50)
        
        # Assert - SBOM performance
        self.assertTrue(result["meets_threshold"],
                       f"SBOM generation time must be <= {self.performance_benchmark.performance_thresholds['sbom_generation_time_s']}s")
        
        self.assertTrue(result["signed_successfully"],
                       "SBOM must be successfully signed")
        
        self.assertGreater(result["components_per_second"], 10,
                          "Should process at least 10 components per second")
        
        # Generation should be faster than signing
        self.assertLess(result["generation_time_s"], 
                       result["total_time_s"],
                       "Generation should be part of total time")
    
    def test_audit_logging_performance_overhead(self):
        """
        Security Contract: Audit logging must have minimal performance overhead.
        Tests audit logging impact on system performance.
        """
        # Act
        result = self.performance_benchmark.benchmark_audit_logging_overhead(operations_count=1000)
        
        # Assert - Logging overhead
        self.assertTrue(result["meets_threshold"],
                       f"Audit logging overhead must be <= {self.performance_benchmark.performance_thresholds['audit_logging_overhead_percent']}%")
        
        self.assertLess(result["overhead_percent"], 10,
                       "Audit logging overhead should be minimal")
        
        # Both baseline and logging operations should complete
        self.assertGreater(result["operations_per_second_baseline"], 0,
                          "Baseline operations must execute")
        self.assertGreater(result["operations_per_second_logging"], 0,
                          "Logging operations must execute")
        
        # Logging shouldn't be drastically slower
        slowdown_factor = result["operations_per_second_baseline"] / result["operations_per_second_logging"]
        self.assertLess(slowdown_factor, 2.0,
                       "Logging shouldn't cause >2x slowdown")
    
    def test_concurrent_security_operations_scalability(self):
        """
        Security Contract: Security operations must maintain performance under concurrent load.
        Tests security operation performance with concurrent access.
        """
        # Act
        result = self.performance_benchmark.benchmark_concurrent_security_operations(
            thread_count=5, operations_per_thread=100
        )
        
        # Assert - Concurrent performance
        self.assertGreaterEqual(result["concurrent_efficiency"], 95,
                               "Concurrent operations should have high success rate")
        
        self.assertGreater(result["operations_per_second"], 100,
                          "Should handle at least 100 operations per second concurrently")
        
        self.assertLess(result["average_operation_time_ms"], 100,
                       "Concurrent operations should maintain low latency")
        
        # All operations should succeed
        self.assertEqual(result["successful_operations"], result["total_operations"],
                        "All concurrent security operations should succeed")
    
    def test_security_performance_regression_detection(self):
        """
        Security Contract: Performance regressions in security controls must be detectable.
        Tests ability to detect performance degradation in security systems.
        """
        # Run multiple benchmarks
        auth_result = self.performance_benchmark.benchmark_authentication_performance(iterations=50)
        authz_result = self.performance_benchmark.benchmark_authorization_performance(iterations=500)
        
        # Get comprehensive summary
        summary = self.performance_benchmark.get_performance_summary()
        
        # Assert - Performance tracking
        self.assertGreaterEqual(summary["overall_performance_score"], 70,
                               "Overall security performance score should be acceptable")
        
        self.assertEqual(summary["benchmarks_completed"], 2,
                        "Should track completed benchmarks")
        
        self.assertIn("benchmark_results", summary,
                     "Should provide detailed benchmark results")
        
        # Check for performance recommendations
        if summary["overall_performance_score"] < 90:
            self.assertGreater(len(summary["recommendations"]), 0,
                              "Should provide performance recommendations when needed")
    
    def test_memory_usage_security_operations(self):
        """
        Security Contract: Security operations must not cause excessive memory usage.
        Tests memory efficiency of security implementations.
        """
        import psutil
        import gc
        
        # Get baseline memory usage
        gc.collect()  # Force garbage collection
        process = psutil.Process()
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Perform memory-intensive security operations
        auth_result = self.performance_benchmark.benchmark_authentication_performance(iterations=100)
        crypto_result = self.performance_benchmark.benchmark_encryption_performance(data_size_mb=5)
        
        # Check memory usage after operations
        gc.collect()  # Force garbage collection
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = final_memory_mb - baseline_memory_mb
        
        # Assert - Memory efficiency
        self.assertLess(memory_increase_mb, 50,
                       "Security operations should not use excessive memory (>50MB)")
        
        # Memory increase should be reasonable relative to data processed
        if crypto_result:
            data_processed_mb = 5  # From benchmark parameter
            memory_efficiency = memory_increase_mb / data_processed_mb
            self.assertLess(memory_efficiency, 2.0,
                           "Memory usage should be efficient relative to data processed")
    
    def test_security_performance_under_stress(self):
        """
        Security Contract: Security controls must maintain performance under stress conditions.
        Tests security system behavior under high load conditions.
        """
        # Simulate stress conditions with high concurrent load
        stress_result = self.performance_benchmark.benchmark_concurrent_security_operations(
            thread_count=20, operations_per_thread=50
        )
        
        # Assert - Stress performance
        self.assertGreaterEqual(stress_result["concurrent_efficiency"], 90,
                               "Security should maintain high efficiency under stress")
        
        self.assertLess(stress_result["average_operation_time_ms"], 200,
                       "Operations should remain responsive under stress")
        
        # System should handle the stress load
        total_ops = stress_result["total_operations"]
        self.assertEqual(stress_result["successful_operations"], total_ops,
                        "All operations should succeed even under stress")
        
        # Performance degradation should be acceptable
        if hasattr(self, '_previous_concurrent_result'):
            previous_avg = self._previous_concurrent_result["average_operation_time_ms"]
            current_avg = stress_result["average_operation_time_ms"]
            degradation = (current_avg - previous_avg) / previous_avg * 100
            
            self.assertLess(degradation, 50,
                           "Performance degradation under stress should be <50%")
        
        # Store result for future comparison
        self._previous_concurrent_result = stress_result


if __name__ == "__main__":
    # Run tests with performance focus
    unittest.main(verbosity=2, buffer=True)