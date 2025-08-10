#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite
Tests all CODEX integration points end-to-end
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """Comprehensive integration tests for AIVillage"""
    
    def __init__(self):
        self.base_path = Path.cwd()
        self.data_dir = self.base_path / "data"
        self.test_results = []
        self.start_time = time.time()
    
    def test_service_connectivity(self) -> Dict:
        """Test basic service connectivity"""
        logger.info("Testing service connectivity...")
        
        services = [
            ("LibP2P TCP", "localhost", 4001, "tcp"),
            ("LibP2P WebSocket", "localhost", 4002, "tcp"),
            ("Digital Twin API", "localhost", 8080, "http", "/health/twin"),
            ("Evolution Metrics API", "localhost", 8081, "http", "/health/evolution"),
            ("RAG Pipeline API", "localhost", 8082, "http", "/health/rag"),
            ("Redis Cache", "localhost", 6379, "tcp"),
        ]
        
        results = []
        for name, host, port, protocol, *path in services:
            if protocol == "tcp":
                import socket
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex((host, port)) == 0
                    sock.close()
                except:
                    result = False
            elif protocol == "http":
                url = f"http://{host}:{port}{path[0] if path else '/'}"
                try:
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=2) as response:
                        result = response.status == 200
                except:
                    result = False
            else:
                result = False
            
            results.append({"service": name, "status": "PASS" if result else "FAIL"})
        
        return {
            "test": "Service Connectivity",
            "passed": sum(1 for r in results if r["status"] == "PASS"),
            "total": len(results),
            "details": results
        }
    
    def test_database_operations(self) -> Dict:
        """Test database CRUD operations"""
        logger.info("Testing database operations...")
        
        results = []
        
        # Test Evolution Metrics DB
        try:
            conn = sqlite3.connect(str(self.data_dir / "evolution_metrics.db"))
            cursor = conn.cursor()
            
            # Insert test data
            cursor.execute("""
                INSERT INTO evolution_rounds (round_number, avg_fitness, best_fitness)
                VALUES (99999, 0.85, 0.95)
            """)
            conn.commit()
            
            # Read back
            cursor.execute("SELECT * FROM evolution_rounds WHERE round_number = 99999")
            data = cursor.fetchone()
            
            # Delete test data
            cursor.execute("DELETE FROM evolution_rounds WHERE round_number = 99999")
            conn.commit()
            conn.close()
            
            results.append({
                "database": "evolution_metrics",
                "operation": "CRUD",
                "status": "PASS" if data else "FAIL"
            })
        except Exception as e:
            results.append({
                "database": "evolution_metrics",
                "operation": "CRUD",
                "status": "FAIL",
                "error": str(e)
            })
        
        # Test Digital Twin DB
        try:
            conn = sqlite3.connect(str(self.data_dir / "digital_twin.db"))
            cursor = conn.cursor()
            
            # Get schema info
            cursor.execute("SELECT sql FROM sqlite_master WHERE name = 'learning_profiles'")
            schema = cursor.fetchone()
            
            # Insert based on actual schema
            if "user_id" in str(schema):
                cursor.execute("""
                    INSERT INTO learning_profiles (user_id, learning_style)
                    VALUES ('test_99999', 'visual')
                """)
            else:
                # Try with minimal columns
                cursor.execute("""
                    INSERT INTO learning_profiles (id)
                    VALUES (99999)
                """)
            
            conn.commit()
            
            # Clean up
            if "user_id" in str(schema):
                cursor.execute("DELETE FROM learning_profiles WHERE user_id = 'test_99999'")
            else:
                cursor.execute("DELETE FROM learning_profiles WHERE id = 99999")
            
            conn.commit()
            conn.close()
            
            results.append({
                "database": "digital_twin",
                "operation": "CRUD",
                "status": "PASS"
            })
        except Exception as e:
            results.append({
                "database": "digital_twin",
                "operation": "CRUD",
                "status": "FAIL",
                "error": str(e)
            })
        
        return {
            "test": "Database Operations",
            "passed": sum(1 for r in results if r["status"] == "PASS"),
            "total": len(results),
            "details": results
        }
    
    def test_api_endpoints(self) -> Dict:
        """Test API endpoint functionality"""
        logger.info("Testing API endpoints...")
        
        results = []
        
        # Test Evolution Metrics API
        try:
            url = "http://localhost:8081/metrics/latest"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read())
                results.append({
                    "api": "Evolution Metrics",
                    "endpoint": "/metrics/latest",
                    "status": "PASS" if "round" in data else "FAIL"
                })
        except:
            results.append({
                "api": "Evolution Metrics",
                "endpoint": "/metrics/latest",
                "status": "FAIL"
            })
        
        # Test RAG Pipeline API
        try:
            url = "http://localhost:8082/query"
            data = json.dumps({"query": "test query"}).encode()
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=2) as response:
                result = json.loads(response.read())
                results.append({
                    "api": "RAG Pipeline",
                    "endpoint": "/query",
                    "status": "PASS" if "results" in result else "FAIL"
                })
        except:
            results.append({
                "api": "RAG Pipeline",
                "endpoint": "/query",
                "status": "FAIL"
            })
        
        # Test Digital Twin API
        try:
            url = "http://localhost:8080/profile/test_user"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read())
                results.append({
                    "api": "Digital Twin",
                    "endpoint": "/profile/{id}",
                    "status": "PASS" if "profile_id" in data else "FAIL"
                })
        except:
            results.append({
                "api": "Digital Twin",
                "endpoint": "/profile/{id}",
                "status": "FAIL"
            })
        
        return {
            "test": "API Endpoints",
            "passed": sum(1 for r in results if r["status"] == "PASS"),
            "total": len(results),
            "details": results
        }
    
    def test_concurrent_operations(self) -> Dict:
        """Test concurrent database and API operations"""
        logger.info("Testing concurrent operations...")
        
        import threading
        
        results = {"success": 0, "failed": 0}
        lock = threading.Lock()
        
        def worker():
            try:
                # Concurrent DB write
                conn = sqlite3.connect(str(self.data_dir / "evolution_metrics.db"))
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO fitness_metrics (round_id, agent_id, metric_name, metric_value)
                    VALUES (1, 'test_agent', 'test_metric', 0.5)
                """)
                conn.commit()
                conn.close()
                
                with lock:
                    results["success"] += 1
            except:
                with lock:
                    results["failed"] += 1
        
        # Run 10 concurrent operations
        threads = []
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        return {
            "test": "Concurrent Operations",
            "passed": results["success"],
            "total": results["success"] + results["failed"],
            "details": [results]
        }
    
    def test_data_persistence(self) -> Dict:
        """Test data persistence across restarts"""
        logger.info("Testing data persistence...")
        
        try:
            # Write test data
            conn = sqlite3.connect(str(self.data_dir / "evolution_metrics.db"))
            cursor = conn.cursor()
            
            test_value = time.time()
            cursor.execute("""
                INSERT INTO kpi_tracking (kpi_name, kpi_value)
                VALUES ('persistence_test', ?)
            """, (test_value,))
            conn.commit()
            conn.close()
            
            # Simulate restart by reopening
            conn = sqlite3.connect(str(self.data_dir / "evolution_metrics.db"))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT kpi_value FROM kpi_tracking 
                WHERE kpi_name = 'persistence_test'
                ORDER BY id DESC LIMIT 1
            """)
            result = cursor.fetchone()
            
            # Clean up
            cursor.execute("DELETE FROM kpi_tracking WHERE kpi_name = 'persistence_test'")
            conn.commit()
            conn.close()
            
            passed = result and abs(result[0] - test_value) < 0.001
            
            return {
                "test": "Data Persistence",
                "passed": 1 if passed else 0,
                "total": 1,
                "details": [{"status": "PASS" if passed else "FAIL"}]
            }
        except Exception as e:
            return {
                "test": "Data Persistence",
                "passed": 0,
                "total": 1,
                "details": [{"status": "FAIL", "error": str(e)}]
            }
    
    def test_performance_metrics(self) -> Dict:
        """Test performance metrics"""
        logger.info("Testing performance metrics...")
        
        results = []
        
        # Database query performance
        try:
            conn = sqlite3.connect(str(self.data_dir / "evolution_metrics.db"))
            cursor = conn.cursor()
            
            start = time.time()
            cursor.execute("SELECT COUNT(*) FROM fitness_metrics")
            _ = cursor.fetchone()
            db_latency = (time.time() - start) * 1000
            
            results.append({
                "metric": "DB Query Latency",
                "value": f"{db_latency:.2f}ms",
                "target": "<10ms",
                "status": "PASS" if db_latency < 10 else "WARN"
            })
            
            conn.close()
        except:
            results.append({
                "metric": "DB Query Latency",
                "status": "FAIL"
            })
        
        # API response time
        try:
            start = time.time()
            req = urllib.request.Request("http://localhost:8081/health/evolution")
            with urllib.request.urlopen(req, timeout=2) as response:
                _ = response.read()
            api_latency = (time.time() - start) * 1000
            
            results.append({
                "metric": "API Response Time",
                "value": f"{api_latency:.2f}ms",
                "target": "<100ms",
                "status": "PASS" if api_latency < 100 else "WARN"
            })
        except:
            results.append({
                "metric": "API Response Time",
                "status": "FAIL"
            })
        
        return {
            "test": "Performance Metrics",
            "passed": sum(1 for r in results if r["status"] == "PASS"),
            "total": len(results),
            "details": results
        }
    
    def run_all_tests(self) -> Tuple[List[Dict], Dict]:
        """Run all integration tests"""
        print("\n" + "="*80)
        print("RUNNING INTEGRATION TEST SUITE")
        print("="*80)
        
        tests = [
            self.test_service_connectivity,
            self.test_database_operations,
            self.test_api_endpoints,
            self.test_concurrent_operations,
            self.test_data_persistence,
            self.test_performance_metrics
        ]
        
        results = []
        total_passed = 0
        total_tests = 0
        
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                total_passed += result["passed"]
                total_tests += result["total"]
                
                # Print progress
                status = "PASS" if result["passed"] == result["total"] else "PARTIAL" if result["passed"] > 0 else "FAIL"
                print(f"\n{result['test']}: [{status}] {result['passed']}/{result['total']} passed")
                
                if result.get("details"):
                    for detail in result["details"][:3]:  # Show first 3 details
                        if isinstance(detail, dict):
                            print(f"  - {detail}")
                
            except Exception as e:
                logger.error(f"Test failed: {e}")
                results.append({
                    "test": test_func.__name__,
                    "passed": 0,
                    "total": 1,
                    "error": str(e)
                })
                total_tests += 1
        
        # Calculate summary
        elapsed = time.time() - self.start_time
        summary = {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_tests - total_passed,
            "pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "execution_time": f"{elapsed:.2f}s",
            "timestamp": datetime.now().isoformat()
        }
        
        return results, summary
    
    def print_summary(self, results: List[Dict], summary: Dict):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        print(f"\nTotal Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"Execution Time: {summary['execution_time']}")
        
        if summary['pass_rate'] >= 90:
            print("\n[SUCCESS] Integration tests passed!")
        elif summary['pass_rate'] >= 70:
            print("\n[WARNING] Some integration tests failed")
        else:
            print("\n[FAILURE] Major integration issues detected")
        
        # Show failed tests
        if summary['failed'] > 0:
            print("\nFailed Tests:")
            for result in results:
                if result["passed"] < result["total"]:
                    print(f"  - {result['test']}: {result['passed']}/{result['total']} passed")
    
    def save_results(self, results: List[Dict], summary: Dict):
        """Save test results to file"""
        report = {
            "summary": summary,
            "results": results
        }
        
        output_file = Path("integration_test_results.json")
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nResults saved to {output_file}")


def main():
    """Main entry point"""
    suite = IntegrationTestSuite()
    
    # Check if services are running first
    print("Checking service availability...")
    connectivity = suite.test_service_connectivity()
    
    if connectivity["passed"] == 0:
        print("\n[ERROR] No services are running!")
        print("\nPlease start services first:")
        print("  1. python src/api/start_api_servers.py")
        print("  2. python src/core/p2p/start_p2p_services.py")
        print("\nThen run this test suite again.")
        return 1
    
    # Run full test suite
    results, summary = suite.run_all_tests()
    
    # Print and save results
    suite.print_summary(results, summary)
    suite.save_results(results, summary)
    
    return 0 if summary['pass_rate'] >= 70 else 1


if __name__ == "__main__":
    sys.exit(main())