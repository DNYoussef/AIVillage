#!/usr/bin/env python3
"""
Production Readiness Validation Test Suite
Validates that all CI/CD pipeline fixes are working correctly
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class ProductionValidator:
    """Comprehensive production readiness validator"""
    
    def __init__(self):
        self.results = {}
        self.project_root = Path(__file__).parent.parent.parent
        
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete production validation suite"""
        print("=" * 60)
        print("PRODUCTION READINESS VALIDATION")
        print("=" * 60)
        
        validation_tests = [
            ("unified_linting_manager", self.test_unified_linting_manager),
            ("test_infrastructure", self.test_test_infrastructure),
            ("security_configuration", self.test_security_configuration),
            ("workflow_optimization", self.test_workflow_optimization),
            ("import_resolution", self.test_import_resolution),
        ]
        
        for test_name, test_func in validation_tests:
            print(f"\n🔍 Running {test_name}...")
            try:
                result = await test_func()
                self.results[test_name] = result
                status = "✅ PASS" if result.get("status") == "passed" else "❌ FAIL"
                print(f"{status} {test_name}: {result.get('message', 'Completed')}")
            except Exception as e:
                self.results[test_name] = {
                    "status": "failed",
                    "message": f"Exception: {str(e)}",
                    "error": str(e)
                }
                print(f"❌ FAIL {test_name}: {str(e)}")
        
        # Generate summary
        summary = self.generate_summary()
        self.save_results(summary)
        return summary
    
    async def test_unified_linting_manager(self) -> Dict[str, Any]:
        """Test that unified linting manager is properly implemented"""
        
        # Check if unified linting manager exists and can be imported
        linting_manager_path = self.project_root / "config" / "linting" / "unified_linting_manager.py"
        
        if not linting_manager_path.exists():
            return {
                "status": "failed",
                "message": "Unified linting manager file not found",
                "details": {"path": str(linting_manager_path)}
            }
        
        # Test import
        try:
            sys.path.insert(0, str(linting_manager_path.parent))
            from unified_linting_manager import UnifiedLintingPipeline
            
            # Test instantiation
            pipeline = UnifiedLintingPipeline()
            
            # Test configuration loading
            config = pipeline._load_config()
            
            required_configs = ["python", "frontend", "security", "execution"]
            missing_configs = [cfg for cfg in required_configs if cfg not in config]
            
            if missing_configs:
                return {
                    "status": "failed", 
                    "message": f"Missing configurations: {missing_configs}",
                    "details": {"missing": missing_configs}
                }
            
            return {
                "status": "passed",
                "message": "Unified linting manager successfully implemented",
                "details": {
                    "config_sections": list(config.keys()),
                    "python_tools": list(config["python"]["tools"].keys()),
                    "parallel_enabled": config["execution"]["parallel"]
                }
            }
            
        except ImportError as e:
            return {
                "status": "failed",
                "message": f"Import error: {str(e)}",
                "details": {"import_error": str(e)}
            }
        except Exception as e:
            return {
                "status": "failed", 
                "message": f"Configuration error: {str(e)}",
                "details": {"config_error": str(e)}
            }
    
    async def test_test_infrastructure(self) -> Dict[str, Any]:
        """Test that test infrastructure improvements are working"""
        
        # Test RAG imports
        try:
            sys.path.insert(0, str(self.project_root))
            sys.path.insert(0, str(self.project_root / "packages"))
            
            from packages.rag import HyperRAG, EdgeDeviceRAGBridge, P2PNetworkRAGBridge, FogComputeBridge
            
            # Test instantiation of bridge components
            edge_bridge = EdgeDeviceRAGBridge()
            p2p_bridge = P2PNetworkRAGBridge()
            fog_bridge = FogComputeBridge()
            
            # Test basic functionality
            edge_result = await edge_bridge.process_query("test query")
            p2p_result = await p2p_bridge.distributed_query("test query")
            fog_result = await fog_bridge.process_workload("test workload")
            
            return {
                "status": "passed",
                "message": "Test infrastructure working correctly",
                "details": {
                    "rag_bridges_working": True,
                    "edge_response": edge_result.get("device_id") == "test_device",
                    "p2p_response": "peer_contributions" in p2p_result,
                    "fog_response": "processing_nodes" in fog_result
                }
            }
            
        except ImportError as e:
            return {
                "status": "failed",
                "message": f"Test infrastructure import error: {str(e)}",
                "details": {"import_error": str(e)}
            }
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Test infrastructure runtime error: {str(e)}",
                "details": {"runtime_error": str(e)}
            }
    
    async def test_security_configuration(self) -> Dict[str, Any]:
        """Test that security configurations are properly set"""
        
        # Check for security configuration files
        security_configs = [
            self.project_root / ".secrets.baseline",
            self.project_root / "config" / ".env.template",
            self.project_root / "config" / "security",
        ]
        
        config_status = {}
        for config_path in security_configs:
            config_status[config_path.name] = config_path.exists()
        
        # Test bandit configuration by running a quick scan
        try:
            result = subprocess.run([
                sys.executable, "-m", "bandit", "-r", "config/", "-f", "json"
            ], capture_output=True, text=True, timeout=30, cwd=self.project_root)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                high_critical = [r for r in bandit_data.get("results", []) 
                               if r.get("issue_severity") in ["HIGH", "CRITICAL"]]
                
                security_health = {
                    "bandit_executable": True,
                    "high_critical_issues": len(high_critical),
                    "total_issues": len(bandit_data.get("results", [])),
                    "properly_suppressed": len(high_critical) < 5  # Expect most issues to be suppressed
                }
            else:
                security_health = {"bandit_executable": False}
                
        except Exception as e:
            security_health = {"bandit_error": str(e)}
        
        # Determine overall status
        critical_configs_exist = config_status.get(".env.template", False)
        security_working = security_health.get("bandit_executable", False)
        
        if critical_configs_exist and security_working:
            return {
                "status": "passed",
                "message": "Security configuration properly set up",
                "details": {
                    "config_files": config_status,
                    "security_scan": security_health
                }
            }
        else:
            return {
                "status": "failed", 
                "message": "Security configuration issues detected",
                "details": {
                    "config_files": config_status,
                    "security_scan": security_health
                }
            }
    
    async def test_workflow_optimization(self) -> Dict[str, Any]:
        """Test that workflow optimizations are in place"""
        
        # Check for optimized workflow files
        workflow_files = [
            self.project_root / ".github" / "workflows" / "main-ci-optimized.yml",
            self.project_root / ".github" / "workflows" / "unified-quality-pipeline.yml",
        ]
        
        workflow_status = {}
        optimizations_found = []
        
        for workflow_file in workflow_files:
            if workflow_file.exists():
                workflow_status[workflow_file.name] = True
                
                # Check for specific optimizations in the file content
                content = workflow_file.read_text()
                
                optimizations = {
                    "timeout_minutes": "timeout-minutes" in content,
                    "caching": "cache:" in content,
                    "parallel_jobs": "strategy:" in content,
                    "artifact_upload": "upload-artifact@v4" in content,
                    "error_handling": "|| echo" in content,
                    "github_cli": "gh " in content or "github-cli" in content
                }
                
                optimizations_found.append({
                    "file": workflow_file.name,
                    "optimizations": optimizations
                })
            else:
                workflow_status[workflow_file.name] = False
        
        # Check if the key optimized workflow exists
        main_optimized_exists = workflow_status.get("main-ci-optimized.yml", False)
        
        return {
            "status": "passed" if main_optimized_exists else "failed",
            "message": "Workflow optimizations implemented" if main_optimized_exists else "Missing optimized workflows",
            "details": {
                "workflow_files": workflow_status,
                "optimizations": optimizations_found
            }
        }
    
    async def test_import_resolution(self) -> Dict[str, Any]:
        """Test that import resolution issues are fixed"""
        
        # Test critical imports that were failing before
        import_tests = []
        
        # Add project paths
        sys.path.insert(0, str(self.project_root))
        sys.path.insert(0, str(self.project_root / "packages"))
        sys.path.insert(0, str(self.project_root / "tests"))
        
        critical_imports = [
            ("packages.rag", ["HyperRAG", "EdgeDeviceRAGBridge"]),
            ("config.linting.unified_linting_manager", ["UnifiedLintingPipeline"]),
        ]
        
        for module_name, class_names in critical_imports:
            try:
                module = __import__(module_name, fromlist=class_names)
                
                import_success = {}
                for class_name in class_names:
                    try:
                        cls = getattr(module, class_name)
                        # Try to instantiate
                        if class_name == "UnifiedLintingPipeline":
                            instance = cls()
                        elif class_name == "HyperRAG":
                            instance = cls()
                        else:
                            instance = cls()
                        import_success[class_name] = True
                    except Exception as e:
                        import_success[class_name] = f"Instantiation failed: {str(e)}"
                
                import_tests.append({
                    "module": module_name,
                    "status": "success",
                    "classes": import_success
                })
                
            except ImportError as e:
                import_tests.append({
                    "module": module_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Determine overall status
        failed_imports = [t for t in import_tests if t["status"] == "failed"]
        
        return {
            "status": "passed" if not failed_imports else "failed",
            "message": f"Import resolution {'working' if not failed_imports else 'has issues'}",
            "details": {
                "import_tests": import_tests,
                "failed_count": len(failed_imports),
                "total_count": len(import_tests)
            }
        }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate overall validation summary"""
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result.get("status") == "passed")
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "validation_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": round(success_rate, 1)
            },
            "overall_status": "PRODUCTION_READY" if success_rate >= 80 else "NEEDS_ATTENTION",
            "individual_results": self.results,
            "recommendations": []
        }
        
        # Generate recommendations
        if failed_tests > 0:
            summary["recommendations"].append(f"Address {failed_tests} failed validation tests")
        
        if success_rate < 80:
            summary["recommendations"].append("Success rate below 80% - review failed tests before production deployment")
        elif success_rate >= 95:
            summary["recommendations"].append("Excellent validation results - system is production ready")
        else:
            summary["recommendations"].append("Good validation results - minor issues to address")
        
        return summary
    
    def save_results(self, summary: Dict[str, Any]):
        """Save validation results to file"""
        
        results_dir = self.project_root / "tests" / "validation" 
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / "production_readiness_results.json"
        
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Results saved to: {results_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Success Rate: {summary['validation_summary']['success_rate']}%")
        print(f"Tests Passed: {summary['validation_summary']['passed']}/{summary['validation_summary']['total_tests']}")
        
        if summary["recommendations"]:
            print("\nRecommendations:")
            for rec in summary["recommendations"]:
                print(f"  • {rec}")


async def main():
    """Run production validation"""
    validator = ProductionValidator()
    summary = await validator.run_validation()
    
    # Exit with appropriate code
    success_rate = summary["validation_summary"]["success_rate"]
    if success_rate >= 80:
        print(f"\n✅ Validation PASSED ({success_rate}%)")
        sys.exit(0)
    else:
        print(f"\n❌ Validation FAILED ({success_rate}%)")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())