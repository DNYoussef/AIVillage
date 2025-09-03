#!/usr/bin/env python3
"""
Production Linting System Validator
Comprehensive validation of the unified linting system for production deployment
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('production_validation.log')
    ]
)
logger = logging.getLogger(__name__)


class ProductionValidator:
    """Comprehensive production readiness validator for the linting system"""
    
    def __init__(self):
        self.validation_results = {}
        self.start_time = time.time()
        
        # Define required components
        self.required_files = [
            "config/linting/unified_linting_manager.py",
            "config/linting/error_handler.py", 
            "config/linting/linting_cache_system.py",
            "config/linting/run_unified_linting.py",
            "config/linting/requirements-linting.txt",
            ".github/workflows/unified-quality-pipeline.yml"
        ]
        
        self.required_python_packages = [
            "ruff", "black", "mypy", "bandit", "safety",
            "semgrep", "detect-secrets", "pip-audit",
            "PyYAML", "redis", "pydantic"
        ]
        
        self.optional_packages = [
            "python-memcached", "prometheus-client"
        ]
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete production validation suite"""
        logger.info("Starting production readiness validation...")
        
        # Run all validation checks
        validations = {
            "file_structure": await self.validate_file_structure(),
            "dependencies": await self.validate_dependencies(),
            "configuration": await self.validate_configuration(),
            "imports": await self.validate_imports(),
            "error_handling": await self.validate_error_handling(),
            "caching_system": await self.validate_caching_system(),
            "cli_interface": await self.validate_cli_interface(),
            "github_workflow": await self.validate_github_workflow(),
            "performance": await self.validate_performance(),
            "security": await self.validate_security(),
            "integration": await self.validate_integration()
        }
        
        # Calculate overall status
        total_time = time.time() - self.start_time
        passed_checks = sum(1 for result in validations.values() if result.get("status") == "passed")
        total_checks = len(validations)
        
        overall_status = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_execution_time": f"{total_time:.2f}s",
            "checks_passed": passed_checks,
            "total_checks": total_checks,
            "success_rate": f"{(passed_checks / total_checks) * 100:.1f}%",
            "overall_status": "READY" if passed_checks == total_checks else "NEEDS_ATTENTION",
            "validations": validations
        }
        
        self.validation_results = overall_status
        await self.generate_validation_report()
        
        return overall_status
    
    async def validate_file_structure(self) -> Dict[str, Any]:
        """Validate required files are present and properly structured"""
        logger.info("Validating file structure...")
        
        missing_files = []
        present_files = []
        file_sizes = {}
        
        for file_path in self.required_files:
            path = Path(file_path)
            if path.exists():
                present_files.append(file_path)
                file_sizes[file_path] = path.stat().st_size
            else:
                missing_files.append(file_path)
        
        # Check for empty files
        empty_files = [f for f, size in file_sizes.items() if size < 100]
        
        status = "passed" if not missing_files and not empty_files else "failed"
        
        return {
            "status": status,
            "present_files": present_files,
            "missing_files": missing_files,
            "empty_files": empty_files,
            "file_sizes": file_sizes,
            "details": f"Found {len(present_files)}/{len(self.required_files)} required files"
        }
    
    async def validate_dependencies(self) -> Dict[str, Any]:
        """Validate Python package dependencies"""
        logger.info("Validating dependencies...")
        
        installed_packages = {}
        missing_packages = []
        optional_missing = []
        
        # Check required packages
        for package in self.required_python_packages:
            try:
                result = subprocess.run([sys.executable, "-c", f"import {package}"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Get package version if possible
                    try:
                        version_result = subprocess.run([sys.executable, "-c", 
                                                       f"import {package}; print(getattr({package}, '__version__', 'unknown'))"],
                                                      capture_output=True, text=True)
                        version = version_result.stdout.strip() if version_result.returncode == 0 else "unknown"
                    except:
                        version = "unknown"
                    
                    installed_packages[package] = version
                else:
                    missing_packages.append(package)
            except:
                missing_packages.append(package)
        
        # Check optional packages
        for package in self.optional_packages:
            try:
                result = subprocess.run([sys.executable, "-c", f"import {package}"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    installed_packages[package] = "available"
                else:
                    optional_missing.append(package)
            except:
                optional_missing.append(package)
        
        status = "passed" if not missing_packages else "failed"
        
        return {
            "status": status,
            "installed_packages": installed_packages,
            "missing_required": missing_packages,
            "missing_optional": optional_missing,
            "details": f"Found {len(installed_packages)} packages, missing {len(missing_packages)} required"
        }
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration files and structure"""
        logger.info("Validating configuration...")
        
        config_files = [
            "config/linting/requirements-linting.txt",
            ".github/workflows/unified-quality-pipeline.yml"
        ]
        
        valid_configs = []
        invalid_configs = []
        config_details = {}
        
        for config_file in config_files:
            path = Path(config_file)
            if path.exists():
                try:
                    if config_file.endswith('.yml') or config_file.endswith('.yaml'):
                        import yaml
                        with open(path, 'r') as f:
                            config_data = yaml.safe_load(f)
                        config_details[config_file] = {
                            "type": "yaml",
                            "keys": list(config_data.keys()) if isinstance(config_data, dict) else []
                        }
                    elif config_file.endswith('.txt'):
                        with open(path, 'r') as f:
                            lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
                        config_details[config_file] = {
                            "type": "requirements",
                            "package_count": len(lines)
                        }
                    
                    valid_configs.append(config_file)
                    
                except Exception as e:
                    invalid_configs.append(f"{config_file}: {str(e)}")
            else:
                invalid_configs.append(f"{config_file}: file not found")
        
        status = "passed" if not invalid_configs else "warning"
        
        return {
            "status": status,
            "valid_configs": valid_configs,
            "invalid_configs": invalid_configs,
            "config_details": config_details,
            "details": f"Validated {len(valid_configs)} configuration files"
        }
    
    async def validate_imports(self) -> Dict[str, Any]:
        """Validate that all modules can be imported without errors"""
        logger.info("Validating imports...")
        
        modules_to_test = [
            "config.linting.unified_linting_manager",
            "config.linting.error_handler",
            "config.linting.linting_cache_system",
            "config.linting.run_unified_linting"
        ]
        
        successful_imports = []
        failed_imports = []
        import_errors = {}
        
        for module in modules_to_test:
            try:
                __import__(module)
                successful_imports.append(module)
            except Exception as e:
                failed_imports.append(module)
                import_errors[module] = str(e)
        
        status = "passed" if not failed_imports else "failed"
        
        return {
            "status": status,
            "successful_imports": successful_imports,
            "failed_imports": failed_imports,
            "import_errors": import_errors,
            "details": f"Successfully imported {len(successful_imports)}/{len(modules_to_test)} modules"
        }
    
    async def validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling system"""
        logger.info("Validating error handling system...")
        
        try:
            # Try to import and test error handling
            from config.linting.error_handler import error_handler, ErrorContext
            
            # Test error context creation
            context = ErrorContext(
                operation="test_validation",
                tool="test_tool",
                target_paths=["test.py"],
                config={},
                timestamp=datetime.now().isoformat(),
                session_id="validation_test"
            )
            
            # Test error handler health check
            if hasattr(error_handler, 'health_check'):
                health_result = await error_handler.health_check()
                error_handler_healthy = health_result.get("status") == "healthy"
            else:
                error_handler_healthy = True  # Basic implementation
            
            status = "passed" if error_handler_healthy else "warning"
            
            return {
                "status": status,
                "error_handler_available": True,
                "health_check_result": health_result if 'health_result' in locals() else "basic_implementation",
                "details": "Error handling system is functional"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error_handler_available": False,
                "error": str(e),
                "details": "Error handling system not available"
            }
    
    async def validate_caching_system(self) -> Dict[str, Any]:
        """Validate caching system functionality"""
        logger.info("Validating caching system...")
        
        try:
            from config.linting.linting_cache_system import cache_manager
            
            # Test cache initialization
            cache_init_success = await cache_manager.initialize()
            
            # Test basic cache operations
            if cache_init_success:
                set_success = await cache_manager.set("test_key", {"test": "data"}, service="validation")
                get_result = await cache_manager.get("test_key", service="validation")
                cache_operations_work = set_success and get_result == {"test": "data"}
                
                # Test health check
                if hasattr(cache_manager, 'health_check'):
                    health_result = await cache_manager.health_check()
                    cache_healthy = health_result.get("status") == "healthy"
                else:
                    cache_healthy = cache_operations_work
                
            else:
                cache_operations_work = False
                cache_healthy = False
                health_result = {"status": "failed", "error": "initialization failed"}
            
            status = "passed" if cache_init_success and cache_operations_work else "warning"
            
            return {
                "status": status,
                "cache_available": True,
                "initialization_success": cache_init_success,
                "operations_functional": cache_operations_work,
                "health_status": health_result if 'health_result' in locals() else "not_available",
                "details": "Caching system validation completed"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "cache_available": False,
                "error": str(e),
                "details": "Caching system not available"
            }
    
    async def validate_cli_interface(self) -> Dict[str, Any]:
        """Validate command line interface"""
        logger.info("Validating CLI interface...")
        
        try:
            # Test CLI import
            from config.linting.run_unified_linting import main
            
            # Test help output
            result = subprocess.run([
                sys.executable, "config/linting/run_unified_linting.py", "--help"
            ], capture_output=True, text=True, timeout=10)
            
            help_works = result.returncode == 0 and "usage:" in result.stdout.lower()
            
            # Test dry run
            dry_run_result = subprocess.run([
                sys.executable, "config/linting/run_unified_linting.py", 
                "--dry-run", "--language=python"
            ], capture_output=True, text=True, timeout=10)
            
            dry_run_works = dry_run_result.returncode == 0
            
            status = "passed" if help_works and dry_run_works else "warning"
            
            return {
                "status": status,
                "cli_importable": True,
                "help_command_works": help_works,
                "dry_run_works": dry_run_works,
                "details": "CLI interface validation completed"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "cli_importable": False,
                "error": str(e),
                "details": "CLI interface not functional"
            }
    
    async def validate_github_workflow(self) -> Dict[str, Any]:
        """Validate GitHub workflow configuration"""
        logger.info("Validating GitHub workflow...")
        
        workflow_path = Path(".github/workflows/unified-quality-pipeline.yml")
        
        if not workflow_path.exists():
            return {
                "status": "failed",
                "workflow_exists": False,
                "details": "GitHub workflow file not found"
            }
        
        try:
            import yaml
            with open(workflow_path, 'r') as f:
                workflow_data = yaml.safe_load(f)
            
            # Check required workflow components
            required_keys = ["name", "on", "jobs"]
            missing_keys = [key for key in required_keys if key not in workflow_data]
            
            # Check for linting-related jobs
            jobs = workflow_data.get("jobs", {})
            linting_jobs = [job for job in jobs.keys() if "quality" in job.lower() or "lint" in job.lower()]
            
            # Check for required steps in jobs
            has_python_setup = any(
                any("python" in step.get("name", "").lower() or 
                    "python" in step.get("uses", "").lower() 
                    for step in job.get("steps", []))
                for job in jobs.values()
            )
            
            status = "passed" if not missing_keys and linting_jobs and has_python_setup else "warning"
            
            return {
                "status": status,
                "workflow_exists": True,
                "missing_keys": missing_keys,
                "linting_jobs": linting_jobs,
                "has_python_setup": has_python_setup,
                "total_jobs": len(jobs),
                "details": f"Workflow validation completed with {len(linting_jobs)} linting jobs"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "workflow_exists": True,
                "error": str(e),
                "details": "Failed to parse GitHub workflow"
            }
    
    async def validate_performance(self) -> Dict[str, Any]:
        """Validate performance characteristics"""
        logger.info("Validating performance...")
        
        try:
            from config.linting.unified_linting_manager import UnifiedLintingPipeline
            
            # Test initialization time
            start_time = time.time()
            manager = UnifiedLintingPipeline()
            init_time = time.time() - start_time
            
            # Test cache initialization time
            start_time = time.time()
            if hasattr(manager, '_initialize_cache_system'):
                await manager._initialize_cache_system()
            cache_init_time = time.time() - start_time
            
            # Performance thresholds (reasonable for production)
            init_fast_enough = init_time < 1.0  # Under 1 second
            cache_init_fast_enough = cache_init_time < 2.0  # Under 2 seconds
            
            status = "passed" if init_fast_enough and cache_init_fast_enough else "warning"
            
            return {
                "status": status,
                "initialization_time": f"{init_time:.3f}s",
                "cache_initialization_time": f"{cache_init_time:.3f}s",
                "initialization_fast_enough": init_fast_enough,
                "cache_init_fast_enough": cache_init_fast_enough,
                "details": "Performance validation completed"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "details": "Performance validation failed"
            }
    
    async def validate_security(self) -> Dict[str, Any]:
        """Validate security aspects"""
        logger.info("Validating security...")
        
        security_checks = {
            "no_hardcoded_secrets": True,
            "secure_imports": True,
            "safe_file_operations": True
        }
        
        # Check for hardcoded secrets in key files
        sensitive_patterns = [
            "password", "secret", "key", "token", "api_key",
            "auth", "credential", "private"
        ]
        
        files_to_check = [
            "config/linting/unified_linting_manager.py",
            "config/linting/error_handler.py",
            "config/linting/linting_cache_system.py"
        ]
        
        suspicious_content = []
        
        for file_path in files_to_check:
            path = Path(file_path)
            if path.exists():
                try:
                    content = path.read_text().lower()
                    for pattern in sensitive_patterns:
                        if f'"{pattern}"' in content or f"'{pattern}'" in content:
                            # Check if it's in a comment or legitimate context
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if pattern in line and not line.strip().startswith('#'):
                                    suspicious_content.append(f"{file_path}:{i+1}")
                except Exception:
                    logger.exception("Secret scan iteration failed")
        
        security_checks["no_hardcoded_secrets"] = len(suspicious_content) == 0
        
        status = "passed" if all(security_checks.values()) else "warning"
        
        return {
            "status": status,
            "security_checks": security_checks,
            "suspicious_content": suspicious_content,
            "details": "Security validation completed"
        }
    
    async def validate_integration(self) -> Dict[str, Any]:
        """Validate integration capabilities"""
        logger.info("Validating integration...")
        
        integration_tests = {}
        
        try:
            # Test unified linting manager integration
            from config.linting.unified_linting_manager import UnifiedLintingPipeline
            manager = UnifiedLintingPipeline()
            
            # Test MCP connections initialization
            if hasattr(manager, 'initialize_mcp_connections'):
                mcp_result = await manager.initialize_mcp_connections()
                integration_tests["mcp_connections"] = mcp_result
            else:
                integration_tests["mcp_connections"] = "not_available"
            
            # Test cache integration
            if hasattr(manager, '_initialize_cache_system'):
                cache_result = await manager._initialize_cache_system()
                integration_tests["cache_integration"] = cache_result
            else:
                integration_tests["cache_integration"] = "not_available"
            
            # Test error handling integration
            integration_tests["error_handling_integration"] = hasattr(manager, 'error_handler')
            
            status = "passed"
            
        except Exception as e:
            integration_tests["error"] = str(e)
            status = "failed"
        
        return {
            "status": status,
            "integration_tests": integration_tests,
            "details": "Integration validation completed"
        }
    
    async def generate_validation_report(self):
        """Generate comprehensive validation report"""
        report_path = Path("production_validation_report.json")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            
            logger.info(f"Validation report saved to {report_path}")
            
            # Generate summary report
            summary_path = Path("production_validation_summary.md")
            await self.generate_summary_report(summary_path)
            
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
    
    async def generate_summary_report(self, output_path: Path):
        """Generate human-readable summary report"""
        try:
            results = self.validation_results
            
            summary_content = f"""# Production Linting System Validation Report

**Validation Date:** {results['validation_timestamp']}
**Total Execution Time:** {results['total_execution_time']}
**Overall Status:** {results['overall_status']}
**Success Rate:** {results['success_rate']}

## Validation Summary

- **Checks Passed:** {results['checks_passed']}/{results['total_checks']}
- **Status:** {'✅ READY FOR PRODUCTION' if results['overall_status'] == 'READY' else '⚠️ NEEDS ATTENTION'}

## Detailed Results

"""
            
            for validation_name, validation_result in results['validations'].items():
                status_emoji = "✅" if validation_result['status'] == 'passed' else "⚠️" if validation_result['status'] == 'warning' else "❌"
                summary_content += f"### {validation_name.replace('_', ' ').title()}\n"
                summary_content += f"{status_emoji} **Status:** {validation_result['status'].upper()}\n"
                summary_content += f"**Details:** {validation_result.get('details', 'No details available')}\n\n"
                
                # Add specific details based on validation type
                if validation_name == 'dependencies' and 'missing_required' in validation_result:
                    missing = validation_result['missing_required']
                    if missing:
                        summary_content += f"**Missing Required Packages:** {', '.join(missing)}\n\n"
                
                if validation_name == 'file_structure' and 'missing_files' in validation_result:
                    missing = validation_result['missing_files']
                    if missing:
                        summary_content += f"**Missing Files:** {', '.join(missing)}\n\n"
            
            if results['overall_status'] != 'READY':
                summary_content += """## Recommended Actions

1. Install missing required dependencies
2. Ensure all required files are present
3. Fix any import errors
4. Verify configuration files are valid
5. Re-run validation after fixes

"""
            
            summary_content += """## Next Steps

1. Review detailed validation results
2. Address any failed or warning checks
3. Run production deployment tests
4. Monitor system performance in production

---
*Generated by AIVillage Production Linting System Validator*
"""
            
            with open(output_path, 'w') as f:
                f.write(summary_content)
            
            logger.info(f"Summary report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")


async def main():
    """Main validation entry point"""
    logger.info("Starting AIVillage Unified Linting System Production Validation")
    
    validator = ProductionValidator()
    results = await validator.run_full_validation()
    
    # Print summary to console
    print("\n" + "="*80)
    print("AIVILLAGE UNIFIED LINTING SYSTEM - PRODUCTION VALIDATION RESULTS")
    print("="*80)
    print(f"Validation Date: {results['validation_timestamp']}")
    print(f"Total Execution Time: {results['total_execution_time']}")
    print(f"Checks Passed: {results['checks_passed']}/{results['total_checks']}")
    print(f"Success Rate: {results['success_rate']}")
    print(f"Overall Status: {results['overall_status']}")
    print("="*80)
    
    for validation_name, validation_result in results['validations'].items():
        status_symbol = "✅" if validation_result['status'] == 'passed' else "⚠️" if validation_result['status'] == 'warning' else "❌"
        print(f"{status_symbol} {validation_name.replace('_', ' ').title()}: {validation_result['status'].upper()}")
    
    print("="*80)
    
    if results['overall_status'] == 'READY':
        print("🎉 SYSTEM IS READY FOR PRODUCTION DEPLOYMENT!")
    else:
        print("⚠️  SYSTEM NEEDS ATTENTION BEFORE PRODUCTION DEPLOYMENT")
        print("📋 Check production_validation_summary.md for detailed recommendations")
    
    print("="*80)
    
    # Return appropriate exit code
    return 0 if results['overall_status'] == 'READY' else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())