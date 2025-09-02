"""
AIVillage Unified Linting Manager - Integration Test Suite
Validates all components and configurations work together
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
import tempfile
import yaml

# Setup logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_import_system():
    """Test all import systems work correctly"""
    test_results = {"import_tests": {}}
    
    try:
        # Test caching system imports
        try:
            from linting_manager_fallback import cache_manager, CacheKey
            test_results["import_tests"]["cache_fallback"] = "✓ SUCCESS"
        except ImportError as e:
            test_results["import_tests"]["cache_fallback"] = f"✗ FAILED: {e}"
        
        # Test error handler imports
        try:
            from error_handler import error_handler, ErrorContext
            test_results["import_tests"]["error_handler"] = "✓ SUCCESS"
        except ImportError as e:
            test_results["import_tests"]["error_handler"] = f"✗ FAILED: {e}"
        
        # Test unified linting manager
        try:
            from unified_linting_manager import UnifiedLintingPipeline, unified_linting_manager
            test_results["import_tests"]["unified_manager"] = "✓ SUCCESS"
        except ImportError as e:
            test_results["import_tests"]["unified_manager"] = f"✗ FAILED: {e}"
        
        # Test run_unified_linting CLI
        try:
            from run_unified_linting import main, setup_logging
            test_results["import_tests"]["cli_runner"] = "✓ SUCCESS"
        except ImportError as e:
            test_results["import_tests"]["cli_runner"] = f"✗ FAILED: {e}"
        
        # Test security linting manager
        try:
            from security_linting_manager import SecurityLintingManager
            test_results["import_tests"]["security_manager"] = "✓ SUCCESS"
        except ImportError as e:
            test_results["import_tests"]["security_manager"] = f"✗ FAILED: {e}"
        
    except Exception as e:
        test_results["import_tests"]["general"] = f"✗ UNEXPECTED ERROR: {e}"
    
    return test_results


async def test_configuration_loading():
    """Test configuration file loading and validation"""
    test_results = {"config_tests": {}}
    
    try:
        # Test configuration file exists
        config_path = Path("unified_config.yml")
        if config_path.exists():
            test_results["config_tests"]["file_exists"] = "✓ SUCCESS"
            
            # Test YAML parsing
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                test_results["config_tests"]["yaml_parsing"] = "✓ SUCCESS"
                
                # Test required sections
                required_sections = ["python", "security", "quality_gates", "paths"]
                for section in required_sections:
                    if section in config:
                        test_results["config_tests"][f"section_{section}"] = "✓ SUCCESS"
                    else:
                        test_results["config_tests"][f"section_{section}"] = "✗ MISSING"
                
                # Test configuration validation
                from unified_linting_manager import UnifiedLintingPipeline
                manager = UnifiedLintingPipeline(config_path)
                if manager.config:
                    test_results["config_tests"]["manager_loading"] = "✓ SUCCESS"
                else:
                    test_results["config_tests"]["manager_loading"] = "✗ CONFIG EMPTY"
                    
            except yaml.YAMLError as e:
                test_results["config_tests"]["yaml_parsing"] = f"✗ YAML ERROR: {e}"
            except Exception as e:
                test_results["config_tests"]["manager_loading"] = f"✗ MANAGER ERROR: {e}"
        else:
            test_results["config_tests"]["file_exists"] = "✗ CONFIG FILE NOT FOUND"
    
    except Exception as e:
        test_results["config_tests"]["general"] = f"✗ UNEXPECTED ERROR: {e}"
    
    return test_results


async def test_tool_availability():
    """Test availability of required linting tools"""
    test_results = {"tool_tests": {}}
    
    tools_to_test = [
        "ruff", "black", "mypy", "bandit", 
        "semgrep", "detect-secrets", "pip-audit"
    ]
    
    for tool in tools_to_test:
        try:
            import subprocess
            result = subprocess.run([tool, "--help"], capture_output=True, timeout=10)
            if result.returncode == 0:
                test_results["tool_tests"][tool] = "✓ AVAILABLE"
            else:
                test_results["tool_tests"][tool] = f"✗ ERROR (exit code: {result.returncode})"
        except FileNotFoundError:
            test_results["tool_tests"][tool] = "✗ NOT INSTALLED"
        except subprocess.TimeoutExpired:
            test_results["tool_tests"][tool] = "✗ TIMEOUT"
        except Exception as e:
            test_results["tool_tests"][tool] = f"✗ ERROR: {e}"
    
    return test_results


async def test_cache_system():
    """Test caching system functionality"""
    test_results = {"cache_tests": {}}
    
    try:
        from linting_manager_fallback import cache_manager, CacheKey
        
        # Test basic cache operations
        test_key = "test_key_123"
        test_value = {"test": "data", "timestamp": "2025-01-09"}
        
        # Test set operation
        set_result = await cache_manager.set(test_key, test_value, ttl=300)
        if set_result:
            test_results["cache_tests"]["set_operation"] = "✓ SUCCESS"
        else:
            test_results["cache_tests"]["set_operation"] = "✗ FAILED"
        
        # Test get operation
        get_result = await cache_manager.get(test_key)
        if get_result == test_value:
            test_results["cache_tests"]["get_operation"] = "✓ SUCCESS"
        else:
            test_results["cache_tests"]["get_operation"] = f"✗ MISMATCH: {get_result}"
        
        # Test cache key generation
        params = {"tool": "ruff", "paths": ["/test/path"], "config_hash": "abc123"}
        cache_key = CacheKey.hash_params(params)
        if isinstance(cache_key, str) and len(cache_key) == 16:
            test_results["cache_tests"]["key_generation"] = "✓ SUCCESS"
        else:
            test_results["cache_tests"]["key_generation"] = f"✗ INVALID KEY: {cache_key}"
        
        # Test cache statistics
        stats = cache_manager.get_stats()
        if isinstance(stats, dict) and "hits" in stats:
            test_results["cache_tests"]["statistics"] = "✓ SUCCESS"
        else:
            test_results["cache_tests"]["statistics"] = f"✗ INVALID STATS: {stats}"
    
    except Exception as e:
        test_results["cache_tests"]["general"] = f"✗ UNEXPECTED ERROR: {e}"
    
    return test_results


async def test_error_handling():
    """Test error handling system"""
    test_results = {"error_tests": {}}
    
    try:
        from error_handler import error_handler, ErrorContext
        
        # Test error context creation
        context = ErrorContext(
            operation="test_operation",
            tool="test_tool", 
            target_paths=["/test"],
            config={"test": True}
        )
        
        if hasattr(context, 'operation') and context.operation == "test_operation":
            test_results["error_tests"]["context_creation"] = "✓ SUCCESS"
        else:
            test_results["error_tests"]["context_creation"] = "✗ INVALID CONTEXT"
        
        # Test error handler availability
        if hasattr(error_handler, 'handle_operation'):
            test_results["error_tests"]["handler_methods"] = "✓ SUCCESS"
        else:
            test_results["error_tests"]["handler_methods"] = "✗ MISSING METHODS"
        
        # Test statistics
        stats = error_handler.get_error_stats()
        if isinstance(stats, dict):
            test_results["error_tests"]["statistics"] = "✓ SUCCESS"
        else:
            test_results["error_tests"]["statistics"] = f"✗ INVALID STATS: {stats}"
    
    except Exception as e:
        test_results["error_tests"]["general"] = f"✗ UNEXPECTED ERROR: {e}"
    
    return test_results


async def test_linting_pipeline_basic():
    """Test basic linting pipeline functionality"""
    test_results = {"pipeline_tests": {}}
    
    try:
        from unified_linting_manager import UnifiedLintingPipeline
        
        # Test pipeline instantiation
        pipeline = UnifiedLintingPipeline()
        if pipeline:
            test_results["pipeline_tests"]["instantiation"] = "✓ SUCCESS"
        else:
            test_results["pipeline_tests"]["instantiation"] = "✗ FAILED"
        
        # Test configuration loading
        if hasattr(pipeline, 'config') and pipeline.config:
            test_results["pipeline_tests"]["config_loading"] = "✓ SUCCESS"
            
            # Test specific config sections
            if "python" in pipeline.config:
                test_results["pipeline_tests"]["python_config"] = "✓ SUCCESS"
            else:
                test_results["pipeline_tests"]["python_config"] = "✗ MISSING"
        else:
            test_results["pipeline_tests"]["config_loading"] = "✗ NO CONFIG"
        
        # Test tool lists
        if hasattr(pipeline, 'python_tools') and pipeline.python_tools:
            test_results["pipeline_tests"]["tool_lists"] = "✓ SUCCESS"
        else:
            test_results["pipeline_tests"]["tool_lists"] = "✗ MISSING TOOLS"
        
        # Test quality metrics calculation (basic)
        try:
            # This should work even with empty results
            quality_metrics = await pipeline.calculate_quality_metrics()
            if quality_metrics:
                test_results["pipeline_tests"]["quality_metrics"] = "✓ SUCCESS"
            else:
                test_results["pipeline_tests"]["quality_metrics"] = "✗ NO METRICS"
        except Exception as e:
            test_results["pipeline_tests"]["quality_metrics"] = f"✗ ERROR: {e}"
    
    except Exception as e:
        test_results["pipeline_tests"]["general"] = f"✗ UNEXPECTED ERROR: {e}"
    
    return test_results


async def test_cli_interface():
    """Test CLI interface functionality"""
    test_results = {"cli_tests": {}}
    
    try:
        # Test argument parsing
        import argparse
        from run_unified_linting import main
        
        # This should be importable
        test_results["cli_tests"]["import_main"] = "✓ SUCCESS"
        
        # Test setup_logging function
        from run_unified_linting import setup_logging
        try:
            setup_logging(debug=False)
            test_results["cli_tests"]["logging_setup"] = "✓ SUCCESS"
        except Exception as e:
            test_results["cli_tests"]["logging_setup"] = f"✗ ERROR: {e}"
        
        # Test utility functions
        from run_unified_linting import print_summary, convert_to_sarif, format_text_results
        test_results["cli_tests"]["utility_functions"] = "✓ SUCCESS"
    
    except Exception as e:
        test_results["cli_tests"]["general"] = f"✗ UNEXPECTED ERROR: {e}"
    
    return test_results


async def run_comprehensive_test():
    """Run comprehensive integration test"""
    logger.info("Starting AIVillage Unified Linting Integration Test")
    logger.info("=" * 60)
    
    all_results = {}
    
    # Run all test categories
    test_functions = [
        ("Import System", test_import_system),
        ("Configuration", test_configuration_loading),
        ("Tool Availability", test_tool_availability), 
        ("Cache System", test_cache_system),
        ("Error Handling", test_error_handling),
        ("Linting Pipeline", test_linting_pipeline_basic),
        ("CLI Interface", test_cli_interface)
    ]
    
    for test_name, test_func in test_functions:
        logger.info(f"Running {test_name} tests...")
        try:
            results = await test_func()
            all_results.update(results)
            
            # Print immediate results
            for category, tests in results.items():
                for test, result in tests.items():
                    status_color = "✓" if result.startswith("✓") else "✗"
                    print(f"  {status_color} {test}: {result}")
        
        except Exception as e:
            logger.error(f"Test category {test_name} failed: {e}")
            all_results[f"{test_name.lower()}_error"] = {"general": f"✗ CATEGORY FAILED: {e}"}
    
    # Generate comprehensive report
    logger.info("\n" + "=" * 60)
    logger.info("COMPREHENSIVE TEST RESULTS")
    logger.info("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for category, tests in all_results.items():
        logger.info(f"\n{category.upper()}:")
        logger.info("-" * 30)
        
        for test_name, result in tests.items():
            total_tests += 1
            if result.startswith("✓"):
                passed_tests += 1
                status = "PASS"
            else:
                failed_tests += 1
                status = "FAIL"
            
            logger.info(f"  {test_name}: {status}")
            if not result.startswith("✓"):
                logger.info(f"    Details: {result}")
    
    # Summary statistics
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Pass Rate: {pass_rate:.1f}%")
    
    # Recommendations based on results
    logger.info("\nRECOMMENDations:")
    logger.info("-" * 20)
    
    if failed_tests == 0:
        logger.info("✓ All tests passed! The unified linting manager is ready for use.")
    else:
        logger.info("⚠ Some tests failed. Review the following:")
        
        if any("NOT INSTALLED" in str(result) for results in all_results.values() for result in results.values()):
            logger.info("  • Install missing linting tools: pip install -r requirements-linting.txt")
        
        if any("IMPORT" in str(result) or "MODULE" in str(result) for results in all_results.values() for result in results.values()):
            logger.info("  • Check Python path and virtual environment")
        
        if any("CONFIG" in str(result) for results in all_results.values() for result in results.values()):
            logger.info("  • Verify configuration file syntax and location")
        
        logger.info("  • Check system permissions and dependencies")
        logger.info("  • Review error messages above for specific issues")
    
    # Save results to file
    results_file = Path("integration_test_results.json")
    try:
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": str(datetime.now()),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": pass_rate,
                "detailed_results": all_results
            }, f, indent=2, default=str)
        
        logger.info(f"\nDetailed results saved to: {results_file}")
    except Exception as e:
        logger.warning(f"Could not save results to file: {e}")
    
    return pass_rate >= 80  # Consider success if 80% or more tests pass


if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)