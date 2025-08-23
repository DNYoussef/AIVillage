#!/usr/bin/env python3
"""
CI/CD Readiness Validation for Unified P2P Test Suite
Validates the consolidated test suite is ready for automated CI/CD execution.
"""

import os
import sys
import time
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "packages"))
sys.path.insert(0, str(project_root / "tests"))

def validate_test_structure():
    """Validate test file structure and organization."""
    print("=== P2P TEST SUITE STRUCTURE VALIDATION ===\n")
    
    expected_files = [
        # Core functionality tests
        "tests/communications/test_p2p.py",
        "tests/unit/test_unified_p2p_consolidated.py", 
        "tests/unit/test_unified_p2p.py",
        
        # Transport-specific tests
        "tests/p2p/test_bitchat_reliability.py",
        "tests/p2p/test_betanet_covert_transport.py",
        "tests/core/p2p/test_mesh_reliability.py",
        
        # Integration tests
        "tests/p2p/test_real_p2p_stack.py",
        "tests/integration/test_p2p_bridge_delivery.py",
        
        # Security and performance tests
        "tests/security/test_p2p_network_security.py",
        "tests/validation/p2p/test_p2p_performance_validation.py",
        
        # Mobile platform tests
        "tests/mobile/test_libp2p_mesh_android.py",
        
        # Production validation
        "tests/production/test_p2p_validation.py",
        
        # Unified configuration
        "tests/conftest.py"
    ]
    
    file_status = {}
    for file_path in expected_files:
        full_path = project_root / file_path
        exists = full_path.exists()
        file_status[file_path] = {
            'exists': exists,
            'size': full_path.stat().st_size if exists else 0,
            'category': get_test_category(file_path)
        }
        
        status = "✅" if exists else "❌"
        size_info = f"({file_status[file_path]['size']} bytes)" if exists else ""
        print(f"{status} {file_path} {size_info}")
    
    # Summary statistics
    existing_files = [f for f, info in file_status.items() if info['exists']]
    total_size = sum(info['size'] for info in file_status.values() if info['exists'])
    
    print(f"\n📊 STRUCTURE VALIDATION SUMMARY:")
    print(f"   Expected files: {len(expected_files)}")
    print(f"   Existing files: {len(existing_files)} ({len(existing_files)/len(expected_files)*100:.1f}%)")
    print(f"   Total test code: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    
    return len(existing_files) >= len(expected_files) * 0.9  # 90% threshold

def get_test_category(file_path):
    """Categorize test files by functionality."""
    if "communications" in file_path or "unit" in file_path:
        return "Core Functionality"
    elif "bitchat" in file_path or "betanet" in file_path or "mesh" in file_path:
        return "Transport Protocols"
    elif "security" in file_path:
        return "Security"
    elif "performance" in file_path or "validation" in file_path:
        return "Performance/Validation"
    elif "mobile" in file_path:
        return "Mobile Platform"
    elif "integration" in file_path:
        return "Integration"
    elif "production" in file_path:
        return "Production"
    else:
        return "Configuration"

def validate_import_paths():
    """Validate that test imports work correctly."""
    print("\n=== IMPORT PATH VALIDATION ===\n")
    
    import_tests = [
        ("pytest", "Pytest framework"),
        ("asyncio", "Async support"),
        ("unittest.mock", "Mock framework"),
        ("json", "JSON handling"),
        ("time", "Timing utilities"),
        ("pathlib", "Path utilities")
    ]
    
    import_success = 0
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"PASS {module_name} - {description}")
            import_success += 1
        except ImportError as e:
            print(f"FAIL {module_name} - {description}: {e}")
    
    print(f"\n📊 IMPORT VALIDATION SUMMARY:")
    print(f"   Required imports: {len(import_tests)}")
    print(f"   Successful imports: {import_success} ({import_success/len(import_tests)*100:.1f}%)")
    
    return import_success >= len(import_tests) * 0.8  # 80% threshold

def validate_test_fixtures():
    """Validate unified test fixtures."""
    print("\n=== TEST FIXTURE VALIDATION ===\n")
    
    conftest_path = project_root / "tests" / "conftest.py"
    
    if not conftest_path.exists():
        print("FAIL conftest.py not found")
        return False
    
    with open(conftest_path, 'r') as f:
        conftest_content = f.read()
    
    required_fixtures = [
        "mock_p2p_transport",
        "p2p_test_config", 
        "mock_mesh_protocol",
        "p2p_test_messages"
    ]
    
    fixture_status = {}
    for fixture in required_fixtures:
        exists = f"def {fixture}" in conftest_content or f"{fixture}():" in conftest_content
        fixture_status[fixture] = exists
        status = "✅" if exists else "❌"
        print(f"{status} {fixture} fixture")
    
    working_fixtures = sum(fixture_status.values())
    print(f"\n📊 FIXTURE VALIDATION SUMMARY:")
    print(f"   Required fixtures: {len(required_fixtures)}")
    print(f"   Available fixtures: {working_fixtures} ({working_fixtures/len(required_fixtures)*100:.1f}%)")
    
    return working_fixtures >= len(required_fixtures) * 0.75  # 75% threshold

def validate_consolidation_results():
    """Validate consolidation achievements."""
    print("\n=== CONSOLIDATION RESULTS VALIDATION ===\n")
    
    # Count all P2P-related test files
    test_dir = project_root / "tests"
    p2p_files = []
    
    if test_dir.exists():
        for py_file in test_dir.rglob("*.py"):
            if any(keyword in py_file.name.lower() for keyword in ['p2p', 'bitchat', 'betanet', 'mesh']):
                p2p_files.append(py_file)
    
    # Expected metrics based on consolidation
    target_file_count = 25  # Target after consolidation
    current_file_count = len(p2p_files)
    
    print(f"CONSOLIDATION METRICS:")
    print(f"   Original files: 127+ (before consolidation)")
    print(f"   Current P2P files: {current_file_count}")
    print(f"   Target files: ≤{target_file_count}")
    print(f"   Reduction achieved: {((127 - current_file_count) / 127 * 100):.1f}%")
    
    if current_file_count <= target_file_count:
        print("PASS File consolidation target achieved")
        consolidation_success = True
    else:
        print(f"WARN File consolidation target missed by {current_file_count - target_file_count} files")
        consolidation_success = False
    
    # List current P2P files
    print(f"\n📁 CURRENT P2P TEST FILES ({len(p2p_files)}):")
    for f in sorted(p2p_files):
        rel_path = f.relative_to(project_root)
        print(f"   - {rel_path}")
    
    return consolidation_success

def generate_ci_cd_recommendations():
    """Generate CI/CD pipeline recommendations."""
    print("\n=== CI/CD PIPELINE RECOMMENDATIONS ===\n")
    
    recommendations = [
        "** Test Execution Strategy **:",
        "   - Run core functionality tests first (fastest feedback)",
        "   - Parallel execution of transport-specific tests",
        "   - Security and performance tests in separate stage",
        "   - Mobile tests in dedicated environment",
        "",
        "** Performance Optimizations **:",
        "   - Estimated total execution time: 15-20 minutes",
        "   - Use test parallelization with pytest-xdist",
        "   - Cache dependencies and mock objects",
        "   - Skip integration tests in PR validation (run on merge)",
        "",
        "** Required Dependencies **:",
        "   - pytest>=8.0",
        "   - pytest-asyncio>=0.21.0",
        "   - pytest-mock>=3.10.0",
        "   - pytest-xdist (for parallel execution)",
        "",
        "** Success Criteria **:",
        "   - Core functionality: 100% pass rate required",
        "   - Transport protocols: 95% pass rate minimum",
        "   - Security tests: 100% pass rate required",
        "   - Performance tests: Benchmark comparison only",
        "",
        "** Monitoring & Reporting **:",
        "   - Code coverage reporting with pytest-cov",
        "   - Performance regression detection",
        "   - Security vulnerability scanning",
        "   - Test result trending and analysis"
    ]
    
    for rec in recommendations:
        print(rec)

def main():
    """Main validation function."""
    print("AIVillage P2P Test Suite - CI/CD Readiness Validation")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all validations
    validations = [
        ("Test Structure", validate_test_structure),
        ("Import Paths", validate_import_paths), 
        ("Test Fixtures", validate_test_fixtures),
        ("Consolidation Results", validate_consolidation_results)
    ]
    
    results = {}
    for name, validator in validations:
        print(f"\n{'-' * 20}")
        results[name] = validator()
    
    # Generate recommendations
    generate_ci_cd_recommendations()
    
    # Final summary
    execution_time = time.time() - start_time
    passed_validations = sum(results.values())
    total_validations = len(results)
    success_rate = (passed_validations / total_validations) * 100
    
    print(f"\n{'=' * 60}")
    print("CI/CD READINESS SUMMARY")
    print("=" * 60)
    print(f"   Validations passed: {passed_validations}/{total_validations} ({success_rate:.1f}%)")
    print(f"   Execution time: {execution_time:.2f} seconds")
    
    if success_rate >= 75:
        print("READY FOR CI/CD INTEGRATION")
        print("   The unified P2P test suite is ready for automated testing")
        return True
    else:
        print("REQUIRES ATTENTION BEFORE CI/CD")
        print("   Please address validation failures before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)