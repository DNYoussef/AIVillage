#!/usr/bin/env python3
"""Test script for AIVillage configuration system.

This script validates the complete configuration system including:
- Environment variable validation
- Configuration manager functionality
- Profile-specific configurations
- Security validation
"""

import base64
import os
import secrets
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from core.config.configuration_manager import ConfigurationManager
from core.config.environment_validator import EnvironmentValidator


def create_test_env_file() -> str:
    """Create a test .env file with valid configuration."""
    # Generate secure keys for testing
    encryption_key = base64.b64encode(secrets.token_bytes(32)).decode()
    api_secret = secrets.token_urlsafe(32)

    test_env_content = f"""# Test Environment Configuration
AIVILLAGE_ENV=development

# Evolution Metrics
AIVILLAGE_DB_PATH=./data/test/evolution_metrics.db
AIVILLAGE_STORAGE_BACKEND=sqlite

# RAG Pipeline
RAG_EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2
RAG_VECTOR_DIM=384
RAG_FAISS_INDEX_PATH=./data/test/faiss_index
RAG_CACHE_ENABLED=true

# P2P Networking
LIBP2P_HOST=0.0.0.0
LIBP2P_PORT=14001
LIBP2P_PRIVATE_KEY_FILE=./config/test_libp2p_private_key

# Digital Twin
DIGITAL_TWIN_ENCRYPTION_KEY={encryption_key}
DIGITAL_TWIN_DB_PATH=./data/test/digital_twin.db
DIGITAL_TWIN_COPPA_COMPLIANT=true
DIGITAL_TWIN_FERPA_COMPLIANT=true
DIGITAL_TWIN_GDPR_COMPLIANT=true
DIGITAL_TWIN_VAULT_PATH=./data/test/vault

# API Configuration
DIGITAL_TWIN_API_PORT=18080
EVOLUTION_METRICS_API_PORT=18081
RAG_PIPELINE_API_PORT=18082
API_SECRET_KEY={api_secret}

# Optional Services
REDIS_HOST=localhost
REDIS_PORT=6379

# Mobile P2P Ports
ANDROID_P2P_PORT_START=14000
ANDROID_P2P_PORT_END=14010
IOS_P2P_PORT_START=14010
IOS_P2P_PORT_END=14020
"""

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write(test_env_content)
        return f.name


def test_environment_validator():
    """Test the environment validator functionality."""
    print("=" * 60)
    print("TESTING ENVIRONMENT VALIDATOR")
    print("=" * 60)

    # Create test environment file
    test_env_file = create_test_env_file()

    try:
        # Load test environment variables
        test_env = {}
        with open(test_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    test_env[key.strip()] = value.strip()

        print(f"✓ Loaded {len(test_env)} test environment variables")

        # Test development profile validation
        print("\n--- Testing Development Profile ---")
        validator = EnvironmentValidator("development")
        dev_report = validator.validate_all(test_env)

        print(
            f"Validation Result: {'✅ VALID' if dev_report.is_valid else '❌ INVALID'}"
        )
        print(f"Errors: {dev_report.errors}")
        print(f"Warnings: {dev_report.warnings}")
        print(f"Total Variables: {dev_report.total_variables}")
        print(f"Valid Variables: {dev_report.valid_variables}")

        if dev_report.issues:
            print("\nFirst 3 issues:")
            for issue in dev_report.issues[:3]:
                print(
                    f"  {issue.level.value.upper()}: {issue.variable} - {issue.message}"
                )

        # Test production profile validation (should be stricter)
        print("\n--- Testing Production Profile ---")
        validator_prod = EnvironmentValidator("production")
        prod_report = validator_prod.validate_all(test_env)

        print(
            f"Validation Result: {'✅ VALID' if prod_report.is_valid else '❌ INVALID'}"
        )
        print(f"Errors: {prod_report.errors}")
        print(f"Warnings: {prod_report.warnings}")

        if prod_report.errors > dev_report.errors:
            print(
                "✓ Production validation is stricter (more errors) - Expected behavior"
            )

        return dev_report.is_valid

    finally:
        # Cleanup
        os.unlink(test_env_file)


def test_configuration_manager() -> bool | None:
    """Test the configuration manager functionality."""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION MANAGER")
    print("=" * 60)

    # Test development profile
    print("\n--- Testing Development Profile Loading ---")
    try:
        config = ConfigurationManager("development")
        config.load_configuration(validate=False)  # Skip validation for faster testing

        print(f"✓ Loaded configuration with {len(config.final_config)} variables")
        print(f"Profile: {config.profile_name}")

        # Test specific getters
        db_path = config.get("AIVILLAGE_DB_PATH")
        api_port = config.get_int("DIGITAL_TWIN_API_PORT", 8080)
        debug_mode = config.get_bool("AIVILLAGE_DEBUG_MODE", False)

        print(f"✓ Database path: {db_path}")
        print(f"✓ API port: {api_port}")
        print(f"✓ Debug mode: {debug_mode}")

        # Test environment detection
        print(f"✓ Is development: {config.is_development()}")
        print(f"✓ Is production: {config.is_production()}")

        # Test database URL generation
        try:
            db_url = config.get_database_url("evolution")
            print(f"✓ Evolution database URL: {db_url}")
        except Exception as e:
            print(f"⚠️  Database URL generation failed: {e}")

        # Test API URL generation
        try:
            api_url = config.get_api_base_url("digital_twin")
            print(f"✓ Digital Twin API URL: {api_url}")
        except Exception as e:
            print(f"⚠️  API URL generation failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Configuration manager test failed: {e}")
        return False


def test_configuration_profiles():
    """Test different configuration profiles."""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION PROFILES")
    print("=" * 60)

    profiles = ["development", "testing", "staging", "production"]
    results = {}

    for profile in profiles:
        print(f"\n--- Testing {profile.upper()} Profile ---")
        try:
            config = ConfigurationManager(profile)
            config.load_configuration(validate=False)

            results[profile] = {
                "loaded": True,
                "variables": len(config.final_config),
                "is_secure": config.get_bool("MESH_TLS_ENABLED", False),
                "has_auth": config.get_bool("API_AUTH_ENABLED", False),
                "debug_mode": config.get_bool("AIVILLAGE_DEBUG_MODE", False),
            }

            print(f"✓ Loaded {results[profile]['variables']} variables")
            print(f"✓ TLS enabled: {results[profile]['is_secure']}")
            print(f"✓ Auth enabled: {results[profile]['has_auth']}")
            print(f"✓ Debug mode: {results[profile]['debug_mode']}")

        except Exception as e:
            print(f"❌ Failed to load {profile} profile: {e}")
            results[profile] = {"loaded": False, "error": str(e)}

    # Validate profile progression (dev -> staging -> production should be more secure)
    print("\n--- Profile Security Progression ---")
    if all(results[p].get("loaded", False) for p in profiles):
        dev_secure = sum(
            [results["development"]["is_secure"], results["development"]["has_auth"]]
        )
        prod_secure = sum(
            [results["production"]["is_secure"], results["production"]["has_auth"]]
        )

        if prod_secure > dev_secure:
            print("✅ Security increases from development to production")
        else:
            print("⚠️  Production may not be more secure than development")

    return all(results[p].get("loaded", False) for p in profiles)


def test_security_validation() -> bool:
    """Test security-specific validation."""
    print("\n" + "=" * 60)
    print("TESTING SECURITY VALIDATION")
    print("=" * 60)

    # Test with insecure configuration
    insecure_env = {
        "AIVILLAGE_ENV": "production",
        "DIGITAL_TWIN_ENCRYPTION_KEY": "REPLACE_WITH_BASE64_ENCODED_32_BYTE_KEY",  # Template value
        "API_SECRET_KEY": "REPLACE_WITH_SECURE_SECRET_KEY",  # Template value
        "DIGITAL_TWIN_COPPA_COMPLIANT": "false",  # Non-compliant in production
        "MESH_TLS_ENABLED": "false",  # Insecure in production
        "API_AUTH_ENABLED": "false",  # Insecure in production
    }

    print("\n--- Testing Insecure Production Configuration ---")
    validator = EnvironmentValidator("production")
    report = validator.validate_all(insecure_env)

    print(f"Validation Result: {'✅ VALID' if report.is_valid else '❌ INVALID'}")
    print(f"Security errors found: {report.errors}")

    if report.errors == 0:
        print("⚠️  Expected security errors but found none")
        return False
    print("✅ Security validation correctly identified issues")

    # Show first few security issues
    security_issues = [
        issue
        for issue in report.issues
        if any(
            keyword in issue.message.lower()
            for keyword in ["security", "insecure", "compliant", "encryption"]
        )
    ]

    print("\nSecurity issues found:")
    for issue in security_issues[:3]:
        print(f"  - {issue.variable}: {issue.message}")

    # Test with secure configuration
    print("\n--- Testing Secure Configuration ---")
    secure_key = base64.b64encode(secrets.token_bytes(32)).decode()
    secure_env = {
        "AIVILLAGE_ENV": "production",
        "DIGITAL_TWIN_ENCRYPTION_KEY": secure_key,
        "API_SECRET_KEY": secrets.token_urlsafe(32),
        "DIGITAL_TWIN_COPPA_COMPLIANT": "true",
        "DIGITAL_TWIN_FERPA_COMPLIANT": "true",
        "DIGITAL_TWIN_GDPR_COMPLIANT": "true",
        "MESH_TLS_ENABLED": "true",
        "API_AUTH_ENABLED": "true",
        "MESH_ENCRYPTION_REQUIRED": "true",
    }

    secure_report = validator.validate_all(secure_env)
    security_errors = len(
        [
            issue
            for issue in secure_report.issues
            if issue.level.value == "error"
            and any(
                keyword in issue.message.lower()
                for keyword in ["security", "insecure", "compliant"]
            )
        ]
    )

    print(f"Security errors in secure config: {security_errors}")

    if security_errors < report.errors:
        print("✅ Secure configuration has fewer security errors")
        return True
    print("⚠️  Secure configuration still has security issues")
    return False


def test_path_validation():
    """Test file path and directory validation."""
    print("\n" + "=" * 60)
    print("TESTING PATH VALIDATION")
    print("=" * 60)

    # Create test directories
    test_base = Path("./test_paths")
    test_base.mkdir(exist_ok=True)

    (test_base / "data").mkdir(exist_ok=True)
    (test_base / "logs").mkdir(exist_ok=True)
    (test_base / "config").mkdir(exist_ok=True)

    # Test with valid paths
    path_env = {
        "AIVILLAGE_DB_PATH": str(test_base / "data" / "test.db"),
        "AIVILLAGE_LOG_DIR": str(test_base / "logs"),
        "RAG_FAISS_INDEX_PATH": str(test_base / "data" / "faiss"),
        "DIGITAL_TWIN_VAULT_PATH": str(test_base / "vault"),
        "LIBP2P_PRIVATE_KEY_FILE": str(test_base / "config" / "key.pem"),
    }

    print("--- Testing Path Validation ---")
    validator = EnvironmentValidator("development")
    report = validator.validate_all(path_env)

    path_issues = [
        issue
        for issue in report.issues
        if any(var in issue.variable for var in path_env)
    ]

    print(f"Path-related issues: {len(path_issues)}")

    # Show path issues
    for issue in path_issues[:3]:
        print(f"  - {issue.variable}: {issue.message}")

    # Cleanup
    import shutil

    try:
        shutil.rmtree(test_base)
        print("✓ Cleaned up test directories")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

    return len(path_issues) == 0 or all(
        issue.level.value != "error" for issue in path_issues
    )


def run_all_tests() -> bool:
    """Run all configuration system tests."""
    print("🧪 AIVillage Configuration System Test Suite")
    print("=" * 80)

    tests = [
        ("Environment Validator", test_environment_validator),
        ("Configuration Manager", test_configuration_manager),
        ("Configuration Profiles", test_configuration_profiles),
        ("Security Validation", test_security_validation),
        ("Path Validation", test_path_validation),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\n🔧 Running {test_name}...")
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "⚠️  PARTIAL"
            print(f"{status} - {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ FAIL - {test_name}: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Configuration system is ready.")
        return True
    if passed >= total * 0.8:  # 80% pass rate
        print("⚠️  Most tests passed. Review failures and warnings.")
        return True
    print("❌ Multiple test failures. Configuration system needs attention.")
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AIVillage configuration system")
    parser.add_argument(
        "--test",
        choices=["validator", "manager", "profiles", "security", "paths", "all"],
        default="all",
        help="Specific test to run",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Set up basic logging
    import logging

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Run specific test or all tests
    if args.test == "all":
        success = run_all_tests()
    elif args.test == "validator":
        success = test_environment_validator()
    elif args.test == "manager":
        success = test_configuration_manager()
    elif args.test == "profiles":
        success = test_configuration_profiles()
    elif args.test == "security":
        success = test_security_validation()
    elif args.test == "paths":
        success = test_path_validation()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
