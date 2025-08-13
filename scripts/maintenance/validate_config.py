"""Simple configuration validation test."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_basic_validation():
    """Test basic environment validation."""
    try:
        from core.config.environment_validator import (
            EnvironmentValidator,
            ValidationLevel,
        )

        print("✅ Successfully imported EnvironmentValidator")

        # Create test environment
        test_env = {
            "AIVILLAGE_ENV": "development",
            "AIVILLAGE_DB_PATH": "./data/test.db",
            "AIVILLAGE_STORAGE_BACKEND": "sqlite",
            "RAG_EMBEDDING_MODEL": "paraphrase-MiniLM-L3-v2",
            "RAG_VECTOR_DIM": "384",
            "LIBP2P_HOST": "0.0.0.0",
            "LIBP2P_PORT": "4001",
            "DIGITAL_TWIN_COPPA_COMPLIANT": "true",
            "DIGITAL_TWIN_FERPA_COMPLIANT": "true",
            "DIGITAL_TWIN_GDPR_COMPLIANT": "true",
        }

        # Test development profile
        validator = EnvironmentValidator("development")
        print("✅ Created development validator")

        report = validator.validate_all(test_env)
        print(
            f"✅ Validation completed - Errors: {report.errors}, Warnings: {report.warnings}"
        )

        # Show first few issues if any
        if report.issues:
            print("\nTop issues found:")
            for issue in report.issues[:3]:
                level_symbol = "❌" if issue.level == ValidationLevel.ERROR else "⚠️"
                print(f"  {level_symbol} {issue.variable}: {issue.message}")

        return report.errors == 0

    except Exception as e:
        print(f"❌ Error during validation test: {e}")
        return False


def test_configuration_manager() -> bool | None:
    """Test configuration manager."""
    try:
        from core.config.configuration_manager import ConfigurationManager

        print("✅ Successfully imported ConfigurationManager")

        # Create configuration manager
        config = ConfigurationManager("development")
        print("✅ Created development configuration manager")

        # Load configuration (without external validation to avoid dependencies)
        config.load_configuration(validate=False)
        print(f"✅ Loaded {len(config.final_config)} configuration variables")

        # Test basic getters
        env_type = config.get("AIVILLAGE_ENV", "unknown")
        debug_mode = config.get_bool("AIVILLAGE_DEBUG_MODE", False)
        api_port = config.get_int("DIGITAL_TWIN_API_PORT", 8080)

        print(f"✅ Environment: {env_type}")
        print(f"✅ Debug mode: {debug_mode}")
        print(f"✅ API port: {api_port}")

        # Test environment detection
        print(f"✅ Is development: {config.is_development()}")

        return True

    except Exception as e:
        print(f"❌ Error during configuration manager test: {e}")
        return False


def test_template_completeness() -> bool | None:
    """Test that .env.template has all required variables."""
    try:
        template_path = Path(".env.template")
        if not template_path.exists():
            print("❌ .env.template file not found")
            return False

        with open(template_path) as f:
            template_content = f.read()

        # Check for key sections
        required_sections = [
            "EVOLUTION METRICS SYSTEM",
            "RAG PIPELINE SYSTEM",
            "P2P NETWORKING",
            "DIGITAL TWIN SYSTEM",
            "API SERVER CONFIGURATION",
        ]

        missing_sections = []
        for section in required_sections:
            if section not in template_content:
                missing_sections.append(section)

        if missing_sections:
            print(f"❌ Missing sections in .env.template: {missing_sections}")
            return False

        # Count variables
        variable_lines = [
            line
            for line in template_content.split("\n")
            if "=" in line and not line.strip().startswith("#")
        ]

        print(f"✅ .env.template has {len(variable_lines)} configuration variables")
        print("✅ All required sections present")

        return True

    except Exception as e:
        print(f"❌ Error checking .env.template: {e}")
        return False


def test_yaml_config_files() -> bool | None:
    """Test YAML configuration files."""
    try:
        config_dir = Path("config")
        if not config_dir.exists():
            print("❌ config/ directory not found")
            return False

        config_files = [
            "aivillage_config_development.yaml",
            "aivillage_config_production.yaml",
        ]

        for config_file in config_files:
            config_path = config_dir / config_file
            if not config_path.exists():
                print(f"❌ Missing config file: {config_file}")
                return False

            with open(config_path) as f:
                content = f.read()

            # Basic structure check
            if len(content) < 1000:  # Should be substantial files
                print(f"❌ Config file {config_file} seems too small")
                return False

            print(f"✅ Found {config_file} ({len(content)} characters)")

        print("✅ All configuration files present")
        return True

    except Exception as e:
        print(f"❌ Error checking YAML config files: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🔧 AIVillage Configuration System Validation")
    print("=" * 60)

    tests = [
        ("Environment Template", test_template_completeness),
        ("YAML Config Files", test_yaml_config_files),
        ("Environment Validator", test_basic_validation),
        ("Configuration Manager", test_configuration_manager),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} - {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ FAIL - {test_name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("Configuration system is properly set up and ready to use.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
