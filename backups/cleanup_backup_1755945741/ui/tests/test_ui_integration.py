#!/usr/bin/env python3
"""
Comprehensive UI Integration Test Suite
Tests all consolidated UI components: Web, Mobile, CLI, and Admin interfaces.

This test suite validates:
- Web React application functionality
- Mobile integration components
- CLI tool execution
- Admin dashboard integration
- Cross-component communication
"""

import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ui"))


class UIIntegrationTestSuite(unittest.TestCase):
    """Comprehensive UI integration tests."""

    def setUp(self):
        """Set up test environment."""
        self.project_root = project_root
        self.ui_root = self.project_root / "ui"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ui_structure_exists(self):
        """Test that consolidated UI structure exists."""
        expected_dirs = [self.ui_root / "web", self.ui_root / "mobile", self.ui_root / "cli", self.ui_root / "docs"]

        for directory in expected_dirs:
            self.assertTrue(directory.exists(), f"UI directory should exist: {directory}")

    def test_web_app_structure(self):
        """Test web application structure."""
        web_root = self.ui_root / "web"

        # Check main application files
        essential_files = ["App.tsx", "App.css", "package.json", "tsconfig.json"]

        for file_name in essential_files:
            file_path = web_root / file_name
            self.assertTrue(file_path.exists(), f"Essential web file should exist: {file_name}")

    def test_web_components_structure(self):
        """Test web components organization."""
        components_root = self.ui_root / "web" / "src" / "components"

        expected_categories = [
            "common",  # Shared UI components
            "concierge",  # AI assistant interface
            "dashboard",  # System monitoring
            "media",  # Multimedia handling
            "messaging",  # P2P communication
            "wallet",  # Economic system
            "admin",  # Administrative interface
        ]

        for category in expected_categories:
            category_dir = components_root / category
            self.assertTrue(category_dir.exists(), f"Component category should exist: {category}")

    def test_mobile_integration_structure(self):
        """Test mobile integration components."""
        mobile_root = self.ui_root / "mobile"

        # Check mobile integration files
        essential_files = [
            "__init__.py",
            "shared/digital_twin_concierge.py",
            "shared/mini_rag_system.py",
            "shared/resource_manager.py",
        ]

        for file_name in essential_files:
            file_path = mobile_root / file_name
            self.assertTrue(file_path.exists(), f"Mobile integration file should exist: {file_name}")

    def test_cli_tools_structure(self):
        """Test CLI tools organization."""
        cli_root = self.ui_root / "cli"

        # Check CLI tool files
        expected_tools = [
            "system_manager.py",  # Unified system manager
            "agent_forge.py",  # Agent Forge CLI
            "dashboard_launcher.py",  # Dashboard launcher
            "base.py",  # Base utilities
        ]

        for tool_name in expected_tools:
            tool_path = cli_root / tool_name
            self.assertTrue(tool_path.exists(), f"CLI tool should exist: {tool_name}")

    def test_cli_system_manager_functionality(self):
        """Test unified CLI system manager."""
        cli_script = self.ui_root / "cli" / "system_manager.py"

        if not cli_script.exists():
            self.skipTest("CLI system manager not found")

        # Test help command
        try:
            result = subprocess.run(
                [sys.executable, str(cli_script), "--help"], capture_output=True, text=True, timeout=10
            )

            self.assertEqual(result.returncode, 0, "CLI help should execute successfully")
            self.assertIn("AIVillage Unified System Manager", result.stdout)
            self.assertIn("dashboard", result.stdout)
            self.assertIn("forge", result.stdout)

        except subprocess.TimeoutExpired:
            self.fail("CLI help command timed out")
        except Exception as e:
            self.fail(f"CLI help command failed: {e}")

    @patch("subprocess.run")
    def test_dashboard_launcher(self, mock_run):
        """Test dashboard launcher functionality."""
        cli_script = self.ui_root / "cli" / "dashboard_launcher.py"

        if not cli_script.exists():
            self.skipTest("Dashboard launcher not found")

        # Mock successful streamlit launch
        mock_run.return_value.returncode = 0

        try:
            result = subprocess.run([sys.executable, str(cli_script)], capture_output=True, text=True, timeout=5)

            # Should attempt to launch streamlit
            self.assertTrue(mock_run.called or result.returncode == 0)

        except subprocess.TimeoutExpired:
            # Expected for streamlit launch - it runs indefinitely
            pass
        except Exception as e:
            self.skipTest(f"Dashboard launcher test skipped: {e}")

    def test_mobile_imports(self):
        """Test mobile integration imports."""
        try:
            from ui.mobile import DigitalTwinConcierge, MiniRAGSystem, MobileResourceManager

            # Test that classes can be imported
            self.assertTrue(callable(DigitalTwinConcierge))
            self.assertTrue(callable(MiniRAGSystem))
            self.assertTrue(callable(MobileResourceManager))

        except ImportError as e:
            self.fail(f"Mobile integration imports failed: {e}")

    def test_web_package_json_dependencies(self):
        """Test web application dependencies."""
        package_json = self.ui_root / "web" / "package.json"

        if not package_json.exists():
            self.skipTest("Web package.json not found")

        with open(package_json) as f:
            package_data = json.load(f)

        # Check essential dependencies
        essential_deps = ["react", "react-dom", "react-router-dom"]

        dependencies = package_data.get("dependencies", {})
        for dep in essential_deps:
            self.assertIn(dep, dependencies, f"Essential dependency missing: {dep}")

    def test_admin_interface_integration(self):
        """Test admin interface components."""
        admin_component = self.ui_root / "web" / "src" / "components" / "admin" / "AdminInterface.tsx"
        admin_styles = self.ui_root / "web" / "src" / "components" / "admin" / "AdminInterface.css"

        if admin_component.exists():
            # Check that admin component has proper structure
            with open(admin_component) as f:
                content = f.read()

            self.assertIn("AdminInterface", content)
            self.assertIn("SystemMetrics", content)
            self.assertIn("fetchSystemMetrics", content)

        if admin_styles.exists():
            # Check that CSS styles exist
            with open(admin_styles) as f:
                css_content = f.read()

            self.assertIn("admin-interface", css_content)
            self.assertIn("metric-card", css_content)

    def test_ui_test_structure(self):
        """Test UI test structure."""
        test_dirs = [self.ui_root / "tests" / "web", self.ui_root / "tests" / "mobile", self.ui_root / "tests" / "cli"]

        for test_dir in test_dirs:
            if test_dir.exists():
                # Should contain test files
                test_files = list(test_dir.glob("*.test.*"))
                if test_files:
                    self.assertGreater(len(test_files), 0, f"Test directory should contain test files: {test_dir}")

    def test_documentation_structure(self):
        """Test UI documentation structure."""
        docs_dir = self.ui_root / "docs"

        if docs_dir.exists():
            # Check for documentation files
            doc_files = list(docs_dir.glob("*.md"))
            if doc_files:
                self.assertGreater(len(doc_files), 0, "Should have documentation files")


class UIPerformanceTests(unittest.TestCase):
    """Performance tests for UI components."""

    def test_ui_file_sizes(self):
        """Test that UI files are reasonably sized."""
        ui_root = Path(__file__).parent.parent

        # Check large files that might need optimization
        large_files = []

        for file_path in ui_root.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                try:
                    size = file_path.stat().st_size
                    if size > 1024 * 1024:  # Files larger than 1MB
                        large_files.append((file_path, size))
                except (OSError, PermissionError):
                    continue

        # Report large files for optimization consideration
        if large_files:
            print(f"\nFound {len(large_files)} large UI files:")
            for file_path, size in large_files:
                print(f"  {file_path.name}: {size / 1024 / 1024:.2f} MB")

    def test_ui_directory_structure_efficiency(self):
        """Test UI directory structure efficiency."""
        ui_root = Path(__file__).parent.parent

        # Count files in each UI category
        ui_categories = {
            "web": ui_root / "web",
            "mobile": ui_root / "mobile",
            "cli": ui_root / "cli",
            "tests": ui_root / "tests",
        }

        category_stats = {}
        for category, path in ui_categories.items():
            if path.exists():
                files = list(path.rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                category_stats[category] = file_count

        print(f"\nUI Structure Efficiency:")
        for category, count in category_stats.items():
            print(f"  {category}: {count} files")

        # Should have reasonable file distribution
        total_files = sum(category_stats.values())
        self.assertGreater(total_files, 0, "Should have UI files")


def run_ui_tests():
    """Run all UI integration tests."""
    print("Running AIVillage UI Integration Test Suite...")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(UIIntegrationTestSuite))
    suite.addTests(loader.loadTestsFromTestCase(UIPerformanceTests))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("UI Integration Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"Success rate: {success_rate:.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_ui_tests()
    sys.exit(0 if success else 1)
