#!/usr/bin/env python3
"""
Test Migration Script for TDD London School Standardization
==========================================================

Automated script to migrate existing unittest-based tests to pytest with
TDD London School patterns and behavior verification.
"""

import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import logging
from dataclasses import dataclass


@dataclass
class TestMigrationResult:
    """Results of test migration process."""
    file_path: str
    original_framework: str
    migration_status: str
    issues_found: List[str]
    changes_made: List[str]


class TestMigrationTool:
    """Tool for migrating tests to TDD London School patterns."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)
        self.migration_results: List[TestMigrationResult] = []
        
        # Patterns for detecting testing frameworks
        self.unittest_patterns = [
            r'import unittest',
            r'from unittest import',
            r'class.*Test.*unittest\.TestCase',
            r'self\.assert',
        ]
        
        self.pytest_patterns = [
            r'import pytest',
            r'@pytest\.',
            r'def test_',
            r'assert '
        ]
        
        # Mock patterns
        self.old_mock_patterns = [
            r'from unittest\.mock import',
            r'unittest\.mock\.',
            r'@patch\(',
            r'MagicMock\(',
            r'Mock\('
        ]

    def analyze_test_file(self, file_path: Path) -> Dict[str, any]:
        """Analyze a test file to determine its current structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Failed to read {file_path}: {e}")
            return {}
        
        analysis = {
            'file_path': str(file_path),
            'has_unittest': any(re.search(pattern, content) for pattern in self.unittest_patterns),
            'has_pytest': any(re.search(pattern, content) for pattern in self.pytest_patterns),
            'has_old_mocks': any(re.search(pattern, content) for pattern in self.old_mock_patterns),
            'unittest_classes': self._find_unittest_classes(content),
            'test_methods': self._find_test_methods(content),
            'mock_usage': self._analyze_mock_usage(content),
            'needs_migration': False
        }
        
        # Determine if migration is needed
        analysis['needs_migration'] = (
            analysis['has_unittest'] or 
            (analysis['has_old_mocks'] and not analysis['has_pytest'])
        )
        
        return analysis

    def _find_unittest_classes(self, content: str) -> List[str]:
        """Find unittest.TestCase classes in content."""
        pattern = r'class\s+(\w+)\(.*unittest\.TestCase.*\):'
        matches = re.findall(pattern, content)
        return matches

    def _find_test_methods(self, content: str) -> List[str]:
        """Find test methods in content."""
        pattern = r'def\s+(test_\w+)\s*\('
        matches = re.findall(pattern, content)
        return matches

    def _analyze_mock_usage(self, content: str) -> Dict[str, int]:
        """Analyze mock usage patterns."""
        return {
            'patch_decorators': len(re.findall(r'@patch\(', content)),
            'magic_mock_usage': len(re.findall(r'MagicMock\(', content)),
            'mock_usage': len(re.findall(r'Mock\(', content)),
            'side_effect_usage': len(re.findall(r'\.side_effect', content))
        }

    def migrate_unittest_to_pytest(self, file_path: Path) -> TestMigrationResult:
        """Migrate a unittest file to pytest with London School patterns."""
        self.logger.info(f"Migrating {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            return TestMigrationResult(
                file_path=str(file_path),
                original_framework='unknown',
                migration_status='failed',
                issues_found=[f"Cannot read file: {e}"],
                changes_made=[]
            )
        
        analysis = self.analyze_test_file(file_path)
        issues_found = []
        changes_made = []
        
        # Start with original content
        migrated_content = original_content
        
        # 1. Replace unittest imports with pytest imports
        if analysis['has_unittest']:
            migrated_content = self._replace_unittest_imports(migrated_content)
            changes_made.append("Replaced unittest imports with pytest imports")
        
        # 2. Convert unittest.TestCase classes to pytest functions/classes
        if analysis['unittest_classes']:
            migrated_content, class_changes = self._convert_unittest_classes(migrated_content, analysis['unittest_classes'])
            changes_made.extend(class_changes)
        
        # 3. Replace unittest assertions with pytest assertions
        migrated_content, assert_changes = self._convert_assertions(migrated_content)
        changes_made.extend(assert_changes)
        
        # 4. Add TDD London School imports and fixtures
        migrated_content = self._add_london_school_imports(migrated_content)
        changes_made.append("Added TDD London School imports")
        
        # 5. Convert mocks to behavior verification mocks
        if analysis['has_old_mocks']:
            migrated_content, mock_changes = self._convert_mocks_to_behavior_verification(migrated_content)
            changes_made.extend(mock_changes)
        
        # 6. Add pytest markers based on file location and content
        migrated_content = self._add_pytest_markers(migrated_content, file_path)
        changes_made.append("Added appropriate pytest markers")
        
        # Write the migrated content
        if not self.dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(migrated_content)
                self.logger.info(f"Successfully migrated {file_path}")
                migration_status = 'completed'
            except Exception as e:
                self.logger.error(f"Failed to write migrated content to {file_path}: {e}")
                migration_status = 'failed'
                issues_found.append(f"Cannot write file: {e}")
        else:
            migration_status = 'dry_run_completed'
            self.logger.info(f"Dry run completed for {file_path}")
        
        return TestMigrationResult(
            file_path=str(file_path),
            original_framework='unittest',
            migration_status=migration_status,
            issues_found=issues_found,
            changes_made=changes_made
        )

    def _replace_unittest_imports(self, content: str) -> str:
        """Replace unittest imports with pytest equivalents."""
        # Replace common unittest imports
        replacements = [
            (r'import unittest\n', 'import pytest\n'),
            (r'from unittest import TestCase\n', 'import pytest\n'),
            (r'from unittest\.mock import', 'from tests.fixtures.tdd_london_mocks import'),
            (r'unittest\.mock\.', '')
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        return content

    def _convert_unittest_classes(self, content: str, class_names: List[str]) -> Tuple[str, List[str]]:
        """Convert unittest.TestCase classes to pytest classes."""
        changes_made = []
        
        for class_name in class_names:
            # Replace class definition
            old_pattern = f'class {class_name}\\(.*unittest\\.TestCase.*\\):'
            new_class_def = f'class {class_name}(ContractTestingMixin):'
            content = re.sub(old_pattern, new_class_def, content)
            changes_made.append(f"Converted {class_name} to pytest class")
            
            # Remove setUp and tearDown methods, replace with fixtures
            content = self._convert_setup_teardown_to_fixtures(content, class_name)
            changes_made.append(f"Converted setUp/tearDown to fixtures for {class_name}")
        
        return content, changes_made

    def _convert_setup_teardown_to_fixtures(self, content: str, class_name: str) -> str:
        """Convert setUp/tearDown methods to pytest fixtures."""
        # This is a simplified conversion - in practice, this would be more complex
        setup_pattern = r'def setUp\(self\):(.*?)(?=def|\Z)'
        teardown_pattern = r'def tearDown\(self\):(.*?)(?=def|\Z)'
        
        # Replace setUp with fixture
        content = re.sub(
            setup_pattern,
            '''@pytest.fixture(autouse=True)
    def setup_method(self, mock_factory):
        """Setup for each test method."""\\1''',
            content,
            flags=re.DOTALL
        )
        
        # Replace tearDown with cleanup
        content = re.sub(
            teardown_pattern,
            '''def teardown_method(self):
        """Cleanup after each test method."""\\1''',
            content,
            flags=re.DOTALL
        )
        
        return content

    def _convert_assertions(self, content: str) -> Tuple[str, List[str]]:
        """Convert unittest assertions to pytest assertions."""
        changes_made = []
        
        # Common unittest assertion conversions
        assertion_conversions = [
            (r'self\.assertTrue\((.*?)\)', r'assert \1'),
            (r'self\.assertFalse\((.*?)\)', r'assert not \1'),
            (r'self\.assertEqual\((.*?),\s*(.*?)\)', r'assert \1 == \2'),
            (r'self\.assertNotEqual\((.*?),\s*(.*?)\)', r'assert \1 != \2'),
            (r'self\.assertIsNone\((.*?)\)', r'assert \1 is None'),
            (r'self\.assertIsNotNone\((.*?)\)', r'assert \1 is not None'),
            (r'self\.assertIn\((.*?),\s*(.*?)\)', r'assert \1 in \2'),
            (r'self\.assertNotIn\((.*?),\s*(.*?)\)', r'assert \1 not in \2'),
            (r'self\.assertRaises\((.*?)\)', r'pytest.raises(\1)'),
            (r'self\.assertRaisesRegex\((.*?),\s*(.*?)\)', r'pytest.raises(\1, match=\2)'),
        ]
        
        for pattern, replacement in assertion_conversions:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                changes_made.append(f"Converted assertion: {pattern[:20]}...")
        
        return content, changes_made

    def _add_london_school_imports(self, content: str) -> str:
        """Add TDD London School imports to the file."""
        london_imports = '''
from tests.fixtures.tdd_london_mocks import MockFactory, ContractTestingMixin
from tests.fixtures.unified_conftest import *
'''
        
        # Add imports after existing imports
        import_end_pattern = r'((?:^(?:import|from).*\n)+)'
        if re.search(import_end_pattern, content, re.MULTILINE):
            content = re.sub(
                import_end_pattern,
                r'\1' + london_imports,
                content,
                count=1,
                flags=re.MULTILINE
            )
        else:
            content = london_imports + '\n' + content
        
        return content

    def _convert_mocks_to_behavior_verification(self, content: str) -> Tuple[str, List[str]]:
        """Convert standard mocks to behavior verification mocks."""
        changes_made = []
        
        # Replace MagicMock with behavior verification mock
        if 'MagicMock' in content:
            content = content.replace('MagicMock()', 'create_behavior_mock("component")')
            changes_made.append("Replaced MagicMock with behavior verification mock")
        
        # Replace Mock with behavior verification mock
        if 'Mock()' in content:
            content = content.replace('Mock()', 'create_behavior_mock("mock")')
            changes_made.append("Replaced Mock with behavior verification mock")
        
        # Add mock factory fixture if needed
        if any(pattern in content for pattern in ['create_behavior_mock', 'MockFactory']):
            # Add fixture parameter if not already present
            content = self._add_mock_factory_fixture_parameter(content)
            changes_made.append("Added mock_factory fixture parameter")
        
        return content, changes_made

    def _add_mock_factory_fixture_parameter(self, content: str) -> str:
        """Add mock_factory parameter to test methods that need it."""
        # This is a simplified implementation
        # In practice, would need more sophisticated AST parsing
        test_method_pattern = r'(def test_\w+\(self)(.*?\):)'
        
        def add_mock_factory(match):
            method_def = match.group(1)
            params_and_body = match.group(2)
            
            if 'mock_factory' not in params_and_body:
                if params_and_body.strip() == '):':
                    return f"{method_def}, mock_factory):"
                else:
                    return f"{method_def}, mock_factory{params_and_body}"
            return match.group(0)
        
        content = re.sub(test_method_pattern, add_mock_factory, content)
        return content

    def _add_pytest_markers(self, content: str, file_path: Path) -> str:
        """Add appropriate pytest markers based on file location and content."""
        markers = []
        
        # Determine markers based on file path
        path_str = str(file_path).lower()
        if 'unit' in path_str:
            markers.append('unit')
        if 'integration' in path_str:
            markers.append('integration')
        if 'security' in path_str:
            markers.append('security')
        if 'performance' in path_str:
            markers.append('performance')
        
        # Determine markers based on content
        if 'mock' in content.lower():
            markers.append('mockist')
        if any(pattern in content for pattern in ['behavior_verification', 'collaboration']):
            markers.append('behavior_verification')
        
        # Add markers to test classes and methods
        if markers:
            marker_decorators = '\n'.join([f'@pytest.mark.{marker}' for marker in markers])
            
            # Add to test classes
            content = re.sub(
                r'(class Test\w+.*:)',
                f'{marker_decorators}\n\\1',
                content
            )
            
            # Add to test methods that don't already have markers
            content = re.sub(
                r'(?<!@pytest\.mark\.)\n(\s+def test_\w+.*:)',
                f'\n{marker_decorators}\n\\1',
                content
            )
        
        return content

    def migrate_test_directory(self, directory: Path, pattern: str = "test_*.py") -> List[TestMigrationResult]:
        """Migrate all test files in a directory."""
        self.logger.info(f"Migrating test files in {directory}")
        
        results = []
        test_files = list(directory.rglob(pattern))
        
        for test_file in test_files:
            analysis = self.analyze_test_file(test_file)
            
            if analysis.get('needs_migration', False):
                result = self.migrate_unittest_to_pytest(test_file)
                results.append(result)
                self.migration_results.append(result)
            else:
                self.logger.info(f"Skipping {test_file} - no migration needed")
        
        return results

    def generate_migration_report(self) -> str:
        """Generate a detailed migration report."""
        report = ["TDD London School Migration Report", "=" * 50, ""]
        
        total_files = len(self.migration_results)
        successful = sum(1 for r in self.migration_results if r.migration_status == 'completed')
        failed = sum(1 for r in self.migration_results if r.migration_status == 'failed')
        
        report.extend([
            f"Total files processed: {total_files}",
            f"Successful migrations: {successful}",
            f"Failed migrations: {failed}",
            f"Success rate: {(successful/total_files*100):.1f}%" if total_files > 0 else "N/A",
            ""
        ])
        
        # Detailed results
        for result in self.migration_results:
            report.extend([
                f"File: {result.file_path}",
                f"Status: {result.migration_status}",
                f"Changes: {len(result.changes_made)}",
                ""
            ])
            
            if result.issues_found:
                report.extend(["Issues found:"])
                for issue in result.issues_found:
                    report.append(f"  - {issue}")
                report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate tests to TDD London School patterns"
    )
    parser.add_argument(
        'directory',
        type=Path,
        help='Directory containing test files to migrate'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without making changes'
    )
    parser.add_argument(
        '--pattern',
        default='test_*.py',
        help='File pattern to match (default: test_*.py)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create migration tool
    migration_tool = TestMigrationTool(dry_run=args.dry_run)
    
    # Perform migration
    results = migration_tool.migrate_test_directory(args.directory, args.pattern)
    
    # Generate and print report
    report = migration_tool.generate_migration_report()
    print(report)
    
    # Save report to file
    report_file = args.directory / 'migration_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_file}")


if __name__ == '__main__':
    main()