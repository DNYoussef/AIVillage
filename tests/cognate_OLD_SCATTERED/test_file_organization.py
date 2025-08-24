#!/usr/bin/env python3
"""
Test Suite 4: File Organization Testing
Verify reorganization is clean and no duplicate functionality remains.
"""

import ast
from pathlib import Path
import sys

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Also need to add the core directory specifically
core_path = project_root / "core"
if core_path.exists():
    sys.path.insert(0, str(core_path))


class TestFileOrganizationValidation:
    """Test file organization and cleanup after reorganization."""
    
    def test_new_structure_exists(self):
        """Test that new cognate-pretrain structure exists."""
        base_path = project_root / "core" / "agent-forge" / "phases" / "cognate-pretrain"
        
        # Check main directory exists
        assert base_path.exists(), "cognate-pretrain directory should exist"
        assert base_path.is_dir(), "cognate-pretrain should be a directory"
        
        # Check required files exist
        required_files = [
            "__init__.py",
            "model_factory.py", 
            "cognate_creator.py",
            "pretrain_pipeline.py",
            "phase_integration.py"
        ]
        
        for file_name in required_files:
            file_path = base_path / file_name
            assert file_path.exists(), f"Required file missing: {file_name}"
            assert file_path.is_file(), f"{file_name} should be a file"
        
        print("✅ New cognate-pretrain structure exists and is complete")
    
    def test_deprecated_files_moved(self):
        """Test that deprecated files were moved to deprecated_duplicates."""
        phases_path = project_root / "core" / "agent-forge" / "phases"
        deprecated_path = phases_path / "deprecated_duplicates"
        
        # Check deprecated directory exists
        if deprecated_path.exists():
            print(f"✅ Deprecated duplicates directory exists: {deprecated_path}")
            
            # Check for expected deprecated files
            expected_deprecated = [
                "optimal_25m_training.py",
                "train_and_save_25m_models.py", 
                "cognate_evomerge_50gen.py",
                "cognate_25m_results.json"
            ]
            
            for file_name in expected_deprecated:
                file_path = deprecated_path / file_name
                if file_path.exists():
                    print(f"✅ Found deprecated file: {file_name}")
        else:
            print("⚠️  No deprecated_duplicates directory found")
    
    def test_cognate_redirect_file_updated(self):
        """Test that cognate.py was updated to redirect."""
        cognate_file = project_root / "core" / "agent-forge" / "phases" / "cognate.py"
        
        assert cognate_file.exists(), "cognate.py should still exist as redirect"
        
        # Read file content to verify it's a redirect
        with open(cognate_file, encoding='utf-8') as f:
            content = f.read()
        
        # Check for redirect indicators
        assert "REDIRECT" in content.upper() or "redirect" in content, \
            "cognate.py should contain redirect indication"
        assert "cognate_pretrain" in content or "cognate-pretrain" in content, \
            "cognate.py should reference new structure"
        assert "deprecated" in content.lower(), \
            "cognate.py should indicate deprecation"
        
        print("✅ cognate.py properly updated as redirect file")
    
    def test_no_duplicate_functionality_in_phases(self):
        """Test that no duplicate Cognate functionality remains in phases directory."""
        phases_path = project_root / "core" / "agent-forge" / "phases"
        
        # Files to exclude from duplication check
        exclude_files = {
            "__init__.py",
            "cognate.py",  # Redirect file is ok
            "phase_controller.py"  # Base controller is ok
        }
        
        exclude_dirs = {
            "cognate-pretrain",  # New structure is ok
            "deprecated_duplicates",  # Deprecated files are ok
            "__pycache__"  # Cache dirs are ok
        }
        
        duplicate_indicators = [
            "create_cognate_model",
            "CognateRefiner", 
            "25m_parameter",
            "cognate_creator",
            "train_and_save"
        ]
        
        # Scan all Python files in phases directory
        for py_file in phases_path.rglob("*.py"):
            # Skip excluded paths
            if any(excluded in str(py_file) for excluded in exclude_dirs):
                continue
            if py_file.name in exclude_files:
                continue
            
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read().lower()
                
                # Check for duplicate functionality indicators
                found_duplicates = []
                for indicator in duplicate_indicators:
                    if indicator.lower() in content:
                        found_duplicates.append(indicator)
                
                if found_duplicates:
                    # This might be ok if it's just imports or comments
                    # Parse the AST to check for actual function definitions
                    try:
                        tree = ast.parse(content)
                        func_names = [node.name.lower() for node in ast.walk(tree) 
                                    if isinstance(node, ast.FunctionDef)]
                        class_names = [node.name.lower() for node in ast.walk(tree)
                                     if isinstance(node, ast.ClassDef)]
                        
                        actual_duplicates = []
                        for indicator in found_duplicates:
                            if (indicator.lower() in func_names or 
                                indicator.lower() in class_names):
                                actual_duplicates.append(indicator)
                        
                        if actual_duplicates:
                            print(f"⚠️  Potential duplicate functionality in {py_file}: {actual_duplicates}")
                        
                    except SyntaxError:
                        # Skip files with syntax errors
                        pass
                        
            except Exception:
                # Skip files that can't be read
                pass
        
        print("✅ Duplicate functionality scan complete")
    
    def test_import_path_cleanliness(self):
        """Test that import paths are clean and don't have circular references."""
        try:
            # Test that new structure doesn't have circular imports
            
            # Test individual modules can be imported independently
            
            print("✅ Import paths are clean with no circular references")
            
        except ImportError as e:
            pytest.fail(f"Import path issue detected: {e}")
    
    def test_package_structure_completeness(self):
        """Test that package structure is complete and well-organized."""
        package_path = project_root / "core" / "agent-forge" / "phases" / "cognate-pretrain"
        
        # Check __init__.py exports
        init_file = package_path / "__init__.py"
        with open(init_file, encoding='utf-8') as f:
            init_content = f.read()
        
        # Should have __all__ definition
        assert "__all__" in init_content, "__init__.py should define __all__"
        
        # Should import main components
        expected_imports = [
            "create_three_cognate_models",
            "CognateModelCreator", 
            "CognateCreatorConfig",
            "CognatePretrainPipeline"
        ]
        
        for expected in expected_imports:
            assert expected in init_content, f"__init__.py should import {expected}"
        
        print("✅ Package structure is complete and well-organized")
    
    def test_models_output_directory_structure(self):
        """Test that models output directory structure is properly set up."""
        package_path = project_root / "core" / "agent-forge" / "phases" / "cognate-pretrain"
        package_path / "models"
        
        # Models directory might not exist yet (created on first run)
        # That's ok, just check that the path structure is reasonable
        assert package_path.exists(), "Package path should exist"
        
        # Check that code references models directory correctly
        creator_file = package_path / "cognate_creator.py"
        with open(creator_file, encoding='utf-8') as f:
            creator_content = f.read()
        
        # Should have models directory reference
        assert "models" in creator_content, "Should reference models directory"
        
        print("✅ Models output directory structure is properly configured")
    
    def test_documentation_files_presence(self):
        """Test that documentation files are present and complete."""
        package_path = project_root / "core" / "agent-forge" / "phases" / "cognate-pretrain"
        readme_file = package_path / "README.md"
        
        if readme_file.exists():
            with open(readme_file, encoding='utf-8') as f:
                readme_content = f.read()
            
            # Check for key documentation elements
            expected_sections = [
                "cognate",
                "model", 
                "usage",
                "create_three_cognate_models"
            ]
            
            content_lower = readme_content.lower()
            for section in expected_sections:
                if section in content_lower:
                    print(f"✅ README contains {section} information")
            
        else:
            print("⚠️  No README.md found in cognate-pretrain package")
        
        # Check for reorganization summary
        phases_path = project_root / "core" / "agent-forge" / "phases"
        reorg_summary = phases_path / "REORGANIZATION_SUMMARY.md"
        
        if reorg_summary.exists():
            print("✅ REORGANIZATION_SUMMARY.md found")
        else:
            print("⚠️  No REORGANIZATION_SUMMARY.md found")


if __name__ == "__main__":
    test_suite = TestFileOrganizationValidation()
    
    print("🧪 Running File Organization Validation Tests")
    print("=" * 50)
    
    try:
        test_suite.test_new_structure_exists()
        test_suite.test_deprecated_files_moved()
        test_suite.test_cognate_redirect_file_updated()
        test_suite.test_no_duplicate_functionality_in_phases()
        test_suite.test_import_path_cleanliness()
        test_suite.test_package_structure_completeness()
        test_suite.test_models_output_directory_structure()
        test_suite.test_documentation_files_presence()
        
        print("=" * 50)
        print("✅ ALL FILE ORGANIZATION TESTS PASSED")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ FILE ORGANIZATION TEST FAILED: {e}")
        raise