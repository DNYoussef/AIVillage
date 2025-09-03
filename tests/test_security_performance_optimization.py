#!/usr/bin/env python3
"""
Test suite for Phase 3 Security Performance Optimizations

Validates that security script performance improvements work correctly:
- Progress logging every 500 files processed
- Early directory skipping for excluded directories  
- File size limits to skip files >2MB
- Timeout adjustments from 180s to 240s
"""

import pytest
import tempfile
import time
from pathlib import Path
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from validate_secret_sanitization import SecretSanitizationValidator
except ImportError:
    pytest.skip("Security validation script not available", allow_module_level=True)


class TestSecurityPerformanceOptimizations:
    """Test security script performance optimizations."""
    
    def test_file_size_limit_optimization(self):
        """Test that files >2MB are skipped for performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = SecretSanitizationValidator(temp_dir)
            
            # Create a large file >2MB
            large_file = Path(temp_dir) / "large_file.py"
            with open(large_file, 'w') as f:
                # Write >2MB of content
                for i in range(100000):
                    f.write(f"# This is line {i} with some content to make it large\n")
            
            # Verify file is >2MB
            file_size_mb = large_file.stat().st_size / (1024 * 1024)
            assert file_size_mb > 2.0, f"Test file should be >2MB, got {file_size_mb}MB"
            
            # Test skip detection
            should_skip, reason = validator._should_skip_file(large_file)
            assert should_skip, "Large file should be skipped"
            assert "too large" in reason.lower(), f"Skip reason should mention size: {reason}"
    
    def test_excluded_directory_optimization(self):
        """Test that files in excluded directories are skipped early."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = SecretSanitizationValidator(temp_dir)
            
            # Create files in excluded directories
            excluded_dirs = ['.git', '__pycache__', 'node_modules', '.venv', 'target']
            
            for excluded_dir in excluded_dirs:
                excluded_path = Path(temp_dir) / excluded_dir / "file.py"
                excluded_path.parent.mkdir(exist_ok=True)
                excluded_path.write_text("# some content")
                
                should_skip, reason = validator._should_skip_file(excluded_path)
                assert should_skip, f"File in {excluded_dir} should be skipped"
                assert excluded_dir in reason, f"Skip reason should mention {excluded_dir}: {reason}"
    
    def test_normal_files_not_skipped(self):
        """Test that normal files are not skipped."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = SecretSanitizationValidator(temp_dir)
            
            # Create a normal small file
            normal_file = Path(temp_dir) / "normal_file.py"
            normal_file.write_text("# A normal Python file\nprint('hello')")
            
            should_skip, reason = validator._should_skip_file(normal_file)
            assert not should_skip, f"Normal file should not be skipped: {reason}"
    
    def test_performance_timing_tracking(self):
        """Test that processing time is tracked."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = SecretSanitizationValidator(temp_dir)
            
            # Create a test file
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("# Test file\npassword = 'test123'  # pragma: allowlist secret")
            
            # Validate file
            start_time = time.time()
            result = validator.validate_file(test_file)
            end_time = time.time()
            
            # Check timing is tracked
            assert "processing_time" in result
            assert result["processing_time"] > 0
            assert result["processing_time"] <= (end_time - start_time) + 0.1  # Small tolerance
    
    def test_skip_tracking_in_results(self):
        """Test that skipped files are tracked in results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = SecretSanitizationValidator(temp_dir)
            
            # Create a file that will be skipped (large file)
            large_file = Path(temp_dir) / "large.py"
            with open(large_file, 'w') as f:
                # Write >2MB
                content = "# " + "x" * 1024 * 1024 * 3  # 3MB of content
                f.write(content)
            
            result = validator.validate_file(large_file)
            
            # Check skip tracking
            assert result["skipped"] is True
            assert "skip_reason" in result
            assert "too large" in result["skip_reason"].lower()
    
    def test_progress_logging_integration(self, caplog):
        """Test that progress logging works during validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override target_files to test with temporary files
            validator = SecretSanitizationValidator(temp_dir)
            
            # Create test files
            test_files = []
            for i in range(3):
                test_file = Path(temp_dir) / f"test_{i}.py"
                test_file.write_text(f"# Test file {i}\nprint('test')")
                test_files.append(f"test_{i}.py")
            
            # Override target_files for this test
            validator.target_files = test_files
            
            # Run validation
            with caplog.at_level('INFO'):
                results = validator.validate_all_files()
            
            # Check that progress logging occurred
            log_messages = [record.message for record in caplog.records]
            progress_logs = [msg for msg in log_messages if "Validating:" in msg or "Progress:" in msg]
            
            assert len(progress_logs) >= 3, f"Expected progress logs, got: {log_messages}"
    
    def test_validation_summary_includes_performance_metrics(self):
        """Test that validation summary includes performance metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = SecretSanitizationValidator(temp_dir)
            
            # Create a small test file
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("# Test content")
            
            # Override target_files
            validator.target_files = ["test.py"]
            
            results = validator.validate_all_files()
            
            # Check performance metrics are included
            summary = results["validation_summary"]
            assert "total_processing_time" in summary
            assert "wall_clock_time" in summary
            assert "files_skipped" in summary
            
            # Verify metrics are reasonable
            assert summary["total_processing_time"] >= 0
            assert summary["wall_clock_time"] >= 0
            assert summary["files_skipped"] >= 0


class TestSecurityOptimizationIntegration:
    """Integration tests for security performance optimizations."""
    
    def test_excluded_directories_list_complete(self):
        """Test that excluded directories list covers common build/cache dirs."""
        validator = SecretSanitizationValidator(".")
        
        expected_excluded = {
            '.git', '.github', '__pycache__', '.pytest_cache',
            'node_modules', '.venv', 'venv', 'env', '.env',
            'target', 'build', 'dist', 'coverage', '.nyc_output',
            'artifacts', 'logs', 'tmp', 'temp', '.tmp'
        }
        
        assert validator.excluded_dirs >= expected_excluded, \
            f"Missing excluded dirs: {expected_excluded - validator.excluded_dirs}"
    
    def test_file_size_limit_reasonable(self):
        """Test that file size limit is reasonable (2MB)."""
        validator = SecretSanitizationValidator(".")
        
        # Should be 2MB
        assert validator.max_file_size_mb == 2.0
        assert validator.max_file_size_mb >= 1.0  # At least 1MB
        assert validator.max_file_size_mb <= 5.0  # But not too large
    
    def test_progress_interval_reasonable(self):
        """Test that progress logging interval is reasonable."""
        validator = SecretSanitizationValidator(".")
        
        # Should be every 500 files
        assert validator.progress_interval == 500
        assert validator.progress_interval >= 100  # Not too frequent
        assert validator.progress_interval <= 1000  # Not too infrequent


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])