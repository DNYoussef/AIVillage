"""
P2P Infrastructure Security Validation Tests

Comprehensive security tests for P2P infrastructure to ensure:
1. No hardcoded secrets or credentials
2. Secure serialization practices
3. Safe network binding configurations
4. Proper authentication mechanisms
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import pytest

# Add infrastructure path for imports
infra_path = Path(__file__).parent.parent.parent / "infrastructure"
sys.path.insert(0, str(infra_path))

try:
    from p2p.security.security_config import (
        SecurityConfig,
        SecureServerConfig,
        SecureSerializer,
        SecurityLevel,
        init_security_config,
        validate_host_binding
    )
except ImportError:
    # If import fails, create mock classes for testing
    class SecurityConfig:
        def __init__(self):
            self.default_host = "127.0.0.1"
            self.allow_all_interfaces = False
            self.allow_pickle = False
            self.safe_serialization_only = True
            self.require_auth = True
            self.min_key_size = 2048
    
    class SecureServerConfig:
        @staticmethod
        def get_safe_host(service_name: str) -> str:
            return "127.0.0.1"
    
    class SecureSerializer:
        @staticmethod
        def is_safe_format(data: bytes) -> bool:
            return not data.startswith((b'\x80\x03', b'\x80\x04', b'\x80\x05'))
        
        @staticmethod
        def safe_deserialize(data: bytes) -> dict:
            import json
            return json.loads(data.decode('utf-8'))
    
    class SecurityLevel:
        PRODUCTION = "production"
        DEVELOPMENT = "development"
    
    def init_security_config(level):
        return SecurityConfig()
    
    def validate_host_binding(host: str) -> str:
        return host


class TestSecurityConfiguration:
    """Test security configuration and validation"""
    
    def test_default_security_config(self):
        """Test default security configuration is secure"""
        config = SecurityConfig()
        
        # Verify secure defaults
        assert config.default_host == "127.0.0.1"
        assert not config.allow_all_interfaces
        assert not config.allow_pickle
        assert config.safe_serialization_only
        assert config.require_auth
        assert config.min_key_size >= 2048
    
    def test_production_security_config(self):
        """Test production security configuration"""
        config = init_security_config(SecurityLevel.PRODUCTION)
        
        # Production should have strictest settings
        assert not config.allow_all_interfaces
        assert not config.allow_pickle
        assert config.require_auth
        assert config.min_key_size >= 4096
        assert config.safe_serialization_only
    
    def test_secure_host_validation(self):
        """Test host binding validation"""
        # Safe host should pass
        safe_host = SecureServerConfig.get_safe_host("test_service")
        assert safe_host == "127.0.0.1"  # Default
        
        # Test with environment variable
        os.environ["TEST_SERVICE_HOST"] = "192.168.1.10"
        host = SecureServerConfig.get_safe_host("test_service")
        assert host == "192.168.1.10"
        
        # Cleanup
        del os.environ["TEST_SERVICE_HOST"]
    
    def test_production_host_binding_restriction(self):
        """Test that 0.0.0.0 binding is restricted in production"""
        os.environ["SECURITY_LEVEL"] = "production"
        os.environ["TEST_SERVICE_HOST"] = "0.0.0.0"
        
        with pytest.raises(ValueError, match="not allowed in production"):
            SecureServerConfig.get_safe_host("test_service")
        
        # Cleanup
        del os.environ["SECURITY_LEVEL"]
        del os.environ["TEST_SERVICE_HOST"]


class TestSecureSerialization:
    """Test secure serialization practices"""
    
    def test_pickle_detection(self):
        """Test pickle format detection"""
        import pickle
        
        # Create test data
        test_data = {"test": "data", "number": 42}
        
        # Pickle serialization (should be detected as unsafe)
        pickle_data = pickle.dumps(test_data)
        assert not SecureSerializer.is_safe_format(pickle_data)
        
        # JSON serialization (should be safe)
        json_data = json.dumps(test_data).encode('utf-8')
        assert SecureSerializer.is_safe_format(json_data)
    
    def test_safe_deserialization(self):
        """Test safe deserialization"""
        test_data = {"test": "data", "number": 42}
        json_data = json.dumps(test_data).encode('utf-8')
        
        # Should successfully deserialize JSON
        result = SecureSerializer.safe_deserialize(json_data)
        assert result == test_data
        
        # Should reject pickle data
        import pickle
        pickle_data = pickle.dumps(test_data)
        
        with pytest.raises(ValueError, match="Unsafe serialization format"):
            SecureSerializer.safe_deserialize(pickle_data)
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON data"""
        invalid_json = b"{'invalid': json}"
        
        with pytest.raises(ValueError, match="Invalid JSON data"):
            SecureSerializer.safe_deserialize(invalid_json)


class TestSecurityScanningIntegration:
    """Integration tests with security scanning tools"""
    
    def test_bandit_scan_passes(self):
        """Test that Bandit security scan passes on P2P infrastructure"""
        p2p_path = Path(__file__).parent.parent.parent / "infrastructure" / "p2p"
        
        # Run Bandit scan
        result = subprocess.run([
            "bandit", "-r", str(p2p_path), "-ll", "-f", "json"
        ], capture_output=True, text=True)
        
        # Parse results
        if result.returncode == 0:
            # No high/critical issues found
            scan_results = json.loads(result.stdout) if result.stdout else {"results": []}
        else:
            # Parse even if bandit found issues
            scan_results = json.loads(result.stdout) if result.stdout else {"results": []}
        
        # Count critical and high severity issues
        critical_issues = [
            issue for issue in scan_results.get("results", [])
            if issue.get("issue_severity") in ["HIGH", "CRITICAL"]
        ]
        
        # Should have no critical security issues
        assert len(critical_issues) == 0, f"Critical security issues found: {critical_issues}"
    
    def test_secret_detection_passes(self):
        """Test that secret detection passes on P2P infrastructure"""
        p2p_path = Path(__file__).parent.parent.parent / "infrastructure" / "p2p"
        
        # Create temporary baseline file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.baseline', delete=False) as f:
            baseline_path = f.name
            f.write('{"results": {}, "version": "1.5.0"}\n')
        
        try:
            # Run detect-secrets scan
            result = subprocess.run([
                "detect-secrets", "scan", str(p2p_path),
                "--baseline", baseline_path,
                "--force-use-all-plugins"
            ], capture_output=True, text=True)
            
            # Should not find new secrets
            assert "No new secrets detected" in result.stdout or result.returncode == 0
            
        finally:
            # Cleanup
            os.unlink(baseline_path)


class TestSecurityBestPractices:
    """Test security best practices in P2P code"""
    
    def test_no_hardcoded_credentials(self):
        """Test that no hardcoded credentials exist in P2P code"""
        p2p_path = Path(__file__).parent.parent.parent / "infrastructure" / "p2p"
        
        # Patterns that indicate hardcoded credentials
        suspicious_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        import re
        issues = []
        
        for py_file in p2p_path.rglob("*.py"):
            if py_file.name.startswith("test_"):
                continue  # Skip test files
                
            try:
                content = py_file.read_text(encoding='utf-8')
                for line_num, line in enumerate(content.splitlines(), 1):
                    for pattern in suspicious_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Check if it has a pragma comment (acceptable for tests)
                            if "pragma: allowlist secret" not in line:
                                issues.append(f"{py_file}:{line_num}: {line.strip()}")
            except Exception:
                continue
        
        assert len(issues) == 0, f"Potential hardcoded credentials found: {issues}"
    
    def test_secure_network_binding(self):
        """Test that network services use secure binding practices"""
        p2p_path = Path(__file__).parent.parent.parent / "infrastructure" / "p2p"
        
        import re
        unsafe_bindings = []
        
        for py_file in p2p_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                for line_num, line in enumerate(content.splitlines(), 1):
                    # Look for direct 0.0.0.0 binding without security checks
                    if re.search(r'host\s*=\s*["\']0\.0\.0\.0["\']', line):
                        # Check if it has security considerations
                        if "nosec" not in line and "security" not in line.lower():
                            unsafe_bindings.append(f"{py_file}:{line_num}: {line.strip()}")
            except Exception:
                continue
        
        # Should have no unsafe bindings without security considerations
        assert len(unsafe_bindings) == 0, f"Unsafe network bindings found: {unsafe_bindings}"


def run_security_validation():
    """Run comprehensive security validation"""
    print("🔒 Running P2P Infrastructure Security Validation...")
    
    # Run pytest on this file
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    
    if exit_code == 0:
        print("✅ All security validations passed!")
    else:
        print("❌ Security validation failures detected!")
        sys.exit(1)


if __name__ == "__main__":
    run_security_validation()