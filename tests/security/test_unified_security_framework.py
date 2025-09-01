"""
Comprehensive Test Suite for Unified Security Framework
Tests all consolidated security components with MCP integration
"""

import asyncio
import pytest
from datetime import datetime, UTC
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

# Import the unified security framework
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from security.unified_security_framework import (
    UnifiedSecurityFramework,
    UnifiedSecurityContext,
    UnifiedSecurityEvent,
    SecurityLevel,
    AuthenticationMethod,
    ThreatType,
    create_security_context,
    authenticate_user,
    authorize_operation,
    detect_security_threats
)

from security.consolidated_security_config import (
    ConsolidatedSecurityConfigService,
    SecurityConfiguration,
    ConfigurationCategory,
    ConfigurationEnvironment
)

from security.mcp_security_coordinator import (
    SecurityAutomationOrchestrator,
    GitHubSecurityCoordinator
)


class TestUnifiedSecurityFramework:
    """Test suite for the unified security framework"""
    
    @pytest.fixture
    async def security_framework(self):
        """Create a test security framework instance"""
        framework = UnifiedSecurityFramework()
        await framework.initialize()
        return framework
    
    @pytest.fixture
    def security_context(self):
        """Create a test security context"""
        return create_security_context(
            user_id="test_user",
            roles={"developer", "user"},
            capabilities={"data.read", "model.train", "agent.execute"},
            security_level=SecurityLevel.MEDIUM,
            source_ip="192.168.1.100",
            user_agent="TestAgent/1.0"
        )
    
    @pytest.mark.asyncio
    async def test_framework_initialization(self):
        """Test framework initialization"""
        framework = UnifiedSecurityFramework()
        assert not framework.initialized
        
        await framework.initialize()
        assert framework.initialized
        
        status = await framework.get_security_status()
        assert status["framework_status"] == "operational"
        assert isinstance(status["active_sessions"], int)
        assert isinstance(status["active_threats"], int)
    
    @pytest.mark.asyncio
    async def test_authentication_integration(self, security_framework, security_context):
        """Test unified authentication system"""
        credentials = {
            "password": "SecurePassword123!",
            "hardware_token": "123456"
        }
        
        result = await security_framework.authenticate_user(security_context, credentials)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "session_id" in result
        assert "authentication_score" in result
        assert "methods_used" in result
        
        if result["success"]:
            assert result["authentication_score"] > 0
            assert len(result["methods_used"]) > 0
    
    @pytest.mark.asyncio
    async def test_authorization_integration(self, security_framework, security_context):
        """Test unified authorization system"""
        result = await security_framework.authorize_operation(
            security_context, 
            "data.model", 
            "train"
        )
        
        assert isinstance(result, dict)
        assert "authorized" in result
        assert "reason" in result
        
        if not result["authorized"]:
            assert "missing_capabilities" in result or "policy_violations" in result
    
    @pytest.mark.asyncio
    async def test_threat_detection_integration(self, security_framework, security_context):
        """Test unified threat detection system"""
        activity_data = {
            "failed_attempts": 3,
            "resource_usage": {"cpu_percent": 85, "memory_percent": 70},
            "consensus_messages": [
                {"id": 1, "timestamp": 1234567890, "type": "vote"},
                {"id": 1, "timestamp": 1234567891, "type": "vote"}  # Duplicate
            ]
        }
        
        threats = await security_framework.detect_threats(security_context, activity_data)
        
        assert isinstance(threats, list)
        for threat in threats:
            assert isinstance(threat, UnifiedSecurityEvent)
            assert threat.event_type
            assert threat.security_level in SecurityLevel
            assert 0 <= threat.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_mcp_integration_status(self, security_framework):
        """Test MCP integration status"""
        status = await security_framework.get_security_status()
        
        mcp_integration = status.get("mcp_integration", {})
        assert isinstance(mcp_integration, dict)
        
        # Check MCP service statuses
        expected_services = ["github_enabled", "memory_enabled", "sequential_thinking_enabled", "context7_enabled"]
        for service in expected_services:
            assert service in mcp_integration
            assert isinstance(mcp_integration[service], bool)
    
    @pytest.mark.asyncio
    async def test_security_incident_handling(self, security_framework):
        """Test security incident handling"""
        incident_data = {
            "type": "authentication_failure",
            "severity": "high",
            "details": {
                "user_id": "test_user",
                "failed_attempts": 5,
                "source_ip": "192.168.1.100"
            }
        }
        
        response = await security_framework.handle_security_incident(incident_data)
        
        assert isinstance(response, dict)
        assert "incident_id" in response
        assert "status" in response
        assert "analysis" in response
        assert response["status"] == "processed"
    
    @pytest.mark.asyncio
    async def test_security_config_update(self, security_framework):
        """Test security configuration updates"""
        config_updates = {
            "default_security_level": SecurityLevel.HIGH,
            "max_failed_auth_attempts": 3,
            "session_timeout_hours": 4
        }
        
        await security_framework.update_security_config(config_updates)
        
        # Verify updates were applied
        assert security_framework.security_config["default_security_level"] == SecurityLevel.HIGH
        assert security_framework.security_config["max_failed_auth_attempts"] == 3
        assert security_framework.security_config["session_timeout_hours"] == 4
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self, security_context):
        """Test convenience functions"""
        # Test authenticate_user convenience function
        auth_result = await authenticate_user(
            user_id="test_user",
            credentials={"password": "TestPassword123!"},
            roles={"user"},
            security_level=SecurityLevel.LOW
        )
        assert isinstance(auth_result, dict)
        assert "success" in auth_result
        
        # Test authorize_operation convenience function
        authz_result = await authorize_operation(
            user_id="test_user",
            resource="data.test",
            operation="read",
            roles={"user"},
            capabilities={"data.read"}
        )
        assert isinstance(authz_result, dict)
        assert "authorized" in authz_result
        
        # Test detect_security_threats convenience function
        threats = await detect_security_threats(
            user_id="test_user",
            activity_data={"failed_attempts": 2},
            security_level=SecurityLevel.MEDIUM
        )
        assert isinstance(threats, list)


class TestConsolidatedSecurityConfig:
    """Test suite for consolidated security configuration service"""
    
    @pytest.fixture
    async def config_service(self):
        """Create a test configuration service instance"""
        service = ConsolidatedSecurityConfigService()
        await service.initialize()
        return service
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test configuration service initialization"""
        service = ConsolidatedSecurityConfigService()
        assert not service.initialized
        
        await service.initialize()
        assert service.initialized
        
        status = await service.get_service_status()
        assert status["service_status"] == "operational"
    
    @pytest.mark.asyncio
    async def test_configuration_creation(self, config_service):
        """Test security configuration creation"""
        config_data = {
            "password_policy": {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True
            },
            "mfa_settings": {
                "enabled": True,
                "methods": ["TOTP", "hardware_token"]
            }
        }
        
        config = await config_service.create_configuration(
            name="test_auth_config",
            category=ConfigurationCategory.AUTHENTICATION,
            environment=ConfigurationEnvironment.TESTING,
            configuration=config_data,
            description="Test authentication configuration"
        )
        
        assert config.name == "test_auth_config"
        assert config.category == ConfigurationCategory.AUTHENTICATION
        assert config.environment == ConfigurationEnvironment.TESTING
        assert config.configuration == config_data
        assert config.get_checksum()
    
    @pytest.mark.asyncio
    async def test_configuration_retrieval(self, config_service):
        """Test security configuration retrieval"""
        # Create a configuration first
        config_data = {"test_setting": "test_value"}
        created_config = await config_service.create_configuration(
            name="test_retrieval",
            category=ConfigurationCategory.AUTHORIZATION,
            environment=ConfigurationEnvironment.TESTING,
            configuration=config_data
        )
        
        # Retrieve by config_id
        retrieved_config = await config_service.get_configuration(config_id=created_config.config_id)
        assert retrieved_config is not None
        assert retrieved_config.config_id == created_config.config_id
        
        # Retrieve by filters
        filtered_config = await config_service.get_configuration(
            name="test_retrieval",
            category=ConfigurationCategory.AUTHORIZATION,
            environment=ConfigurationEnvironment.TESTING
        )
        assert filtered_config is not None
        assert filtered_config.name == "test_retrieval"
    
    @pytest.mark.asyncio
    async def test_configuration_update(self, config_service):
        """Test security configuration update"""
        # Create a configuration
        config_data = {"setting1": "value1", "setting2": "value2"}
        config = await config_service.create_configuration(
            name="test_update",
            category=ConfigurationCategory.CONSENSUS,
            environment=ConfigurationEnvironment.TESTING,
            configuration=config_data
        )
        
        original_checksum = config.get_checksum()
        original_version = config.version
        
        # Update the configuration
        updates = {"setting1": "updated_value1", "setting3": "new_value3"}
        updated_config = await config_service.update_configuration(
            config.config_id,
            updates,
            updated_by="test_user"
        )
        
        assert updated_config.configuration["setting1"] == "updated_value1"
        assert updated_config.configuration["setting2"] == "value2"  # Unchanged
        assert updated_config.configuration["setting3"] == "new_value3"  # New
        assert updated_config.get_checksum() != original_checksum
        assert float(updated_config.version) > float(original_version)
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, config_service):
        """Test configuration validation"""
        # Test valid configuration
        valid_config = {
            "password_policy": {"min_length": 12},
            "mfa_settings": {"enabled": True},
            "session_management": {"timeout_minutes": 60}
        }
        
        config = await config_service.create_configuration(
            name="test_valid",
            category=ConfigurationCategory.AUTHENTICATION,
            environment=ConfigurationEnvironment.TESTING,
            configuration=valid_config
        )
        assert config is not None
        
        # Test invalid configuration (should raise exception)
        invalid_config = {
            "password_policy": {"min_length": 4},  # Too short
            "mfa_settings": {"enabled": "not_a_boolean"},  # Wrong type
            # Missing required field: session_management
        }
        
        with pytest.raises(ValueError):
            await config_service.create_configuration(
                name="test_invalid",
                category=ConfigurationCategory.AUTHENTICATION,
                environment=ConfigurationEnvironment.TESTING,
                configuration=invalid_config
            )
    
    @pytest.mark.asyncio
    async def test_configuration_export_import(self, config_service):
        """Test configuration export and import"""
        # Create a configuration
        config_data = {"export_test": "export_value"}
        original_config = await config_service.create_configuration(
            name="test_export",
            category=ConfigurationCategory.GATEWAY,
            environment=ConfigurationEnvironment.TESTING,
            configuration=config_data
        )
        
        # Export the configuration
        export_data = await config_service.export_configuration(original_config.config_id)
        
        assert "metadata" in export_data
        assert "configuration" in export_data
        assert "history" in export_data
        assert export_data["configuration"]["name"] == "test_export"
        
        # Import the configuration
        imported_config = await config_service.import_configuration(
            export_data,
            imported_by="test_importer"
        )
        
        assert imported_config.name == original_config.name
        assert imported_config.category == original_config.category
        assert imported_config.configuration == original_config.configuration
        assert imported_config.config_id != original_config.config_id  # Different ID
    
    @pytest.mark.asyncio
    async def test_configuration_listing(self, config_service):
        """Test configuration listing with filters"""
        # Create multiple configurations
        auth_config = await config_service.create_configuration(
            name="auth_list_test",
            category=ConfigurationCategory.AUTHENTICATION,
            environment=ConfigurationEnvironment.TESTING,
            configuration={"auth_setting": "auth_value"}
        )
        
        authz_config = await config_service.create_configuration(
            name="authz_list_test",
            category=ConfigurationCategory.AUTHORIZATION,
            environment=ConfigurationEnvironment.TESTING,
            configuration={"authz_setting": "authz_value"}
        )
        
        # List all configurations
        all_configs = await config_service.list_configurations()
        assert len(all_configs) >= 2
        
        # List by category
        auth_configs = await config_service.list_configurations(
            category=ConfigurationCategory.AUTHENTICATION
        )
        auth_names = [config.name for config in auth_configs]
        assert "auth_list_test" in auth_names
        
        # List by environment
        test_configs = await config_service.list_configurations(
            environment=ConfigurationEnvironment.TESTING
        )
        test_names = [config.name for config in test_configs]
        assert "auth_list_test" in test_names
        assert "authz_list_test" in test_names
    
    @pytest.mark.asyncio
    async def test_context7_integration(self, config_service):
        """Test Context7 MCP integration"""
        # Get cache statistics
        cache_stats = await config_service.context7.get_cache_statistics()
        
        assert isinstance(cache_stats, dict)
        assert "enabled" in cache_stats
        assert "cache_prefix" in cache_stats
        assert "default_ttl" in cache_stats
        
        # Test configuration caching
        config_data = {"cache_test": "cache_value"}
        config = await config_service.create_configuration(
            name="test_cache",
            category=ConfigurationCategory.NETWORKING,
            environment=ConfigurationEnvironment.TESTING,
            configuration=config_data
        )
        
        # Configuration should be cached automatically
        assert config.cache_key is not None


class TestSecurityAutomationOrchestrator:
    """Test suite for security automation orchestrator"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create a test orchestrator instance"""
        orchestrator = SecurityAutomationOrchestrator()
        await orchestrator.initialize()
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = SecurityAutomationOrchestrator()
        
        await orchestrator.initialize()
        
        status = await orchestrator.get_automation_status()
        assert status["orchestrator_status"] == "operational"
        assert "automated_workflows" in status
        assert "security_metrics" in status
    
    @pytest.mark.asyncio
    async def test_security_event_handling(self, orchestrator):
        """Test security event handling"""
        event_data = {
            "event_type": "brute_force_attack",
            "severity": "high",
            "details": {
                "user_id": "attacker_user",
                "failed_attempts": 10,
                "source_ip": "10.0.0.1"
            }
        }
        
        event_id = await orchestrator.handle_security_event(event_data)
        
        assert event_id is not None
        assert isinstance(event_id, str)
        
        # Check that metrics were updated
        status = await orchestrator.get_automation_status()
        assert "security_events" in status["security_metrics"]
    
    @pytest.mark.asyncio
    async def test_github_coordinator_integration(self, orchestrator):
        """Test GitHub coordinator integration"""
        github_coordinator = orchestrator.github_coordinator
        
        # Test issue creation
        event_data = {
            "event_type": "vulnerability_detected",
            "title": "SQL Injection Vulnerability",
            "severity": "critical",
            "technical_details": {
                "location": "user_input_handler.py",
                "cve_id": "CVE-2023-12345"
            }
        }
        
        issue_id = await github_coordinator.create_security_issue(event_data)
        assert issue_id is not None
        
        # Test policy validation
        validation_results = await github_coordinator.validate_security_policies()
        assert isinstance(validation_results, dict)
        assert "policies_validated" in validation_results
        assert "policies_passed" in validation_results
        assert "policies_failed" in validation_results
    
    @pytest.mark.asyncio
    async def test_automated_mitigations(self, orchestrator):
        """Test automated security mitigations"""
        # Test DoS mitigation
        dos_event = {"event_type": "dos_attack", "source_ip": "10.0.0.1"}
        mitigations = await orchestrator._apply_automated_mitigations(dos_event)
        assert "rate_limiting_activated" in mitigations
        
        # Test authentication mitigation
        auth_event = {"event_type": "brute_force_authentication"}
        mitigations = await orchestrator._apply_automated_mitigations(auth_event)
        assert "account_restrictions_applied" in mitigations
        
        # Test intrusion mitigation
        intrusion_event = {"event_type": "unauthorized_access"}
        mitigations = await orchestrator._apply_automated_mitigations(intrusion_event)
        assert "network_isolation_activated" in mitigations
        
        # Test data protection mitigation
        data_event = {"event_type": "pii_exposure"}
        mitigations = await orchestrator._apply_automated_mitigations(data_event)
        assert "data_access_restricted" in mitigations


class TestSecurityIntegration:
    """Integration tests for the complete security system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_workflow(self):
        """Test complete end-to-end security workflow"""
        # 1. Initialize all components
        framework = UnifiedSecurityFramework()
        await framework.initialize()
        
        config_service = ConsolidatedSecurityConfigService()
        await config_service.initialize()
        
        orchestrator = SecurityAutomationOrchestrator()
        await orchestrator.initialize()
        
        # 2. Create security context
        context = create_security_context(
            user_id="integration_test_user",
            roles={"developer"},
            capabilities={"data.read", "model.train"},
            security_level=SecurityLevel.HIGH
        )
        
        # 3. Create security configuration
        config = await config_service.create_configuration(
            name="integration_test_config",
            category=ConfigurationCategory.AUTHENTICATION,
            environment=ConfigurationEnvironment.TESTING,
            configuration={
                "password_policy": {"min_length": 16},
                "mfa_settings": {"enabled": True},
                "session_management": {"timeout_minutes": 30}
            }
        )
        
        # 4. Test authentication
        auth_result = await framework.authenticate_user(context, {
            "password": "IntegrationTestPassword123!",
            "hardware_token": "654321"
        })
        
        # 5. Test authorization
        authz_result = await framework.authorize_operation(
            context, "data.model", "train"
        )
        
        # 6. Test threat detection
        threats = await framework.detect_threats(context, {
            "resource_usage": {"cpu_percent": 95}
        })
        
        # 7. Handle any detected threats
        if threats:
            for threat in threats:
                event_data = {
                    "event_type": threat.event_type,
                    "severity": threat.security_level.value,
                    "details": threat.details
                }
                event_id = await orchestrator.handle_security_event(event_data)
                assert event_id is not None
        
        # 8. Verify system status
        framework_status = await framework.get_security_status()
        config_status = await config_service.get_service_status()
        orchestrator_status = await orchestrator.get_automation_status()
        
        assert framework_status["framework_status"] == "operational"
        assert config_status["service_status"] == "operational"
        assert orchestrator_status["orchestrator_status"] == "operational"
    
    @pytest.mark.asyncio
    async def test_cross_system_configuration_sync(self):
        """Test configuration synchronization across systems"""
        # Initialize services
        framework = UnifiedSecurityFramework()
        await framework.initialize()
        
        config_service = ConsolidatedSecurityConfigService()
        await config_service.initialize()
        
        # Create authentication configuration
        auth_config = await config_service.create_configuration(
            name="sync_test_auth",
            category=ConfigurationCategory.AUTHENTICATION,
            environment=ConfigurationEnvironment.PRODUCTION,
            configuration={
                "password_policy": {"min_length": 14},
                "session_management": {"timeout_minutes": 120}
            }
        )
        
        # Update framework configuration based on stored config
        framework_updates = {
            "session_timeout_hours": 2,  # 120 minutes
            "max_failed_auth_attempts": 3
        }
        
        await framework.update_security_config(framework_updates)
        
        # Verify synchronization
        framework_status = await framework.get_security_status()
        assert framework.security_config["session_timeout_hours"] == 2
    
    @pytest.mark.asyncio
    async def test_mcp_integration_flow(self):
        """Test complete MCP integration workflow"""
        # Initialize framework with MCP integration
        framework = UnifiedSecurityFramework()
        framework.security_config["enable_mcp_integration"] = True
        await framework.initialize()
        
        # Test MCP service status
        status = await framework.get_security_status()
        mcp_status = status["mcp_integration"]
        
        # Verify all MCP services are acknowledged
        assert "github_enabled" in mcp_status
        assert "memory_enabled" in mcp_status
        assert "sequential_thinking_enabled" in mcp_status
        assert "context7_enabled" in mcp_status
        
        # Test security incident with MCP integration
        incident_data = {
            "type": "mcp_integration_test",
            "severity": "medium",
            "details": {"test": "mcp_workflow"}
        }
        
        response = await framework.handle_security_incident(incident_data)
        assert response["status"] == "processed"
        assert "analysis" in response


# Test utilities and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_security_framework_performance():
    """Performance test for security framework operations"""
    import time
    
    framework = UnifiedSecurityFramework()
    await framework.initialize()
    
    context = create_security_context(
        user_id="perf_test_user",
        roles={"user"},
        security_level=SecurityLevel.MEDIUM
    )
    
    # Test authentication performance
    start_time = time.time()
    
    for i in range(100):
        await framework.authenticate_user(context, {
            "password": f"TestPassword{i}!"
        })
    
    auth_time = time.time() - start_time
    
    # Test authorization performance
    start_time = time.time()
    
    for i in range(100):
        await framework.authorize_operation(context, f"resource.{i}", "read")
    
    authz_time = time.time() - start_time
    
    # Performance assertions (adjust thresholds as needed)
    assert auth_time < 10.0  # Should complete 100 auths in under 10 seconds
    assert authz_time < 5.0   # Should complete 100 authz in under 5 seconds
    
    print(f"Authentication performance: {auth_time:.2f}s for 100 operations")
    print(f"Authorization performance: {authz_time:.2f}s for 100 operations")


if __name__ == "__main__":
    # Run specific tests
    asyncio.run(test_security_framework_performance())
    print("All performance tests completed successfully!")