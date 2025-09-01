"""
TDD London School Implementation Example
======================================

Demonstrates outside-in TDD with behavior verification and mock-driven development.
This example shows how to test a UserRegistrationService using London School principles.
"""

import pytest
from typing import Dict, Any
from tests.fixtures.tdd_london_mocks import MockFactory, ContractTestingMixin


class TestUserRegistrationService(ContractTestingMixin):
    """
    Example test class demonstrating TDD London School patterns.
    Tests focus on behavior verification and object collaboration.
    """

    @pytest.fixture
    def user_registration_collaborators(self, mock_factory):
        """Setup collaborators for UserRegistrationService."""
        return mock_factory.create_collaborator_set(
            'user_repository',
            'email_service', 
            'password_hasher',
            'audit_logger'
        )

    @pytest.fixture
    def valid_registration_data(self):
        """Standard valid registration data for testing."""
        return {
            'email': 'user@example.com',
            'password': 'secure_password_123',
            'name': 'Test User',
            'terms_accepted': True
        }

    def test_successful_user_registration_behavior(self, user_registration_collaborators, valid_registration_data):
        """
        Test successful user registration workflow using behavior verification.
        
        This test follows London School principles:
        1. Setup collaborator expectations
        2. Exercise the system under test
        3. Verify interactions occurred as expected
        """
        # Arrange - Setup collaborator behaviors
        collaborators = user_registration_collaborators
        
        # Configure repository mock to return None (user doesn't exist)
        collaborators['user_repository']._mock.find_by_email.return_value = None
        collaborators['user_repository']._mock.save.return_value = {
            'id': 'user_123',
            'email': valid_registration_data['email'],
            'created_at': '2025-09-01T12:00:00Z'
        }
        
        # Configure password hasher
        collaborators['password_hasher']._mock.hash_password.return_value = 'hashed_password_abc'
        
        # Configure email service
        collaborators['email_service']._mock.send_welcome_email.return_value = True
        
        # Setup interaction expectations
        collaborators['user_repository'].expect_interaction('find_by_email', valid_registration_data['email'])
        collaborators['password_hasher'].expect_interaction('hash_password', valid_registration_data['password'])
        collaborators['user_repository'].expect_interaction('save', 
            email=valid_registration_data['email'],
            hashed_password='hashed_password_abc',
            name=valid_registration_data['name']
        )
        collaborators['email_service'].expect_interaction('send_welcome_email', 'user_123')
        collaborators['audit_logger'].expect_interaction('log_registration_success', 'user_123')

        # Act - Exercise the system under test
        from tests.examples.user_registration_service import UserRegistrationService
        
        service = UserRegistrationService(
            user_repository=collaborators['user_repository']._mock,
            email_service=collaborators['email_service']._mock,
            password_hasher=collaborators['password_hasher']._mock,
            audit_logger=collaborators['audit_logger']._mock
        )
        
        result = service.register_user(valid_registration_data)

        # Assert - Verify behavior and interactions
        assert result['success'] is True
        assert result['user_id'] == 'user_123'
        assert 'created_at' in result
        
        # Verify collaboration sequence
        expected_sequence = [
            "find_by_email(('user@example.com',), {})",
            "hash_password(('secure_password_123',), {})",
            "save((), {'email': 'user@example.com', 'hashed_password': 'hashed_password_abc', 'name': 'Test User'})",
            "send_welcome_email(('user_123',), {})",
            "log_registration_success(('user_123',), {})"
        ]
        
        # Verify each collaborator was used correctly
        self.assert_interaction_count(collaborators['user_repository'], 'find_by_email', 1)
        self.assert_interaction_count(collaborators['password_hasher'], 'hash_password', 1)
        self.assert_interaction_count(collaborators['user_repository'], 'save', 1)
        self.assert_interaction_count(collaborators['email_service'], 'send_welcome_email', 1)
        self.assert_interaction_count(collaborators['audit_logger'], 'log_registration_success', 1)

    def test_duplicate_email_registration_failure(self, user_registration_collaborators, valid_registration_data):
        """
        Test registration failure when email already exists.
        
        This test demonstrates:
        1. Behavior when business rules are violated
        2. Partial interaction patterns (some collaborators not called)
        3. Exception handling verification
        """
        # Arrange
        collaborators = user_registration_collaborators
        
        # Configure repository to return existing user
        existing_user = {
            'id': 'existing_user_456',
            'email': valid_registration_data['email'],
            'created_at': '2025-08-01T12:00:00Z'
        }
        collaborators['user_repository']._mock.find_by_email.return_value = existing_user
        
        # Setup interaction expectations (only checking email should occur)
        collaborators['user_repository'].expect_interaction('find_by_email', valid_registration_data['email'])

        # Act & Assert
        from tests.examples.user_registration_service import UserRegistrationService, DuplicateEmailError
        
        service = UserRegistrationService(
            user_repository=collaborators['user_repository']._mock,
            email_service=collaborators['email_service']._mock,
            password_hasher=collaborators['password_hasher']._mock,
            audit_logger=collaborators['audit_logger']._mock
        )
        
        with pytest.raises(DuplicateEmailError) as exc_info:
            service.register_user(valid_registration_data)
        
        assert str(exc_info.value) == f"Email {valid_registration_data['email']} already exists"
        
        # Verify interaction behavior - these should NOT have been called
        self.assert_interaction_count(collaborators['user_repository'], 'find_by_email', 1)
        self.assert_never_called(collaborators['password_hasher'], 'hash_password')
        self.assert_never_called(collaborators['user_repository'], 'save')
        self.assert_never_called(collaborators['email_service'], 'send_welcome_email')
        self.assert_never_called(collaborators['audit_logger'], 'log_registration_success')

    def test_email_service_failure_rollback_behavior(self, user_registration_collaborators, valid_registration_data):
        """
        Test system behavior when email service fails.
        
        This test demonstrates:
        1. Partial success scenario handling
        2. Rollback/compensation behavior
        3. Error state interactions
        """
        # Arrange
        collaborators = user_registration_collaborators
        
        # Configure successful initial steps
        collaborators['user_repository']._mock.find_by_email.return_value = None
        collaborators['password_hasher']._mock.hash_password.return_value = 'hashed_password_abc'
        collaborators['user_repository']._mock.save.return_value = {
            'id': 'user_123',
            'email': valid_registration_data['email']
        }
        
        # Configure email service to fail
        collaborators['email_service']._mock.send_welcome_email.side_effect = Exception("Email service unavailable")
        
        # Configure rollback behavior
        collaborators['user_repository']._mock.delete.return_value = True

        # Act
        from tests.examples.user_registration_service import UserRegistrationService, EmailServiceError
        
        service = UserRegistrationService(
            user_repository=collaborators['user_repository']._mock,
            email_service=collaborators['email_service']._mock,
            password_hasher=collaborators['password_hasher']._mock,
            audit_logger=collaborators['audit_logger']._mock
        )
        
        with pytest.raises(EmailServiceError):
            service.register_user(valid_registration_data)

        # Assert - Verify rollback behavior
        self.assert_interaction_count(collaborators['user_repository'], 'find_by_email', 1)
        self.assert_interaction_count(collaborators['password_hasher'], 'hash_password', 1)
        self.assert_interaction_count(collaborators['user_repository'], 'save', 1)
        self.assert_interaction_count(collaborators['email_service'], 'send_welcome_email', 1)
        self.assert_interaction_count(collaborators['user_repository'], 'delete', 1)  # Rollback
        self.assert_interaction_count(collaborators['audit_logger'], 'log_registration_failure', 1)

    @pytest.mark.behavior_verification
    def test_contract_compliance_verification(self, user_registration_collaborators, valid_registration_data):
        """
        Test that service adheres to defined contracts with collaborators.
        
        This test demonstrates:
        1. Contract testing principles
        2. Interface compliance verification
        3. Parameter validation
        """
        # Arrange
        collaborators = user_registration_collaborators
        
        # Configure standard successful flow
        collaborators['user_repository']._mock.find_by_email.return_value = None
        collaborators['password_hasher']._mock.hash_password.return_value = 'hashed_password_abc'
        collaborators['user_repository']._mock.save.return_value = {'id': 'user_123'}
        collaborators['email_service']._mock.send_welcome_email.return_value = True

        # Act
        from tests.examples.user_registration_service import UserRegistrationService
        
        service = UserRegistrationService(
            user_repository=collaborators['user_repository']._mock,
            email_service=collaborators['email_service']._mock,
            password_hasher=collaborators['password_hasher']._mock,
            audit_logger=collaborators['audit_logger']._mock
        )
        
        service.register_user(valid_registration_data)

        # Assert - Contract compliance verification
        self.assert_called_with_contract(
            collaborators['user_repository'], 
            'save',
            email=valid_registration_data['email'],
            hashed_password='hashed_password_abc',
            name=valid_registration_data['name']
        )
        
        self.assert_called_with_contract(
            collaborators['email_service'],
            'send_welcome_email',
            user_id='user_123'
        )

    @pytest.mark.outside_in
    def test_outside_in_development_example(self, mock_factory):
        """
        Example of outside-in TDD development process.
        
        This test demonstrates:
        1. Starting with acceptance criteria
        2. Driving design through tests
        3. Discovering collaborators through testing
        """
        # Start with user story: "As a user, I want to register for an account"
        
        # Acceptance criteria test (outside layer)
        registration_data = {
            'email': 'newuser@example.com',
            'password': 'secure_pass_456',
            'name': 'New User'
        }
        
        # Discover what collaborators we need through testing
        needed_collaborators = mock_factory.create_collaborator_set(
            'user_repository',     # For checking existing users and saving
            'password_hasher',     # For securing passwords
            'email_service',       # For sending welcome emails
            'audit_logger',        # For compliance logging
            'validation_service'   # Discovered: need input validation
        )
        
        # Configure the happy path behavior
        needed_collaborators['validation_service']._mock.validate_registration_data.return_value = True
        needed_collaborators['user_repository']._mock.find_by_email.return_value = None
        needed_collaborators['password_hasher']._mock.hash_password.return_value = 'hashed_456'
        needed_collaborators['user_repository']._mock.save.return_value = {'id': 'user_789'}
        needed_collaborators['email_service']._mock.send_welcome_email.return_value = True

        # This drives us to implement UserRegistrationService with these dependencies
        from tests.examples.user_registration_service import UserRegistrationService
        
        service = UserRegistrationService(
            user_repository=needed_collaborators['user_repository']._mock,
            email_service=needed_collaborators['email_service']._mock,
            password_hasher=needed_collaborators['password_hasher']._mock,
            audit_logger=needed_collaborators['audit_logger']._mock,
            validation_service=needed_collaborators['validation_service']._mock
        )
        
        result = service.register_user(registration_data)
        
        # Verify the acceptance criteria are met
        assert result['success'] is True
        assert result['user_id'] == 'user_789'
        
        # The test drives us to discover and implement all necessary collaborations