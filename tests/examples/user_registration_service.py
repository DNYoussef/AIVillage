"""
UserRegistrationService Implementation
====================================

Example service class for demonstrating TDD London School testing patterns.
This is the implementation driven by the tests in tdd_london_example.py.
"""

from typing import Dict, Any, Protocol


class DuplicateEmailError(Exception):
    """Raised when attempting to register with an existing email."""
    pass


class EmailServiceError(Exception):
    """Raised when email service operations fail."""
    pass


# Protocol definitions (contracts discovered through TDD)
class UserRepository(Protocol):
    """Repository for user data operations."""
    
    def find_by_email(self, email: str) -> Dict[str, Any] | None:
        """Find user by email address."""
        ...
    
    def save(self, *, email: str, hashed_password: str, name: str) -> Dict[str, Any]:
        """Save new user and return user data with ID."""
        ...
    
    def delete(self, user_id: str) -> bool:
        """Delete user by ID (for rollback scenarios)."""
        ...


class EmailService(Protocol):
    """Service for email operations."""
    
    def send_welcome_email(self, user_id: str) -> bool:
        """Send welcome email to newly registered user."""
        ...


class PasswordHasher(Protocol):
    """Service for password hashing operations."""
    
    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        ...


class AuditLogger(Protocol):
    """Service for audit logging."""
    
    def log_registration_success(self, user_id: str) -> None:
        """Log successful registration event."""
        ...
    
    def log_registration_failure(self, email: str, reason: str) -> None:
        """Log failed registration attempt."""
        ...


class ValidationService(Protocol):
    """Service for input validation."""
    
    def validate_registration_data(self, data: Dict[str, Any]) -> bool:
        """Validate registration data format and business rules."""
        ...


class UserRegistrationService:
    """
    Service for user registration operations.
    
    This class demonstrates the London School approach:
    - Depends on abstractions (protocols) not concrete implementations
    - Coordinates between collaborators
    - Focuses on behavior and workflows rather than data structures
    """
    
    def __init__(
        self,
        user_repository: UserRepository,
        email_service: EmailService,
        password_hasher: PasswordHasher,
        audit_logger: AuditLogger,
        validation_service: ValidationService | None = None
    ):
        self._user_repository = user_repository
        self._email_service = email_service
        self._password_hasher = password_hasher
        self._audit_logger = audit_logger
        self._validation_service = validation_service

    def register_user(self, registration_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new user following the business workflow.
        
        Workflow:
        1. Validate input data (if validation service available)
        2. Check if email already exists
        3. Hash the password
        4. Save user to repository
        5. Send welcome email
        6. Log success
        
        Returns:
            Dict containing registration result with user_id and success status
            
        Raises:
            DuplicateEmailError: If email already exists
            EmailServiceError: If email service fails
        """
        email = registration_data['email']
        password = registration_data['password']
        name = registration_data['name']
        
        # Optional validation (discovered through outside-in TDD)
        if self._validation_service:
            is_valid = self._validation_service.validate_registration_data(registration_data)
            if not is_valid:
                raise ValueError("Invalid registration data")
        
        # Check for duplicate email
        existing_user = self._user_repository.find_by_email(email)
        if existing_user is not None:
            raise DuplicateEmailError(f"Email {email} already exists")
        
        # Hash password
        hashed_password = self._password_hasher.hash_password(password)
        
        # Save user
        saved_user = self._user_repository.save(
            email=email,
            hashed_password=hashed_password,
            name=name
        )
        
        user_id = saved_user['id']
        
        # Send welcome email with rollback on failure
        try:
            email_sent = self._email_service.send_welcome_email(user_id)
            if not email_sent:
                raise EmailServiceError("Failed to send welcome email")
        except Exception as e:
            # Rollback: delete the created user
            self._user_repository.delete(user_id)
            self._audit_logger.log_registration_failure(email, f"Email service failed: {str(e)}")
            raise EmailServiceError(f"Registration failed due to email service: {str(e)}")
        
        # Log successful registration
        self._audit_logger.log_registration_success(user_id)
        
        return {
            'success': True,
            'user_id': user_id,
            'created_at': saved_user.get('created_at'),
            'message': 'User registered successfully'
        }


# Additional example: Order Processing Service for more complex workflows
class OrderProcessingService:
    """
    Example of a more complex service with multiple collaboration patterns.
    Demonstrates advanced London School testing scenarios.
    """
    
    def __init__(
        self,
        order_repository,
        payment_gateway,
        inventory_service,
        shipping_service,
        notification_service,
        audit_logger
    ):
        self._order_repository = order_repository
        self._payment_gateway = payment_gateway
        self._inventory_service = inventory_service
        self._shipping_service = shipping_service
        self._notification_service = notification_service
        self._audit_logger = audit_logger

    def process_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an order with complex coordination between services.
        
        This method demonstrates:
        - Multi-step workflows with rollback capabilities
        - Complex collaboration patterns
        - Error handling with partial completion
        """
        order_id = order_data['order_id']
        customer_id = order_data['customer_id']
        items = order_data['items']
        payment_info = order_data['payment']
        
        # Reserve inventory first
        reservation_ids = []
        try:
            for item in items:
                reservation_id = self._inventory_service.reserve_item(
                    item['product_id'], 
                    item['quantity']
                )
                reservation_ids.append(reservation_id)
        except Exception as e:
            # Rollback any successful reservations
            for res_id in reservation_ids:
                self._inventory_service.cancel_reservation(res_id)
            raise Exception(f"Inventory reservation failed: {str(e)}")
        
        # Process payment
        try:
            payment_result = self._payment_gateway.charge(
                customer_id=customer_id,
                amount=order_data['total_amount'],
                payment_info=payment_info
            )
        except Exception as e:
            # Rollback inventory reservations
            for res_id in reservation_ids:
                self._inventory_service.cancel_reservation(res_id)
            raise Exception(f"Payment processing failed: {str(e)}")
        
        # Save order
        try:
            order_record = self._order_repository.save_order({
                'order_id': order_id,
                'customer_id': customer_id,
                'items': items,
                'payment_id': payment_result['payment_id'],
                'status': 'confirmed'
            })
        except Exception as e:
            # Rollback payment and inventory
            self._payment_gateway.refund(payment_result['payment_id'])
            for res_id in reservation_ids:
                self._inventory_service.cancel_reservation(res_id)
            raise Exception(f"Order save failed: {str(e)}")
        
        # Schedule shipping
        try:
            shipping_id = self._shipping_service.schedule_shipment({
                'order_id': order_id,
                'items': items,
                'address': order_data['shipping_address']
            })
        except Exception as e:
            # Log warning but don't fail the order - shipping can be retried
            self._audit_logger.log_warning(f"Shipping scheduling failed for order {order_id}: {str(e)}")
            shipping_id = None
        
        # Send confirmation notification
        try:
            self._notification_service.send_order_confirmation(
                customer_id=customer_id,
                order_id=order_id,
                order_details=order_record
            )
        except Exception as e:
            # Log warning but don't fail the order - notification can be retried
            self._audit_logger.log_warning(f"Notification failed for order {order_id}: {str(e)}")
        
        # Log successful order processing
        self._audit_logger.log_order_success(order_id)
        
        return {
            'success': True,
            'order_id': order_id,
            'payment_id': payment_result['payment_id'],
            'shipping_id': shipping_id,
            'status': 'confirmed'
        }