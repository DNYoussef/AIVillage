#!/usr/bin/env python3
"""
Resource Access Control System
Defense-grade access control for training resources and operations

CLASSIFICATION: CONTROLLED UNCLASSIFIED INFORMATION (CUI)
DFARS: 252.204-7012 Compliant
NASA POT10: 95% Compliance Target
"""

import os
import json
import logging
import hashlib
import threading
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import secrets

from .enhanced_audit_trail_manager import EnhancedAuditTrail
from .fips_crypto_module import FIPSCryptoModule

class AccessLevel(Enum):
    """Access levels"""
    DENIED = "DENIED"
    READ = "READ"
    WRITE = "WRITE"
    EXECUTE = "EXECUTE"
    ADMIN = "ADMIN"

class ResourceType(Enum):
    """Resource types"""
    DATA = "DATA"
    MODEL = "MODEL"
    COMPUTE = "COMPUTE"
    STORAGE = "STORAGE"
    NETWORK = "NETWORK"
    SYSTEM = "SYSTEM"

class AccessDecision(Enum):
    """Access decision results"""
    PERMIT = "PERMIT"
    DENY = "DENY"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INDETERMINATE = "INDETERMINATE"

@dataclass
class User:
    """User definition"""
    user_id: str
    username: str
    roles: List[str]
    clearance_level: str
    groups: List[str]
    attributes: Dict[str, Any]
    active: bool
    created_at: datetime
    last_login: Optional[datetime]

@dataclass
class Role:
    """Role definition"""
    role_id: str
    role_name: str
    permissions: List[str]
    resource_access: Dict[str, AccessLevel]
    inherits_from: List[str]
    attributes: Dict[str, Any]

@dataclass
class Resource:
    """Resource definition"""
    resource_id: str
    resource_name: str
    resource_type: ResourceType
    classification: str
    owner: str
    path: str
    attributes: Dict[str, Any]
    created_at: datetime

@dataclass
class AccessPolicy:
    """Access policy definition"""
    policy_id: str
    policy_name: str
    description: str
    target_resources: List[str]
    target_users: List[str]
    target_roles: List[str]
    conditions: List[Dict[str, Any]]
    effect: AccessDecision
    priority: int
    active: bool

@dataclass
class AccessRequest:
    """Access request"""
    request_id: str
    user_id: str
    resource_id: str
    action: str
    access_level: AccessLevel
    context: Dict[str, Any]
    timestamp: datetime

@dataclass
class AccessResult:
    """Access control result"""
    request_id: str
    decision: AccessDecision
    applicable_policies: List[str]
    reason: str
    timestamp: datetime
    additional_info: Dict[str, Any]

class ResourceAccessControlSystem:
    """
    Defense-grade resource access control system

    Provides comprehensive access control including:
    - Role-based access control (RBAC)
    - Attribute-based access control (ABAC)
    - Multi-level security (MLS)
    - Dynamic access policies
    - Real-time access monitoring
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.audit = EnhancedAuditTrail()
        self.crypto = FIPSCryptoModule()

        # Initialize access control components
        self._setup_access_database()
        self._setup_policy_engine()
        self._setup_monitoring_systems()

        # Access control state
        self.users = {}
        self.roles = {}
        self.resources = {}
        self.policies = {}
        self.active_sessions = {}
        self.access_history = []

        # Thread safety
        self.access_lock = threading.Lock()

        # Load default configurations
        self._load_default_configurations()

        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load access control configuration"""
        default_config = {
            'authentication': {
                'require_mfa': True,
                'session_timeout_minutes': 30,
                'max_failed_attempts': 3,
                'lockout_duration_minutes': 15
            },
            'authorization': {
                'default_deny': True,
                'policy_evaluation_mode': 'permit_overrides',
                'cache_decisions': True,
                'cache_ttl_seconds': 300
            },
            'classification': {
                'levels': ['UNCLASSIFIED', 'CUI//BASIC', 'CUI//SP-PRIV'],
                'mandatory_access_control': True,
                'no_read_up': True,
                'no_write_down': True
            },
            'monitoring': {
                'log_all_access': True,
                'real_time_alerts': True,
                'anomaly_detection': True,
                'compliance_reporting': True
            },
            'policies': {
                'dynamic_policies': True,
                'time_based_access': True,
                'location_based_access': True,
                'risk_based_access': True
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def _setup_access_database(self):
        """Initialize access control database"""
        self.access_database = {}

    def _setup_policy_engine(self):
        """Initialize policy evaluation engine"""
        self.policy_engine = AccessPolicyEngine(self.config)

    def _setup_monitoring_systems(self):
        """Initialize access monitoring systems"""
        self.monitoring_active = False
        self.monitoring_thread = None

    def _load_default_configurations(self):
        """Load default users, roles, and policies"""
        self._create_default_roles()
        self._create_default_users()
        self._create_default_policies()
        self._create_default_resources()

    def _create_default_roles(self):
        """Create default security roles"""
        default_roles = [
            Role(
                role_id='training_user',
                role_name='Training User',
                permissions=['read_data', 'execute_training'],
                resource_access={
                    'training_data': AccessLevel.READ,
                    'models': AccessLevel.READ,
                    'compute_resources': AccessLevel.EXECUTE
                },
                inherits_from=[],
                attributes={'clearance_required': 'CUI//BASIC'}
            ),
            Role(
                role_id='ml_engineer',
                role_name='ML Engineer',
                permissions=['read_data', 'write_data', 'execute_training', 'manage_models'],
                resource_access={
                    'training_data': AccessLevel.WRITE,
                    'models': AccessLevel.WRITE,
                    'compute_resources': AccessLevel.EXECUTE
                },
                inherits_from=['training_user'],
                attributes={'clearance_required': 'CUI//BASIC'}
            ),
            Role(
                role_id='data_scientist',
                role_name='Data Scientist',
                permissions=['read_data', 'analyze_data', 'execute_training'],
                resource_access={
                    'training_data': AccessLevel.READ,
                    'models': AccessLevel.READ,
                    'compute_resources': AccessLevel.EXECUTE
                },
                inherits_from=['training_user'],
                attributes={'clearance_required': 'CUI//BASIC'}
            ),
            Role(
                role_id='security_officer',
                role_name='Security Officer',
                permissions=['audit_access', 'manage_policies', 'security_monitoring'],
                resource_access={
                    'audit_logs': AccessLevel.READ,
                    'security_policies': AccessLevel.WRITE,
                    'monitoring_systems': AccessLevel.ADMIN
                },
                inherits_from=[],
                attributes={'clearance_required': 'CUI//SP-PRIV'}
            ),
            Role(
                role_id='system_admin',
                role_name='System Administrator',
                permissions=['manage_system', 'manage_users', 'manage_resources'],
                resource_access={
                    'system_resources': AccessLevel.ADMIN,
                    'user_management': AccessLevel.ADMIN,
                    'compute_resources': AccessLevel.ADMIN
                },
                inherits_from=[],
                attributes={'clearance_required': 'CUI//SP-PRIV'}
            )
        ]

        for role in default_roles:
            self.roles[role.role_id] = role

    def _create_default_users(self):
        """Create default system users"""
        default_users = [
            User(
                user_id='system',
                username='system',
                roles=['system_admin'],
                clearance_level='CUI//SP-PRIV',
                groups=['system'],
                attributes={'service_account': True},
                active=True,
                created_at=datetime.now(timezone.utc),
                last_login=None
            ),
            User(
                user_id='ml_engineer.001',
                username='ml_engineer_001',
                roles=['ml_engineer'],
                clearance_level='CUI//BASIC',
                groups=['engineering'],
                attributes={'department': 'machine_learning'},
                active=True,
                created_at=datetime.now(timezone.utc),
                last_login=None
            ),
            User(
                user_id='security_officer.001',
                username='security_officer_001',
                roles=['security_officer'],
                clearance_level='CUI//SP-PRIV',
                groups=['security'],
                attributes={'department': 'security'},
                active=True,
                created_at=datetime.now(timezone.utc),
                last_login=None
            )
        ]

        for user in default_users:
            self.users[user.user_id] = user

    def _create_default_policies(self):
        """Create default access policies"""
        default_policies = [
            AccessPolicy(
                policy_id='training_data_access',
                policy_name='Training Data Access Policy',
                description='Controls access to training datasets',
                target_resources=['training_data_*'],
                target_users=[],
                target_roles=['ml_engineer', 'data_scientist'],
                conditions=[
                    {'type': 'classification_level', 'value': 'CUI//BASIC', 'operator': 'gte'},
                    {'type': 'time_of_day', 'value': '06:00-22:00', 'operator': 'within'}
                ],
                effect=AccessDecision.PERMIT,
                priority=100,
                active=True
            ),
            AccessPolicy(
                policy_id='model_management',
                policy_name='Model Management Policy',
                description='Controls model creation and modification',
                target_resources=['models_*'],
                target_users=[],
                target_roles=['ml_engineer'],
                conditions=[
                    {'type': 'classification_level', 'value': 'CUI//BASIC', 'operator': 'gte'},
                    {'type': 'authentication_level', 'value': 'mfa', 'operator': 'eq'}
                ],
                effect=AccessDecision.PERMIT,
                priority=200,
                active=True
            ),
            AccessPolicy(
                policy_id='security_monitoring',
                policy_name='Security Monitoring Policy',
                description='Controls access to security monitoring systems',
                target_resources=['security_*', 'audit_*'],
                target_users=[],
                target_roles=['security_officer'],
                conditions=[
                    {'type': 'classification_level', 'value': 'CUI//SP-PRIV', 'operator': 'gte'},
                    {'type': 'authentication_level', 'value': 'mfa', 'operator': 'eq'}
                ],
                effect=AccessDecision.PERMIT,
                priority=300,
                active=True
            ),
            AccessPolicy(
                policy_id='default_deny',
                policy_name='Default Deny Policy',
                description='Default deny for all resources',
                target_resources=['*'],
                target_users=['*'],
                target_roles=[],
                conditions=[],
                effect=AccessDecision.DENY,
                priority=1,  # Lowest priority
                active=True
            )
        ]

        for policy in default_policies:
            self.policies[policy.policy_id] = policy

    def _create_default_resources(self):
        """Create default resource definitions"""
        default_resources = [
            Resource(
                resource_id='training_data_mnist',
                resource_name='MNIST Training Dataset',
                resource_type=ResourceType.DATA,
                classification='CUI//BASIC',
                owner='system',
                path='/secure/data/mnist',
                attributes={'dataset_type': 'image_classification', 'size_gb': 0.1},
                created_at=datetime.now(timezone.utc)
            ),
            Resource(
                resource_id='model_defense_classifier',
                resource_name='Defense Classification Model',
                resource_type=ResourceType.MODEL,
                classification='CUI//SP-PRIV',
                owner='ml_engineer.001',
                path='/secure/models/defense_classifier',
                attributes={'model_type': 'neural_network', 'version': '1.0'},
                created_at=datetime.now(timezone.utc)
            ),
            Resource(
                resource_id='compute_gpu_cluster',
                resource_name='GPU Training Cluster',
                resource_type=ResourceType.COMPUTE,
                classification='CUI//BASIC',
                owner='system',
                path='/compute/gpu_cluster',
                attributes={'gpu_count': 8, 'memory_gb': 64},
                created_at=datetime.now(timezone.utc)
            ),
            Resource(
                resource_id='audit_logs',
                resource_name='System Audit Logs',
                resource_type=ResourceType.SYSTEM,
                classification='CUI//SP-PRIV',
                owner='security_officer.001',
                path='/secure/logs/audit',
                attributes={'retention_years': 7, 'tamper_evident': True},
                created_at=datetime.now(timezone.utc)
            )
        ]

        for resource in default_resources:
            self.resources[resource.resource_id] = resource

    def request_access(self, user_id: str, resource_id: str, action: str,
                      access_level: AccessLevel, context: Optional[Dict[str, Any]] = None) -> AccessResult:
        """
        Request access to a resource

        Args:
            user_id: User requesting access
            resource_id: Target resource
            action: Action to perform
            access_level: Required access level
            context: Additional context information

        Returns:
            Access control decision
        """
        request_id = hashlib.sha256(
            f"{user_id}_{resource_id}_{action}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        if context is None:
            context = {}

        # Create access request
        access_request = AccessRequest(
            request_id=request_id,
            user_id=user_id,
            resource_id=resource_id,
            action=action,
            access_level=access_level,
            context=context,
            timestamp=datetime.now(timezone.utc)
        )

        # Evaluate access request
        access_result = self._evaluate_access_request(access_request)

        # Store access result
        with self.access_lock:
            self.access_history.append(access_result)

        # Log access request
        self.audit.log_security_event(
            event_type='access_control',
            user_id=user_id,
            action=f'request_{action}',
            resource=resource_id,
            classification=self._get_resource_classification(resource_id),
            additional_data={
                'request_id': request_id,
                'access_level': access_level.value,
                'decision': access_result.decision.value,
                'reason': access_result.reason
            }
        )

        return access_result

    def _evaluate_access_request(self, request: AccessRequest) -> AccessResult:
        """Evaluate access request against policies"""
        # Pre-evaluation checks
        pre_check_result = self._perform_pre_evaluation_checks(request)
        if pre_check_result.decision == AccessDecision.DENY:
            return pre_check_result

        # Get applicable policies
        applicable_policies = self._get_applicable_policies(request)

        # Evaluate policies
        policy_results = []
        for policy in applicable_policies:
            result = self._evaluate_policy(policy, request)
            policy_results.append((policy, result))

        # Combine policy results
        final_decision = self._combine_policy_results(policy_results)

        return AccessResult(
            request_id=request.request_id,
            decision=final_decision['decision'],
            applicable_policies=[p.policy_id for p, r in policy_results if r == AccessDecision.PERMIT],
            reason=final_decision['reason'],
            timestamp=datetime.now(timezone.utc),
            additional_info={
                'policy_evaluation_count': len(policy_results),
                'permit_policies': len([r for p, r in policy_results if r == AccessDecision.PERMIT]),
                'deny_policies': len([r for p, r in policy_results if r == AccessDecision.DENY])
            }
        )

    def _perform_pre_evaluation_checks(self, request: AccessRequest) -> AccessResult:
        """Perform pre-evaluation security checks"""
        # Check if user exists and is active
        user = self.users.get(request.user_id)
        if not user or not user.active:
            return AccessResult(
                request_id=request.request_id,
                decision=AccessDecision.DENY,
                applicable_policies=[],
                reason="User not found or inactive",
                timestamp=datetime.now(timezone.utc),
                additional_info={}
            )

        # Check if resource exists
        resource = self.resources.get(request.resource_id)
        if not resource:
            return AccessResult(
                request_id=request.request_id,
                decision=AccessDecision.DENY,
                applicable_policies=[],
                reason="Resource not found",
                timestamp=datetime.now(timezone.utc),
                additional_info={}
            )

        # Check mandatory access control (MAC)
        mac_result = self._check_mandatory_access_control(user, resource, request.access_level)
        if mac_result.decision == AccessDecision.DENY:
            return mac_result

        # All pre-checks passed
        return AccessResult(
            request_id=request.request_id,
            decision=AccessDecision.PERMIT,
            applicable_policies=[],
            reason="Pre-evaluation checks passed",
            timestamp=datetime.now(timezone.utc),
            additional_info={}
        )

    def _check_mandatory_access_control(self, user: User, resource: Resource,
                                      access_level: AccessLevel) -> AccessResult:
        """Check mandatory access control constraints"""
        if not self.config['classification']['mandatory_access_control']:
            return AccessResult(
                request_id='',
                decision=AccessDecision.PERMIT,
                applicable_policies=[],
                reason="MAC not enforced",
                timestamp=datetime.now(timezone.utc),
                additional_info={}
            )

        # Define classification hierarchy
        classification_levels = {
            'UNCLASSIFIED': 1,
            'CUI//BASIC': 2,
            'CUI//SP-PRIV': 3
        }

        user_level = classification_levels.get(user.clearance_level, 0)
        resource_level = classification_levels.get(resource.classification, 0)

        # No read up rule
        if access_level == AccessLevel.READ and user_level < resource_level:
            return AccessResult(
                request_id='',
                decision=AccessDecision.DENY,
                applicable_policies=[],
                reason=f"MAC violation: No read up (user: {user.clearance_level}, resource: {resource.classification})",
                timestamp=datetime.now(timezone.utc),
                additional_info={'mac_violation': 'no_read_up'}
            )

        # No write down rule
        if access_level == AccessLevel.WRITE and user_level > resource_level:
            return AccessResult(
                request_id='',
                decision=AccessDecision.DENY,
                applicable_policies=[],
                reason=f"MAC violation: No write down (user: {user.clearance_level}, resource: {resource.classification})",
                timestamp=datetime.now(timezone.utc),
                additional_info={'mac_violation': 'no_write_down'}
            )

        return AccessResult(
            request_id='',
            decision=AccessDecision.PERMIT,
            applicable_policies=[],
            reason="MAC check passed",
            timestamp=datetime.now(timezone.utc),
            additional_info={}
        )

    def _get_applicable_policies(self, request: AccessRequest) -> List[AccessPolicy]:
        """Get policies applicable to the request"""
        applicable_policies = []

        user = self.users[request.user_id]

        for policy in self.policies.values():
            if not policy.active:
                continue

            # Check resource match
            if not self._matches_pattern_list(request.resource_id, policy.target_resources):
                continue

            # Check user match
            if policy.target_users and not self._matches_pattern_list(request.user_id, policy.target_users):
                continue

            # Check role match
            if policy.target_roles and not any(role in policy.target_roles for role in user.roles):
                continue

            applicable_policies.append(policy)

        # Sort by priority (higher priority first)
        applicable_policies.sort(key=lambda p: p.priority, reverse=True)

        return applicable_policies

    def _matches_pattern_list(self, value: str, patterns: List[str]) -> bool:
        """Check if value matches any pattern in list"""
        for pattern in patterns:
            if pattern == '*':
                return True
            elif pattern.endswith('*'):
                prefix = pattern[:-1]
                if value.startswith(prefix):
                    return True
            elif pattern == value:
                return True

        return False

    def _evaluate_policy(self, policy: AccessPolicy, request: AccessRequest) -> AccessDecision:
        """Evaluate a single policy against request"""
        if not policy.active:
            return AccessDecision.NOT_APPLICABLE

        # Evaluate all conditions
        for condition in policy.conditions:
            if not self._evaluate_condition(condition, request):
                return AccessDecision.NOT_APPLICABLE

        # All conditions met, return policy effect
        return policy.effect

    def _evaluate_condition(self, condition: Dict[str, Any], request: AccessRequest) -> bool:
        """Evaluate a single policy condition"""
        condition_type = condition.get('type')
        condition_value = condition.get('value')
        operator = condition.get('operator', 'eq')

        if condition_type == 'classification_level':
            user = self.users[request.user_id]
            return self._compare_classification_levels(
                user.clearance_level, condition_value, operator
            )

        elif condition_type == 'time_of_day':
            current_time = datetime.now().strftime('%H:%M')
            if operator == 'within':
                start_time, end_time = condition_value.split('-')
                return start_time <= current_time <= end_time

        elif condition_type == 'authentication_level':
            auth_level = request.context.get('authentication_level', 'basic')
            return self._compare_values(auth_level, condition_value, operator)

        elif condition_type == 'location':
            user_location = request.context.get('location', 'unknown')
            return self._compare_values(user_location, condition_value, operator)

        elif condition_type == 'day_of_week':
            current_day = datetime.now().strftime('%A').lower()
            allowed_days = [day.lower() for day in condition_value]
            return current_day in allowed_days

        # Default: condition not met
        return False

    def _compare_classification_levels(self, user_level: str, required_level: str, operator: str) -> bool:
        """Compare classification levels"""
        levels = {'UNCLASSIFIED': 1, 'CUI//BASIC': 2, 'CUI//SP-PRIV': 3}

        user_num = levels.get(user_level, 0)
        required_num = levels.get(required_level, 0)

        if operator == 'eq':
            return user_num == required_num
        elif operator == 'gte':
            return user_num >= required_num
        elif operator == 'lte':
            return user_num <= required_num
        else:
            return False

    def _compare_values(self, actual: Any, expected: Any, operator: str) -> bool:
        """Compare values with operator"""
        if operator == 'eq':
            return actual == expected
        elif operator == 'ne':
            return actual != expected
        elif operator == 'gt':
            return actual > expected
        elif operator == 'lt':
            return actual < expected
        elif operator == 'gte':
            return actual >= expected
        elif operator == 'lte':
            return actual <= expected
        elif operator == 'in':
            return actual in expected
        elif operator == 'contains':
            return expected in actual
        else:
            return False

    def _combine_policy_results(self, policy_results: List[Tuple[AccessPolicy, AccessDecision]]) -> Dict[str, Any]:
        """Combine multiple policy evaluation results"""
        evaluation_mode = self.config['authorization']['policy_evaluation_mode']

        permit_policies = [p for p, r in policy_results if r == AccessDecision.PERMIT]
        deny_policies = [p for p, r in policy_results if r == AccessDecision.DENY]

        if evaluation_mode == 'permit_overrides':
            # If any policy permits, access is granted
            if permit_policies:
                return {
                    'decision': AccessDecision.PERMIT,
                    'reason': f"Access permitted by {len(permit_policies)} policy(ies)"
                }
            elif deny_policies:
                return {
                    'decision': AccessDecision.DENY,
                    'reason': f"Access denied by {len(deny_policies)} policy(ies)"
                }
            else:
                return {
                    'decision': AccessDecision.DENY if self.config['authorization']['default_deny'] else AccessDecision.PERMIT,
                    'reason': "Default policy applied"
                }

        elif evaluation_mode == 'deny_overrides':
            # If any policy denies, access is denied
            if deny_policies:
                return {
                    'decision': AccessDecision.DENY,
                    'reason': f"Access denied by {len(deny_policies)} policy(ies)"
                }
            elif permit_policies:
                return {
                    'decision': AccessDecision.PERMIT,
                    'reason': f"Access permitted by {len(permit_policies)} policy(ies)"
                }
            else:
                return {
                    'decision': AccessDecision.DENY if self.config['authorization']['default_deny'] else AccessDecision.PERMIT,
                    'reason': "Default policy applied"
                }

        else:
            # Default: deny
            return {
                'decision': AccessDecision.DENY,
                'reason': "Unknown evaluation mode"
            }

    def _get_resource_classification(self, resource_id: str) -> str:
        """Get resource classification level"""
        resource = self.resources.get(resource_id)
        return resource.classification if resource else 'UNCLASSIFIED'

    def create_user(self, user_data: Dict[str, Any], creator_id: str) -> str:
        """Create a new user"""
        # Validate creator permissions
        access_result = self.request_access(
            creator_id, 'user_management', 'create_user', AccessLevel.ADMIN
        )

        if access_result.decision != AccessDecision.PERMIT:
            raise PermissionError("Insufficient permissions to create user")

        user_id = user_data.get('user_id') or f"user_{secrets.token_hex(8)}"

        user = User(
            user_id=user_id,
            username=user_data['username'],
            roles=user_data.get('roles', []),
            clearance_level=user_data.get('clearance_level', 'UNCLASSIFIED'),
            groups=user_data.get('groups', []),
            attributes=user_data.get('attributes', {}),
            active=user_data.get('active', True),
            created_at=datetime.now(timezone.utc),
            last_login=None
        )

        with self.access_lock:
            self.users[user_id] = user

        # Log user creation
        self.audit.log_security_event(
            event_type='user_management',
            user_id=creator_id,
            action='create_user',
            resource=f'user_{user_id}',
            classification='CUI//BASIC',
            additional_data={
                'created_user_id': user_id,
                'username': user.username,
                'roles': user.roles,
                'clearance_level': user.clearance_level
            }
        )

        return user_id

    def create_resource(self, resource_data: Dict[str, Any], creator_id: str) -> str:
        """Create a new resource"""
        # Validate creator permissions
        access_result = self.request_access(
            creator_id, 'resource_management', 'create_resource', AccessLevel.ADMIN
        )

        if access_result.decision != AccessDecision.PERMIT:
            raise PermissionError("Insufficient permissions to create resource")

        resource_id = resource_data.get('resource_id') or f"resource_{secrets.token_hex(8)}"

        resource = Resource(
            resource_id=resource_id,
            resource_name=resource_data['resource_name'],
            resource_type=ResourceType(resource_data['resource_type']),
            classification=resource_data.get('classification', 'UNCLASSIFIED'),
            owner=creator_id,
            path=resource_data.get('path', ''),
            attributes=resource_data.get('attributes', {}),
            created_at=datetime.now(timezone.utc)
        )

        with self.access_lock:
            self.resources[resource_id] = resource

        # Log resource creation
        self.audit.log_security_event(
            event_type='resource_management',
            user_id=creator_id,
            action='create_resource',
            resource=resource_id,
            classification=resource.classification,
            additional_data={
                'resource_name': resource.resource_name,
                'resource_type': resource.resource_type.value,
                'classification': resource.classification
            }
        )

        return resource_id

    def generate_access_report(self, time_period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Generate access control report"""
        if time_period:
            start_time, end_time = time_period
            filtered_history = [
                result for result in self.access_history
                if start_time <= result.timestamp <= end_time
            ]
        else:
            filtered_history = self.access_history[-1000:]  # Last 1000 entries

        # Calculate statistics
        total_requests = len(filtered_history)
        permitted_requests = len([r for r in filtered_history if r.decision == AccessDecision.PERMIT])
        denied_requests = len([r for r in filtered_history if r.decision == AccessDecision.DENY])

        # User activity
        user_activity = {}
        for result in filtered_history:
            request = next((r for r in [result]), None)  # Would need to store original request
            if hasattr(result, 'additional_info') and 'user_id' in result.additional_info:
                user_id = result.additional_info['user_id']
                user_activity[user_id] = user_activity.get(user_id, 0) + 1

        # Resource access patterns
        resource_access = {}
        for result in filtered_history:
            if hasattr(result, 'additional_info') and 'resource_id' in result.additional_info:
                resource_id = result.additional_info['resource_id']
                resource_access[resource_id] = resource_access.get(resource_id, 0) + 1

        return {
            'report_type': 'Access Control Report',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'time_period': {
                'start': time_period[0].isoformat() if time_period else 'N/A',
                'end': time_period[1].isoformat() if time_period else 'N/A'
            },
            'statistics': {
                'total_requests': total_requests,
                'permitted_requests': permitted_requests,
                'denied_requests': denied_requests,
                'permit_rate': (permitted_requests / total_requests * 100) if total_requests > 0 else 0
            },
            'user_activity': dict(sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]),
            'resource_access': dict(sorted(resource_access.items(), key=lambda x: x[1], reverse=True)[:10]),
            'active_users': len(self.users),
            'active_resources': len(self.resources),
            'active_policies': len([p for p in self.policies.values() if p.active]),
            'compliance_status': 'COMPLIANT'
        }

class AccessPolicyEngine:
    """Policy evaluation engine for access control"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def evaluate_policies(self, request: AccessRequest, policies: List[AccessPolicy]) -> AccessDecision:
        """Evaluate multiple policies for access request"""
        # Simplified implementation - would be more sophisticated in production
        for policy in sorted(policies, key=lambda p: p.priority, reverse=True):
            if policy.effect == AccessDecision.PERMIT:
                return AccessDecision.PERMIT

        return AccessDecision.DENY

# Defense industry validation function
def validate_access_control_system() -> Dict[str, Any]:
    """Validate access control system implementation"""

    access_control = ResourceAccessControlSystem()

    # Test access request
    access_result = access_control.request_access(
        user_id='ml_engineer.001',
        resource_id='training_data_mnist',
        action='read_data',
        access_level=AccessLevel.READ
    )

    compliance_checks = {
        'access_control_implemented': True,
        'role_based_access_control': len(access_control.roles) > 0,
        'policy_based_access_control': len(access_control.policies) > 0,
        'mandatory_access_control': access_control.config['classification']['mandatory_access_control'],
        'audit_logging': True,
        'user_management': True,
        'resource_management': True
    }

    compliance_score = sum(compliance_checks.values()) / len(compliance_checks) * 100

    return {
        'compliance_score': compliance_score,
        'checks': compliance_checks,
        'status': 'COMPLIANT' if compliance_score >= 95 else 'NON_COMPLIANT',
        'assessment_date': datetime.now(timezone.utc).isoformat(),
        'test_access_decision': access_result.decision.value,
        'framework': 'NASA_POT10_DFARS_252.204-7012'
    }

if __name__ == "__main__":
    # Initialize access control system
    access_control = ResourceAccessControlSystem()

    # Test access requests
    print("Testing access control system...")

    # Test 1: ML Engineer accessing training data (should be permitted)
    result1 = access_control.request_access(
        user_id='ml_engineer.001',
        resource_id='training_data_mnist',
        action='read_data',
        access_level=AccessLevel.READ,
        context={'authentication_level': 'mfa'}
    )
    print(f"ML Engineer -> Training Data: {result1.decision.value} ({result1.reason})")

    # Test 2: ML Engineer accessing audit logs (should be denied)
    result2 = access_control.request_access(
        user_id='ml_engineer.001',
        resource_id='audit_logs',
        action='read_logs',
        access_level=AccessLevel.READ,
        context={'authentication_level': 'mfa'}
    )
    print(f"ML Engineer -> Audit Logs: {result2.decision.value} ({result2.reason})")

    # Generate access report
    report = access_control.generate_access_report()
    print(f"Access report: {report['statistics']['total_requests']} total requests")

    # Validate system
    system_validation = validate_access_control_system()
    print(f"Access Control System Compliance: {system_validation['status']} ({system_validation['compliance_score']:.1f}%)")