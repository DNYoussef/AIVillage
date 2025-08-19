"""
AIVillage RBAC and Multi-Tenant Security System.

This module provides comprehensive role-based access control (RBAC) and multi-tenant
isolation for the AIVillage platform, ensuring secure access to all system components.
"""

from .aivillage_rbac_integration import (
    AgentForgeIntegration,
    AgentSystemIntegration,
    AIVillageRBACIntegration,
    DigitalTwinIntegration,
    MobileEdgeIntegration,
    P2PNetworkIntegration,
    RAGSystemIntegration,
    initialize_aivillage_rbac,
)
from .multi_tenant_manager import MultiTenantManager, ResourceType, TenantResource, initialize_multi_tenant_manager
from .rbac_api_server import RBACAPIServer, create_rbac_server
from .rbac_system import (
    Permission,
    RBACMiddleware,
    RBACSystem,
    Role,
    Session,
    TenantConfig,
    User,
    initialize_rbac_system,
)

__all__ = [
    # Core RBAC
    "RBACSystem",
    "Role",
    "Permission",
    "TenantConfig",
    "User",
    "Session",
    "RBACMiddleware",
    "initialize_rbac_system",
    # Multi-tenant management
    "MultiTenantManager",
    "TenantResource",
    "ResourceType",
    "initialize_multi_tenant_manager",
    # AIVillage integration
    "AIVillageRBACIntegration",
    "AgentSystemIntegration",
    "RAGSystemIntegration",
    "P2PNetworkIntegration",
    "AgentForgeIntegration",
    "DigitalTwinIntegration",
    "MobileEdgeIntegration",
    "initialize_aivillage_rbac",
    # API server
    "RBACAPIServer",
    "create_rbac_server",
]

__version__ = "1.0.0"
__author__ = "AIVillage Security Team"
__description__ = "RBAC and Multi-Tenant Isolation System for AIVillage"
