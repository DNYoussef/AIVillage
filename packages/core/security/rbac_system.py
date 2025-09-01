# Bridge for RBAC security module
try:
    from core.security.rbac_system import *
except ImportError:
    try:
        from infrastructure.shared.security.rbac_system import *
    except ImportError:
        # Define minimal stubs
        class RBACSystem:
            pass

        class Role:
            pass
