# Bridge for nat_traversal module
try:
    from infrastructure.p2p.nat_traversal import *
except ImportError:
    try:
        from core.p2p.nat_traversal import *
    except ImportError:
        # Define minimal stubs
        class NATInfo:
            pass

        class NATTraversal:
            pass

        class NATType:
            pass
