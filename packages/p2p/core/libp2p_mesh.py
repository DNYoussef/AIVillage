# Bridge for libp2p_mesh module
# First try infrastructure location, then core

try:
    from infrastructure.p2p.core.libp2p_mesh import *
except ImportError:
    try:
        from core.p2p.core.libp2p_mesh import *
    except ImportError:
        # Define minimal stubs if needed
        class LibP2PMeshNetwork:
            pass

        class MeshMessage:
            pass

        class MeshMessageType:
            pass
