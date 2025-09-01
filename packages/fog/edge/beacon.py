# Bridge for fog edge beacon module
try:
    from infrastructure.fog.edge.beacon import *
except ImportError:
    # Define minimal stubs
    class CapabilityBeacon:
        pass

    class DeviceType:
        pass

    class PowerProfile:
        pass
