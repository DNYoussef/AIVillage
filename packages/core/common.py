# Bridge for core.common module
try:
    from core.common import *
except ImportError:
    # Define minimal stubs for common utilities
    pass
