# P2P package initialization
# Import routing to actual module locations

import sys
from pathlib import Path

# Add actual p2p locations to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "infrastructure"))
sys.path.insert(0, str(project_root / "core"))

# Re-export from actual locations
try:
    from infrastructure.p2p import *
except ImportError:
    pass

try:
    from core.p2p import *
except ImportError:
    pass
