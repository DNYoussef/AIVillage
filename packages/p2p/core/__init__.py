# P2P Core module bridge
# Routes imports to actual locations in infrastructure/p2p and core/p2p

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent

# Try infrastructure/p2p first
infra_p2p = project_root / "infrastructure" / "p2p"
if infra_p2p.exists():
    sys.path.insert(0, str(infra_p2p.parent))

# Also try core/p2p
core_p2p = project_root / "core" / "p2p"
if core_p2p.exists():
    sys.path.insert(0, str(core_p2p.parent))
