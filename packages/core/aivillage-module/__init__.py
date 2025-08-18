"""Namespace package redirecting to src modules.
This allows imports using the `AIVillage` prefix without requiring an install.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Expose submodules under src through this package
__path__ = [str(_src)]

# Support legacy imports like `AIVillage.src.*`
import types

_legacy = types.ModuleType("AIVillage.src")
_legacy.__path__ = [str(_src)]
sys.modules[__name__ + ".src"] = _legacy
