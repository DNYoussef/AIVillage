"""DEPRECATED: Mesh Protocol Implementation Script.

This script has been superseded by implement_mesh_protocol_fixed.py which
contains bug fixes and improvements.

The fixed version provides:
- Corrected routing table initialization
- Better message handling
- Improved error handling
- Enhanced simulation capabilities

Please use the fixed version instead:
  python scripts/implement_mesh_protocol_fixed.py

This script has been archived to deprecated/scripts/implement_mesh_protocol_archived.py
"""

import os
import sys
import warnings

warnings.warn(
    "scripts/implement_mesh_protocol.py is deprecated. "
    "Use scripts/implement_mesh_protocol_fixed.py instead. "
    "This script will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

print("=" * 60)
print("DEPRECATED SCRIPT WARNING")
print("=" * 60)
print("This script (implement_mesh_protocol.py) is deprecated.")
print("")
print("Please use the improved version instead:")
print("  python scripts/implement_mesh_protocol_fixed.py")
print("")
print("The fixed version contains important bug fixes and improvements.")
print("=" * 60)
print("")

# Ask user if they want to continue with deprecated version or switch
try:
    choice = input("Continue with deprecated version? (y/N): ").strip().lower()
    if choice not in ["y", "yes"]:
        print("Switching to fixed version...")
        # Try to run the fixed version
        fixed_script = os.path.join(os.path.dirname(__file__), "implement_mesh_protocol_fixed.py")
        if os.path.exists(fixed_script):
            os.system(f"python {fixed_script}")
        else:
            print("ERROR: Fixed version not found!")
            print("Please run: python scripts/implement_mesh_protocol_fixed.py")
        sys.exit(0)
    else:
        print("WARNING: Running deprecated version with known issues...")
        print("")
except KeyboardInterrupt:
    print("\nOperation cancelled.")
    sys.exit(0)

# If user insists on running deprecated version, load the archived code
archived_script = os.path.join(
    os.path.dirname(__file__),
    "..",
    "deprecated",
    "scripts",
    "implement_mesh_protocol_archived.py",
)
if os.path.exists(archived_script):
    print(f"Loading archived implementation from: {archived_script}")
    exec(open(archived_script).read())
else:
    print("ERROR: Archived script not found!")
    print("Please use: python scripts/implement_mesh_protocol_fixed.py")
    sys.exit(1)
