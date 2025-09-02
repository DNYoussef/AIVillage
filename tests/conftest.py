# Test Configuration
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "packages"))
sys.path.insert(0, str(project_root / "core"))
sys.path.insert(0, str(project_root / "infrastructure"))

# Set test environment
os.environ.setdefault("AIVILLAGE_ENV", "testing")
os.environ.setdefault("PYTHONPATH", ":".join(sys.path))

# Configure asyncio for testing
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
