#!/usr/bin/env python3
"""Start Agent Forge Controller API on port 8083 - Enhanced Version."""

from pathlib import Path
import sys

# Add API directory to path
api_dir = Path(__file__).parent / "api"
sys.path.insert(0, str(api_dir))

if __name__ == "__main__":
    import uvicorn

    try:
        # Try enhanced controller first
        from api.agent_forge_controller_enhanced import app

        print("🚀 Starting Enhanced Agent Forge Controller API on port 8083...")
        print("✅ Real 25M parameter model creation enabled")
        print("✅ Real-time WebSocket progress updates enabled")
        print("✅ Consolidated cognate_pretrain integration enabled")
        uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info")
    except ImportError as e:
        print(f"Enhanced controller not available: {e}")
        print("Falling back to original controller...")
        try:
            from api.agent_forge_controller import app

            print("🚀 Starting Agent Forge Controller API on port 8083...")
            uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info")
        except ImportError as e2:
            print(f"Original controller import error: {e2}")
            print("Trying direct import...")
            import agent_forge_controller

            print("🚀 Starting Agent Forge Controller API on port 8083...")
            uvicorn.run(agent_forge_controller.app, host="0.0.0.0", port=8083, log_level="info")
