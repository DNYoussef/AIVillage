#!/usr/bin/env python3
"""Start WebSocket Manager on port 8085."""

from pathlib import Path
import sys

# Add API directory to path
api_dir = Path(__file__).parent / "api"
sys.path.insert(0, str(api_dir))

if __name__ == "__main__":
    import uvicorn

    try:
        from api.websocket_manager import app

        print("🚀 Starting WebSocket Manager on port 8085...")
        uvicorn.run(app, host="0.0.0.0", port=8085, log_level="info")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Trying direct import...")
        import websocket_manager

        print("🚀 Starting WebSocket Manager on port 8085...")
        uvicorn.run(websocket_manager.app, host="0.0.0.0", port=8085, log_level="info")
