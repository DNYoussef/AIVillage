#!/usr/bin/env python3
"""Start Model Chat API on port 8084."""

from pathlib import Path
import sys

# Add API directory to path
api_dir = Path(__file__).parent / "api"
sys.path.insert(0, str(api_dir))

if __name__ == "__main__":
    import uvicorn

    try:
        from api.model_chat import app

        print("🚀 Starting Model Chat API on port 8084...")
        uvicorn.run(app, host="0.0.0.0", port=8084, log_level="info")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Trying direct import...")
        import model_chat

        print("🚀 Starting Model Chat API on port 8084...")
        uvicorn.run(model_chat.app, host="0.0.0.0", port=8084, log_level="info")
