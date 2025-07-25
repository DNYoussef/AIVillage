#!/usr/bin/env python3
"""
Dashboard Launcher for Agent Forge

Launches the Streamlit dashboard with proper configuration
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Agent Forge dashboard"""

    # Set up environment
    dashboard_path = Path(__file__).parent.parent / "monitoring" / "dashboard.py"

    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        return 1

    # Launch Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.serverAddress", "localhost",
        "--browser.serverPort", "8501"
    ]

    print("🚀 Launching Agent Forge Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("⚠️  Press Ctrl+C to stop the dashboard")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard failed to start: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
