#!/usr/bin/env python3
"""
Startup script for AIVillage Admin Dashboard
Convenient way to start the admin monitoring dashboard
"""

from pathlib import Path
import subprocess
import sys


def main():
    """Start the admin dashboard server"""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    admin_server_path = project_root / "packages" / "core" / "bin" / "admin_server.py"

    if not admin_server_path.exists():
        print(f"Error: Admin server not found at {admin_server_path}")
        sys.exit(1)

    print("Starting AIVillage Admin Dashboard...")
    print("Dashboard will be available at: http://localhost:3006")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        # Start the admin server
        subprocess.run([sys.executable, str(admin_server_path)], check=True)
    except KeyboardInterrupt:
        print("\nShutting down admin dashboard...")
    except subprocess.CalledProcessError as e:
        print(f"Error starting admin dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
