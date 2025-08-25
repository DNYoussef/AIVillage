#!/usr/bin/env python3
"""
Simple UI Server - Serves the Agent Forge admin interface
No external dependencies, just serves the HTML interface
"""

import http.server
import os
import socketserver
import webbrowser
from pathlib import Path


def start_simple_ui():
    """Start a simple HTTP server for the Agent Forge UI."""

    # Change to the gateway directory to serve files
    gateway_dir = Path(__file__).parent
    os.chdir(gateway_dir)

    PORT = 8080

    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print("=" * 60)
            print("AGENT FORGE - SIMPLE UI SERVER")
            print("=" * 60)
            print(f"Server running at: http://localhost:{PORT}")
            print(f"Admin Interface: http://localhost:{PORT}/admin_interface.html")
            print(f"Serving from: {gateway_dir}")
            print("\nAvailable interfaces:")
            print("- admin_interface.html - Simple HTML interface")
            print("- React UI (if built) - Full TypeScript interface")
            print("\nPress Ctrl+C to stop the server")
            print("=" * 60)

            # Try to open the browser
            try:
                webbrowser.open(f"http://localhost:{PORT}/admin_interface.html")
                print("Opening browser automatically...")
            except:
                print("Could not open browser automatically")

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")


if __name__ == "__main__":
    start_simple_ui()
