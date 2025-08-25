#!/usr/bin/env python3
"""
Start All Agent Forge Services - Enhanced Version

Starts all backend services required for the Agent Forge system:
1. Agent Forge Controller (Enhanced) - Port 8083
2. Model Chat API - Port 8084
3. WebSocket Manager - Port 8085
4. Agent Fleet Manager - Port 8086
"""

import subprocess
import sys
import time
from pathlib import Path


def start_service(script_name, port, description):
    """Start a service script."""
    try:
        print(f"🚀 Starting {description} on port {port}...")

        # Use Python to start the service
        process = subprocess.Popen([sys.executable, script_name], cwd=Path(__file__).parent)

        print(f"✅ {description} started (PID: {process.pid})")
        return process

    except Exception as e:
        print(f"❌ Failed to start {description}: {e}")
        return None


def main():
    """Start all Agent Forge services."""
    print("=" * 60)
    print("🎯 Starting Enhanced Agent Forge Backend Services")
    print("=" * 60)

    services = [
        ("start_agent_forge_api.py", 8083, "Agent Forge Controller (Enhanced)"),
        ("start_model_chat_api.py", 8084, "Model Chat API"),
        ("start_websocket_api.py", 8085, "WebSocket Manager"),
        ("api/agent_management.py", 8086, "Agent Fleet Manager"),
    ]

    processes = []

    for script, port, desc in services:
        process = start_service(script, port, desc)
        if process:
            processes.append((process, desc))
        time.sleep(2)  # Stagger startup

    print("\n" + "=" * 60)
    print("🎉 All Enhanced Agent Forge Services Started!")
    print("=" * 60)
    print("📊 Service Endpoints:")
    print("   • Agent Forge Control:  http://localhost:8083")
    print("   • Model Chat:          http://localhost:8084")
    print("   • WebSocket Updates:   ws://localhost:8085/ws")
    print("   • Agent Fleet:         http://localhost:8086")
    print("\n🌐 Admin Interface:")
    print("   • React UI:            http://localhost:3000 (npm run dev)")
    print("   • Simple HTML:         infrastructure/gateway/admin_interface.html")
    print("\n🔧 Enhanced Features:")
    print("   ✅ Real 25M parameter Cognate model creation")
    print("   ✅ Real-time WebSocket progress updates")
    print("   ✅ Consolidated cognate_pretrain integration")
    print("   ✅ Model testing and chat interface")
    print("   ✅ System resource monitoring")

    try:
        print(f"\n⏳ Services running... Press Ctrl+C to stop all services")

        # Keep main process alive and monitor services
        while True:
            time.sleep(5)

            # Check if any services died
            for process, desc in processes:
                if process.poll() is not None:
                    print(f"⚠️  {desc} stopped unexpectedly (exit code: {process.returncode})")

    except KeyboardInterrupt:
        print(f"\n🛑 Stopping all Agent Forge services...")

        for process, desc in processes:
            if process.poll() is None:  # Still running
                print(f"   Stopping {desc}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        print("✅ All services stopped")


if __name__ == "__main__":
    main()
