#!/usr/bin/env python3
"""
Start Script for Minimal Agent Forge Backend

Starts all necessary services for Agent Forge:
- Main API server on port 8083 (with WebSocket)
- Gateway server on port 8080
- Optional additional services
"""

import subprocess
import sys
import threading
import time
from pathlib import Path


def start_service(name, script_path, port, wait_time=2):
    """Start a service and return the process."""
    print(f"Starting {name} on port {port}...")

    try:
        # Use Python to run the script
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Wait a bit for startup
        time.sleep(wait_time)

        # Check if process is still running
        if process.poll() is None:
            print(f"[OK] {name} started successfully (PID: {process.pid})")
            return process
        else:
            print(f"[ERROR] {name} failed to start")
            return None

    except Exception as e:
        print(f"[ERROR] Error starting {name}: {e}")
        return None


def monitor_process(name, process):
    """Monitor a process and print its output."""
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(f"[{name}] {line.strip()}")
    except:
        pass


def main():
    print("=" * 80)
    print("🏗️  AGENT FORGE MINIMAL BACKEND STARTUP")
    print("=" * 80)

    # Get current directory
    current_dir = Path(__file__).parent

    services = []

    # Start main backend (API + WebSocket)
    main_backend_path = current_dir / "minimal_agent_forge_backend.py"
    if main_backend_path.exists():
        main_process = start_service("Agent Forge API + WebSocket", str(main_backend_path), 8083, wait_time=3)
        if main_process:
            services.append(("Agent Forge API + WebSocket", main_process))
    else:
        print(f"❌ Main backend script not found: {main_backend_path}")
        return

    # Start simple gateway server (optional)
    gateway_path = current_dir / "simple_server.py"
    if gateway_path.exists():
        gateway_process = start_service("Gateway Server", str(gateway_path), 8080, wait_time=2)
        if gateway_process:
            services.append(("Gateway Server", gateway_process))

    if not services:
        print("❌ No services started successfully")
        return

    print("\n" + "=" * 80)
    print("✅ AGENT FORGE BACKEND READY")
    print("=" * 80)
    print("🔗 Service URLs:")
    print("  📡 API Server:     http://localhost:8083")
    print("  🧪 Test Interface: http://localhost:8083/test")
    print("  🔌 WebSocket:      ws://localhost:8083/ws")
    if len(services) > 1:
        print("  🌐 Gateway:        http://localhost:8080")
    print("\n📋 Quick Test Commands:")
    print("  curl http://localhost:8083/health")
    print("  curl -X POST http://localhost:8083/phases/cognate/start")
    print("  curl http://localhost:8083/phases/status")
    print("  curl http://localhost:8083/models")
    print("\n🔍 Open Test Interface: http://localhost:8083/test")
    print("=" * 80)

    # Start monitoring threads for service output
    monitor_threads = []
    for name, process in services:
        thread = threading.Thread(target=monitor_process, args=(name, process))
        thread.daemon = True
        thread.start()
        monitor_threads.append(thread)

    try:
        print("\n🎯 Services running. Press Ctrl+C to stop all services...\n")

        # Wait for processes
        while True:
            time.sleep(1)

            # Check if any process has died
            running_services = []
            for name, process in services:
                if process.poll() is None:
                    running_services.append((name, process))
                else:
                    print(f"⚠️ {name} process has stopped")

            services = running_services
            if not services:
                print("❌ All services have stopped")
                break

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")

        # Terminate all processes
        for name, process in services:
            print(f"⏹️ Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                print(f"⚠️ Force killing {name}...")
                process.kill()
                process.wait()
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")

        print("\n👋 All services stopped. Goodbye!")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

        # Emergency cleanup
        for name, process in services:
            try:
                process.kill()
            except:
                pass


if __name__ == "__main__":
    main()
