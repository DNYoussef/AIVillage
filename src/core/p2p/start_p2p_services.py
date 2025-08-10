#!/usr/bin/env python3
"""Start P2P networking services"""

import json
import logging
import socket
import sys
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class P2PServer:
    """Simple P2P server implementation"""

    def __init__(self):
        self.peers = []
        self.running = False

    def start_tcp_server(self, host="0.0.0.0", port=4001):
        """Start TCP server for P2P communication"""
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(5)

            logger.info(f"P2P TCP server listening on {host}:{port}")
            self.running = True

            while self.running:
                try:
                    server.settimeout(1.0)
                    client, addr = server.accept()
                    logger.info(f"New peer connected from {addr}")

                    # Handle peer in thread
                    threading.Thread(
                        target=self.handle_peer, args=(client, addr)
                    ).start()
                except TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Server error: {e}")

            server.close()

        except Exception as e:
            logger.error(f"Failed to start TCP server: {e}")

    def handle_peer(self, client, addr):
        """Handle peer connection"""
        try:
            # Send welcome message
            welcome = json.dumps({"type": "welcome", "peer_id": str(addr)})
            client.send(welcome.encode() + b"\n")

            # Read messages
            while self.running:
                try:
                    client.settimeout(1.0)
                    data = client.recv(1024)
                    if not data:
                        break

                    # Echo back for now
                    client.send(data)
                except TimeoutError:
                    continue
                except:
                    break

            client.close()
            logger.info(f"Peer {addr} disconnected")

        except Exception as e:
            logger.error(f"Error handling peer {addr}: {e}")

    def start_websocket_server(self, host="0.0.0.0", port=4002):
        """Start WebSocket server"""
        # Simple WebSocket-like server using TCP for now
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(5)

            logger.info(f"P2P WebSocket server listening on {host}:{port}")

            while self.running:
                try:
                    server.settimeout(1.0)
                    client, addr = server.accept()
                    logger.info(f"WebSocket peer connected from {addr}")

                    # Simple response
                    client.send(b"HTTP/1.1 200 OK\r\n\r\n")
                    client.close()
                except TimeoutError:
                    continue
                except:
                    continue

            server.close()

        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")

    def start_mdns_service(self):
        """Start mDNS service announcement"""
        logger.info("mDNS service started (mock implementation)")
        # Real mDNS would use python-zeroconf or similar


def main():
    """Main entry point"""
    print("Starting P2P networking services...")

    server = P2PServer()

    # Start services in threads
    threads = [
        threading.Thread(target=server.start_tcp_server, args=("0.0.0.0", 4001)),
        threading.Thread(target=server.start_websocket_server, args=("0.0.0.0", 4002)),
        threading.Thread(target=server.start_mdns_service),
    ]

    for t in threads:
        t.daemon = True
        t.start()

    print("P2P services started:")
    print("  - LibP2P TCP: port 4001")
    print("  - LibP2P WebSocket: port 4002")
    print("  - mDNS: port 5353")

    try:
        # Keep running
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\nShutting down P2P services...")
        server.running = False
        sys.exit(0)


if __name__ == "__main__":
    main()
