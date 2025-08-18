/**
 * Betanet C Echo Server Example
 *
 * Demonstrates usage of the Betanet C FFI library for creating
 * a simple echo server supporting both TCP-443 and QUIC-443 transports.
 *
 * Usage:
 *   ./c_echo_server tcp    # Listen via TCP on port 443
 *   ./c_echo_server quic   # Listen via QUIC on port 443
 *   ./c_echo_server        # Default: TCP
 *
 * Memory Management:
 *   - All server resources are properly cleaned up via betanet_htx_server_destroy()
 *   - String parameters are copied internally by the library
 *   - Connection handles are managed internally
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include "../include/betanet.h"

#ifdef _WIN32
    #include <windows.h>
    #define sleep(x) Sleep(x * 1000)
    #define usleep(x) Sleep(x / 1000)
#else
    #include <unistd.h>
#endif

// Global server handle for cleanup
static struct BetanetHtxServer* g_server = NULL;
static volatile int g_running = 1;
static int g_server_started = 0;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down server...\n", sig);
    g_running = 0;
}

// Server state callback - called when server state changes
void on_server_state_changed(void* user_data, enum BetanetConnectionState state) {
    const char* state_str;
    switch (state) {
        case Disconnected:
            state_str = "Stopped";
            g_server_started = 0;
            break;
        case Connecting:
            state_str = "Starting";
            break;
        case Connected:
            state_str = "Running";
            g_server_started = 1;
            break;
        case Disconnecting:
            state_str = "Stopping";
            break;
        case ConnectionError:
            state_str = "Error";
            g_server_started = 0;
            break;
        default:
            state_str = "Unknown";
    }

    printf("[SERVER] State: %s\n", state_str);
}

void print_usage(const char* program_name) {
    printf("Usage: %s [transport] [listen_addr]\n", program_name);
    printf("  transport: tcp (default) or quic\n");
    printf("  listen_addr: address to bind (default: 0.0.0.0:443)\n");
    printf("\nExamples:\n");
    printf("  %s tcp                    # TCP on port 443\n", program_name);
    printf("  %s quic                   # QUIC on port 443\n", program_name);
    printf("  %s tcp 0.0.0.0:8443      # TCP on port 8443\n", program_name);
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    const char* transport_str = "tcp";
    const char* listen_addr = "0.0.0.0:443";
    enum BetanetTransport transport = Tcp;

    if (argc > 1) {
        transport_str = argv[1];
        if (strcmp(transport_str, "tcp") == 0) {
            transport = Tcp;
        } else if (strcmp(transport_str, "quic") == 0) {
            transport = Quic;
        } else {
            printf("Error: Unknown transport '%s'\n\n", transport_str);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (argc > 2) {
        listen_addr = argv[2];
    }

    printf("=== Betanet Echo Server Example ===\n");
    printf("Transport: %s\n", transport_str);
    printf("Listen address: %s\n", listen_addr);
    printf("Library version: %s\n", betanet_get_version());
    printf("Press Ctrl+C to stop server\n\n");

    // Setup signal handler for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize Betanet library
    enum BetanetResult result = betanet_init();
    if (result != Success) {
        fprintf(stderr, "Failed to initialize Betanet library\n");
        return 1;
    }
    printf("[SERVER] Library initialized\n");

    // Create configuration for the specified transport
    struct BetanetConfig config = {
        .listen_addr = listen_addr,
        .server_name = NULL,  // Not needed for server
        .transport = transport,
        .max_connections = 100,  // Allow multiple concurrent clients
        .connection_timeout_secs = 60,
        .keepalive_interval_secs = 30,
        .enable_compression = 0
    };

    // Create HTX server
    g_server = betanet_htx_server_create(&config);
    if (!g_server) {
        const char* error = betanet_get_last_error();
        fprintf(stderr, "Failed to create server: %s\n", error ? error : "Unknown error");
        return 1;
    }
    printf("[SERVER] HTX server created\n");

    // Start server asynchronously
    result = betanet_htx_server_start_async(
        g_server,
        on_server_state_changed,
        NULL  // No user data needed
    );

    if (result != Success) {
        fprintf(stderr, "Failed to start server: %d\n", result);
        const char* error = betanet_get_last_error();
        if (error) {
            fprintf(stderr, "Error details: %s\n", error);
        }
        betanet_htx_server_destroy(g_server);
        return 1;
    }

    // Wait for server to start
    printf("[SERVER] Starting %s server on %s...\n", transport_str, listen_addr);
    int startup_wait = 0;
    while (!g_server_started && g_running && startup_wait < 50) {
        usleep(100000); // 100ms
        startup_wait++;
    }

    if (!g_server_started) {
        printf("[SERVER] Server startup timeout or failed\n");
        betanet_htx_server_destroy(g_server);
        return 1;
    }

    printf("[SERVER] Server is running and accepting connections\n");
    printf("[SERVER] Waiting for client connections...\n\n");

    // Main server loop - handle connections and echo messages
    unsigned int connection_id;
    int active_connections = 0;
    int total_connections = 0;

    while (g_running && g_server_started) {
        // Try to accept a new connection (non-blocking in a real implementation)
        // This is a simplified example - in practice you'd use proper async I/O
        result = betanet_htx_server_accept(g_server, &connection_id);

        if (result == Success) {
            active_connections++;
            total_connections++;
            printf("[SERVER] New client connected (ID: %u). Active: %d, Total: %d\n",
                   connection_id, active_connections, total_connections);

            // In a real implementation, you would:
            // 1. Store the connection ID for tracking
            // 2. Set up message handlers for this connection
            // 3. Echo back any received messages
            // 4. Handle connection cleanup when client disconnects

            // For this demo, we'll simulate the echo process
            printf("[SERVER] Connection %u ready for echo service\n", connection_id);

            // Simulate some echo activity
            usleep(500000); // 500ms

            // Simulate connection activity (in real code, this would be event-driven)
            printf("[SERVER] Echoing messages for connection %u...\n", connection_id);

        } else if (result != NotConnected) {
            // NotConnected means no pending connections, which is normal
            const char* error = betanet_get_last_error();
            if (error) {
                printf("[SERVER] Accept error: %s\n", error);
            }
        }

        // Sleep briefly to prevent busy waiting
        usleep(1000000); // 1 second

        // For demo purposes, limit the number of connections processed
        if (total_connections >= 3) {
            printf("[SERVER] Demo complete after serving %d connections\n", total_connections);
            break;
        }
    }

    // Cleanup resources
    printf("\n[SERVER] Shutting down server...\n");
    if (g_server) {
        betanet_htx_server_destroy(g_server);
        g_server = NULL;
        printf("[SERVER] Server destroyed\n");
    }

    // Clear any remaining errors
    betanet_clear_error();
    printf("[SERVER] Cleanup complete\n");
    printf("[SERVER] Total connections served: %d\n", total_connections);

    return 0;
}
