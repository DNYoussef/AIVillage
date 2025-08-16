/**
 * Betanet C Echo Server Example
 *
 * Demonstrates usage of the Betanet C FFI library for creating
 * a simple echo server using HTX transport.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include "../include/betanet.h"

// Global server handle for cleanup
static BetanetHtxServer* g_server = NULL;
static volatile int g_running = 1;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down...\n", sig);
    g_running = 0;
}

// Connection callback
void on_server_state_changed(void* user_data, BetanetConnectionState state) {
    const char* state_str;
    switch (state) {
        case BETANET_CONNECTION_STATE_DISCONNECTED:
            state_str = "Stopped";
            break;
        case BETANET_CONNECTION_STATE_CONNECTING:
            state_str = "Starting";
            break;
        case BETANET_CONNECTION_STATE_CONNECTED:
            state_str = "Running";
            break;
        case BETANET_CONNECTION_STATE_DISCONNECTING:
            state_str = "Stopping";
            break;
        case BETANET_CONNECTION_STATE_ERROR:
            state_str = "Error";
            break;
        default:
            state_str = "Unknown";
    }

    printf("[SERVER] State: %s\n", state_str);
}

// Error callback
void on_error(void* user_data, BetanetResult error_code, const char* error_msg) {
    printf("[SERVER] Error %d: %s\n", error_code, error_msg ? error_msg : "Unknown error");
}

// Connection handler structure
typedef struct {
    uint32_t connection_id;
    BetanetHtxServer* server;
} ConnectionContext;

int main(int argc, char* argv[]) {
    const char* listen_addr = "0.0.0.0:9000";
    if (argc > 1) {
        listen_addr = argv[1];
    }

    printf("Betanet Echo Server Example\n");
    printf("Listening on: %s\n", listen_addr);
    printf("Press Ctrl+C to exit\n\n");

    // Setup signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize Betanet library
    BetanetResult result = betanet_init();
    if (result != BETANET_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to initialize Betanet library\n");
        return 1;
    }

    // Create configuration
    BetanetConfig config = {
        .listen_addr = listen_addr,
        .server_name = NULL,
        .transport = BETANET_TRANSPORT_TCP,
        .max_connections = 100,
        .connection_timeout_secs = 60,
        .keepalive_interval_secs = 15,
        .enable_compression = 0
    };

    // Create HTX server
    g_server = betanet_htx_server_create(&config);
    if (!g_server) {
        const char* error = betanet_get_last_error();
        fprintf(stderr, "Failed to create server: %s\n", error ? error : "Unknown error");
        return 1;
    }

    // Start server
    result = betanet_htx_server_start_async(
        g_server,
        on_server_state_changed,
        NULL
    );

    if (result != BETANET_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to start server: %d\n", result);
        betanet_htx_server_destroy(g_server);
        return 1;
    }

    printf("[SERVER] Server started successfully\n");

    // Main loop - accept connections and echo data
    ConnectionContext* connections[100] = {0};
    int connection_count = 0;

    while (g_running) {
        // Accept new connections
        uint32_t new_conn_id;
        result = betanet_htx_server_accept(g_server, &new_conn_id);

        if (result == BETANET_RESULT_SUCCESS) {
            printf("[SERVER] New connection accepted: ID=%u\n", new_conn_id);

            // Store connection context
            if (connection_count < 100) {
                ConnectionContext* ctx = malloc(sizeof(ConnectionContext));
                ctx->connection_id = new_conn_id;
                ctx->server = g_server;
                connections[connection_count++] = ctx;
            }
        }

        // Echo server logic would go here
        // In a real implementation, we would:
        // 1. Receive data from connections
        // 2. Echo it back to the sender
        // 3. Handle disconnections

        // For demo purposes, just sleep
        sleep(1);

        // Print server status periodically
        static int status_counter = 0;
        if (++status_counter % 10 == 0) {
            printf("[SERVER] Status: Running, %d connections\n", connection_count);
        }
    }

    // Cleanup connections
    printf("\n[SERVER] Shutting down...\n");
    for (int i = 0; i < connection_count; i++) {
        if (connections[i]) {
            free(connections[i]);
        }
    }

    // Destroy server
    betanet_htx_server_destroy(g_server);
    printf("[SERVER] Cleanup complete\n");

    return 0;
}
