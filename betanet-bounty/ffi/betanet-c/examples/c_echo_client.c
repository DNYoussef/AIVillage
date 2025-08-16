/**
 * Betanet C Echo Client Example
 *
 * Demonstrates usage of the Betanet C FFI library for creating
 * a simple echo client using HTX transport.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include "../include/betanet.h"

// Global client handle for cleanup
static BetanetHtxClient* g_client = NULL;
static volatile int g_running = 1;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down...\n", sig);
    g_running = 0;
}

// Connection callback
void on_connection_state_changed(void* user_data, BetanetConnectionState state) {
    const char* state_str;
    switch (state) {
        case BETANET_CONNECTION_STATE_DISCONNECTED:
            state_str = "Disconnected";
            break;
        case BETANET_CONNECTION_STATE_CONNECTING:
            state_str = "Connecting";
            break;
        case BETANET_CONNECTION_STATE_CONNECTED:
            state_str = "Connected";
            break;
        case BETANET_CONNECTION_STATE_DISCONNECTING:
            state_str = "Disconnecting";
            break;
        case BETANET_CONNECTION_STATE_ERROR:
            state_str = "Error";
            break;
        default:
            state_str = "Unknown";
    }

    printf("[CLIENT] Connection state: %s\n", state_str);
}

// Error callback
void on_error(void* user_data, BetanetResult error_code, const char* error_msg) {
    printf("[CLIENT] Error %d: %s\n", error_code, error_msg ? error_msg : "Unknown error");
}

// Data receive callback
void on_data_received(void* user_data, const uint8_t* data, uint32_t len) {
    printf("[CLIENT] Received %u bytes: ", len);
    fwrite(data, 1, len, stdout);
    printf("\n");
}

int main(int argc, char* argv[]) {
    const char* server_addr = "127.0.0.1:9000";
    if (argc > 1) {
        server_addr = argv[1];
    }

    printf("Betanet Echo Client Example\n");
    printf("Connecting to: %s\n", server_addr);
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
        .listen_addr = NULL,  // Client doesn't need to listen
        .server_name = server_addr,
        .transport = BETANET_TRANSPORT_TCP,
        .max_connections = 1,
        .connection_timeout_secs = 30,
        .keepalive_interval_secs = 10,
        .enable_compression = 0
    };

    // Create HTX client
    g_client = betanet_htx_client_create(&config);
    if (!g_client) {
        const char* error = betanet_get_last_error();
        fprintf(stderr, "Failed to create client: %s\n", error ? error : "Unknown error");
        return 1;
    }

    // Connect to server
    result = betanet_htx_client_connect_async(
        g_client,
        server_addr,
        on_connection_state_changed,
        NULL
    );

    if (result != BETANET_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to initiate connection: %d\n", result);
        betanet_htx_client_destroy(g_client);
        return 1;
    }

    // Wait for connection
    sleep(1);

    // Main loop - send messages and receive responses
    char input_buffer[1024];
    uint8_t recv_buffer[4096];
    uint32_t received_len;
    int message_count = 0;

    while (g_running) {
        // Send a message every 2 seconds
        snprintf(input_buffer, sizeof(input_buffer),
                 "Echo message #%d from client", ++message_count);

        printf("[CLIENT] Sending: %s\n", input_buffer);

        result = betanet_htx_client_send_async(
            g_client,
            (const uint8_t*)input_buffer,
            strlen(input_buffer),
            on_error,
            NULL
        );

        if (result != BETANET_RESULT_SUCCESS) {
            printf("[CLIENT] Send failed: %d\n", result);
        }

        // Check for received data (non-blocking)
        result = betanet_htx_client_recv(
            g_client,
            recv_buffer,
            sizeof(recv_buffer),
            &received_len
        );

        if (result == BETANET_RESULT_SUCCESS && received_len > 0) {
            printf("[CLIENT] Received %u bytes: ", received_len);
            fwrite(recv_buffer, 1, received_len, stdout);
            printf("\n");
        }

        sleep(2);
    }

    // Cleanup
    printf("\n[CLIENT] Shutting down...\n");
    betanet_htx_client_destroy(g_client);
    printf("[CLIENT] Cleanup complete\n");

    return 0;
}
