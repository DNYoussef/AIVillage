/**
 * Betanet C Echo Client Example
 *
 * Demonstrates usage of the Betanet C FFI library for creating
 * a simple echo client supporting both TCP-443 and QUIC-443 transports.
 *
 * Usage:
 *   ./c_echo_client tcp    # Connect via TCP to port 443
 *   ./c_echo_client quic   # Connect via QUIC to port 443
 *   ./c_echo_client        # Default: TCP
 *
 * Memory Management:
 *   - All client resources are properly cleaned up via betanet_htx_client_destroy()
 *   - String parameters are copied internally by the library
 *   - User data pointers remain owned by this program
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include "../include/betanet.h"

// Global client handle for cleanup
static struct BetanetHtxClient* g_client = NULL;
static volatile int g_running = 1;
static int g_connected = 0;

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down...\n", sig);
    g_running = 0;
}

// Connection callback - called when connection state changes
void on_connection_state_changed(void* user_data, enum BetanetConnectionState state) {
    const char* state_str;
    switch (state) {
        case Disconnected:
            state_str = "Disconnected";
            g_connected = 0;
            break;
        case Connecting:
            state_str = "Connecting";
            break;
        case Connected:
            state_str = "Connected";
            g_connected = 1;
            break;
        case Disconnecting:
            state_str = "Disconnecting";
            break;
        case Error:
            state_str = "Error";
            g_connected = 0;
            break;
        default:
            state_str = "Unknown";
    }

    printf("[CLIENT] Connection state: %s\n", state_str);
}

// Error callback - called when async operations fail
void on_error(void* user_data, enum BetanetResult error_code, const char* error_msg) {
    printf("[CLIENT] Error %d: %s\n", error_code, error_msg ? error_msg : "Unknown error");
}

void print_usage(const char* program_name) {
    printf("Usage: %s [transport] [server]\n", program_name);
    printf("  transport: tcp (default) or quic\n");
    printf("  server: server address (default: 127.0.0.1:443)\n");
    printf("\nExamples:\n");
    printf("  %s tcp                    # TCP on port 443\n", program_name);
    printf("  %s quic                   # QUIC on port 443\n", program_name);
    printf("  %s tcp 192.168.1.100:443  # TCP to specific server\n", program_name);
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    const char* transport_str = "tcp";
    const char* server_addr = "127.0.0.1:443";
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
        server_addr = argv[2];
    }

    printf("=== Betanet Echo Client Example ===\n");
    printf("Transport: %s\n", transport_str);
    printf("Server: %s\n", server_addr);
    printf("Library version: %s\n", betanet_get_version());
    printf("Press Ctrl+C to exit\n\n");

    // Setup signal handler for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize Betanet library
    enum BetanetResult result = betanet_init();
    if (result != Success) {
        fprintf(stderr, "Failed to initialize Betanet library\n");
        return 1;
    }
    printf("[CLIENT] Library initialized\n");

    // Create configuration for the specified transport
    struct BetanetConfig config = {
        .listen_addr = NULL,  // Client doesn't need to listen
        .server_name = server_addr,
        .transport = transport,
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
    printf("[CLIENT] HTX client created\n");

    // Connect to server asynchronously
    result = betanet_htx_client_connect_async(
        g_client,
        server_addr,
        on_connection_state_changed,
        NULL  // No user data needed
    );

    if (result != Success) {
        fprintf(stderr, "Failed to initiate connection: %d\n", result);
        const char* error = betanet_get_last_error();
        if (error) {
            fprintf(stderr, "Error details: %s\n", error);
        }
        betanet_htx_client_destroy(g_client);
        return 1;
    }

    // Wait for connection to establish
    printf("[CLIENT] Connecting to %s via %s...\n", server_addr, transport_str);
    int connection_wait = 0;
    while (!g_connected && g_running && connection_wait < 100) {
        usleep(100000); // 100ms
        connection_wait++;
    }

    if (!g_connected) {
        printf("[CLIENT] Connection timeout or failed\n");
        betanet_htx_client_destroy(g_client);
        return 1;
    }

    // Main communication loop
    char input_buffer[1024];
    unsigned char recv_buffer[4096];
    unsigned int received_len;
    int message_count = 0;

    printf("[CLIENT] Starting echo communication...\n");

    while (g_running && g_connected) {
        // Send a message every 3 seconds
        message_count++;
        snprintf(input_buffer, sizeof(input_buffer),
                 "Echo message #%d via %s transport", message_count, transport_str);

        printf("[CLIENT] Sending: %s\n", input_buffer);

        result = betanet_htx_client_send_async(
            g_client,
            (const unsigned char*)input_buffer,
            (unsigned int)strlen(input_buffer),
            on_error,
            NULL  // No user data needed
        );

        if (result != Success) {
            printf("[CLIENT] Send failed: %d\n", result);
            const char* error = betanet_get_last_error();
            if (error) {
                printf("[CLIENT] Send error: %s\n", error);
            }
        }

        // Check for received data (non-blocking)
        result = betanet_htx_client_recv(
            g_client,
            recv_buffer,
            sizeof(recv_buffer),
            &received_len
        );

        if (result == Success && received_len > 0) {
            printf("[CLIENT] Received %u bytes: ", received_len);
            fwrite(recv_buffer, 1, received_len, stdout);
            printf("\n");
        } else if (result == BufferTooSmall) {
            printf("[CLIENT] Warning: Received data larger than buffer\n");
        }

        // Sleep before next message
        sleep(3);

        // Send a limited number of messages for demo
        if (message_count >= 5) {
            printf("[CLIENT] Demo complete, exiting...\n");
            break;
        }
    }

    // Cleanup resources
    printf("\n[CLIENT] Shutting down...\n");
    if (g_client) {
        betanet_htx_client_destroy(g_client);
        g_client = NULL;
        printf("[CLIENT] Client destroyed\n");
    }

    // Clear any remaining errors
    betanet_clear_error();
    printf("[CLIENT] Cleanup complete\n");

    return 0;
}