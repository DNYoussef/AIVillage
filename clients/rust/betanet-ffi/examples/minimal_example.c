/**
 * Minimal Betanet FFI Example
 *
 * Demonstrates basic FFI functionality for Day 8-9 deliverable
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/betanet.h"

int main() {
    printf("=== Betanet Minimal FFI Example ===\n\n");

    // Initialize Betanet library
    if (betanet_init() != 0) {
        fprintf(stderr, "Failed to initialize Betanet library\n");
        return 1;
    }

    printf("Library version: %s\n", betanet_version());
    printf("FFI demo support: %s\n", betanet_feature_supported("ffi_demo") == BETANET_SUCCESS ? "Yes" : "No");
    printf("Buffer management: %s\n", betanet_feature_supported("buffer_management") == BETANET_SUCCESS ? "Yes" : "No");
    printf("Version info: %s\n", betanet_feature_supported("version_info") == BETANET_SUCCESS ? "Yes" : "No");
    printf("Unknown feature: %s\n", betanet_feature_supported("unknown") == BETANET_SUCCESS ? "Yes" : "No");
    printf("\n");

    // Test echo function
    printf("Testing echo function...\n");
    BetanetBuffer echo_output;
    BetanetResult result = betanet_echo("Hello from C!", &echo_output);
    if (result == BETANET_SUCCESS) {
        printf("Echo result: ");
        for (size_t i = 0; i < echo_output.len; i++) {
            printf("%c", echo_output.data[i]);
        }
        printf("\n");
        betanet_buffer_free(echo_output);
    } else {
        printf("Echo failed: %s\n", betanet_error_message(result));
    }
    printf("\n");

    // Test buffer allocation
    printf("Testing buffer allocation...\n");
    BetanetBuffer test_buffer = betanet_buffer_alloc(100);
    if (test_buffer.data != NULL && test_buffer.len == 100) {
        printf("Buffer allocated successfully: %zu bytes\n", test_buffer.len);

        // Fill buffer with test data
        for (size_t i = 0; i < test_buffer.len; i++) {
            test_buffer.data[i] = (uint8_t)(i % 256);
        }

        betanet_buffer_free(test_buffer);
        printf("Buffer freed successfully\n");
    } else {
        printf("Buffer allocation failed\n");
    }
    printf("\n");

    // Test packet encoding/decoding
    printf("Testing packet encoding/decoding...\n");
    const char* message = "Test packet payload";
    BetanetBuffer input_buffer;
    input_buffer.data = (uint8_t*)message;
    input_buffer.len = strlen(message);
    input_buffer.capacity = 0; // Read-only

    // Encode packet
    BetanetBuffer encoded_packet;
    result = betanet_packet_encode(input_buffer, &encoded_packet);
    if (result == BETANET_SUCCESS) {
        printf("Packet encoded: %zu bytes\n", encoded_packet.len);

        // Decode packet
        BetanetBuffer decoded_packet;
        result = betanet_packet_decode(encoded_packet, &decoded_packet);
        if (result == BETANET_SUCCESS) {
            printf("Packet decoded: %zu bytes\n", decoded_packet.len);
            printf("Decoded content: ");
            for (size_t i = 0; i < decoded_packet.len; i++) {
                printf("%c", decoded_packet.data[i]);
            }
            printf("\n");

            // Verify content matches
            if (decoded_packet.len == strlen(message) &&
                memcmp(decoded_packet.data, message, decoded_packet.len) == 0) {
                printf("✓ Encode/decode roundtrip successful!\n");
            } else {
                printf("✗ Content mismatch after roundtrip\n");
            }

            betanet_buffer_free(decoded_packet);
        } else {
            printf("Packet decode failed: %s\n", betanet_error_message(result));
        }

        betanet_buffer_free(encoded_packet);
    } else {
        printf("Packet encode failed: %s\n", betanet_error_message(result));
    }

    printf("\n");

    // Test error codes
    printf("Testing error codes...\n");
    printf("Success: %s\n", betanet_error_message(BETANET_SUCCESS));
    printf("Invalid argument: %s\n", betanet_error_message(BETANET_INVALID_ARGUMENT));
    printf("Network error: %s\n", betanet_error_message(BETANET_NETWORK_ERROR));
    printf("Parse error: %s\n", betanet_error_message(BETANET_PARSE_ERROR));
    printf("\n");

    betanet_cleanup();

    printf("=== Minimal FFI Example Complete ===\n");
    return 0;
}
