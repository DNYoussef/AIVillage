/**
 * HTX Protocol C Example
 *
 * Demonstrates how to use the Betanet HTX protocol from C
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/betanet.h"

int main() {
    printf("=== Betanet HTX Protocol Example ===\n\n");

    // Initialize Betanet library
    if (betanet_init() != 0) {
        fprintf(stderr, "Failed to initialize Betanet library\n");
        return 1;
    }

    printf("Library version: %s\n", betanet_version());
    printf("HTX support: %s\n", betanet_feature_supported("htx") == BETANET_SUCCESS ? "Yes" : "No");
    printf("\n");

    // Create a DATA frame
    const char* payload_data = "Hello, HTX World!";
    betanet_Buffer payload = betanet_buffer_alloc(strlen(payload_data));
    memcpy(payload.data, payload_data, strlen(payload_data));
    payload.len = strlen(payload_data);

    printf("Creating HTX DATA frame...\n");
    HTXFrame* frame = htx_frame_create(1, 0, payload); // stream_id=1, type=DATA
    if (!frame) {
        fprintf(stderr, "Failed to create HTX frame\n");
        betanet_buffer_free(payload);
        return 1;
    }

    printf("Frame created successfully!\n");
    printf("  Stream ID: %u\n", htx_frame_stream_id(frame));
    printf("  Frame Type: %u\n", htx_frame_type(frame));

    // Encode frame to bytes
    betanet_Buffer encoded_frame;
    betanet_Result result = htx_frame_encode(frame, &encoded_frame);
    if (result != betanet_Success) {
        fprintf(stderr, "Failed to encode frame: %s\n", betanet_error_message(result));
        htx_frame_free(frame);
        betanet_buffer_free(payload);
        return 1;
    }

    printf("Frame encoded: %u bytes\n", encoded_frame.len);

    // Decode frame back
    printf("\nDecoding frame...\n");
    HTXFrame* decoded_frame = htx_frame_decode(encoded_frame);
    if (!decoded_frame) {
        fprintf(stderr, "Failed to decode frame\n");
        htx_frame_free(frame);
        betanet_buffer_free(payload);
        betanet_buffer_free(encoded_frame);
        return 1;
    }

    printf("Frame decoded successfully!\n");
    printf("  Stream ID: %u\n", htx_frame_stream_id(decoded_frame));
    printf("  Frame Type: %u\n", htx_frame_type(decoded_frame));

    // Get payload back
    betanet_Buffer decoded_payload;
    result = htx_frame_payload(decoded_frame, &decoded_payload);
    if (result == betanet_Success) {
        printf("  Payload: ");
        for (unsigned int i = 0; i < decoded_payload.len; i++) {
            printf("%c", decoded_payload.data[i]);
        }
        printf("\n");
    }

    // Create HTX client
    printf("\nCreating HTX client...\n");
    HTXClient* client = htx_client_create();
    if (client) {
        printf("HTX client created successfully!\n");

        // Note: In a real application, you would call htx_client_connect()
        // and htx_client_send_frame() here

        htx_client_free(client);
    } else {
        printf("Failed to create HTX client\n");
    }

    // Cleanup
    htx_frame_free(frame);
    htx_frame_free(decoded_frame);
    betanet_buffer_free(payload);
    betanet_buffer_free(encoded_frame);

    betanet_cleanup();

    printf("\n=== HTX Example Complete ===\n");
    return 0;
}
