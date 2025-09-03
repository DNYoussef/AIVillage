/**
 * Error Handling Example
 *
 * Demonstrates using betanet_error_message when API calls fail.
 */

#include <stdio.h>
#include "../include/betanet.h"

int main() {
    if (betanet_init() != 0) {
        fprintf(stderr, "Failed to initialize Betanet library\n");
        return 1;
    }

    BetanetResult rt = betanet_init_runtime();
    if (rt != betanet_Success) {
        fprintf(stderr, "Runtime init failed: %s\n", betanet_error_message(rt));
        return 1;
    }

    // Intentionally call with a NULL frame to trigger an error
    BetanetBuffer buffer;
    BetanetResult result = htx_frame_encode(NULL, &buffer);
    if (result != betanet_Success) {
        printf("htx_frame_encode failed: %s\n", betanet_error_message(result));
    }

    betanet_cleanup();
    return 0;
}
