/**
 * uTLS Fingerprinting C Example
 *
 * Demonstrates JA3/JA4 fingerprint generation from C
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/betanet.h"

// Sample ClientHello data (simplified)
static const unsigned char sample_client_hello[] = {
    0x16, 0x03, 0x03, 0x00, 0xf4, 0x01, 0x00, 0x00, 0xf0, 0x03, 0x03,
    // ... rest of ClientHello would be here in real application
};

int main() {
    printf("=== Betanet uTLS Fingerprinting Example ===\n\n");

    // Initialize library
    if (betanet_init() != 0) {
        fprintf(stderr, "Failed to initialize Betanet library\n");
        return 1;
    }

    printf("Library version: %s\n", betanet_version());
    printf("uTLS support: %s\n", betanet_feature_supported("utls") == BETANET_SUCCESS ? "Yes" : "No");
    printf("JA3 support: %s\n", betanet_feature_supported("ja3") == BETANET_SUCCESS ? "Yes" : "No");
    printf("JA4 support: %s\n", betanet_feature_supported("ja4") == BETANET_SUCCESS ? "Yes" : "No");
    printf("\n");

    // Run self-test first
    printf("Running uTLS self-test...\n");
    if (utls_self_test()) {
        printf("✓ Self-test passed!\n\n");
    } else {
        printf("✗ Self-test failed!\n\n");
    }

    // Create ClientHello buffer
    betanet_Buffer client_hello;
    client_hello.data = (unsigned char*)sample_client_hello;
    client_hello.len = sizeof(sample_client_hello);
    client_hello.capacity = 0; // Read-only

    // Generate JA3 fingerprint
    printf("Generating JA3 fingerprint...\n");
    JA3Generator* ja3_gen = utls_ja3_generator_create();
    if (!ja3_gen) {
        fprintf(stderr, "Failed to create JA3 generator\n");
        betanet_cleanup();
        return 1;
    }

    betanet_Buffer ja3_fingerprint;
    betanet_Result result = utls_ja3_generate(ja3_gen, client_hello, &ja3_fingerprint);
    if (result == betanet_Success) {
        printf("JA3 fingerprint: ");
        for (size_t i = 0; i < ja3_fingerprint.len; i++) {
            printf("%c", ja3_fingerprint.data[i]);
        }
        printf("\n");
        betanet_buffer_free(ja3_fingerprint);
    } else {
        printf("Failed to generate JA3 fingerprint: %s\n", betanet_error_message(result));
    }

    utls_ja3_generator_free(ja3_gen);

    // Generate JA4 fingerprint
    printf("\nGenerating JA4 fingerprint...\n");
    JA4Generator* ja4_gen = utls_ja4_generator_create();
    if (!ja4_gen) {
        fprintf(stderr, "Failed to create JA4 generator\n");
        betanet_cleanup();
        return 1;
    }

    betanet_Buffer ja4_fingerprint;
    result = utls_ja4_generate(ja4_gen, client_hello, 0, &ja4_fingerprint); // 0 = TCP
    if (result == betanet_Success) {
        printf("JA4 fingerprint (TCP): ");
        for (size_t i = 0; i < ja4_fingerprint.len; i++) {
            printf("%c", ja4_fingerprint.data[i]);
        }
        printf("\n");
        betanet_buffer_free(ja4_fingerprint);
    } else {
        printf("Failed to generate JA4 fingerprint: %s\n", betanet_error_message(result));
    }

    // Generate JA4 for QUIC
    result = utls_ja4_generate(ja4_gen, client_hello, 1, &ja4_fingerprint); // 1 = QUIC
    if (result == betanet_Success) {
        printf("JA4 fingerprint (QUIC): ");
        for (size_t i = 0; i < ja4_fingerprint.len; i++) {
            printf("%c", ja4_fingerprint.data[i]);
        }
        printf("\n");
        betanet_buffer_free(ja4_fingerprint);
    } else {
        printf("Failed to generate JA4 QUIC fingerprint: %s\n", betanet_error_message(result));
    }

    utls_ja4_generator_free(ja4_gen);

    // Create uTLS template
    printf("\nCreating Chrome uTLS template...\n");
    UTLSTemplate* template = utls_template_create("chrome");
    if (template) {
        printf("Chrome template created successfully!\n");

        // Generate ClientHello from template
        betanet_Buffer generated_client_hello;
        result = utls_template_generate_client_hello(template, "example.com", &generated_client_hello);
        if (result == betanet_Success) {
            printf("Generated ClientHello: %zu bytes\n", generated_client_hello.len);
            betanet_buffer_free(generated_client_hello);
        } else {
            printf("Failed to generate ClientHello: %s\n", betanet_error_message(result));
        }

        utls_template_free(template);
    } else {
        printf("Failed to create Chrome template\n");
    }

    betanet_cleanup();

    printf("\n=== uTLS Example Complete ===\n");
    return 0;
}
