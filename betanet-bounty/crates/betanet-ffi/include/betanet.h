/* Betanet FFI - C bindings for Betanet protocol components */
#ifndef BETANET_FFI_H
#define BETANET_FFI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Result codes for Betanet FFI functions
 */
typedef enum {
    /** Operation completed successfully */
    BETANET_SUCCESS = 0,
    /** Invalid argument provided */
    BETANET_INVALID_ARGUMENT = -1,
    /** Out of memory */
    BETANET_OUT_OF_MEMORY = -2,
    /** Network error */
    BETANET_NETWORK_ERROR = -3,
    /** Parsing error */
    BETANET_PARSE_ERROR = -4,
    /** Cryptographic error */
    BETANET_CRYPTO_ERROR = -5,
    /** Internal error */
    BETANET_INTERNAL_ERROR = -6,
    /** Feature not supported */
    BETANET_NOT_SUPPORTED = -7,
    /** Operation timed out */
    BETANET_TIMEOUT = -8,
} BetanetResult;

/**
 * Buffer structure for passing data between C and Rust
 */
typedef struct {
    /** Pointer to data */
    uint8_t *data;
    /** Length of data in bytes */
    uint32_t len;
    /** Capacity of buffer (for owned buffers) */
    uint32_t capacity;
} BetanetBuffer;

/**
 * Initialize the Betanet library
 *
 * Must be called before using any other functions.
 * Returns 0 on success, negative error code on failure.
 */
int betanet_init(void);

/**
 * Cleanup and shutdown the Betanet library
 *
 * Should be called when finished using the library.
 */
void betanet_cleanup(void);

/**
 * Get library version string
 *
 * Returns a null-terminated string with the library version.
 * The caller must not free the returned pointer.
 */
const char* betanet_version(void);

/**
 * Check if a feature is supported
 *
 * @param feature - Feature name to check (null-terminated string)
 *
 * Returns:
 * - 1 if feature is supported
 * - 0 if feature is not supported
 * - -1 if feature name is invalid
 */
int betanet_feature_supported(const char* feature);

/**
 * Free a buffer allocated by Betanet
 *
 * @param buffer - Buffer to free
 *
 * SAFETY: Buffer must have been allocated by Betanet library
 */
void betanet_buffer_free(BetanetBuffer buffer);

/**
 * Allocate a new buffer
 *
 * @param size - Size in bytes to allocate
 *
 * Returns new buffer on success, empty buffer on failure
 */
BetanetBuffer betanet_buffer_alloc(uint32_t size);

/**
 * Get error message for result code
 *
 * @param result - Result code
 *
 * Returns null-terminated error message string.
 * The caller must not free the returned pointer.
 */
const char* betanet_error_message(BetanetResult result);

/**
 * Simple packet encoder demo
 *
 * @param input - Input data
 * @param output - Output buffer (will be allocated)
 *
 * Returns result code
 */
BetanetResult betanet_packet_encode(BetanetBuffer input, BetanetBuffer* output);

/**
 * Simple packet decoder demo
 *
 * @param input - Input encoded data
 * @param output - Output buffer (will be allocated)
 *
 * Returns result code
 */
BetanetResult betanet_packet_decode(BetanetBuffer input, BetanetBuffer* output);

/**
 * Echo function for testing
 *
 * @param input - Input string (null-terminated)
 * @param output - Output buffer (will be allocated)
 *
 * Returns result code
 */
BetanetResult betanet_echo(const char* input, BetanetBuffer* output);

#ifdef __cplusplus
}
#endif

#endif /* BETANET_FFI_H */
/* End of Betanet FFI header */
