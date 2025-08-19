/* Betanet FFI - C bindings for Betanet protocol components */
#ifndef BETANET_FFI_H
#define BETANET_FFI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Result codes for Betanet FFI functions */
typedef enum {
    BETANET_SUCCESS = 0,
    BETANET_INVALID_ARGUMENT = -1,
    BETANET_OUT_OF_MEMORY = -2,
    BETANET_NETWORK_ERROR = -3,
    BETANET_PARSE_ERROR = -4,
    BETANET_CRYPTO_ERROR = -5,
    BETANET_INTERNAL_ERROR = -6,
    BETANET_NOT_SUPPORTED = -7,
    BETANET_TIMEOUT = -8,
} BetanetResult;

/* Backwards compatible aliases used in examples */
typedef BetanetResult betanet_Result;
#define betanet_Success BETANET_SUCCESS
#define betanet_InvalidArgument BETANET_INVALID_ARGUMENT
#define betanet_OutOfMemory BETANET_OUT_OF_MEMORY
#define betanet_NetworkError BETANET_NETWORK_ERROR
#define betanet_ParseError BETANET_PARSE_ERROR
#define betanet_CryptoError BETANET_CRYPTO_ERROR
#define betanet_InternalError BETANET_INTERNAL_ERROR
#define betanet_NotSupported BETANET_NOT_SUPPORTED
#define betanet_Timeout BETANET_TIMEOUT

/* Buffer structure for passing data between C and Rust */
typedef struct {
    uint8_t *data;
    size_t len;
    size_t capacity;
} BetanetBuffer;

typedef BetanetBuffer betanet_Buffer;

/* Opaque handle type used for protocol objects */
typedef struct BetanetHandle BetanetHandle;

typedef BetanetHandle HTXFrame;
typedef BetanetHandle HTXClient;
typedef BetanetHandle HTXServer;

typedef BetanetHandle MixnodeHandle;
typedef BetanetHandle SphinxPacketHandle;

typedef BetanetHandle UTLSTemplate;
typedef BetanetHandle JA3Generator;
typedef BetanetHandle JA4Generator;

typedef BetanetHandle LinterHandle;

/* Mixnode configuration */
typedef struct {
    uint32_t max_pps;
    uint32_t delay_pool_size;
    double cover_traffic_rate;
    int32_t enable_vrf_delay;
} MixnodeConfigFFI;

typedef MixnodeConfigFFI betanet_MixnodeConfigFFI;

/* Linter configuration */
typedef struct {
    const char* target_dir;
    uint32_t min_severity;
    int32_t all_checks;
} LinterConfigFFI;

typedef LinterConfigFFI betanet_LinterConfigFFI;

/* Linter results */
typedef struct {
    uint32_t files_checked;
    uint32_t rules_executed;
    uint32_t critical_issues;
    uint32_t error_issues;
    uint32_t warning_issues;
    uint32_t info_issues;
} LinterResultsFFI;

typedef LinterResultsFFI betanet_LinterResultsFFI;

/*
 * Base library functions
 */
int betanet_init(void);
void betanet_cleanup(void);
const char* betanet_version(void);
BetanetResult betanet_feature_supported(const char* feature);
void betanet_buffer_free(BetanetBuffer buffer);
BetanetBuffer betanet_buffer_alloc(size_t size);
const char* betanet_error_message(BetanetResult result);
BetanetResult betanet_packet_encode(BetanetBuffer input, BetanetBuffer* output);
BetanetResult betanet_packet_decode(BetanetBuffer input, BetanetBuffer* output);
BetanetResult betanet_echo(const char* input, BetanetBuffer* output);

/*
 * HTX protocol bindings
 */
HTXFrame* htx_frame_create(uint32_t stream_id, uint32_t frame_type, BetanetBuffer payload);
BetanetResult htx_frame_encode(const HTXFrame* frame, BetanetBuffer* buffer);
HTXFrame* htx_frame_decode(BetanetBuffer data);
uint32_t htx_frame_stream_id(const HTXFrame* frame);
uint32_t htx_frame_type(const HTXFrame* frame);
BetanetResult htx_frame_payload(const HTXFrame* frame, BetanetBuffer* buffer);
void htx_frame_free(HTXFrame* frame);

HTXClient* htx_client_create(void);
BetanetResult htx_client_connect(HTXClient* client, const char* address, uint32_t port);
BetanetResult htx_client_send_frame(HTXClient* client, const HTXFrame* frame);
void htx_client_free(HTXClient* client);

/*
 * Mixnode and Sphinx bindings
 */
MixnodeHandle* mixnode_create(const MixnodeConfigFFI* config);
BetanetResult mixnode_start(MixnodeHandle* mixnode);
BetanetResult mixnode_stop(MixnodeHandle* mixnode);
BetanetResult mixnode_process_packet(MixnodeHandle* mixnode, BetanetBuffer packet_data, BetanetBuffer* output_packet);
BetanetResult mixnode_get_stats(const MixnodeHandle* mixnode,
                                uint32_t* packets_processed,
                                uint32_t* packets_forwarded,
                                uint32_t* packets_dropped,
                                double* current_pps);
void mixnode_free(MixnodeHandle* mixnode);

SphinxPacketHandle* sphinx_packet_create(BetanetBuffer payload, const char* route);
BetanetResult sphinx_packet_encode(const SphinxPacketHandle* packet, BetanetBuffer* buffer);
SphinxPacketHandle* sphinx_packet_decode(BetanetBuffer data);
void sphinx_packet_free(SphinxPacketHandle* packet);

/*
 * uTLS and fingerprinting bindings
 */
JA3Generator* utls_ja3_generator_create(void);
BetanetResult utls_ja3_generate(JA3Generator* generator, BetanetBuffer client_hello, BetanetBuffer* fingerprint);
void utls_ja3_generator_free(JA3Generator* generator);

JA4Generator* utls_ja4_generator_create(void);
BetanetResult utls_ja4_generate(JA4Generator* generator, BetanetBuffer client_hello, int32_t is_quic, BetanetBuffer* fingerprint);
void utls_ja4_generator_free(JA4Generator* generator);

UTLSTemplate* utls_template_create(const char* browser_type);
BetanetResult utls_template_generate_client_hello(UTLSTemplate* template_handle, const char* server_name, BetanetBuffer* client_hello);
void utls_template_free(UTLSTemplate* template_handle);
int utls_self_test(void);

/*
 * Linter bindings
 */
LinterHandle* linter_create(const LinterConfigFFI* config);
BetanetResult linter_run(LinterHandle* linter, LinterResultsFFI* results);
BetanetResult linter_check_rule(LinterHandle* linter, const char* rule_name, LinterResultsFFI* results);
BetanetResult linter_generate_sbom(const char* directory, const char* format, BetanetBuffer* output);
void linter_free(LinterHandle* linter);

#ifdef __cplusplus
}
#endif

#endif /* BETANET_FFI_H */
/* End of Betanet FFI header */
