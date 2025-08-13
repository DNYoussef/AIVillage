#ifndef BETANET_H
#define BETANET_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BNContext BNContext;
typedef struct BNSession BNSession;

BNContext* bn_init(const char* cfg_path);
BNSession* bn_dial(BNContext* ctx, const char* origin);
int32_t bn_stream_write(BNSession* sess, const uint8_t* data, uintptr_t len);
int32_t bn_stream_read(BNSession* sess, uint8_t* buf, uintptr_t len);
void bn_close(BNSession* sess);
void bn_free(BNContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* BETANET_H */
