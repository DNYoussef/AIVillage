#include <stdio.h>
#include "betanet.h"

int main() {
    BNContext* ctx = bn_init(NULL);
    if (!ctx) return 1;
    BNSession* sess = bn_dial(ctx, "listen:127.0.0.1:5555");
    if (!sess) return 1;
    uint8_t buf[1024];
    int n = bn_stream_read(sess, buf, sizeof(buf));
    if (n > 0) {
        bn_stream_write(sess, buf, (uintptr_t)n);
    }
    bn_close(sess);
    bn_free(ctx);
    return 0;
}
