#include <stdio.h>
#include <string.h>
#include "betanet.h"

int main() {
    BNContext* ctx = bn_init(NULL);
    if (!ctx) return 1;
    BNSession* sess = bn_dial(ctx, "127.0.0.1:5555");
    if (!sess) return 1;
    const char* msg = "hello";
    bn_stream_write(sess, (const uint8_t*)msg, strlen(msg));
    uint8_t buf[1024];
    int n = bn_stream_read(sess, buf, sizeof(buf));
    if (n > 0) {
        buf[n] = '\0';
        printf("echo:%s\n", buf);
    }
    bn_close(sess);
    bn_free(ctx);
    return 0;
}
