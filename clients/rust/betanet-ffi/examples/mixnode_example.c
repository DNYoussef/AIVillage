/**
 * Mixnode C Example
 *
 * Demonstrates mixnode operation and Sphinx packet processing from C
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/betanet.h"

int main() {
    printf("=== Betanet Mixnode Example ===\n\n");

    // Initialize library
    if (betanet_init() != 0) {
        fprintf(stderr, "Failed to initialize Betanet library\n");
        return 1;
    }

    printf("Library version: %s\n", betanet_version());
    printf("Mixnode support: %s\n", betanet_feature_supported("mixnode") == BETANET_SUCCESS ? "Yes" : "No");
    printf("Sphinx support: %s\n", betanet_feature_supported("sphinx") == BETANET_SUCCESS ? "Yes" : "No");
    printf("\n");

    // Configure mixnode
    betanet_MixnodeConfigFFI config = {
        .max_pps = 1000,           // 1000 packets per second
        .delay_pool_size = 100,    // 100 packet delay pool
        .cover_traffic_rate = 5.0, // 5 cover packets per second
        .enable_vrf_delay = 1      // Enable VRF-based delays
    };

    printf("Creating mixnode with configuration:\n");
    printf("  Max PPS: %u\n", config.max_pps);
    printf("  Delay pool size: %u\n", config.delay_pool_size);
    printf("  Cover traffic rate: %.1f\n", config.cover_traffic_rate);
    printf("  VRF delay: %s\n", config.enable_vrf_delay ? "enabled" : "disabled");
    printf("\n");

    // Create mixnode
    MixnodeHandle* mixnode = mixnode_create(&config);
    if (!mixnode) {
        fprintf(stderr, "Failed to create mixnode\n");
        betanet_cleanup();
        return 1;
    }

    printf("Mixnode created successfully!\n");

    // Start mixnode
    betanet_Result result = mixnode_start(mixnode);
    if (result != betanet_Success) {
        fprintf(stderr, "Failed to start mixnode: %s\n", betanet_error_message(result));
        mixnode_free(mixnode);
        betanet_cleanup();
        return 1;
    }

    printf("Mixnode started!\n\n");

    // Create a Sphinx packet
    const char* payload_data = "Secret message through the mix network";
    betanet_Buffer payload = betanet_buffer_alloc(strlen(payload_data));
    memcpy(payload.data, payload_data, strlen(payload_data));
    payload.len = strlen(payload_data);

    printf("Creating Sphinx packet...\n");
    printf("  Payload: %s\n", payload_data);
    printf("  Route: mix1.example.com,mix2.example.com,destination.example.com\n");

    SphinxPacketHandle* packet = sphinx_packet_create(payload, "mix1.example.com,mix2.example.com,destination.example.com");
    if (!packet) {
        fprintf(stderr, "Failed to create Sphinx packet\n");
        betanet_buffer_free(payload);
        mixnode_stop(mixnode);
        mixnode_free(mixnode);
        betanet_cleanup();
        return 1;
    }

    printf("Sphinx packet created successfully!\n");

    // Encode packet
    betanet_Buffer encoded_packet;
    result = sphinx_packet_encode(packet, &encoded_packet);
    if (result != betanet_Success) {
        fprintf(stderr, "Failed to encode Sphinx packet: %s\n", betanet_error_message(result));
        sphinx_packet_free(packet);
        betanet_buffer_free(payload);
        mixnode_stop(mixnode);
        mixnode_free(mixnode);
        betanet_cleanup();
        return 1;
    }

    printf("Packet encoded: %u bytes\n", encoded_packet.len);

    // Process packet through mixnode
    printf("\nProcessing packet through mixnode...\n");
    betanet_Buffer output_packet;
    result = mixnode_process_packet(mixnode, encoded_packet, &output_packet);
    if (result != betanet_Success) {
        fprintf(stderr, "Failed to process packet: %s\n", betanet_error_message(result));
    } else {
        if (output_packet.len > 0) {
            printf("Packet forwarded: %u bytes\n", output_packet.len);
            betanet_buffer_free(output_packet);
        } else {
            printf("Packet consumed (final destination or dropped)\n");
        }
    }

    // Get mixnode statistics
    printf("\nMixnode statistics:\n");
    unsigned int packets_processed, packets_forwarded, packets_dropped;
    double current_pps;

    result = mixnode_get_stats(mixnode, &packets_processed, &packets_forwarded, &packets_dropped, &current_pps);
    if (result == betanet_Success) {
        printf("  Packets processed: %u\n", packets_processed);
        printf("  Packets forwarded: %u\n", packets_forwarded);
        printf("  Packets dropped: %u\n", packets_dropped);
        printf("  Current PPS: %.2f\n", current_pps);
    } else {
        printf("  Failed to get statistics: %s\n", betanet_error_message(result));
    }

    // Decode packet example
    printf("\nDecoding Sphinx packet...\n");
    SphinxPacketHandle* decoded_packet = sphinx_packet_decode(encoded_packet);
    if (decoded_packet) {
        printf("Packet decoded successfully!\n");
        sphinx_packet_free(decoded_packet);
    } else {
        printf("Failed to decode packet\n");
    }

    // Cleanup
    sphinx_packet_free(packet);
    betanet_buffer_free(payload);
    betanet_buffer_free(encoded_packet);

    mixnode_stop(mixnode);
    mixnode_free(mixnode);

    betanet_cleanup();

    printf("\n=== Mixnode Example Complete ===\n");
    return 0;
}
