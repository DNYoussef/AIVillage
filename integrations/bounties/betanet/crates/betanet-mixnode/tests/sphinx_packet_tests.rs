use betanet_mixnode::sphinx::{SphinxPacket, RoutingInfo};

#[test]
fn test_routing_decisions() {
    // Final destination packet
    let mut final_packet = SphinxPacket::new();
    final_packet.header.routing_info = RoutingInfo::new([0u8;16], 0, 0, true).to_bytes();
    final_packet.payload[..5].copy_from_slice(b"hello");
    let result = final_packet.process().expect("process");
    assert_eq!(result, Some(final_packet.payload.to_vec()));

    // Intermediate hop packet should not return payload
    let mut interm_packet = SphinxPacket::new();
    interm_packet.header.routing_info = RoutingInfo::new([0u8;16], 0, 0, false).to_bytes();
    let result2 = interm_packet.process().expect("process");
    assert!(result2.is_none());
}

#[test]
fn test_replay_protection() {
    let mut packet = SphinxPacket::new();
    packet.header.routing_info = RoutingInfo::new([0u8;16], 0, 0, true).to_bytes();
    packet.payload[..4].copy_from_slice(b"test");

    // First processing succeeds
    assert!(packet.process().unwrap().is_some());
    // Second processing should be dropped due to replay detection
    assert!(packet.process().unwrap().is_none());
}
