#![cfg(feature = "vrf")]

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use betanet_mixnode::vrf_delay::VrfKeyPair;
use betanet_mixnode::vrf_neighbor::{NeighborSelectionConfig, MixnodeInfo, VrfNeighborSelector};
use schnorrkel::{signing_context, PublicKey};

#[test]
fn vrf_delay_is_deterministic_and_verifiable() {
    let keypair = VrfKeyPair::from_seed([42u8; 32]).unwrap();
    let message = b"delay-seed";
    let proof = keypair.prove(message).unwrap();
    assert!(keypair.verify(message, &proof));

    let min_delay = Duration::from_millis(100);
    let max_delay = Duration::from_millis(1000);
    let delay1 = proof.extract_delay(min_delay, max_delay);
    let delay2 = proof.extract_delay(min_delay, max_delay);
    assert_eq!(delay1, delay2);
}

#[test]
fn neighbor_selection_returns_proof() {
    let config = NeighborSelectionConfig { max_neighbors: 4, ..Default::default() };
    let mut selector = VrfNeighborSelector::with_vrf_key([7u8; 32], config);

    // add nodes across different AS numbers
    for i in 0..5u16 {
        let addr = SocketAddr::new(IpAddr::from([127, 0, 0, 1]), 8000 + i);
        let kp = VrfKeyPair::from_seed([i as u8 + 1; 32]).unwrap();
        let node = MixnodeInfo::new(addr, 1000 + i as u32, kp.public_key());
        selector.add_node(node);
    }

    let seed = b"integration-seed";
    let (neighbors1, proof1) = selector.select_neighbors(seed).unwrap();
    let (neighbors2, proof2) = selector.select_neighbors(seed).unwrap();
    assert_eq!(neighbors1, neighbors2);

    // verify proof
    let pk = PublicKey::from_bytes(selector.vrf_public_key()).unwrap();
    let ctx = signing_context(b"betanet-mixnode-vrf");
    assert!(pk
        .vrf_verify(ctx.bytes(seed), &proof1.io.to_preout(), &proof1.proof)
        .is_ok());
    assert_eq!(proof1.io.as_output_bytes(), proof2.io.as_output_bytes());
}
