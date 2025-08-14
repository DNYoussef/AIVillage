use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use chacha20poly1305::{aead::Aead, ChaCha20Poly1305, Key, Nonce};
use serde_json::json;
use tokio::runtime::Runtime;

use betanet_gateway::aead::{AeadManager, FrameType};
use betanet_gateway::config::GatewayConfig;
use betanet_gateway::metrics::MetricsCollector;

#[test]
fn kat_and_timing() {
    // RFC8439 ChaCha20-Poly1305 test vector
    let key = Key::from_slice(&hex::decode("1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0").unwrap());
    let nonce = Nonce::from_slice(&hex::decode("000000000102030405060708").unwrap());
    let plaintext = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
    let cipher = ChaCha20Poly1305::new(key);
    let ciphertext = cipher.encrypt(nonce, plaintext.as_ref()).unwrap();
    let expected = hex::decode("61af9619629b5fe123d030e198fc44f15a9f5cb37041f4cff1406645c77580a45138e05e231b16befe9ec6476a4467f3bfa3a3317f010a46dee43b75335594cd322d8e466d4593808881b50ba24484c440b0022a078c22ac3ebcdcf4fc2ec7459181e92bf5a2b861d69939022c244335624a4c3fa1adbe247fc01e20818a546fc77d").unwrap();
    assert_eq!(ciphertext, expected);

    // Write KAT result
    let verify_dir = Path::new("../tmp_scion_verify");
    fs::create_dir_all(verify_dir).unwrap();
    fs::write(verify_dir.join("kat_results.txt"), "KAT passed\n").unwrap();

    // Measure encryption/decryption times using AeadManager
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        let gateway_config = Arc::new(GatewayConfig::default());
        let metrics = Arc::new(MetricsCollector::new(gateway_config.clone()).unwrap());
        let manager = AeadManager::new(gateway_config.aead.clone(), metrics, [0u8; 32]);

        let peer = "timing_peer";
        let payload = b"hello";
        let aad = b"";

        let mut enc_times = Vec::new();
        let mut dec_times = Vec::new();

        for _ in 0..50 {
            let start = Instant::now();
            let frame = manager.encrypt_frame(peer, FrameType::ScionData, payload, aad).await.unwrap();
            enc_times.push(start.elapsed().as_micros() as u64);

            let start = Instant::now();
            manager.decrypt_frame(peer, &frame).await.unwrap();
            dec_times.push(start.elapsed().as_micros() as u64);
        }

        // Calculate median (p50)
        let mut enc_sorted = enc_times.clone();
        enc_sorted.sort_unstable();
        let mut dec_sorted = dec_times.clone();
        dec_sorted.sort_unstable();
        let p50_enc = enc_sorted[enc_sorted.len() / 2];
        let p50_dec = dec_sorted[dec_sorted.len() / 2];
        assert!(p50_enc <= 120);
        assert!(p50_dec <= 120);

        let hist = json!({
            "encrypt_us": enc_times,
            "decrypt_us": dec_times,
        });
        fs::write(
            verify_dir.join("timing_hist.json"),
            serde_json::to_string(&hist).unwrap(),
        )
        .unwrap();
    });
}
