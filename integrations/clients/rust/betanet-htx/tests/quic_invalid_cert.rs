#![cfg(feature = "quic")]
use betanet_htx::quic::QuicTransport;
use betanet_htx::HtxConfig;
use std::net::SocketAddr;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn rejects_untrusted_certificate() {
    let addr: SocketAddr = "127.0.0.1:14444".parse().unwrap();

    let mut server_cfg = HtxConfig::default();
    server_cfg.listen_addr = addr;
    server_cfg.enable_quic = true;

    let handle = tokio::spawn(async move {
        let _ = QuicTransport::listen(server_cfg, |_| {}).await;
    });

    sleep(Duration::from_millis(100)).await;

    let client_cfg = HtxConfig::default();
    let res = QuicTransport::connect(addr, &client_cfg, None).await;
    assert!(
        res.is_err(),
        "connection with invalid certificate succeeded"
    );

    handle.abort();
}
