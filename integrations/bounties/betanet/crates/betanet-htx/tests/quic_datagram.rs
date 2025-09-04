#[cfg(feature = "quic")]
use betanet_htx::{quic::QuicTransport, Frame, HtxConfig};
#[cfg(feature = "quic")]
use bytes::Bytes;
#[cfg(feature = "quic")]
use tokio::time::{sleep, Duration};

#[cfg(feature = "quic")]
#[tokio::test]
async fn quic_datagram_echo() {
    let mut config = HtxConfig::default();
    config.enable_tcp = false;
    config.enable_quic = true;
    config.listen_addr = "127.0.0.1:9443".parse().unwrap();

    let server_cfg = config.clone();
    let server = tokio::spawn(async move {
        QuicTransport::listen(server_cfg, |mut conn| {
            tokio::spawn(async move {
                if let Ok(Some(frame)) = conn.recv_datagram().await {
                    let _ = conn.send_datagram(frame).await;
                }
            });
        })
        .await
        .unwrap();
    });

    // Give server time to start
    sleep(Duration::from_millis(100)).await;

    let mut client = QuicTransport::connect(config.listen_addr, &config)
        .await
        .expect("client connect");
    let frame = Frame::data(1, Bytes::from_static(b"hello")).unwrap();
    client.send_datagram(frame.clone()).await.unwrap();
    let echoed = client.recv_datagram().await.unwrap().expect("no frame");
    assert_eq!(echoed.payload, frame.payload);
    client.close().await.unwrap();

    server.abort();
}
