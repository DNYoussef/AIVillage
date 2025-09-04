use betanet_htx::{HtxConfig, Frame};
use betanet_htx::quic::QuicTransport;
use bytes::Bytes;
use quinn::{Endpoint, ServerConfig, TransportConfig};
use rcgen::generate_simple_self_signed;
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer};
use std::sync::Arc;

#[tokio::test]
async fn test_quic_datagram_echo() {
    // Generate a self-signed certificate for the server
    let cert = generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
    let cert_der = CertificateDer::from(cert.cert);
    let key_der = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());

    let mut server_config = ServerConfig::with_single_cert(vec![cert_der], key_der.into()).unwrap();
    let mut transport = TransportConfig::default();
    transport.max_datagram_frame_size(Some(65535));
    server_config.transport = Arc::new(transport);

    let mut endpoint = Endpoint::server(server_config, "127.0.0.1:0".parse().unwrap()).unwrap();
    let server_addr = endpoint.local_addr().unwrap();

    // Echo server task
    tokio::spawn(async move {
        if let Some(connecting) = endpoint.accept().await {
            if let Ok(conn) = connecting.await {
                if let Ok(data) = conn.read_datagram().await {
                    let _ = conn.send_datagram(data);
                }
                conn.close(0u32.into(), b"done");
            }
        }
    });

    // Client configuration
    let mut config = HtxConfig::default();
    config.enable_quic = true;
    config.enable_tls_camouflage = false;
    config.alpn_protocols = vec!["h3".to_string()];

    let mut transport = QuicTransport::connect(server_addr, &config).await.unwrap();
    let frame = Frame::data(1, Bytes::from_static(b"hello")).unwrap();
    transport.send_datagram(frame.clone()).await.unwrap();

    let received = transport.recv_datagram().await.unwrap().expect("no datagram");
    assert_eq!(received.payload, frame.payload);

    transport.close().await.unwrap();
}
