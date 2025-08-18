#![no_main]

use libfuzzer_sys::fuzz_target;
use betanet_htx::parse_frame;
use bytes::Bytes;
use tokio::runtime::Runtime;
use std::net::SocketAddr;
use quinn::{Endpoint, ServerConfig, ClientConfig};
use quinn::rustls::{RootCertStore};
use quinn::rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer};

fuzz_target!(|data: &[u8]| {
    // QUIC datagram fuzzing exercises HTX frame parsing over quinn datagrams
    if data.is_empty() {
        return;
    }

    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return,
    };

    rt.block_on(async {
        // Generate a self-signed certificate for the test server
        let cert = match rcgen::generate_simple_self_signed(vec!["localhost".to_string()]) {
            Ok(c) => c,
            Err(_) => return,
        };
        let cert_der = CertificateDer::from(cert.cert);
        let key_der = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());

        // Set up server endpoint with DATAGRAM support
        let server_cfg = match ServerConfig::with_single_cert(vec![cert_der.clone()], key_der.into()) {
            Ok(cfg) => cfg,
            Err(_) => return,
        };
        let mut server_ep = match Endpoint::server(server_cfg, "127.0.0.1:0".parse::<SocketAddr>().unwrap()) {
            Ok(ep) => ep,
            Err(_) => return,
        };

        // Configure client endpoint trusting the generated cert
        let mut roots = RootCertStore::empty();
        if roots.add(cert_der).is_err() {
            return;
        }
        let client_cfg = match ClientConfig::with_root_certificates(std::sync::Arc::new(roots)) {
            Ok(cfg) => cfg,
            Err(_) => return,
        };
        let mut client_ep = match Endpoint::client("127.0.0.1:0".parse::<SocketAddr>().unwrap()) {
            Ok(ep) => ep,
            Err(_) => return,
        };
        client_ep.set_default_client_config(client_cfg);

        let server_addr = match server_ep.local_addr() {
            Ok(addr) => addr,
            Err(_) => return,
        };

        // Establish connection
        let connecting = match client_ep.connect(server_addr, "localhost") {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut client_conn = match connecting.await {
            Ok(conn) => conn,
            Err(_) => return,
        };
        let mut server_conn = match server_ep.accept().await {
            Some(c) => match c.await {
                Ok(conn) => conn,
                Err(_) => return,
            },
            None => return,
        };

        // Send fuzz input as QUIC DATAGRAM from client to server
        let _ = client_conn.send_datagram(Bytes::copy_from_slice(data));

        // Read datagram on server side and attempt HTX frame parsing
        if let Ok(received) = server_conn.read_datagram().await {
            let _ = parse_frame(received.as_ref());
        }

        // Echo back fuzz data to exercise client receive path
        let _ = server_conn.send_datagram(Bytes::copy_from_slice(data));
        if let Ok(received) = client_conn.read_datagram().await {
            let _ = parse_frame(received.as_ref());
        }
    });
});

