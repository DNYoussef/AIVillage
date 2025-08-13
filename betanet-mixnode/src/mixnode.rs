use crate::{policy::select_next_hop, sphinx::SphinxPacket};
use anyhow::Result;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::{net::UdpSocket, time::{self, Duration}};

/// Basic mixnode handling UDP packets with simplified Sphinx peeling.
pub struct MixNode {
    pub bind: SocketAddr,
    pub peers: Vec<SocketAddr>,
}

impl MixNode {
    /// Run the mixnode until process is killed.
    pub async fn run(&self) -> Result<()> {
        static PPS: AtomicU64 = AtomicU64::new(0);
        static DROPS: AtomicU64 = AtomicU64::new(0);
        let socket = UdpSocket::bind(self.bind).await?;
        let mut buf = vec![0u8; 1500];
        loop {
            tokio::select! {
                res = socket.recv_from(&mut buf) => {
                    if let Ok((n, _src)) = res {
                        PPS.fetch_add(1, Ordering::Relaxed);
                        match SphinxPacket::parse(&buf[..n]) {
                            Ok(pkt) => {
                                let (hop, payload) = pkt.peel();
                                let _ = socket.send_to(&payload, hop).await;
                            }
                            Err(_) => {
                                DROPS.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
                _ = time::sleep(Duration::from_secs(1)) => {
                    // cover traffic
                    if let Some(hop) = select_next_hop(b"cover", &self.peers) {
                        let _ = socket.send_to(b"cover", hop).await;
                    }
                }
            }
        }
    }
}
