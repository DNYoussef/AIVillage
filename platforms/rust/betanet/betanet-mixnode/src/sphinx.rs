use anyhow::{anyhow, Result};
use std::net::SocketAddr;

/// Simplified Sphinx packet used for testing.
/// Format: [len:u8][next_hop_bytes][payload]
pub struct SphinxPacket {
    pub next_hop: SocketAddr,
    pub payload: Vec<u8>,
}

impl SphinxPacket {
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(anyhow!("empty packet"));
        }
        let len = data[0] as usize;
        if data.len() < 1 + len {
            return Err(anyhow!("truncated packet"));
        }
        let addr_str = std::str::from_utf8(&data[1..1 + len])?;
        let next_hop: SocketAddr = addr_str.parse()?;
        Ok(SphinxPacket {
            next_hop,
            payload: data[1 + len..].to_vec(),
        })
    }

    pub fn peel(self) -> (SocketAddr, Vec<u8>) {
        (self.next_hop, self.payload)
    }
}

/// Serialize a packet for tests.
pub fn serialize(next_hop: SocketAddr, payload: &[u8]) -> Vec<u8> {
    let addr = next_hop.to_string();
    let mut out = Vec::with_capacity(1 + addr.len() + payload.len());
    out.push(addr.len() as u8);
    out.extend_from_slice(addr.as_bytes());
    out.extend_from_slice(payload);
    out
}
