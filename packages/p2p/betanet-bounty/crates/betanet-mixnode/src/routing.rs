//! Routing table implementation

use std::collections::HashMap;
use std::net::SocketAddr;

use crate::packet::Packet;

/// Routing table for mixnode
pub struct RoutingTable {
    routes: HashMap<u8, Vec<SocketAddr>>,
}

impl RoutingTable {
    /// Create new routing table
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
        }
    }

    /// Add route for layer
    pub fn add_route(&mut self, layer: u8, nodes: Vec<SocketAddr>) {
        self.routes.insert(layer, nodes);
    }

    /// Get next hop for packet
    pub async fn get_next_hop(&self, packet: &Packet) -> Option<SocketAddr> {
        let layer = packet.layer();

        if let Some(nodes) = self.routes.get(&layer) {
            if !nodes.is_empty() {
                // Simple round-robin selection
                let index = (rand::random::<usize>()) % nodes.len();
                return Some(nodes[index]);
            }
        }

        None
    }

    /// Remove route for layer
    pub fn remove_route(&mut self, layer: u8) {
        self.routes.remove(&layer);
    }

    /// Get all routes
    pub fn routes(&self) -> &HashMap<u8, Vec<SocketAddr>> {
        &self.routes
    }

    /// Check if route exists
    pub fn has_route(&self, layer: u8) -> bool {
        self.routes.contains_key(&layer)
    }
}

impl Default for RoutingTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_routing_table() {
        let mut table = RoutingTable::new();
        let addr: SocketAddr = "127.0.0.1:9002".parse().unwrap();

        table.add_route(1, vec![addr]);
        assert!(table.has_route(1));

        let packet = Packet::data(Bytes::from("test"), 1);
        let next_hop = table.get_next_hop(&packet).await;
        assert!(next_hop.is_some());
        assert_eq!(next_hop.unwrap(), addr);
    }
}
