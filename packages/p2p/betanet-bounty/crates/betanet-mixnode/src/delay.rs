//! Delay queue implementation

use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

/// Delayed packet entry
#[derive(Debug)]
struct DelayedPacket {
    packet: Vec<u8>,
    release_time: Instant,
}

impl PartialEq for DelayedPacket {
    fn eq(&self, other: &Self) -> bool {
        self.release_time == other.release_time
    }
}

impl Eq for DelayedPacket {}

impl PartialOrd for DelayedPacket {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DelayedPacket {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for min-heap behavior
        other.release_time.cmp(&self.release_time)
    }
}

/// Delay queue for packet processing
pub struct DelayQueue {
    queue: BinaryHeap<DelayedPacket>,
}

impl DelayQueue {
    /// Create new delay queue
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
        }
    }

    /// Add packet with delay
    pub async fn add_packet(&mut self, packet: Vec<u8>, delay: Duration) {
        let release_time = Instant::now() + delay;
        let delayed_packet = DelayedPacket {
            packet,
            release_time,
        };
        self.queue.push(delayed_packet);
    }

    /// Pop ready packet
    pub async fn pop_ready(&mut self) -> Option<Vec<u8>> {
        let now = Instant::now();

        if let Some(top) = self.queue.peek() {
            if top.release_time <= now {
                return Some(self.queue.pop().unwrap().packet);
            }
        }

        None
    }

    /// Get queue size
    pub fn size(&self) -> usize {
        self.queue.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl Default for DelayQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_delay_queue() {
        let mut queue = DelayQueue::new();
        let packet = vec![1, 2, 3, 4];

        // Add packet with 10ms delay
        queue
            .add_packet(packet.clone(), Duration::from_millis(10))
            .await;

        // Should not be ready immediately
        assert!(queue.pop_ready().await.is_none());

        // Wait and check again
        sleep(Duration::from_millis(15)).await;
        let result = queue.pop_ready().await;
        assert!(result.is_some());
        assert_eq!(result.unwrap(), packet);
    }
}
