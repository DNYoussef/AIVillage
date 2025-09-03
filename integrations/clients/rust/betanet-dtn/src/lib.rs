//! Bundle Protocol v7 implementation for Delay-Tolerant Networking (DTN)
//!
//! This crate implements Bundle Protocol v7 (RFC 9171) as a session layer for
//! convergence layers like BitChat (BLE) and Betanet (TCP/QUIC). It provides
//! store-and-forward messaging with support for intermittent connectivity.
//!
//! # Key Features
//!
//! - Bundle Protocol v7 with Primary, Canonical, and Payload blocks
//! - CRC integrity protection and age tracking
//! - Sled-backed persistent storage
//! - Contact Graph Routing with policy support
//! - Convergence layer abstraction
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐
//! │   Application   │    │   Application   │
//! └─────────────────┘    └─────────────────┘
//!          │                       │
//! ┌─────────────────────────────────────────┐
//! │           DTN Bundle Layer              │
//! │   (Bundle Protocol v7 Session Plane)   │
//! └─────────────────────────────────────────┘
//!          │                       │
//! ┌─────────────────┐    ┌─────────────────┐
//! │  BitChat CLA    │    │  Betanet CLA    │
//! │     (BLE)       │    │  (TCP/QUIC)     │
//! └─────────────────┘    └─────────────────┘
//! ```

#![deny(clippy::all)]
#![allow(missing_docs)]
#![allow(dead_code)]

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// Core DTN modules
pub mod api;
pub mod bundle;
pub mod router;
pub mod sched;
pub mod storage;

// Security test modules
#[cfg(any(test, feature = "security-testing"))]
pub mod test_plaintext_guard;

// Re-export main types
pub use api::{DtnError, DtnNode, RegistrationInfo, SendBundleOptions};
pub use bundle::{
    BlockType, Bundle, BundleId, CanonicalBlock, EndpointId, PayloadBlock, PrimaryBlock,
};
pub use router::{ContactGraphRouter, ContactPlan, RoutingPolicy};
pub use sched::{LyapunovConfig, LyapunovScheduler, QueueState, SchedulingDecision};
pub use storage::{BundleStore, StorageError};

/// DTN protocol version (Bundle Protocol v7)
pub const DTN_VERSION: u8 = 7;

/// Maximum bundle size (16MB)
pub const MAX_BUNDLE_SIZE: usize = 16 * 1024 * 1024;

/// Default bundle lifetime (24 hours)
pub const DEFAULT_BUNDLE_LIFETIME: Duration = Duration::from_secs(24 * 60 * 60);

/// Bundle processing flags per BPv7 specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BundleControlFlags(u32);

impl BundleControlFlags {
    pub const NONE: Self = Self(0);
    pub const IS_FRAGMENT: Self = Self(1 << 0);
    pub const ADMINISTRATIVE_RECORD: Self = Self(1 << 1);
    pub const MUST_NOT_FRAGMENT: Self = Self(1 << 2);
    pub const CUSTODY_TRANSFER_REQUESTED: Self = Self(1 << 3);
    pub const DESTINATION_SINGLETON: Self = Self(1 << 4);
    pub const ACKNOWLEDGMENT_REQUESTED: Self = Self(1 << 5);
    pub const RESERVED_6: Self = Self(1 << 6);
    pub const RESERVED_7: Self = Self(1 << 7);
    pub const RESERVED_8: Self = Self(1 << 8);
    pub const RESERVED_9: Self = Self(1 << 9);
    pub const RESERVED_10: Self = Self(1 << 10);
    pub const RESERVED_11: Self = Self(1 << 11);
    pub const RESERVED_12: Self = Self(1 << 12);
    pub const RESERVED_13: Self = Self(1 << 13);
    pub const RESERVED_14: Self = Self(1 << 14);
    pub const RESERVED_15: Self = Self(1 << 15);
    pub const RESERVED_16: Self = Self(1 << 16);
    pub const RESERVED_17: Self = Self(1 << 17);
    pub const RESERVED_18: Self = Self(1 << 18);
    pub const RESERVED_19: Self = Self(1 << 19);
    pub const RESERVED_20: Self = Self(1 << 20);
    /// Bits 21-22 encode bundle priority as defined by BPv7
    pub const PRIORITY_SHIFT: u32 = 21;
    pub const PRIORITY_MASK: u32 = 0b11 << 21;

    pub fn new(value: u32) -> Self {
        Self(value)
    }

    pub fn value(self) -> u32 {
        self.0
    }

    pub fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) != 0
    }

    pub fn set(&mut self, flag: Self) {
        self.0 |= flag.0;
    }

    pub fn unset(&mut self, flag: Self) {
        self.0 &= !flag.0;
    }

    /// Extract the priority encoded in the control flags
    pub fn priority(self) -> u8 {
        ((self.0 & Self::PRIORITY_MASK) >> Self::PRIORITY_SHIFT) as u8
    }

    /// Create flags with the given priority value
    pub fn with_priority(priority: u8) -> Self {
        Self((priority as u32) << Self::PRIORITY_SHIFT)
    }

    /// Set the priority bits, preserving other flags
    pub fn set_priority(&mut self, priority: u8) {
        self.0 = (self.0 & !Self::PRIORITY_MASK) | ((priority as u32) << Self::PRIORITY_SHIFT);
    }
}

/// Bundle creation timestamp
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CreationTimestamp {
    /// Timestamp in DTN time (seconds since year 2000)
    pub dtn_time: u64,
    /// Sequence number for bundles created at the same time
    pub sequence_number: u64,
}

impl CreationTimestamp {
    pub fn now() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        // DTN time epoch is January 1, 2000 00:00:00 UTC
        // Unix epoch is January 1, 1970 00:00:00 UTC
        // Difference is 30 years = 946684800 seconds
        const DTN_EPOCH_OFFSET: u64 = 946684800;

        let dtn_time = now.as_secs().saturating_sub(DTN_EPOCH_OFFSET);

        Self {
            dtn_time,
            sequence_number: rand::random(),
        }
    }

    pub fn is_expired(&self, lifetime_ms: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        const DTN_EPOCH_OFFSET: u64 = 946684800;
        let current_dtn_time = now.as_secs().saturating_sub(DTN_EPOCH_OFFSET);

        let age_ms = (current_dtn_time - self.dtn_time) * 1000;
        age_ms >= lifetime_ms
    }
}

/// Priority levels for bundle processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Priority {
    Bulk = 0,
    Normal = 1,
    Expedited = 2,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// CLA (Convergence Layer Adapter) trait for different transport mechanisms
#[async_trait::async_trait]
pub trait ConvergenceLayer: Send + Sync + 'static {
    /// Name of this convergence layer
    fn name(&self) -> &'static str;

    /// Maximum transmission unit for this CLA
    fn mtu(&self) -> usize;

    /// Send a bundle to a destination
    async fn send_bundle(
        &self,
        destination: &str, // String representation of address
        bundle: &Bundle,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // DTN INVARIANT: No plaintext at gateway boundaries
        // This debug assertion ensures payloads are encrypted before egress
        debug_assert!(
            bundle.payload.is_ciphertext(),
            "DTN gateway invariant violated: plaintext payload detected at egress boundary"
        );

        self.send_bundle_impl(destination, bundle).await
    }

    /// Implementation-specific bundle sending (called after plaintext guard)
    async fn send_bundle_impl(
        &self,
        destination: &str,
        bundle: &Bundle,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Start listening for incoming bundles
    async fn start_listening(
        &self,
        local_address: &str,
        bundle_handler: Box<dyn Fn(Bundle) + Send + Sync>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Stop the convergence layer
    async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Bundle statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BundleStats {
    pub bundles_created: u64,
    pub bundles_sent: u64,
    pub bundles_received: u64,
    pub bundles_forwarded: u64,
    pub bundles_delivered: u64,
    pub bundles_expired: u64,
    pub bundles_dropped: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

impl BundleStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_created(&mut self) {
        self.bundles_created += 1;
    }

    pub fn record_sent(&mut self, size: usize) {
        self.bundles_sent += 1;
        self.bytes_sent += size as u64;
    }

    pub fn record_received(&mut self, size: usize) {
        self.bundles_received += 1;
        self.bytes_received += size as u64;
    }

    pub fn record_forwarded(&mut self) {
        self.bundles_forwarded += 1;
    }

    pub fn record_delivered(&mut self) {
        self.bundles_delivered += 1;
    }

    pub fn record_expired(&mut self) {
        self.bundles_expired += 1;
    }

    pub fn record_dropped(&mut self) {
        self.bundles_dropped += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_control_flags() {
        let mut flags = BundleControlFlags::NONE;
        assert!(!flags.contains(BundleControlFlags::IS_FRAGMENT));

        flags.set(BundleControlFlags::IS_FRAGMENT);
        assert!(flags.contains(BundleControlFlags::IS_FRAGMENT));

        flags.unset(BundleControlFlags::IS_FRAGMENT);
        assert!(!flags.contains(BundleControlFlags::IS_FRAGMENT));
    }

    #[test]
    fn test_creation_timestamp() {
        let ts1 = CreationTimestamp::now();
        let ts2 = CreationTimestamp::now();

        // Should have different sequence numbers even if created at same time
        assert_ne!(ts1.sequence_number, ts2.sequence_number);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Expedited > Priority::Normal);
        assert!(Priority::Normal > Priority::Bulk);
    }

    #[test]
    fn test_bundle_stats() {
        let mut stats = BundleStats::new();
        assert_eq!(stats.bundles_created, 0);

        stats.record_created();
        assert_eq!(stats.bundles_created, 1);

        stats.record_sent(1024);
        assert_eq!(stats.bundles_sent, 1);
        assert_eq!(stats.bytes_sent, 1024);
    }
}
