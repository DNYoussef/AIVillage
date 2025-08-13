//! SCION-inspired gateway with signed CBOR control messages
//!
//! Provides path-aware networking with multi-path routing, signed control messages,
//! and governance integration for the Betanet overlay.

use crate::{
    config::GatewayConfig,
    error::{BetanetError, Result},
    BetanetMessage, BetanetPeer,
};

use bytes::Bytes;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// SCION-ish gateway for path-aware networking
pub struct ScionGateway {
    /// Configuration
    config: GatewayConfig,
    /// Gateway ID
    gateway_id: String,
    /// Signing key for control messages
    signing_key: SigningKey,
    /// Public key for verification
    public_key: VerifyingKey,
    /// Path database
    path_db: Arc<RwLock<PathDatabase>>,
    /// Beacon service
    beacon_service: Arc<BeaconService>,
    /// Control message handler
    control_handler: Arc<ControlMessageHandler>,
    /// AS-level topology
    as_topology: Arc<RwLock<AsTopology>>,
    /// Running state
    is_running: Arc<RwLock<bool>>,
}

/// Path database for storing available paths
#[derive(Debug, Clone)]
struct PathDatabase {
    /// Available paths to destinations
    paths: HashMap<String, Vec<PathInfo>>,
    /// Path selection strategy
    selection_strategy: PathSelectionStrategy,
    /// Path freshness timeout
    path_timeout: Duration,
    /// Last update timestamp
    last_update: SystemTime,
}

/// Information about a network path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathInfo {
    /// Path ID
    pub path_id: String,
    /// Destination AS
    pub destination_as: u32,
    /// Intermediate ASes
    pub intermediate_ases: Vec<u32>,
    /// Path latency
    pub latency_ms: u32,
    /// Path bandwidth
    pub bandwidth_mbps: u32,
    /// Path reliability score
    pub reliability: f32,
    /// Path cost
    pub cost: u32,
    /// Path expiration
    pub expires_at: u64,
    /// Path metadata
    pub metadata: HashMap<String, String>,
}

/// Path selection strategies
#[derive(Debug, Clone)]
pub enum PathSelectionStrategy {
    /// Lowest latency first
    LatencyFirst,
    /// Highest bandwidth first
    BandwidthFirst,
    /// Load balancing across paths
    LoadBalance,
    /// Cost-optimized selection
    CostOptimized,
    /// AS diversity maximization
    AsDiversity,
}

/// Beacon service for path discovery
struct BeaconService {
    /// Local AS number
    local_as: u32,
    /// Beacon broadcast interval
    beacon_interval: Duration,
    /// Active beacons
    active_beacons: Arc<Mutex<HashMap<String, BeaconInfo>>>,
    /// Beacon statistics
    stats: Arc<Mutex<BeaconStats>>,
}

/// Beacon information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BeaconInfo {
    /// Beacon ID
    pub beacon_id: String,
    /// Origin AS
    pub origin_as: u32,
    /// Path segments
    pub path_segments: Vec<PathSegment>,
    /// Beacon timestamp
    pub timestamp: u64,
    /// Beacon signature
    pub signature: Option<Bytes>,
}

/// Path segment in a beacon
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PathSegment {
    /// AS number
    pub as_number: u32,
    /// Ingress interface
    pub ingress_if: u32,
    /// Egress interface
    pub egress_if: u32,
    /// Segment latency
    pub latency_ms: u32,
    /// Segment bandwidth
    pub bandwidth_mbps: u32,
}

/// Control message handler for signed CBOR messages
struct ControlMessageHandler {
    /// Signing key
    signing_key: SigningKey,
    /// Known public keys for verification
    known_keys: Arc<RwLock<HashMap<String, VerifyingKey>>>,
    /// Message sequence numbers
    sequence_numbers: Arc<Mutex<HashMap<String, u64>>>,
    /// Control message stats
    stats: Arc<Mutex<ControlMessageStats>>,
}

/// CBOR control message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlMessage {
    /// Message type
    pub msg_type: ControlMessageType,
    /// Source gateway ID
    pub source: String,
    /// Destination gateway ID
    pub destination: String,
    /// Message sequence number
    pub sequence: u64,
    /// Message timestamp
    pub timestamp: u64,
    /// Message payload
    pub payload: ControlPayload,
    /// Digital signature
    pub signature: Option<Bytes>,
}

/// Control message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessageType {
    /// Path advertisement
    PathAdvertisement,
    /// Path withdrawal
    PathWithdrawal,
    /// Beacon propagation
    BeaconPropagation,
    /// AS topology update
    TopologyUpdate,
    /// Governance announcement
    GovernanceAnnouncement,
    /// Gateway keepalive
    Keepalive,
}

/// Control message payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlPayload {
    /// Path advertisement payload
    PathAd(PathAdvertisement),
    /// Path withdrawal payload
    PathWithdrawal(PathWithdrawal),
    /// Beacon payload
    Beacon(BeaconInfo),
    /// Topology update payload
    Topology(TopologyUpdate),
    /// Governance payload
    Governance(GovernanceMessage),
    /// Keepalive payload
    Keepalive(KeepaliveMessage),
}

/// Path advertisement message
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PathAdvertisement {
    pub paths: Vec<PathInfo>,
    pub advertiser_as: u32,
    pub advertisement_id: String,
}

/// Path withdrawal message
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PathWithdrawal {
    pub withdrawn_paths: Vec<String>,
    pub withdrawer_as: u32,
    pub withdrawal_reason: String,
}

/// Topology update message
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TopologyUpdate {
    pub as_links: Vec<AsLink>,
    pub update_type: TopologyUpdateType,
    pub update_id: String,
}

/// AS link information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AsLink {
    pub from_as: u32,
    pub to_as: u32,
    pub link_type: LinkType,
    pub bandwidth_mbps: u32,
    pub latency_ms: u32,
    pub cost: u32,
}

/// Link types
#[derive(Debug, Clone, Serialize, Deserialize)]
enum LinkType {
    /// Provider-customer link
    ProviderCustomer,
    /// Peer-to-peer link
    PeerToPeer,
    /// Sibling link
    Sibling,
}

/// Topology update types
#[derive(Debug, Clone, Serialize, Deserialize)]
enum TopologyUpdateType {
    /// Link addition
    LinkAdd,
    /// Link removal
    LinkRemove,
    /// Link modification
    LinkModify,
}

/// Governance message
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GovernanceMessage {
    pub proposal_id: String,
    pub proposal_type: String,
    pub voting_weight_caps: HashMap<String, f32>, // AS/Org -> weight cap
    pub subnet_vote_limits: HashMap<String, f32>, // /24, /48 -> limit
}

/// Keepalive message
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KeepaliveMessage {
    pub gateway_status: GatewayStatus,
    pub load_metrics: LoadMetrics,
}

/// Gateway status
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GatewayStatus {
    pub uptime_seconds: u64,
    pub active_connections: u32,
    pub path_count: u32,
    pub beacon_count: u32,
}

/// Load metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LoadMetrics {
    pub cpu_usage_percent: f32,
    pub memory_usage_percent: f32,
    pub network_utilization_percent: f32,
    pub messages_per_second: u32,
}

/// AS topology manager
#[derive(Debug, Clone)]
struct AsTopology {
    /// AS relationships
    as_relationships: HashMap<u32, AsInfo>,
    /// AS links
    as_links: HashMap<(u32, u32), AsLink>,
    /// Policy configurations
    policies: HashMap<u32, AsPolicy>,
}

/// AS information
#[derive(Debug, Clone)]
struct AsInfo {
    pub as_number: u32,
    pub as_name: String,
    pub as_type: AsType,
    pub governance_weight: f32,
    pub reputation_score: f32,
}

/// AS types
#[derive(Debug, Clone)]
enum AsType {
    /// Internet Service Provider
    Isp,
    /// Content Provider
    ContentProvider,
    /// Enterprise
    Enterprise,
    /// Academic
    Academic,
    /// Government
    Government,
}

/// AS policy configuration
#[derive(Debug, Clone)]
struct AsPolicy {
    pub export_policies: Vec<ExportPolicy>,
    pub import_policies: Vec<ImportPolicy>,
    pub transit_policies: Vec<TransitPolicy>,
}

/// Export policy
#[derive(Debug, Clone)]
struct ExportPolicy {
    pub target_as: Option<u32>,
    pub path_filter: PathFilter,
    pub action: PolicyAction,
}

/// Import policy
#[derive(Debug, Clone)]
struct ImportPolicy {
    pub source_as: Option<u32>,
    pub path_filter: PathFilter,
    pub action: PolicyAction,
}

/// Transit policy
#[derive(Debug, Clone)]
struct TransitPolicy {
    pub ingress_as: u32,
    pub egress_as: u32,
    pub action: PolicyAction,
}

/// Path filter criteria
#[derive(Debug, Clone)]
struct PathFilter {
    pub min_bandwidth: Option<u32>,
    pub max_latency: Option<u32>,
    pub max_cost: Option<u32>,
    pub required_ases: Vec<u32>,
    pub forbidden_ases: Vec<u32>,
}

/// Policy actions
#[derive(Debug, Clone)]
enum PolicyAction {
    Allow,
    Deny,
    Modify(PathModification),
}

/// Path modification
#[derive(Debug, Clone)]
struct PathModification {
    pub cost_multiplier: Option<f32>,
    pub latency_addition: Option<u32>,
    pub bandwidth_limit: Option<u32>,
}

/// Statistics structures
#[derive(Debug, Clone, Default)]
struct BeaconStats {
    pub beacons_sent: u64,
    pub beacons_received: u64,
    pub beacon_errors: u64,
    pub paths_discovered: u64,
}

#[derive(Debug, Clone, Default)]
struct ControlMessageStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub signature_verifications: u64,
    pub signature_failures: u64,
}

impl ScionGateway {
    /// Create new SCION gateway
    pub async fn new(config: GatewayConfig) -> Result<Self> {
        info!("Creating SCION gateway with AS: {}", config.local_as);

        // Generate or load signing keypair
        let signing_key = SigningKey::generate(&mut OsRng);
        let public_key = signing_key.verifying_key();

        let gateway_id = format!("gateway_{}_{}", config.local_as, hex::encode(&public_key.to_bytes()[..8]));

        // Initialize components
        let path_db = Arc::new(RwLock::new(PathDatabase {
            paths: HashMap::new(),
            selection_strategy: PathSelectionStrategy::LoadBalance,
            path_timeout: Duration::from_secs(300), // 5 minutes
            last_update: SystemTime::now(),
        }));

        let beacon_service = Arc::new(BeaconService {
            local_as: config.local_as,
            beacon_interval: Duration::from_secs(30),
            active_beacons: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(BeaconStats::default())),
        });

        let control_handler = Arc::new(ControlMessageHandler {
            signing_key: signing_key.clone(),
            known_keys: Arc::new(RwLock::new(HashMap::new())),
            sequence_numbers: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(ControlMessageStats::default())),
        });

        let as_topology = Arc::new(RwLock::new(AsTopology {
            as_relationships: HashMap::new(),
            as_links: HashMap::new(),
            policies: HashMap::new(),
        }));

        Ok(Self {
            config,
            gateway_id,
            signing_key,
            public_key,
            path_db,
            beacon_service,
            control_handler,
            as_topology,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the gateway
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            warn!("SCION gateway already running");
            return Ok(());
        }

        info!("Starting SCION gateway: {}", self.gateway_id);

        // Start beacon service
        self.start_beacon_service().await;

        // Start control message processing
        self.start_control_message_processing().await;

        // Start path management
        self.start_path_management().await;

        *is_running = true;
        info!("SCION gateway started successfully");

        Ok(())
    }

    /// Stop the gateway
    pub async fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if !*is_running {
            warn!("SCION gateway not running");
            return Ok(());
        }

        info!("Stopping SCION gateway: {}", self.gateway_id);

        // Clear state
        self.path_db.write().await.paths.clear();
        self.beacon_service.active_beacons.lock().await.clear();

        *is_running = false;
        info!("SCION gateway stopped");

        Ok(())
    }

    /// Select best path to destination
    pub async fn select_path(&self, destination_as: u32, requirements: &PathRequirements) -> Option<PathInfo> {
        let path_db = self.path_db.read().await;
        let destination_key = destination_as.to_string();

        if let Some(paths) = path_db.paths.get(&destination_key) {
            let mut suitable_paths: Vec<&PathInfo> = paths
                .iter()
                .filter(|path| self.path_meets_requirements(path, requirements))
                .collect();

            if suitable_paths.is_empty() {
                return None;
            }

            // Apply selection strategy
            match path_db.selection_strategy {
                PathSelectionStrategy::LatencyFirst => {
                    suitable_paths.sort_by_key(|p| p.latency_ms);
                }
                PathSelectionStrategy::BandwidthFirst => {
                    suitable_paths.sort_by_key(|p| std::cmp::Reverse(p.bandwidth_mbps));
                }
                PathSelectionStrategy::CostOptimized => {
                    suitable_paths.sort_by_key(|p| p.cost);
                }
                PathSelectionStrategy::LoadBalance => {
                    // Simple round-robin for load balancing
                    use rand::seq::SliceRandom;
                    suitable_paths.shuffle(&mut rand::thread_rng());
                }
                PathSelectionStrategy::AsDiversity => {
                    // Prefer paths with more AS diversity
                    suitable_paths.sort_by_key(|p| std::cmp::Reverse(p.intermediate_ases.len()));
                }
            }

            suitable_paths.first().cloned().cloned()
        } else {
            None
        }
    }

    /// Send control message
    pub async fn send_control_message(&self, message: ControlMessage) -> Result<()> {
        let signed_message = self.control_handler.sign_message(message).await?;
        let cbor_data = self.serialize_control_message(&signed_message).await?;

        // In real implementation, would send over network
        debug!("Sending control message: {} bytes", cbor_data.len());

        let mut stats = self.control_handler.stats.lock().await;
        stats.messages_sent += 1;

        Ok(())
    }

    /// Handle incoming control message
    pub async fn handle_control_message(&self, cbor_data: Bytes) -> Result<()> {
        let message = self.deserialize_control_message(cbor_data).await?;

        // Verify signature
        if !self.control_handler.verify_message(&message).await? {
            return Err(BetanetError::Gateway("Invalid message signature".to_string()));
        }

        let mut stats = self.control_handler.stats.lock().await;
        stats.messages_received += 1;
        stats.signature_verifications += 1;
        drop(stats);

        // Process message based on type
        match message.payload {
            ControlPayload::PathAd(path_ad) => {
                self.handle_path_advertisement(path_ad).await?;
            }
            ControlPayload::PathWithdrawal(withdrawal) => {
                self.handle_path_withdrawal(withdrawal).await?;
            }
            ControlPayload::Beacon(beacon) => {
                self.handle_beacon(beacon).await?;
            }
            ControlPayload::Topology(topology_update) => {
                self.handle_topology_update(topology_update).await?;
            }
            ControlPayload::Governance(governance) => {
                self.handle_governance_message(governance).await?;
            }
            ControlPayload::Keepalive(keepalive) => {
                self.handle_keepalive(keepalive).await?;
            }
        }

        Ok(())
    }

    /// Get gateway statistics
    pub async fn get_statistics(&self) -> GatewayStatistics {
        let path_db = self.path_db.read().await;
        let beacon_stats = self.beacon_service.stats.lock().await.clone();
        let control_stats = self.control_handler.stats.lock().await.clone();

        GatewayStatistics {
            gateway_id: self.gateway_id.clone(),
            local_as: self.config.local_as,
            total_paths: path_db.paths.values().map(|p| p.len()).sum::<usize>() as u64,
            active_beacons: self.beacon_service.active_beacons.lock().await.len() as u64,
            beacon_stats,
            control_message_stats: control_stats,
            uptime_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    // Private helper methods

    async fn path_meets_requirements(&self, path: &PathInfo, requirements: &PathRequirements) -> bool {
        if let Some(max_latency) = requirements.max_latency_ms {
            if path.latency_ms > max_latency {
                return false;
            }
        }

        if let Some(min_bandwidth) = requirements.min_bandwidth_mbps {
            if path.bandwidth_mbps < min_bandwidth {
                return false;
            }
        }

        if let Some(max_cost) = requirements.max_cost {
            if path.cost > max_cost {
                return false;
            }
        }

        // Check forbidden ASes
        for forbidden_as in &requirements.forbidden_ases {
            if path.intermediate_ases.contains(forbidden_as) {
                return false;
            }
        }

        true
    }

    async fn start_beacon_service(&self) {
        let beacon_service = self.beacon_service.clone();
        let gateway_id = self.gateway_id.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(beacon_service.beacon_interval);

            loop {
                interval.tick().await;

                // Generate and propagate beacon
                let beacon = BeaconInfo {
                    beacon_id: format!("{}_{}", gateway_id, uuid::Uuid::new_v4()),
                    origin_as: beacon_service.local_as,
                    path_segments: vec![], // Would be populated with real path segments
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    signature: None, // Would be signed in real implementation
                };

                let mut stats = beacon_service.stats.lock().await;
                stats.beacons_sent += 1;
                drop(stats);

                debug!("Propagated beacon: {}", beacon.beacon_id);
            }
        });
    }

    async fn start_control_message_processing(&self) {
        // This would start a server to listen for incoming control messages
        info!("Control message processing started");
    }

    async fn start_path_management(&self) {
        let path_db = self.path_db.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                // Clean up expired paths
                let mut db = path_db.write().await;
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                for (_dest, paths) in db.paths.iter_mut() {
                    paths.retain(|path| path.expires_at > now);
                }

                // Remove empty destinations
                db.paths.retain(|_dest, paths| !paths.is_empty());
            }
        });
    }

    async fn handle_path_advertisement(&self, path_ad: PathAdvertisement) -> Result<()> {
        let mut path_db = self.path_db.write().await;

        for path in path_ad.paths {
            let dest_key = path.destination_as.to_string();
            path_db.paths.entry(dest_key).or_insert_with(Vec::new).push(path);
        }

        info!("Processed path advertisement from AS {}", path_ad.advertiser_as);
        Ok(())
    }

    async fn handle_path_withdrawal(&self, withdrawal: PathWithdrawal) -> Result<()> {
        let mut path_db = self.path_db.write().await;

        for path_id in withdrawal.withdrawn_paths {
            for (_dest, paths) in path_db.paths.iter_mut() {
                paths.retain(|path| path.path_id != path_id);
            }
        }

        info!("Processed path withdrawal from AS {}: {}",
              withdrawal.withdrawer_as, withdrawal.withdrawal_reason);
        Ok(())
    }

    async fn handle_beacon(&self, beacon: BeaconInfo) -> Result<()> {
        let mut active_beacons = self.beacon_service.active_beacons.lock().await;
        active_beacons.insert(beacon.beacon_id.clone(), beacon.clone());

        let mut stats = self.beacon_service.stats.lock().await;
        stats.beacons_received += 1;
        stats.paths_discovered += beacon.path_segments.len() as u64;

        debug!("Received beacon from AS {}", beacon.origin_as);
        Ok(())
    }

    async fn handle_topology_update(&self, topology_update: TopologyUpdate) -> Result<()> {
        let mut as_topology = self.as_topology.write().await;

        for link in topology_update.as_links {
            let link_key = (link.from_as, link.to_as);

            match topology_update.update_type {
                TopologyUpdateType::LinkAdd | TopologyUpdateType::LinkModify => {
                    as_topology.as_links.insert(link_key, link);
                }
                TopologyUpdateType::LinkRemove => {
                    as_topology.as_links.remove(&link_key);
                }
            }
        }

        info!("Processed topology update: {}", topology_update.update_id);
        Ok(())
    }

    async fn handle_governance_message(&self, governance: GovernanceMessage) -> Result<()> {
        info!("Received governance message for proposal: {}", governance.proposal_id);

        // Apply governance constraints
        // This would integrate with the main governance system
        for (entity, weight_cap) in governance.voting_weight_caps {
            debug!("Applied weight cap for {}: {}", entity, weight_cap);
        }

        for (subnet, vote_limit) in governance.subnet_vote_limits {
            debug!("Applied vote limit for {}: {}", subnet, vote_limit);
        }

        Ok(())
    }

    async fn handle_keepalive(&self, keepalive: KeepaliveMessage) -> Result<()> {
        debug!("Received keepalive: uptime={}s, connections={}",
               keepalive.gateway_status.uptime_seconds,
               keepalive.gateway_status.active_connections);
        Ok(())
    }

    async fn serialize_control_message(&self, message: &ControlMessage) -> Result<Bytes> {
        let cbor_data = ciborium::ser::to_vec(message).map_err(|e| {
            BetanetError::Serialization(crate::error::SerializationError::Cbor(e.to_string()))
        })?;
        Ok(Bytes::from(cbor_data))
    }

    async fn deserialize_control_message(&self, cbor_data: Bytes) -> Result<ControlMessage> {
        let message: ControlMessage = ciborium::de::from_reader(cbor_data.as_ref()).map_err(|e| {
            BetanetError::Serialization(crate::error::SerializationError::Cbor(e.to_string()))
        })?;
        Ok(message)
    }
}

impl ControlMessageHandler {
    /// Sign a control message
    async fn sign_message(&self, mut message: ControlMessage) -> Result<ControlMessage> {
        // Serialize message without signature
        let message_bytes = ciborium::ser::to_vec(&message).map_err(|e| {
            BetanetError::Serialization(crate::error::SerializationError::Cbor(e.to_string()))
        })?;

        // Sign the serialized message
        let signature = self.signing_key.sign(&message_bytes);
        message.signature = Some(Bytes::from(signature.to_bytes().to_vec()));

        Ok(message)
    }

    /// Verify a control message signature
    async fn verify_message(&self, message: &ControlMessage) -> Result<bool> {
        let signature_bytes = match &message.signature {
            Some(sig) => sig.clone(),
            None => return Ok(false), // No signature
        };

        // Get public key for source
        let known_keys = self.known_keys.read().await;
        let public_key = match known_keys.get(&message.source) {
            Some(key) => key,
            None => {
                // Unknown source - in real implementation would fetch key
                return Ok(false);
            }
        };

        // Create message copy without signature for verification
        let mut message_copy = message.clone();
        message_copy.signature = None;

        let message_bytes = ciborium::ser::to_vec(&message_copy).map_err(|e| {
            BetanetError::Serialization(crate::error::SerializationError::Cbor(e.to_string()))
        })?;

        // Verify signature
        if let Ok(signature) = Signature::from_bytes(signature_bytes.as_ref().try_into().unwrap_or(&[0u8; 64])) {
            Ok(public_key.verify(&message_bytes, &signature).is_ok())
        } else {
            let mut stats = self.stats.lock().await;
            stats.signature_failures += 1;
            Ok(false)
        }
    }
}

/// Path requirements for path selection
#[derive(Debug, Clone, Default)]
pub struct PathRequirements {
    pub max_latency_ms: Option<u32>,
    pub min_bandwidth_mbps: Option<u32>,
    pub max_cost: Option<u32>,
    pub forbidden_ases: Vec<u32>,
    pub required_ases: Vec<u32>,
}

/// Gateway statistics
#[derive(Debug, Clone)]
pub struct GatewayStatistics {
    pub gateway_id: String,
    pub local_as: u32,
    pub total_paths: u64,
    pub active_beacons: u64,
    pub beacon_stats: BeaconStats,
    pub control_message_stats: ControlMessageStats,
    pub uptime_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GatewayConfig;

    #[tokio::test]
    async fn test_scion_gateway_creation() {
        let config = GatewayConfig {
            local_as: 64512,
            gateway_port: 8080,
            beacon_interval_secs: 30,
            path_timeout_secs: 300,
        };

        let gateway = ScionGateway::new(config).await;
        assert!(gateway.is_ok());
    }

    #[test]
    fn test_path_info_creation() {
        let path = PathInfo {
            path_id: "test_path".to_string(),
            destination_as: 64513,
            intermediate_ases: vec![64512, 64514],
            latency_ms: 50,
            bandwidth_mbps: 1000,
            reliability: 0.99,
            cost: 100,
            expires_at: 1234567890,
            metadata: HashMap::new(),
        };

        assert_eq!(path.path_id, "test_path");
        assert_eq!(path.destination_as, 64513);
        assert_eq!(path.intermediate_ases.len(), 2);
    }

    #[test]
    fn test_control_message_creation() {
        let message = ControlMessage {
            msg_type: ControlMessageType::Keepalive,
            source: "gateway_1".to_string(),
            destination: "gateway_2".to_string(),
            sequence: 1,
            timestamp: 1234567890,
            payload: ControlPayload::Keepalive(KeepaliveMessage {
                gateway_status: GatewayStatus {
                    uptime_seconds: 3600,
                    active_connections: 10,
                    path_count: 5,
                    beacon_count: 3,
                },
                load_metrics: LoadMetrics {
                    cpu_usage_percent: 25.5,
                    memory_usage_percent: 60.0,
                    network_utilization_percent: 15.0,
                    messages_per_second: 100,
                },
            }),
            signature: None,
        };

        assert_eq!(message.source, "gateway_1");
        assert_eq!(message.sequence, 1);
    }
}
