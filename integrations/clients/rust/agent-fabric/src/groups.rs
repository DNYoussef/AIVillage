//! MLS group cohorts for secure group communication
//!
//! Provides end-to-end encrypted group messaging for agent coordination,
//! training sessions, alerts, and voting protocols.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::{debug, info, warn};

#[cfg(feature = "mls")]
use openmls::group::MlsGroup as MlsGroup_;
#[cfg(feature = "mls")]
use openmls::prelude::*;
#[cfg(feature = "mls")]
use openmls_rust_crypto::OpenMlsRustCrypto;
#[cfg(feature = "mls")]
use openmls_basic_credential::SignatureKeyPair;
#[cfg(feature = "mls")]
use std::io::Cursor;

use crate::{AgentFabricError, AgentId, Result};

/// MLS group configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupConfig {
    pub group_id: String,
    pub ciphersuite: u16,
    pub max_members: usize,
    pub admin_only_add: bool,
    pub admin_only_remove: bool,
    pub require_unanimous_votes: bool,
    pub vote_timeout_seconds: u64,
}

impl Default for GroupConfig {
    fn default() -> Self {
        Self {
            group_id: "default".to_string(),
            #[cfg(feature = "mls")]
            ciphersuite: 1, // Default to a basic ciphersuite ID
            #[cfg(not(feature = "mls"))]
            ciphersuite: 1, // Placeholder when MLS not available
            max_members: 100,
            admin_only_add: false,
            admin_only_remove: false,
            require_unanimous_votes: false,
            vote_timeout_seconds: 300, // 5 minutes
        }
    }
}

/// Group message types for different coordination activities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMessage {
    pub message_id: String,
    pub from: AgentId,
    pub message_type: GroupMessageType,
    pub payload: Bytes,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroupMessageType {
    /// Training coordination messages
    Training {
        session_id: String,
        action: TrainingAction,
    },
    /// Alert broadcasting
    Alert {
        alert_level: AlertLevel,
        category: String,
    },
    /// Voting and governance
    Vote {
        proposal_id: String,
        action: VoteAction,
    },
    /// General group communication
    Broadcast { category: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingAction {
    StartSession,
    UpdateModel,
    ShareGradients,
    SyncParameters,
    EndSession,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoteAction {
    Propose,
    Vote { approve: bool },
    Tally,
    Execute,
}

/// Group member information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMember {
    pub agent_id: AgentId,
    pub member_id: Vec<u8>,
    pub is_admin: bool,
    pub joined_at: u64,
    pub last_active: u64,
    pub vote_weight: u32,
    #[cfg(feature = "mls")]
    pub leaf_index: Option<u32>,
}

/// Voting proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteProposal {
    pub proposal_id: String,
    pub proposer: AgentId,
    pub title: String,
    pub description: String,
    pub votes_for: u32,
    pub votes_against: u32,
    pub total_weight: u32,
    pub deadline: u64,
    pub executed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistentState {
    members: Vec<GroupMember>,
    proposals: Vec<VoteProposal>,
}

#[async_trait]
pub trait GroupCallback: Send + Sync {
    async fn member_added(&self, _member: &GroupMember) {}
    async fn member_removed(&self, _agent: &AgentId) {}
    async fn vote_executed(&self, _proposal: &VoteProposal) {}
}

/// Group statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroupStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub members_added: u64,
    pub members_removed: u64,
    pub votes_initiated: u64,
    pub votes_completed: u64,
    pub training_sessions: u64,
    pub alerts_sent: u64,
}

/// MLS group implementation
pub struct MlsGroup {
    group_id: String,
    config: GroupConfig,
    #[cfg(feature = "mls")]
    mls_group: Arc<Mutex<Option<MlsGroup_>>>,
    #[cfg(not(feature = "mls"))]
    mls_group: Arc<Mutex<Option<()>>>, // Placeholder
    members: Arc<RwLock<HashMap<Vec<u8>, GroupMember>>>,
    proposals: Arc<RwLock<HashMap<String, VoteProposal>>>,
    message_queue: Arc<Mutex<mpsc::Sender<GroupMessage>>>,
    stats: Arc<RwLock<GroupStats>>,
    #[cfg(feature = "mls")]
    crypto_provider: Arc<OpenMlsRustCrypto>,
    #[cfg(not(feature = "mls"))]
    crypto_provider: (), // Placeholder
    #[cfg(feature = "mls")]
    key_store: Arc<RwLock<HashMap<Vec<u8>, KeyPackage>>>,
    #[cfg(not(feature = "mls"))]
    key_store: (), // Placeholder
    state_path: Arc<RwLock<Option<PathBuf>>>,
    callbacks: Arc<RwLock<Vec<Arc<dyn GroupCallback>>>>,
    #[cfg(feature = "mls")]
    signer: Arc<SignatureKeyPair>,
}

impl MlsGroup {
    /// Create a new MLS group
    pub async fn new(group_id: String, config: GroupConfig) -> Result<Self> {
        let (sender, _receiver) = mpsc::channel(1000);
        #[cfg(feature = "mls")]
        let signer = SignatureKeyPair::new(SignatureScheme::ED25519)
            .map_err(|e| AgentFabricError::MlsError(e.to_string()))?;

        let group = Self {
            group_id: group_id.clone(),
            config,
            mls_group: Arc::new(Mutex::new(None)),
            members: Arc::new(RwLock::new(HashMap::new())),
            proposals: Arc::new(RwLock::new(HashMap::new())),
            message_queue: Arc::new(Mutex::new(sender)),
            stats: Arc::new(RwLock::new(GroupStats::default())),
            #[cfg(feature = "mls")]
            crypto_provider: Arc::new(OpenMlsRustCrypto::default()),
            #[cfg(not(feature = "mls"))]
            crypto_provider: (),
            #[cfg(feature = "mls")]
            key_store: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(not(feature = "mls"))]
            key_store: (),
            state_path: Arc::new(RwLock::new(None)),
            callbacks: Arc::new(RwLock::new(Vec::new())),
            #[cfg(feature = "mls")]
            signer: Arc::new(signer),
        };

        info!("Created MLS group: {}", group_id);
        Ok(group)
    }

    pub async fn set_state_path(&self, path: PathBuf) -> Result<()> {
        {
            let mut guard = self.state_path.write().await;
            *guard = Some(path);
        }
        self.load_state().await
    }

    pub async fn register_callback(&self, cb: Arc<dyn GroupCallback>) {
        let mut callbacks = self.callbacks.write().await;
        callbacks.push(cb);
    }

    async fn load_state(&self) -> Result<()> {
        let path = { self.state_path.read().await.clone() };
        if let Some(path) = path {
            if let Ok(data) = fs::read(&path).await {
                if let Ok(state) = serde_json::from_slice::<PersistentState>(&data) {
                    let mut members = self.members.write().await;
                    members.clear();
                    for m in state.members {
                        members.insert(m.member_id.clone(), m);
                    }
                    let mut proposals = self.proposals.write().await;
                    proposals.clear();
                    for p in state.proposals {
                        proposals.insert(p.proposal_id.clone(), p);
                    }
                }
            }
        }
        Ok(())
    }

    async fn save_state(&self) -> Result<()> {
        let path = { self.state_path.read().await.clone() };
        if let Some(path) = path {
            let state = PersistentState {
                members: self.members.read().await.values().cloned().collect(),
                proposals: self.proposals.read().await.values().cloned().collect(),
            };
            let data = serde_json::to_vec(&state)
                .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
            fs::write(path, data)
                .await
                .map_err(|e| AgentFabricError::MlsError(e.to_string()))?;
        }
        Ok(())
    }

    /// Initialize group as creator (admin)
    pub async fn initialize_as_creator(&self, creator: AgentId) -> Result<()> {
        #[cfg(feature = "mls")]
        {
            let backend = self.crypto_provider.as_ref();
            let credential = Credential::new(
                creator.to_string().into_bytes(),
                CredentialType::Basic,
            )
            .map_err(|e| AgentFabricError::MlsError(e.to_string()))?;
            let credential_with_key = CredentialWithKey {
                credential,
                signature_key: self.signer.public().into(),
            };
            let group_id = GroupId::from_slice(self.group_id.as_bytes());
            let ciphersuite = Ciphersuite::MLS_128_DHKEMX25519_AES128GCM_SHA256_Ed25519;
            let mls_group_config = MlsGroupConfig::builder()
                .crypto_config(CryptoConfig::with_default_version(ciphersuite))
                .build();
            let mls_group = MlsGroup_::new_with_group_id(
                backend,
                self.signer.as_ref(),
                &mls_group_config,
                group_id,
                credential_with_key,
            )
            .map_err(|e| AgentFabricError::MlsError(format!("{e:?}")))?;
            {
                let mut g = self.mls_group.lock().await;
                *g = Some(mls_group);
            }
            let member_id = creator.to_string().into_bytes();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let member = GroupMember {
                agent_id: creator,
                member_id: member_id.clone(),
                is_admin: true,
                joined_at: now,
                last_active: now,
                vote_weight: 1,
                leaf_index: Some(0),
            };
            {
                let mut members = self.members.write().await;
                members.insert(member_id, member);
            }
            self.save_state().await?;
            info!("Initialized MLS group {} as creator", self.group_id);
            Ok(())
        }
        #[cfg(not(feature = "mls"))]
        {
            info!("Group initialization (simulated): {}", creator);
            let member_id = creator.to_string().into_bytes();
            let member = GroupMember {
                agent_id: creator,
                member_id: member_id.clone(),
                is_admin: true,
                joined_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                last_active: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                vote_weight: 1,
            };
            {
                let mut members = self.members.write().await;
                members.insert(member_id, member);
            }
            Ok(())
        }
    }

    /// Send message to group
    pub async fn send_message(&self, message: GroupMessage) -> Result<Vec<u8>> {
        let payload = serde_json::to_vec(&message)
            .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
        #[cfg(feature = "mls")]
        {
            let mut group_lock = self.mls_group.lock().await;
            let group = group_lock.as_mut().ok_or_else(|| {
                AgentFabricError::MlsError("Group not initialized".to_string())
            })?;
            let msg_out = group
                .create_message(
                    self.crypto_provider.as_ref(),
                    self.signer.as_ref(),
                    &payload,
                )
                .map_err(|e| AgentFabricError::MlsError(format!("{e:?}")))?;
            let bytes = msg_out
                .tls_serialize_detached()
                .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
            {
                let mut stats = self.stats.write().await;
                stats.messages_sent += 1;
                match &message.message_type {
                    GroupMessageType::Alert { .. } => stats.alerts_sent += 1,
                    _ => {}
                }
            }
            {
                let sender = self.message_queue.lock().await;
                if sender.send(message).await.is_err() {
                    warn!("Failed to queue message locally");
                }
            }
            Ok(bytes)
        }
        #[cfg(not(feature = "mls"))]
        {
            debug!("Group message sent (simulated) for group {}", self.group_id);
            {
                let mut stats = self.stats.write().await;
                stats.messages_sent += 1;
            }
            {
                let sender = self.message_queue.lock().await;
                if sender.send(message).await.is_err() {
                    warn!("Failed to queue message locally");
                }
            }
            Ok(payload)
        }
    }

    /// Process received MLS message
    pub async fn process_message(&self, mls_message: &[u8]) -> Result<Option<GroupMessage>> {
        #[cfg(feature = "mls")]
        {
            let mut group_lock = self.mls_group.lock().await;
            let group = group_lock.as_mut().ok_or_else(|| {
                AgentFabricError::MlsError("Group not initialized".to_string())
            })?;
            let mut cursor = Cursor::new(mls_message);
            let msg_in = MlsMessageIn::tls_deserialize(&mut cursor)
                .map_err(|e| AgentFabricError::MlsError(e.to_string()))?;
            let protocol = msg_in
                .into_protocol_message()
                .ok_or_else(|| AgentFabricError::MlsError("Invalid message".to_string()))?;
            let processed = group
                .process_message(self.crypto_provider.as_ref(), protocol)
                .map_err(|e| AgentFabricError::MlsError(e.to_string()))?;
            if let ProcessedMessageContent::ApplicationMessage(app) = processed.into_content() {
                let payload = app.into_bytes();
                let message: GroupMessage = serde_json::from_slice(&payload)
                    .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
                {
                    let mut stats = self.stats.write().await;
                    stats.messages_received += 1;
                }
                self.handle_group_message(&message).await?;
                return Ok(Some(message));
            }
            Ok(None)
        }
        #[cfg(not(feature = "mls"))]
        {
            let message: GroupMessage = serde_json::from_slice(mls_message)
                .map_err(|e| AgentFabricError::SerializationError(e.to_string()))?;
            {
                let mut stats = self.stats.write().await;
                stats.messages_received += 1;
            }
            self.handle_group_message(&message).await?;
            Ok(Some(message))
        }
    }

    /// Add member to group
    #[cfg(feature = "mls")]
    pub async fn add_member(&self, agent_id: AgentId, key_package: KeyPackage) -> Result<()> {
        let mut group_lock = self.mls_group.lock().await;
        let group = group_lock.as_mut().ok_or_else(|| {
            AgentFabricError::MlsError("Group not initialized".to_string())
        })?;
        group
            .add_members(
                self.crypto_provider.as_ref(),
                self.signer.as_ref(),
                &[key_package.clone()],
            )
            .map_err(|e| AgentFabricError::MlsError(format!("{e:?}")))?;
        group
            .merge_pending_commit(self.crypto_provider.as_ref())
            .map_err(|e| AgentFabricError::MlsError(format!("{e:?}")))?;
        let member_id = agent_id.to_string().into_bytes();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let leaf_index = group
            .members()
            .find(|m| m.credential.identity() == agent_id.id.as_bytes())
            .map(|m| m.index.u32())
            .unwrap_or(0);
        let member = GroupMember {
            agent_id: agent_id.clone(),
            member_id: member_id.clone(),
            is_admin: false,
            joined_at: now,
            last_active: now,
            vote_weight: 1,
            leaf_index: Some(leaf_index),
        };
        {
            let mut members = self.members.write().await;
            members.insert(member_id.clone(), member.clone());
        }
        {
            let mut stats = self.stats.write().await;
            stats.members_added += 1;
        }
        {
            let mut store = self.key_store.write().await;
            store.insert(member_id, key_package);
        }
        self.save_state().await?;
        let callbacks = self.callbacks.read().await.clone();
        for cb in callbacks {
            cb.member_added(&member).await;
        }
        info!("Added member {} to group {}", agent_id, self.group_id);
        Ok(())
    }

    /// Add member to group (simplified version without MLS)
    #[cfg(not(feature = "mls"))]
    pub async fn add_member(&self, agent_id: AgentId) -> Result<()> {
        // Simplified implementation for demo purposes
        // In a real implementation, this would use MLS key packages and proposals

        // For now, directly add the member
        let member_id = agent_id.to_string().into_bytes();
        let member = GroupMember {
            agent_id: agent_id.clone(),
            member_id: member_id.clone(),
            is_admin: false,
            joined_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            last_active: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            vote_weight: 1,
        };

        {
            let mut members = self.members.write().await;
            members.insert(member_id, member);
        }

        {
            let mut stats = self.stats.write().await;
            stats.members_added += 1;
        }

        info!("Added member {} to group {}", agent_id, self.group_id);
        Ok(())
    }

    /// Remove member from group
    pub async fn remove_member(&self, agent_id: &AgentId) -> Result<()> {
        let member_id = agent_id.to_string().into_bytes();
        #[cfg(feature = "mls")]
        {
            let mut group_lock = self.mls_group.lock().await;
            let group = group_lock.as_mut().ok_or_else(|| {
                AgentFabricError::MlsError("Group not initialized".to_string())
            })?;
            let index = {
                let members = self.members.read().await;
                members
                    .get(&member_id)
                    .and_then(|m| m.leaf_index)
                    .unwrap_or(0)
            };
            group
                .remove_members(
                    self.crypto_provider.as_ref(),
                    self.signer.as_ref(),
                    &[LeafNodeIndex::new(index)],
                )
                .map_err(|e| AgentFabricError::MlsError(format!("{e:?}")))?;
            group
                .merge_pending_commit(self.crypto_provider.as_ref())
                .map_err(|e| AgentFabricError::MlsError(format!("{e:?}")))?;
        }

        {
            let mut members = self.members.write().await;
            members.remove(&member_id);
        }
        {
            let mut stats = self.stats.write().await;
            stats.members_removed += 1;
        }
        self.save_state().await?;
        let callbacks = self.callbacks.read().await.clone();
        for cb in callbacks {
            cb.member_removed(agent_id).await;
        }
        info!("Removed member {} from group {}", agent_id, self.group_id);
        Ok(())
    }

    /// Start a vote
    pub async fn start_vote(
        &self,
        proposer: AgentId,
        title: String,
        description: String,
        timeout_seconds: u64,
    ) -> Result<String> {
        let proposal_id = uuid::Uuid::new_v4().to_string();
        let deadline = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            + timeout_seconds;

        let proposal = VoteProposal {
            proposal_id: proposal_id.clone(),
            proposer: proposer.clone(),
            title,
            description,
            votes_for: 0,
            votes_against: 0,
            total_weight: self.get_total_vote_weight().await,
            deadline,
            executed: false,
        };

        {
            let mut proposals = self.proposals.write().await;
            proposals.insert(proposal_id.clone(), proposal);
        }
        self.save_state().await?;

        // Send vote proposal message
        let vote_message = GroupMessage {
            message_id: uuid::Uuid::new_v4().to_string(),
            from: proposer,
            message_type: GroupMessageType::Vote {
                proposal_id: proposal_id.clone(),
                action: VoteAction::Propose,
            },
            payload: Bytes::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let _ = self.send_message(vote_message).await?;

        {
            let mut stats = self.stats.write().await;
            stats.votes_initiated += 1;
        }

        info!("Started vote {} in group {}", proposal_id, self.group_id);
        Ok(proposal_id)
    }

    /// Cast a vote
    pub async fn cast_vote(
        &self,
        proposal_id: String,
        voter: AgentId,
        approve: bool,
    ) -> Result<()> {
        let vote_weight = self.get_member_vote_weight(&voter).await.unwrap_or(0);

        if vote_weight == 0 {
            return Err(AgentFabricError::MlsError(
                "Voter not found or no vote weight".to_string(),
            ));
        }

        // Update proposal
        {
            let mut proposals = self.proposals.write().await;
            if let Some(proposal) = proposals.get_mut(&proposal_id) {
                if approve {
                    proposal.votes_for += vote_weight;
                } else {
                    proposal.votes_against += vote_weight;
                }
            } else {
                return Err(AgentFabricError::MlsError("Proposal not found".to_string()));
            }
        }
        self.save_state().await?;

        // Send vote message
        let vote_message = GroupMessage {
            message_id: uuid::Uuid::new_v4().to_string(),
            from: voter.clone(),
            message_type: GroupMessageType::Vote {
                proposal_id: proposal_id.clone(),
                action: VoteAction::Vote { approve },
            },
            payload: Bytes::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let _ = self.send_message(vote_message).await?;

        debug!(
            "Vote cast for proposal {} by {}: {}",
            proposal_id, voter, approve
        );
        Ok(())
    }

    /// Get group statistics
    pub async fn get_stats(&self) -> GroupStats {
        self.stats.read().await.clone()
    }

    /// Get group members
    pub async fn get_members(&self) -> Vec<GroupMember> {
        self.members.read().await.values().cloned().collect()
    }

    /// Get group configuration
    pub fn get_config(&self) -> &GroupConfig {
        &self.config
    }

    // Private helper methods

    async fn handle_group_message(&self, message: &GroupMessage) -> Result<()> {
        match &message.message_type {
            GroupMessageType::Training { session_id, action } => {
                self.handle_training_message(session_id, action).await?;
            }
            GroupMessageType::Alert {
                alert_level,
                category,
            } => {
                self.handle_alert_message(alert_level, category).await?;
            }
            GroupMessageType::Vote {
                proposal_id,
                action,
            } => {
                self.handle_vote_message(proposal_id, action).await?;
            }
            GroupMessageType::Broadcast { category: _ } => {
                // General broadcast handling
            }
        }

        // Update member activity
        let member_id = message.from.to_string().into_bytes();
        {
            let mut members = self.members.write().await;
            if let Some(member) = members.get_mut(&member_id) {
                member.last_active = message.timestamp;
            }
        }

        Ok(())
    }

    async fn handle_training_message(
        &self,
        _session_id: &str,
        action: &TrainingAction,
    ) -> Result<()> {
        match action {
            TrainingAction::StartSession => {
                let mut stats = self.stats.write().await;
                stats.training_sessions += 1;
            }
            TrainingAction::UpdateModel => {
                debug!("Model update in training session");
            }
            TrainingAction::ShareGradients => {
                debug!("Gradients shared in training session");
            }
            TrainingAction::SyncParameters => {
                debug!("Parameters synchronized in training session");
            }
            TrainingAction::EndSession => {
                debug!("Training session ended");
            }
        }
        Ok(())
    }

    async fn handle_alert_message(&self, level: &AlertLevel, category: &str) -> Result<()> {
        match level {
            AlertLevel::Emergency => {
                warn!("EMERGENCY ALERT in {}: {}", self.group_id, category);
            }
            AlertLevel::Critical => {
                warn!("Critical alert in {}: {}", self.group_id, category);
            }
            AlertLevel::Warning => {
                info!("Warning in {}: {}", self.group_id, category);
            }
            AlertLevel::Info => {
                debug!("Info alert in {}: {}", self.group_id, category);
            }
        }
        Ok(())
    }

    async fn handle_vote_message(&self, proposal_id: &str, action: &VoteAction) -> Result<()> {
        match action {
            VoteAction::Propose => {
                debug!("Vote proposal {} created", proposal_id);
            }
            VoteAction::Vote { approve: _ } => {
                debug!("Vote cast for proposal {}", proposal_id);
            }
            VoteAction::Tally => {
                self.tally_vote(proposal_id).await?;
            }
            VoteAction::Execute => {
                debug!("Executing proposal {}", proposal_id);
            }
        }
        Ok(())
    }

    async fn tally_vote(&self, proposal_id: &str) -> Result<()> {
        let mut proposals = self.proposals.write().await;
        if let Some(proposal) = proposals.get_mut(proposal_id) {
            let passed = if self.config.require_unanimous_votes {
                proposal.votes_for == proposal.total_weight && proposal.votes_against == 0
            } else {
                proposal.votes_for > proposal.votes_against
            };

            if passed {
                proposal.executed = true;
                info!(
                    "Vote {} PASSED ({} for, {} against)",
                    proposal_id, proposal.votes_for, proposal.votes_against
                );
                let callbacks = self.callbacks.read().await.clone();
                for cb in callbacks {
                    cb.vote_executed(proposal).await;
                }
            } else {
                info!(
                    "Vote {} FAILED ({} for, {} against)",
                    proposal_id, proposal.votes_for, proposal.votes_against
                );
            }

            let mut stats = self.stats.write().await;
            stats.votes_completed += 1;
        }
        self.save_state().await?;
        Ok(())
    }

    async fn get_total_vote_weight(&self) -> u32 {
        let members = self.members.read().await;
        members.values().map(|m| m.vote_weight).sum()
    }

    async fn get_member_vote_weight(&self, agent_id: &AgentId) -> Option<u32> {
        let member_id = agent_id.to_string().into_bytes();
        let members = self.members.read().await;
        members.get(&member_id).map(|m| m.vote_weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "mls")]
    fn make_key_package(identity: &str, backend: &OpenMlsRustCrypto) -> KeyPackage {
        let credential = Credential::new(identity.as_bytes().to_vec(), CredentialType::Basic).unwrap();
        let signer = SignatureKeyPair::new(SignatureScheme::ED25519).unwrap();
        signer.store(backend.key_store()).unwrap();
        let cred_with_key = CredentialWithKey { credential, signature_key: signer.public().into() };
        KeyPackage::builder()
            .build(
                CryptoConfig::with_default_version(Ciphersuite::MLS_128_DHKEMX25519_AES128GCM_SHA256_Ed25519),
                backend,
                &signer,
                cred_with_key,
            )
            .unwrap()
    }

    #[tokio::test]
    async fn test_group_creation() {
        let config = GroupConfig::default();
        let group = MlsGroup::new("test-group".to_string(), config)
            .await
            .unwrap();

        assert_eq!(group.group_id, "test-group");
        assert_eq!(group.config.group_id, "default");
    }

    #[test]
    fn test_group_message_serialization() {
        let message = GroupMessage {
            message_id: "msg-123".to_string(),
            from: AgentId::new("agent1", "node1"),
            message_type: GroupMessageType::Training {
                session_id: "session-1".to_string(),
                action: TrainingAction::StartSession,
            },
            payload: Bytes::from("test payload"),
            timestamp: 1234567890,
        };

        let serialized = serde_json::to_vec(&message).unwrap();
        let deserialized: GroupMessage = serde_json::from_slice(&serialized).unwrap();

        assert_eq!(message.message_id, deserialized.message_id);
        assert_eq!(message.from, deserialized.from);
        assert_eq!(message.timestamp, deserialized.timestamp);
    }

    #[test]
    fn test_group_config_default() {
        let config = GroupConfig::default();
        assert_eq!(config.group_id, "default");
        assert_eq!(config.max_members, 100);
        assert!(!config.admin_only_add);
        assert!(!config.require_unanimous_votes);
    }

    #[test]
    fn test_vote_proposal() {
        let proposal = VoteProposal {
            proposal_id: "prop-1".to_string(),
            proposer: AgentId::new("proposer", "node1"),
            title: "Test Proposal".to_string(),
            description: "A test proposal".to_string(),
            votes_for: 0,
            votes_against: 0,
            total_weight: 10,
            deadline: 1234567890,
            executed: false,
        };

        assert_eq!(proposal.proposal_id, "prop-1");
        assert_eq!(proposal.total_weight, 10);
        assert!(!proposal.executed);
    }

    #[cfg(feature = "mls")]
    #[tokio::test]
    async fn test_member_add_remove() {
        let config = GroupConfig::default();
        let group = MlsGroup::new("g1".to_string(), config).await.unwrap();
        let creator = AgentId::new("creator", "n1");
        group.initialize_as_creator(creator.clone()).await.unwrap();
        let backend = group.crypto_provider.clone();
        let kp = make_key_package("bob", &backend);
        let bob = AgentId::new("bob", "n1");
        group.add_member(bob.clone(), kp).await.unwrap();
        assert_eq!(group.get_members().await.len(), 2);
        group.remove_member(&bob).await.unwrap();
        assert_eq!(group.get_members().await.len(), 1);
    }

    #[cfg(feature = "mls")]
    #[tokio::test]
    async fn test_vote_tally() {
        let config = GroupConfig::default();
        let group = MlsGroup::new("g2".to_string(), config).await.unwrap();
        let creator = AgentId::new("creator", "n1");
        group.initialize_as_creator(creator.clone()).await.unwrap();
        let proposal_id = group
            .start_vote(
                creator.clone(),
                "title".into(),
                "desc".into(),
                60,
            )
            .await
            .unwrap();
        group
            .cast_vote(proposal_id.clone(), creator.clone(), true)
            .await
            .unwrap();
        group.tally_vote(&proposal_id).await.unwrap();
        let proposals = group.proposals.read().await;
        assert!(proposals.get(&proposal_id).unwrap().executed);
    }

    #[cfg(feature = "mls")]
    #[tokio::test]
    async fn test_encrypted_roundtrip() {
        let config = GroupConfig::default();
        let group = MlsGroup::new("g3".to_string(), config).await.unwrap();
        let creator = AgentId::new("creator", "n1");
        group.initialize_as_creator(creator.clone()).await.unwrap();
        let message = GroupMessage {
            message_id: "m1".to_string(),
            from: creator.clone(),
            message_type: GroupMessageType::Broadcast { category: "test".into() },
            payload: Bytes::from("hello"),
            timestamp: 1,
        };
        let bytes = group.send_message(message.clone()).await.unwrap();
        assert!(!bytes.is_empty());
    }
}
